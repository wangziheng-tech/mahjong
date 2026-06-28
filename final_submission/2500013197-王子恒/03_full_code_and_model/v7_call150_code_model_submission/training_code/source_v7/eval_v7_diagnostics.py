import argparse
import importlib
import json
import os
import sys
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Subset

from dataset_v2 import MahjongGBDataset


TYPE_NAMES = ["Pass", "Hu", "Play", "Chi", "Peng", "Gang", "AnGang", "BuGang"]
TYPE_TO_ID = {name: i for i, name in enumerate(TYPE_NAMES)}

ACTION_RANGES = OrderedDict(
    [
        ("Pass", (0, 1)),
        ("Hu", (1, 2)),
        ("Play", (2, 36)),
        ("Chi", (36, 99)),
        ("Peng", (99, 133)),
        ("Gang", (133, 167)),
        ("AnGang", (167, 201)),
        ("BuGang", (201, 235)),
    ]
)

TILE_NAMES = [
    *("W%d" % (i + 1) for i in range(9)),
    *("T%d" % (i + 1) for i in range(9)),
    *("B%d" % (i + 1) for i in range(9)),
    *("F%d" % (i + 1) for i in range(4)),
    *("J%d" % (i + 1) for i in range(3)),
]

TILE_GROUPS = OrderedDict(
    [
        ("W", list(range(0, 9))),
        ("T", list(range(9, 18))),
        ("B", list(range(18, 27))),
        ("Honor", list(range(27, 34))),
        ("Terminal", [0, 8, 9, 17, 18, 26]),
        ("Yaojiu", [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]),
    ]
)


def configure_stdout():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def safe_div(num, den):
    return None if den == 0 else float(num) / float(den)


def pct(value):
    return "n/a" if value is None else "%.4f" % value


def action_type_id(action):
    out = torch.empty_like(action, dtype=torch.long)
    out[action == 0] = TYPE_TO_ID["Pass"]
    out[action == 1] = TYPE_TO_ID["Hu"]
    out[(2 <= action) & (action < 36)] = TYPE_TO_ID["Play"]
    out[(36 <= action) & (action < 99)] = TYPE_TO_ID["Chi"]
    out[(99 <= action) & (action < 133)] = TYPE_TO_ID["Peng"]
    out[(133 <= action) & (action < 167)] = TYPE_TO_ID["Gang"]
    out[(167 <= action) & (action < 201)] = TYPE_TO_ID["AnGang"]
    out[action >= 201] = TYPE_TO_ID["BuGang"]
    return out


def legal_any(mask, names):
    out = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
    for name in names:
        begin, end = ACTION_RANGES[name]
        out |= mask[:, begin:end].sum(dim=1) > 0
    return out


def legal_play_group(mask, tile_ids):
    cols = torch.as_tensor([2 + t for t in tile_ids], dtype=torch.long, device=mask.device)
    return mask.index_select(1, cols).sum(dim=1) > 0


def tile_in(tile_ids, group):
    ids = torch.as_tensor(TILE_GROUPS[group], dtype=torch.long, device=tile_ids.device)
    return (tile_ids.unsqueeze(1) == ids.unsqueeze(0)).any(dim=1)


def make_input(obs, mask, device):
    return {
        "is_training": False,
        "obs": {
            "observation": obs.to(device).float(),
            "action_mask": mask.to(device).float(),
        },
    }


def load_model(module_name, ckpt, device):
    model = importlib.import_module(module_name).CNNModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()
    return model


def forward_logits(model, obs, mask, device):
    input_dict = make_input(obs, mask, device)
    if hasattr(model, "forward_all"):
        logits = model.forward_all(input_dict)[0]
    else:
        logits = model(input_dict)
    mask = mask.to(device).float()
    return logits + torch.clamp(torch.log(mask), -1e9, 0)


def parse_model_spec(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--model must be name:module:ckpt, got %r" % spec)
    return {"name": parts[0], "module": parts[1], "ckpt": parts[2]}


def init_stats():
    return {
        "total": 0,
        "exact": 0,
        "top3": 0,
        "top5": 0,
        "by_type": {
            name: {"total": 0, "exact": 0, "type_hit": 0, "top3": 0, "top5": 0, "pred_pass": 0}
            for name in TYPE_NAMES
        },
        "opportunities": {},
        "discard": {
            "play_total": 0,
            "target_play_total": 0,
            "target_play_exact": 0,
            "target_group": {
                name: {"target": 0, "exact": 0, "pred_same_group": 0}
                for name in TILE_GROUPS
            },
            "legal_group": {
                name: {"legal": 0, "label": 0, "pred": 0}
                for name in TILE_GROUPS
            },
            "hand_average_sum": {name: 0.0 for name in ("W", "T", "B", "Honor")},
            "singleton_honor": {
                "opportunity": 0,
                "label": 0,
                "pred": 0,
                "exact_when_label": 0,
                "top3_when_label": 0,
            },
            "honor_tiles": {
                TILE_NAMES[i]: {"target": 0, "exact": 0, "pred_when_target": 0}
                for i in TILE_GROUPS["Honor"]
            },
        },
    }


def add_opp(stats, name, cond, label_group, pred_group, target_type, pred_type, exact):
    total = int(cond.sum().item())
    if total == 0:
        return
    item = stats["opportunities"].setdefault(
        name,
        {
            "total": 0,
            "exact": 0,
            "label": 0,
            "pred": 0,
            "hit": 0,
            "pred_pass": 0,
            "pred_play": 0,
            "label_type_hist": {n: 0 for n in TYPE_NAMES},
            "pred_type_hist": {n: 0 for n in TYPE_NAMES},
        },
    )
    item["total"] += total
    item["exact"] += int((exact & cond).sum().item())
    item["label"] += int((label_group & cond).sum().item())
    item["pred"] += int((pred_group & cond).sum().item())
    item["hit"] += int((label_group & pred_group & cond).sum().item())
    item["pred_pass"] += int(((pred_type == TYPE_TO_ID["Pass"]) & cond).sum().item())
    item["pred_play"] += int(((pred_type == TYPE_TO_ID["Play"]) & cond).sum().item())
    for i, n in enumerate(TYPE_NAMES):
        item["label_type_hist"][n] += int(((target_type == i) & cond).sum().item())
        item["pred_type_hist"][n] += int(((pred_type == i) & cond).sum().item())


def update_batch(stats, obs, mask, act, pred, topk):
    target_type = action_type_id(act)
    pred_type = action_type_id(pred)
    exact = pred == act
    top3_hit = (topk[:, : min(3, topk.shape[1])] == act.unsqueeze(1)).any(dim=1)
    top5_hit = (topk == act.unsqueeze(1)).any(dim=1)

    n = act.numel()
    stats["total"] += n
    stats["exact"] += int(exact.sum().item())
    stats["top3"] += int(top3_hit.sum().item())
    stats["top5"] += int(top5_hit.sum().item())

    for i, name in enumerate(TYPE_NAMES):
        cond = target_type == i
        item = stats["by_type"][name]
        item["total"] += int(cond.sum().item())
        item["exact"] += int((exact & cond).sum().item())
        item["type_hit"] += int(((pred_type == i) & cond).sum().item())
        item["top3"] += int((top3_hit & cond).sum().item())
        item["top5"] += int((top5_hit & cond).sum().item())
        item["pred_pass"] += int(((pred_type == TYPE_TO_ID["Pass"]) & cond).sum().item())

    respond_meld = ["Chi", "Peng", "Gang"]
    exposed = ["Chi", "Peng", "Gang", "BuGang"]
    set_names = ["Chi", "Peng", "Gang", "AnGang", "BuGang"]
    kong = ["Gang", "AnGang", "BuGang"]
    respond_sample = mask[:, ACTION_RANGES["Pass"][0]] > 0
    for name, action_names in [
        ("respond_meld_opportunity", respond_meld),
        ("open_meld_opportunity", exposed),
        ("set_opportunity_including_angang", set_names),
        ("kong_opportunity", kong),
    ]:
        ids = torch.as_tensor([TYPE_TO_ID[x] for x in action_names], dtype=torch.long, device=act.device)
        label_group = (target_type.unsqueeze(1) == ids.unsqueeze(0)).any(dim=1)
        pred_group = (pred_type.unsqueeze(1) == ids.unsqueeze(0)).any(dim=1)
        cond = legal_any(mask, action_names)
        if name == "respond_meld_opportunity":
            cond &= respond_sample
        add_opp(stats, name, cond, label_group, pred_group, target_type, pred_type, exact)

    for name in ("Chi", "Peng", "Gang", "AnGang", "BuGang"):
        add_opp(
            stats,
            name.lower() + "_legal_opportunity",
            legal_any(mask, [name]),
            target_type == TYPE_TO_ID[name],
            pred_type == TYPE_TO_ID[name],
            target_type,
            pred_type,
            exact,
        )

    play_sample = legal_any(mask, ["Play"])
    target_play = target_type == TYPE_TO_ID["Play"]
    pred_play = pred_type == TYPE_TO_ID["Play"]
    d = stats["discard"]
    d["play_total"] += int(play_sample.sum().item())
    d["target_play_total"] += int(target_play.sum().item())
    d["target_play_exact"] += int((target_play & exact).sum().item())

    act_tile = (act - 2).clamp(min=0, max=33)
    pred_tile = (pred - 2).clamp(min=0, max=33)
    for group, tile_ids in TILE_GROUPS.items():
        label_group = target_play & tile_in(act_tile, group)
        pred_group = pred_play & tile_in(pred_tile, group)
        legal_group = play_sample & legal_play_group(mask, tile_ids)

        tg = d["target_group"][group]
        tg["target"] += int(label_group.sum().item())
        tg["exact"] += int((label_group & exact).sum().item())
        tg["pred_same_group"] += int((label_group & pred_group).sum().item())

        lg = d["legal_group"][group]
        lg["legal"] += int(legal_group.sum().item())
        lg["label"] += int((legal_group & label_group).sum().item())
        lg["pred"] += int((legal_group & pred_group).sum().item())

    obs_flat = obs.to(act.device).float().reshape(obs.shape[0], obs.shape[1], 36)
    hand_count = obs_flat[:, 2:6, :34].sum(dim=1)
    for group in ("W", "T", "B", "Honor"):
        ids = torch.as_tensor(TILE_GROUPS[group], dtype=torch.long, device=act.device)
        d["hand_average_sum"][group] += float(hand_count.index_select(1, ids).sum(dim=1)[play_sample].sum().item())

    honor_ids = torch.as_tensor(TILE_GROUPS["Honor"], dtype=torch.long, device=act.device)
    singleton_honor_tile = hand_count.index_select(1, honor_ids) == 1
    legal_singleton_honor = (
        singleton_honor_tile & (mask.index_select(1, 2 + honor_ids) > 0)
    ).any(dim=1) & play_sample
    act_count = hand_count.gather(1, act_tile.unsqueeze(1)).squeeze(1)
    pred_count = hand_count.gather(1, pred_tile.unsqueeze(1)).squeeze(1)
    act_singleton_honor = target_play & tile_in(act_tile, "Honor") & (act_count == 1)
    pred_singleton_honor = pred_play & tile_in(pred_tile, "Honor") & (pred_count == 1)
    sh = d["singleton_honor"]
    sh["opportunity"] += int(legal_singleton_honor.sum().item())
    sh["label"] += int((legal_singleton_honor & act_singleton_honor).sum().item())
    sh["pred"] += int((legal_singleton_honor & pred_singleton_honor).sum().item())
    sh["exact_when_label"] += int((legal_singleton_honor & act_singleton_honor & exact).sum().item())
    sh["top3_when_label"] += int((legal_singleton_honor & act_singleton_honor & top3_hit).sum().item())

    for tile_id in TILE_GROUPS["Honor"]:
        action = 2 + tile_id
        tile = TILE_NAMES[tile_id]
        target_tile = act == action
        item = d["honor_tiles"][tile]
        item["target"] += int(target_tile.sum().item())
        item["exact"] += int((target_tile & exact).sum().item())
        item["pred_when_target"] += int((target_tile & (pred == action)).sum().item())


def finalize(stats):
    out = {
        "total": stats["total"],
        "overall": {
            "exact_acc": safe_div(stats["exact"], stats["total"]),
            "top3": safe_div(stats["top3"], stats["total"]),
            "top5": safe_div(stats["top5"], stats["total"]),
        },
        "by_type": {},
        "opportunities": {},
        "discard": {},
    }

    for name, item in stats["by_type"].items():
        total = item["total"]
        out["by_type"][name] = {
            "total": total,
            "exact_acc": safe_div(item["exact"], total),
            "type_recall": safe_div(item["type_hit"], total),
            "top3": safe_div(item["top3"], total),
            "top5": safe_div(item["top5"], total),
            "pred_pass_rate": safe_div(item["pred_pass"], total),
        }

    for name, item in stats["opportunities"].items():
        total = item["total"]
        out["opportunities"][name] = {
            "total": total,
            "exact_acc": safe_div(item["exact"], total),
            "label_group_rate": safe_div(item["label"], total),
            "pred_group_rate": safe_div(item["pred"], total),
            "group_recall": safe_div(item["hit"], item["label"]),
            "group_precision": safe_div(item["hit"], item["pred"]),
            "pred_pass_rate": safe_div(item["pred_pass"], total),
            "pred_play_rate": safe_div(item["pred_play"], total),
            "label_type_hist": item["label_type_hist"],
            "pred_type_hist": item["pred_type_hist"],
        }

    d = stats["discard"]
    out_d = out["discard"]
    out_d["play_total"] = d["play_total"]
    out_d["target_play_total"] = d["target_play_total"]
    out_d["target_play_exact_acc"] = safe_div(d["target_play_exact"], d["target_play_total"])
    out_d["target_group"] = {}
    for group, item in d["target_group"].items():
        total = item["target"]
        out_d["target_group"][group] = {
            "target": total,
            "target_rate_among_play_labels": safe_div(total, d["target_play_total"]),
            "exact_acc": safe_div(item["exact"], total),
            "pred_same_group_rate": safe_div(item["pred_same_group"], total),
        }
    out_d["legal_group"] = {}
    for group, item in d["legal_group"].items():
        total = item["legal"]
        out_d["legal_group"][group] = {
            "legal": total,
            "label_rate_when_legal": safe_div(item["label"], total),
            "pred_rate_when_legal": safe_div(item["pred"], total),
        }
    out_d["hand_avg_on_play_sample"] = {
        group: safe_div(value, d["play_total"])
        for group, value in d["hand_average_sum"].items()
    }
    sh = d["singleton_honor"]
    out_d["singleton_honor"] = {
        "opportunity": sh["opportunity"],
        "label_rate": safe_div(sh["label"], sh["opportunity"]),
        "pred_rate": safe_div(sh["pred"], sh["opportunity"]),
        "exact_when_label": safe_div(sh["exact_when_label"], sh["label"]),
        "top3_when_label": safe_div(sh["top3_when_label"], sh["label"]),
    }
    out_d["honor_tiles"] = {
        tile: {
            "target": item["target"],
            "exact_acc": safe_div(item["exact"], item["target"]),
            "pred_when_target_rate": safe_div(item["pred_when_target"], item["target"]),
        }
        for tile, item in d["honor_tiles"].items()
    }
    return out


def evaluate_model(model_spec, loader, device, progress):
    print("loading model:", model_spec["name"], model_spec["module"], model_spec["ckpt"])
    model = load_model(model_spec["module"], model_spec["ckpt"], device)
    stats = init_stats()
    with torch.no_grad():
        for batch_id, (obs, mask, act) in enumerate(loader):
            act = act.to(device).long()
            mask = mask.to(device).float()
            logits = forward_logits(model, obs, mask, device)
            pred = logits.argmax(dim=1)
            topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            update_batch(stats, obs, mask, act, pred, topk)
            if progress and batch_id % progress == 0:
                print(model_spec["name"], "batch", batch_id, "samples", stats["total"])
    out = finalize(stats)
    out["model"] = model_spec
    return out


def print_compact(results):
    print("\n=== Compact Comparison ===")
    print("model                 acc     top3    play    pass    chi     peng    resp_pred resp_prec honor_pred singleton_pred")
    for name, r in results.items():
        opp = r["opportunities"].get("respond_meld_opportunity", {})
        legal_honor = r["discard"]["legal_group"]["Honor"]
        singleton = r["discard"]["singleton_honor"]
        print(
            "%-20s %7s %7s %7s %7s %7s %7s %9s %9s %10s %14s"
            % (
                name,
                pct(r["overall"]["exact_acc"]),
                pct(r["overall"]["top3"]),
                pct(r["by_type"]["Play"]["exact_acc"]),
                pct(r["by_type"]["Pass"]["exact_acc"]),
                pct(r["by_type"]["Chi"]["exact_acc"]),
                pct(r["by_type"]["Peng"]["exact_acc"]),
                pct(opp.get("pred_group_rate")),
                pct(opp.get("group_precision")),
                pct(legal_honor["pred_rate_when_legal"]),
                pct(singleton["pred_rate"]),
            )
        )

    print("\n=== Suit Discard Rates When Legal ===")
    print("model                 W_label W_pred  T_label T_pred  B_label B_pred  H_label H_pred")
    for name, r in results.items():
        lg = r["discard"]["legal_group"]
        print(
            "%-20s %7s %7s %7s %7s %7s %7s %7s %7s"
            % (
                name,
                pct(lg["W"]["label_rate_when_legal"]),
                pct(lg["W"]["pred_rate_when_legal"]),
                pct(lg["T"]["label_rate_when_legal"]),
                pct(lg["T"]["pred_rate_when_legal"]),
                pct(lg["B"]["label_rate_when_legal"]),
                pct(lg["B"]["pred_rate_when_legal"]),
                pct(lg["Honor"]["label_rate_when_legal"]),
                pct(lg["Honor"]["pred_rate_when_legal"]),
            )
        )


def main():
    configure_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name:module:ckpt")
    parser.add_argument("--begin", type=float, default=0.95)
    parser.add_argument("--end", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="eval/v7_diagnostics.json")
    parser.add_argument("--progress", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("split:", args.begin, args.end)

    ds = MahjongGBDataset(args.begin, args.end, augment=False)
    if args.max_samples and args.max_samples < len(ds):
        ds = Subset(ds, range(args.max_samples))
    print("samples:", len(ds))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    results = {}
    for spec in args.model:
        model_spec = parse_model_spec(spec)
        results[model_spec["name"]] = evaluate_model(model_spec, loader, device, args.progress)

    payload = {
        "config": vars(args),
        "results": results,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print("wrote:", args.output)

    print_compact(results)


if __name__ == "__main__":
    main()
