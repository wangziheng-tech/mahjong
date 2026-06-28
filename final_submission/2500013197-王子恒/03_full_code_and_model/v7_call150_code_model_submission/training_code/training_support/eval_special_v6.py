import argparse
import importlib
import json
from collections import OrderedDict

import torch
import torch.nn.functional as F
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

HONOR_TILE_IDS = list(range(27, 34))
TERMINAL_TILE_IDS = [0, 8, 9, 17, 18, 26]
HONOR_PLAY_ACTIONS = [2 + i for i in HONOR_TILE_IDS]
TERMINAL_PLAY_ACTIONS = [2 + i for i in TERMINAL_TILE_IDS]
YAOJIU_PLAY_ACTIONS = TERMINAL_PLAY_ACTIONS + HONOR_PLAY_ACTIONS


def configure_stdout():
    try:
        import sys

        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def pct(x):
    if x is None:
        return "n/a"
    return "%.4f" % x


def safe_div(num, den):
    return None if den == 0 else float(num) / float(den)


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


def type_in(type_tensor, names):
    out = torch.zeros_like(type_tensor, dtype=torch.bool)
    for name in names:
        out |= type_tensor == TYPE_TO_ID[name]
    return out


def action_in(action, actions):
    out = torch.zeros_like(action, dtype=torch.bool)
    for a in actions:
        out |= action == a
    return out


def legal_any(mask, names):
    out = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
    for name in names:
        begin, end = ACTION_RANGES[name]
        out |= mask[:, begin:end].sum(dim=1) > 0
    return out


def legal_actions(mask, actions):
    cols = torch.as_tensor(actions, dtype=torch.long, device=mask.device)
    return mask.index_select(1, cols).sum(dim=1) > 0


def load_model(module_name, ckpt, device):
    model = importlib.import_module(module_name).CNNModel().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_input(obs, mask, device):
    return {
        "is_training": False,
        "obs": {
            "observation": obs.to(device).float(),
            "action_mask": mask.to(device).float(),
        },
    }


def forward_logits(model, obs, mask, device):
    input_dict = make_input(obs, mask, device)
    if hasattr(model, "forward_all"):
        out = model.forward_all(input_dict)
        logits = out[0]
        type_logits = out[1] if len(out) > 1 else None
    else:
        logits = model(input_dict)
        type_logits = None
    mask = mask.to(device).float()
    logits = logits + torch.clamp(torch.log(mask), -1e9, 0)
    return logits, type_logits


def make_summary(total, exact, target_group, pred_group, target_type, pred_type):
    target_group_n = int(target_group.sum().item())
    pred_group_n = int(pred_group.sum().item())
    hit_group_n = int((target_group & pred_group).sum().item())
    return {
        "total": int(total),
        "exact_acc": safe_div(exact, total),
        "label_group_rate": safe_div(target_group_n, total),
        "pred_group_rate": safe_div(pred_group_n, total),
        "group_recall": safe_div(hit_group_n, target_group_n),
        "group_precision": safe_div(hit_group_n, pred_group_n),
        "pred_pass_rate": safe_div(int((pred_type == TYPE_TO_ID["Pass"]).sum().item()), total),
        "pred_hu_rate": safe_div(int((pred_type == TYPE_TO_ID["Hu"]).sum().item()), total),
        "pred_play_rate": safe_div(int((pred_type == TYPE_TO_ID["Play"]).sum().item()), total),
        "label_type_hist": {
            name: int((target_type == i).sum().item()) for i, name in enumerate(TYPE_NAMES)
        },
        "pred_type_hist": {
            name: int((pred_type == i).sum().item()) for i, name in enumerate(TYPE_NAMES)
        },
    }


def empty_summary():
    return {
        "total": 0,
        "exact_acc": None,
        "label_group_rate": None,
        "pred_group_rate": None,
        "group_recall": None,
        "group_precision": None,
        "pred_pass_rate": None,
        "pred_hu_rate": None,
        "pred_play_rate": None,
        "label_type_hist": {name: 0 for name in TYPE_NAMES},
        "pred_type_hist": {name: 0 for name in TYPE_NAMES},
    }


def init_counter():
    return {
        "total": 0,
        "exact": 0,
        "top3": 0,
        "top5": 0,
        "type_confusion": [[0 for _ in TYPE_NAMES] for _ in TYPE_NAMES],
        "target_type_total": {name: 0 for name in TYPE_NAMES},
        "target_type_exact": {name: 0 for name in TYPE_NAMES},
        "target_type_type_correct": {name: 0 for name in TYPE_NAMES},
        "target_type_top3": {name: 0 for name in TYPE_NAMES},
        "target_type_top5": {name: 0 for name in TYPE_NAMES},
        "target_type_pred_pass": {name: 0 for name in TYPE_NAMES},
        "target_type_pred_hu": {name: 0 for name in TYPE_NAMES},
        "opportunity_raw": {},
        "discard": {
            "target_honor": {
                "total": 0,
                "exact": 0,
                "top3": 0,
                "top5": 0,
                "pred_honor": 0,
                "pred_play_nonhonor": 0,
            },
            "legal_honor": {
                "total": 0,
                "label_honor": 0,
                "pred_honor": 0,
                "exact_when_label_honor": 0,
            },
            "target_terminal": {"total": 0, "exact": 0, "pred_terminal": 0},
            "legal_terminal": {"total": 0, "label_terminal": 0, "pred_terminal": 0},
            "target_yaojiu": {"total": 0, "exact": 0, "pred_yaojiu": 0},
            "legal_yaojiu": {"total": 0, "label_yaojiu": 0, "pred_yaojiu": 0},
            "honor_tiles": {
                TILE_NAMES[i]: {"target": 0, "exact": 0, "pred_when_target": 0}
                for i in HONOR_TILE_IDS
            },
        },
    }


def add_opportunity(counter, name, cond, target_group, pred_group, target_type, pred_type, exact):
    total = int(cond.sum().item())
    if total == 0:
        return
    c = counter["opportunity_raw"].setdefault(
        name,
        {
            "total": 0,
            "exact": 0,
            "target_group": 0,
            "pred_group": 0,
            "hit_group": 0,
            "pred_pass": 0,
            "pred_hu": 0,
            "pred_play": 0,
            "label_type_hist": {n: 0 for n in TYPE_NAMES},
            "pred_type_hist": {n: 0 for n in TYPE_NAMES},
        },
    )
    c["total"] += total
    c["exact"] += int((exact & cond).sum().item())
    c["target_group"] += int((target_group & cond).sum().item())
    c["pred_group"] += int((pred_group & cond).sum().item())
    c["hit_group"] += int((target_group & pred_group & cond).sum().item())
    c["pred_pass"] += int(((pred_type == TYPE_TO_ID["Pass"]) & cond).sum().item())
    c["pred_hu"] += int(((pred_type == TYPE_TO_ID["Hu"]) & cond).sum().item())
    c["pred_play"] += int(((pred_type == TYPE_TO_ID["Play"]) & cond).sum().item())
    for i, n in enumerate(TYPE_NAMES):
        c["label_type_hist"][n] += int(((target_type == i) & cond).sum().item())
        c["pred_type_hist"][n] += int(((pred_type == i) & cond).sum().item())


def finalize(counter):
    out = {
        "total": counter["total"],
        "overall": {
            "exact_acc": safe_div(counter["exact"], counter["total"]),
            "top3": safe_div(counter["top3"], counter["total"]),
            "top5": safe_div(counter["top5"], counter["total"]),
        },
        "by_target_type": {},
        "type_confusion": {
            target: {pred: counter["type_confusion"][i][j] for j, pred in enumerate(TYPE_NAMES)}
            for i, target in enumerate(TYPE_NAMES)
        },
        "opportunities": {},
        "discard": {},
    }

    for name in TYPE_NAMES:
        total = counter["target_type_total"][name]
        out["by_target_type"][name] = {
            "total": total,
            "exact_acc": safe_div(counter["target_type_exact"][name], total),
            "type_recall": safe_div(counter["target_type_type_correct"][name], total),
            "top3": safe_div(counter["target_type_top3"][name], total),
            "top5": safe_div(counter["target_type_top5"][name], total),
            "pred_pass_rate": safe_div(counter["target_type_pred_pass"][name], total),
            "pred_hu_rate": safe_div(counter["target_type_pred_hu"][name], total),
        }

    for name, c in counter["opportunity_raw"].items():
        total = c["total"]
        out["opportunities"][name] = {
            "total": total,
            "exact_acc": safe_div(c["exact"], total),
            "label_group_rate": safe_div(c["target_group"], total),
            "pred_group_rate": safe_div(c["pred_group"], total),
            "group_recall": safe_div(c["hit_group"], c["target_group"]),
            "group_precision": safe_div(c["hit_group"], c["pred_group"]),
            "pred_pass_rate": safe_div(c["pred_pass"], total),
            "pred_hu_rate": safe_div(c["pred_hu"], total),
            "pred_play_rate": safe_div(c["pred_play"], total),
            "label_type_hist": c["label_type_hist"],
            "pred_type_hist": c["pred_type_hist"],
        }

    d = counter["discard"]
    th = d["target_honor"]
    out["discard"]["target_honor_play"] = {
        "total": th["total"],
        "exact_acc": safe_div(th["exact"], th["total"]),
        "top3": safe_div(th["top3"], th["total"]),
        "top5": safe_div(th["top5"], th["total"]),
        "pred_honor_rate": safe_div(th["pred_honor"], th["total"]),
        "miss_to_nonhonor_play_rate": safe_div(th["pred_play_nonhonor"], th["total"]),
    }
    lh = d["legal_honor"]
    out["discard"]["legal_honor_play"] = {
        "total": lh["total"],
        "label_honor_rate": safe_div(lh["label_honor"], lh["total"]),
        "pred_honor_rate": safe_div(lh["pred_honor"], lh["total"]),
        "exact_when_label_honor": safe_div(lh["exact_when_label_honor"], lh["label_honor"]),
    }

    for key in ("target_terminal", "legal_terminal", "target_yaojiu", "legal_yaojiu"):
        item = d[key]
        total = item["total"]
        out["discard"][key] = {"total": total}
        for k, v in item.items():
            if k != "total":
                out["discard"][key][k + "_rate"] = safe_div(v, total)

    out["discard"]["honor_tiles"] = {
        tile: {
            "target": item["target"],
            "exact_acc": safe_div(item["exact"], item["target"]),
            "pred_when_target_rate": safe_div(item["pred_when_target"], item["target"]),
        }
        for tile, item in d["honor_tiles"].items()
    }
    return out


def evaluate(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.model_module, args.ckpt, device)
    ds = MahjongGBDataset(args.begin, args.end, augment=False)
    if args.max_samples and args.max_samples < len(ds):
        ds = Subset(ds, range(args.max_samples))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    counter = init_counter()
    with torch.no_grad():
        for batch_id, (obs, mask, act) in enumerate(loader):
            act = act.to(device).long()
            mask = mask.to(device).float()
            logits, _type_logits = forward_logits(model, obs, mask, device)
            pred = logits.argmax(dim=1)
            topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            top3_hit = (topk[:, : min(3, topk.shape[1])] == act.unsqueeze(1)).any(dim=1)
            top5_hit = (topk == act.unsqueeze(1)).any(dim=1)
            exact = pred == act
            target_type = action_type_id(act)
            pred_type = action_type_id(pred)

            n = act.numel()
            counter["total"] += n
            counter["exact"] += int(exact.sum().item())
            counter["top3"] += int(top3_hit.sum().item())
            counter["top5"] += int(top5_hit.sum().item())

            conf = torch.bincount(target_type * len(TYPE_NAMES) + pred_type, minlength=64).view(8, 8)
            for i, target in enumerate(TYPE_NAMES):
                counter["target_type_total"][target] += int((target_type == i).sum().item())
                counter["target_type_exact"][target] += int((exact & (target_type == i)).sum().item())
                counter["target_type_type_correct"][target] += int(
                    ((pred_type == i) & (target_type == i)).sum().item()
                )
                counter["target_type_top3"][target] += int((top3_hit & (target_type == i)).sum().item())
                counter["target_type_top5"][target] += int((top5_hit & (target_type == i)).sum().item())
                counter["target_type_pred_pass"][target] += int(
                    ((pred_type == TYPE_TO_ID["Pass"]) & (target_type == i)).sum().item()
                )
                counter["target_type_pred_hu"][target] += int(
                    ((pred_type == TYPE_TO_ID["Hu"]) & (target_type == i)).sum().item()
                )
                for j in range(8):
                    counter["type_confusion"][i][j] += int(conf[i, j].item())

            exposed_names = ["Chi", "Peng", "Gang", "BuGang"]
            respond_meld_names = ["Chi", "Peng", "Gang"]
            set_names = ["Chi", "Peng", "Gang", "AnGang", "BuGang"]
            kong_names = ["Gang", "AnGang", "BuGang"]

            target_exposed = type_in(target_type, exposed_names)
            pred_exposed = type_in(pred_type, exposed_names)
            target_respond_meld = type_in(target_type, respond_meld_names)
            pred_respond_meld = type_in(pred_type, respond_meld_names)
            target_set = type_in(target_type, set_names)
            pred_set = type_in(pred_type, set_names)
            target_kong = type_in(target_type, kong_names)
            pred_kong = type_in(pred_type, kong_names)

            respond_sample = mask[:, ACTION_RANGES["Pass"][0]] > 0
            play_sample = legal_any(mask, ["Play"])
            add_opportunity(
                counter,
                "respond_meld_opportunity",
                respond_sample & legal_any(mask, respond_meld_names),
                target_respond_meld,
                pred_respond_meld,
                target_type,
                pred_type,
                exact,
            )
            add_opportunity(
                counter,
                "open_meld_opportunity",
                legal_any(mask, exposed_names),
                target_exposed,
                pred_exposed,
                target_type,
                pred_type,
                exact,
            )
            add_opportunity(
                counter,
                "set_opportunity_including_angang",
                legal_any(mask, set_names),
                target_set,
                pred_set,
                target_type,
                pred_type,
                exact,
            )
            add_opportunity(
                counter,
                "kong_opportunity",
                legal_any(mask, kong_names),
                target_kong,
                pred_kong,
                target_type,
                pred_type,
                exact,
            )

            for name in ("Chi", "Peng", "Gang", "AnGang", "BuGang"):
                add_opportunity(
                    counter,
                    name.lower() + "_legal_opportunity",
                    legal_any(mask, [name]),
                    target_type == TYPE_TO_ID[name],
                    pred_type == TYPE_TO_ID[name],
                    target_type,
                    pred_type,
                    exact,
                )

            pred_honor = action_in(pred, HONOR_PLAY_ACTIONS)
            act_honor = action_in(act, HONOR_PLAY_ACTIONS)
            legal_honor = play_sample & legal_actions(mask, HONOR_PLAY_ACTIONS)
            pred_terminal = action_in(pred, TERMINAL_PLAY_ACTIONS)
            act_terminal = action_in(act, TERMINAL_PLAY_ACTIONS)
            legal_terminal = play_sample & legal_actions(mask, TERMINAL_PLAY_ACTIONS)
            pred_yaojiu = action_in(pred, YAOJIU_PLAY_ACTIONS)
            act_yaojiu = action_in(act, YAOJIU_PLAY_ACTIONS)
            legal_yaojiu = play_sample & legal_actions(mask, YAOJIU_PLAY_ACTIONS)

            d = counter["discard"]
            th = d["target_honor"]
            th["total"] += int(act_honor.sum().item())
            th["exact"] += int((act_honor & exact).sum().item())
            th["top3"] += int((act_honor & top3_hit).sum().item())
            th["top5"] += int((act_honor & top5_hit).sum().item())
            th["pred_honor"] += int((act_honor & pred_honor).sum().item())
            th["pred_play_nonhonor"] += int(
                (act_honor & (pred_type == TYPE_TO_ID["Play"]) & ~pred_honor).sum().item()
            )

            lh = d["legal_honor"]
            lh["total"] += int(legal_honor.sum().item())
            lh["label_honor"] += int((legal_honor & act_honor).sum().item())
            lh["pred_honor"] += int((legal_honor & pred_honor).sum().item())
            lh["exact_when_label_honor"] += int((legal_honor & act_honor & exact).sum().item())

            for key, target_flag, pred_flag, legal_flag in [
                ("terminal", act_terminal, pred_terminal, legal_terminal),
                ("yaojiu", act_yaojiu, pred_yaojiu, legal_yaojiu),
            ]:
                target_item = d["target_" + key]
                target_item["total"] += int(target_flag.sum().item())
                target_item["exact"] += int((target_flag & exact).sum().item())
                target_item["pred_" + key] += int((target_flag & pred_flag).sum().item())
                legal_item = d["legal_" + key]
                legal_item["total"] += int(legal_flag.sum().item())
                legal_item["label_" + key] += int((legal_flag & target_flag).sum().item())
                legal_item["pred_" + key] += int((legal_flag & pred_flag).sum().item())

            for tile_id in HONOR_TILE_IDS:
                action = 2 + tile_id
                tile = TILE_NAMES[tile_id]
                target_tile = act == action
                item = d["honor_tiles"][tile]
                item["target"] += int(target_tile.sum().item())
                item["exact"] += int((target_tile & exact).sum().item())
                item["pred_when_target"] += int((target_tile & (pred == action)).sum().item())

            if args.progress and batch_id % args.progress == 0:
                print("batch", batch_id, "samples", counter["total"])

    return finalize(counter)


def print_report(result):
    print("\n=== Overall ===")
    print("samples:", result["total"])
    print("exact:", pct(result["overall"]["exact_acc"]))
    print("top3 :", pct(result["overall"]["top3"]))
    print("top5 :", pct(result["overall"]["top5"]))

    print("\n=== By Target Type ===")
    print("type      total    exact   type_recall  top3    pass_when_label")
    for name in TYPE_NAMES:
        item = result["by_target_type"][name]
        print(
            "%-8s %7d  %7s  %11s  %6s  %15s"
            % (
                name,
                item["total"],
                pct(item["exact_acc"]),
                pct(item["type_recall"]),
                pct(item["top3"]),
                pct(item["pred_pass_rate"]),
            )
        )

    print("\n=== Meld Opportunities ===")
    print("name                             total label_rate pred_rate recall precision pass_rate exact")
    for name in [
        "respond_meld_opportunity",
        "open_meld_opportunity",
        "set_opportunity_including_angang",
        "kong_opportunity",
        "chi_legal_opportunity",
        "peng_legal_opportunity",
        "gang_legal_opportunity",
        "angang_legal_opportunity",
        "bugang_legal_opportunity",
    ]:
        item = result["opportunities"].get(name, empty_summary())
        print(
            "%-32s %6d %10s %9s %6s %9s %9s %6s"
            % (
                name,
                item["total"],
                pct(item["label_group_rate"]),
                pct(item["pred_group_rate"]),
                pct(item["group_recall"]),
                pct(item["group_precision"]),
                pct(item["pred_pass_rate"]),
                pct(item["exact_acc"]),
            )
        )

    print("\n=== Honor / Big Tile Discard ===")
    target_honor = result["discard"]["target_honor_play"]
    legal_honor = result["discard"]["legal_honor_play"]
    print(
        "target honor discard: total=%d exact=%s top3=%s pred_honor=%s miss_to_nonhonor_play=%s"
        % (
            target_honor["total"],
            pct(target_honor["exact_acc"]),
            pct(target_honor["top3"]),
            pct(target_honor["pred_honor_rate"]),
            pct(target_honor["miss_to_nonhonor_play_rate"]),
        )
    )
    print(
        "legal honor discard : total=%d label_honor=%s pred_honor=%s exact_when_label=%s"
        % (
            legal_honor["total"],
            pct(legal_honor["label_honor_rate"]),
            pct(legal_honor["pred_honor_rate"]),
            pct(legal_honor["exact_when_label_honor"]),
        )
    )
    print("honor tile breakdown:")
    for tile, item in result["discard"]["honor_tiles"].items():
        print(
            "  %-2s target=%6d exact=%s pred_when_target=%s"
            % (tile, item["target"], pct(item["exact_acc"]), pct(item["pred_when_target_rate"]))
        )


def main():
    configure_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-module", default="model_v6")
    parser.add_argument("--ckpt", default="model/checkpoint/v6_refine2_095/mahjong_v6_best.pkl")
    parser.add_argument("--begin", type=float, default=0.95)
    parser.add_argument("--end", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--progress", type=int, default=0)
    args = parser.parse_args()

    result = evaluate(args)
    result["config"] = vars(args)
    print_report(result)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("\nwrote:", args.output)


if __name__ == "__main__":
    main()
