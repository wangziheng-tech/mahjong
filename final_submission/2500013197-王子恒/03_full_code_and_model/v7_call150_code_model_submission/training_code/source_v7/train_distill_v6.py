import argparse
import importlib
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_rl_v2 import MahjongGBRLDataset
from train_cnn_v2 import augment_batch, build_perm_tensors, save_state_dict


def configure_stdout():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(module_name, ckpt_path, device, trainable):
    model = importlib.import_module(module_name).CNNModel().to(device)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.train(mode=trainable)
    for p in model.parameters():
        p.requires_grad_(trainable)
    return model


def action_type_id(action):
    action = action.long()
    out = torch.empty_like(action)
    out[action == 0] = 0
    out[action == 1] = 1
    out[(2 <= action) & (action < 36)] = 2
    out[(36 <= action) & (action < 99)] = 3
    out[(99 <= action) & (action < 133)] = 4
    out[(133 <= action) & (action < 167)] = 5
    out[(167 <= action) & (action < 201)] = 6
    out[action >= 201] = 7
    return out


def action_type_name(action):
    action = int(action)
    if action == 0:
        return "Pass"
    if action == 1:
        return "Hu"
    if action < 36:
        return "Play"
    if action < 99:
        return "Chi"
    if action < 133:
        return "Peng"
    if action < 167:
        return "Gang"
    if action < 201:
        return "AnGang"
    return "BuGang"


def make_input(obs, mask, device, is_training):
    return {
        "is_training": is_training,
        "obs": {
            "observation": obs.to(device).float(),
            "action_mask": mask.to(device).float(),
        },
    }


def distill_loss(student_logits, teacher_logits, temperature):
    t = max(float(temperature), 1e-6)
    log_prob = F.log_softmax(student_logits / t, dim=1)
    target_prob = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(log_prob, target_prob, reduction="batchmean") * (t * t)


def evaluate(model, loader, device):
    model.eval()
    total = correct = top3 = top5 = 0
    value_mse_sum = 0.0
    value_l1_sum = 0.0
    type_total = Counter()
    type_correct = Counter()
    group_total = Counter()
    group_correct = Counter()
    honor_total = 0
    honor_correct = 0
    with torch.no_grad():
        for obs, mask, act, reward, player in loader:
            act = act.to(device).long()
            reward = reward.to(device).float()
            logits, type_logits, value = model.forward_all(make_input(obs, mask, device, False))
            pred = logits.argmax(dim=1)
            k = min(5, logits.shape[1])
            topk = logits.topk(k=k, dim=1).indices
            total += act.numel()
            correct += (pred == act).sum().item()
            top3 += (topk[:, :min(3, k)] == act.unsqueeze(1)).any(dim=1).sum().item()
            top5 += (topk == act.unsqueeze(1)).any(dim=1).sum().item()
            value_mse_sum += F.mse_loss(value, reward, reduction="sum").item()
            value_l1_sum += F.l1_loss(value, reward, reduction="sum").item()
            act_cpu = act.detach().cpu().numpy()
            pred_cpu = pred.detach().cpu().numpy()
            reward_cpu = reward.detach().cpu().numpy()
            for y, p, r in zip(act_cpu, pred_cpu, reward_cpu):
                typ = action_type_name(y)
                type_total[typ] += 1
                type_correct[typ] += int(y == p)
                grp = "win" if r > 0 else "lose"
                group_total[grp] += 1
                group_correct[grp] += int(y == p)
                if 28 <= int(y) <= 34:
                    honor_total += 1
                    honor_correct += int(y == p)
    return {
        "acc": correct / total,
        "top3": top3 / total,
        "top5": top5 / total,
        "value_mse": value_mse_sum / total,
        "value_l1": value_l1_sum / total,
        "group_acc": {k: group_correct[k] / max(group_total[k], 1) for k in sorted(group_total)},
        "type_acc": {k: type_correct[k] / max(type_total[k], 1) for k in sorted(type_total)},
        "honor_discard_label_acc": honor_correct / max(honor_total, 1),
        "honor_discard_label_total": honor_total,
    }


def main():
    configure_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", type=float, default=0.95)
    parser.add_argument("--val-begin", type=float, default=0.95)
    parser.add_argument("--val-end", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", default="model/checkpoint/v6_distill")
    parser.add_argument("--model-module", default="model_v6")
    parser.add_argument("--init-ckpt", default="")
    parser.add_argument("--teacher-module", default="model_blend_v45")
    parser.add_argument("--teacher-ckpt", default="model/checkpoint/blend_v45_alpha_75.pkl")
    parser.add_argument("--reward-dir", default="data_reward_v5")
    parser.add_argument("--reward-scale", type=float, default=64.0)
    parser.add_argument("--distill-coef", type=float, default=0.45)
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--type-coef", type=float, default=0.08)
    parser.add_argument("--value-coef", type=float, default=0.06)
    parser.add_argument("--advantage-beta", type=float, default=0.0)
    parser.add_argument("--advantage-clip", type=float, default=2.0)
    parser.add_argument("--augment-mode", choices=["batch", "dataset", "none"], default="batch")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--legacy-save", action="store_true")
    parser.add_argument("--save-prefix", default="mahjong_v6")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("torch:", torch.__version__)
    print("model_module:", args.model_module)
    print("teacher_module:", args.teacher_module)
    print("teacher_ckpt:", args.teacher_ckpt)
    print("distill_coef:", args.distill_coef)
    print("distill_temp:", args.distill_temp)
    print("type_coef:", args.type_coef)
    print("value_coef:", args.value_coef)
    print("advantage_beta:", args.advantage_beta)
    print("augment_mode:", args.augment_mode)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_ds = MahjongGBRLDataset(0, args.train_end, args.augment_mode == "dataset", args.reward_dir, args.reward_scale)
    val_ds = MahjongGBRLDataset(args.val_begin, args.val_end, False, args.reward_dir, args.reward_scale)
    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))

    pin_memory = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = load_model(args.model_module, args.init_ckpt, device, True)
    teacher = None
    if args.distill_coef > 0 and args.teacher_ckpt:
        teacher = load_model(args.teacher_module, args.teacher_ckpt, device, False)
        teacher.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    perm_tensors = build_perm_tensors(device) if args.augment_mode == "batch" else None
    best_score = -1e9

    for epoch in range(args.epochs):
        model.train()
        loss_sum = policy_sum = distill_sum = type_sum = value_sum = 0.0
        for i, (obs, mask, act, reward, player) in enumerate(train_loader):
            obs = obs.to(device, non_blocking=True).float()
            mask = mask.to(device, non_blocking=True).float()
            act = act.to(device, non_blocking=True).long()
            reward = reward.to(device, non_blocking=True).float()
            obs, mask, act = augment_batch(obs, mask, act, perm_tensors)
            input_train = make_input(obs, mask, device, True)
            logits, type_logits, value = model.forward_all(input_train)
            ce = F.cross_entropy(logits, act, reduction="none")
            if args.advantage_beta > 0:
                adv = reward - reward.mean()
                weights = torch.exp(args.advantage_beta * adv).clamp(max=args.advantage_clip)
                policy_loss = (ce * weights).mean()
            else:
                policy_loss = ce.mean()
            type_loss = F.cross_entropy(type_logits, action_type_id(act))
            value_loss = F.mse_loss(value, reward)
            d_loss = logits.new_tensor(0.0)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(make_input(obs, mask, device, False))
                d_loss = distill_loss(logits, teacher_logits, args.distill_temp)
            loss = policy_loss + args.distill_coef * d_loss + args.type_coef * type_loss + args.value_coef * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += loss.item()
            policy_sum += policy_loss.item()
            distill_sum += d_loss.item()
            type_sum += type_loss.item()
            value_sum += value_loss.item()
            if i % 50 == 0:
                print(
                    f"epoch={epoch} iter={i}/{len(train_loader)} loss={loss.item():.4f} "
                    f"policy={policy_loss.item():.4f} distill={d_loss.item():.4f} "
                    f"type={type_loss.item():.4f} value={value_loss.item():.4f}"
                )

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        avg = max(1, len(train_loader))
        win_acc = metrics["group_acc"].get("win", 0.0)
        select_score = metrics["acc"] + 0.02 * metrics["top3"] + 0.03 * win_acc
        print(
            f"epoch={epoch} avg_loss={loss_sum/avg:.4f} policy={policy_sum/avg:.4f} "
            f"distill={distill_sum/avg:.4f} type={type_sum/avg:.4f} value={value_sum/avg:.4f} "
            f"val_acc={metrics['acc']:.4f} top3={metrics['top3']:.4f} top5={metrics['top5']:.4f} "
            f"value_mse={metrics['value_mse']:.4f} value_l1={metrics['value_l1']:.4f} "
            f"select_score={select_score:.6f}"
        )
        print("group_acc:", json.dumps(metrics["group_acc"], sort_keys=True))
        print("type_acc:", json.dumps(metrics["type_acc"], sort_keys=True))
        print("honor_discard_label_acc:", metrics["honor_discard_label_acc"], "total:", metrics["honor_discard_label_total"])
        ckpt = os.path.join(args.out_dir, f"{args.save_prefix}_epoch_{epoch}.pkl")
        save_state_dict(model, ckpt, args.legacy_save)
        print("saved:", ckpt)
        if select_score > best_score:
            best_score = select_score
            best = os.path.join(args.out_dir, f"{args.save_prefix}_best.pkl")
            save_state_dict(model, best, args.legacy_save)
            with open(os.path.join(args.out_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            print("saved best:", best, "select_score:", best_score)


if __name__ == "__main__":
    main()
