import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_v2 import MahjongGBDataset
from model_v2 import CNNModel


def make_input(obs, mask, device, is_training):
    return {
        "is_training": is_training,
        "obs": {
            "observation": obs.to(device).float(),
            "action_mask": mask.to(device).float(),
        },
    }


def evaluate(model, loader, device):
    model.eval()
    total = correct = top3 = 0
    with torch.no_grad():
        for obs, mask, act in loader:
            act = act.to(device).long()
            logits = model(make_input(obs, mask, device, False))
            pred = logits.argmax(dim=1)
            correct += (pred == act).sum().item()
            topk = logits.topk(k=min(3, logits.shape[1]), dim=1).indices
            top3 += (topk == act.unsqueeze(1)).any(dim=1).sum().item()
            total += act.numel()
    return correct / total, top3 / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", type=float, default=0.01)
    parser.add_argument("--val-begin", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out-dir", default="model/checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("torch:", torch.__version__)

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = MahjongGBDataset(0, args.train_end, True)
    val_ds = MahjongGBDataset(args.val_begin, 1.0, False)
    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0

        for i, (obs, mask, act) in enumerate(train_loader):
            act = act.to(device).long()
            logits = model(make_input(obs, mask, device, True))
            loss = F.cross_entropy(logits, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            if i % 50 == 0:
                print(f"epoch={epoch} iter={i}/{len(train_loader)} loss={loss.item():.4f}")

        acc, top3_acc = evaluate(model, val_loader, device)
        avg_loss = loss_sum / max(1, len(train_loader))
        print(f"epoch={epoch} avg_loss={avg_loss:.4f} val_acc={acc:.4f} top3={top3_acc:.4f}")

        ckpt = os.path.join(args.out_dir, f"cnn_epoch_{epoch}.pkl")
        torch.save(model.state_dict(), ckpt)
        print("saved:", ckpt)

        if acc > best_acc:
            best_acc = acc
            best = os.path.join(args.out_dir, "best.pkl")
            torch.save(model.state_dict(), best)
            print("saved best:", best)


if __name__ == "__main__":
    main()