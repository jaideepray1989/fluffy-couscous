#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qd_suite.data.dataset import load_from_root
from qd_suite.models.dataset import PointSeqDataset, collate_pad
from qd_suite.models.transformer_encoder import StrokeTransformer, TransformerConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train a transformer encoder on stroke sequences.")
    p.add_argument("--data", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--limit-per-class", type=int, default=1000)
    p.add_argument("--offset-per-class", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--run-dir", default="runs/transformer")
    return p.parse_args()


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    ds = load_from_root(args.data, classes, limit_per_class=args.limit_per_class, offset_per_class=args.offset_per_class)
    dataset = PointSeqDataset(ds.samples, class_to_idx={c: i for i, c in enumerate(classes)}, include_time=True, deltas=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)

    cfg = TransformerConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        num_classes=len(classes),
        include_time=True,
        deltas=True,
    )
    model = StrokeTransformer(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, lengths, labels in loader:
            pad_mask = torch.arange(x.size(1))[None, :].expand(x.size(0), -1) >= lengths[:, None]
            x, pad_mask, labels = x.to(device), pad_mask.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(x, src_key_padding_mask=pad_mask)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"epoch {epoch+1}/{args.epochs} loss={total_loss/total:.4f} acc={correct/total:.3f}")
        torch.save({"model_state": model.state_dict(), "config": cfg.__dict__, "classes": classes}, run_dir / "transformer.pt")


if __name__ == "__main__":
    main()
