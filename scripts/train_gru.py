#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from qd_suite.data.dataset import load_from_root
from qd_suite.models.dataset import PointSeqDataset, collate_pad
from qd_suite.models.gru_classifier import GRUConfig, build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train a simple GRU classifier on QuickDraw sequences.")
    p.add_argument("--data", required=True, help="Directory containing class ndjson files")
    p.add_argument("--classes", required=True, help="Comma-separated class names")
    p.add_argument("--limit-per-class", type=int, default=2000, help="Limit samples per class for quick training")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--run-dir", default="runs/gru_baseline")
    return p.parse_args()


def make_loaders(dataset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    return train_loader, val_loader


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, lengths, labels in loader:
        x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, lengths, labels in loader:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    dataset = load_from_root(args.data, classes, limit_per_class=args.limit_per_class)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ds = PointSeqDataset(dataset.samples, class_to_idx, include_time=True, deltas=True)
    train_loader, val_loader = make_loaders(ds, args.batch_size)

    cfg = GRUConfig(hidden_dim=args.hidden_dim, num_layers=args.layers, include_time=True, deltas=True)
    model = build_model(num_classes=len(classes), config=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(f"epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} acc={train_acc:.3f} val_loss={val_loss:.4f} acc={val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "config": cfg.__dict__,
                },
                run_dir / "gru_checkpoint.pt",
            )
    # Save final for reference
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "config": cfg.__dict__,
        },
        run_dir / "gru_last.pt",
    )


if __name__ == "__main__":
    main()
