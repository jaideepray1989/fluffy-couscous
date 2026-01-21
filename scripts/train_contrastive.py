#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qd_suite.data.dataset import load_from_root
from qd_suite.models.contrastive import ContrastiveConfig, ContrastiveModel, pad_batch


def parse_args():
    p = argparse.ArgumentParser(description="Train a contrastive encoder (triplet loss) for QuickDraw.")
    p.add_argument("--data", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--limit-per-class", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--run-dir", default="runs/contrastive")
    return p.parse_args()


def sample_triplets(samples, class_to_idx, k=64):
    rng = random.Random(0)
    batch = []
    for _ in range(k):
        anchor = rng.choice(samples)
        positives = [s for s in samples if s.label == anchor.label and s is not anchor]
        negatives = [s for s in samples if s.label != anchor.label]
        if not positives or not negatives:
            continue
        pos = rng.choice(positives)
        neg = rng.choice(negatives)
        batch.append((anchor.sequence, pos.sequence, neg.sequence))
    return batch


def collate_triplets(batch):
    anchors, positives, negatives = zip(*batch)
    a, la = pad_batch(anchors)
    p, lp = pad_batch(positives)
    n, ln = pad_batch(negatives)
    return a, la, p, lp, n, ln


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    dataset = load_from_root(args.data, classes, limit_per_class=args.limit_per_class)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cfg = ContrastiveConfig(hidden_dim=args.hidden_dim, proj_dim=args.proj_dim, num_layers=args.layers)
    model = ContrastiveModel(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        batch = sample_triplets(dataset.samples, class_to_idx, k=args.batch_size)
        if not batch:
            continue
        a, la, p, lp, n, ln = collate_triplets(batch)
        a, la, p, lp, n, ln = a.to(device), la.to(device), p.to(device), lp.to(device), n.to(device), ln.to(device)
        opt.zero_grad()
        za = model.embed(a, la)
        zp = model.embed(p, lp)
        zn = model.embed(n, ln)
        loss = model.loss(za, zp, zn)
        loss.backward()
        opt.step()
        print(f"epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}")

    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__, "class_to_idx": class_to_idx}, run_dir / "contrastive.pt")


if __name__ == "__main__":
    main()
