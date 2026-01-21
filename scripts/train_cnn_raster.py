#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from qd_suite.data.dataset import load_from_root
from qd_suite.models.cnn_raster import CNNConfig, SimpleCNN
from qd_suite.utils.rasterize import rasterize


class RasterDataset(torch.utils.data.Dataset):
    def __init__(self, samples, size: int = 64, stroke_width: int = 2, padding: int = 2):
        self.samples = samples
        self.size = size
        self.stroke_width = stroke_width
        self.padding = padding
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = rasterize(sample.sequence, size=self.size, stroke_width=self.stroke_width, padding=self.padding)
        tensor = self.to_tensor(img)
        return tensor, sample.label


def parse_args():
    p = argparse.ArgumentParser(description="Train a simple CNN on rasterized QuickDraw sketches.")
    p.add_argument("--data", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--limit-per-class", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--run-dir", default="runs/cnn_raster")
    return p.parse_args()


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    dataset = load_from_root(args.data, classes, limit_per_class=args.limit_per_class)
    train_loader = DataLoader(RasterDataset(dataset.samples), batch_size=args.batch_size, shuffle=True)
    cfg = CNNConfig(num_classes=len(classes))
    model = SimpleCNN(cfg)
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
        for x, y in train_loader:
            x = x.to(device)
            y_idx = torch.tensor([classes.index(lbl) for lbl in y], device=device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y_idx)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y_idx).sum().item()
            total += x.size(0)
        print(f"epoch {epoch+1}/{args.epochs} loss={total_loss/total:.4f} acc={correct/total:.3f}")
        torch.save({"model_state": model.state_dict(), "classes": classes}, run_dir / "cnn.pt")


if __name__ == "__main__":
    main()
