from __future__ import annotations

import os
from typing import List

import torch
from torchvision import transforms

from qd_suite.adapters.base import ModelAdapter
from qd_suite.models.cnn_raster import CNNConfig, SimpleCNN
from qd_suite.repr.pointseq import PointSequence
from qd_suite.utils.rasterize import rasterize


class CNNRasterAdapter(ModelAdapter):
    def __init__(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        classes = ckpt["classes"]
        cfg = CNNConfig(num_classes=len(classes))
        self.model = SimpleCNN(cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.classes = classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.to_tensor = transforms.ToTensor()

    def predict(self, batch: List[PointSequence]) -> List[str]:
        imgs = [self.to_tensor(rasterize(seq)) for seq in batch]
        x = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
        return [self.classes[i] for i in preds]

    def predict_proba(self, batch: List[PointSequence]) -> torch.Tensor:
        imgs = [self.to_tensor(rasterize(seq)) for seq in batch]
        x = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            return torch.softmax(logits, dim=1).cpu()


def get_adapter():
    ckpt = os.environ.get("CNN_CHECKPOINT", "runs/cnn_raster/cnn.pt")
    if not os.path.exists(ckpt):
        raise RuntimeError("Set CNN_CHECKPOINT env var to a trained CNN checkpoint.")
    return CNNRasterAdapter(ckpt)
