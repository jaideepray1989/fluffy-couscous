from __future__ import annotations

import torch

import os

from qd_suite.adapters.base import ModelAdapter
from qd_suite.models.dataset import pad_batch
from qd_suite.models.gru_classifier import GRUConfig, build_model
from qd_suite.repr.pointseq import PointSequence


class GRUAdapter(ModelAdapter):
    def __init__(self, checkpoint_path: str, device: str | None = None):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.class_to_idx = ckpt["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        cfg = GRUConfig(**ckpt["config"])
        self.model = build_model(num_classes=len(self.class_to_idx), config=cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.include_time = cfg.include_time
        self.deltas = cfg.deltas

    def predict(self, batch: list[PointSequence]) -> list[str]:
        tokens, lengths = pad_batch(batch, include_time=self.include_time, deltas=self.deltas)
        tokens = tokens.to(self.device)
        lengths = lengths.to(self.device)
        with torch.no_grad():
            logits = self.model(tokens, lengths)
            preds = logits.argmax(dim=1).cpu().tolist()
        return [self.idx_to_class[i] for i in preds]


def get_adapter():
    ckpt = os.environ.get("GRU_CHECKPOINT") or "runs/gru_baseline/gru_checkpoint.pt"
    if not os.path.exists(ckpt):
        raise RuntimeError("Set GRU_CHECKPOINT env var to a trained checkpoint path.")
    return GRUAdapter(checkpoint_path=ckpt)
