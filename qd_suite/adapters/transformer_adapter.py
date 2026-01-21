from __future__ import annotations

import os
from typing import List

import torch

from qd_suite.adapters.base import ModelAdapter
from qd_suite.models.transformer_encoder import StrokeTransformer, TransformerConfig
from qd_suite.models.dataset import pad_batch
from qd_suite.repr.pointseq import PointSequence


class TransformerAdapter(ModelAdapter):
    def __init__(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = TransformerConfig(**ckpt["config"])
        self.classes = ckpt["classes"]
        self.model = StrokeTransformer(cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.include_time = cfg.include_time
        self.deltas = cfg.deltas

    def _predict_logits_batch(self, batch: List[PointSequence], batch_size: int = 64) -> torch.Tensor:
        logits_list = []
        for i in range(0, len(batch), batch_size):
            sub = batch[i : i + batch_size]
            x, lengths = pad_batch(sub, include_time=self.include_time, deltas=self.deltas)
            pad_mask = torch.arange(x.size(1))[None, :].expand(x.size(0), -1) >= lengths[:, None]
            x, pad_mask = x.to(self.device), pad_mask.to(self.device)
            with torch.no_grad():
                logits = self.model(x, src_key_padding_mask=pad_mask)
            logits_list.append(logits.cpu())
        return torch.cat(logits_list, dim=0)

    def predict(self, batch: List[PointSequence]) -> List[str]:
        logits = self._predict_logits_batch(batch)
        preds = logits.argmax(dim=1).tolist()
        return [self.classes[i] for i in preds]

    def predict_proba(self, batch: List[PointSequence]) -> torch.Tensor:
        logits = self._predict_logits_batch(batch)
        return torch.softmax(logits, dim=1)


def get_adapter():
    candidates = [
        os.environ.get("TRANSFORMER_CHECKPOINT"),
        "runs/transformer/transformer.pt",
        "runs/transformer_v1/transformer.pt",
    ]
    for ckpt in candidates:
        if ckpt and os.path.exists(ckpt):
            return TransformerAdapter(ckpt)
    raise RuntimeError("Set TRANSFORMER_CHECKPOINT to a trained transformer checkpoint.")
