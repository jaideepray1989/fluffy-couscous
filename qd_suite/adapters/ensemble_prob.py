from __future__ import annotations

import os
from typing import List

import torch

from qd_suite.adapters.base import ModelAdapter
from qd_suite.adapters.cnn_raster_adapter import CNNRasterAdapter
from qd_suite.adapters.transformer_adapter import TransformerAdapter
from qd_suite.repr.pointseq import PointSequence


class EnsembleProbAdapter(ModelAdapter):
    """Average probabilities from CNN raster and transformer stroke encoder."""

    def __init__(self, cnn_ckpt: str, transformer_ckpt: str):
        self.cnn = CNNRasterAdapter(cnn_ckpt)
        self.transformer = TransformerAdapter(transformer_ckpt)
        if self.cnn.classes != self.transformer.classes:
            raise ValueError("CNN and transformer class lists must match")
        self.classes = self.cnn.classes

    def predict(self, batch: List[PointSequence]) -> List[str]:
        cnn_probs = self.cnn.predict_proba(batch)
        tx_probs = self.transformer.predict_proba(batch)
        avg = (cnn_probs + tx_probs) / 2.0
        preds = avg.argmax(dim=1).tolist()
        return [self.classes[i] for i in preds]


def get_adapter():
    cnn_ckpt = os.environ.get("CNN_CHECKPOINT", "runs/cnn_raster_v2/cnn.pt")
    tx_candidates = [
        os.environ.get("TRANSFORMER_CHECKPOINT"),
        "runs/transformer/transformer.pt",
        "runs/transformer_v1/transformer.pt",
    ]
    tx_ckpt = next((c for c in tx_candidates if c and os.path.exists(c)), None)
    if tx_ckpt is None:
        raise RuntimeError("Set TRANSFORMER_CHECKPOINT to a trained transformer checkpoint.")
    return EnsembleProbAdapter(cnn_ckpt, tx_ckpt)
