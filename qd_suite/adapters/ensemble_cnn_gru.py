from __future__ import annotations

import os
from collections import Counter
from typing import List

from qd_suite.adapters.base import ModelAdapter
from qd_suite.adapters.cnn_raster_adapter import CNNRasterAdapter
from qd_suite.adapters.gru_adapter import GRUAdapter
from qd_suite.repr.pointseq import PointSequence


class EnsembleCnnGruAdapter(ModelAdapter):
    """
    Simple ensemble of raster CNN and stroke-sequence GRU.
    Voting: majority; tie breaks in favor of CNN prediction.
    """

    def __init__(self, cnn_ckpt: str, gru_ckpt: str):
        self.cnn = CNNRasterAdapter(cnn_ckpt)
        self.gru = GRUAdapter(gru_ckpt)

    def predict(self, batch: List[PointSequence]) -> List[str]:
        cnn_preds = self.cnn.predict(batch)
        gru_preds = self.gru.predict(batch)
        preds = []
        for c, g in zip(cnn_preds, gru_preds):
            if c == g:
                preds.append(c)
            else:
                preds.append(c)  # tie-break to CNN
        return preds


def get_adapter():
    cnn_ckpt = os.environ.get("CNN_CHECKPOINT", "runs/cnn_raster_v2/cnn.pt")
    gru_ckpt = os.environ.get("GRU_CHECKPOINT", "runs/gru_small_v2/gru_checkpoint.pt")
    return EnsembleCnnGruAdapter(cnn_ckpt, gru_ckpt)
