from __future__ import annotations

import random
from typing import List

from qd_suite.adapters.base import ModelAdapter
from qd_suite.repr.pointseq import PointSequence


class ExampleDummyAdapter(ModelAdapter):
    """
    Dummy adapter that predicts the ground-truth label from metadata when available.
    Fallback: random label from the observed batch labels.
    """

    def predict(self, batch: List[PointSequence]) -> List[str]:
        labels = [seq.metadata.get("word", "unknown") for seq in batch]
        fallback = list({lbl for lbl in labels if lbl != "unknown"}) or ["unknown"]
        pred = []
        rng = random.Random(0)
        for lbl in labels:
            if lbl != "unknown":
                pred.append(lbl)
            else:
                pred.append(rng.choice(fallback))
        return pred


def get_adapter() -> ModelAdapter:
    return ExampleDummyAdapter()
