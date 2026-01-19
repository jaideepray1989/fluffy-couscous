from __future__ import annotations

import random
from typing import List

from ..repr.pointseq import PointSequence
from .common import seq_from_strokes, strokes_from_seq


def stroke_dropout(seq: PointSequence, p: float | None = None, seed: int | None = None, fraction: float | None = None) -> PointSequence:
    drop = fraction if fraction is not None else p
    if drop is None:
        raise ValueError("p (drop fraction) is required")
    rng = random.Random(seed)
    strokes = strokes_from_seq(seq)
    kept: List[List] = []
    for stroke in strokes:
        if rng.random() >= drop:
            kept.append(stroke)
    if not kept and strokes:
        kept.append(strokes[0])
    return seq_from_strokes(kept, metadata=dict(seq.metadata))


def point_dropout(seq: PointSequence, p: float | None = None, seed: int | None = None, fraction: float | None = None) -> PointSequence:
    drop = fraction if fraction is not None else p
    if drop is None:
        raise ValueError("p (drop fraction) is required")
    rng = random.Random(seed)
    strokes = strokes_from_seq(seq)
    new_strokes: List[List] = []
    for stroke in strokes:
        filtered = [pt for pt in stroke if rng.random() >= drop]
        if filtered:
            new_strokes.append(filtered)
    if not new_strokes and strokes:
        new_strokes.append([strokes[0][0]])
    return seq_from_strokes(new_strokes, metadata=dict(seq.metadata))
