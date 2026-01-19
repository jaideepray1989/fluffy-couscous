from __future__ import annotations

import math
import random
from typing import List

from ..repr.pointseq import PointSequence
from .common import seq_from_strokes, strokes_from_seq


def split_stroke(seq: PointSequence, k: int, seed: int | None = None) -> PointSequence:
    rng = random.Random(seed)
    strokes = []
    for stroke in strokes_from_seq(seq):
        if len(stroke) < k:
            strokes.append(stroke)
            continue
        cut_points = sorted(rng.sample(range(1, len(stroke)), k - 1))
        start = 0
        for cp in cut_points + [len(stroke)]:
            strokes.append(stroke[start:cp])
            start = cp
    return seq_from_strokes(strokes, metadata=dict(seq.metadata), renumber_time=True)


def merge_strokes(seq: PointSequence, threshold: float) -> PointSequence:
    strokes_in = strokes_from_seq(seq)
    if not strokes_in:
        return seq
    merged: List[List] = [strokes_in[0]]
    for stroke in strokes_in[1:]:
        last = merged[-1]
        if not last or not stroke:
            merged.append(stroke)
            continue
        dx = stroke[0].x - last[-1].x
        dy = stroke[0].y - last[-1].y
        if math.hypot(dx, dy) < threshold:
            merged[-1].extend(stroke)
        else:
            merged.append(stroke)
    return seq_from_strokes(merged, metadata=dict(seq.metadata), renumber_time=True)


def penup_noise(seq: PointSequence, drop_prob: float = 0.2, insert_prob: float = 0.2, seed: int | None = None) -> PointSequence:
    rng = random.Random(seed)
    strokes = strokes_from_seq(seq)
    noisy: List[List] = []
    for stroke in strokes:
        noisy.append(stroke)
        # Maybe insert an empty gap as noise.
        if rng.random() < insert_prob:
            noisy.append([])
    # Drop empty strokes introduced by noise or small fragments.
    noisy = [s for s in noisy if s]
    # Optionally drop separators by merging strokes.
    filtered: List[List] = []
    skip_next = False
    for idx, stroke in enumerate(noisy):
        if skip_next:
            skip_next = False
            continue
        if rng.random() < drop_prob and idx + 1 < len(noisy):
            combined = stroke + noisy[idx + 1]
            filtered.append(combined)
            skip_next = True
        else:
            filtered.append(stroke)
    return seq_from_strokes(filtered, metadata=dict(seq.metadata), renumber_time=True)
