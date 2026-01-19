from __future__ import annotations

import random
from typing import List

from ..repr.pointseq import Point, PointSequence
from .common import seq_from_strokes, strokes_from_seq


def reverse_strokes(seq: PointSequence) -> PointSequence:
    strokes = list(reversed(strokes_from_seq(seq)))
    return seq_from_strokes(strokes, metadata=dict(seq.metadata), renumber_time=True)


def local_shuffle_strokes(seq: PointSequence, window: int, seed: int | None = None) -> PointSequence:
    rng = random.Random(seed)
    strokes = strokes_from_seq(seq)
    shuffled: List[List] = []
    for i in range(0, len(strokes), window):
        block = strokes[i : i + window]
        rng.shuffle(block)
        shuffled.extend(block)
    return seq_from_strokes(shuffled, metadata=dict(seq.metadata), renumber_time=True)


def reverse_points_within_stroke(seq: PointSequence) -> PointSequence:
    strokes = [list(reversed(stroke)) for stroke in strokes_from_seq(seq)]
    return seq_from_strokes(strokes, metadata=dict(seq.metadata), renumber_time=True)


def time_warp(seq: PointSequence, factor: float) -> PointSequence:
    strokes = strokes_from_seq(seq)
    warped: List[List[Point]] = []
    for stroke in strokes:
        if not stroke:
            warped.append(stroke)
            continue
        start_t = stroke[0].t
        warped_stroke: List[Point] = []
        for i, pt in enumerate(stroke):
            new_t = start_t + (pt.t - start_t) * factor
            warped_stroke.append(Point(pt.x, pt.y, new_t, pt.pen))
        warped.append(warped_stroke)
    return seq_from_strokes(warped, metadata=dict(seq.metadata))
