from __future__ import annotations

import math
from typing import List

from ..repr.pointseq import Point, PointSequence
from .common import seq_from_strokes, strokes_from_seq


def translate(seq: PointSequence, dx: float, dy: float) -> PointSequence:
    shifted = [Point(pt.x + dx, pt.y + dy, pt.t, pt.pen) for pt in seq.points]
    return seq.copy_with(shifted)


def scale(seq: PointSequence, factor: float) -> PointSequence:
    scaled = [Point(pt.x * factor, pt.y * factor, pt.t, pt.pen) for pt in seq.points]
    return seq.copy_with(scaled)


def rotate(seq: PointSequence, degrees: float) -> PointSequence:
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotated = [
        Point(pt.x * cos_t - pt.y * sin_t, pt.x * sin_t + pt.y * cos_t, pt.t, pt.pen)
        for pt in seq.points
    ]
    return seq.copy_with(rotated)


def start_point_shift(seq: PointSequence, shift_pct: float) -> PointSequence:
    strokes = strokes_from_seq(seq)
    shifted: List[List] = []
    for stroke in strokes:
        if not stroke:
            shifted.append(stroke)
            continue
        k = max(0, int(len(stroke) * shift_pct / 100.0)) % len(stroke)
        shifted.append(stroke[k:] + stroke[:k])
    return seq_from_strokes(shifted, metadata=dict(seq.metadata), renumber_time=True)
