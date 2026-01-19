from __future__ import annotations

import random
from typing import List

from ..repr.pointseq import Point, PointSequence, ensure_monotonic_time


def jitter_xy(seq: PointSequence, sigma: float, seed: int | None = None) -> PointSequence:
    rng = random.Random(seed)
    jittered: List[Point] = []
    for pt in seq.points:
        dx = rng.gauss(0.0, sigma)
        dy = rng.gauss(0.0, sigma)
        jittered.append(Point(pt.x + dx, pt.y + dy, pt.t, pt.pen))
    return seq.copy_with(jittered)


def quantize_xy(seq: PointSequence, bits: int) -> PointSequence:
    levels = max(2, 2**bits)
    quantized: List[Point] = []
    for pt in seq.points:
        step = 2 / (levels - 1)
        qx = round(pt.x / step) * step
        qy = round(pt.y / step) * step
        quantized.append(Point(qx, qy, pt.t, pt.pen))
    return seq.copy_with(quantized)


def timestamp_jitter(seq: PointSequence, sigma: float, seed: int | None = None) -> PointSequence:
    rng = random.Random(seed)
    jittered = [
        Point(pt.x, pt.y, pt.t + rng.gauss(0.0, sigma), pt.pen)
        for pt in seq.points
    ]
    return seq.copy_with(ensure_monotonic_time(jittered))
