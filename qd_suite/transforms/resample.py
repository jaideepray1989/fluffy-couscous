from __future__ import annotations

import math
import random
from typing import List

from ..repr.pointseq import Point
from ..repr.pointseq import PointSequence
from .common import seq_from_strokes, strokes_from_seq


def _interp(p0: Point, p1: Point, alpha: float) -> Point:
    return Point(
        x=p0.x + (p1.x - p0.x) * alpha,
        y=p0.y + (p1.y - p0.y) * alpha,
        t=p0.t + (p1.t - p0.t) * alpha,
        pen=p0.pen,
    )


def _resample_uniform(stroke: List[Point], n: int) -> List[Point]:
    if not stroke:
        return []
    if len(stroke) == 1:
        return [Point(stroke[0].x, stroke[0].y, stroke[0].t + i, stroke[0].pen) for i in range(n)]
    resampled: List[Point] = []
    for i in range(n):
        pos = i * (len(stroke) - 1) / max(1, n - 1)
        idx = int(math.floor(pos))
        frac = pos - idx
        if idx >= len(stroke) - 1:
            resampled.append(stroke[-1])
        else:
            resampled.append(_interp(stroke[idx], stroke[idx + 1], frac))
    return resampled


def _arc_lengths(stroke: List[Point]) -> List[float]:
    dists = [0.0]
    for i in range(1, len(stroke)):
        dx = stroke[i].x - stroke[i - 1].x
        dy = stroke[i].y - stroke[i - 1].y
        dist = math.hypot(dx, dy)
        dists.append(dists[-1] + dist)
    return dists


def _resample_arclength(stroke: List[Point], n: int) -> List[Point]:
    if not stroke:
        return []
    if len(stroke) == 1:
        return [Point(stroke[0].x, stroke[0].y, stroke[0].t + i, stroke[0].pen) for i in range(n)]
    dists = _arc_lengths(stroke)
    total = dists[-1]
    if total == 0:
        return _resample_uniform(stroke, n)
    targets = [i * total / max(1, n - 1) for i in range(n)]
    resampled: List[Point] = []
    j = 0
    for tgt in targets:
        while j < len(dists) - 2 and dists[j + 1] < tgt:
            j += 1
        span = max(dists[j + 1] - dists[j], 1e-9)
        alpha = (tgt - dists[j]) / span
        resampled.append(_interp(stroke[j], stroke[j + 1], alpha))
    return resampled


def resample_uniform(seq: PointSequence, n_points: int | None = None, N: int | None = None) -> PointSequence:
    target = n_points if n_points is not None else N
    if target is None:
        raise ValueError("n_points (N) is required")
    strokes = [_resample_uniform(stroke, target) for stroke in strokes_from_seq(seq)]
    return seq_from_strokes(strokes, metadata=dict(seq.metadata), renumber_time=True)


def resample_arclength(seq: PointSequence, n_points: int | None = None, N: int | None = None) -> PointSequence:
    target = n_points if n_points is not None else N
    if target is None:
        raise ValueError("n_points (N) is required")
    strokes = [_resample_arclength(stroke, target) for stroke in strokes_from_seq(seq)]
    return seq_from_strokes(strokes, metadata=dict(seq.metadata), renumber_time=True)


def subsample_ratio(seq: PointSequence, ratio: float | None = None, seed: int | None = None, r: float | None = None) -> PointSequence:
    target_ratio = ratio if ratio is not None else r
    if target_ratio is None:
        raise ValueError("ratio (r) is required")
    rng = random.Random(seed)
    strokes = strokes_from_seq(seq)
    new_strokes: List[List[Point]] = []
    for stroke in strokes:
        if not stroke:
            continue
        keep_n = max(1, int(math.ceil(len(stroke) * target_ratio)))
        indices = list(range(len(stroke)))
        rng.shuffle(indices)
        keep_idx = sorted(indices[:keep_n])
        new_strokes.append([stroke[i] for i in keep_idx])
    return seq_from_strokes(new_strokes, metadata=dict(seq.metadata), renumber_time=True)
