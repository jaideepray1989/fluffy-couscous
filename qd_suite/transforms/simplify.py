from __future__ import annotations

import math
from typing import List

from ..repr.pointseq import Point, PointSequence
from .common import seq_from_strokes, strokes_from_seq


def _perp_distance(p: Point, start: Point, end: Point) -> float:
    if start.x == end.x and start.y == end.y:
        return math.hypot(p.x - start.x, p.y - start.y)
    num = abs((end.y - start.y) * p.x - (end.x - start.x) * p.y + end.x * start.y - end.y * start.x)
    den = math.hypot(end.y - start.y, end.x - start.x)
    return num / den


def _rdp(stroke: List[Point], epsilon: float) -> List[Point]:
    if len(stroke) < 3:
        return stroke
    start, end = stroke[0], stroke[-1]
    max_dist = -1.0
    idx = 0
    for i, p in enumerate(stroke[1:-1], start=1):
        dist = _perp_distance(p, start, end)
        if dist > max_dist:
            max_dist = dist
            idx = i
    if max_dist > epsilon:
        left = _rdp(stroke[: idx + 1], epsilon)
        right = _rdp(stroke[idx:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def simplify_rdp(seq: PointSequence, epsilon: float) -> PointSequence:
    strokes = [_rdp(stroke, epsilon) for stroke in strokes_from_seq(seq)]
    return seq_from_strokes(strokes, metadata=dict(seq.metadata), renumber_time=True)


def simplify_ratio(seq: PointSequence, ratio: float) -> PointSequence:
    strokes = []
    for stroke in strokes_from_seq(seq):
        target = max(2, int(len(stroke) * ratio))
        if len(stroke) <= target:
            strokes.append(stroke)
        else:
            step = max(1, len(stroke) // target)
            strokes.append([p for i, p in enumerate(stroke) if i % step == 0][:target])
    return seq_from_strokes(strokes, metadata=dict(seq.metadata), renumber_time=True)
