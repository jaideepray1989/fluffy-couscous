from __future__ import annotations

from typing import Iterable, List, Tuple

from .pointseq import PenState, Point, PointSequence


def _bounds(points: Iterable[Point]) -> Tuple[float, float, float, float]:
    xs = [p.x for p in points if p.pen == PenState.DOWN]
    ys = [p.y for p in points if p.pen == PenState.DOWN]
    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), max(xs), min(ys), max(ys)


def normalize_xy(seq: PointSequence, eps: float = 1e-6) -> PointSequence:
    """Center and scale to fit roughly in [-1, 1]."""
    min_x, max_x, min_y, max_y = _bounds(seq.points)
    width = max(max_x - min_x, eps)
    height = max(max_y - min_y, eps)
    scale = max(width, height) / 2.0
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    normalized: List[Point] = []
    for pt in seq.points:
        if pt.pen == PenState.DOWN:
            normalized.append(Point((pt.x - cx) / scale, (pt.y - cy) / scale, pt.t, pt.pen))
        else:
            normalized.append(Point(pt.x - cx, pt.y - cy, pt.t, pt.pen))
    return PointSequence(points=normalized, metadata=dict(seq.metadata))


def normalize_time(seq: PointSequence, mode: str = "relative") -> PointSequence:
    """Normalize timestamps to start at zero. If mode='unit', also divide by final time."""
    if not seq.points:
        return seq
    start = seq.points[0].t
    shifted: List[Point] = []
    for pt in seq.points:
        shifted.append(Point(pt.x, pt.y, pt.t - start, pt.pen))
    if mode == "unit":
        end = shifted[-1].t
        divisor = end if end > 0 else 1.0
        shifted = [Point(pt.x, pt.y, pt.t / divisor, pt.pen) for pt in shifted]
    return PointSequence(points=shifted, metadata=dict(seq.metadata))
