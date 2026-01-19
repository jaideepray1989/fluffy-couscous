from __future__ import annotations

import math
from typing import List

from ..repr.pointseq import PenState, Point, PointSequence, ensure_monotonic_time
from .common import seq_from_strokes, strokes_from_seq


def prefix_points(seq: PointSequence, k: float | None = None, k_pct: float | None = None) -> PointSequence:
    """Keep first k% of non-separator points."""
    pct = k_pct if k_pct is not None else k
    if pct is None:
        raise ValueError("k (percentage) is required")
    total_down = sum(1 for p in seq.points if p.pen == PenState.DOWN)
    target = max(1, math.ceil(total_down * pct / 100.0))
    kept: List[Point] = []
    down_seen = 0
    for pt in seq.points:
        if pt.pen == PenState.DOWN:
            down_seen += 1
            kept.append(pt)
            if down_seen >= target:
                break
        else:
            if down_seen > 0:
                kept.append(pt)
    return seq.copy_with(ensure_monotonic_time(kept))


def prefix_strokes(seq: PointSequence, k: float | None = None, k_pct: float | None = None) -> PointSequence:
    pct = k_pct if k_pct is not None else k
    if pct is None:
        raise ValueError("k (percentage) is required")
    strokes = strokes_from_seq(seq)
    if not strokes:
        return seq
    target = max(1, math.ceil(len(strokes) * pct / 100.0))
    selected = strokes[:target]
    return seq_from_strokes(selected, metadata=dict(seq.metadata))
