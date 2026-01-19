from __future__ import annotations

from typing import List, Tuple

from .pointseq import PenState, PointSequence


def to_tokens(seq: PointSequence, include_time: bool = False, deltas: bool = False) -> List[Tuple[float, ...]]:
    """
    Convert a PointSequence to model-ready tokens.

    include_time controls whether timestamps are included.
    deltas uses dx, dy, dt relative encoding; otherwise absolute values.
    """
    tokens: List[Tuple[float, ...]] = []
    prev = None
    for pt in seq.points:
        if deltas and prev is not None:
            dx = pt.x - prev.x
            dy = pt.y - prev.y
            dt = pt.t - prev.t
            if include_time:
                tokens.append((dx, dy, dt, 1.0 if pt.pen == PenState.DOWN else 0.0))
            else:
                tokens.append((dx, dy, 1.0 if pt.pen == PenState.DOWN else 0.0))
        else:
            if include_time:
                tokens.append((pt.x, pt.y, pt.t, 1.0 if pt.pen == PenState.DOWN else 0.0))
            else:
                tokens.append((pt.x, pt.y, 1.0 if pt.pen == PenState.DOWN else 0.0))
        prev = pt
    return tokens
