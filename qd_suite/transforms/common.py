from __future__ import annotations

from typing import List

from ..repr.pointseq import Point, PointSequence, ensure_monotonic_time, flatten_strokes


def strokes_from_seq(seq: PointSequence) -> List[List[Point]]:
    return seq.strokes()


def seq_from_strokes(strokes: List[List[Point]], metadata: dict | None = None, renumber_time: bool = False) -> PointSequence:
    if renumber_time:
        t = 0.0
        flat = []
        for stroke in strokes:
            new_stroke = []
            for pt in stroke:
                new_stroke.append((pt.x, pt.y, t))
                t += 1.0
            flat.append(new_stroke)
    else:
        flat = [[(pt.x, pt.y, pt.t) for pt in stroke] for stroke in strokes]
    seq = flatten_strokes(flat, metadata=metadata or {})
    seq = seq.copy_with(ensure_monotonic_time(seq.points))
    return seq
