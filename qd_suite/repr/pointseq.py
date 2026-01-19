from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class PenState(str, Enum):
    DOWN = "DOWN"
    UP = "UP"


@dataclass
class Point:
    x: float
    y: float
    t: float
    pen: PenState


@dataclass
class PointSequence:
    points: List[Point] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def strokes(self) -> List[List[Point]]:
        """Split the flat sequence into strokes based on UP separators."""
        strokes: List[List[Point]] = []
        current: List[Point] = []
        for pt in self.points:
            if pt.pen == PenState.DOWN:
                current.append(pt)
            else:
                if current:
                    strokes.append(current)
                    current = []
        if current:
            strokes.append(current)
        return strokes

    def copy_with(self, points: List[Point]) -> "PointSequence":
        return PointSequence(points=points, metadata=dict(self.metadata))


def flatten_strokes(
    strokes: Sequence[Sequence[Tuple[float, float, float]]],
    metadata: Optional[Dict[str, object]] = None,
    penup_time: str = "carry",
) -> PointSequence:
    """
    Build a PointSequence from strokes (each point: (x, y, t)).

    penup_time:
        carry: reuse last point's timestamp for the separator.
        plus_one: add +1 relative step for each separator.
    """
    pts: List[Point] = []
    meta = metadata or {}
    current_time = 0.0
    for idx, stroke in enumerate(strokes):
        for x, y, t in stroke:
            current_time = max(current_time, float(t))
            pts.append(Point(float(x), float(y), current_time, PenState.DOWN))
        if idx < len(strokes) - 1:
            if penup_time == "plus_one":
                current_time += 1.0
            pts.append(Point(pts[-1].x, pts[-1].y, current_time, PenState.UP))
    return PointSequence(points=pts, metadata=meta)


def ensure_monotonic_time(points: Iterable[Point]) -> List[Point]:
    """Return a copy with non-decreasing timestamps."""
    fixed: List[Point] = []
    last_t = 0.0
    for pt in points:
        t = max(last_t, float(pt.t))
        fixed.append(Point(pt.x, pt.y, t, pt.pen))
        last_t = t
    return fixed


def relabel_time(points: Iterable[Point], start_at: float = 0.0, step: float = 1.0) -> List[Point]:
    """Assign monotonically increasing timestamps with a fixed step."""
    relabeled: List[Point] = []
    t = start_at
    for pt in points:
        relabeled.append(Point(pt.x, pt.y, t, pt.pen))
        t += step
    return relabeled
