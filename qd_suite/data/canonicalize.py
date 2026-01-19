from __future__ import annotations

from typing import Iterable, List

from .parse_ndjson import RawSketch
from ..repr.normalize import normalize_time, normalize_xy
from ..repr.pointseq import PointSequence, flatten_strokes, ensure_monotonic_time


def to_pointseq(raw: RawSketch, normalize_xy_first: bool = True, normalize_time_mode: str = "relative") -> PointSequence:
    """Convert a RawSketch into the canonical PointSequence representation."""
    seq = flatten_strokes(raw.strokes, metadata={"key_id": raw.key_id, "word": raw.word})
    seq = seq.copy_with(ensure_monotonic_time(seq.points))
    if normalize_xy_first:
        seq = normalize_xy(seq)
    if normalize_time_mode:
        seq = normalize_time(seq, mode=normalize_time_mode)
    return seq


def batch_to_pointseq(raws: Iterable[RawSketch], **kwargs) -> List[PointSequence]:
    return [to_pointseq(sketch, **kwargs) for sketch in raws]
