from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence, Tuple


@dataclass
class RawSketch:
    key_id: str
    word: str
    strokes: List[List[Tuple[float, float, float]]]
    metadata: dict


def _validate_stroke(stroke: Sequence[Sequence[float]]) -> List[Tuple[float, float, float]]:
    if len(stroke) != 3:
        raise ValueError(f"Expected stroke with 3 arrays, got {len(stroke)}")
    xs, ys, ts = stroke
    if not (len(xs) == len(ys) == len(ts)):
        raise ValueError("Mismatched lengths in stroke arrays")
    pts: List[Tuple[float, float, float]] = []
    last_t = None
    for x, y, t in zip(xs, ys, ts):
        if last_t is not None and t < last_t:
            raise ValueError("Timestamps must be monotonic within a stroke")
        pts.append((float(x), float(y), float(t)))
        last_t = t
    return pts


def parse_line(line: str) -> RawSketch:
    obj = json.loads(line)
    drawing = obj.get("drawing")
    if drawing is None:
        raise ValueError("Missing drawing field")
    strokes = [_validate_stroke(stk) for stk in drawing]
    key_id = str(obj.get("key_id", ""))
    word = str(obj.get("word", ""))
    metadata = {k: v for k, v in obj.items() if k not in {"drawing"}}
    return RawSketch(key_id=key_id, word=word, strokes=strokes, metadata=metadata)


def iter_sketches(path: Path | str) -> Generator[RawSketch, None, None]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield parse_line(line)


def load_many(paths: Iterable[Path | str]) -> List[RawSketch]:
    sketches: List[RawSketch] = []
    for path in paths:
        sketches.extend(list(iter_sketches(path)))
    return sketches
