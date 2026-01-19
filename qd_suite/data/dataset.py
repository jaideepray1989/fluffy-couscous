from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from .canonicalize import to_pointseq
from .parse_ndjson import RawSketch, iter_sketches
from ..repr.pointseq import PointSequence


def ndjson_for_class(root: Path, class_name: str) -> Path:
    return root / f"{class_name}.ndjson"


@dataclass
class Sample:
    sequence: PointSequence
    label: str


class QuickDrawDataset:
    """
    Lightweight loader for QuickDraw raw ndjson files.
    """

    def __init__(
        self,
        paths: Sequence[Path | str],
        limit: Optional[int] = None,
        normalize_xy_first: bool = True,
        normalize_time_mode: str = "relative",
    ):
        self.samples: List[Sample] = []
        for path in paths:
            for raw in iter_sketches(path):
                seq = to_pointseq(raw, normalize_xy_first=normalize_xy_first, normalize_time_mode=normalize_time_mode)
                self.samples.append(Sample(sequence=seq, label=raw.word))
                if limit is not None and len(self.samples) >= limit:
                    break
            if limit is not None and len(self.samples) >= limit:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)


def load_from_root(root: Path | str, classes: Iterable[str], limit_per_class: Optional[int] = None) -> QuickDrawDataset:
    root_path = Path(root)
    paths: List[Path] = []
    for name in classes:
        candidate = ndjson_for_class(root_path, name)
        if candidate.exists():
            paths.append(candidate)
    return QuickDrawDataset(paths=paths, limit=limit_per_class)
