from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..repr.pointseq import PointSequence


class ModelAdapter(ABC):
    """Thin wrapper so any stroke-sequence model can plug into the suite."""

    @abstractmethod
    def predict(self, batch: List[PointSequence]) -> List[str]:
        """Return one predicted class label per input sequence."""
        raise NotImplementedError
