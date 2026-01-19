from __future__ import annotations

from typing import Dict


def weighted_score(family_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    total_w = 0.0
    for name, score in family_scores.items():
        w = weights.get(name, 0.0)
        total += score * w
        total_w += w
    return total / total_w if total_w else 0.0
