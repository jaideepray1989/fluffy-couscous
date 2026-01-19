from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def accuracy(truth: Sequence[str], pred: Sequence[str]) -> float:
    if not truth:
        return 0.0
    correct = sum(1 for t, p in zip(truth, pred) if t == p)
    return correct / len(truth)


def curve_auc(points: Iterable[Tuple[float, float]]) -> float:
    pts = sorted(points, key=lambda x: x[0])
    if len(pts) < 2:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def robustness_score(clean_acc: float, degraded: Iterable[float], weights: Iterable[float] | None = None) -> float:
    values = list(degraded)
    if not values:
        return 0.0
    if clean_acc == 0:
        return 0.0
    if weights is None:
        weights = [1.0] * len(values)
    weighted = sum(v / clean_acc * w for v, w in zip(values, weights))
    total_w = sum(weights)
    return weighted / total_w if total_w else 0.0


def aggregate_metrics(per_family: Dict[str, List[float]], clean_acc: float) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for family, scores in per_family.items():
        summary[f"{family}_mean"] = sum(scores) / len(scores) if scores else 0.0
        summary[f"{family}_robustness"] = robustness_score(clean_acc, scores)
    return summary
