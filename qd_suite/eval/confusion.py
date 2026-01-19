from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


def cluster_accuracy(truth: Sequence[str], pred: Sequence[str], classes: Iterable[str]) -> float:
    mask = [i for i, t in enumerate(truth) if t in classes]
    if not mask:
        return 0.0
    filtered_truth = [truth[i] for i in mask]
    filtered_pred = [pred[i] for i in mask]
    correct = sum(1 for t, p in zip(filtered_truth, filtered_pred) if t == p)
    return correct / len(filtered_truth)


def pair_accuracy(truth: Sequence[str], pred: Sequence[str], pair: Tuple[str, str]) -> float:
    a, b = pair
    mask = [i for i, t in enumerate(truth) if t in pair]
    if not mask:
        return 0.0
    filtered_truth = [truth[i] for i in mask]
    filtered_pred = [pred[i] for i in mask]
    correct = sum(1 for t, p in zip(filtered_truth, filtered_pred) if t == p)
    return correct / len(filtered_truth)


def confusion_entropy(truth: Sequence[str], pred: Sequence[str], classes: Iterable[str]) -> float:
    mask = [i for i, t in enumerate(truth) if t in classes]
    if not mask:
        return 0.0
    filtered_pred = [pred[i] for i in mask]
    counts = Counter(filtered_pred)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + 1e-12, 2)
    return entropy


def summarize_cluster(truth: Sequence[str], pred: Sequence[str], cluster: Dict) -> Dict[str, float]:
    classes = cluster.get("classes", [])
    pairs = cluster.get("pairs", [])
    metrics = {
        "cluster_acc": cluster_accuracy(truth, pred, classes),
        "confusion_entropy": confusion_entropy(truth, pred, classes),
    }
    worst = 1.0
    for pair in pairs:
        acc = pair_accuracy(truth, pred, tuple(pair))
        worst = min(worst, acc)
    metrics["worst_pair_acc"] = worst if pairs else 0.0
    return metrics
