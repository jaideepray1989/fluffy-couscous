from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..adapters.base import ModelAdapter
from ..data.dataset import QuickDrawDataset, Sample, load_from_root
from ..repr.pointseq import Point, PointSequence
from ..repr.pointseq import PenState
from ..repr.pointseq import flatten_strokes
from ..transforms import affine, dropout, noise, prefix, reorder, resample, segment, simplify
from . import confusion, metrics, scoring
from .reporting import write_metrics_json, write_report


TRANSFORM_REGISTRY = {
    "prefix_points": prefix.prefix_points,
    "prefix_strokes": prefix.prefix_strokes,
    "stroke_dropout": dropout.stroke_dropout,
    "point_dropout": dropout.point_dropout,
    "resample_uniform": resample.resample_uniform,
    "resample_arclength": resample.resample_arclength,
    "subsample_ratio": resample.subsample_ratio,
    "simplify_rdp": simplify.simplify_rdp,
    "simplify_ratio": simplify.simplify_ratio,
    "split_stroke": segment.split_stroke,
    "merge_strokes": segment.merge_strokes,
    "penup_noise": segment.penup_noise,
    "reverse_strokes": reorder.reverse_strokes,
    "local_shuffle_strokes": reorder.local_shuffle_strokes,
    "reverse_points_within_stroke": reorder.reverse_points_within_stroke,
    "time_warp": reorder.time_warp,
    "jitter_xy": noise.jitter_xy,
    "quantize_xy": noise.quantize_xy,
    "timestamp_jitter": noise.timestamp_jitter,
    "translate": affine.translate,
    "scale": affine.scale,
    "rotate": affine.rotate,
    "start_point_shift": affine.start_point_shift,
}


def param_grid(params: Dict[str, List]) -> Iterable[Dict]:
    if not params:
        yield {}
        return
    keys = sorted(params.keys())
    values = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def load_adapter(module_path: str) -> ModelAdapter:
    spec = importlib.util.spec_from_file_location("adapter_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load adapter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, "get_adapter"):
        raise ValueError("Adapter module must expose get_adapter()")
    return module.get_adapter()


def synthetic_samples() -> List[Sample]:
    # Simple two-stroke sketches to keep tests lightweight.
    cat_strokes = [
        [(0.0, 0.0, 0.0), (1.0, 0.2, 1.0), (1.2, 1.0, 2.0)],
        [(0.2, 1.0, 3.0), (0.0, 0.0, 4.0)],
    ]
    dog_strokes = [
        [(0.0, 0.0, 0.0), (-1.0, 0.2, 1.0), (-1.0, 1.0, 2.0)],
        [(-0.2, 1.0, 3.0), (0.0, 0.0, 4.0)],
    ]
    samples = [
        Sample(sequence=flatten_strokes(cat_strokes, metadata={"word": "cat"}), label="cat"),
        Sample(sequence=flatten_strokes(dog_strokes, metadata={"word": "dog"}), label="dog"),
    ]
    return samples


def load_dataset(data_dir: str | None, classes: List[str] | None, limit: int | None) -> List[Sample]:
    if data_dir:
        class_list = classes or []
        if not class_list:
            class_list = [p.stem for p in Path(data_dir).glob("*.ndjson")]
        dataset = load_from_root(Path(data_dir), class_list, limit_per_class=limit)
        return dataset.samples
    return synthetic_samples()


def _apply_transform(seq: PointSequence, name: str, params: Dict, seed: int) -> PointSequence:
    fn = TRANSFORM_REGISTRY[name]
    kwargs = dict(params)
    if "seed" in fn.__code__.co_varnames and "seed" not in kwargs:
        kwargs["seed"] = seed
    return fn(seq, **kwargs)


def run_suite(samples: List[Sample], adapter: ModelAdapter, spec: Dict, seed: int = 0) -> Tuple[Dict, Dict, Dict]:
    truth = [s.label for s in samples]
    sequences = [s.sequence for s in samples]
    clean_pred = adapter.predict(sequences)
    clean_acc = metrics.accuracy(truth, clean_pred)

    per_family_scores: Dict[str, List[float]] = {}
    curves: Dict[str, list] = {}
    test_results: Dict[str, List[Dict]] = {}

    for test in spec.get("tests", []):
        name = test["name"]
        params = test.get("params", {})
        fn = TRANSFORM_REGISTRY.get(name)
        if fn is None:
            continue
        for combo in param_grid(params):
            transformed_batch = [_apply_transform(seq, name, combo, seed=seed) for seq in sequences]
            pred = adapter.predict(transformed_batch)
            acc = metrics.accuracy(truth, pred)
            per_family_scores.setdefault(name, []).append(acc)
            test_results.setdefault(name, []).append({"params": combo, "acc": acc})
            if name.startswith("prefix"):
                key = combo.get("k")
                if key is not None:
                    curves.setdefault(name, []).append((key, acc))

    summary = metrics.aggregate_metrics(per_family_scores, clean_acc)
    weights = spec.get("scoring", {}).get("weights", {})
    summary["clean_acc"] = clean_acc
    summary["overall_score"] = scoring.weighted_score({k: v for k, v in summary.items() if k.endswith("_robustness")}, weights)

    confusion_sets = spec.get("confusion_sets", [])
    confusion_metrics = []
    for cluster in confusion_sets:
        confusion_metrics.append({"name": cluster.get("name", ""), **confusion.summarize_cluster(truth, clean_pred, cluster)})
    metrics_dict = {
        "clean": {"acc": clean_acc, "clean_acc": clean_acc},
        "tests": test_results,
        "summary": summary,
        "confusion": confusion_metrics,
    }
    return metrics_dict, summary, curves


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QuickDraw capability suite.")
    parser.add_argument("--spec", required=True, help="Path to suite spec JSON")
    parser.add_argument("--model", required=True, help="Path to adapter module implementing get_adapter()")
    parser.add_argument("--data", help="Directory containing class ndjson files")
    parser.add_argument("--classes", help="Comma-separated class names to load")
    parser.add_argument("--limit", type=int, help="Limit number of samples per class")
    parser.add_argument("--run-dir", default="runs/latest", help="Output directory for metrics/report")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    spec_data = json.loads(Path(args.spec).read_text())
    adapter = load_adapter(args.model)
    classes = args.classes.split(",") if args.classes else None
    samples = load_dataset(args.data, classes, args.limit)
    metrics_dict, summary, curves = run_suite(samples, adapter, spec_data)
    run_dir = Path(args.run_dir)
    write_metrics_json(run_dir, metrics_dict)
    write_report(run_dir, summary, curves=curves)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
