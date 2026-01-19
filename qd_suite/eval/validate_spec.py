from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

REQUIRED_KEYS = {"version", "seed_policy", "canonical_representation", "tests", "confusion_sets", "scoring"}


def validate_spec(spec: Dict) -> None:
    missing = REQUIRED_KEYS - spec.keys()
    if missing:
        raise ValueError(f"Spec missing keys: {missing}")
    if not isinstance(spec.get("tests"), list):
        raise ValueError("tests must be a list")
    for test in spec["tests"]:
        if "name" not in test or "params" not in test:
            raise ValueError(f"Invalid test entry: {test}")
    if not isinstance(spec.get("confusion_sets"), list):
        raise ValueError("confusion_sets must be a list")


def pretty_print(spec: Dict) -> str:
    return json.dumps(spec, indent=2, sort_keys=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a capability suite spec file.")
    parser.add_argument("spec", help="Path to JSON spec")
    args = parser.parse_args(argv)
    spec_data = json.loads(Path(args.spec).read_text())
    validate_spec(spec_data)
    print(pretty_print(spec_data))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
