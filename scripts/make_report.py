#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from qd_suite.eval.reporting import write_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create report.md from a run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing metrics.json")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics.json at {metrics_path}")
    data = json.loads(metrics_path.read_text())
    summary = data.get("summary", {})
    curves = data.get("curves", {})
    write_report(run_dir, summary, curves=curves)
    print(f"Wrote report to {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
