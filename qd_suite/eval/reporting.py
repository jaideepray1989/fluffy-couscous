from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def write_report(run_dir: Path | str, summary: Dict[str, float], curves: Dict[str, list] | None = None) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    lines = ["# QuickDraw Capability Report", ""]
    lines.append("## Headline metrics")
    for k, v in sorted(summary.items()):
        lines.append(f"- {k}: {v:.4f}")
    if curves:
        lines.append("")
        lines.append("## Curves")
        for name, pts in curves.items():
            lines.append(f"- {name}: {pts}")
    report_path = run_path / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_metrics_json(run_dir: Path | str, metrics: Dict) -> Path:
    path = Path(run_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
