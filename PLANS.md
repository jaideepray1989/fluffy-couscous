# Goals

- Implement a deterministic, vector-only capability suite for QuickDraw stroke-sequence models.
- Provide plug-in model adapters and reproducible evaluation outputs (metrics.json, report.md, plots).

# Milestones

- Task 1: Scaffold repo, packaging, and scripts (`python -m qd_suite` imports, `scripts/run_suite.py --help`).
- Task 2: Data download + parsing utilities for QuickDraw raw ndjson.
- Task 3: Canonical point-sequence representation and normalization helpers.
- Task 4: Deterministic transform library for capability tests.
- Task 5: Suite spec schema + runner plumbing with adapter loading.
- Task 6: Metrics, scoring, reporting, and plotting stubs.
- Task 7: Agent handoff utilities (prompt template, spec validation).

# Risks

- Large dataset downloads (mitigate via class subsets, HTTPS fallback).
- Runtime/memory when applying many transforms (mitigate with sampling caps).

# Done Criteria

- Smoke suite run completes on a toy dataset with metrics.json and report.md generated.
- Placeholder tests under `tests/` pass via `pytest`.
