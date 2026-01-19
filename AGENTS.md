# Agent Setup

- Python: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .[dev,plotting]`
- Tests: `pytest`
- Smoke run: `python scripts/run_suite.py --spec qd_suite/eval/default_spec.json --model qd_suite/adapters/example_dummy.py`

# Constraints

- Evaluation is vector-only; avoid rasterization when adding features or examples.
- Keep transforms deterministic when a seed is provided.

# Notes

- QuickDraw raw ndjson is the canonical data source (x, y, t per stroke).
- Model adapters live in `qd_suite/adapters/` and expose `get_adapter()`.
