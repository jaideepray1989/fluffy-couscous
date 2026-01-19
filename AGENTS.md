# Agent Setup

- Python: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .[dev,plotting]`
- Tests: `pytest`
- Smoke run: `python scripts/run_suite.py --spec qd_suite/eval/default_spec.json --model qd_suite/adapters/example_dummy.py`
- Train baseline GRU: `python scripts/train_gru.py --data data/raw --classes <comma-list> --limit-per-class 2000 --epochs 5`
- Run GRU adapter: `python scripts/run_suite.py --spec qd_suite/eval/default_spec.json --model qd_suite/adapters/gru_adapter.py --data data/raw --classes <comma-list> --limit 500 --run-dir runs/gru_eval` (pass checkpoint path via env or modify adapter to load your file)

# Constraints

- Evaluation is vector-only; avoid rasterization when adding features or examples.
- Keep transforms deterministic when a seed is provided.

# Notes

- QuickDraw raw ndjson is the canonical data source (x, y, t per stroke).
- Model adapters live in `qd_suite/adapters/` and expose `get_adapter()`.
