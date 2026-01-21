# Agent Setup

- Python: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .[dev,plotting]`
- Tests: `pytest`
- Smoke run: `python scripts/run_suite.py --spec qd_suite/eval/default_spec.json --model qd_suite/adapters/example_dummy.py`
- Train baseline GRU: `python scripts/train_gru.py --data data/raw --classes <comma-list> --limit-per-class 2000 --epochs 5`
- Run GRU adapter: `python scripts/run_suite.py --spec qd_suite/eval/default_spec.json --model qd_suite/adapters/gru_adapter.py --data data/raw --classes <comma-list> --limit 500 --run-dir runs/gru_eval` (pass checkpoint path via env or modify adapter to load your file)
- Train contrastive encoder: `python scripts/train_contrastive.py --data data/raw --classes <comma-list>`
- Run contrastive kNN adapter: set `CONTRASTIVE_CHECKPOINT`, `CONTRASTIVE_SUPPORT_DATA`, `CONTRASTIVE_SUPPORT_CLASSES`, then `python scripts/run_suite.py --spec ... --model qd_suite/adapters/contrastive_knn.py --data ...`
- Train CNN raster baseline (allows rasterization): `python scripts/train_cnn_raster.py --data data/raw --classes <comma-list>`
- Run CNN adapter: set `CNN_CHECKPOINT` and `python scripts/run_suite.py --spec ... --model qd_suite/adapters/cnn_raster_adapter.py --data ...`
- Train transformer stroke encoder: `python scripts/train_transformer.py --data data/raw --classes <comma-list>`
- Run transformer adapter: set `TRANSFORMER_CHECKPOINT` and `python scripts/run_suite.py --spec ... --model qd_suite/adapters/transformer_adapter.py --data ...`
- Probabilistic ensemble (CNN + transformer): set `CNN_CHECKPOINT` and `TRANSFORMER_CHECKPOINT`, then use `qd_suite/adapters/ensemble_prob.py`

# Constraints

- Evaluation is vector-only; avoid rasterization when adding features or examples.
- Keep transforms deterministic when a seed is provided.

# Notes

- QuickDraw raw ndjson is the canonical data source (x, y, t per stroke).
- Model adapters live in `qd_suite/adapters/` and expose `get_adapter()`.
