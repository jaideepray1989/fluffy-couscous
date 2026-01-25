#!/usr/bin/env bash
set -euo pipefail

# Minimal GPU training script for Lightning Studio or any GPU box.
# Avoids editable install; uses PYTHONPATH to load the local package.

VENV_DIR="${VENV_DIR:-.venv}"
DATA_DIR="${DATA_DIR:-data/raw}"
CLASSES="${CLASSES:-airplane,cat,dog,house,tree}"
LIMIT_PER_CLASS="${LIMIT_PER_CLASS:-2000}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys
import torch
print("python:", sys.version.split()[0])
print("cuda:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

export PYTHONPATH="${PYTHONPATH:-.}"

python scripts/train_transformer.py \
  --data "${DATA_DIR}" \
  --classes "${CLASSES}" \
  --limit-per-class "${LIMIT_PER_CLASS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}"
