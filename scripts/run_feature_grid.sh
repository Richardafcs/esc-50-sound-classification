#!/usr/bin/env bash
# Run 12 prespecified cases x 3 seeds on a user-provided ESC-50 dataset.
# Do not silently skip, overwrite, or mix unfinished evidence.
set -euo pipefail
MODE="${MODE:-cnn}"
EPOCHS="${EPOCHS:-60}"
DEVICE="${DEVICE:-auto}"
RAW_ROOT="${RAW_ROOT:-data/ESC-50}"
CACHE_DIR="${CACHE_DIR:-data/feature_ablation}"
RESULTS_DIR="${RESULTS_DIR:-results/feature_ablation}"
for case in $(PYTHONPATH=src python -c 'from esc50_sound.feature_ablation import GRID; print(" ".join(s.name for s in GRID))'); do
  for seed in 101 202 303; do
    echo "Running ${MODE} ${case} seed=${seed}" >&2
    PYTHONPATH=src python scripts/feature_ablation.py train \
      --raw-root "$RAW_ROOT" --cache-dir "$CACHE_DIR" \
      --results-dir "$RESULTS_DIR" --case "$case" \
      --model "$MODE" --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE"
  done
done
