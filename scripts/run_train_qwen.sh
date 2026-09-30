#!/usr/bin/env bash
# Invoke with bash scripts/run_train_qwen.sh ML [extra run_train.py arguments].
set -euo pipefail
DATASET="${1:-ML}"
if [ "$#" -gt 0 ]; then shift; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
python scripts/run_train.py \
  --dataset "$DATASET" \
  --data_dir "data/$DATASET" \
  --config configs/memrec_frozen_protocol.yaml \
  "$@"
