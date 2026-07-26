#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/RAD}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

export HF_HOME HF_HUB_CACHE HF_HUB_OFFLINE

"${PYTHON_BIN}" scripts/run_official_anomalib_patchcore_sanity.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --backbone wide_resnet50_2 \
  --disable-local-weights \
  --train-batch-size 32 \
  --eval-batch-size 32 \
  --num-workers 8
