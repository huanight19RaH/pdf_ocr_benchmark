#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/kaggle_doclaynet_science.yaml}"
OUT="${OUT:-/kaggle/working/ocr_benchmark_outputs}"
LIMIT="${LIMIT:-20}"
ENGINES="${ENGINES:-docling paddleocr_vl paddleocr surya}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python -m ocr_benchmark.benchmark \
  --config "${CONFIG}" \
  --engines ${ENGINES} \
  --limit "${LIMIT}" \
  --output-dir "${OUT}"
