#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/kaggle_doclaynet_science.yaml}"
BENCH_OUT="${BENCH_OUT:-/kaggle/working/ocr_benchmark_outputs}"
FT_OUT="${FT_OUT:-/kaggle/working/ocr_finetune_outputs}"
LIMIT="${LIMIT:-20}"
EPOCHS="${EPOCHS:-10}"
PRETRAINED_ENGINES="${PRETRAINED_ENGINES:-docling paddleocr_vl paddleocr surya omnidocbench protonx_legal_tc}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python -m ocr_benchmark.benchmark \
  --config "${CONFIG}" \
  --engines ${PRETRAINED_ENGINES} \
  --limit "${LIMIT}" \
  --output-dir "${BENCH_OUT}/pretrained_all"

python -m ocr_benchmark.finetune \
  --config "${CONFIG}" \
  --models paddleocr docling paddleocr_vl surya omnidocbench protonx_legal_tc \
  --limit "${LIMIT}" \
  --epochs "${EPOCHS}" \
  --execute \
  --output-dir "${FT_OUT}"

if [ -d "${FT_OUT}/paddleocr/inference" ]; then
  python -m ocr_benchmark.benchmark \
    --config "${CONFIG}" \
    --engines paddleocr_ft \
    --limit "${LIMIT}" \
    --output-dir "${BENCH_OUT}/finetuned_paddleocr"
fi
