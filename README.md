# DocLayNet OCR Benchmark + Finetune on Kaggle

Benchmark all listed OCR/document parsing engines on scientific paper pages from DocLayNet, then run a 10-epoch finetune path where public Kaggle-feasible training is available.

```text
GitHub repo -> Kaggle git clone -> benchmark all pretrained models -> finetune -> benchmark finetuned model -> download CSV/report
```

## Dataset and Metrics

Default dataset:

```text
docling-project/DocLayNet-v1.1
split: val
doc_category: scientific_articles
limit: 20
seed: 42
```

The benchmark uses `pdf_cells` as pseudo-ground-truth text and compares each prediction with normalized reference text.

Metrics:

- `cer`: Character Error Rate, lower is better.
- `wer`: Word Error Rate, lower is better.
- `char_f1`: character bag F1, higher is better.
- `exact_match_normalized`: exact match after normalization.
- `latency_s`: average inference time per page.
- `chars_per_second`: generated normalized characters per second.
- `success_pages`, `failed_pages`, `failure_rate`.

## Model Registry

Pretrained benchmark engines:

- `docling`: Docling `DocumentConverter`.
- `paddleocr_vl`: PaddleOCR-VL 1.6.
- `paddleocr`: PaddleOCR classic baseline.
- `surya`: Surya OCR.

Finetuned benchmark engine:

- `paddleocr_ft`: exported PaddleOCR recognition model after 10 epochs.

Reference/non-OCR entries:

- `omnidocbench`: benchmark suite, recorded as skipped.
- `protonx_legal_tc`: legal text classification model, recorded as skipped.

## Run on Kaggle From GitHub

1. Push this folder to GitHub.
2. Create a Kaggle Notebook.
3. Enable Internet and GPU T4/P100.
4. Open or paste cells from `notebooks/kaggle_run_from_github.ipynb`.
5. Change:

```python
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/ocr_benchmark.git"
```

6. Run all cells.

The notebook will:

- install base dependencies and OCR engine dependencies;
- run smoke test with `noop`;
- benchmark all pretrained OCR engines: `docling paddleocr_vl paddleocr surya`;
- record `omnidocbench` and `protonx_legal_tc` as skipped non-OCR/reference entries;
- prepare finetune data for all listed models;
- train PaddleOCR recognition for `10` epochs when `--execute` is enabled;
- export the PaddleOCR inference model;
- benchmark `paddleocr_ft`;
- write `/kaggle/working/combined_benchmark_summary.csv`.

## Kaggle Commands

Benchmark all pretrained models:

```bash
PYTHONPATH=src python -m ocr_benchmark.benchmark \
  --config configs/kaggle_doclaynet_science.yaml \
  --engines docling paddleocr_vl paddleocr surya omnidocbench protonx_legal_tc \
  --limit 20 \
  --output-dir /kaggle/working/ocr_benchmark_outputs/pretrained_all
```

Prepare and run 10-epoch finetune:

```bash
PYTHONPATH=src python -m ocr_benchmark.finetune \
  --config configs/kaggle_doclaynet_science.yaml \
  --models paddleocr docling paddleocr_vl surya omnidocbench protonx_legal_tc \
  --limit 20 \
  --epochs 10 \
  --execute \
  --output-dir /kaggle/working/ocr_finetune_outputs
```

Benchmark exported finetuned PaddleOCR:

```bash
PYTHONPATH=src python -m ocr_benchmark.benchmark \
  --config configs/kaggle_doclaynet_science.yaml \
  --engines paddleocr_ft \
  --limit 20 \
  --output-dir /kaggle/working/ocr_benchmark_outputs/finetuned_paddleocr
```

All-in-one shell wrapper:

```bash
LIMIT=20 EPOCHS=10 bash scripts/run_kaggle_all.sh
```

## Install Options

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-kaggle-docling.txt
python -m pip install -r requirements-kaggle-paddleocr.txt
python -m pip install -r requirements-kaggle-surya.txt
python -m pip install -r requirements-kaggle-paddleocr-vl.txt
```

PaddleOCR-VL is heavy. If it fails due to CUDA/package conflicts, run fewer engines per Kaggle session while keeping the same config/seed/limit.

## Outputs

Benchmark output:

```text
manifest.jsonl
predictions.jsonl
metrics_by_page.csv
summary.csv
errors.jsonl
errors.csv
benchmark_report.md
sample_failures.csv
pages/*.png
```

Finetune output:

```text
finetune_status.csv
finetune_report.md
paddleocr/paddleocr_rec_dataset/
paddleocr/train_command.sh
paddleocr/export_command.sh
paddleocr/train_output/
paddleocr/inference/
```

## Finetune Support

This repo benchmarks all listed OCR engines, but it only trains models with a public Kaggle-feasible training entrypoint:

- `paddleocr`: supported. The runner crops DocLayNet `pdf_cells`, creates PaddleOCR recognition labels, trains for 10 epochs, exports an inference model, then benchmarks `paddleocr_ft`.
- `docling`: skipped for finetune because public tooling is inference-focused, not a stable Kaggle training CLI.
- `paddleocr_vl`: skipped by default for finetune because its SFT stack is separate and too heavy for the single T4/P100 all-in-one path.
- `surya`: skipped for finetune because the public repo does not provide a stable self-service Kaggle finetune command.
- `omnidocbench`: skipped because it is a benchmark suite.
- `protonx_legal_tc`: skipped because it is a legal text classifier, not OCR.

## Local Checks

```bash
python -m pytest
python -m py_compile src/ocr_benchmark/*.py
python -m json.tool notebooks/kaggle_run_from_github.ipynb
```
