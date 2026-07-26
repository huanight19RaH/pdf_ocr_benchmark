# DocLayNet OCR Benchmark on Kaggle

Benchmark text OCR/document parsing models on scientific paper pages from DocLayNet. The repo is designed for this workflow:

```text
GitHub repo -> Kaggle Notebook git clone -> run benchmark -> download CSV/report
```

## What This Benchmarks

Default dataset:

```text
docling-project/DocLayNet-v1.1
```

Default subset:

```text
split: val
doc_category: scientific_articles
limit: 20
seed: 42
```

The benchmark uses `pdf_cells` as pseudo-ground-truth text and compares each OCR/parser prediction with normalized reference text.

Metrics:

- `cer`: Character Error Rate, lower is better.
- `wer`: Word Error Rate, lower is better.
- `char_f1`: character bag F1, higher is better.
- `exact_match_normalized`: exact match after normalization.
- `latency_s`: average inference time per page.
- `chars_per_second`: generated normalized characters per second.
- `success_pages`, `failed_pages`, `failure_rate`.

## Model Registry

Runnable OCR engines:

- `docling`: Docling `DocumentConverter`.
- `paddleocr_vl`: PaddleOCR-VL 1.6 via `PaddleOCRVL(pipeline_version="v1.6")`.
- `paddleocr`: PaddleOCR classic baseline.
- `surya`: Surya OCR, Python API first and CLI fallback.
- `noop`: empty prediction smoke test.

Reference/non-OCR links:

- `omnidocbench`: benchmark suite, recorded as skipped.
- `protonx_legal_tc`: legal text classification model, recorded as skipped.

## Run on Kaggle From GitHub

1. Push this repo to GitHub.
2. Create a Kaggle Notebook.
3. Enable Internet and choose GPU T4/P100.
4. Open `notebooks/kaggle_run_from_github.ipynb`.
5. Change this line:

```python
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/ocr_benchmark.git"
```

6. Run all cells.

The notebook first runs:

```bash
PYTHONPATH=src python -m ocr_benchmark.benchmark \
  --config configs/kaggle_doclaynet_science.yaml \
  --engines noop omnidocbench protonx_legal_tc \
  --limit 3 \
  --output-dir /kaggle/working/ocr_benchmark_outputs
```

Then run the full benchmark after installing the OCR engines you want:

```bash
PYTHONPATH=src python -m ocr_benchmark.benchmark \
  --config configs/kaggle_doclaynet_science.yaml \
  --engines docling paddleocr_vl paddleocr surya omnidocbench protonx_legal_tc \
  --limit 20 \
  --output-dir /kaggle/working/ocr_benchmark_outputs
```

You can also run the shell wrapper:

```bash
ENGINES="docling paddleocr_vl paddleocr surya" LIMIT=20 bash scripts/run_kaggle_benchmark.sh
```

## Install Options on Kaggle

Base requirements:

```bash
python -m pip install -r requirements.txt
```

Install only the engines you need:

```bash
python -m pip install -r requirements-kaggle-docling.txt
python -m pip install -r requirements-kaggle-paddleocr.txt
python -m pip install -r requirements-kaggle-surya.txt
python -m pip install -r requirements-kaggle-paddleocr-vl.txt
```

PaddleOCR-VL is heavy. If it fails with CUDA/package conflicts, run it in a fresh Kaggle session and keep the same `limit`, `seed`, and config.

## Output Files

The benchmark writes to `/kaggle/working/ocr_benchmark_outputs` by default:

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

`summary.csv` is the main benchmark table. `benchmark_report.md` contains a readable ranking table plus run notes and top errors.

## Local Checks

```bash
python -m pytest
python -m py_compile src/ocr_benchmark/*.py
python -m json.tool notebooks/kaggle_run_from_github.ipynb
```

## Notes

- This version benchmarks text OCR only, not layout mAP, table accuracy, or reading order.
- DocLayNet is primarily a layout dataset; `pdf_cells` is digital PDF text used as pseudo-ground-truth.
- `docling-project/DocLayNet` is the original asset repository with PNG, COCO annotations, PDFs, and JSON cells. `docling-project/DocLayNet-v1.1` is easier to load directly in Kaggle with Hugging Face `datasets`.
