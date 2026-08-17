# OCR Benchmark Command Cheatsheet

File nay gom cac lenh hay dung de clone, chay benchmark, finetune model, submit Kaggle jobs, tai ket qua va commit code.

## 1. Clone Repo

```powershell
git clone https://github.com/huanight19RaH/pdf_ocr_benchmark.git
cd pdf_ocr_benchmark
```

Neu dang dung repo local hien tai:

```powershell
cd D:\THStudy\UniversityStudy\extra_classes\ocr_benchmark
```

## 2. Cai Moi Moi Truong Local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Test Nhanh Local

```powershell
python -m pytest -q
python -m compileall -q src kaggle_runner_ui scripts
```

## 4. Chay Smoke Test Benchmark

```powershell
python -m ocr_benchmark.benchmark `
  --config configs/kaggle_doclaynet_science.yaml `
  --engines noop `
  --limit 3 `
  --output-dir outputs/local_smoke
```

## 5. Chay Benchmark Local Mot Engine

Docling:

```powershell
python -m pip install -r requirements-kaggle-docling.txt
python -m ocr_benchmark.benchmark `
  --config configs/kaggle_doclaynet_science.yaml `
  --engines docling `
  --limit 3 `
  --output-dir outputs/local_docling
```

PaddleOCR:

```powershell
python -m pip install -r requirements-kaggle-paddleocr.txt
python -m ocr_benchmark.benchmark `
  --config configs/kaggle_doclaynet_science.yaml `
  --engines paddleocr `
  --limit 3 `
  --output-dir outputs/local_paddleocr
```

Surya:

```powershell
python -m pip install -r requirements-kaggle-surya.txt
python -m ocr_benchmark.benchmark `
  --config configs/kaggle_doclaynet_science.yaml `
  --engines surya `
  --limit 3 `
  --output-dir outputs/local_surya
```

PaddleOCR-VL:

```powershell
python -m pip install -r requirements-kaggle-paddleocr-vl.txt
python -m ocr_benchmark.benchmark `
  --config configs/kaggle_doclaynet_science.yaml `
  --engines paddleocr_vl `
  --limit 3 `
  --output-dir outputs/local_paddleocr_vl
```

## 6. Chay Finetuning & Benchmark So Sanh (PaddleOCR Pretrained vs. Finetuned)

### 6.1 Chay Finetune Local:

```powershell
# Chuan bi dataset va train 10 epochs tren DocLayNet scientific_articles:
python -m ocr_benchmark.finetune `
  --config configs/kaggle_doclaynet_science.yaml `
  --models paddleocr `
  --limit 20 `
  --epochs 10 `
  --output-dir outputs/local_finetune `
  --execute

# Benchmark so sanh Pretrained baseline vs Finetuned model:
python -m ocr_benchmark.benchmark `
  --config configs/kaggle_doclaynet_science.yaml `
  --engines paddleocr paddleocr_ft `
  --limit 20 `
  --output-dir outputs/local_finetune_eval
```

### 6.2 Chay Finetune tren Kaggle qua Orchestrator:

```powershell
# Chay rieng job paddleocr-ft:
python scripts/kaggle_orchestrator.py --config configs/kaggle_accounts.yaml --job paddleocr-ft --action push
python scripts/kaggle_orchestrator.py --config configs/kaggle_accounts.yaml --job paddleocr-ft --action status
python scripts/kaggle_orchestrator.py --config configs/kaggle_accounts.yaml --job paddleocr-ft --action output
```

## 7. Chay Portable Kaggle Runner UI

```powershell
cd kaggle_runner_ui
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mo browser theo URL Streamlit hien ra, thuong la:

```text
http://localhost:8501
```

## 8. Chuan Bi Config Kaggle Accounts

Neu chua co file config rieng:

```powershell
copy configs\kaggle_accounts.example.yaml configs\kaggle_accounts.yaml
```

Sua `configs/kaggle_accounts.yaml`:

```yaml
repo_url: https://github.com/huanight19RaH/pdf_ocr_benchmark.git
limit: 20
machine_shape: NvidiaTeslaT4
jobs:
  - name: docling
    username: your_kaggle_username_1
    token_dir: C:/Users/YOU/.kaggle/account1
    engines: [docling]
    install_files:
      - requirements-kaggle-docling.txt

  - name: paddleocr-ft
    username: your_kaggle_username_2
    token_dir: C:/Users/YOU/.kaggle/account2
    job_type: finetune
    finetune_model: paddleocr
    epochs: 10
    engines: [paddleocr, paddleocr_ft]
    install_files:
      - requirements-kaggle-paddleocr.txt
```

Moi `token_dir` can co file:

```text
kaggle.json
```

## 9. Submit Jobs Len Kaggle Bang Script Windows

Chay tung lenh rieng trong PowerShell, khong copy ca dau prompt `PS ...>`:

```powershell
.\scripts\windows\01_prepare_jobs.bat
.\scripts\windows\02_push_jobs.bat
.\scripts\windows\03_check_status.bat
.\scripts\windows\04_download_outputs.bat
```

## 10. Merge Ket Qua Benchmark & Tao Bao Cao

```powershell
python scripts\build_latest_report.py `
  --input-dir kaggle_remote_jobs\outputs `
  --output-dir outputs\final_benchmark_report_latest
```

Hoac dung file bat:

```powershell
.\scripts\windows\07_build_latest_report.bat
```

File can xem:

```text
outputs/final_benchmark_report_latest/combined_summary.csv
outputs/final_benchmark_report_latest/latest_benchmark_report.xlsx
outputs/final_benchmark_report_latest/combined_errors.csv
outputs/final_benchmark_report_latest/combined_finetune_status.csv
outputs/final_benchmark_report_latest/LATEST_BENCHMARK_REPORT.md
```

## 11. Commit Va Push Len GitHub

```powershell
git status --short
git diff
git add .
git commit -m "feat: complete 10-epoch DocLayNet finetuning pipeline and reporting"
git push
```
