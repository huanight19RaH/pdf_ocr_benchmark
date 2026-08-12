# OCR Benchmark Command Cheatsheet

File nay gom cac lenh hay dung de clone, chay benchmark, submit Kaggle jobs, tai ket qua va commit code.

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

## 6. Chay Portable Kaggle Runner UI

```powershell
cd kaggle_runner_ui
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mo browser theo URL Streamlit hien ra, thuong la:

```text
http://localhost:8501
```

## 7. Chuan Bi Config Kaggle Accounts

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
```

Moi `token_dir` can co file:

```text
kaggle.json
```

## 8. Submit Jobs Len Kaggle Bang Script Windows

Chay tung lenh rieng trong PowerShell, khong copy ca dau prompt `PS ...>`:

```powershell
.\scripts\windows\01_prepare_jobs.bat
.\scripts\windows\02_push_jobs.bat
.\scripts\windows\03_check_status.bat
.\scripts\windows\04_download_outputs.bat
.\scripts\windows\05_merge_results.bat
```

Neu bi loi current directory da bi xoa tren Kaggle/local shell, chuyen ve folder ton tai truoc:

```powershell
cd D:\THStudy\UniversityStudy\extra_classes\ocr_benchmark
```

## 9. Check Trang Thai Kaggle Job Thu Cong

```powershell
python -m kaggle kernels status OWNER/KERNEL-SLUG
```

Vi du:

```powershell
python -m kaggle kernels status thung192/ocr-docling
```

Dung slug trong URL notebook Kaggle:

```text
https://www.kaggle.com/code/OWNER/KERNEL-SLUG
```

## 10. Merge Ket Qua Benchmark

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
outputs/final_benchmark_report_latest/LATEST_BENCHMARK_REPORT.md
```

## 11. Commit Va Push Len GitHub

Xem file thay doi:

```powershell
git status --short
git diff
```

Commit cac file code can push:

```powershell
git add .
git commit -m "fix: support PaddleOCR Surya and PaddleOCR-VL Kaggle runtimes"
git push
```

Neu chi muon commit file lenh nay:

```powershell
git add COMMANDS.md
git commit -m "docs: add command cheatsheet"
git push
```

## 12. Kaggle Notebook Clone Va Chay Truc Tiep

Cell clone:

```python
import os

REPO_DIR = "/kaggle/working/ocr_benchmark"
os.chdir("/kaggle/working")

if not os.path.exists(REPO_DIR):
    !git clone --depth 1 https://github.com/huanight19RaH/pdf_ocr_benchmark.git {REPO_DIR}

os.chdir(REPO_DIR)
```

Cell install base:

```python
!python -m pip install -q -r requirements.txt
```

Cell smoke test:

```python
!PYTHONPATH=/kaggle/working/ocr_benchmark/src python -m ocr_benchmark.benchmark \
  --config configs/kaggle_doclaynet_science.yaml \
  --engines noop \
  --limit 3 \
  --output-dir /kaggle/working/ocr_benchmark_outputs/noop
```

Cell chay mot engine:

```python
!python -m pip install -q -r requirements-kaggle-docling.txt
!PYTHONPATH=/kaggle/working/ocr_benchmark/src python -m ocr_benchmark.benchmark \
  --config configs/kaggle_doclaynet_science.yaml \
  --engines docling \
  --limit 20 \
  --output-dir /kaggle/working/ocr_benchmark_outputs/docling
```

Cell zip output:

```python
!cd /kaggle/working && zip -r ocr_benchmark_outputs.zip ocr_benchmark_outputs
```
