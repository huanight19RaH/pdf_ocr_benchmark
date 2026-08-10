# Portable Kaggle Runner UI

Local Streamlit dashboard for submitting and monitoring Kaggle jobs across multiple accounts.

Run:

```bash
cd kaggle_runner_ui
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows you can also double click:

```text
run_windows.bat
```

If this folder is still inside the OCR benchmark repo, import your existing accounts/project once:

```bash
python import_current_ocr_project.py
```

## First Setup

1. Open the `Accounts` tab.
2. Add each Kaggle account.
3. Import `kaggle.json`, paste `username + key`, or paste `access_token`.
4. Open the `Projects` tab and edit the default project.
5. Open `Run` and click `Prepare`, `Push`, `Status`, then `Download`.

Sensitive files are local-only and ignored:

```text
data/accounts.yaml
data/projects.yaml
data/tokens/**
outputs/**
work/**
```

## Default Project

The default config is for:

```text
https://github.com/huanight19RaH/pdf_ocr_benchmark.git
```

Jobs:

- `docling`
- `paddleocr`
- `surya`
- `paddleocr-vl`

You can copy this folder to another project and update `data/projects.yaml` in the UI.
