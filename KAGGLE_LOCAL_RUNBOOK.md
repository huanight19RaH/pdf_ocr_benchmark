# Local -> Kaggle Runbook

## 1. Prepare GitHub

Push the latest repo code:

```bat
git add .
git commit -m "feat: add Kaggle multi-account runner"
git push
```

## 2. Install Local Kaggle CLI

Double click or run:

```bat
scripts\windows\00_install_local_deps.bat
```

## 3. Add Kaggle Tokens

For each Kaggle account:

1. Open Kaggle.
2. Go to `Settings -> API`.
3. Click `Create New Token`.
4. Put the downloaded `kaggle.json` here:

```text
.kaggle_tokens/account1/kaggle.json
.kaggle_tokens/account2/kaggle.json
.kaggle_tokens/account3/kaggle.json
```

Do not commit real tokens.

## 4. Fill Config

Open:

```text
configs/kaggle_accounts.yaml
```

Fill:

```yaml
username: your_real_kaggle_username
token_dir: .kaggle_tokens/account1
```

Keep `repo_url` as your GitHub repo:

```yaml
repo_url: https://github.com/huanight19RaH/pdf_ocr_benchmark.git
```

## 5. Submit Jobs

Prepare Kaggle kernel folders:

```bat
scripts\windows\01_prepare_jobs.bat
```

Submit all jobs:

```bat
scripts\windows\02_push_jobs.bat
```

Check status:

```bat
scripts\windows\03_check_status.bat
```

Download outputs:

```bat
scripts\windows\04_download_outputs.bat
```

Or do push, wait, and download in one command:

```bat
scripts\windows\05_run_all_and_wait.bat
```

## 6. Read Results

Downloaded outputs are here:

```text
kaggle_remote_jobs/outputs/
```

Important files per job:

```text
summary.csv
errors.csv
benchmark_report.md
prefetch_status.jsonl
results_<job>.zip
```

If a model fails, inspect:

```text
prefetch_status.jsonl
errors.csv
```

