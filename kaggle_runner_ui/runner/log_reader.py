import json
import zipfile
from pathlib import Path

import pandas as pd


def find_job_files(output_dir):
    output_dir = Path(output_dir)
    files = {
        "logs": sorted(output_dir.rglob("job_debug_*.log")),
        "summaries": sorted(output_dir.rglob("summary.csv")),
        "errors": sorted(output_dir.rglob("errors.csv")),
        "prefetch": sorted(output_dir.rglob("prefetch_status.jsonl")),
        "zips": sorted(output_dir.rglob("results_*.zip")),
    }
    for zip_path in files["zips"]:
        extract_dir = zip_path.with_suffix("")
        if not extract_dir.exists():
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_dir)
            except zipfile.BadZipFile:
                pass
    files["logs"] = sorted(output_dir.rglob("job_debug_*.log"))
    files["summaries"] = sorted(output_dir.rglob("summary.csv"))
    files["errors"] = sorted(output_dir.rglob("errors.csv"))
    files["prefetch"] = sorted(output_dir.rglob("prefetch_status.jsonl"))
    return files


def read_text_tail(path, max_chars=20000):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:]


def read_csv(path):
    return pd.read_csv(path)


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def combine_summaries(project_output_dir):
    rows = []
    for summary_path in Path(project_output_dir).rglob("summary.csv"):
        df = pd.read_csv(summary_path)
        parts = summary_path.parts
        job_name = parts[-3] if len(parts) >= 3 else summary_path.parent.name
        df.insert(0, "job", job_name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

