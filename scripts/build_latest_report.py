import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build one final benchmark report from downloaded Kaggle outputs.")
    parser.add_argument("--input-dir", default="kaggle_remote_jobs/outputs")
    parser.add_argument("--output-dir", default="outputs/final_benchmark_report_latest")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = collect_summaries(input_dir)
    errors = collect_errors(input_dir)
    prefetch = collect_prefetch(input_dir)
    logs = collect_logs(input_dir)

    summary_df = merge_diagnostics(summaries, prefetch, errors)
    summary_df = sort_summary(summary_df)

    write_csv(summary_df, output_dir / "combined_summary.csv")
    write_csv(errors, output_dir / "combined_errors.csv")
    write_csv(prefetch, output_dir / "combined_prefetch_status.csv")
    write_csv(logs, output_dir / "combined_log_tails.csv")

    xlsx_path = output_dir / "latest_benchmark_report.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        excel_safe(summary_df).to_excel(writer, sheet_name="summary", index=False)
        excel_safe(errors).to_excel(writer, sheet_name="errors", index=False)
        excel_safe(prefetch).to_excel(writer, sheet_name="prefetch", index=False)
        excel_safe(logs).to_excel(writer, sheet_name="logs", index=False)

    report_path = output_dir / "LATEST_BENCHMARK_REPORT.md"
    report_path.write_text(render_markdown(summary_df), encoding="utf-8")

    print(f"Wrote: {output_dir / 'combined_summary.csv'}")
    print(f"Wrote: {xlsx_path}")
    print(f"Wrote: {report_path}")


def collect_summaries(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in input_dir.rglob("summary.csv"):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            rows.append({"job": infer_job(path), "status": "read_failed", "source_path": str(path), "read_error": repr(exc)})
            continue
        df["job"] = infer_job(path)
        df["source_path"] = str(path)
        df["source_mtime"] = path.stat().st_mtime
        rows.extend(df.to_dict("records"))
    return pd.DataFrame(rows)


def collect_errors(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in input_dir.rglob("errors.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {"error": line}
            row["job"] = infer_job(path)
            row["source_path"] = str(path)
            rows.append(row)
    for path in input_dir.rglob("errors.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["job"] = infer_job(path)
        df["source_path"] = str(path)
        rows.extend(df.to_dict("records"))
    return dedupe(pd.DataFrame(rows))


def collect_prefetch(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in input_dir.rglob("prefetch_status.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {"status": "read_failed", "error": line}
            row["job"] = infer_job(path)
            row["source_path"] = str(path)
            rows.append(row)
    return pd.DataFrame(rows)


def collect_logs(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in input_dir.rglob("job_debug*.log"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rows.append(
            {
                "job": infer_job(path),
                "source_path": str(path),
                "log_tail": text[-4000:],
            }
        )
    return pd.DataFrame(rows)


def merge_diagnostics(summary: pd.DataFrame, prefetch: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "job",
                "engine",
                "status",
                "success_pages",
                "failed_pages",
                "failure_rate",
                "cer",
                "wer",
                "char_f1",
                "diagnosis",
            ]
        )

    df = summary.copy()
    if not prefetch.empty:
        prefetch_cols = ["job", "engine", "status", "chars", "error"]
        available = [col for col in prefetch_cols if col in prefetch.columns]
        prefetch_small = prefetch[available].rename(
            columns={"status": "prefetch_status", "chars": "prefetch_chars", "error": "prefetch_error"}
        )
        df = df.merge(prefetch_small, on=["job", "engine"], how="left")

    if not errors.empty:
        error_text = (
            errors.assign(error=lambda x: x.get("error", "").astype(str))
            .groupby(["job", "engine"], dropna=False)["error"]
            .apply(lambda values: " | ".join(dict.fromkeys(v for v in values if v and v != "nan"))[:4000])
            .reset_index()
            .rename(columns={"error": "sample_error"})
        )
        df = df.merge(error_text, on=["job", "engine"], how="left")

    df["diagnosis"] = df.apply(diagnose_row, axis=1)
    return df


def diagnose_row(row) -> str:
    status = str(row.get("status", "")).lower()
    error = " ".join(
        str(row.get(col, ""))
        for col in ["sample_error", "prefetch_error", "read_error"]
        if col in row and pd.notna(row.get(col))
    )
    if status == "ok":
        return "OK"
    if "cls" in error and "PaddleOCR" in error:
        return "PaddleOCR API cu dang cu; can push/rerun adapter fix bo cls=True."
    if "ncclCommShrink" in error or "libtorch_cuda" in error:
        return "PaddleOCR-VL loi CUDA/NCCL; dung requirement CPU default moi hoac fresh Kaggle session."
    if "--langs" in error and "surya" in error.lower():
        return "Surya CLI dang chay code cu co --langs; can push/rerun adapter fix."
    if status in {"failed", "load_failed"}:
        return "Failed; xem combined_errors.csv va combined_log_tails.csv."
    return ""


def sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = ["cer", "wer", "failure_rate"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    sort_cols = [col for col in ["status", "cer", "wer", "engine"] if col in df.columns]
    return df.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def render_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Latest OCR Benchmark Report\n\nNo summary.csv files found.\n"
    display_cols = [
        "job",
        "engine",
        "status",
        "success_pages",
        "failed_pages",
        "failure_rate",
        "cer",
        "wer",
        "char_f1",
        "latency_s",
        "chars_per_second",
        "prefetch_status",
        "diagnosis",
    ]
    cols = [col for col in display_cols if col in df.columns]
    table = df[cols].to_markdown(index=False)
    return "# Latest OCR Benchmark Report\n\n## Summary\n\n" + table + "\n\n## Files\n\n- `combined_summary.csv`\n- `latest_benchmark_report.xlsx`\n- `combined_errors.csv`\n- `combined_prefetch_status.csv`\n- `combined_log_tails.csv`\n"


def infer_job(path: Path) -> str:
    parts = list(path.parts)
    if "outputs" in parts:
        idx = len(parts) - 1 - parts[::-1].index("outputs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    for part in parts:
        if part.startswith("results_ocr-"):
            return part.replace("results_ocr-", "")
        if part.startswith("prefetch_ocr-"):
            return part.replace("prefetch_ocr-", "")
    return path.parent.name


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = df.copy()
    illegal = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].map(lambda value: illegal.sub("", value) if isinstance(value, str) else value)
    return cleaned


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.astype(str).drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    main()
