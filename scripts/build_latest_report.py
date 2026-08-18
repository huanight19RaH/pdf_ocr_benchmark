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
    finetune = collect_finetune(input_dir)
    logs = collect_logs(input_dir)

    summary_df = merge_diagnostics(summaries, prefetch, errors)
    summary_df = sort_summary(summary_df)
    comparison_df = build_comparison_table(summary_df)

    write_csv(summary_df, output_dir / "combined_summary.csv")
    write_csv(errors, output_dir / "combined_errors.csv")
    write_csv(prefetch, output_dir / "combined_prefetch_status.csv")
    if not finetune.empty:
        write_csv(finetune, output_dir / "combined_finetune_status.csv")
    write_csv(logs, output_dir / "combined_log_tails.csv")

    xlsx_path = output_dir / "latest_benchmark_report.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        excel_safe(summary_df).to_excel(writer, sheet_name="summary", index=False)
        if not comparison_df.empty:
            excel_safe(comparison_df).to_excel(writer, sheet_name="ft_comparison", index=False)
        excel_safe(errors).to_excel(writer, sheet_name="errors", index=False)
        excel_safe(prefetch).to_excel(writer, sheet_name="prefetch", index=False)
        if not finetune.empty:
            excel_safe(finetune).to_excel(writer, sheet_name="finetune", index=False)
        excel_safe(logs).to_excel(writer, sheet_name="logs", index=False)

    report_path = output_dir / "LATEST_BENCHMARK_REPORT.md"
    report_path.write_text(render_markdown(summary_df, comparison_df, finetune), encoding="utf-8")

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


def collect_finetune(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in input_dir.rglob("finetune_status.csv"):
        try:
            df = pd.read_csv(path)
            df["job"] = infer_job(path)
            df["source_path"] = str(path)
            rows.extend(df.to_dict("records"))
        except Exception:
            continue
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
    engine = str(row.get("engine", "")).lower()
    error = " ".join(
        str(row.get(col, ""))
        for col in ["sample_error", "prefetch_error", "read_error"]
        if col in row and pd.notna(row.get(col))
    )
    if status == "ok":
        if engine == "paddleocr_ft":
            return "OK (Finetuned)"
        if engine == "paddleocr":
            return "OK (Pretrained Baseline)"
        return "OK"
    if "cls" in error and "PaddleOCR" in error:
        return "PaddleOCR API cu dang cu; can push/rerun adapter fix bo cls=True."
    if "ncclCommShrink" in error or "libtorch_cuda" in error:
        return "PaddleOCR-VL loi CUDA/NCCL; dung requirement CPU default moi hoac fresh Kaggle session."
    if "pad_token_id" in error and "surya" in error.lower():
        return "SuryaDecoderConfig thieu pad_token_id; can pin transformers<5.0 va patch adapter."
    if "docker binary not found" in error and "surya" in error.lower():
        return "Surya v2 can Docker/vLLM; can pin surya-ocr==0.17.1 va rerun fresh kernel."
    if "--langs" in error and "surya" in error.lower():
        return "Surya CLI fallback cu bi fail; xem loi truoc do trong combined_errors.csv."
    if "rec_model_dir" in error or "inference model" in error or (engine == "paddleocr_ft" and status in {"load_failed", "failed"}):
        return "PaddleOCR-FT: Finetuned model chua duoc export hoac inference dir khong ton tai."
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


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "engine" not in df.columns:
        return pd.DataFrame()

    paddle_rows = df[df["engine"].isin(["paddleocr", "paddleocr_ft"])]
    if len(paddle_rows) < 2:
        return pd.DataFrame()

    base_rows = df[df["engine"] == "paddleocr"]
    ft_rows = df[df["engine"] == "paddleocr_ft"]
    if base_rows.empty or ft_rows.empty:
        return pd.DataFrame()

    base = base_rows.iloc[0]
    ft = ft_rows.iloc[0]

    def fmt_num(val, dec=4):
        try:
            return f"{float(val):.{dec}f}"
        except Exception:
            return "N/A"

    def fmt_diff(ft_val, base_val, lower_is_better=True, dec=4):
        try:
            fv, bv = float(ft_val), float(base_val)
            diff = fv - bv
            pct = (diff / bv * 100) if bv != 0 else 0
            sign = "+" if diff > 0 else ""
            better = (diff < 0) if lower_is_better else (diff > 0)
            tag = " [Improved]" if better else (" [Degraded]" if diff != 0 else " [Same]")
            return f"{sign}{diff:.{dec}f} ({sign}{pct:.1f}%){tag}"
        except Exception:
            return "N/A"

    comparison_data = [
        {
            "Metric": "Character Error Rate (CER)",
            "Pretrained Baseline (paddleocr)": fmt_num(base.get("cer")),
            "Finetuned 10-Epochs (paddleocr_ft)": fmt_num(ft.get("cer")),
            "Difference / Impact": fmt_diff(ft.get("cer"), base.get("cer"), lower_is_better=True),
        },
        {
            "Metric": "Word Error Rate (WER)",
            "Pretrained Baseline (paddleocr)": fmt_num(base.get("wer")),
            "Finetuned 10-Epochs (paddleocr_ft)": fmt_num(ft.get("wer")),
            "Difference / Impact": fmt_diff(ft.get("wer"), base.get("wer"), lower_is_better=True),
        },
        {
            "Metric": "Character F1 Score",
            "Pretrained Baseline (paddleocr)": fmt_num(base.get("char_f1")),
            "Finetuned 10-Epochs (paddleocr_ft)": fmt_num(ft.get("char_f1")),
            "Difference / Impact": fmt_diff(ft.get("char_f1"), base.get("char_f1"), lower_is_better=False),
        },
        {
            "Metric": "Latency per Page (s)",
            "Pretrained Baseline (paddleocr)": fmt_num(base.get("latency_s"), 2),
            "Finetuned 10-Epochs (paddleocr_ft)": fmt_num(ft.get("latency_s"), 2),
            "Difference / Impact": fmt_diff(ft.get("latency_s"), base.get("latency_s"), lower_is_better=True, dec=2),
        },
        {
            "Metric": "Chars / Second",
            "Pretrained Baseline (paddleocr)": fmt_num(base.get("chars_per_second"), 1),
            "Finetuned 10-Epochs (paddleocr_ft)": fmt_num(ft.get("chars_per_second"), 1),
            "Difference / Impact": fmt_diff(ft.get("chars_per_second"), base.get("chars_per_second"), lower_is_better=False, dec=1),
        },
    ]
    return pd.DataFrame(comparison_data)


def render_markdown(df: pd.DataFrame, comparison_df: pd.DataFrame = None, finetune_df: pd.DataFrame = None) -> str:
    if df.empty:
        return "# Latest OCR Benchmark Report\n\nNo summary.csv files found.\n"

    # Flexible argument support
    if comparison_df is not None and "model" in comparison_df.columns and finetune_df is None:
        finetune_df = comparison_df
        comparison_df = build_comparison_table(df)

    if comparison_df is None:
        comparison_df = build_comparison_table(df)

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

    sections = [
        "# Latest OCR Benchmark Report",
        "",
        "## Overall Model Benchmark Summary",
        "",
        table,
    ]

    if comparison_df is not None and not comparison_df.empty:
        sections.extend([
            "",
            "## Pretrained vs. Finetuned Head-to-Head Comparison (`paddleocr` vs `paddleocr_ft`)",
            "",
            comparison_df.to_markdown(index=False),
            "",
            "> **Note:** Finetuning is performed for 10 epochs on DocLayNet `scientific_articles` subset crops.",
        ])

    if finetune_df is not None and not finetune_df.empty:
        sections.extend([
            "",
            "## Finetune Status",
            "",
            finetune_df.to_markdown(index=False),
        ])

    sections.extend([
        "",
        "## Output Files",
        "",
        "- `combined_summary.csv`: Aggregated benchmark metrics for all engines.",
        "- `latest_benchmark_report.xlsx`: Multi-sheet workbook (summary, ft_comparison, errors, prefetch, finetune, logs).",
        "- `combined_errors.csv`: Detailed failure records and stack traces.",
        "- `combined_prefetch_status.csv`: Diagnostic results from model prefetch step.",
        "- `combined_finetune_status.csv`: Finetuning execution logs and crop stats.",
        "- `combined_log_tails.csv`: Execution log tails from remote Kaggle workers.",
    ])

    return "\n".join(sections) + "\n"


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
