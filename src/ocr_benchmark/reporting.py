from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "engine",
    "status",
    "success_pages",
    "failed_pages",
    "failure_rate",
    "cer",
    "wer",
    "char_f1",
    "exact_match_normalized",
    "latency_s",
    "chars_per_second",
]


def build_summary(metrics_df: pd.DataFrame, errors_df: pd.DataFrame, requested_engines, expected_pages=None) -> pd.DataFrame:
    rows = []
    for engine in requested_engines:
        engine_metrics = _filter_engine(metrics_df, engine)
        engine_errors = _filter_engine(errors_df, engine)
        success_pages = len(engine_metrics)
        status = _status_for(engine_metrics, engine_errors)
        failed_pages = _failed_page_count(engine_errors, status, expected_pages)
        total_pages = success_pages + failed_pages
        if success_pages:
            rows.append(
                {
                    "engine": engine,
                    "status": status,
                    "success_pages": success_pages,
                    "failed_pages": failed_pages,
                    "failure_rate": failed_pages / max(1, total_pages),
                    "cer": engine_metrics["cer"].mean(),
                    "wer": engine_metrics["wer"].mean(),
                    "char_f1": engine_metrics["char_f1"].mean(),
                    "exact_match_normalized": engine_metrics["exact_match_normalized"].mean(),
                    "latency_s": engine_metrics["latency_s"].mean(),
                    "chars_per_second": engine_metrics["chars_per_second"].mean(),
                }
            )
        else:
            rows.append(
                {
                    "engine": engine,
                    "status": status,
                    "success_pages": 0,
                    "failed_pages": failed_pages,
                    "failure_rate": 1.0 if failed_pages else 0.0,
                    "cer": None,
                    "wer": None,
                    "char_f1": None,
                    "exact_match_normalized": None,
                    "latency_s": None,
                    "chars_per_second": None,
                }
            )
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return summary.sort_values(["cer", "wer"], na_position="last").reset_index(drop=True)


def write_report(output_dir: Path, summary: pd.DataFrame, metrics_df: pd.DataFrame, errors_df: pd.DataFrame, config) -> None:
    output_dir = Path(output_dir)
    report_path = output_dir / "benchmark_report.md"
    failures_path = output_dir / "sample_failures.csv"

    if not metrics_df.empty:
        failures = metrics_df.sort_values(["cer", "wer"], ascending=False).head(25)
        failures.to_csv(failures_path, index=False)
    else:
        pd.DataFrame().to_csv(failures_path, index=False)

    lines = [
        "# DocLayNet OCR Benchmark Report",
        "",
        "## Config",
        "",
        f"- Dataset: `{config['dataset'].get('name')}`",
        f"- Split: `{config['dataset'].get('split')}`",
        f"- Document category: `{config['dataset'].get('doc_category')}`",
        f"- Seed: `{config['dataset'].get('seed')}`",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No benchmark rows were produced.")
    else:
        lines.append(summary.to_markdown(index=False))
    lines.extend(["", "## Notes", ""])
    lines.append("- CER/WER/char-F1 are computed after text normalization.")
    lines.append("- `pdf_cells` is used as pseudo-ground-truth text from the digital PDF.")
    lines.append("- Reference-only/non-OCR entries are listed as skipped and are not ranked.")
    if not errors_df.empty:
        lines.extend(["", "## Errors", ""])
        lines.append(errors_df.head(20).to_markdown(index=False))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _filter_engine(df: pd.DataFrame, engine: str) -> pd.DataFrame:
    if df.empty or "engine" not in df.columns:
        return pd.DataFrame()
    return df[df["engine"] == engine]


def _status_for(engine_metrics: pd.DataFrame, engine_errors: pd.DataFrame) -> str:
    if len(engine_metrics) and len(engine_errors):
        return "partial"
    if len(engine_metrics):
        return "ok"
    if not len(engine_errors):
        return "not_run"
    stages = set(engine_errors.get("stage", []))
    if "reference_only" in stages or "non_ocr_skipped" in stages:
        return "skipped"
    if "load" in stages:
        return "load_failed"
    return "failed"


def _failed_page_count(engine_errors: pd.DataFrame, status: str, expected_pages) -> int:
    if status == "skipped":
        return 0
    if status == "load_failed" and expected_pages is not None:
        return int(expected_pages)
    if engine_errors.empty or "sample_id" not in engine_errors.columns:
        return len(engine_errors)
    return int(engine_errors["sample_id"].notna().sum() or len(engine_errors))
