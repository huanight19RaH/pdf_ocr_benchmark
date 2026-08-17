import pandas as pd

from ocr_benchmark.reporting import build_summary


def test_build_summary_counts_success_and_failures():
    metrics = pd.DataFrame(
        [
            {
                "engine": "docling",
                "sample_id": "a",
                "cer": 0.1,
                "wer": 0.2,
                "char_f1": 0.9,
                "exact_match_normalized": 0.0,
                "latency_s": 2.0,
                "chars_per_second": 50.0,
            }
        ]
    )
    errors = pd.DataFrame(
        [
            {
                "engine": "docling",
                "stage": "predict",
                "sample_id": "b",
                "latency_s": 1.0,
                "error": "boom",
            }
        ]
    )

    summary = build_summary(metrics, errors, ["docling"])

    assert summary.loc[0, "status"] == "partial"
    assert summary.loc[0, "success_pages"] == 1
    assert summary.loc[0, "failed_pages"] == 1
    assert summary.loc[0, "failure_rate"] == 0.5


def test_build_summary_marks_non_ocr_reference_as_skipped():
    errors = pd.DataFrame(
        [
            {
                "engine": "omnidocbench",
                "stage": "reference_only",
                "sample_id": None,
                "latency_s": 0.0,
                "error": "not an ocr engine",
            }
        ]
    )

    summary = build_summary(pd.DataFrame(), errors, ["omnidocbench"])

    assert summary.loc[0, "status"] == "skipped"
    assert summary.loc[0, "success_pages"] == 0


def test_build_latest_report_diagnostics_and_finetune_render():
    from scripts.build_latest_report import diagnose_row, render_markdown

    ok_ft_row = {"status": "ok", "engine": "paddleocr_ft"}
    assert diagnose_row(ok_ft_row) == "OK (Finetuned)"

    failed_ft_row = {"status": "failed", "engine": "paddleocr_ft", "sample_error": "inference model not found"}
    assert "PaddleOCR-FT" in diagnose_row(failed_ft_row)

    df_summary = pd.DataFrame(
        [
            {
                "job": "finetune-paddleocr",
                "engine": "paddleocr_ft",
                "status": "ok",
                "success_pages": 20,
                "failed_pages": 0,
                "failure_rate": 0.0,
                "cer": 0.05,
                "wer": 0.10,
                "char_f1": 0.95,
                "latency_s": 1.2,
                "chars_per_second": 120.0,
                "prefetch_status": "ok",
                "diagnosis": "OK (Finetuned)",
            }
        ]
    )
    df_finetune = pd.DataFrame(
        [
            {
                "job": "finetune-paddleocr",
                "model": "paddleocr",
                "status": "trained",
                "epochs": 10,
                "train_samples": 180,
                "val_samples": 20,
                "note": "Training and export completed.",
            }
        ]
    )

    md = render_markdown(df_summary, df_finetune)
    assert "## Finetune Status" in md
    assert "paddleocr_ft" in md
    assert "combined_finetune_status.csv" in md


