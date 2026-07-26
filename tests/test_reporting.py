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

