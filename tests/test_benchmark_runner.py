import sys

import pandas as pd
import yaml
from PIL import Image

from ocr_benchmark import dataset
from ocr_benchmark import benchmark


def test_benchmark_runner_writes_required_artifacts(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "fake-doclaynet",
                    "split": "val",
                    "doc_category": "scientific_articles",
                    "limit": 1,
                    "seed": 42,
                },
                "runtime": {
                    "output_dir": str(output_dir),
                    "image_format": "png",
                    "image_max_side": 256,
                    "language": "en",
                },
                "engines": [{"name": "noop", "enabled": True}],
            }
        ),
        encoding="utf-8",
    )

    def fake_load_dataset(name, split):
        assert name == "fake-doclaynet"
        assert split == "val"
        return [
            {
                "image": Image.new("RGB", (32, 32), color="white"),
                "pdf_cells": [[{"text": "Scientific paper title"}]],
                "metadata": {
                    "doc_category": "scientific_articles",
                    "doc_name": "paper.pdf",
                    "page_no": 1,
                    "page_hash": "page-1",
                },
            }
        ]

    monkeypatch.setattr(dataset, "_load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--config",
            str(config_path),
            "--engines",
            "noop",
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )

    benchmark.main()

    for filename in [
        "manifest.jsonl",
        "predictions.jsonl",
        "metrics_by_page.csv",
        "summary.csv",
        "errors.jsonl",
        "errors.csv",
        "benchmark_report.md",
        "sample_failures.csv",
    ]:
        assert (output_dir / filename).exists()

    summary = pd.read_csv(output_dir / "summary.csv")
    assert summary.loc[0, "engine"] == "noop"
    assert summary.loc[0, "success_pages"] == 1
