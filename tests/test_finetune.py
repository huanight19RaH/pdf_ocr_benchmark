from PIL import Image

from ocr_benchmark.finetune import prepare_paddleocr_rec_dataset


def test_prepare_paddleocr_rec_dataset_writes_labels_and_config(tmp_path):
    samples = [
        {
            "sample_id": "page-1",
            "image": Image.new("RGB", (100, 50), color="white"),
            "reference_text": "Title Abstract",
            "pdf_cells": [
                {"text": "Title", "bbox": [0, 0, 50, 20]},
                {"text": "Abstract", "bbox": [0.0, 0.5, 1.0, 1.0]},
            ],
        }
    ]

    result = prepare_paddleocr_rec_dataset(samples, tmp_path, {"image_max_side": 100}, epochs=10)
    data_dir = result["data_dir"]

    assert (data_dir / "train.txt").exists()
    assert (data_dir / "val.txt").exists()
    assert (data_dir / "dict.txt").exists()
    assert (data_dir / "rec_doclaynet.yml").exists()
    assert result["train_samples"] >= 1
    assert "epoch_num: 10" in (data_dir / "rec_doclaynet.yml").read_text(encoding="utf-8")
