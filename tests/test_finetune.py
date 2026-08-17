from pathlib import Path
from PIL import Image

from ocr_benchmark.finetune import (
    build_paddleocr_export_command,
    build_safe_export_shell_script,
    extract_text_cells,
    prepare_paddleocr_rec_dataset,
    resolve_checkpoint_prefix,
    write_paddleocr_config,
)


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

    content = (data_dir / "rec_doclaynet.yml").read_text(encoding="utf-8")
    assert "epoch_num: 10" in content
    assert "drop_last: false" in content
    assert "num_workers: 0" in content


def test_extract_text_cells_various_bbox_formats():
    image_size = (1000, 1000)
    pdf_cells = [
        {"text": "Cell 1", "bbox": {"l": 10, "t": 10, "r": 100, "b": 50}},
        {"text": "Cell 2", "bbox": {"x": 200, "y": 200, "w": 80, "h": 30}},
        {"text": "Cell 3", "bbox": [0.1, 0.1, 0.5, 0.2]},
        {"text": "Cell 4", "bbox": [500, 500, 100, 40]},  # [x, y, w, h] where w < x
    ]
    extracted = extract_text_cells(pdf_cells, image_size)
    assert len(extracted) == 4
    assert extracted[0][1] == "Cell 1"
    assert extracted[0][0] == (10, 10, 100, 50)
    assert extracted[1][0] == (200, 200, 280, 230)
    assert extracted[2][0] == (100, 100, 500, 200)


def test_write_paddleocr_config_gpu_and_cpu_flags(tmp_path):
    cfg_gpu = tmp_path / "gpu.yml"
    write_paddleocr_config(cfg_gpu, tmp_path, epochs=5, use_gpu=True)
    assert "use_gpu: true" in cfg_gpu.read_text(encoding="utf-8")

    cfg_cpu = tmp_path / "cpu.yml"
    write_paddleocr_config(cfg_cpu, tmp_path, epochs=5, use_gpu=False)
    assert "use_gpu: false" in cfg_cpu.read_text(encoding="utf-8")


def test_resolve_checkpoint_prefix_fallback(tmp_path):
    # Case 1: Empty folder falls back to best_accuracy
    assert resolve_checkpoint_prefix(tmp_path) == tmp_path / "best_accuracy"

    # Case 2: Specific latest.pdparams
    latest_file = tmp_path / "latest.pdparams"
    latest_file.write_text("dummy", encoding="utf-8")
    assert resolve_checkpoint_prefix(tmp_path) == tmp_path / "latest"

    # Case 3: best_accuracy.pdparams takes top priority
    best_file = tmp_path / "best_accuracy.pdparams"
    best_file.write_text("dummy", encoding="utf-8")
    assert resolve_checkpoint_prefix(tmp_path) == tmp_path / "best_accuracy"

    # Case 4: Any arbitrary .pdparams in subfolder
    sub_dir = tmp_path / "nested"
    sub_dir.mkdir()
    iter_file = sub_dir / "iter_epoch_10.pdparams"
    iter_file.write_text("dummy", encoding="utf-8")
    best_file.unlink()
    latest_file.unlink()
    assert resolve_checkpoint_prefix(tmp_path) == sub_dir / "iter_epoch_10"


def test_build_paddleocr_export_command_strips_pdparams_suffix(tmp_path):
    repo = Path("/kaggle/working/PaddleOCR")
    data_dir = tmp_path / "data"
    ckpt = tmp_path / "train_output" / "best_accuracy.pdparams"
    inference_dir = tmp_path / "inference"

    cmd = build_paddleocr_export_command(repo, data_dir, ckpt, inference_dir)
    assert f"Global.pretrained_model={(tmp_path / 'train_output' / 'best_accuracy').as_posix()}" in cmd
    assert f"Global.save_inference_dir={inference_dir.as_posix()}" in cmd


def test_build_safe_export_shell_script(tmp_path):
    repo = Path("/kaggle/working/PaddleOCR")
    data_dir = tmp_path / "data"
    train_out = tmp_path / "train_output"
    inference_dir = tmp_path / "inference"

    script = build_safe_export_shell_script(repo, data_dir, train_out, inference_dir)
    assert "best_accuracy.pdparams" in script
    assert "latest.pdparams" in script
    assert "export_model.py" in script
