import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm.auto import tqdm

from .dataset import load_doclaynet_subset
from .text_utils import normalize_text


SUPPORTED_FINETUNE = {
    "paddleocr": "supported",
    "docling": "unsupported_public_training",
    "paddleocr_vl": "requires_separate_paddleocr_vl_sft_stack",
    "surya": "unsupported_public_training",
    "omnidocbench": "not_model",
    "protonx_legal_tc": "not_ocr_model",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and optionally run DocLayNet OCR finetuning jobs.")
    parser.add_argument("--config", default="configs/kaggle_doclaynet_science.yaml")
    parser.add_argument("--models", nargs="+", default=["paddleocr", "docling", "paddleocr_vl", "surya"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", default="/kaggle/working/ocr_finetune_outputs")
    parser.add_argument("--execute", action="store_true", help="Run supported training commands instead of preparing them only.")
    parser.add_argument("--paddleocr-repo", default="/kaggle/working/PaddleOCR")
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = dict(config["dataset"])
    runtime_cfg = config["runtime"]
    limit = args.limit or int(dataset_cfg.get("finetune_limit", dataset_cfg.get("limit", 20)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list(load_doclaynet_subset(dataset_cfg, limit))
    status_rows = []
    for model in args.models:
        status = SUPPORTED_FINETUNE.get(model, "unknown_model")
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        if model == "paddleocr":
            try:
                prepared = prepare_paddleocr_rec_dataset(samples, model_dir, runtime_cfg, epochs=args.epochs)
                train_output_dir = model_dir / "train_output"
                inference_dir = model_dir / "inference"
                command = build_paddleocr_train_command(
                    paddleocr_repo=Path(args.paddleocr_repo),
                    data_dir=prepared["data_dir"],
                    epochs=args.epochs,
                    output_dir=train_output_dir,
                )
                export_command = build_paddleocr_export_command(
                    paddleocr_repo=Path(args.paddleocr_repo),
                    data_dir=prepared["data_dir"],
                    checkpoint_prefix=train_output_dir / "best_accuracy",
                    inference_dir=inference_dir,
                )
                safe_export_script = build_safe_export_shell_script(
                    paddleocr_repo=Path(args.paddleocr_repo),
                    data_dir=prepared["data_dir"],
                    train_output_dir=train_output_dir,
                    inference_dir=inference_dir,
                )
                (model_dir / "train_command.sh").write_text(" ".join(command) + "\n", encoding="utf-8")
                (model_dir / "export_command.sh").write_text(safe_export_script, encoding="utf-8")
                row = {
                    "model": model,
                    "status": "prepared",
                    "epochs": args.epochs,
                    "train_samples": prepared["train_samples"],
                    "val_samples": prepared["val_samples"],
                    "output_dir": str(model_dir),
                    "note": "PaddleOCR recognition finetune dataset, train command, and safe export script prepared.",
                }
                if args.execute:
                    ensure_paddleocr_repo(Path(args.paddleocr_repo))
                    subprocess.run(command, check=True)
                    ckpt_prefix = resolve_checkpoint_prefix(train_output_dir)
                    actual_export_command = build_paddleocr_export_command(
                        paddleocr_repo=Path(args.paddleocr_repo),
                        data_dir=prepared["data_dir"],
                        checkpoint_prefix=ckpt_prefix,
                        inference_dir=inference_dir,
                    )
                    subprocess.run(actual_export_command, check=True)
                    row["status"] = "trained"
                    row["note"] = f"Training ({args.epochs} epochs) and export completed from checkpoint '{ckpt_prefix.name}'."
                status_rows.append(row)
            except Exception as exc:
                status_rows.append(
                    {
                        "model": model,
                        "status": "failed",
                        "epochs": args.epochs,
                        "train_samples": 0,
                        "val_samples": 0,
                        "output_dir": str(model_dir),
                        "note": repr(exc),
                    }
                )
            continue

        status_rows.append(
            {
                "model": model,
                "status": "skipped",
                "epochs": args.epochs,
                "train_samples": 0,
                "val_samples": 0,
                "output_dir": str(model_dir),
                "note": finetune_note(model, status),
            }
        )

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(output_dir / "finetune_status.csv", index=False)
    write_finetune_report(output_dir, status_df)
    print(status_df.to_string(index=False))
    print(f"\nWrote finetune outputs to: {output_dir}")


def detect_use_gpu() -> bool:
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return True
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


def prepare_paddleocr_rec_dataset(samples, output_dir: Path, runtime_cfg, epochs: int):
    data_dir = output_dir / "paddleocr_rec_dataset"
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    max_side = runtime_cfg.get("image_max_side") if runtime_cfg else None

    for sample in tqdm(samples, desc="prepare-paddleocr-rec"):
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        if max_side and max(image.size) > int(max_side):
            image.thumbnail((int(max_side), int(max_side)))

        cell_records = extract_text_cells(sample.get("pdf_cells") or [], image.size)
        if not cell_records:
            # Full-page fallback keeps the finetune job usable when cell boxes are unavailable.
            ref_text = normalize_text(sample.get("reference_text", ""))[:512]
            if ref_text:
                cell_records = [((0, 0, image.width, image.height), ref_text)]

        for idx, (bbox, text) in enumerate(cell_records):
            clean_text = text.strip()
            if not clean_text:
                continue
            crop = image.crop(bbox)
            if crop.width < 4 or crop.height < 4:
                continue
            rel_path = Path("images") / f"{sample['sample_id']}_{idx:04d}.png"
            crop.save(data_dir / rel_path)
            records.append((rel_path.as_posix(), clean_text))

    if not records:
        raise RuntimeError("No PaddleOCR recognition crops could be created from DocLayNet pdf_cells.")

    split = max(1, int(len(records) * 0.9))
    train_records = records[:split]
    val_records = records[split:] or records[-1:]
    write_label_file(data_dir / "train.txt", train_records)
    write_label_file(data_dir / "val.txt", val_records)
    write_dict_file(data_dir / "dict.txt", records)
    use_gpu = runtime_cfg.get("use_gpu") if runtime_cfg else None
    write_paddleocr_config(data_dir / "rec_doclaynet.yml", data_dir, epochs=epochs, use_gpu=use_gpu)
    return {"data_dir": data_dir, "train_samples": len(train_records), "val_samples": len(val_records)}


def extract_text_cells(pdf_cells, image_size):
    if isinstance(pdf_cells, str):
        try:
            pdf_cells = json.loads(pdf_cells)
        except Exception:
            pass
    cells = []
    for item in walk_cells(pdf_cells):
        text = item.get("text") or item.get("content") or item.get("value")
        bbox = item.get("bbox") or item.get("box") or item.get("rect")
        if text and bbox:
            parsed = parse_bbox(bbox, image_size)
            if parsed:
                cells.append((parsed, str(text)))
    return cells


def walk_cells(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk_cells(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_cells(item)


def parse_bbox(bbox, image_size):
    width, height = image_size
    if isinstance(bbox, dict):
        if {"l", "t", "r", "b"}.issubset(bbox):
            coords = [bbox["l"], bbox["t"], bbox["r"], bbox["b"]]
        elif {"x1", "y1", "x2", "y2"}.issubset(bbox):
            coords = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        elif {"x", "y", "w", "h"}.issubset(bbox):
            coords = [bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]]
        elif {"x", "y", "x2", "y2"}.issubset(bbox):
            coords = [bbox["x"], bbox["y"], bbox["x2"], bbox["y2"]]
        else:
            return None
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        coords = list(bbox[:4])
        if coords[2] < coords[0] or coords[3] < coords[1]:
            coords = [coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]]
    else:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in coords]
    except Exception:
        return None
    if max(x1, y1, x2, y2) <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def write_label_file(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for rel_path, text in records:
            f.write(f"{rel_path}\t{text}\n")


def write_dict_file(path: Path, records):
    chars = sorted({char for _, text in records for char in text if char.strip()})
    path.write_text("\n".join(chars) + "\n", encoding="utf-8")


def write_paddleocr_config(path: Path, data_dir: Path, epochs: int, use_gpu: bool = None):
    if use_gpu is None:
        use_gpu = detect_use_gpu()
    use_gpu_str = "true" if use_gpu else "false"
    save_model_dir = (data_dir.parent / "train_output").as_posix()
    config = f"""Global:
  use_gpu: {use_gpu_str}
  epoch_num: {epochs}
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {save_model_dir}
  save_epoch_step: 1
  eval_batch_step: [0, 50]
  cal_metric_during_train: true
  pretrained_model:
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img:
  character_dict_path: {data_dir.as_posix()}/dict.txt
  max_text_length: 512
  infer_mode: false
  use_space_char: true

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0005
  regularizer:
    name: L2
    factor: 0.00001

Architecture:
  model_type: rec
  algorithm: CRNN
  Transform:
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: small
  Neck:
    name: SequenceEncoder
    encoder_type: rnn
    hidden_size: 96
  Head:
    name: CTCHead
    fc_decay: 0.00001

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {data_dir.as_posix()}
    label_file_list:
      - {data_dir.as_posix()}/train.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug:
      - CTCLabelEncode:
      - RecResizeImg:
          image_shape: [3, 32, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: true
    batch_size_per_card: 32
    drop_last: false
    num_workers: 0

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {data_dir.as_posix()}
    label_file_list:
      - {data_dir.as_posix()}/val.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - CTCLabelEncode:
      - RecResizeImg:
          image_shape: [3, 32, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 32
    num_workers: 0
"""
    path.write_text(config, encoding="utf-8")


def resolve_checkpoint_prefix(train_output_dir: Path) -> Path:
    train_output_dir = Path(train_output_dir)
    best_ckpt = train_output_dir / "best_accuracy"
    if (train_output_dir / "best_accuracy.pdparams").exists():
        return best_ckpt
    latest_ckpt = train_output_dir / "latest"
    if (train_output_dir / "latest.pdparams").exists():
        return latest_ckpt
    pdparams_files = sorted(train_output_dir.glob("*.pdparams"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pdparams_files:
        stem = pdparams_files[0].stem
        return train_output_dir / stem
    return latest_ckpt


def build_paddleocr_train_command(paddleocr_repo: Path, data_dir: Path, epochs: int, output_dir: Path):
    config_path = data_dir / "rec_doclaynet.yml"
    return [
        "python",
        str(paddleocr_repo / "tools" / "train.py"),
        "-c",
        str(config_path),
        "-o",
        f"Global.epoch_num={epochs}",
        f"Global.save_model_dir={output_dir.as_posix()}",
    ]


def build_paddleocr_export_command(paddleocr_repo: Path, data_dir: Path, checkpoint_prefix: Path, inference_dir: Path):
    config_path = data_dir / "rec_doclaynet.yml"
    return [
        "python",
        str(paddleocr_repo / "tools" / "export_model.py"),
        "-c",
        str(config_path),
        "-o",
        f"Global.pretrained_model={checkpoint_prefix.as_posix()}",
        f"Global.save_inference_dir={inference_dir.as_posix()}",
    ]


def build_safe_export_shell_script(paddleocr_repo: Path, data_dir: Path, train_output_dir: Path, inference_dir: Path) -> str:
    config_path = data_dir / "rec_doclaynet.yml"
    tools_export = paddleocr_repo / "tools" / "export_model.py"
    return f"""#!/usr/bin/env bash
set -e

TRAIN_DIR="{train_output_dir.as_posix()}"
if [ -f "$TRAIN_DIR/best_accuracy.pdparams" ]; then
    CKPT="$TRAIN_DIR/best_accuracy"
elif [ -f "$TRAIN_DIR/latest.pdparams" ]; then
    CKPT="$TRAIN_DIR/latest"
else
    CKPT_FILE=$(ls -t "$TRAIN_DIR"/*.pdparams 2>/dev/null | head -n 1 || true)
    if [ -n "$CKPT_FILE" ]; then
        CKPT="${{CKPT_FILE%.pdparams}}"
    else
        CKPT="$TRAIN_DIR/latest"
    fi
fi

echo "Exporting PaddleOCR checkpoint: $CKPT"
python "{tools_export.as_posix()}" \\
    -c "{config_path.as_posix()}" \\
    -o Global.pretrained_model="$CKPT" \\
       Global.save_inference_dir="{inference_dir.as_posix()}"
"""


def ensure_paddleocr_repo(paddleocr_repo: Path):
    if (paddleocr_repo / "tools" / "train.py").exists():
        return
    if paddleocr_repo.exists():
        shutil.rmtree(paddleocr_repo)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/PaddlePaddle/PaddleOCR.git", str(paddleocr_repo)],
        check=True,
    )


def finetune_note(model, status):
    notes = {
        "unsupported_public_training": "No stable public Kaggle finetune entrypoint is provided for this engine; benchmark pretrained inference instead.",
        "requires_separate_paddleocr_vl_sft_stack": "PaddleOCR-VL finetuning uses a separate SFT stack and is too heavy for the shared single-notebook T4/P100 path.",
        "not_model": "Reference benchmark suite, not a trainable OCR model.",
        "not_ocr_model": "Text classification model, not an OCR model.",
        "unknown_model": "No finetune adapter registered.",
    }
    return notes.get(status, status)


def write_finetune_report(output_dir: Path, status_df: pd.DataFrame):
    lines = [
        "# Finetune Report",
        "",
        "This run prepares 10-epoch finetuning for models with a public Kaggle-feasible training path.",
        "",
        status_df.to_markdown(index=False),
        "",
        "Only `paddleocr` is trained by this repo by default. Other engines remain in the pretrained benchmark table and are marked skipped for finetuning.",
    ]
    (output_dir / "finetune_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
