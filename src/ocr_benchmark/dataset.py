import random
from pathlib import Path

from PIL import Image

from .text_utils import flatten_pdf_cells, normalize_text


def load_doclaynet_subset(dataset_cfg, limit):
    ds = _load_dataset(dataset_cfg["name"], split=dataset_cfg.get("split", "val"))
    target_category = dataset_cfg.get("doc_category")
    seed = int(dataset_cfg.get("seed", 42))
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    yielded = 0
    for idx in indices:
        row = ds[idx]
        metadata = row.get("metadata") or {}
        doc_category = metadata.get("doc_category") or row.get("doc_category")
        if target_category and doc_category != target_category:
            continue
        reference = flatten_pdf_cells(row.get("pdf_cells"))
        if not normalize_text(reference):
            continue
        sample_id = metadata.get("page_hash") or metadata.get("file_name") or f"sample_{idx:06d}"
        yield {
            "sample_id": str(sample_id).replace("/", "_").replace("\\", "_"),
            "image": row["image"],
            "reference_text": reference,
            "pdf_cells": row.get("pdf_cells") or [],
            "doc_category": doc_category,
            "doc_name": metadata.get("doc_name"),
            "page_no": metadata.get("page_no"),
            "metadata": metadata,
        }
        yielded += 1
        if yielded >= limit:
            break


def save_page_image(sample, pages_dir: Path, runtime_cfg):
    image = sample["image"]
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    max_side = runtime_cfg.get("image_max_side")
    if max_side and max(image.size) > int(max_side):
        image.thumbnail((int(max_side), int(max_side)))
    image_path = pages_dir / f"{sample['sample_id']}.{runtime_cfg.get('image_format', 'png')}"
    if not image_path.exists():
        image.save(image_path)
    return image_path


def _load_dataset(name, split):
    from datasets import load_dataset

    return load_dataset(name, split=split)
