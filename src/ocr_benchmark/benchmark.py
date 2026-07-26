import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm.auto import tqdm

from .adapters import NON_RUNNABLE_ENGINES, build_engine
from .dataset import load_doclaynet_subset, save_page_image
from .reporting import build_summary, write_report
from .text_utils import char_f1, exact_match_normalized, normalize_text, safe_cer, safe_wer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark OCR engines on a DocLayNet subset.")
    parser.add_argument("--config", default="configs/kaggle_doclaynet_science.yaml")
    parser.add_argument("--engines", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--keep-going", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    runtime_cfg = config["runtime"]

    output_dir = Path(args.output_dir or runtime_cfg["output_dir"])
    pages_dir = output_dir / "pages"
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit or int(dataset_cfg.get("limit", 20))
    engines = _selected_engines(config, args.engines)

    samples = list(load_doclaynet_subset(dataset_cfg, limit))
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps({k: v for k, v in sample.items() if k != "image"}, ensure_ascii=False) + "\n")

    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"
    rows = []
    error_rows = []

    with predictions_path.open("w", encoding="utf-8") as pred_f, errors_path.open("w", encoding="utf-8") as err_f:
        for engine_name, engine_cfg in engines:
            engine_runtime_cfg = {**runtime_cfg, **engine_cfg}
            print(f"\n=== Loading engine: {engine_name} ===", flush=True)
            if engine_name in NON_RUNNABLE_ENGINES:
                meta = NON_RUNNABLE_ENGINES[engine_name]
                error = {
                    "engine": engine_name,
                    "stage": meta["stage"],
                    "sample_id": None,
                    "latency_s": 0.0,
                    "error": meta["reason"],
                }
                error_rows.append(error)
                err_f.write(json.dumps(error, ensure_ascii=False) + "\n")
                err_f.flush()
                print(f"Skip {engine_name}: {meta['reason']}", flush=True)
                continue
            try:
                engine = build_engine(engine_name, config=engine_runtime_cfg)
            except Exception as exc:
                error = {"engine": engine_name, "stage": "load", "sample_id": None, "latency_s": 0.0, "error": repr(exc)}
                error_rows.append(error)
                err_f.write(json.dumps(error, ensure_ascii=False) + "\n")
                print(f"Skip {engine_name}: {exc}", flush=True)
                continue

            for sample in tqdm(samples, desc=engine_name):
                image_path = save_page_image(sample, pages_dir, runtime_cfg)
                started = time.perf_counter()
                try:
                    result = engine.predict(image_path)
                    latency = time.perf_counter() - started
                    prediction = result.text
                    metrics = compute_metrics(sample["reference_text"], prediction)
                    row = {
                        "engine": engine_name,
                        "sample_id": sample["sample_id"],
                        "doc_name": sample.get("doc_name"),
                        "page_no": sample.get("page_no"),
                        "doc_category": sample.get("doc_category"),
                        "image_path": str(image_path),
                        "latency_s": latency,
                        "reference_chars": len(normalize_text(sample["reference_text"])),
                        "prediction_chars": len(normalize_text(prediction)),
                        "chars_per_second": len(normalize_text(prediction)) / max(1e-9, latency),
                        "reference_preview": normalize_text(sample["reference_text"])[:500],
                        "prediction_preview": normalize_text(prediction)[:500],
                        **metrics,
                    }
                    rows.append(row)
                    pred_f.write(
                        json.dumps(
                            {
                                **row,
                                "reference_text": sample["reference_text"],
                                "prediction_text": prediction,
                                "raw": result.get("raw"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    pred_f.flush()
                except Exception as exc:
                    latency = time.perf_counter() - started
                    error = {
                        "engine": engine_name,
                        "stage": "predict",
                        "sample_id": sample["sample_id"],
                        "latency_s": latency,
                        "error": repr(exc),
                    }
                    error_rows.append(error)
                    err_f.write(json.dumps(error, ensure_ascii=False) + "\n")
                    err_f.flush()

    metrics_df = pd.DataFrame(rows)
    errors_df = pd.DataFrame(error_rows)
    metrics_path = output_dir / "metrics_by_page.csv"
    summary_path = output_dir / "summary.csv"
    requested_engine_names = [name for name, _ in engines]
    metrics_df.to_csv(metrics_path, index=False)
    errors_df.to_csv(output_dir / "errors.csv", index=False)
    summary = build_summary(metrics_df, errors_df, requested_engine_names, expected_pages=len(samples))
    summary.to_csv(summary_path, index=False)
    write_report(output_dir, summary, metrics_df, errors_df, config)
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("No successful predictions. Check errors.jsonl.")
    print(f"\nWrote outputs to: {output_dir}")


def _selected_engines(config, cli_engines):
    configured = {item["name"]: item for item in config.get("engines", [])}
    names = cli_engines or [item["name"] for item in config.get("engines", []) if item.get("enabled")]
    return [(name, configured.get(name, {})) for name in names]


def compute_metrics(reference, prediction):
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    return {
        "cer": safe_cer(ref, pred),
        "wer": safe_wer(ref, pred),
        "char_f1": char_f1(ref, pred),
        "exact_match_normalized": exact_match_normalized(ref, pred),
    }


if __name__ == "__main__":
    main()
