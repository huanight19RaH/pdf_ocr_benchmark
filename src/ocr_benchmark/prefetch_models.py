import argparse
import json
import subprocess
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from .adapters import build_engine

LOAD_ONLY_ENGINES = {"paddleocr_vl"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prefetch and smoke-test OCR model downloads.")
    parser.add_argument("--engines", nargs="+", required=True)
    parser.add_argument("--config", default="configs/kaggle_doclaynet_science.yaml")
    parser.add_argument("--output-dir", default="/kaggle/working/model_prefetch")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "smoke.png"
    make_smoke_image(image_path)

    engine_configs = {}
    if Path(args.config).exists():
        try:
            cfg_data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            runtime_cfg = cfg_data.get("runtime", {})
            for item in cfg_data.get("engines", []):
                engine_configs[item["name"]] = {**runtime_cfg, **item}
        except Exception:
            pass

    rows = []
    for engine in args.engines:
        print(f"\n=== Prefetch/check: {engine} ===", flush=True)
        engine_cfg = engine_configs.get(engine, {"language": "en"})
        try:
            if engine == "docling":
                run_optional(["docling-tools", "models", "download"])

            # If paddleocr_ft specified rec_model_dir that does not exist yet, record status
            if engine == "paddleocr_ft":
                rec_dir = engine_cfg.get("rec_model_dir")
                if rec_dir and not Path(rec_dir).exists():
                    rows.append(
                        {
                            "engine": engine,
                            "status": "pending_finetune",
                            "chars": 0,
                            "error": f"Finetuned inference directory {rec_dir} not yet created.",
                        }
                    )
                    print(f"{engine}: pending finetune ({rec_dir} not found)", flush=True)
                    continue

            predictor = build_engine(engine, config=engine_cfg)
            if engine in LOAD_ONLY_ENGINES:
                rows.append({"engine": engine, "status": "loaded", "chars": 0, "error": "load-only prefetch"})
                print(f"{engine}: loaded; skipped smoke inference for heavy engine", flush=True)
                continue
            result = predictor.predict(image_path)
            rows.append({"engine": engine, "status": "ok", "chars": len(result.text), "error": ""})
            print(f"{engine}: ok, chars={len(result.text)}", flush=True)
        except Exception as exc:
            rows.append({"engine": engine, "status": "failed", "chars": 0, "error": repr(exc)})
            print(f"{engine}: failed: {exc}", flush=True)

    status_path = output_dir / "prefetch_status.jsonl"
    with status_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {status_path}")


def make_smoke_image(path: Path):
    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 70), "DocLayNet OCR benchmark smoke test", fill="black")
    draw.text((40, 140), "Scientific paper page 1", fill="black")
    image.save(path)


def run_optional(command):
    try:
        subprocess.run(command, check=False, timeout=900)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
