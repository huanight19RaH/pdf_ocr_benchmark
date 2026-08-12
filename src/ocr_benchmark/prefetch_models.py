import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from .adapters import build_engine

LOAD_ONLY_ENGINES = {"paddleocr_vl"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prefetch and smoke-test OCR model downloads.")
    parser.add_argument("--engines", nargs="+", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/model_prefetch")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "smoke.png"
    make_smoke_image(image_path)

    rows = []
    for engine in args.engines:
        print(f"\n=== Prefetch/check: {engine} ===", flush=True)
        try:
            if engine == "docling":
                run_optional(["docling-tools", "models", "download"])
            predictor = build_engine(engine, config={"language": "en"})
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
