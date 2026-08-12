import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class EngineResult(dict):
    @property
    def text(self) -> str:
        return self.get("text", "")


class BaseEngine:
    name = "base"
    runnable = True

    def __init__(self, config=None):
        self.config = config or {}

    def predict(self, image_path: Path) -> EngineResult:
        raise NotImplementedError


class NoopEngine(BaseEngine):
    name = "noop"

    def predict(self, image_path: Path) -> EngineResult:
        return EngineResult(text="", raw={"note": "empty prediction smoke-test"})


class ReferenceOnlyEngine(BaseEngine):
    runnable = False
    stage = "reference_only"
    reason = "Reference benchmark suite, not an OCR inference engine."

    def __init__(self, config=None):
        super().__init__(config)
        raise RuntimeError(self.reason)


class NonOCRSkippedEngine(BaseEngine):
    runnable = False
    stage = "non_ocr_skipped"
    reason = "Text classification model, not an OCR inference engine."

    def __init__(self, config=None):
        super().__init__(config)
        raise RuntimeError(self.reason)


class DoclingEngine(BaseEngine):
    name = "docling"

    def __init__(self, config=None):
        super().__init__(config)
        from docling.document_converter import DocumentConverter

        self.converter = DocumentConverter()

    def predict(self, image_path: Path) -> EngineResult:
        result = self.converter.convert(str(image_path))
        document = result.document
        if hasattr(document, "export_to_markdown"):
            text = document.export_to_markdown()
        elif hasattr(document, "export_to_text"):
            text = document.export_to_text()
        else:
            text = str(document)
        return EngineResult(text=text, raw={"format": "docling_document"})


class PaddleOCRVLEngine(BaseEngine):
    name = "paddleocr_vl"

    def __init__(self, config=None):
        super().__init__(config)
        from paddleocr import PaddleOCRVL

        version = self.config.get("pipeline_version", "v1.6")
        try:
            self.pipeline = PaddleOCRVL(pipeline_version=version)
        except TypeError:
            self.pipeline = PaddleOCRVL()

    def predict(self, image_path: Path) -> EngineResult:
        output = self.pipeline.predict(str(image_path))
        text_parts = []
        raw_items = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for item in output:
                raw_items.append(_safe_repr(item))
                if hasattr(item, "save_to_markdown"):
                    item.save_to_markdown(save_path=str(tmp_path))
                text_parts.extend(_collect_strings(item))
            for md in tmp_path.rglob("*.md"):
                text_parts.append(md.read_text(encoding="utf-8", errors="ignore"))
        return EngineResult(text="\n".join(text_parts), raw={"items": raw_items[:3]})


class PaddleOCREngine(BaseEngine):
    name = "paddleocr"

    def __init__(self, config=None):
        super().__init__(config)
        from paddleocr import PaddleOCR

        lang = self.config.get("language", "en")
        kwargs = {}
        rec_model_dir = self.config.get("rec_model_dir")
        if rec_model_dir:
            kwargs["rec_model_dir"] = rec_model_dir
        init_attempts = [
            {"use_angle_cls": True, "lang": lang, **kwargs},
            {"lang": lang, **kwargs},
            kwargs,
        ]
        last_error = None
        for init_kwargs in init_attempts:
            try:
                self.ocr = PaddleOCR(**init_kwargs)
                return
            except TypeError as exc:
                last_error = exc
        raise last_error

    def predict(self, image_path: Path) -> EngineResult:
        errors = []
        result = None
        if hasattr(self.ocr, "ocr"):
            for kwargs in ({"cls": True}, {}):
                try:
                    result = self.ocr.ocr(str(image_path), **kwargs)
                    break
                except TypeError as exc:
                    errors.append(f"ocr({kwargs}): {exc}")
        if result is None and hasattr(self.ocr, "predict"):
            try:
                result = self.ocr.predict(str(image_path))
            except TypeError as exc:
                errors.append(f"predict(): {exc}")
        if result is None:
            detail = "; ".join(errors) if errors else "No compatible PaddleOCR prediction method found."
            raise RuntimeError(f"PaddleOCR prediction failed: {detail}")
        return EngineResult(text="\n".join(_collect_strings(result)), raw={"items": _safe_repr(result)[:2000]})


class SuryaEngine(BaseEngine):
    name = "surya"

    def __init__(self, config=None):
        super().__init__(config)
        os.environ.setdefault("RECOGNITION_BATCH_SIZE", str(self.config.get("recognition_batch_size", 128)))
        os.environ.setdefault("DETECTOR_BATCH_SIZE", str(self.config.get("detector_batch_size", 24)))
        self.python_engine = self._try_python_engine()
        if self.python_engine is not None:
            return
        self.command = shutil.which("surya_ocr") or shutil.which("surya")
        if not self.command:
            raise RuntimeError("Surya CLI not found. Install with: pip install surya-ocr")

    def predict(self, image_path: Path) -> EngineResult:
        if self.python_engine is not None:
            return self._predict_python(image_path)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            input_dir = out_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            input_image = input_dir / image_path.name
            shutil.copy2(image_path, input_image)
            langs = self.config.get("language", "en")
            commands = [
                [self.command, str(input_dir), "--output_dir", str(out_dir), "--images"],
                [self.command, str(image_path), "--output_dir", str(out_dir), "--images"],
                [self.command, str(input_dir), "--output_dir", str(out_dir)],
                [self.command, str(image_path), "--output_dir", str(out_dir)],
                [self.command, str(image_path), "--output_dir", str(out_dir), "--langs", langs],
                [self.command, "ocr", str(image_path), "--output_dir", str(out_dir), "--langs", langs],
            ]
            attempt_errors = []
            for cmd in commands:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if proc.returncode == 0:
                    text = []
                    for path in out_dir.rglob("*.json"):
                        text.extend(_collect_strings(json.loads(path.read_text(encoding="utf-8", errors="ignore"))))
                    for path in out_dir.rglob("*.md"):
                        text.append(path.read_text(encoding="utf-8", errors="ignore"))
                    for path in out_dir.rglob("*.txt"):
                        text.append(path.read_text(encoding="utf-8", errors="ignore"))
                    return EngineResult(
                        text="\n".join(text),
                        raw={"stdout": proc.stdout[-2000:], "command": " ".join(cmd)},
                    )
                message = proc.stderr[-2000:] or proc.stdout[-2000:] or f"exit code {proc.returncode}"
                attempt_errors.append(f"$ {' '.join(cmd)}\n{message}")
            raise RuntimeError("Surya command failed after all CLI variants:\n\n" + "\n\n---\n\n".join(attempt_errors))

    def _try_python_engine(self):
        try:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            foundation = FoundationPredictor()
            return {
                "api": "surya_v1",
                "recognition_predictor": RecognitionPredictor(foundation),
                "detection_predictor": DetectionPredictor(),
            }
        except Exception:
            pass
        try:
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.detection.processor import load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor

            return {
                "api": "surya_legacy_run_ocr",
                "run_ocr": run_ocr,
                "det_model": load_det_model(),
                "det_processor": load_det_processor(),
                "rec_model": load_rec_model(),
                "rec_processor": load_rec_processor(),
            }
        except Exception:
            return None

    def _predict_python(self, image_path: Path) -> EngineResult:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        if self.python_engine.get("api") == "surya_v1":
            result = self.python_engine["recognition_predictor"](
                [image],
                det_predictor=self.python_engine["detection_predictor"],
            )
            return EngineResult(text="\n".join(_collect_strings(result)), raw={"items": _safe_repr(result)[:2000]})
        lang = self.config.get("language", "en")
        langs = [lang] if isinstance(lang, str) else lang
        result = self.python_engine["run_ocr"](
            [image],
            [langs],
            self.python_engine["det_model"],
            self.python_engine["det_processor"],
            self.python_engine["rec_model"],
            self.python_engine["rec_processor"],
        )
        return EngineResult(text="\n".join(_collect_strings(result)), raw={"items": _safe_repr(result)[:2000]})


ENGINE_REGISTRY = {
    "docling": DoclingEngine,
    "paddleocr_vl": PaddleOCRVLEngine,
    "paddleocr": PaddleOCREngine,
    "paddleocr_ft": PaddleOCREngine,
    "surya": SuryaEngine,
    "noop": NoopEngine,
    "omnidocbench": ReferenceOnlyEngine,
    "protonx_legal_tc": NonOCRSkippedEngine,
}

NON_RUNNABLE_ENGINES = {
    "omnidocbench": {
        "stage": "reference_only",
        "reason": "OmniDocBench is a benchmark suite, not an OCR engine for inference.",
    },
    "protonx_legal_tc": {
        "stage": "non_ocr_skipped",
        "reason": "protonx-legal-tc is a text classification model, not an OCR engine.",
    },
}


def build_engine(name: str, config=None):
    if name not in ENGINE_REGISTRY:
        raise KeyError(f"Unknown engine '{name}'. Available: {sorted(ENGINE_REGISTRY)}")
    return ENGINE_REGISTRY[name](config=config or {})


def _safe_repr(value):
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _collect_strings(value):
    strings = []
    if value is None:
        return strings
    if isinstance(value, str):
        clean = value.strip()
        if clean:
            strings.append(clean)
        return strings
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"text", "content", "transcription", "rec_text", "markdown"}:
                strings.extend(_collect_strings(item))
            elif isinstance(item, (dict, list, tuple)):
                strings.extend(_collect_strings(item))
        return strings
    if isinstance(value, (list, tuple)):
        # PaddleOCR classic often returns [box, (text, score)].
        if len(value) == 2 and isinstance(value[1], (list, tuple)) and value[1] and isinstance(value[1][0], str):
            return [value[1][0]]
        for item in value:
            strings.extend(_collect_strings(item))
        return strings
    for attr in ("text", "content", "markdown"):
        if hasattr(value, attr):
            strings.extend(_collect_strings(getattr(value, attr)))
    return strings
