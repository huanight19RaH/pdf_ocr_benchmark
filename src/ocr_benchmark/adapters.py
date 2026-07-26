import json
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
        self.pipeline = PaddleOCRVL(pipeline_version=version)

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
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        except TypeError:
            self.ocr = PaddleOCR(lang=lang)

    def predict(self, image_path: Path) -> EngineResult:
        if hasattr(self.ocr, "ocr"):
            result = self.ocr.ocr(str(image_path), cls=True)
        else:
            result = self.ocr.predict(str(image_path))
        return EngineResult(text="\n".join(_collect_strings(result)), raw={"items": _safe_repr(result)[:2000]})


class SuryaEngine(BaseEngine):
    name = "surya"

    def __init__(self, config=None):
        super().__init__(config)
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
            langs = self.config.get("language", "en")
            commands = [
                [self.command, str(image_path), "--output_dir", str(out_dir), "--langs", langs],
                [self.command, "ocr", str(image_path), "--output_dir", str(out_dir), "--langs", langs],
            ]
            last_error = None
            for cmd in commands:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if proc.returncode == 0:
                    text = []
                    for path in out_dir.rglob("*.json"):
                        text.extend(_collect_strings(json.loads(path.read_text(encoding="utf-8", errors="ignore"))))
                    for path in out_dir.rglob("*.txt"):
                        text.append(path.read_text(encoding="utf-8", errors="ignore"))
                    return EngineResult(text="\n".join(text), raw={"stdout": proc.stdout[-2000:]})
                last_error = proc.stderr[-2000:] or proc.stdout[-2000:]
            raise RuntimeError(f"Surya command failed: {last_error}")

    def _try_python_engine(self):
        try:
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.detection.processor import load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor

            return {
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
