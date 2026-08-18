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
            rec_model_path = Path(rec_model_dir)
            if not rec_model_path.exists():
                raise FileNotFoundError(f"rec_model_dir does not exist: {rec_model_dir}")
            kwargs["rec_model_dir"] = str(rec_model_path)
            rec_char_dict_path = self.config.get("rec_char_dict_path")
            if not rec_char_dict_path:
                candidates = [
                    rec_model_path / "dict.txt",
                    rec_model_path.parent / "paddleocr_rec_dataset" / "dict.txt",
                    rec_model_path.parent / "dict.txt",
                ]
                for cand in candidates:
                    if cand.exists():
                        rec_char_dict_path = str(cand)
                        break
            if rec_char_dict_path and Path(rec_char_dict_path).exists():
                kwargs["rec_char_dict_path"] = str(rec_char_dict_path)

        base_kwargs = {}
        if "rec_model_dir" in kwargs:
            base_kwargs["rec_model_dir"] = kwargs["rec_model_dir"]
        if "rec_char_dict_path" in kwargs:
            base_kwargs["rec_char_dict_path"] = kwargs["rec_char_dict_path"]

        # Detect GPU availability for legacy PaddleOCR 2.x support
        gpu_kwargs = {}
        try:
            import paddle
            cuda_avail = False
            if hasattr(paddle, "is_compiled_with_cuda") and paddle.is_compiled_with_cuda():
                if hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "device_count"):
                    cuda_avail = paddle.device.cuda.device_count() > 0
                else:
                    cuda_avail = True
            if not cuda_avail:
                gpu_kwargs["use_gpu"] = False
        except Exception:
            gpu_kwargs["use_gpu"] = False

        init_attempts = [
            {"use_angle_cls": True, "lang": lang, **base_kwargs, **gpu_kwargs},
            {"use_angle_cls": True, "lang": lang, **base_kwargs},
            {"lang": lang, **base_kwargs},
            base_kwargs,
            {"lang": lang},
            {},
        ]
        last_error = None
        for init_kwargs in init_attempts:
            try:
                self.ocr = PaddleOCR(**init_kwargs)
                return
            except (TypeError, ValueError, RuntimeError, Exception) as exc:
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


def patch_surya_transformers_compatibility():
    """Defensive monkey-patch for Surya OCR + Transformers compatibility.

    Handles:
    - Missing config attributes: `pad_token_id`, `decoder_pad_token_id`, `bbox_size`, token IDs across
      `PretrainedConfig`, static config classes, and dynamically loaded classes (e.g. SuryaDecoderConfig).
    - Missing `find_pruneable_heads_and_indices` in transformers.pytorch_utils or modeling_utils.
    - Dynamic module loader patching to ensure dynamically loaded config classes from Hugging Face hub
      have the necessary attributes defined.
    """
    import sys

    # 1. Patch find_pruneable_heads_and_indices if removed/missing in transformers
    try:
        def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            try:
                import torch
                if len(heads) == 0:
                    return heads, None
                heads = set(heads) - set(already_pruned_heads)
                if len(heads) == 0:
                    return heads, None
                mask = torch.ones(n_heads, head_size)
                for head in sorted(heads):
                    head_idx = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                    if head_idx < n_heads:
                        mask[head_idx] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index
            except Exception:
                return heads, None

        for mod_name in ("transformers.pytorch_utils", "transformers.modeling_utils", "transformers"):
            try:
                mod = sys.modules.get(mod_name)
                if mod is None:
                    import importlib
                    mod = importlib.import_module(mod_name)
                if not hasattr(mod, "find_pruneable_heads_and_indices"):
                    setattr(mod, "find_pruneable_heads_and_indices", _find_pruneable_heads_and_indices)
            except Exception:
                pass
    except Exception:
        pass

    # 2. Patch PretrainedConfig class and __init__
    try:
        from transformers.configuration_utils import PretrainedConfig

        # Class level fallbacks
        for attr in ("pad_token_id", "decoder_pad_token_id", "bbox_size", "bos_token_id", "eos_token_id", "sep_token_id"):
            if not hasattr(PretrainedConfig, attr):
                setattr(PretrainedConfig, attr, None)

        if not getattr(PretrainedConfig, "_surya_pad_token_patched", False):
            orig_init = PretrainedConfig.__init__

            def _patched_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                if "pad_token_id" in kwargs:
                    self.pad_token_id = kwargs["pad_token_id"]
                elif getattr(self, "pad_token_id", None) is None:
                    self.pad_token_id = getattr(self, "decoder_pad_token_id", None) or getattr(self, "pad_token_id", None)
                if not hasattr(self, "decoder_pad_token_id"):
                    self.decoder_pad_token_id = getattr(self, "pad_token_id", None)
                if not hasattr(self, "bbox_size"):
                    self.bbox_size = None

            PretrainedConfig.__init__ = _patched_init
            PretrainedConfig._surya_pad_token_patched = True
    except Exception:
        pass

    # 3. Patch dynamic_module_utils so any Hugging Face Hub dynamically loaded class receives attributes
    try:
        import transformers.dynamic_module_utils as dmu

        if hasattr(dmu, "get_class_from_dynamic_module") and not getattr(dmu, "_surya_patched", False):
            orig_get_class = dmu.get_class_from_dynamic_module

            def _patched_get_class(*args, **kwargs):
                cls = orig_get_class(*args, **kwargs)
                if isinstance(cls, type):
                    for attr_name in ("pad_token_id", "decoder_pad_token_id", "bbox_size", "bos_token_id", "eos_token_id", "sep_token_id"):
                        if not hasattr(cls, attr_name):
                            setattr(cls, attr_name, None)
                return cls

            dmu.get_class_from_dynamic_module = _patched_get_class
            dmu._surya_patched = True
    except Exception:
        pass

    # 4. Scan all currently loaded modules for config classes
    try:
        target_attrs = ("pad_token_id", "decoder_pad_token_id", "bbox_size", "bos_token_id", "eos_token_id", "sep_token_id")
        target_names = ("SuryaDecoderConfig", "SuryaConfig", "DecoderConfig", "EfficientViTConfig", "DonutSwinConfig", "MBartConfig")

        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            if any(key in mod_name for key in ("surya", "transformers", "dynamic_module")):
                for name, obj in list(vars(mod).items()):
                    if isinstance(obj, type) and (name in target_names or name.endswith("Config")):
                        for attr in target_attrs:
                            if not hasattr(obj, attr):
                                setattr(obj, attr, None)
    except Exception:
        pass


class SuryaEngine(BaseEngine):
    name = "surya"

    def __init__(self, config=None):
        super().__init__(config)
        os.environ.setdefault("RECOGNITION_BATCH_SIZE", str(self.config.get("recognition_batch_size", 128)))
        os.environ.setdefault("DETECTOR_BATCH_SIZE", str(self.config.get("detector_batch_size", 24)))
        patch_surya_transformers_compatibility()
        self.python_engine = self._try_python_engine()
        if self.python_engine is not None:
            return
        self.command = shutil.which("surya_ocr") or shutil.which("surya")
        if not self.command:
            raise RuntimeError("Surya CLI not found and Python engine failed. Install with: pip install surya-ocr")

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
        patch_surya_transformers_compatibility()
        errors = []
        try:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            patch_surya_transformers_compatibility()
            foundation = FoundationPredictor()
            return {
                "api": "surya_v1",
                "recognition_predictor": RecognitionPredictor(foundation),
                "detection_predictor": DetectionPredictor(),
            }
        except Exception as exc:
            errors.append(f"surya_v1 init failed: {exc}")

        try:
            patch_surya_transformers_compatibility()
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
        except Exception as exc:
            errors.append(f"surya_legacy init failed: {exc}")

        if errors:
            print("[SuryaEngine] Python engine initialization attempts encountered:\n" + "\n".join(f"  - {e}" for e in errors), flush=True)
        return None

    def _predict_python(self, image_path: Path) -> EngineResult:
        from PIL import Image

        patch_surya_transformers_compatibility()
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
    for attr in ("text", "content", "markdown", "text_lines", "lines", "words", "paragraphs", "blocks"):
        if hasattr(value, attr):
            strings.extend(_collect_strings(getattr(value, attr)))
    return strings
