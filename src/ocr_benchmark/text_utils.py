import re
from collections import Counter


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def flatten_pdf_cells(pdf_cells) -> str:
    chunks = []
    for group in pdf_cells or []:
        if isinstance(group, dict):
            value = group.get("text") or group.get("content") or group.get("value")
            if value:
                chunks.append(str(value))
            continue
        if isinstance(group, (list, tuple)):
            for cell in group:
                if isinstance(cell, dict):
                    value = cell.get("text") or cell.get("content") or cell.get("value")
                    if value:
                        chunks.append(str(value))
                elif isinstance(cell, str):
                    chunks.append(cell)
        elif isinstance(group, str):
            chunks.append(group)
    return "\n".join(chunks)


def levenshtein(a, b) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def char_f1(reference: str, prediction: str) -> float:
    ref_counter = Counter(reference)
    pred_counter = Counter(prediction)
    overlap = sum((ref_counter & pred_counter).values())
    if not reference and not prediction:
        return 1.0
    if not reference or not prediction:
        return 0.0
    precision = overlap / max(1, len(prediction))
    recall = overlap / max(1, len(reference))
    return 2 * precision * recall / max(1e-12, precision + recall)


def safe_cer(reference: str, prediction: str) -> float:
    reference = reference or ""
    prediction = prediction or ""
    if not reference:
        return 0.0 if not prediction else 1.0
    try:
        from jiwer import cer

        return float(cer(reference, prediction))
    except Exception:
        return levenshtein(reference, prediction) / max(1, len(reference))


def safe_wer(reference: str, prediction: str) -> float:
    reference = reference or ""
    prediction = prediction or ""
    if not reference:
        return 0.0 if not prediction else 1.0
    try:
        from jiwer import wer

        return float(wer(reference, prediction))
    except Exception:
        return levenshtein(reference.split(), prediction.split()) / max(1, len(reference.split()))


def exact_match_normalized(reference: str, prediction: str) -> float:
    return float(normalize_text(reference) == normalize_text(prediction))
