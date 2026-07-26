from ocr_benchmark.text_utils import (
    char_f1,
    exact_match_normalized,
    flatten_pdf_cells,
    normalize_text,
    safe_cer,
    safe_wer,
)


def test_normalize_text_dehyphenates_and_collapses_space():
    assert normalize_text("Deep-\n learning   OCR") == "deeplearning ocr"


def test_flatten_pdf_cells_supports_nested_cells():
    cells = [
        [{"text": "Title"}, {"content": "Abstract"}],
        {"value": "Conclusion"},
        "References",
    ]
    assert flatten_pdf_cells(cells) == "Title\nAbstract\nConclusion\nReferences"


def test_metric_wrappers_have_expected_bounds():
    assert safe_cer("abc", "abc") == 0.0
    assert safe_wer("hello world", "hello world") == 0.0
    assert char_f1("abc", "abc") == 1.0
    assert exact_match_normalized("Hello   World", "hello world") == 1.0
