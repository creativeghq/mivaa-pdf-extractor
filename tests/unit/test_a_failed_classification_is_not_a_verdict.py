"""A classifier outage must never be written down as "this image is not a material".

The reclassify backfill stamped `API error: ... credit balance` as is_material=false, so
12 trim images lost their place in search with a plausible-looking verdict. It also wrote a
`classification` column document_images does not have, so the endpoint twin could never save.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_verdict_module():
    path = Path(__file__).resolve().parents[2] / "app" / "services" / "images" / "classification_verdict.py"
    spec = importlib.util.spec_from_file_location("classification_verdict", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_verdict = _load_verdict_module()
VERDICT_COLUMNS = _verdict.VERDICT_COLUMNS
MIN_CLASSIFY_SIDE_PX = _verdict.MIN_CLASSIFY_SIDE_PX
classification_failed = _verdict.classification_failed
failed_result = _verdict.failed_result
verdict_update = _verdict.verdict_update
_below_minimum_size = _verdict.below_minimum_size

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _blank(path: str) -> str:
    spec = importlib.util.spec_from_file_location("comment_budget", _ROOT / "scripts" / "comment_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.blank_comments((_ROOT / path).read_text(encoding="utf-8"))


REAL = {"is_material": True, "confidence": 0.93, "reason": "tile swatch", "classification": "PRODUCT_IMAGE", "model": "claude-opus-5"}


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        failed_result("claude-opus-5", "API error: credit balance is too low"),
        {"is_material": False, "confidence": 0.0, "reason": "API error: 400"},
        {"is_material": False, "confidence": 0.0, "reason": "No tool_use block returned"},
        {"is_material": False, "confidence": 0.0, "reason": "API key missing"},
        {"is_material": False, "confidence": 0.0, "reason": "Error: boom"},
        {"is_material": False, "model": "claude-opus-5_failed"},
    ],
)
def test_outages_are_failures(result):
    assert classification_failed(result)


def test_a_real_verdict_is_not_a_failure():
    assert not classification_failed(REAL)
    assert not classification_failed(dict(REAL, is_material=False, classification="DECORATIVE"))


def test_a_failed_result_cannot_be_written():
    with pytest.raises(ValueError):
        verdict_update({}, failed_result("claude-opus-5", "API error: x"), "test")


def test_verdict_update_writes_only_real_columns_and_clears_quarantine():
    update = verdict_update({"ai_classification": {"classification_pending": True}, "keep": 1}, REAL, "test")
    assert set(update) <= VERDICT_COLUMNS
    assert "classification" not in VERDICT_COLUMNS
    assert update["image_type"] == "PRODUCT_IMAGE"
    assert update["category"] == "product"
    assert update["metadata"]["keep"] == 1
    assert update["metadata"]["ai_classification"]["classification_pending"] is False


def test_every_writer_goes_through_the_helper():
    backfill = _blank("app/services/embeddings/classification_backfill.py")
    endpoint = _blank("app/api/images.py")
    assert "verdict_update(" in backfill and "classification_failed(" in backfill
    assert "verdict_update(" in endpoint and "classification_failed(" in endpoint
    assert '"classification": "material"' not in backfill + endpoint


def test_the_classifier_marks_its_own_failures():
    rag = _blank("app/services/search/rag_service.py")
    start = rag.index("async def _classify_image_material(")
    body = rag[start : rag.index("\n    async def ", start + 1)]
    assert "'reason': 'API key missing'" not in body
    assert "'reason': f'API error" not in body
    assert body.count("failed_result(") >= 4


def test_tiny_images_skip_the_model_call():
    assert _below_minimum_size({"width": 60, "height": 40})
    assert _below_minimum_size({"width": MIN_CLASSIFY_SIDE_PX - 1, "height": 20})
    assert not _below_minimum_size({"width": MIN_CLASSIFY_SIDE_PX, "height": 30})
    assert not _below_minimum_size({"width": 800, "height": 600})
    assert not _below_minimum_size({})
    assert not _below_minimum_size({"width": "x", "height": None})


def test_the_vision_rollup_does_not_publish_page_text_on_the_product():
    src = _blank("app/api/pdf_processing/stage_4_products.py")
    assert 'candidates["detected_text"]' not in src
    assert 'fill_if_empty("detected_text"' not in src


def test_the_classification_model_is_declared_once():
    src = _blank("app/config.py")
    assert src.count("anthropic_model_classification: str") == 1
