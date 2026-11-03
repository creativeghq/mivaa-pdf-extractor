"""The discovered list is reviewed before anything is created, and an edit is applied whole or not at all (#513)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dr = _load("discovery_review_under_test", "app/services/discovery/discovery_review.py")
budget = _load("comment_budget_for_review", "scripts/comment_budget.py")

CATALOG = {"model_used": "x", "products": [
    {"name": "LINEAR MAX", "page_range": [4, 5], "metadata": {"variants": [{}, {}]}},
    {"name": "LINEAR MAX - M", "page_range": [6], "metadata": {}},
    {"name": "LINEAR MAX, LINEAR MAX - M", "page_range": [4, 7], "metadata": {}},
    {"name": "HM FITTINGS", "page_range": [40], "metadata": {}},
]}


def names(catalog):
    return [p["name"] for p in catalog["products"]]


def test_a_mislabelled_merge_folds_into_the_real_product_with_its_pages():
    out, log = dr.apply_review(CATALOG, [{"name": "LINEAR MAX, LINEAR MAX - M", "action": "merge_into", "into": "LINEAR MAX"},
                                         {"name": "HM FITTINGS", "action": "drop"}])
    assert names(out) == ["LINEAR MAX", "LINEAR MAX - M"]
    assert out["products"][0]["page_range"] == [4, 5, 7]
    assert out["products"][0]["metadata"]["merged_from"] == ["LINEAR MAX, LINEAR MAX - M"]
    assert len(log) == 2 and out["model_used"] == "x"


def test_a_rename_keeps_the_product_and_its_pages():
    out, _ = dr.apply_review(CATALOG, [{"name": "linear max - m", "action": "rename", "new_name": "Linear Max M"}])
    assert names(out)[1] == "Linear Max M" and out["products"][1]["page_range"] == [6]


@pytest.mark.parametrize("edits", [
    [{"name": "NOPE", "action": "drop"}],
    [{"name": "LINEAR MAX", "action": "explode"}],
    [{"name": "LINEAR MAX", "action": "rename"}],
    [{"name": "LINEAR MAX", "action": "merge_into", "into": "LINEAR MAX"}],
    [{"name": "LINEAR MAX - M", "action": "merge_into", "into": "HM FITTINGS"}, {"name": "HM FITTINGS", "action": "drop"}],
    [{"name": "HM FITTINGS", "action": "rename", "new_name": "linear max"}],
])
def test_a_bad_edit_applies_nothing(edits):
    with pytest.raises(dr.ReviewError):
        dr.apply_review(CATALOG, edits)
    assert names(CATALOG)[0] == "LINEAR MAX" and CATALOG["products"][0]["page_range"] == [4, 5]


def test_the_review_row_shows_what_spots_a_duplicate():
    row = dr.review_rows(CATALOG["products"])[0]
    assert row == {"name": "LINEAR MAX", "page_range": [4, 5], "category": None, "dimensions": [], "variants": 2, "confidence": None}


def test_a_dry_run_cap_is_a_small_whole_number():
    assert dr.product_limit(None) is None and dr.product_limit("3") == 3
    for bad in ("x", -1, 501):
        with pytest.raises(dr.ReviewError):
            dr.product_limit(bad)


def test_the_gate_parks_before_any_product_is_created():
    src = budget.blank_comments((_ROOT / "app/api/pdf_processing/stage_0_discovery.py").read_text(encoding="utf-8"))
    assert src.index("'review_discovery'") < src.index("create_single_product(")
    assert src.index("_product_limit(") < src.index("products_to_create =")


def test_the_orchestrator_stops_on_a_parked_job_before_the_product_pipeline():
    src = budget.blank_comments((_ROOT / "app/api/rag_routes.py").read_text(encoding="utf-8"))
    body = src[src.index("async def process_document_with_discovery"):]
    assert body.index('stage_0_result.get("awaiting_review")') < body.index("process_products_parallel")


def test_approving_claims_the_parked_job_before_resuming_it():
    src = budget.blank_comments((_ROOT / "app/api/rag_routes.py").read_text(encoding="utf-8"))
    body = src[src.index("async def decide_discovery_review"):]
    assert body.index(".eq('status', 'awaiting_review')") < body.index("restart_job_from_checkpoint(")
