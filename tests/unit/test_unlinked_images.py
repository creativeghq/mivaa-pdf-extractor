"""Every live material image ends linked or excluded with a reason, and a model answer never guesses a series."""

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("unlinked_images_under_test", _ROOT / "app/services/images/unlinked_images.py")
ui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ui)


def test_candidates_are_the_printed_series_then_the_series_whose_pages_hold_the_image():
    product_of = {"calma": "p1", "saboya": "p2", "tiber": "p3"}
    pages_of = {"p3": {7}, "p2": {7, 8}, "p9": {9}}
    assert ui.page_candidates(7, {"calma", "saboya"}, product_of, pages_of) == ["p1", "p2", "p3"]
    assert ui.page_candidates(9, set(), product_of, pages_of) == ["p9"]
    assert ui.page_candidates(1, {"unknown"}, product_of, pages_of) == []


def test_a_verdict_counts_only_when_it_names_a_real_image_and_a_real_series():
    answer = {"images": [
        {"image": 1, "series": 2, "kind": "product"},
        {"image": 2, "series": None, "kind": "decorative"},
        {"image": 3, "series": 9, "kind": "product"},
        {"image": 4, "series": None, "kind": "unclear"},
        {"image": 7, "series": 1, "kind": "product"},
        {"image": 5, "series": 1, "kind": "maybe"},
    ]}
    assert ui.verdicts(answer, n_images=5, n_candidates=2) == {
        0: ("product", 1), 1: ("decorative", None), 2: ("unclear", None), 3: ("unclear", None)}


def test_each_non_link_outcome_has_a_stated_reason():
    assert ui.exclusion_reason("decorative") == "decorative_image"
    assert ui.exclusion_reason("unclear") == "unattributable_on_shared_page"


def _body(name):
    src = (_ROOT / "app/services/images/unlinked_images.py").read_text(encoding="utf-8")
    return ast.unparse(next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.AsyncFunctionDef) and n.name == name))


def test_a_failed_vision_call_decides_nothing():
    body = _body("settle_unlinked_images")
    handler = body[body.index("except Exception"):]
    assert "undecided" in handler and "exclude(" not in handler and "link(" not in handler


def test_an_excluded_image_loses_its_vectors_and_says_who_excluded_it():
    body = _body("settle_unlinked_images")
    assert "delete_embeddings_by_image_ids" in body and "'by': EXCLUDED_BY" in body


def test_settling_runs_after_label_linking_in_the_repair():
    src = (_ROOT / "app/services/pdf/post_ingestion_repair.py").read_text(encoding="utf-8")
    assert src.index('step("label_links"') < src.index('step("unlinked_images"')
