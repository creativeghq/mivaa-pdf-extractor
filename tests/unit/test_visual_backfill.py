"""An image whose visual vector failed at creation is repaired, and only a live material image is."""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC = (Path(__file__).resolve().parents[2] / "app" / "services" / "embeddings" / "visual_backfill.py").read_text(encoding="utf-8")


def _load(name):
    node = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"Any": object, "Dict": dict, "NOT_RETIRED_FILTER": "R", "NOT_QUARANTINED_FILTER": "Q",
             "NOT_AWAITING_SOURCING_FILTER": "S"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "visual_backfill", "exec"), scope)
    return scope[name]


owes = _load("owes_visual_vector")
LIVE = {"image_url": "https://x/a.jpg", "metadata": {"ai_classification": {"is_material": True}}}


def test_a_live_material_image_is_owed_a_vector():
    assert owes(LIVE)
    assert owes({"image_url": "https://x/a.jpg", "metadata": {}})


@pytest.mark.parametrize("meta", [
    {"excluded_from_products": {"reason": "replaced"}},
    {"bundle_skipped_reason": "icon_candidate_spec_path"},
    {"ai_classification": {"is_material": False}},
    {"ai_classification": {"classification_pending": True}},
    {"image_quality": {"sourcing": "pending"}},
])
def test_a_row_another_step_owns_is_left_alone(meta):
    assert not owes({"image_url": "https://x/a.jpg", "metadata": meta})


def test_no_url_means_nothing_to_embed():
    assert not owes({"image_url": None, "metadata": {}})


def test_the_filter_selects_missing_vectors_and_honours_the_shared_exclusions():
    f = _load("missing_visual_filter")("document_id.is.null")
    assert f.startswith("and(or(has_slig_embedding.is.null,has_slig_embedding.is.false),R,")
    assert "metadata->>bundle_skipped_reason.is.null" in f and "or(Q)" in f and "or(S)" in f
    assert f.endswith(",or(document_id.is.null))")


def test_a_failed_embed_is_counted_never_substituted():
    fn = next(n for n in ast.walk(ast.parse(SRC)) if isinstance(n, ast.AsyncFunctionDef) and n.name == "backfill_visual_embeddings")
    body = ast.unparse(fn)
    assert "_generate_visual_embedding" in body and "upsert_image_embedding" in body
    assert "summary['failed'] += 1" in body and "failed_ids" in body
