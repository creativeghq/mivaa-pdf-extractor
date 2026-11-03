"""A deferred job builds no vectors anywhere, counts none of that as a failure, and one embed run builds them (#513)."""

import asyncio
import importlib.util
import sys
import types
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


ed = _load("embedding_deferral_under_test", "app/services/embeddings/embedding_deferral.py")
budget = _load("comment_budget_for_deferral", "scripts/comment_budget.py")


def _code(relative):
    return budget.blank_comments((_ROOT / relative).read_text(encoding="utf-8"))


def test_the_switch_reaches_child_tasks_and_does_not_leak_between_runs():
    async def run(on):
        ed.defer_embeddings(on)

        async def child():
            return ed.embeddings_deferred()

        return await asyncio.create_task(child()), await asyncio.to_thread(ed.embeddings_deferred)

    assert asyncio.run(run(True)) == (True, True)
    assert asyncio.run(run(False)) == (False, False)
    assert ed.embeddings_deferred() is False


@pytest.mark.parametrize("relative, before", [
    ("app/api/pdf_processing/stage_4_products.py", "build_product_page_body_text(\n                supabase,"),
    ("app/services/search/rag_service.py", "generate_contextualized_embeddings("),
    ("app/services/images/image_processing_service.py", "awaiting_image_sourcing"),
    ("app/services/images/image_sourcing.py", "_embed_confirmed_material(\n"),
    ("app/api/rag_routes.py", "embed_document_pages(\n                        document_id=document_id,"),
])
def test_every_inline_embedding_site_asks_the_switch_first(relative, before):
    src = _code(relative)
    at = src.index(before)
    assert "embeddings_deferred()" in src[max(0, at - 2500):at], f"{relative}: no deferral check before {before!r}"


def test_a_deferred_product_vector_is_not_counted_as_a_failure():
    src = _code("app/api/pdf_processing/stage_4_products.py")
    handler = src[src.index("except EmbeddingDeferred:"):src.index("except Exception as emb_err:")]
    assert "embedding_failed = False" in handler


def test_the_orchestrator_sets_the_switch_before_any_stage_and_audits_pre_embed():
    body = _code("app/api/rag_routes.py")
    body = body[body.index("async def process_document_with_discovery"):]
    assert body.index("defer_embeddings(") < body.index("process_stage_0_discovery(")
    stage5 = _code("app/api/pdf_processing/stage_5_quality.py")
    assert '"pre_embed" if embeddings_deferred()' in stage5


def test_a_deferred_catalogue_is_never_published_unembedded():
    body = _code("app/api/rag_routes.py")
    deferred = body[body.index('"ready_to_embed"'):]
    assert deferred.index("publish_on_complete = False") < deferred.index("if publish_on_complete:")


for _pkg in ("app", "app.services", "app.services.pdf"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_load("app.services.pdf.ocr_completeness", "app/services/pdf/ocr_completeness.py")
_load("app.services.pdf.ingestion_audit", "app/services/pdf/ingestion_audit.py")
er = _load("document_embed_run_under_test", "app/services/embeddings/document_embed_run.py")


def test_the_embed_run_takes_only_live_linked_images_without_a_vector():
    row = {"id": "a", "metadata": {}, "has_slig_embedding": False}
    assert er.owes_image_vector(row, {"a"})
    assert not er.owes_image_vector(row, set())
    assert not er.owes_image_vector({**row, "has_slig_embedding": True}, {"a"})
    assert not er.owes_image_vector({**row, "metadata": {"image_quality": {"sourcing": "pending"}}}, {"a"})
    assert not er.owes_image_vector({**row, "metadata": {"excluded_from_products": {}}}, {"a"})
    assert not er.owes_image_vector({**row, "image_type": "DECORATIVE"}, {"a"})
