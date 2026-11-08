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


def test_a_deferred_image_is_counted_as_deferred_not_as_a_failed_save():
    src = _code("app/services/images/image_processing_service.py")
    tally = src[src.index('if error == "awaiting_image_sourcing":'):]
    assert tally.index('elif error == "embedding_deferred":') < tally.index("elif error:")


def test_work_outside_the_job_reads_the_documents_own_marker():
    src = _code("app/api/pdf_processing/stage_4_products.py")
    body = src[src.index("async def reembed_product_text"):]
    assert body.index("document_deferred(supabase, document_id)") < body.index("build_product_page_body_text(")
    repair = _code("app/services/pdf/post_ingestion_repair.py")
    assert "embed = not embeddings_deferred()" in repair and repair.count("embed=embed") == 4
    assert repair.count("if embed else") == 1 and "if ids and embed else" in repair
    run = _code("app/services/embeddings/document_embed_run.py")
    run = run[run.index("async def embed_document"):]
    assert run.index("mark_document, db, document_id, False") < run.index("backfill_product_text_embeddings(")


def test_image_analysis_written_after_stage_4_7_still_reaches_the_product():
    run = _code("app/services/embeddings/document_embed_run.py")
    run = run[run.index("async def embed_document"):]
    images = run.index("_embed_confirmed_material(")
    rollup = run.index("reroll_products(document_id, product_ids)")
    assert images < rollup < run.index("backfill_product_text_embeddings(")
    stage = _code("app/api/pdf_processing/stage_4_products.py")
    assert "product_vision = linked_vision.get(product_id) or" in stage
    assert 'markers_changed = "facet_canonicalization" in new_metadata' in stage
    sourcing = _code("app/services/images/image_sourcing.py")
    body = sourcing[sourcing.index("async def run_all"):]
    assert "reroll_products(self.document_id, analysed)" in body and "not embeddings_deferred()" in body


def test_no_backfill_reaches_a_document_waiting_on_its_embed_run():
    src = _code("app/services/embeddings/text_embedding_backfill.py")
    products = src[src.index("async def _fetch_products_missing_embedding"):src.index("async def _fetch_chunks_missing_embedding")]
    assert 'outside_documents("source_document_id", deferred)' in products
    chunks = src[src.index("async def _fetch_chunks_missing_embedding"):src.index("async def backfill_product_text_embeddings")]
    assert "outside_documents('document_id', deferred)" in chunks and ".or_(wanted)" in chunks
    assert ed.outside_documents("document_id", ["a", "b"]) == "document_id.is.null,document_id.not.in.(a,b)"


def test_the_documents_marker_is_set_and_cleared_without_touching_other_keys():
    writes = []

    class Q:
        def __init__(self, rows):
            self.rows = rows

        def select(self, *_):
            return self

        def eq(self, *_):
            return self

        def limit(self, *_):
            return self

        def update(self, payload):
            writes.append(payload)
            return self

        def execute(self):
            return types.SimpleNamespace(data=self.rows)

    db = types.SimpleNamespace(table=lambda _: Q([{"metadata": {"title": "x"}}]))
    ed.mark_document(db, "d", True)
    assert writes[-1] == {"metadata": {"title": "x", "embedding_deferred": True}}
    db = types.SimpleNamespace(table=lambda _: Q([{"metadata": {"title": "x", "embedding_deferred": True}}]))
    assert ed.document_deferred(db, "d") is True
    ed.mark_document(db, "d", False)
    assert writes[-1] == {"metadata": {"title": "x"}}
    assert ed.document_deferred(db, None) is False
