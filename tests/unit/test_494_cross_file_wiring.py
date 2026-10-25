"""Guards for the #494 wiring that spans files: registry partition, stage 4 markers,
category registry handling, recanonicalize `force`, and product-scoped image vectors.

Source rules read code with comments and docstrings blanked, so prose cannot satisfy them.
"""

import ast
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
DISC = APP / "services" / "discovery" / "product_discovery_service.py"
STAGE4 = APP / "api" / "pdf_processing" / "stage_4_products.py"
IMPORT = APP / "services" / "integrations" / "data_import_service.py"
FACETS = APP / "api" / "facet_routes.py"
IMAGES = APP / "services" / "images" / "image_processing_service.py"


def _func(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            sys.path.insert(0, str(ROOT / "scripts"))
            try:
                from comment_budget import blank_comments
                return blank_comments(ast.get_source_segment(source, node) or "")
            finally:
                sys.path.pop(0)
    raise AssertionError(f"{path.name} no longer defines {name}()")


def test_partition_asks_whether_the_key_is_registered():
    # destination_of() answers 'metadata' for an unknown key, so it made every key "known".
    body = _func(DISC, "_partition_by_registry")
    assert "is_registered(key)" in body
    assert "destination_of" not in body


def test_spec_extraction_gets_the_category_and_records_degradation():
    body = _func(STAGE4, "enrich_products_from_chunks_and_vision")
    assert 'category=metadata.get("material_category")' in body
    assert '"_degraded_tiers"' in body and '"status": "degraded"' in body
    assert 'new_metadata["_spec_extraction"] = spec_status' in body


def test_a_failed_functional_extraction_is_recorded_not_read_as_empty():
    body = _func(STAGE4, "enrich_products_from_chunks_and_vision")
    failed = body.index('prop_result.status == "failed"')
    assert failed < body.index("prop_result.coverage_pct > 0")
    assert 'new_metadata["_functional_extraction"]' in body
    assert '"category_confidence": prop_result.category_confidence' in body
    assert "if not filled and not markers_changed:" in body


def test_registry_validation_is_category_scoped():
    body = _func(STAGE4, "create_single_product")
    assert "validate_metadata_against_registry(\n            metadata, category=" in body


@pytest.mark.parametrize("path", [STAGE4, IMPORT])
def test_category_registry_is_cached_and_its_failure_is_never_an_empty_map(path):
    src = path.read_text(encoding="utf-8")
    assert "get_category_registry(" in src
    assert "load_category_registry(" not in src
    assert "except CategoryRegistryUnavailable" in src


def test_recanonicalize_route_passes_force():
    body = _func(FACETS, "recanonicalize")
    assert "force=req.force" in body


def test_product_scoped_images_anchor_tenancy_on_the_product():
    body = _func(IMAGES, "save_images_and_generate_clips")
    product = body.index("_assert_product_in_workspace(product_id, workspace_id)")
    assert product < body.index("icon_candidates =")
    embed = _func(IMAGES, "embed_product_images")
    assert embed.index("_assert_product_in_workspace(") < embed.index(".download")
    assert "staging_failures" in embed and "_mark_embedding_failed(" in embed


def test_imported_images_are_embedded_inline_and_failures_are_marked():
    body = _func(IMPORT, "create_product_from_payload")
    assert "await self._embed_linked_images(" in body
    embed = _func(IMPORT, "_embed_linked_images")
    assert "embed_product_images(" in embed
    assert "'status': 'failed'" in embed


def test_inline_image_embedding_checks_credits_before_the_paid_call():
    """Invariant 10: the balance is read BEFORE Claude/Voyage run, and a refusal is a marker."""
    embed = _func(IMPORT, "_embed_linked_images")
    assert embed.index("_embed_credit_block(") < embed.index("embed_product_images(")
    assert "'skipped_no_credits'" in embed
    block = _func(IMPORT, "_embed_credit_block")
    assert "get_available_credits(" in block
    assert block.count("return None") == 1, "only a sufficient balance may let the call run"


def test_the_upload_category_fills_an_uncategorised_product_before_it_is_resolved():
    body = _func(STAGE4, "create_single_product")
    classify = body.index("await _classify_product(")
    hint = body.index("_upload_category_hint(supabase, document_id)")
    resolve = body.index("_resolve_category_id(metadata.get('material_category')")
    assert classify < hint < resolve
    assert "metadata.get(\"material_category\") != upload_category" in body[hint:resolve]
    assert '"general_materials"' not in body[hint:resolve]


def test_a_pdf_product_insert_is_identity_keyed_and_safe_to_repeat():
    body = _func(STAGE4, "create_single_product")
    identity = body.index("import_identity_for(")
    insert = body.index(".upsert(product_data, on_conflict='id', ignore_duplicates=True)")
    assert identity < insert
    assert "product_data['id'] = str(uuid.uuid4())" in body[identity:insert]
    assert "allow_name_identity=bool(_factory_name)" in body
    assert ".insert(product_data)" not in body
    race = body[insert:]
    assert "'23505'" in race and "_reuse_existing_product(" in race


def test_ai_calls_with_only_a_job_bill_the_job_payer_before_debiting():
    logger_path = APP / "services" / "core" / "ai_call_logger.py"
    for method in ("log_claude_call", "log_replicate_call", "log_time_based_call"):
        body = _func(logger_path, method)
        assert body.index("await self._job_payer(job_id)") < body.index("try:"), method
    assert "not system_initiated" in _func(logger_path, "log_claude_call")
    assert "background_jobs" in _func(logger_path, "_job_payer")


def test_an_upload_carries_its_content_hash_and_a_duplicate_is_a_409():
    body = _func(APP / "api" / "rag_routes.py", "upload_document")
    assert '"p_content_sha256": None if allow_duplicate else _sha256_file(file_path)' in body
    rpc = body.index("rpc('create_document_upload_job'")
    dup = body.index('outcome.get("status") == "duplicate"')
    assert rpc < dup < body.index("background_tasks.add_task(")
    assert "HTTP_409_CONFLICT" in body[dup:] and "return JSONResponse(" in body[dup:]


def test_a_job_wipe_reaches_every_table_keyed_to_its_document():
    body = _func(APP / "services" / "utilities" / "cleanup_service.py", "delete_job_completely")
    for table, column in (("document_entities", "source_document_id"), ("processed_documents", "pdf_document_id")):
        assert f"('{table}', '{column}'" in body, table
    assert body.index("('processed_documents'") < body.index("doc_response = supabase_client.client.table('documents')")
    assert "'image_validations'" not in body, "dropped table"
    assert ".in_('product_id', product_ids)" in body and "table('image_product_associations')" in body
