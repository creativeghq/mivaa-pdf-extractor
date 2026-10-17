"""Guards for #494 product data: linking, vision validation, rollups, propagation, creation.

Pure helpers are exercised by extracting them from the source with `ast` (CI installs
pytest only, so the modules themselves cannot be imported). Source rules read code with
comments and docstrings blanked, so prose can never satisfy them.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
STAGE3 = APP / "api" / "pdf_processing" / "stage_3_images.py"
STAGE4 = APP / "api" / "pdf_processing" / "stage_4_products.py"
STAGE5 = APP / "api" / "pdf_processing" / "stage_5_quality.py"
PROCESSOR = APP / "api" / "pdf_processing" / "product_processor.py"
LINK = APP / "services" / "discovery" / "entity_linking_service.py"
ASSOC = APP / "services" / "images" / "multi_modal_image_product_association_service.py"
IMAGES = APP / "services" / "images" / "image_processing_service.py"
TEXT_FOLD = APP / "utils" / "text_fold.py"


def _code(path: Path) -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from comment_budget import blank_comments
        return blank_comments(path.read_text(encoding="utf-8"))
    finally:
        sys.path.pop(0)


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


def _fold_identity():
    spec = importlib.util.spec_from_file_location("_text_fold_for_test", TEXT_FOLD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fold_identity


def _extract(path: Path, names: set, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import re
    ns: Dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "re": re,
        "fold_identity": _fold_identity(),
    }
    ns.update(extra or {})
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns


# ── B4: every association row carries the tenant; each link step reports itself ──

def test_every_link_step_has_its_own_failure_marker():
    for helper in ("_link_images_for_product", "_link_chunks_for_product", "_apply_table_metadata"):
        body = _func(LINK, helper)
        tail = body.split("except Exception")[-1]
        assert "LINKING_FAILED" in tail and "'failed'" in tail, f"{helper} hides its failure"
    body = _func(LINK, "link_product_entities")
    assert "step_status" in body and "link_failed_steps" in body


def test_association_rows_are_tenant_bound():
    assert "'workspace_id': workspace_id" in _func(LINK, "_link_images_for_product")
    assert "'workspace_id': workspace_id" in _func(LINK, "link_images_to_products")
    assert "'workspace_id': workspace_id" in _func(STAGE4, "create_single_product")


# ── B5: the vision validator is the schema the model was constrained to ──────

def test_vision_analysis_is_validated_by_the_schema():
    body = _func(IMAGES, "_validate_vision_analysis")
    assert "VisionAnalysis.model_validate" in body
    assert "color_palette" not in _code(IMAGES)


# ── B6: icon rollup is the product's own pages and only fills gaps ────────────

def test_icon_rollup_reads_only_the_products_pages():
    body = _func(STAGE4, "_merge_icon_metadata_into_product")
    assert ".in_('page_number', pages)" in body
    create = _func(STAGE4, "create_single_product")
    assert "if _is_empty_value(metadata.get(spec_field)):" in create


# ── B7: only verified catalogue-level facts propagate; dimensions are per product ──

def test_propagation_never_copies_per_product_specs():
    code = _func(STAGE4, "propagate_common_fields_to_products")
    for per_product in ("available_sizes", "'thickness'", "body_type", "composition"):
        assert per_product not in code, f"{per_product} is copied across products again"
    assert "_unanimous_value" in code


def test_unanimity_is_required_for_a_shared_value():
    ns = _extract(STAGE4, {"_unanimous_value", "_is_empty_value"})
    agree = [{"metadata": {"factory_name": "Acme"}}, {"metadata": {"factory_name": "ACME"}}, {"metadata": {}}]
    assert ns["_unanimous_value"](agree, "factory_name") == ("Acme", 2)
    clash = [{"metadata": {"factory_name": "Acme"}}, {"metadata": {"factory_name": "Other"}}]
    assert ns["_unanimous_value"](clash, "factory_name") == (None, 0)


def test_dimension_scan_keeps_the_stated_unit_and_skips_unitless():
    ns = _extract(STAGE4, {"_scan_dimensions", "_dimension_patterns", "_SIZE_RE", "_THICKNESS_RE"})
    found = ns["_scan_dimensions"]("Formats 100x200 mm and 60x120 cm. Bare 30x30. Thickness 9 mm")
    assert found["available_sizes"] == ["100×200 mm", "60×120 cm"]
    assert found["thickness_mm"] == 9.0
    assert ns["_scan_dimensions"]("spessore 0,8 cm")["thickness_mm"] == 8.0
    assert ns["_scan_dimensions"]("only 30x30 here") == {}


def test_dimension_scan_reads_each_products_own_chunks():
    body = _func(STAGE4, "extract_dimensions_from_document_chunks")
    assert "text_by_product.get(product_id)" in body
    assert "combined_text" not in body


# ── B8: shared pages need product-specific evidence ───────────────────────────

def test_ambiguous_same_page_pairs_stay_unlinked():
    body = _func(ASSOC, "create_document_associations")
    assert "if len(on_page) > 1:" in body and "product_evidence(" in body
    assert "ambiguous_unlinked" in body
    assert "product_evidence(" in _func(LINK, "_link_images_for_product")


def test_product_evidence_names_the_product():
    ns = _extract(ASSOC, {"product_evidence", "_name_core"})
    evidence = ns["product_evidence"]
    product = {"name": "MAISON by ONSET"}
    assert evidence({"product_name": "Maison"}, product) == "vision_region"
    assert evidence({"caption": "The Maison range in white"}, product) == "caption_name"
    assert evidence({"caption": "Image from page 4"}, product) is None
    assert evidence({"caption": "Maisonette tiles"}, product) is None


def test_spread_images_resolve_against_the_whole_sheet():
    ns = _extract(STAGE3, {"_sheet_physical_pages"})
    mapping = {4: (2, "left"), 5: (2, "right"), 6: (3, "left")}
    assert ns["_sheet_physical_pages"](2, True, mapping) == [4, 5]
    assert ns["_sheet_physical_pages"](2, False, mapping) == []
    body = _func(STAGE3, "_process_product_images")
    assert "spread_images_dropped_other_page" in body
    failure = body.split("except Exception as region_err")[1][:300]
    assert "'status': 'failed'" in failure and "error_type" in failure


# ── creation: destination, identity, lifecycle ────────────────────────────────

def test_create_routes_columns_sets_identity_and_draft():
    body = _func(STAGE4, "create_single_product")
    for needle in ("'status': 'draft'", "'category_id': _category_id,",
                   "_category_id = _resolve_category_id(", "except CategoryRegistryUnavailable",
                   "'confidence_score': _confidence_score(", "**column_values"):
        assert needle in body, needle
    dedup = body.index("_find_workspace_duplicate(")
    assert dedup < body.index("_classify_product("), "dedup must run before the paid classifier"
    assert dedup < body.index(".insert(product_data)")


def test_confidence_score_is_a_fraction_or_unknown():
    ns = _extract(STAGE4, {"_confidence_score"})
    assert ns["_confidence_score"](0.7) == 0.7
    assert ns["_confidence_score"](85) == 0.85
    assert ns["_confidence_score"](None) is None
    assert ns["_confidence_score"](-1) is None


def test_enriched_values_are_recanonicalised():
    for fn in ("enrich_products_from_chunks_and_vision", "propagate_common_fields_to_products",
               "extract_dimensions_from_document_chunks"):
        assert "recanonicalize_product(" in _func(STAGE4, fn), fn
    assert "recanonicalize_product(" in _func(LINK, "_apply_table_metadata")


def test_vision_category_uses_the_registry_resolver():
    body = _func(STAGE4, "_normalize_material_category")
    assert "field_registry.resolve_category(" in body
    assert "_MATERIAL_TYPE_TO_CATEGORY" not in _code(STAGE4)


def test_enrichment_has_no_document_wide_fallback():
    body = _func(STAGE4, "enrich_products_from_chunks_and_vision")
    assert "product_chunks = all_chunks" not in body
    assert "product_vision = all_vision" not in body


# ── stage 3 / 3.5 / 5 / processor ─────────────────────────────────────────────

def test_embedding_to_text_is_retired():
    assert not (APP / "services" / "embeddings" / "embedding_to_text_service.py").exists()
    code = _code(IMAGES)
    assert "VisualMetadataService" not in code and "embedding_to_text" not in code


def test_vision_ocr_runs_off_the_event_loop_and_no_tool_use_is_quarantined():
    body = _func(IMAGES, "_try_claude_material_analysis")
    assert "asyncio.to_thread(\n                self._ocr_context_for_vision" in body
    assert "'_no_tool_use' in _cls_model" in _code(IMAGES)


def test_vector_counts_come_from_confirmed_writes():
    body = _func(IMAGES, "_process_single_image_with_retry")
    assert "'understanding': understanding_saved" in body
    assert "'color' in aspects_saved" in body


def test_open_breaker_is_reported_as_skipped():
    body = _func(STAGE5, "process_stage_5_quality")
    breaker = body.index("except CircuitBreakerError")
    assert "'status': 'skipped'" in body[breaker:breaker + 300]


def test_failed_merge_does_not_complete_the_creation_stage():
    body = _func(PROCESSOR, "process_single_product")
    merge_fail = body.index('raise RuntimeError(f"metadata merge failed for')
    assert merge_fail <body.index("ProductStage.CREATION,\n            {\"product_db_id\"")


def test_icon_extraction_gets_its_category_and_a_failure_is_stamped():
    body = _func(IMAGES, "_process_icon_candidate")
    assert "category=material_category" in body and "job_id=job_id" in body
    assert "'status': 'failed'" in body.split("except Exception as ocr_err")[1][:900]


def test_partial_chunking_and_failed_region_crops_are_recorded():
    body = _func(PROCESSOR, "process_single_product")
    assert '"failed_pages": chunk_result.get(\'failed_pages\', [])' in body
    assert "chunking_failed_pages_product_" in body
    assert '"region_crop_status": image_result.get(\'region_crop_status\')' in body
