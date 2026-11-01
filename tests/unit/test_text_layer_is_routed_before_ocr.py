import importlib.util
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _blank_comments():
    spec = importlib.util.spec_from_file_location("comment_budget_for_guard", _ROOT / "scripts/comment_budget.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["comment_budget_for_guard"] = module
    spec.loader.exec_module(module)
    return module.blank_comments


def _code(relative: str) -> str:
    return _blank_comments()((_ROOT / relative).read_text(encoding="utf-8"))


STAGE_1 = "app/api/pdf_processing/stage_1_layout_precompute.py"


def _worker_body() -> str:
    src = _code(STAGE_1)
    start = src.index("async def _process_one_page")
    end = src.index("payload = {", start)
    return src[start:end]


def test_the_text_layer_is_planned_before_any_page_is_rendered_or_sent_to_ocr():
    body = _worker_body()
    assert body.index("_plan_page") < body.index("_render_physical_page") < body.index("run_structural_pass")


def test_ocr_only_runs_when_the_text_layer_was_not_used():
    body = _worker_body()
    gate = body.index("if not use_text_layer:")
    assert gate < body.index("_render_physical_page") and gate < body.index("run_structural_pass")


def test_text_layer_rows_are_written_with_their_own_version():
    src = _code(STAGE_1)
    assert '"processing_version": processing_version' in src
    assert "processing_version = TEXT_LAYER_VERSION" in src


def test_no_reader_still_compares_the_layout_version_to_the_ocr_literal():
    offenders = []
    for path in (_ROOT / "app").rglob("*.py"):
        code = _blank_comments()(path.read_text(encoding="utf-8"))
        if re.search(r"processing_version[^\n]{0,40}(!=|==)\s*[\"']paddleocr-vl[\"']", code):
            offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == []


def test_every_layout_reader_uses_the_shared_version_set():
    for relative in (STAGE_1, "app/services/pdf/region_crop_extractor.py"):
        assert "LAYOUT_PROCESSING_VERSIONS" in _code(relative), relative


def test_a_text_layer_table_is_not_cropped_as_a_product_image():
    code = _code("app/services/pdf/region_crop_extractor.py")
    assert 'region_type == "TABLE"' in code and "TEXT_LAYER_VERSION" in code


def test_a_page_that_loses_numbers_in_the_text_path_falls_back_to_ocr():
    code = _code(STAGE_1)
    assert code.index("lost_number_share(") < code.index("elements_lost_numbers")


def test_table_checks_run_on_ocr_pages_and_are_recorded_on_the_page():
    code = _code(STAGE_1)
    assert "verify_ocr_tables" in code and '"table_checks": page_tables' in code and '"needs_review"' in code
    assert "if _text_layer_enabled() else" in code


def test_the_rollback_switch_turns_every_new_behaviour_off():
    code = _code(STAGE_1)
    assert "return _text_layer_enabled() and" in code
    assert 'extraction_path != "text_layer"' in code


def test_a_rotated_page_is_never_read_from_the_text_layer():
    body = _code(STAGE_1)
    plan = body[body.index("def _plan_page"):body.index("async def precompute_document_layout")]
    assert plan.index("page.rotation") < plan.index("extract_page_content")
    assert plan.index("assess_text_layer") < plan.index("extract_tables(page, clip)")


def test_edge_extraction_deletes_only_the_sources_whose_call_succeeded():
    code = _code("app/services/products/product_relationship_service.py")
    assert "succeeded" in code
    delete_block = code[code.index("batch = succeeded["):]
    assert "src_product_id" in delete_block.split("deduped")[0]
    assert "batch = pids[i:i + 100]" not in code


def test_flagged_tables_do_not_enrich_the_product():
    code = _code("app/services/metadata/table_metadata_extractor.py")
    assert "needs_review" in code and code.index("needs_review") < code.index("table_type = table.get")
