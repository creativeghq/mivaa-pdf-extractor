"""What the pipeline fixed on its own is reported, and a fix done worse than intended is never counted as success (#536)."""

import ast
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


ic = _load("ingestion_corrections_under_test", "app/services/pdf/ingestion_corrections.py")
budget = _load("comment_budget_for_corrections", "scripts/comment_budget.py")


def _code(relative):
    return budget.blank_comments((_ROOT / relative).read_text(encoding="utf-8"))


def _by_code(corrections):
    return {c.code: c for c in corrections}


def test_a_repair_that_fell_back_from_gemini_is_degraded_and_not_counted_as_repaired():
    summary = {"restored_ids": ["a", "b", "c"], "restored_without_gemini": ["b", "c"],
               "gemini_unavailable_reasons": ["HTTP 500: prepayment depleted", "HTTP 500: prepayment depleted"]}
    got = _by_code(ic.sourcing_corrections(summary))
    assert got["images_repaired"].outcome == ic.CORRECTED and got["images_repaired"].subjects == ["a"]
    degraded = got["images_repaired_without_gemini"]
    assert degraded.outcome == ic.DEGRADED and degraded.subjects == ["b", "c"] and "prepayment depleted" in degraded.message


def test_a_crop_kept_at_low_resolution_is_degraded():
    got = _by_code(ic.sourcing_corrections({"kept_source_ids": ["x"]}))
    assert got["images_kept_low_resolution"].outcome == ic.DEGRADED and got["images_kept_low_resolution"].count == 1


def test_sourcing_with_nobody_to_bill_needs_a_decision():
    got = _by_code(ic.sourcing_corrections({"unbilled": True}))
    assert got["ai_work_unbilled"].outcome == ic.NEEDS_DECISION


def test_a_quiet_run_reports_nothing():
    assert ic.sourcing_corrections({"restored_ids": [], "unbilled": False}) == []
    assert ic.settle_corrections({"excluded": {}, "undecided": 0}) == []


def test_settling_reports_links_exclusions_room_scenes_and_what_it_could_not_decide():
    got = _by_code(ic.settle_corrections({"linked_by_label": 2, "linked_by_vision": 3, "multi_linked": ["r1"],
                                          "excluded": {"decorative_image": 4}, "undecided": 1,
                                          "failed_pages": [{"pdf_page": 3}]}))
    assert got["images_linked"].count == 5
    assert got["room_scenes_linked_to_several_series"].subjects == ["r1"]
    assert got["images_excluded"].detail == {"by_reason": {"decorative_image": 4}}
    assert got["images_undecided"].outcome == ic.NEEDS_DECISION and "1 page(s) failed" in got["images_undecided"].message


def test_counts_per_outcome_for_the_job():
    cs = [ic.Correction("a", ic.CORRECTED, "", 3), ic.Correction("b", ic.DEGRADED, "", 2), ic.Correction("c", ic.CORRECTED, "", 1)]
    assert ic.outcome_counts(cs) == {"corrected": 4, "degraded": 2, "needs_decision": 0}


def test_sourcing_records_its_corrections_except_on_a_dry_run():
    src = _code("app/services/images/image_sourcing.py")
    body = src[src.index("async def run_all"):src.index("async def run_image_sourcing")]
    assert "if not self.dry_run" in body and body.index("if not self.dry_run") < body.index("sourcing_corrections(")


def test_the_gemini_fallback_is_marked_where_the_repair_is_stored():
    src = _code("app/services/images/image_sourcing.py")
    body = src[src.index("async def _repair"):src.index("async def run_product")]
    assert "gemini_failed = self._gemini_failure" in body
    assert body.index('self.summary["restored_ids"]') < body.index('self.summary["restored_without_gemini"]')


def test_every_way_gemini_can_not_run_says_why():
    src = _code("app/services/images/image_sourcing.py")
    body = src[src.index("async def _gemini_restore"):src.index("async def _repair")]
    returns = body.count("return None")
    assert body.count("self._gemini_failure = ") - 1 >= returns


def test_sourcing_and_settling_have_someone_to_bill():
    assert "billing_principal(" in _code("app/services/images/image_sourcing.py")
    repair = _code("app/services/pdf/post_ingestion_repair.py")
    assert "user_id=payer" in repair[repair.index('step("series_reconcile"'):]
    assert "user_id=payer" in repair[repair.index('step("unlinked_images"'):]
    settle = _code("app/services/images/unlinked_images.py")
    assert "user_id=user_id" in settle[settle.index("async def _ask"):settle.index("async def settle_unlinked_images")]


def _func(relative, name):
    src = _code(relative)
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    return ast.get_source_segment(src, node)


def test_firecrawl_is_charged_per_firecrawl_credit_not_per_token():
    body = _func("app/services/core/ai_call_logger.py", "log_firecrawl_call")
    assert "debit_credits_for_firecrawl" in body and "input_tokens=0" not in body.split("log_data")[0]


@pytest.mark.parametrize("method", ["debit_credits_for_ai_operation", "debit_credits_for_firecrawl"])
def test_a_successful_debit_hands_back_the_usage_row_it_wrote(method):
    body = _func("app/services/integrations/credits_integration_service.py", method)
    assert "'usage_log_id': result.get('usage_log_id')" in body
