"""Guards for `app/evaluation/field_judgement.py` and the stage-5 page judge that uses it.

Static where it must be: CI installs pytest alone, so the judge SERVICE is parsed
as source, and only the pure stdlib module is executed (loaded by path inside a
fixture). Each case was watched to fail against a broken version before it was
committed.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
MODULE = APP / "evaluation" / "field_judgement.py"
JUDGE = APP / "services" / "ai_validation" / "product_field_judge.py"
STAGE5 = APP / "api" / "pdf_processing" / "stage_5_quality.py"


@pytest.fixture(scope="module")
def fj():
    spec = importlib.util.spec_from_file_location("_field_judgement", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── the pure part ─────────────────────────────────────────────────────────────

def test_a_verdict_on_a_field_the_judge_was_not_shown_is_dropped_and_counted(fj):
    kept, dropped = fj.validate_verdicts(
        [{"field": "colour", "verdict": "ok", "score": 5, "reason": "printed beside the swatch"},
         {"field": "invented", "verdict": "wrong", "score": 1, "reason": "x"}],
        allowed_fields=["colour", "size_cm"],
    )
    assert [k["field"] for k in kept] == ["colour"]
    assert dropped and dropped[0]["why"] == "field was not put before the judge"


def test_out_of_enum_or_out_of_range_or_reasonless_is_dropped_never_coerced(fj):
    kept, dropped = fj.validate_verdicts(
        [{"field": "a", "verdict": "maybe", "score": 3, "reason": "r"},
         {"field": "b", "verdict": "ok", "score": 9, "reason": "r"},
         {"field": "c", "verdict": "wrong", "score": 1, "reason": ""},
         {"field": "d", "verdict": "ok", "score": 2.5, "reason": "r"}],
        allowed_fields="abcd",
    )
    assert kept == []
    assert {d["why"] for d in dropped} == {
        "verdict outside the enum", "score outside 1..5", "no reason", "score is not an integer",
    }


def test_absent_on_page_carries_no_score_and_duplicates_are_dropped(fj):
    kept, dropped = fj.validate_verdicts(
        [{"field": "a", "verdict": "absent_on_page", "score": 3, "reason": "nothing on the page names it"},
         {"field": "a", "verdict": "ok", "score": 5, "reason": "again"}],
        allowed_fields=["a"],
    )
    assert kept == [{
        "field": "a", "verdict": "absent_on_page", "score": None,
        "reason": "nothing on the page names it", "suggested_value": None,
    }]
    assert dropped[0]["why"] == "duplicate verdict for field"


def test_summary_is_counts_with_every_verdict_present(fj):
    kept, _ = fj.validate_verdicts(
        [{"field": "a", "verdict": "wrong", "score": 1, "reason": "r", "suggested_value": "60x120"},
         {"field": "b", "verdict": "ok", "score": 4, "reason": "r"}],
        allowed_fields=["a", "b"],
    )
    assert fj.summarize_verdicts(kept) == {"ok": 1, "suspect": 0, "wrong": 1, "absent_on_page": 0}
    assert kept[0]["suggested_value"] == "60x120"


def test_fields_for_judgement_drops_empties_caps_and_orders(fj):
    flat = {"z": "1", "a": "", "m": None, "b": "2"}
    assert fj.fields_for_judgement(flat) == {"b": "2", "z": "1"}
    big = {f"f{i:03d}": i for i in range(100)}
    assert len(fj.fields_for_judgement(big, max_fields=60)) == 60


def test_verdict_enum_matches_the_database_check(fj):
    # The DB CHECK on product_field_judgements.verdict is the enforcer; this list must equal it.
    assert fj.VERDICTS == ("ok", "suspect", "wrong", "absent_on_page")


# ── the service, as source ────────────────────────────────────────────────────

def test_judge_service_exists_and_uses_a_forced_tool_call():
    src = _src(JUDGE)
    assert "call_with_tool(" in src, "the judge must go through the forced-tool helper (invariant 9)"
    assert "json.loads(" not in src, "no salvage parser"


def test_judge_prompt_comes_from_the_database_with_no_fallback():
    src = _src(JUDGE)
    assert "load_prompt(" in src
    assert 'JUDGE_TASK = "product_field_judge"' in src
    assert not re.search(r"(?m)^[A-Z_]*PROMPT\s*=\s*[\"']", src), "a prompt constant in code is the banned fallback"


def test_judge_renders_the_page_itself_and_does_not_depend_on_page_embeddings():
    src = _src(JUDGE)
    assert "get_pixmap(" in src
    assert 'table("document_page_embeddings")' not in src, "renders were never written; the judge must not wait for them"


def test_judge_stamps_a_status_on_every_product_it_touched():
    src = _src(JUDGE)
    for status in ("ok", "failed", "skipped"):
        assert re.search(rf'_stamp\(\s*(product|p),\s*"{status}"', src), (
            f'"{status}" must be written to products.metadata.field_judgement'
        )


def test_judge_only_reads_low_confidence_products_of_this_document_in_this_workspace():
    src = _src(JUDGE)
    assert '.eq("source_document_id", document_id)' in src
    assert '.eq("workspace_id", self.workspace_id)' in src
    assert '.lt("confidence_score", confidence_below)' in src


def test_stage_5_runs_the_judge_after_image_validation_and_never_fails_the_job_on_it():
    src = _src(STAGE5)
    image_validation = src.index("process_validation_queue")
    judge = src.index("ProductFieldJudge(")
    assert judge > image_validation
    # A judge failure is recorded on the result, not raised through the stage.
    window = src[judge - 400: judge + 1200]
    assert "except Exception" in window
    assert '"field_judgements"' in src
