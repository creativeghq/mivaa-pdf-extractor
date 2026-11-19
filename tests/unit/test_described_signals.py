"""Described signals (#499): spec cleaning, the evidence check, and the claim → charge → judge order.

Loads the rules module by path (CI installs pytest only); the order check reads the service source
with comments blanked, so a comment can never satisfy it.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = _ROOT / "app" / "services" / "integrations" / "described_signals_service.py"


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rules = _load("app.services.integrations.described_signals_rules",
              "app/services/integrations/described_signals_rules.py")
budget = _load("comment_budget_for_described", "scripts/comment_budget.py")


def test_clean_spec_keeps_known_sources_and_bounds_everything():
    spec = rules.clean_spec({
        "sources": ["job_posting", "tenders", "job_posting", "mention"],
        "keywords": [f"kw{i}" for i in range(60)] + ["KW1"],
        "must_not": ["recruiter"],
        "judge_question": "Does this show a new showroom? " * 40,
        "positive_examples": ["a", "b"],
        "negative_examples": [],
        "min_similarity": 3,
    })
    assert spec["sources"] == ["job_posting", "mention"]
    assert len(spec["keywords"]) == 40
    assert len(spec["judge_question"]) == 600
    assert spec["min_similarity"] == 0.9


@pytest.mark.parametrize("raw", [
    {"sources": ["tenders"], "judge_question": "Q?"},
    {"sources": ["job_posting"], "judge_question": "   "},
])
def test_clean_spec_refuses_rules_with_no_source_or_question(raw):
    with pytest.raises(ValueError):
        rules.clean_spec(raw)


def test_a_bad_similarity_falls_back_to_the_default():
    spec = rules.clean_spec({"sources": ["technology"], "judge_question": "Q?", "min_similarity": "high"})
    assert spec["min_similarity"] == 0.45


def test_the_quote_must_be_a_passage_of_the_item():
    item = "Title: Showroom Manager\n\nWe are opening our NEW   showroom in Athens next spring."
    assert rules.quote_is_in_item("opening our new showroom in Athens", item)
    assert not rules.quote_is_in_item("opening a flagship store in Thessaloniki", item)
    assert not rules.quote_is_in_item("new", item)


def test_the_quote_check_tolerates_typography_and_elisions_but_not_invention():
    item = "Duties — “Upload pick tickets to Salesforce” and related work orders, daily."
    assert rules.quote_is_in_item('"Upload pick tickets to Salesforce" and related work orders.', item)
    assert rules.quote_is_in_item("Duties - Upload pick tickets ... work orders", item)
    assert not rules.quote_is_in_item("work orders ... Upload pick tickets", item)
    assert not rules.quote_is_in_item("Upload pick tickets to HubSpot", item)


def test_an_elided_quote_cannot_be_stitched_from_scattered_words():
    item = "We are hiring a new store manager. Our showroom team is growing. Based in Greece, travel required."
    assert not rules.quote_is_in_item("new ... showroom ... Greece", item)
    assert not rules.quote_is_in_item("hiring a new ... showroom team ... Based in Greece ... travel required", item)
    assert rules.quote_is_in_item("hiring a new store manager ... showroom team is growing", item)


def test_schemas_are_closed_and_require_every_field():
    for schema in (rules.COMPILE_SCHEMA, rules.JUDGE_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    assert rules.COMPILE_SCHEMA["properties"]["sources"]["items"]["enum"] == list(rules.SOURCES)


def _run_one_body() -> str:
    src = budget.blank_comments(_SERVICE.read_text(encoding="utf-8"))
    start = src.index("async def _judge_claimed(")
    end = src.index("\nasync def ", start + 10)
    return src[start:end]


def test_a_judgement_is_claimed_before_the_charge_and_charged_before_the_model_call():
    body = _run_one_body()
    claim = body.index('"claim_described_judgement"')
    charge = body.index("await _charge(")
    judge = body.index("await judge_item(")
    assert claim < charge < judge


def test_a_failed_check_is_refunded_what_the_ledger_took():
    body = _run_one_body()
    failed = body.index('verdict["status"] == "failed"')
    assert body.index("await _refund(ws, paid,", failed) > failed


def test_a_reclaimed_judgement_that_was_already_paid_is_not_charged_again():
    body = _run_one_body()
    assert body.index('claim.get("prepaid")') < body.index("if paid <= 0:") < body.index("await _charge(")
    assert body.index("await _charge(") < body.index('"set_described_judgement_charge"') < body.index("await judge_item(")


def test_no_prompt_text_lives_in_the_service():
    src = budget.blank_comments(_SERVICE.read_text(encoding="utf-8"))
    assert 'load_prompt("classification", "signals_describe_judge"' in src
    assert 'load_prompt("classification", "signals_describe_compile"' in src
    assert not re.search(r"You are a careful analyst|Answer the question for this item", src)
