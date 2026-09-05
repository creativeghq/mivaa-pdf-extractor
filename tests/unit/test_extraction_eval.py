"""Guards for `app/evaluation/extraction_eval.py` — the rules that keep a comparison honest.

Loaded by path inside a fixture: CI installs pytest alone and the module is pure
stdlib. Every case below encodes one of the wrong numbers the GAIK extraction-
evaluation reference records its authors publishing first (adopted 2026-09-05).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "evaluation" / "extraction_eval.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("_extraction_eval", MODULE)
    mod = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: the module uses dataclasses under `from __future__ import
    # annotations`, and dataclasses resolves those strings through sys.modules[cls.__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_is_stdlib_only():
    src = MODULE.read_text(encoding="utf-8")
    for banned in ("from app", "import app", "supabase", "httpx", "pydantic"):
        assert banned not in src, f"{banned!r} would make the module unloadable in CI"


# ── values ────────────────────────────────────────────────────────────────────

def test_numbers_agree_as_values_across_the_ways_a_catalogue_writes_them(ev):
    assert ev.values_match(1000, "1000.0")
    assert ev.values_match("1.000,00", 1000)
    assert ev.values_match("1,234.56", "1234,56")
    assert ev.values_match("3,5", 3.5)
    assert ev.values_match("60 cm", 60)
    assert not ev.values_match("60x60", 60)  # a dimension pair is not one number
    assert not ev.values_match("1000", "1001")


def test_text_agrees_after_case_width_and_whitespace(ev):
    assert ev.values_match("Beige", "  beige ")
    assert ev.values_match("Μπεζ", "μπεζ")
    assert ev.values_match(["a", "b"], ["B", "A"])
    assert not ev.values_match("beige", "grey")


def test_empty_expected_and_empty_extracted_is_a_correct_absence(ev):
    assert ev.values_match(None, "")
    assert ev.values_match("", [])


# ── compare + metrics ─────────────────────────────────────────────────────────

def test_wrong_value_counts_on_both_sides_and_missing_counts_once(ev):
    v = ev.compare({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "9"})
    m = ev.metrics(v)
    assert (m.tp, m.fp, m.fn) == (1, 1, 2)
    assert m.precision == 0.5
    assert round(m.recall, 4) == round(1 / 3, 4)


def test_a_field_expected_empty_but_extracted_is_a_hallucination(ev):
    v = ev.compare({"unit": None}, {"unit": "KG"})
    m = ev.metrics(v)
    assert m.hallucinated == 1 and m.fp == 1 and m.tn == 0
    # And the rule for the golden-case author is written down: leave such fields out.
    assert "does not record scores a correct" in MODULE.read_text(encoding="utf-8")


def test_extra_extracted_fields_are_not_punished_unless_strict(ev):
    lenient = ev.metrics(ev.compare({"a": "1"}, {"a": "1", "colour": "beige"}))
    strict = ev.metrics(ev.compare({"a": "1"}, {"a": "1", "colour": "beige"}, strict=True))
    assert lenient.hallucinated == 0 and lenient.f1 == 1.0
    assert strict.hallucinated == 1 and strict.precision == 0.5


def test_no_extraction_scores_zero_and_stays_in_the_sample(ev):
    m = ev.metrics(ev.compare({"a": "1", "b": "2"}, None))
    assert m.n_fields == 2 and m.tp == 0 and m.fn == 2
    assert m.recall == 0.0
    assert m.precision is None  # nothing was extracted: unknown, not 0


def test_micro_aggregate_sums_counts_rather_than_averaging_averages(ev):
    # Case A: 1 of 1 right. Case B: 0 of 9 right. Per-case average would say 50%.
    a = ev.metrics(ev.compare({"x": "1"}, {"x": "1"}))
    b = ev.metrics(ev.compare({f"f{i}": "1" for i in range(9)}, {}))
    total = ev.micro_aggregate([a, b])
    assert total.tp == 1 and total.n_expected == 10
    assert total.recall == 0.1


def test_exact_match_rate_counts_correct_absences(ev):
    v = ev.compare({"a": "1", "b": None}, {"a": "1", "b": ""})
    m = ev.metrics(v)
    assert m.exact_match_rate == 1.0 and m.tn == 1


# ── run failure classes ───────────────────────────────────────────────────────

def test_failures_are_three_different_findings(ev):
    assert ev.classify_run_failure(product_found=False, extracted=None) == "no_product"
    assert ev.classify_run_failure(product_found=True, extracted={"a": ""}) == "no_extraction"
    assert ev.classify_run_failure(product_found=True, extracted={"a": "1"}, pipeline_ok=False) == "pipeline_failed"
    assert ev.classify_run_failure(product_found=True, extracted={"a": "1"}) == "none"
    assert set(ev.RUN_FAILURE_CLASSES) == {"none", "no_product", "no_extraction", "pipeline_failed"}


# ── agreement across repeats ──────────────────────────────────────────────────

def test_cell_and_byte_agreement_say_different_things(ev):
    runs = [{"total": 1000}, {"total": "1000"}, {"total": "1000.0"}, {"total": 1000}, {"total": "1000"}]
    ag = ev.agreement(runs)
    assert ag.cell["total"] == 1.0
    assert ag.byte["total"] == 0.4  # "1000" twice is the modal byte string


def test_a_run_that_produced_nothing_stays_in_the_agreement_denominator(ev):
    ag = ev.agreement([{"a": "x"}, {"a": "x"}, None, {"a": "x"}])
    assert ag.runs == 4
    assert ag.cell["a"] == 0.75
    assert ag.completeness["a"] == 0.75


def test_stability_alone_rewards_silence_so_completeness_is_reported_beside_it(ev):
    silent = ev.agreement([{"a": ""}, {"a": None}, {}])
    assert silent.cell["a"] == 1.0  # perfectly repeatable
    assert silent.completeness["a"] == 0.0  # and that is why agreement is never read alone


def test_agreement_is_addressed_by_field_name_not_position(ev):
    ag = ev.agreement([{"a": "1", "b": "2"}, {"b": "2", "a": "1"}], fields=["a", "b"])
    assert ag.cell == {"a": 1.0, "b": 1.0}


# ── product flattening ────────────────────────────────────────────────────────

def test_flatten_reads_gold_attributes_and_scalar_columns_only(ev):
    row = {
        "name": "Beige tile", "sku": "T-1", "attributes": {"colour": "beige", "size_cm": "60x60"},
        "attributes_raw": {"colour": "Beige-ish"}, "metadata": {"page_number": 4}, "cost": 12.5,
    }
    flat = ev.flatten_product_fields(row)
    assert flat == {"name": "Beige tile", "sku": "T-1", "colour": "beige", "size_cm": "60x60"}
    assert "cost" not in flat and "page_number" not in flat
