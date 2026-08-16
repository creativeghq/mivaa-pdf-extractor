"""
Guard: extracted values are checked against what the field registry already declares, and a
value the check rejects is FLAGGED, never dropped.

WHY THIS EXISTS
---------------
`material_metadata_fields` declares `field_type` for 387 fields and `dropdown_options` for 51 of
them, and `field_registry.priority_fields_prompt` renders those options into the extraction
prompt as `[one of: ...]`. Until `field_validation.py` landed, nothing on the PDF ingest path
checked the answer against them: the facet canonicalizer only sees the 25 `canonicalize=true`
fields and clusters synonyms rather than judging plausibility, and
`MetadataPrototypeValidator` — which does read `dropdown_options` — is not called from stage 4
at all. So `pei_rating: "PEI 7"` and `bowl_depth_mm: 4000` were written verbatim.

Neither is visible to any other guard in this repo. A wrong value is a well-formed value: the
type checker sees a `str`, the DB sees a `jsonb`, and every integrity probe sees a populated
field. That is the same shape as the money bug in CLAUDE.md, one layer down.

Two properties matter and both are tested here:

  1. The check actually rejects the values the registry says are illegal. A validator that
     accepts everything is the silent zero wearing a validator's coat — and this repo has
     already shipped one (`prototype_descriptions`, a column no row ever carried, which is why
     that validator did nothing for months while looking alive).
  2. A rejection never destroys data. The blast radius of a wrong `max` must be a noisy report,
     not a hole in the catalogue — and 47 of the 51 enum fields are not canonicalizable, so
     there is no `attributes_raw` to replay a dropped value from.

CI CONSTRAINT
-------------
CI installs pytest and NOTHING else (`deploy.yml`: `pip install pytest==7.4.3`). A third-party
import here makes the module uncollectable and takes the ENTIRE suite down, not just this file.
So: no `@pytest.mark.asyncio` (unregistered marker + `--strict-markers` = collection error;
coroutines are driven with `asyncio.run`), and the module under test is loaded BY PATH. That is
possible only because `field_validation` keeps every app import out of module scope — if that
ever regresses, `test_the_module_imports_with_no_dependencies` fails first and says why.
"""
import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _ROOT / "app" / "services" / "metadata" / "field_validation.py"


def _load_module():
    name = "_field_validation_under_test"
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations via
    # `sys.modules[cls.__module__].__dict__`, which is None for a module that is mid-exec and not
    # yet registered.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fv = _load_module()


class Spec:
    """Stand-in for FieldSpec carrying only what the validator reads."""

    def __init__(self, field_type="text", dropdown_options=(), validation_rules=None, name="f"):
        self.name = name
        self.field_type = field_type
        self.dropdown_options = dropdown_options
        self.validation_rules = validation_rules


# ── The module stays dependency-free ─────────────────────────────────────────────────────────

def test_the_module_imports_with_no_dependencies():
    """The load above already proves it. This states WHY, so a regression reads as intent.

    An `from app.services...` at module scope pulls in supabase_client and makes this file
    uncollectable in CI, which fails the whole suite rather than this test.
    """
    source = _MODULE_PATH.read_text(encoding="utf-8")
    offenders = [
        line for line in source.splitlines()
        if (line.startswith("from app.") or line.startswith("import app."))
    ]
    assert offenders == [], (
        "field_validation.py imports an app module at MODULE scope:\n  "
        + "\n  ".join(offenders)
        + "\n\nCI installs pytest and nothing else. Import it lazily inside the function that "
          "needs it, as `validate_metadata_against_registry` does."
    )


# ── 1. Enum membership: the 51 dropdown fields ───────────────────────────────────────────────

PEI = Spec("dropdown", ("PEI I", "PEI II", "PEI III", "PEI IV", "PEI V"), name="pei_rating")
SLIP = Spec("dropdown", ("R9", "R10", "R11", "R12", "R13"), name="slip_resistance")


@pytest.mark.parametrize("value", ["PEI I", "pei i", "PEI  I", "pei-i", "PeI_I"])
def test_enum_accepts_the_value_modulo_case_and_separators(value):
    """Catalogues write `R 11`, `R-11` and `R11` for the same rating."""
    verdict = fv.validate_value(PEI, value)
    assert verdict.status == fv.OK
    assert verdict.value == "PEI I", "should adopt the admin's curated spelling"


def test_enum_records_the_original_when_it_rewrites_the_spelling():
    verdict = fv.validate_value(SLIP, "r 11")
    assert verdict.status == fv.OK
    assert verdict.value == "R11"
    assert verdict.normalized_from == "r 11", "the original spelling must survive the rewrite"


@pytest.mark.parametrize("value", ["PEI 7", "PEI VI", "R99", "excellent", "5"])
def test_enum_flags_a_value_the_registry_does_not_offer(value):
    spec = PEI if value.startswith("PEI") else SLIP
    verdict = fv.validate_value(spec, value)
    assert verdict.status == fv.OUT_OF_ENUM
    assert verdict.value == value, "an out-of-enum value is KEPT, not dropped"


def test_an_enum_field_with_no_options_is_not_silently_enforced():
    """`field_type='dropdown'` with an empty `dropdown_options` constrains nothing.

    Treating it as "no value is legal" would reject every extraction for that field, which is
    the opposite failure and just as invisible.
    """
    verdict = fv.validate_value(Spec("dropdown", ()), "anything")
    assert verdict.status == fv.OK


# ── 2. Numbers: the 86 numeric fields ────────────────────────────────────────────────────────

DEPTH = Spec("number", validation_rules={"min": 0, "max": 1000}, name="bowl_depth_mm")


@pytest.mark.parametrize("value", [120, 120.5, "120", "120 mm", "120mm", "1,5", " 12 "])
def test_number_accepts_real_catalogue_spellings(value):
    assert fv.validate_value(DEPTH, value).status == fv.OK


@pytest.mark.parametrize("value,expected", [
    (4000, fv.OUT_OF_RANGE),      # the bug this was built for
    (-5, fv.OUT_OF_RANGE),
    ("4000 mm", fv.OUT_OF_RANGE),
    ("deep", fv.NOT_A_NUMBER),
    ("mm 120", fv.NOT_A_NUMBER),  # leading unit is malformed, not a formatting quirk
    ("", fv.NOT_A_NUMBER),
])
def test_number_flags_out_of_range_and_unparseable(value, expected):
    verdict = fv.validate_value(DEPTH, value)
    assert verdict.status == expected
    assert verdict.value == value, "a suspect number is KEPT, not dropped"


def test_a_boolean_is_not_the_number_one():
    """`bool` is an `int` subclass in Python. A True in a numeric field is a type confusion."""
    assert fv.validate_value(DEPTH, True).status == fv.NOT_A_NUMBER


def test_a_number_with_no_rules_is_still_checked_for_being_a_number():
    """The 3 numerics with no bound (unknown unit) must not become unchecked entirely."""
    assert fv.validate_value(Spec("number"), "wide").status == fv.NOT_A_NUMBER
    assert fv.validate_value(Spec("number"), "42").status == fv.OK


def test_min_only_leaves_the_ceiling_open():
    """The unit-unknown fields carry `{"min": 0}` and deliberately no max."""
    spec = Spec("number", validation_rules={"min": 0})
    assert fv.validate_value(spec, 10**9).status == fv.OK
    assert fv.validate_value(spec, -1).status == fv.OUT_OF_RANGE


# ── 3. Booleans ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (True, True), ("yes", True), ("Y", True), ("true", True), ("1", True),
    (False, False), ("no", False), ("false", False), ("0", False),
])
def test_boolean_coerces_the_usual_spellings(value, expected):
    verdict = fv.validate_value(Spec("boolean"), value)
    assert verdict.status == fv.OK
    assert verdict.value is expected


def test_boolean_flags_a_value_that_is_not_a_yes_or_no():
    verdict = fv.validate_value(Spec("boolean"), "sometimes")
    assert verdict.status == fv.NOT_A_BOOLEAN
    assert verdict.value == "sometimes"


# ── 4. validation_rules is a CLOSED set ──────────────────────────────────────────────────────

def test_an_unknown_rule_key_is_reported_not_ignored():
    """A rule that silently does nothing is worse than no rule — that is the whole thesis.

    `{"maximum": 10}` (instead of `max`) must not read as "no constraint".
    """
    verdict = fv.validate_value(Spec("number", validation_rules={"maximum": 10}), 999)
    assert verdict.status == fv.BAD_RULE
    assert "maximum" in verdict.detail


@pytest.mark.parametrize("rules", [
    {"pattern": "([unclosed"},
    {"min": "zero"},
    {"max_length": "long"},
])
def test_a_malformed_rule_is_reported_against_the_registry(rules):
    verdict = fv.validate_value(Spec("number", validation_rules=rules), 5)
    assert verdict.status == fv.BAD_RULE


def test_pattern_and_max_length_apply_to_text():
    """The three retyped composite fields (pallet_dimensions_cm, flush_volume_l,
    color_temperature_range_k) are `text` + a pattern."""
    spec = Spec("text", validation_rules={"pattern": r"^\s*\d+(?:[.,]\d+)?\s*(?:/\s*\d+(?:[.,]\d+)?\s*)?$"})
    assert fv.validate_value(spec, "4.5/3").status == fv.OK
    assert fv.validate_value(spec, "6").status == fv.OK
    assert fv.validate_value(spec, "four point five").status == fv.PATTERN_MISMATCH

    long_spec = Spec("text", validation_rules={"max_length": 5})
    assert fv.validate_value(long_spec, "abcdefgh").status == fv.TOO_LONG


def test_every_allowed_rule_key_is_actually_enforced():
    """A key in the allowlist that no branch reads is a rule that silently does nothing —
    the exact defect `bad_rule` exists to surface, hiding one level up in the evaluator."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    body = source.split("def _apply_rules", 1)[1].split("\ndef ", 1)[0]
    unenforced = [k for k in fv.ALLOWED_RULE_KEYS if f'"{k}"' not in body]
    # `unit` is metadata for a human reading the registry, not a constraint: the numeric parser
    # already ignores a trailing unit, and rejecting "120 cm" on a mm field would need a
    # conversion table this deliberately does not have.
    assert unenforced == ["unit"], (
        f"validation_rules keys in the allowlist that _apply_rules never reads: {unenforced}. "
        "Either enforce the key or take it out of ALLOWED_RULE_KEYS — leaving it in means an "
        "admin can write a rule that passes validation and constrains nothing."
    )


# ── 5. The invariant: nothing is ever dropped ────────────────────────────────────────────────

def test_no_verdict_ever_returns_an_empty_value():
    """Swept across every rejecting path at once, so a new status cannot quietly drop data."""
    cases = [
        (PEI, "PEI 7"), (SLIP, "R99"),
        (DEPTH, 4000), (DEPTH, "deep"), (DEPTH, -1),
        (Spec("boolean"), "sometimes"),
        (Spec("number", validation_rules={"maximum": 1}), 5),
        (Spec("text", validation_rules={"max_length": 2}), "abcdef"),
        (Spec("text", validation_rules={"pattern": "^z"}), "abc"),
    ]
    seen = set()
    for spec, value in cases:
        verdict = fv.validate_value(spec, value)
        seen.add(verdict.status)
        assert verdict.is_suspect, f"{value!r} should be suspect for {spec.field_type}"
        assert verdict.value == value, (
            f"{verdict.status} DROPPED {value!r}. An implausible value is flagged and kept: a "
            "wrong rule must not be able to destroy extracted data, and most enum fields have "
            "no attributes_raw to replay from."
        )
    assert seen == fv.SUSPECT - {fv.NOT_A_NUMBER} | {fv.NOT_A_NUMBER}, (
        f"not every suspect status is reachable from these cases: missing {fv.SUSPECT - seen}. "
        "An unreachable marker is a check nobody performs."
    )


def test_a_list_keeps_every_element_and_takes_the_worst_verdict():
    kept, verdict = fv._validate_one(SLIP, ["R10", "R99", "R12"])
    assert kept == ["R10", "R99", "R12"], "no element may be dropped"
    assert verdict.status == fv.OUT_OF_ENUM


def test_empty_values_are_not_failures():
    """'The extractor found nothing' is not 'the extractor found something wrong'."""
    for blank in (None, "", "  ", "N/A", "not specified"):
        assert fv._is_empty(blank), f"{blank!r} should read as empty"


# ── 6. The whole-dict pass, including the silent-zero counter ────────────────────────────────

def _run_with_registry(metadata, specs):
    """Drive the async entry point with a stubbed registry module.

    The stub is injected into sys.modules because the real one is imported lazily INSIDE the
    function — which is what keeps this file importable in a pytest-only CI.
    """
    pkg_names = ["app", "app.services", "app.services.metadata"]
    saved = {n: sys.modules.get(n) for n in pkg_names + ["app.services.metadata.field_registry"]}
    try:
        for name in pkg_names:
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
        stub = types.ModuleType("app.services.metadata.field_registry")

        class _Registry:
            @staticmethod
            async def ensure_loaded(**_kw):
                return None

            @staticmethod
            def spec_for(key):
                return specs.get(key)

        stub.field_registry = _Registry()
        sys.modules["app.services.metadata.field_registry"] = stub
        return asyncio.run(fv.validate_metadata_against_registry(metadata))
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_it_walks_section_containers_with_the_dotted_path():
    """`_extraction_metadata` already keys nested fields as `packaging.pieces_per_box`."""
    metadata = {"pei_rating": "PEI 7", "packaging": {"patterns_count": 5000}}
    specs = {
        "pei_rating": PEI,
        "patterns_count": Spec("number", validation_rules={"min": 0, "max": 200}),
    }
    out, report = _run_with_registry(metadata, specs)

    assert report.suspect == {
        "pei_rating": fv.OUT_OF_ENUM,
        "packaging.patterns_count": fv.OUT_OF_RANGE,
    }
    assert out["pei_rating"] == "PEI 7"
    assert out["packaging"]["patterns_count"] == 5000
    assert out["_extraction_metadata"]["packaging.patterns_count"]["validation"]["status"] == \
        fv.OUT_OF_RANGE


def test_an_unknown_key_counts_as_unchecked_not_as_invalid():
    """The extractor emits keys the registry never claimed to describe (`material_category`,
    `unit`, `factory_name`). Calling those wrong would bury the real findings."""
    out, report = _run_with_registry({"factory_name": "Marazzi", "unit": "m2"}, {})
    assert report.suspect == {}
    assert report.checked == 0
    assert report.unchecked == 2
    assert out["factory_name"] == "Marazzi"


def test_the_report_distinguishes_clean_from_never_ran():
    """`checked: 34, suspect: {}` and "the validator never ran" must not look identical.

    This is the platform's dominant historical failure — a metric sitting at zero forever with
    nothing complaining — so the count is recorded even when everything passed.
    """
    out, report = _run_with_registry({"pei_rating": "PEI IV"}, {"pei_rating": PEI})
    assert report.checked == 1
    assert report.suspect == {}
    assert out["_validation"] == {"checked": 1, "unchecked": 0}


def test_existing_extraction_provenance_is_merged_not_replaced():
    """Stage 4 already writes `{source, confidence}` per field; the verdict joins it."""
    metadata = {
        "pei_rating": "PEI 7",
        "_extraction_metadata": {"pei_rating": {"source": "chunk_regex", "confidence": 0.9}},
    }
    out, _ = _run_with_registry(metadata, {"pei_rating": PEI})
    entry = out["_extraction_metadata"]["pei_rating"]
    assert entry["source"] == "chunk_regex", "existing provenance must survive"
    assert entry["confidence"] == 0.9
    assert entry["validation"]["status"] == fv.OUT_OF_ENUM
