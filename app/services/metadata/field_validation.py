"""
Plausibility validation for extracted metadata, driven by the field registry.

WHY THIS EXISTS
---------------
`material_metadata_fields` already declares, for 387 active fields, what each value is
supposed to BE: `field_type` (number / dropdown / boolean / text) and, for the 51 dropdown
fields, the exact `dropdown_options` an admin curated. `field_registry.priority_fields_prompt`
renders those options into the extraction prompt as `[one of: ...]`.

Nothing checked the answer. The PDF ingest path (stage 4) called the facet canonicalizer —
which only sees the 25 `canonicalize=true` fields, and whose job is clustering synonyms, not
judging plausibility — and wrote the result straight to `products.metadata`. So the prompt
offered an enum and no validator enforced one: `pei_rating` could come back `PEI 7` (the scale
stops at V), `slip_resistance` `R99`, `bowl_depth_mm` `4000`. Every one of those is a valid
string or a valid number, so neither the type checker nor any integrity probe could see it —
the same shape as the money bug in CLAUDE.md: a wrong number is still a `number`.

This module closes that. It is the validator side of the asymmetry `#347` set out to kill,
which until now existed in its purest form: the prompt offered the vocabulary, and there was
no validator at all.

WHAT IT CHECKS, IN ORDER OF WHAT THE REGISTRY ALREADY KNOWS
-----------------------------------------------------------
1. `dropdown` + `dropdown_options` — membership. 51 fields, no new data required.
2. `number` — parses as a number, then `min`/`max` from `validation_rules` if present.
3. `boolean` — parses as a boolean.
4. `validation_rules` — the escape hatch for what the columns cannot express
   (`min`, `max`, `unit`, `pattern`, `max_length`). Small and CLOSED on purpose: an
   unrecognised key is reported, never ignored, because a rule that silently does nothing is
   worse than no rule.

FLAG AND KEEP — NEVER DROP
--------------------------
An implausible value is recorded and RETAINED, never deleted. Two reasons. A bad rule must not
be able to destroy extracted data — the blast radius of a wrong `max` is then a noisy report,
not a silent hole in the catalogue. And for the non-canonicalizable fields there is no
`attributes_raw` to replay from, so a dropped value is gone for good.

The one rewrite this does perform is spelling: a value that matches an option modulo case and
separators (`"MATT"`, `"R 11"`) is rewritten to the option's canonical spelling, with the
original preserved under `normalized_from`. Nothing is lost.

COVERAGE IS REPORTED, NOT ASSUMED
---------------------------------
`ValidationReport.checked` / `.unchecked` exist so that "the validator ran and validated
nothing" is visible. A validator that silently checks zero fields is exactly the silent-zero
shape this platform keeps rediscovering — `MetadataPrototypeValidator.load_prototypes` already
warns about its own version of it. Emptiness alone is ambiguous; the counts are not.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.metadata.field_registry import FieldSpec

# NOTE: `field_registry` is imported lazily inside `validate_metadata_against_registry`, not at
# module scope. It reaches `app.services.core.supabase_client`, and CI installs pytest and
# nothing else (see tests/unit/test_workspace_resolution.py) — a module-scope import here would
# make this file uncollectable and take the whole suite down with it. Everything above that
# function is pure and importable with no dependencies at all, which is what the unit tests load.

logger = logging.getLogger(__name__)

# ── Verdict vocabulary ───────────────────────────────────────────────────────────────────────
# Explicit markers, per pipeline convention 1: a consumer checks the marker. "No verdict" and
# "checked and fine" must not be the same value.
OK = "ok"
OUT_OF_ENUM = "out_of_enum"
OUT_OF_RANGE = "out_of_range"
NOT_A_NUMBER = "not_a_number"
NOT_A_BOOLEAN = "not_a_boolean"
PATTERN_MISMATCH = "pattern_mismatch"
TOO_LONG = "too_long"
BAD_RULE = "bad_rule"

#: Every status that means "this value is suspect". BAD_RULE is in here deliberately: a rule the
#: evaluator cannot understand is a defect in the registry, and staying quiet about it would
#: reproduce the failure this module exists to fix, one level up.
SUSPECT = frozenset({
    OUT_OF_ENUM, OUT_OF_RANGE, NOT_A_NUMBER, NOT_A_BOOLEAN,
    PATTERN_MISMATCH, TOO_LONG, BAD_RULE,
})

#: The closed set of keys `material_metadata_fields.validation_rules` may carry. Adding one means
#: teaching `_apply_rules` to enforce it — an unknown key yields BAD_RULE rather than passing.
ALLOWED_RULE_KEYS = frozenset({"min", "max", "unit", "pattern", "max_length"})

#: Values that mean "the extractor found nothing here". Not a validation failure.
_EMPTY = {"", "n/a", "na", "none", "null", "-", "—", "unknown", "not specified"}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
#: Leading numeric token, optionally signed/decimal, with an optional trailing unit we ignore
#: for the numeric check ("12 mm", "1.5kg"). A value with a LEADING unit ("mm 12") does not
#: match and is reported — that is a genuinely malformed answer, not a formatting quirk.
_LEADING_NUMBER_RE = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*([a-zA-Z%°/²³]*)\s*$")

_TRUE_TOKENS = {"true", "yes", "y", "1", "present", "included"}
_FALSE_TOKENS = {"false", "no", "n", "0", "absent", "not included"}


def _enum_key(value: Any) -> str:
    """Comparison key for enum membership: lowercase, alphanumerics only.

    Deliberately tighter than `facet_canonicalizer.normalize_string`, which collapses separators
    to a SPACE. That would leave `"R 11"` and `"R11"` different keys, and catalogues write slip
    and IP ratings both ways. Here both reduce to `r11`.
    """
    return _NON_ALNUM_RE.sub("", str(value).strip().lower())


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _EMPTY
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _coerce_number(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """Returns (number, unit_seen). `(None, None)` when the value is not numeric."""
    if isinstance(value, bool):
        # bool is an int subclass in Python; a boolean in a numeric field is a type confusion,
        # not the number 1.
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None
    match = _LEADING_NUMBER_RE.match(str(value))
    if not match:
        return None, None
    raw, unit = match.group(1), (match.group(2) or "").strip()
    try:
        return float(raw.replace(",", ".")), (unit or None)
    except ValueError:  # pragma: no cover - the regex already constrains this
        return None, None


def _coerce_boolean(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


@dataclass(frozen=True)
class Verdict:
    """The outcome for ONE value. `value` is what should be kept — never None, never dropped."""
    status: str
    value: Any
    detail: str = ""
    #: Set only when the value was rewritten to an option's canonical spelling.
    normalized_from: Optional[str] = None

    @property
    def is_suspect(self) -> bool:
        return self.status in SUSPECT

    def as_record(self) -> Dict[str, Any]:
        """The shape stamped into `_extraction_metadata[field]['validation']`."""
        record: Dict[str, Any] = {"status": self.status}
        if self.detail:
            record["detail"] = self.detail
        if self.normalized_from is not None:
            record["normalized_from"] = self.normalized_from
        return record


@dataclass
class ValidationReport:
    """Per-product summary. The counts are the point — see the module docstring."""
    checked: int = 0
    unchecked: int = 0
    suspect: Dict[str, str] = dc_field(default_factory=dict)     # "field" -> status
    normalized: Dict[str, str] = dc_field(default_factory=dict)  # "field" -> original spelling

    @property
    def suspect_count(self) -> int:
        return len(self.suspect)

    def as_metadata_block(self) -> Dict[str, Any]:
        """Written to `metadata['_validation']` so a probe can read it off the row.

        Recorded even when everything passed: "0 suspect out of 34 checked" and "the validator
        never ran" have to be distinguishable, and only one of them has a `checked` count.
        """
        block: Dict[str, Any] = {"checked": self.checked, "unchecked": self.unchecked}
        if self.suspect:
            block["suspect"] = dict(sorted(self.suspect.items()))
        if self.normalized:
            block["normalized"] = dict(sorted(self.normalized.items()))
        return block


def _apply_rules(
    spec: "FieldSpec",
    value: Any,
    number: Optional[float],
    text: str,
) -> Optional[Verdict]:
    """Enforce `validation_rules`. Returns None when nothing objects.

    Every verdict carries `value`, the ORIGINAL, never `text`. Judging a value must not
    change its type: returning the stringified form turned a rejected `4000` into `"4000"`,
    so a rule firing silently retyped the field it flagged.
    """
    rules = spec.validation_rules or {}
    if not isinstance(rules, dict):
        return Verdict(BAD_RULE, value, f"validation_rules is {type(rules).__name__}, expected object")

    unknown = sorted(set(rules) - ALLOWED_RULE_KEYS)
    if unknown:
        # Loud, not ignored: an unrecognised key is a rule that would otherwise do nothing while
        # looking like protection.
        return Verdict(BAD_RULE, value, f"unknown validation_rules keys: {', '.join(unknown)}")

    if number is not None:
        low, high = rules.get("min"), rules.get("max")
        try:
            if low is not None and number < float(low):
                return Verdict(OUT_OF_RANGE, value, f"{number:g} < min {low}")
            if high is not None and number > float(high):
                return Verdict(OUT_OF_RANGE, value, f"{number:g} > max {high}")
        except (TypeError, ValueError):
            return Verdict(BAD_RULE, value, f"min/max are not numeric: min={low!r} max={high!r}")

    pattern = rules.get("pattern")
    if pattern:
        try:
            if not re.match(str(pattern), text):
                return Verdict(PATTERN_MISMATCH, value, f"does not match {pattern}")
        except re.error as exc:
            return Verdict(BAD_RULE, value, f"invalid regex {pattern!r}: {exc}")

    max_length = rules.get("max_length")
    if max_length is not None:
        try:
            if len(text) > int(max_length):
                return Verdict(TOO_LONG, value, f"{len(text)} chars > max_length {max_length}")
        except (TypeError, ValueError):
            return Verdict(BAD_RULE, value, f"max_length is not an integer: {max_length!r}")

    return None


def validate_value(spec: "FieldSpec", value: Any) -> Verdict:
    """Validate one scalar against its registry spec. Pure; no I/O."""
    text = str(value).strip()

    if spec.field_type == "dropdown" and spec.dropdown_options:
        wanted = {_enum_key(o): o for o in spec.dropdown_options if str(o).strip()}
        hit = wanted.get(_enum_key(value))
        if hit is None:
            return Verdict(
                OUT_OF_ENUM, value,
                f"not one of: {', '.join(str(o) for o in spec.dropdown_options)}",
            )
        # Match modulo case/separators — adopt the curated spelling, keep the original.
        if str(hit) != text:
            return Verdict(OK, hit, normalized_from=text)
        return Verdict(OK, hit)

    if spec.field_type == "number":
        number, _unit = _coerce_number(value)
        if number is None:
            return Verdict(NOT_A_NUMBER, value, f"{text!r} does not parse as a number")
        return _apply_rules(spec, value, number, text) or Verdict(OK, value)

    if spec.field_type == "boolean":
        as_bool = _coerce_boolean(value)
        if as_bool is None:
            return Verdict(NOT_A_BOOLEAN, value, f"{text!r} is not a yes/no value")
        return Verdict(OK, as_bool, normalized_from=text if not isinstance(value, bool) else None)

    # text: only the explicit rules apply.
    return _apply_rules(spec, value, None, text) or Verdict(OK, value)


def _validate_one(spec: "FieldSpec", value: Any) -> Tuple[Any, Verdict]:
    """Handle the scalar-or-list shape. A list's verdict is its worst element's."""
    if not isinstance(value, list):
        verdict = validate_value(spec, value)
        return verdict.value, verdict

    kept: List[Any] = []
    worst: Optional[Verdict] = None
    for item in value:
        if _is_empty(item):
            kept.append(item)
            continue
        verdict = validate_value(spec, item)
        kept.append(verdict.value)
        if verdict.is_suspect and worst is None:
            worst = verdict
    if worst is not None:
        return kept, Verdict(worst.status, kept, worst.detail)
    return kept, Verdict(OK, kept)


async def validate_metadata_against_registry(
    metadata: Dict[str, Any],
    *,
    stamp: bool = True,
) -> Tuple[Dict[str, Any], ValidationReport]:
    """Validate an extracted metadata dict against the field registry.

    Walks the top level plus one level of section containers (`packaging.pieces_per_box`), the
    same dotted convention `_extraction_metadata` already uses in stage 4.

    Returns `(metadata, report)`. The dict is returned with canonical enum spellings applied and
    NOTHING removed. When `stamp` is set, each suspect field also gets a `validation` record
    merged into its `_extraction_metadata` entry, and the summary lands on `metadata['_validation']`.

    Unknown keys are counted as `unchecked` rather than flagged: the registry does not claim to
    describe every key the extractor emits (`material_category`, `unit`, `factory_name`), and
    treating "I have no opinion" as "this is wrong" would bury the real findings.
    """
    from app.services.metadata.field_registry import field_registry

    await field_registry.ensure_loaded()

    report = ValidationReport()
    extraction_meta = dict(metadata.get("_extraction_metadata") or {})

    def visit(container: Optional[str], holder: Dict[str, Any], lookup: Callable) -> None:
        for key, value in list(holder.items()):
            if not key or key.startswith("_"):
                continue
            path = f"{container}.{key}" if container else key

            spec = lookup(key)
            if spec is None:
                # A nested section (`packaging`, `material_properties`) rather than a field.
                if container is None and isinstance(value, dict):
                    visit(key, value, lookup)
                else:
                    report.unchecked += 1
                continue

            if _is_empty(value):
                continue

            kept, verdict = _validate_one(spec, value)
            holder[key] = kept
            report.checked += 1

            if verdict.normalized_from is not None:
                report.normalized[path] = verdict.normalized_from
            if verdict.is_suspect:
                report.suspect[path] = verdict.status
            if stamp and (verdict.is_suspect or verdict.normalized_from is not None):
                entry = dict(extraction_meta.get(path) or {})
                entry["validation"] = verdict.as_record()
                extraction_meta[path] = entry

    visit(None, metadata, field_registry.spec_for)

    if stamp:
        if extraction_meta:
            metadata["_extraction_metadata"] = extraction_meta
        metadata["_validation"] = report.as_metadata_block()

    if report.checked == 0:
        # The silent zero, caught at the source. A product whose every field is unknown to the
        # registry is possible; a whole catalogue of them means the registry and the extractor
        # have drifted apart, and nothing downstream would say so.
        logger.warning(
            "Registry validation checked 0 of %d metadata keys — no key matched a registered "
            "field. Extractor output and material_metadata_fields have drifted.",
            report.unchecked,
        )
    elif report.suspect:
        logger.warning(
            "⚠️  %d of %d validated fields are implausible (kept, not dropped): %s",
            report.suspect_count, report.checked,
            ", ".join(f"{k}={v}" for k, v in sorted(report.suspect.items())),
        )

    return metadata, report
