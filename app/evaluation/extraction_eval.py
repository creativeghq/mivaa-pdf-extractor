"""Field-level extraction evaluation: precision, recall, F1, hallucination, agreement.

Adopted 2026-09-05 from the GAIK toolkit's `ExtractionEvaluator` and its
`measuring-extraction` reference, which records the wrong numbers its authors
published first. The rules that keep a comparison honest, each encoded below:

* **The denominator does not move.** A metric is computed over every (case, field)
  pair with a non-empty expectation, whether or not anything happened. Averaging a
  loss rate only over the documents that lost something makes two runs incomparable.
* **A failed case scores zero and stays in the sample.** Dropping it lets a pipeline
  raise its own average by crashing on the hardest inputs. A case whose product was
  never created, or whose extraction is empty, is a row with a failure class, not a
  missing row.
* **Failure is a result, not a missing row.** `no_product`, `no_extraction` and
  `pipeline_failed` all score zero and are three different findings.
* **Cell agreement and byte agreement say different things.** One repeat returning
  ``1000.0`` where four returned ``1000`` is cell agreement 1.0 and byte agreement
  0.6. Both are computed; the first is about quality, the second about whether
  outputs can be diffed.
* **Stability alone rewards silence.** A configuration that leaves fields empty is
  perfectly repeatable. Agreement is reported beside completeness, never alone.
* **Only fields the page actually carries belong in the expectation.** An expected
  ``null`` for a field the source system does not record scores a correct ``"KG"``
  as a mistake. That is a rule for whoever writes the golden case; `strict=False`
  (the default) does not punish extra extracted fields, because a golden case here
  is usually a PARTIAL statement of what is on the page.

Pure stdlib. Loaded by path in `tests/unit/test_extraction_eval.py`.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

#: How a case-run failed before any field could be scored. `none` means it was scored.
RUN_FAILURE_CLASSES = ("none", "no_product", "no_extraction", "pipeline_failed")

_EMPTY = (None, "", [], {})

_WS = re.compile(r"\s+")
_NUMBER_JUNK = re.compile(r"[^\d,.\-]")


def is_empty(value: Any) -> bool:
    """None, empty string, empty list, empty dict — the ways "nothing" is written."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def parse_number(value: Any) -> Optional[float]:
    """A number written any of the ways a catalogue writes it, or None.

    ``1.234,56`` (Greek/German), ``1,234.56`` (English), ``1000``, ``1000,00``,
    ``60 cm`` are all numbers; ``60x60`` is not (two of them). Units and currency
    symbols are stripped; the LAST separator present is taken as the decimal one.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # Two numbers glued by x / × / - are a dimension pair, not one number.
    if re.search(r"\d\s*[x×X]\s*\d", s):
        return None
    cleaned = _NUMBER_JUNK.sub("", s)
    if not re.search(r"\d", cleaned):
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        head, _, tail = cleaned.rpartition(",")
        # "1,234" is a thousand; "1000,00" / "3,5" is a decimal.
        cleaned = head + tail if len(tail) == 3 and head else head + "." + tail
    try:
        n = float(cleaned)
    except ValueError:
        return None
    return n if math.isfinite(n) else None


def normalize_text(value: Any) -> str:
    """The comparison form of a scalar: NFKC, casefolded, whitespace collapsed."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        n = float(value)
        return str(int(n)) if n.is_integer() else repr(n)
    if isinstance(value, (list, tuple, set)):
        return "|".join(sorted(normalize_text(v) for v in value))
    if isinstance(value, dict):
        return json.dumps({k: normalize_text(v) for k, v in value.items()}, sort_keys=True, ensure_ascii=False)
    s = unicodedata.normalize("NFKC", str(value)).casefold()
    return _WS.sub(" ", s).strip()


def values_match(expected: Any, extracted: Any) -> bool:
    """Equal as VALUES: ``1000`` == ``"1000.0"`` == ``"1.000,00"``; ``"Beige"`` == ``" beige "``."""
    if is_empty(expected) and is_empty(extracted):
        return True
    if is_empty(expected) != is_empty(extracted):
        return False
    en, xn = parse_number(expected), parse_number(extracted)
    if en is not None and xn is not None:
        return math.isclose(en, xn, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, (list, tuple, set)) or isinstance(extracted, (list, tuple, set)):
        e_items = expected if isinstance(expected, (list, tuple, set)) else [expected]
        x_items = extracted if isinstance(extracted, (list, tuple, set)) else [extracted]
        return sorted(normalize_text(v) for v in e_items) == sorted(normalize_text(v) for v in x_items)
    return normalize_text(expected) == normalize_text(extracted)


@dataclass
class FieldVerdict:
    field: str
    expected: Any
    extracted: Any
    matched: bool
    #: expected non-empty, extracted empty or absent
    missing: bool = False
    #: extracted non-empty where nothing was expected (strict mode only)
    hallucinated: bool = False
    #: both empty — a correct absence
    both_empty: bool = False
    reason: str = ""


@dataclass
class Metrics:
    """Precision / recall / F1 / hallucination / exact-match, with the counts behind them.

    TP: expected non-empty and matched.  FP: extracted non-empty and wrong, or
    hallucinated.  FN: expected non-empty and missing or wrong (a wrong value is
    both an FP and an FN).  TN: both empty.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    hallucinated: int = 0
    n_expected: int = 0
    n_extracted: int = 0
    n_fields: int = 0

    @property
    def precision(self) -> Optional[float]:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def hallucination_rate(self) -> Optional[float]:
        return self.hallucinated / self.n_extracted if self.n_extracted else None

    @property
    def exact_match_rate(self) -> Optional[float]:
        return (self.tp + self.tn) / self.n_fields if self.n_fields else None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(
            precision=_round(self.precision),
            recall=_round(self.recall),
            f1=_round(self.f1),
            hallucination_rate=_round(self.hallucination_rate),
            exact_match_rate=_round(self.exact_match_rate),
        )
        return d

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn, tn=self.tn + other.tn,
            hallucinated=self.hallucinated + other.hallucinated,
            n_expected=self.n_expected + other.n_expected,
            n_extracted=self.n_extracted + other.n_extracted,
            n_fields=self.n_fields + other.n_fields,
        )


def _round(v: Optional[float], places: int = 4) -> Optional[float]:
    return None if v is None else round(v, places)


def compare(expected: Mapping[str, Any], extracted: Optional[Mapping[str, Any]], *, strict: bool = False) -> List[FieldVerdict]:
    """One verdict per expected field, plus one per extra field when ``strict``.

    ``extracted`` may be None (no extraction at all): every expected field is then
    missing. That is deliberate — a failed run scores zero and stays in the sample.
    """
    got: Mapping[str, Any] = extracted or {}
    verdicts: List[FieldVerdict] = []
    for key, exp in expected.items():
        present = key in got
        val = got.get(key)
        if is_empty(exp):
            if is_empty(val):
                verdicts.append(FieldVerdict(key, exp, val, True, both_empty=True, reason="both empty"))
            else:
                verdicts.append(FieldVerdict(key, exp, val, False, hallucinated=True, reason="value where none was expected"))
            continue
        if not present or is_empty(val):
            verdicts.append(FieldVerdict(key, exp, val, False, missing=True, reason="missing from extraction"))
        elif values_match(exp, val):
            verdicts.append(FieldVerdict(key, exp, val, True, reason="match"))
        else:
            verdicts.append(FieldVerdict(key, exp, val, False, reason="value differs"))
    if strict:
        for key, val in got.items():
            if key in expected or is_empty(val):
                continue
            verdicts.append(FieldVerdict(key, None, val, False, hallucinated=True, reason="field not in the expectation"))
    return verdicts


def metrics(verdicts: Iterable[FieldVerdict]) -> Metrics:
    m = Metrics()
    for v in verdicts:
        m.n_fields += 1
        expected_nonempty = not is_empty(v.expected)
        extracted_nonempty = not is_empty(v.extracted)
        if expected_nonempty:
            m.n_expected += 1
        if extracted_nonempty:
            m.n_extracted += 1
        if v.both_empty:
            m.tn += 1
        elif v.hallucinated:
            m.fp += 1
            m.hallucinated += 1
        elif v.matched:
            m.tp += 1
        elif v.missing:
            m.fn += 1
        else:  # wrong value: counted on both sides, as GAIK and the classic definition do
            m.fp += 1
            m.fn += 1
    return m


def micro_aggregate(items: Iterable[Metrics]) -> Metrics:
    """Sum the counts, then derive — never average per-item averages (that moves the denominator)."""
    total = Metrics()
    for m in items:
        total = total + m
    return total


def classify_run_failure(*, product_found: bool, extracted: Optional[Mapping[str, Any]], pipeline_ok: bool = True) -> str:
    """Why a case could not be scored on its fields, or ``none`` if it could."""
    if not pipeline_ok:
        return "pipeline_failed"
    if not product_found:
        return "no_product"
    if extracted is None or all(is_empty(v) for v in extracted.values()):
        return "no_extraction"
    return "none"


@dataclass
class Agreement:
    """Across REPEATS of one case, per field: did the same thing come out each time?"""

    runs: int
    #: field -> share of runs whose VALUE equals the modal value (1000 == "1000.0")
    cell: Dict[str, float] = field(default_factory=dict)
    #: field -> share of runs whose raw BYTES equal the modal bytes
    byte: Dict[str, float] = field(default_factory=dict)
    #: field -> share of runs where the field was non-empty. Read beside `cell`.
    completeness: Dict[str, float] = field(default_factory=dict)

    @property
    def cell_mean(self) -> Optional[float]:
        return _round(sum(self.cell.values()) / len(self.cell)) if self.cell else None

    @property
    def byte_mean(self) -> Optional[float]:
        return _round(sum(self.byte.values()) / len(self.byte)) if self.byte else None

    @property
    def completeness_mean(self) -> Optional[float]:
        return _round(sum(self.completeness.values()) / len(self.completeness)) if self.completeness else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "runs": self.runs,
            "cell_mean": self.cell_mean,
            "byte_mean": self.byte_mean,
            "completeness_mean": self.completeness_mean,
            "cell": {k: _round(v) for k, v in self.cell.items()},
            "byte": {k: _round(v) for k, v in self.byte.items()},
            "completeness": {k: _round(v) for k, v in self.completeness.items()},
        }


def agreement(extractions: Sequence[Optional[Mapping[str, Any]]], fields: Optional[Iterable[str]] = None) -> Agreement:
    """Agreement across repeats. Rows are addressed by FIELD NAME, never by position.

    A run that produced nothing (None) counts as "empty" for every field — it stays in
    the denominator, because dropping it would let a crashing configuration look stable.
    """
    n = len(extractions)
    out = Agreement(runs=n)
    if n == 0:
        return out
    keys = list(fields) if fields is not None else sorted({k for e in extractions if e for k in e.keys()})
    for k in keys:
        raw_values = [None if e is None else e.get(k) for e in extractions]
        cells = [normalize_text(v) for v in raw_values]
        # Numbers agree as values: fold every parsable one to a canonical float string.
        cells = [(_round(parse_number(v)) if parse_number(v) is not None else c) for v, c in zip(raw_values, cells)]
        cells = [str(c) for c in cells]
        # JSON for every value, strings included: the int 1000 and the string "1000" are the
        # same CELL and different BYTES, and byte agreement exists to say so.
        bytes_ = ["" if v is None else json.dumps(v, sort_keys=True, ensure_ascii=False) for v in raw_values]
        out.cell[k] = Counter(cells).most_common(1)[0][1] / n
        out.byte[k] = Counter(bytes_).most_common(1)[0][1] / n
        out.completeness[k] = sum(0 if is_empty(v) else 1 for v in raw_values) / n
    return out


def flatten_product_fields(product: Mapping[str, Any]) -> Dict[str, Any]:
    """The comparable view of a `products` row: scalar columns + `attributes` keys.

    `attributes` is the GOLD layer (canonicalised facets); `attributes_raw` and
    `metadata` are not compared — a golden case states what the page says, and the
    canonical value is what the platform claims it says. A key present in both the
    row and `attributes` is taken from `attributes`.
    """
    out: Dict[str, Any] = {}
    for col in ("name", "sku", "external_sku", "description", "category", "brand", "barcode", "country_of_origin", "measurement_unit_code"):
        if col in product and not is_empty(product.get(col)):
            out[col] = product[col]
    attrs = product.get("attributes")
    if isinstance(attrs, Mapping):
        for k, v in attrs.items():
            if not is_empty(v):
                out[str(k)] = v
    return out
