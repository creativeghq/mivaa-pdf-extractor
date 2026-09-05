"""The judge's verdicts, validated: pure stdlib, loaded by path in tests.

A second model reads the RENDERED PAGE beside the fields the first model extracted
and says, per field, whether the page supports the value. This module is the part
that does not talk to a model: it decides which of the judge's verdicts are usable
and what they add up to.

Rules:

* A verdict on a field that was never put in front of the judge is DROPPED, and the
  drop is counted. A model can name a field it invented; storing that would let the
  judge hallucinate a column into the review queue.
* A verdict outside the enum, or a score outside 1..5, or a verdict with no reason,
  is dropped for the same reason. Never coerced — a coerced verdict is a valid-
  looking row nothing can raise on (the platform's silent-zero rules).
* The summary is COUNTS, not an average: "2 wrong, 1 suspect, 9 ok" is what the
  reviewer needs; a 0.83 is not.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

#: The DB CHECK on product_field_judgements.verdict admits exactly these.
VERDICTS = ("ok", "suspect", "wrong", "absent_on_page")

#: Likert 1..5, severity-aligned the way GAIK's judge prompt enforces it:
#: 1 → wrong, 2-3 → suspect, 4-5 → ok. `absent_on_page` carries no score.
SCORE_MIN, SCORE_MAX = 1, 5

MAX_REASON_CHARS = 300


def _clean_text(v: Any, limit: int) -> str:
    if not isinstance(v, str):
        return ""
    return " ".join(v.split())[:limit]


def validate_verdicts(raw: Any, allowed_fields: Iterable[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(kept, dropped)``. Each dropped entry carries ``why``."""
    allowed = set(allowed_fields)
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return kept, [{"why": "verdicts is not a list", "raw": raw}]
    seen = set()
    for item in raw:
        if not isinstance(item, Mapping):
            dropped.append({"why": "not an object", "raw": item})
            continue
        fld = item.get("field")
        if not isinstance(fld, str) or fld not in allowed:
            dropped.append({"why": "field was not put before the judge", "field": fld})
            continue
        if fld in seen:
            dropped.append({"why": "duplicate verdict for field", "field": fld})
            continue
        verdict = item.get("verdict")
        if verdict not in VERDICTS:
            dropped.append({"why": "verdict outside the enum", "field": fld, "verdict": verdict})
            continue
        reason = _clean_text(item.get("reason"), MAX_REASON_CHARS)
        if not reason:
            dropped.append({"why": "no reason", "field": fld})
            continue
        score = item.get("score")
        if verdict == "absent_on_page":
            score_val = None
        else:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not float(score).is_integer():
                dropped.append({"why": "score is not an integer", "field": fld, "score": score})
                continue
            score_val = int(score)
            if score_val < SCORE_MIN or score_val > SCORE_MAX:
                dropped.append({"why": "score outside 1..5", "field": fld, "score": score})
                continue
        suggested = item.get("suggested_value")
        if isinstance(suggested, str):
            suggested = _clean_text(suggested, 200) or None
        elif not isinstance(suggested, (int, float)) or isinstance(suggested, bool):
            suggested = None
        seen.add(fld)
        kept.append({
            "field": fld,
            "verdict": verdict,
            "score": score_val,
            "reason": reason,
            "suggested_value": suggested,
        })
    return kept, dropped


def summarize_verdicts(kept: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    """Counts per verdict, every key present even when zero."""
    out = {v: 0 for v in VERDICTS}
    for k in kept:
        v = k.get("verdict")
        if v in out:
            out[v] += 1
    return out


def fields_for_judgement(flat: Mapping[str, Any], *, max_fields: int = 60) -> Dict[str, Any]:
    """The fields the judge is shown: non-empty, capped, stable order.

    The cap is the property-count lesson from the extraction skill applied to the
    OUTPUT side: sixty verdicts is already more than a reviewer reads, and a judge
    asked for two hundred starts skipping.
    """
    items = [(k, v) for k, v in flat.items() if v not in (None, "", [], {})]
    items.sort(key=lambda kv: kv[0])
    return dict(items[:max_fields])
