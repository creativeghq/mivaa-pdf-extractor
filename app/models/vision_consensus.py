"""
Two readers, one record: comparing a second vision opinion (issue #393 Step 4).

STDLIB ONLY — same contract as `ocr_context.py` and `vision_category_context.py`.

THE WRITER/CHECKER SPLIT, AND WHY IT IS NOT A VOTE
--------------------------------------------------
`serialize_vision_analysis_to_text` turns a VisionAnalysis into the string Voyage
embeds. The vector therefore encodes the WORD CHOICES of whichever model produced it.
Two models describing the same tile — "Calacatta marble" against "white veined
marble" — land in different places in the same 1024D space.

So a vote that picks per-field winners would assemble a record no single model wrote,
in a dialect neither speaks, and embed that. Half a corpus written that way and half
written by one model puts a systematic descriptive offset through the collection, and
nothing raises: every vector is well-formed, HNSW ranks them all, cosine returns a
confident number. It is the same hazard the no-fallback-embedder rule exists for.

Hence: the WRITER's analysis is persisted and embedded, always, unchanged. The CHECKER
never edits the record. Its only output is agreement — and a disagreement on a SKU is
a signal to have a human look, not a value to silently substitute.

FIELD RULES
-----------
Fields are compared by what they ARE, not uniformly:

- `detected_text` is character-exact. `GU10` and `GU1O` are a real disagreement — that
  is the entire point of reading it twice.
- Descriptive scalars are compared case/whitespace-insensitively; a difference in
  wording is a genuine difference in what was seen.
- Descriptive lists are compared by overlap, because order is meaningless and one
  model listing an extra colour is a weaker signal than two models naming different
  materials.
- `description` and `confidence` are NOT compared. Free prose never matches, and
  scoring it would bury the fields that matter under permanent noise. `confidence` is
  each model's opinion of itself, not a claim about the image.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Character-exact. These are codes, and a near-miss is the failure we are looking for.
EXACT_LIST_FIELDS: Tuple[str, ...] = ("detected_text",)

#: Compared case- and whitespace-insensitively.
SCALAR_FIELDS: Tuple[str, ...] = (
    "material_type",
    "category",
    "subcategory",
    "finish",
    "surface_pattern",
    "style",
)

#: Compared by set overlap.
OVERLAP_LIST_FIELDS: Tuple[str, ...] = ("colors", "textures", "applications")

#: Deliberately never compared — see the module docstring.
IGNORED_FIELDS: Tuple[str, ...] = ("description", "confidence", "schema_version")

#: Below this, the record is flagged for human review. Matches the band
#: `ConsensusValidator.LOW_AGREEMENT` already uses for critical extractions, so the two
#: consensus surfaces do not disagree about what "low" means.
FLAG_BELOW = 0.5


def _norm(v: Any) -> str:
    return " ".join(str(v).strip().lower().split()) if v is not None else ""


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None and str(x).strip()]
    return [str(v)] if str(v).strip() else []


def _overlap(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard over normalised members. Two empty lists AGREE."""
    sa = {_norm(x) for x in a}
    sb = {_norm(x) for x in b}
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def compare_vision_analyses(
    writer: Mapping[str, Any],
    checker: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare two VisionAnalysis payloads. Returns a record, never a merge.

    The returned dict is what gets stored on `document_images.vision_consensus`. It
    names the checker's values so a reviewer can see the alternative, but nothing in
    it is ever written back into `vision_analysis`.
    """
    per_field: Dict[str, float] = {}
    disagreements: List[Dict[str, Any]] = []

    for name in SCALAR_FIELDS:
        w, c = _norm(writer.get(name)), _norm(checker.get(name))
        if not w and not c:
            continue  # neither saw it — no evidence either way, so no score
        score = 1.0 if w == c else 0.0
        per_field[name] = score
        if score < 1.0:
            disagreements.append(
                {"field": name, "writer": writer.get(name), "checker": checker.get(name)}
            )

    for name in OVERLAP_LIST_FIELDS:
        w, c = _as_list(writer.get(name)), _as_list(checker.get(name))
        if not w and not c:
            continue
        score = _overlap(w, c)
        per_field[name] = score
        if score < 1.0:
            disagreements.append({"field": name, "writer": w, "checker": c})

    for name in EXACT_LIST_FIELDS:
        w, c = _as_list(writer.get(name)), _as_list(checker.get(name))
        if not w and not c:
            continue
        # Character-exact set equality. No normalisation: casing IS the signal here.
        sw, sc = set(w), set(c)
        score = 1.0 if sw == sc else 0.0
        per_field[name] = score
        if score < 1.0:
            disagreements.append(
                {
                    "field": name,
                    "writer": sorted(sw),
                    "checker": sorted(sc),
                    "only_writer": sorted(sw - sc),
                    "only_checker": sorted(sc - sw),
                }
            )

    agreement = sum(per_field.values()) / len(per_field) if per_field else 1.0
    # An empty comparison (both models silent on everything comparable) is NOT
    # agreement worth trusting — it is no evidence. Say so rather than scoring 1.0
    # and reading as a confident match.
    compared_any = bool(per_field)

    return {
        "agreement": round(agreement, 4),
        "compared_fields": sorted(per_field),
        "per_field": {k: round(v, 4) for k, v in per_field.items()},
        "disagreements": disagreements,
        "flagged": compared_any and agreement < FLAG_BELOW,
        "no_comparable_fields": not compared_any,
    }


def build_consensus_record(
    *,
    writer_model: str,
    checker_model: str,
    comparison: Optional[Dict[str, Any]] = None,
    checker_failed: bool = False,
    checker_error: Optional[str] = None,
) -> Dict[str, Any]:
    """The jsonb written to `document_images.vision_consensus`.

    A failed checker is recorded as a FAILURE, not as an absent second opinion.
    Storing nothing would make "we never checked" indistinguishable from "we checked
    and it agreed" — the same ambiguity `ocr_failed` exists to prevent.
    """
    record: Dict[str, Any] = {
        "writer_model": writer_model,
        "checker_model": checker_model,
        "checker_failed": checker_failed,
    }
    if checker_failed:
        record["checker_error"] = (checker_error or "unknown")[:500]
        record["flagged"] = False  # nothing was compared; not a disagreement
        return record
    record.update(comparison or {})
    return record
