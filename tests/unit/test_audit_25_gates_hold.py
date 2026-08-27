"""Guards for the mivaa#25 ingestion-service fixes (M12-1 ... M12-5).

M12-1 is the one worth reading twice. `_calculate_clip_score` read three columns that
do not exist on `document_images` -- `clip_embedding`, `visual_embedding`, `embedding` --
and two that do not exist on `products`. The rows are fetched with `select('*')`, so
PostgREST returned what existed and `.get()` answered None for the rest: no KeyError, no
warning. The `or` chain collapsed and the "neutral score" fallback fired on every call
ever made, so a service documented as multi-modal has been text-only for its whole life.

A constant 0.5 at 30% weight is not neutral. It added a fixed +0.15 to every overall
score, it was a third data point in a variance calculation that rewards agreement, and
it cleared the `>= 0.5` branch in `_generate_reasoning` exactly -- so every association
this service ever wrote carried the stored phrase 'moderate visual relevance',
describing a comparison that never happened.

WHY None RATHER THAN 0.0
------------------------
0.0 means "compared them and found no resemblance". None means "no comparison happened".
Those are different claims about the world and only one of them is true here. The column
was NOT NULL, which is why 0.5 was the only representable answer; the migration drops
that so the honest value can be stored.

M12-4 is enforced by the DATABASE, not by this file. `image_product_associations`
gained workspace_id plus a composite FK to (products.id, products.workspace_id), so an
association across tenants is refused by Postgres. The cases here cover the code half:
the tenant is derived once, disagreement raises, and the write carries it.

WATCHED TO FAIL: the whole file was run against the pre-fix tree (the three sources
restored from HEAD). 17 of 22 cases fired. The 5 that pass both ways do so by design:
the four `_blend` cases exercise the renormalisation, which is new logic with no pre-fix
state to fail against, and `test_the_marker_the_whole_finding_rests_on_still_exists`
pins the PREMISE -- `_call_paddleocr` emitting `paddleocr_failed` was always correct,
and every M12-3 case is vacuous if it stops.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

ASSOC = APP / "services" / "images" / "multi_modal_image_product_association_service.py"
OCR = APP / "services" / "pdf" / "ocr_service.py"
SPEC = APP / "services" / "products" / "product_spec_vision_extractor.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _body(src: str, name: str) -> str:
    return _strip_comments(ast.get_source_segment(src, _node(src, name)) or "")


# -------------------------------------------------------------------------
# M12-1 -- the visual signal is absent, not fabricated
# -------------------------------------------------------------------------

@pytest.mark.parametrize("column", ["clip_embedding", "visual_embedding", "'embedding'"])
def test_the_nonexistent_embedding_columns_are_not_read_again(column: str):
    """None of these is a column on `document_images` or `products`. Reading one is
    silent -- select('*') plus .get() yields None rather than raising -- so the only
    thing standing between this and another five years of a constant score is this
    test."""
    src = _strip_comments(_read(ASSOC))
    assert column not in src, (
        f"{column} is back in the association service (#25 M12-1). Image vectors live "
        "in VECS; `has_slig_embedding` / `has_understanding_embedding` are the "
        "canonical O(1) presence checks."
    )


def test_the_scorer_can_return_no_answer():
    body = _body(_read(ASSOC), "_calculate_clip_score")
    assert "return None" in body, (
        "the visual scorer no longer has an 'I could not compute this' answer, so it is "
        "back to inventing one (#25 M12-1)"
    )
    assert "return 0.5" not in body, "the constant neutral score is back"


def test_the_scorer_uses_the_canonical_presence_flags():
    body = _body(_read(ASSOC), "_calculate_clip_score")
    assert "has_slig_embedding" in body or "has_understanding_embedding" in body, (
        "the scorer no longer gates on the canonical boolean flags"
    )


def test_the_blend_renormalises_over_participating_signals():
    """Weights that do not sum to 1 after a signal drops out would silently rescale the
    threshold everything else is compared against."""
    body = _body(_read(ASSOC), "_evaluate_association")
    assert "total_weight" in body and "/ total_weight" in body, (
        "the weighted sum no longer renormalises, so a missing visual score either "
        "contributes a phantom value or shrinks every overall score by 30%"
    )
    assert "clip_score * weights.clip" not in body, (
        "the unconditional clip term is back -- it multiplies None or a constant"
    )


def test_confidence_only_sees_signals_that_were_computed():
    src = _read(ASSOC)
    node = _node(src, "_calculate_confidence")
    args = [a.arg for a in node.args.args]
    assert "clip_score" not in args, (
        "_calculate_confidence takes clip_score positionally again, so an absent signal "
        "must be passed as a placeholder -- which is what let a constant count as "
        "agreement between independent signals (#25 M12-1)"
    )
    assert "scores" in args


def test_the_reasoning_does_not_claim_a_visual_comparison_that_did_not_happen():
    body = _body(_read(ASSOC), "_generate_reasoning")
    at = body.index("clip_score is not None")
    assert at < body.index("moderate visual relevance"), (
        "the visual reasoning branch is no longer guarded, so 'moderate visual "
        "relevance' is written for associations that had no visual signal"
    )


def test_the_stored_metadata_reports_participation():
    src = _strip_comments(_read(ASSOC))
    assert '"participated": clip_score is not None' in src, (
        "the association metadata no longer records whether the visual signal reached "
        "the score -- a reader of the row cannot otherwise tell a real 0.4 from an "
        "absent one"
    )


# -------------------------------------------------------------------------
# M12-1 -- the arithmetic, not just the shape
# -------------------------------------------------------------------------

def _blend(spatial: float, caption: float, clip, w=(0.4, 0.3, 0.3)) -> float:
    """Mirrors the renormalisation in `_evaluate_association`. Kept here rather than
    imported because the module needs supabase at import time and MIVAA's CI installs
    pytest only."""
    components = [(spatial, w[0]), (caption, w[1])]
    if clip is not None:
        components.append((clip, w[2]))
    total = sum(x for _, x in components) or 1.0
    return sum(s * x for s, x in components) / total


def test_an_absent_visual_signal_no_longer_adds_a_phantom_fifteen_points():
    """The exact number that made this matter: spatial 0.0 + a generic caption 0.5 used
    to total 0.30 and clear a 0.30 threshold, writing an association for an image on the
    wrong page -- the thing the spatial gate was added to prevent."""
    assert _blend(0.0, 0.5, 0.5) == pytest.approx(0.30)
    assert _blend(0.0, 0.5, None) == pytest.approx(0.2142857, rel=1e-4)
    assert _blend(0.0, 0.5, None) < 0.30


def test_dropping_a_signal_does_not_shrink_a_strong_association():
    """Renormalising, not just omitting: a perfect spatial+caption match must still
    score 1.0 when the visual signal is absent, or every association would be penalised
    for a signal nobody could compute."""
    assert _blend(1.0, 1.0, None) == pytest.approx(1.0)
    assert _blend(1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_a_real_visual_score_still_counts_for_its_full_weight():
    assert _blend(0.0, 0.0, 1.0) == pytest.approx(0.3)


# -------------------------------------------------------------------------
# M12-4 / M12-5 -- one tenant-bound atomic write
# -------------------------------------------------------------------------

def test_the_write_carries_the_workspace():
    body = _body(_read(ASSOC), "_create_database_associations")
    assert '"workspace_id": workspace_id' in body, (
        "association rows no longer carry a tenant (#25 M12-4) -- and the composite FK "
        "added by the migration will reject the insert outright"
    )


def test_the_tenant_is_derived_once_and_disagreement_refuses():
    body = _body(_read(ASSOC), "create_document_associations")
    assert "workspace_ids" in body and "raise ValueError" in body, (
        "the run no longer verifies that its products resolve to exactly one workspace, "
        "so a corrupt document binds its images to whichever one sorted first"
    )


def test_there_is_exactly_one_upsert():
    """Two upserts of the same rows to the same table meant a failure of the second left
    the first behind -- an association whose stated reason was the placeholder 'depicts'
    and whose metadata was empty, reported as created."""
    body = _body(_read(ASSOC), "_create_database_associations")
    assert body.count(".upsert(") == 1, (
        "the double upsert is back (#25 M12-5); pipeline convention 3 wants one atomic "
        "write, not two that can disagree"
    )
    assert '"depicts"' not in body, "the placeholder reasoning row is back"


def test_a_failed_write_is_not_reported_as_zero_created():
    body = _body(_read(ASSOC), "_create_database_associations")
    assert "raise" in body, (
        "the write swallows its exception again, so a run that stored nothing is "
        "indistinguishable from a catalogue that earned no associations"
    )


# -------------------------------------------------------------------------
# M12-2 -- the spec extractor forces its tool
# -------------------------------------------------------------------------

def test_the_spec_extractor_forces_a_tool_call():
    body = _body(_read(SPEC), "_call_claude_vision")
    assert "tool_choice" in body, "the spec extractor is back on free-form JSON (#25 M12-2)"
    assert "```" not in body, "the markdown-fence repair is back"
    assert "json.loads" not in body


def test_the_spec_tool_schema_stays_permissive():
    """The reply's shape is described by a DB-stored prompt that admins edit. Pinning
    those keys here would create a second source, and because the model is FORCED to
    satisfy the schema, a prompt edit would silently stop taking effect."""
    src = _read(SPEC)
    assert '"additionalProperties": True' in src, (
        "the spec tool grew a closed schema -- it now overrides the stored prompt"
    )


def test_the_spec_extractor_still_returns_none_per_page():
    """`extract_specs_from_pdf_pages` does `if not data: continue`. Raising instead
    would abandon the rest of the catalogue over one bad page."""
    body = _body(_read(SPEC), "_call_claude_vision")
    assert "return None" in body


# -------------------------------------------------------------------------
# M12-3 -- a failed OCR is not an empty one
# -------------------------------------------------------------------------

def test_a_failed_ocr_raises_instead_of_returning_no_items():
    body = _body(_read(OCR), "extract_icon_metadata")
    at = body.find("paddleocr_failed")
    assert at != -1, "the failure marker is no longer inspected at all (#25 M12-3)"
    assert "raise RuntimeError" in body[at:at + 400], (
        "the marker is filtered out again rather than raised, so a PaddleOCR crash and "
        "an empty crop both return [] -- which the caller logs as 'no spec items "
        "extracted' and reports as success"
    )


def test_the_dead_convenience_wrappers_stay_gone():
    """Both discarded `method` and neither had a caller. If one comes back it must
    propagate the marker."""
    src = _read(OCR)
    for name in ("def extract_text_simple(", "def get_text_with_confidence("):
        assert name not in src, (
            f"{name} is back -- it must return `method` alongside the text, or it "
            "reintroduces exactly the ambiguity #25 M12-3 removed"
        )


def test_the_marker_the_whole_finding_rests_on_still_exists():
    """Every case above is about propagating `paddleocr_failed`. If `_call_paddleocr`
    stops producing it, they are all guarding nothing."""
    body = _body(_read(OCR), "_call_paddleocr")
    assert "paddleocr_failed" in body, (
        "_call_paddleocr no longer emits the failure marker, so pipeline convention 1 "
        "is broken at the source and every consumer guard here is vacuous"
    )
