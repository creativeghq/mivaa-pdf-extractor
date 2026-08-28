"""
Guard: the second vision reader annotates, and never rewrites the record.

WHY THIS EXISTS
---------------
Issue #393 Step 4 adds an optional second model that reads the same image. The
obvious design — vote per field, keep the winners — is wrong here, and wrong in a way
that leaves no trace.

`serialize_vision_analysis_to_text` turns a VisionAnalysis into the string Voyage
embeds, so the vector encodes the WORD CHOICES of whichever model produced it. A
per-field vote assembles a record no single model wrote, in a dialect neither speaks,
and embeds that. Half a corpus written that way puts a systematic descriptive offset
through one 1024D space, and nothing raises: every vector is well-formed, HNSW ranks
them, cosine returns a confident number. Exactly the hazard the no-fallback-embedder
rule exists for.

So the writer's analysis is persisted and embedded unchanged; the checker only says
whether a second reader saw the same thing. This file pins that, plus the field rules
and the three-way distinction between "never checked", "checked and failed" and
"checked and agreed".

Behavioural half loads the stdlib-only models by path — MIVAA CI installs pytest and
nothing else.
"""

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_CONSENSUS = _ROOT / "app" / "models" / "vision_consensus.py"
_CATEGORY = _ROOT / "app" / "models" / "vision_category_context.py"
_SERVICE = _ROOT / "app" / "services" / "images" / "image_processing_service.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vc():
    return _load(_CONSENSUS, "_vision_consensus_probe")


@pytest.fixture(scope="module")
def cc():
    return _load(_CATEGORY, "_vision_category_probe")


# ── field comparison rules ───────────────────────────────────────────────────

def test_detected_text_is_character_exact(vc) -> None:
    """The whole reason for a second reader. GU10 vs GU1O must not 'agree'."""
    out = vc.compare_vision_analyses(
        {"detected_text": ["GU10", "IP44"]},
        {"detected_text": ["GU1O", "IP44"]},
    )
    assert out["per_field"]["detected_text"] == 0.0
    d = [x for x in out["disagreements"] if x["field"] == "detected_text"][0]
    assert d["only_writer"] == ["GU10"]
    assert d["only_checker"] == ["GU1O"]


def test_detected_text_case_is_not_normalised_away(vc) -> None:
    out = vc.compare_vision_analyses(
        {"detected_text": ["IP44"]}, {"detected_text": ["ip44"]}
    )
    assert out["per_field"]["detected_text"] == 0.0


def test_descriptive_scalars_ignore_case_and_spacing(vc) -> None:
    out = vc.compare_vision_analyses(
        {"material_type": "Calacatta  Marble"}, {"material_type": "calacatta marble"}
    )
    assert out["per_field"]["material_type"] == 1.0


def test_descriptive_lists_are_scored_by_overlap(vc) -> None:
    out = vc.compare_vision_analyses(
        {"colors": ["white", "grey veining"]},
        {"colors": ["white", "grey veining", "cream"]},
    )
    assert 0.0 < out["per_field"]["colors"] < 1.0


def test_prose_and_self_reported_confidence_are_not_compared(vc) -> None:
    """Free text never matches; scoring it would bury the fields that matter."""
    out = vc.compare_vision_analyses(
        {"description": "A pale tile.", "confidence": 0.9, "material_type": "x"},
        {"description": "Totally different prose.", "confidence": 0.2, "material_type": "x"},
    )
    assert "description" not in out["per_field"]
    assert "confidence" not in out["per_field"]
    assert out["agreement"] == 1.0


def test_a_field_neither_model_saw_is_not_scored_as_agreement(vc) -> None:
    """Two silences are not a match — they are no evidence."""
    out = vc.compare_vision_analyses({"finish": None}, {"finish": ""})
    assert "finish" not in out["per_field"]


def test_nothing_comparable_is_reported_as_such_not_as_perfect_agreement(vc) -> None:
    out = vc.compare_vision_analyses({}, {})
    assert out["no_comparable_fields"] is True
    assert out["flagged"] is False


def test_wide_disagreement_is_flagged(vc) -> None:
    out = vc.compare_vision_analyses(
        {"material_type": "oak wood", "finish": "matte", "category": "flooring"},
        {"material_type": "porcelain tile", "finish": "glossy", "category": "wall"},
    )
    assert out["agreement"] == 0.0
    assert out["flagged"] is True


# ── the record ───────────────────────────────────────────────────────────────

def test_a_failed_checker_is_recorded_not_silently_absent(vc) -> None:
    """'Never checked' and 'checked and agreed' must never render the same."""
    rec = vc.build_consensus_record(
        writer_model="w", checker_model="c",
        checker_failed=True, checker_error="boom",
    )
    assert rec["checker_failed"] is True
    assert rec["flagged"] is False       # nothing compared is not a disagreement
    assert "agreement" not in rec        # and it is not an agreement either


def test_the_record_names_both_models(vc) -> None:
    rec = vc.build_consensus_record(
        writer_model="claude-opus-5", checker_model="other",
        comparison=vc.compare_vision_analyses({"finish": "matte"}, {"finish": "matte"}),
    )
    assert rec["writer_model"] == "claude-opus-5"
    assert rec["checker_model"] == "other"
    assert rec["agreement"] == 1.0


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_checker_result_is_never_written_back_into_vision_analysis() -> None:
    """The writer/checker split, asserted against the source.

    If a future edit persists the checker's analysis, the embedding space silently
    acquires a second dialect. There is no runtime signal for that, so the guard has
    to be structural.
    """
    src = _SERVICE.read_text(encoding="utf-8")
    assert "checker_analysis" in src, "the checker call site vanished"
    # The checker's own payload must reach `compare_vision_analyses` and nothing else.
    uses = re.findall(r"checker_analysis[^\n]*", src)
    for line in uses:
        assert "vision_analysis" not in line or "compare_vision_analyses" in line, (
            f"checker_analysis appears to flow into the persisted record: {line!r}. "
            f"The writer's analysis is the record; the checker only annotates."
        )


def test_the_second_read_is_off_by_default_and_must_differ_from_the_writer() -> None:
    cfg = (_ROOT / "app" / "config.py").read_text(encoding="utf-8")
    assert "anthropic_model_vision_checker" in cfg
    assert re.search(
        r"anthropic_model_vision_checker:\s*str\s*=\s*Field\(\s*\n?\s*default=\"\"", cfg
    ), "the second reader must default to OFF — it doubles the per-image vision bill"

    src = _SERVICE.read_text(encoding="utf-8")
    assert "checker_model == writer_model" in src, (
        "Nothing stops the checker being the same model as the writer. That measures "
        "sampling noise, not agreement, and bills for it."
    )


# ── category context ─────────────────────────────────────────────────────────

def test_unknown_category_says_so_rather_than_rendering_an_empty_block(cc) -> None:
    block = cc.build_category_context_block(category_key=None)
    assert "not known" in block
    assert block.startswith(cc.CATEGORY_BLOCK_OPEN)


def test_category_context_carries_tips_vocab_and_absences(cc) -> None:
    block = cc.build_category_context_block(
        category_key="sanitary",
        display_name="Sanitary Ware",
        extraction_tips=["Check the rim for the flush rating"],
        controlled_vocab=["vitreous china", "fireclay"],
        skip_fields=["slip_rating"],
    )
    assert "Sanitary Ware" in block
    assert "flush rating" in block
    assert "vitreous china" in block
    assert "slip_rating" in block


def test_the_product_schema_is_not_leaked_into_the_vision_prompt() -> None:
    """`priority_fields_prompt` enumerates fields VisionAnalysis cannot emit.

    Feeding it to a `strict`-constrained tool call instructs the model to produce
    fields the tool rejects, and the model resolves that contradiction by guessing.
    """
    src = _CATEGORY.read_text(encoding="utf-8")
    assert "priority_fields_prompt" not in src or "NOT used" in src
    service = _SERVICE.read_text(encoding="utf-8")
    assert "priority_fields_prompt" not in service, (
        "The product-extraction field list is being fed to the vision call. Those "
        "fields are Stage 4's schema, not VisionAnalysis's."
    )


def test_category_models_stay_stdlib_only() -> None:
    for path in (_CONSENSUS, _CATEGORY):
        src = path.read_text(encoding="utf-8")
        for imp in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, re.MULTILINE):
            assert imp.split(".")[0] in {"typing"}, (
                f"{path.name} imports {imp!r}; it must stay stdlib-only so this guard "
                f"can load it by path in a pytest-only CI."
            )
