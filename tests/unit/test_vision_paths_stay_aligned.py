"""
Guard: every path that produces a VisionAnalysis produces it the SAME way.

WHY THIS EXISTS
---------------
`app/models/vision_analysis.py` opens by saying its call paths "MUST stay aligned or
the `vecs.image_understanding_embeddings` collection drifts". That was a comment, and
comments do not hold. By 2026-08-28 there were four paths and they had drifted into
three regimes:

    ingestion              claude-opus-5      8192 tokens   thinking on    DB prompt
    understanding_backfill claude-opus-4-8    4096 tokens   thinking OFF   hardcoded prompt
    aspect_query           claude-opus-4-8    4096 tokens   thinking OFF
    rag_service            claude-opus-4-8    4096 tokens   thinking OFF

Why that is not cosmetic: `serialize_vision_analysis_to_text` turns the result into
the string Voyage embeds, so the vector encodes the model's WORD CHOICES. A
backfilled image and a freshly ingested one therefore sat in one collection
describing the same material under different instructions, from different models,
with different reasoning budgets — and `aspect_query` analysed the SEARCH image in a
third regime, so the query vector spoke a different dialect from the corpus it was
searching.

Nothing raises. Every vector is well-formed, HNSW ranks them all, cosine returns a
confident number. It is the same hazard the no-fallback-embedder rule exists for,
arriving through the text side instead of the vector side.

The backfill case is the sharpest: its entire job is to make stale rows match current
ones, and a private prompt made it manufacture a third regime instead.

Source-based: reads the files, imports nothing. MIVAA CI installs pytest only.
"""


from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"

#: Every module that calls the vision tool and writes into the shared collection.
_VISION_CALL_SITES = [
    _APP / "services" / "images" / "image_processing_service.py",
    _APP / "services" / "embeddings" / "understanding_backfill.py",
    _APP / "services" / "search" / "aspect_query.py",
    _APP / "services" / "search" / "rag_service.py",
]


def _src(p: Path) -> str:
    assert p.exists(), f"missing {p}"
    return p.read_text(encoding="utf-8")


def test_the_call_sites_still_exist() -> None:
    """If a path was renamed, this guard must be repointed, not deleted."""
    for p in _VISION_CALL_SITES:
        assert "VISION_ANALYSIS_TOOL" in _src(p), (
            f"{p.name} no longer references VISION_ANALYSIS_TOOL. If the vision call "
            f"moved, update this list; do not drop the file from it."
        )


@pytest.mark.parametrize("path", _VISION_CALL_SITES, ids=lambda p: p.name)
def test_every_path_resolves_its_model_from_settings(path: Path) -> None:
    """The vision call must not pin a model of its own.

    Checked positively rather than by hunting literals: these files contain OTHER
    Claude calls that legitimately run on a different tier (image classification,
    product enrichment), so "no model literal anywhere in the file" is the wrong
    question. What matters is that the vision call reaches the shared setting.
    """
    assert "anthropic_model_validation" in _src(path), (
        f"{path.name} does not resolve its vision model from "
        f"`anthropic_model_validation`. Three of these four paths were pinned to "
        f"claude-opus-4-8 while ingestion ran Opus 5 — one collection, two models."
    )


@pytest.mark.parametrize("path", _VISION_CALL_SITES, ids=lambda p: p.name)
def test_every_path_uses_the_shared_token_budget(path: Path) -> None:
    """4096 here and 8192 there is a different amount of thinking per image."""
    assert "VISION_MAX_TOKENS" in _src(path), (
        f"{path.name} does not use VISION_MAX_TOKENS. The budget covers adaptive "
        f"thinking as well as the emitted arguments, so a smaller one is a shallower "
        f"analysis, not merely a shorter one."
    )


def test_the_backfill_uses_the_database_prompt_not_its_own() -> None:
    """The path whose job is to make rows MATCH must not use different instructions."""
    src = _src(_APP / "services" / "embeddings" / "understanding_backfill.py")
    assert "load_material_analyzer_prompt" in src, (
        "understanding_backfill no longer loads the DB prompt. It used to send its own "
        "hardcoded sentence, which made the backfill produce a third description "
        "regime in a collection it exists to normalise."
    )
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "Use the emit_vision_analysis tool to" not in code, (
        "the hardcoded backfill prompt is back in live code (the comment explaining "
        "why it was removed is fine and is skipped here)"
    )


def test_the_shared_parameters_live_in_one_place() -> None:
    schema = _src(_APP / "models" / "vision_analysis.py")
    assert "VISION_MAX_TOKENS" in schema and "def vision_call_extra_kwargs" in schema, (
        "the shared vision call parameters left vision_analysis.py. They belong beside "
        "the schema and serialiser they have to stay aligned with."
    )
