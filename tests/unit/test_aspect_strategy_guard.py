"""Guard: /api/rag/search must REFUSE an aspect it cannot honor, never ignore it (#277)."""

import re
from pathlib import Path

import pytest

_ROUTES = Path(__file__).resolve().parents[2] / "app" / "api" / "rag_routes.py"
SOURCE = _ROUTES.read_text(encoding="utf-8")

ASPECTS = ("color", "texture", "style", "material")


def test_route_file_is_readable():
    """A moved/renamed route file must fail loudly, not silently pass every assertion."""
    assert _ROUTES.exists(), f"rag_routes.py not found at {_ROUTES}"
    assert "strategy" in SOURCE


def test_aspect_field_still_exists_on_the_request_model():
    """If the field is gone the rest of this guard is vacuously true."""
    assert re.search(r"^\s*aspect:\s*Optional\[str\]", SOURCE, re.MULTILINE), (
        "SearchRequest.aspect not found — if the parameter was removed, delete this guard; "
        "if it was renamed, update it."
    )


def test_aspect_is_rejected_for_every_strategy_that_cannot_honor_it():
    """The pre-dispatch refusal must exist and must key on `!= multi_vector`."""
    assert re.search(
        r"if\s+_requested_aspect\s+and\s+strategy\s*!=\s*[\"']multi_vector[\"']",
        SOURCE,
    ), (
        "The guard `if _requested_aspect and strategy != 'multi_vector'` is missing. "
        "Without it, strategy='image' and strategy='material' accept an aspect and "
        "silently ignore it, returning unbiased results that look correct."
    )


def test_the_refusal_is_a_400_naming_the_endpoint_that_can_answer():
    """A 400 that does not say where to go just moves the dead end."""
    guard_at = SOURCE.find("_requested_aspect and strategy")
    assert guard_at > 0
    block = SOURCE[guard_at : guard_at + 1200]

    assert "HTTP_400_BAD_REQUEST" in block, "the refusal must be a 400, not a silent pass"
    assert "/api/search/by-" in block, (
        "the error must name /api/search/by-<aspect> — the endpoint that CAN match a "
        "single aspect (it runs vision_analysis on query_image, then queries "
        "image_<aspect>_embeddings in the same Voyage space)."
    )


def test_the_guard_runs_before_the_strategy_dispatch():
    """
    Placement is the whole point. Inside a branch it guards one strategy; before the
    dispatch it guards every strategy that exists now and every one added later.
    """
    guard_at = SOURCE.find("_requested_aspect and strategy")
    dispatch_at = SOURCE.find('if strategy == "multi_vector"')

    assert guard_at > 0, "aspect guard not found"
    assert dispatch_at > 0, "strategy dispatch not found"
    assert guard_at < dispatch_at, (
        "The aspect guard moved after the strategy dispatch. It must run BEFORE it, so a "
        "strategy added later cannot inherit the silent drop by default."
    )


@pytest.mark.parametrize("aspect", ASPECTS)
def test_multi_vector_still_applies_the_bias_for_every_aspect(aspect):
    """
    The other half: refusing elsewhere is only correct if multi_vector actually honors it.
    A guard that rejects every path would 'pass' this file while breaking the feature.
    """
    assert "aspect_bias_weights" in SOURCE, (
        "multi_vector no longer calls aspect_bias_weights — the aspect is refused "
        "everywhere and honored nowhere."
    )
    assert re.search(
        r"if\s+_aspect\s+in\s+\((?:[^)]*)\)", SOURCE
    ), "the multi_vector aspect branch is missing"

    membership = re.search(r"if\s+_aspect\s+in\s+\(([^)]*)\)", SOURCE).group(1)
    assert f"'{aspect}'" in membership or f'"{aspect}"' in membership, (
        f"aspect '{aspect}' dropped out of the multi_vector bias branch — requests naming "
        f"it would fall through to the balanced fusion with no error."
    )
