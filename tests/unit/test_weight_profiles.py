"""
Guard for the canonical search fusion weights.

These weights decide search relevance, and a wrong one is still a valid float — no
typecheck sees it, no integrity probe sees it, and search just quietly gets worse. The
specific failure this guards is adding an embedding vector and missing a profile or a
vocabulary mapping: that vector then scores zero on the path you missed while being
computed, stored and billed, which is the `ops.silent_zero` shape.

Deliberately imports only `weight_profiles` (which imports nothing from `app`) so it
runs with no DB, no secrets and no application bootstrap.
"""

import importlib.util
from pathlib import Path

import pytest

# Loaded by path rather than `from app.services.search.weight_profiles import ...`:
# importing it as a package would execute app/services/__init__.py, which imports the
# Supabase client and therefore the whole runtime dependency set. weight_profiles is
# pure data + arithmetic, so loading the file directly keeps this guard runnable in CI
# in about a second with nothing installed but pytest — which is the difference between
# a guard that runs on every push and one that quietly never runs.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "app" / "services" / "search" / "weight_profiles.py"
_spec = importlib.util.spec_from_file_location("weight_profiles_under_test", _MODULE_PATH)
weight_profiles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(weight_profiles)

EMBEDDING_ASPECTS = weight_profiles.EMBEDDING_ASPECTS
SOURCE_CHANNELS = weight_profiles.SOURCE_CHANNELS
SPECIALIZED_ASPECTS = weight_profiles.SPECIALIZED_ASPECTS
TEXT_SOURCE_SPLIT = weight_profiles.TEXT_SOURCE_SPLIT
WEIGHT_PROFILES = weight_profiles.WEIGHT_PROFILES
DEFAULT_PROFILE = weight_profiles.DEFAULT_PROFILE
PAGE_WEIGHTS = weight_profiles.PAGE_WEIGHTS
_BASE_PROFILES = weight_profiles._BASE_PROFILES
aspect_bias_weights = weight_profiles.aspect_bias_weights
get_profile = weight_profiles.get_profile
image_only_weights = weight_profiles.image_only_weights
normalize = weight_profiles.normalize
profile_to_source_weights = weight_profiles.profile_to_source_weights

TOL = 1e-9

pytestmark = pytest.mark.unit


# ── The profiles themselves ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(WEIGHT_PROFILES))
def test_every_profile_sums_to_one(name):
    """A profile that doesn't sum to 1.0 makes its scores incomparable with the others."""
    assert sum(WEIGHT_PROFILES[name].values()) == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("name", sorted(WEIGHT_PROFILES))
def test_every_profile_covers_every_aspect(name):
    """THE regression this file exists for.

    Add an 8th vector to EMBEDDING_ASPECTS and every profile must be re-weighted. A
    profile missing the new aspect silently scores it at zero on whichever intent
    routes to that profile.
    """
    assert set(WEIGHT_PROFILES[name]) == set(EMBEDDING_ASPECTS)


@pytest.mark.parametrize("name", sorted(WEIGHT_PROFILES))
def test_no_negative_weights(name):
    assert all(w >= 0 for w in WEIGHT_PROFILES[name].values())


def test_default_profile_exists():
    assert DEFAULT_PROFILE in WEIGHT_PROFILES


def test_get_profile_falls_back_for_unknown_names():
    assert get_profile("no-such-intent") == WEIGHT_PROFILES[DEFAULT_PROFILE]


def test_get_profile_returns_a_copy():
    """Callers mutate weights (aspect bias); that must not corrupt the shared table."""
    p = get_profile(DEFAULT_PROFILE)
    p["visual"] = 99.0
    assert WEIGHT_PROFILES[DEFAULT_PROFILE]["visual"] != 99.0


# ── The 7-aspect → 9-source mapping ─────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(WEIGHT_PROFILES))
def test_source_weights_cover_every_channel(name):
    assert set(profile_to_source_weights(WEIGHT_PROFILES[name])) == set(SOURCE_CHANNELS)


@pytest.mark.parametrize("name", sorted(WEIGHT_PROFILES))
def test_source_weights_preserve_total(name):
    """Fanning `text` across three sources must not create or destroy weight."""
    assert sum(profile_to_source_weights(WEIGHT_PROFILES[name]).values()) == pytest.approx(1.0, abs=TOL)


def test_text_split_is_exhaustive():
    assert sum(TEXT_SOURCE_SPLIT.values()) == pytest.approx(1.0, abs=TOL)


def test_source_mapping_matches_the_pre_refactor_formula():
    """Behavior-preservation check against the formula that lived inline in
    rag_routes.py and query_routes.py before it moved into weight_profiles."""
    for name, profile in WEIGHT_PROFILES.items():
        text_w = profile["text"]
        expected = {
            "visual": profile["visual"],
            "chunk": text_w * 0.40,
            "understanding": profile["understanding"],
            "product": text_w * 0.35,
            "keyword": text_w * 0.25,
            "color": profile["color"],
            "texture": profile["texture"],
            "style": profile["style"],
            "material": profile["material"],
        }
        got = profile_to_source_weights(profile)
        for channel, value in expected.items():
            assert got[channel] == pytest.approx(value, abs=TOL), f"{name}.{channel}"


def test_source_mapping_defaults_missing_aspects_from_balanced():
    """A partial dict (e.g. an old cached weight payload) must not yield KeyError."""
    got = profile_to_source_weights({"text": 0.20})
    assert set(got) == set(SOURCE_CHANNELS)
    assert got["chunk"] == pytest.approx(0.20 * TEXT_SOURCE_SPLIT["chunk"], abs=TOL)


# ── Image-only fan-out ──────────────────────────────────────────────────────────

def test_image_only_full_set_matches_pre_refactor_constants():
    """The inline version hardcoded visual 0.30 / understanding 0.20 / 0.50 spread
    across 4 specialized, then renormalized. That total was already 1.0, so the
    derived version must reproduce it exactly."""
    got = image_only_weights(has_understanding=True, specialized_types=list(SPECIALIZED_ASPECTS))
    assert got["visual"] == pytest.approx(0.30, abs=TOL)
    assert got["understanding"] == pytest.approx(0.20, abs=TOL)
    for aspect in SPECIALIZED_ASPECTS:
        assert got[aspect] == pytest.approx(0.125, abs=TOL)


@pytest.mark.parametrize(
    "has_understanding,specialized",
    [
        (True, list(SPECIALIZED_ASPECTS)),
        (False, list(SPECIALIZED_ASPECTS)),
        (True, ["color"]),
        (True, []),
        (False, []),
        (True, ["color", "texture"]),
    ],
)
def test_image_only_always_normalized(has_understanding, specialized):
    """Skipping a collection must redistribute its share, never dilute the rest."""
    got = image_only_weights(has_understanding=has_understanding, specialized_types=specialized)
    assert sum(got.values()) == pytest.approx(1.0, abs=TOL)


def test_image_only_visual_alone_takes_everything():
    assert image_only_weights(has_understanding=False, specialized_types=[]) == {"visual": 1.0}


def test_image_only_ignores_unknown_collection_names():
    got = image_only_weights(has_understanding=False, specialized_types=["color", "bogus"])
    assert "bogus" not in got
    assert sum(got.values()) == pytest.approx(1.0, abs=TOL)


# ── Explicit aspect bias (#277) ─────────────────────────────────────────────────

@pytest.mark.parametrize("aspect", sorted(SPECIALIZED_ASPECTS))
def test_aspect_bias_is_normalized(aspect):
    """The inline version summed to 1.025 — the chosen aspect's 0.025 was overwritten
    by 0.55 rather than swapped, leaving this path's scores on a different scale."""
    assert sum(aspect_bias_weights(aspect).values()) == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("aspect", sorted(SPECIALIZED_ASPECTS))
def test_aspect_bias_dominates(aspect):
    weights = aspect_bias_weights(aspect)
    assert weights[aspect] == max(weights.values())


def test_aspect_bias_rejects_unknown_aspect():
    with pytest.raises(ValueError):
        aspect_bias_weights("text")


@pytest.mark.parametrize("aspect", sorted(SPECIALIZED_ASPECTS))
def test_aspect_bias_covers_every_source_channel(aspect):
    """aspect_bias_weights builds its dict BY HAND rather than deriving it from a
    profile, so it is the one mapping that a new channel can slip past. A missing key
    scores that vector at zero on the explicit-aspect path only — the exact
    computed-stored-billed-unread shape."""
    assert set(aspect_bias_weights(aspect)) == set(SOURCE_CHANNELS)


# ── The page channel (#239) ─────────────────────────────────────────────────────

def test_page_weight_defined_for_every_profile():
    """PAGE_WEIGHTS is what WEIGHT_PROFILES is built from — a profile missing from it
    is a KeyError at import, but this states the requirement where it's readable."""
    assert set(PAGE_WEIGHTS) == set(_BASE_PROFILES)


@pytest.mark.parametrize("name", sorted(WEIGHT_PROFILES))
def test_page_channel_is_present_and_positive(name):
    """A zero page weight is indistinguishable from having forgotten the channel."""
    assert WEIGHT_PROFILES[name]["page"] > 0


@pytest.mark.parametrize("name", sorted(WEIGHT_PROFILES))
def test_page_carve_out_preserves_the_original_ratios(name):
    """THE invariant behind _with_page: the page channel takes a slice of the whole
    profile, it does not rob one aspect. Every original pair keeps its ratio, so the
    seven-aspect tuning survives intact and re-tuning `page` is one number, not eight.
    """
    base = _BASE_PROFILES[name]
    profile = WEIGHT_PROFILES[name]
    scale = 1.0 - PAGE_WEIGHTS[name]
    for aspect, weight in base.items():
        assert profile[aspect] == pytest.approx(weight * scale, abs=TOL), f"{name}.{aspect}"


def test_page_channel_does_not_disturb_the_image_only_fan_out():
    """The image-only path has no page channel (it ranks images, and a page vector
    keys a page). Because the carve-out is proportional and this path renormalizes,
    its weights must be IDENTICAL to the pre-page values — this is what lets the
    channel be added without re-tuning image search. Pinned separately from
    test_image_only_full_set_matches_pre_refactor_constants because that test asserts
    the same numbers for a different reason and could legitimately be retired."""
    got = image_only_weights(has_understanding=True, specialized_types=list(SPECIALIZED_ASPECTS))
    assert "page" not in got
    base = _BASE_PROFILES[DEFAULT_PROFILE]
    expected_visual = base["text"] + base["visual"]
    assert got["visual"] == pytest.approx(expected_visual, abs=TOL)
    assert got["understanding"] == pytest.approx(base["understanding"], abs=TOL)


# ── normalize() ─────────────────────────────────────────────────────────────────

def test_normalize_preserves_ratios():
    got = normalize({"a": 2.0, "b": 6.0})
    assert got["b"] / got["a"] == pytest.approx(3.0, abs=TOL)
    assert sum(got.values()) == pytest.approx(1.0, abs=TOL)


def test_normalize_handles_all_zero_without_dividing_by_zero():
    assert normalize({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
