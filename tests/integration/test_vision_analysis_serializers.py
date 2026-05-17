"""
VisionAnalysis serializer + Stage 4.7 rollup contract tests.

Why this file exists
--------------------
On 2026-05-04 we audited the metadata extraction path and found three classes
of silent bugs that had been bleeding into production data for weeks:

  1. The Stage 4.7 `_rollup_vision_analysis` was reading legacy field names
     (`pattern`, `texture`, `design_style`, `material_subtype`, `primary_color_hex`)
     that the schema-locked `VisionAnalysis` model doesn't emit. Net effect:
     vision-derived fields like Design Style and Material Subtype were never
     populated on `products.metadata`, so the product detail UI rendered empty.

  2. The rollup `.lower()`-ed every text field before write, so the UI saw
     "matte" / "glossy" instead of properly title-cased "Matte" / "Glossy" —
     the modal's title-case helper mostly hid the issue but inconsistencies
     leaked through (e.g. cert chip rendering).

  3. The four aspect serializers (color/texture/style/material) in
     `app.models.vision_analysis` produce the text strings that Voyage
     embeds into the four `vecs.image_<aspect>_embeddings` collections.
     If any serializer drifts from byte-stability, the aspect embedding
     space corrupts silently — same image embedded at two different times
     would land at different points in latent space.

This file pins the contracts:
  - Aspect serializers are byte-deterministic for identical inputs.
  - The full `serialize_vision_analysis_to_text` is byte-deterministic.
  - `_rollup_vision_analysis` reads canonical schema field names AND
    accepts the documented legacy fallbacks.
  - `_rollup_vision_analysis` preserves original case (no `.lower()` clobber).
  - `_most_common_pretty` / `_dedupe_pretty` group case-insensitively but
    emit original-case representatives.

If any of these regress, CI fails before the bad data hits production.
"""

from __future__ import annotations

import pytest

from app.models.vision_analysis import (
    SCHEMA_VERSION,
    VisionAnalysis,
    serialize_vision_analysis_to_text,
    serialize_aspect_color,
    serialize_aspect_texture,
    serialize_aspect_style,
    serialize_aspect_material,
    vision_analysis_from_legacy_dict,
)
from app.api.pdf_processing.stage_4_products import (
    _rollup_vision_analysis,
    _most_common_pretty,
    _dedupe_pretty,
    _case_fold_key,
)


# ────────────────────────────────────────────────────────────────────────
# Aspect serializers — determinism + None handling
# ────────────────────────────────────────────────────────────────────────


def _make_va(**overrides) -> VisionAnalysis:
    """Build a VisionAnalysis with reasonable defaults; override per test."""
    base = dict(
        material_type="warm honey-brown engineered oak",
        category="flooring",
        subcategory="engineered wood",
        colors=["warm honey-brown", "amber accents"],
        textures=["wood grain", "matte open pore"],
        finish="matte",
        surface_pattern="herringbone",
        description="Engineered oak with herringbone layout and matte open-pore finish.",
        applications=["residential floor", "feature wall"],
        style="Scandinavian",
        confidence=0.92,
        detected_text=["AB-1234", "Made in Italy"],
    )
    base.update(overrides)
    return VisionAnalysis(**base)


class TestAspectSerializerDeterminism:
    """Two VisionAnalysis instances with identical field values must produce
    byte-identical aspect text. This is what protects the
    image_<aspect>_embeddings collections from drift."""

    def test_color_aspect_is_deterministic(self) -> None:
        va1 = _make_va()
        va2 = _make_va()
        assert serialize_aspect_color(va1) == serialize_aspect_color(va2)
        # Field-level repeatability across distinct construction orders.
        va3 = VisionAnalysis(
            material_type="x",
            colors=["warm honey-brown", "amber accents"],
        )
        va4 = VisionAnalysis(
            colors=["warm honey-brown", "amber accents"],
            material_type="x",
        )
        assert serialize_aspect_color(va3) == serialize_aspect_color(va4)

    def test_texture_aspect_is_deterministic(self) -> None:
        va1 = _make_va()
        va2 = _make_va()
        assert serialize_aspect_texture(va1) == serialize_aspect_texture(va2)

    def test_style_aspect_is_deterministic(self) -> None:
        va1 = _make_va()
        va2 = _make_va()
        assert serialize_aspect_style(va1) == serialize_aspect_style(va2)

    def test_material_aspect_is_deterministic(self) -> None:
        va1 = _make_va()
        va2 = _make_va()
        assert serialize_aspect_material(va1) == serialize_aspect_material(va2)

    def test_full_serialization_is_deterministic(self) -> None:
        """`serialize_vision_analysis_to_text` is the single source of truth
        for understanding-embedding text. Drift = `image_understanding_embeddings`
        becomes inconsistent. Lock byte-stability here."""
        va1 = _make_va()
        va2 = _make_va()
        assert serialize_vision_analysis_to_text(va1) == serialize_vision_analysis_to_text(va2)


class TestAspectSerializerNoneHandling:
    """When the vision model didn't surface a given aspect, the serializer
    returns None (caller treats this as "skip"). Material is the lone always-
    present aspect because `material_type` is required by the schema."""

    def test_color_aspect_returns_none_when_no_colors(self) -> None:
        va = _make_va(colors=[])
        assert serialize_aspect_color(va) is None

    def test_color_aspect_strips_blanks(self) -> None:
        # Whitespace-only entries shouldn't bleed into the embedding text.
        va = _make_va(colors=["warm white", "   ", "grey veining"])
        out = serialize_aspect_color(va)
        assert out == "warm white, grey veining"

    def test_texture_aspect_falls_back_to_finish_only(self) -> None:
        va = _make_va(textures=[], finish="matte")
        # finish-only is still useful tactile signal.
        assert serialize_aspect_texture(va) == "matte"

    def test_texture_aspect_returns_none_when_neither(self) -> None:
        va = _make_va(textures=[], finish=None)
        assert serialize_aspect_texture(va) is None

    def test_style_aspect_returns_none_when_no_signals(self) -> None:
        va = _make_va(style=None, surface_pattern=None, applications=[])
        assert serialize_aspect_style(va) is None

    def test_material_aspect_always_returns_string(self) -> None:
        # material_type is required (Pydantic-enforced), so this serializer
        # must always return a non-empty string.
        va = VisionAnalysis(material_type="brushed brass")
        assert serialize_aspect_material(va) == "brushed brass"


class TestSerializerOrderStability:
    """List ordering is part of the byte-stable contract. Voyage produces
    different vectors for ['blue', 'red'] vs ['red', 'blue'], so the order
    we feed it must match the order in source."""

    def test_color_aspect_preserves_input_order(self) -> None:
        va_a = _make_va(colors=["red", "green", "blue"])
        va_b = _make_va(colors=["blue", "green", "red"])
        # Different input order → different output (no internal sort).
        assert serialize_aspect_color(va_a) != serialize_aspect_color(va_b)

    def test_full_serialization_preserves_list_order(self) -> None:
        va_a = _make_va(textures=["wood grain", "matte open pore"])
        va_b = _make_va(textures=["matte open pore", "wood grain"])
        assert serialize_vision_analysis_to_text(va_a) != serialize_vision_analysis_to_text(va_b)


class TestLegacyDictCoercion:
    """`vision_analysis_from_legacy_dict` is what the backfill cron uses to
    re-coerce pre-2026-05-01 rows into the strict schema. It must accept
    the documented legacy field names (color_palette, surface_finish, etc.)
    and reject error envelopes."""

    def test_coerces_canonical_dict(self) -> None:
        d = {
            "material_type": "porcelain tile",
            "category": "flooring",
            "subcategory": "porcelain tile",
            "colors": ["warm white"],
            "textures": ["veined"],
            "finish": "matte",
            "surface_pattern": "stacked bond",
            "description": "Warm-white veined porcelain.",
            "applications": ["bathroom floor"],
            "style": "contemporary",
            "confidence": 0.88,
            "detected_text": ["60x60"],
        }
        va = vision_analysis_from_legacy_dict(d)
        assert va is not None
        assert va.material_type == "porcelain tile"
        assert va.surface_pattern == "stacked bond"
        assert va.finish == "matte"

    def test_coerces_legacy_field_names(self) -> None:
        # Legacy names that pre-2026-05-01 rows can carry.
        d = {
            "type": "marble slab",  # legacy alias for material_type
            "color_palette": ["calacatta white", "grey veining"],  # legacy alias for colors
            "surface_texture": "natural stone veining",  # legacy alias for textures
            "surface_finish": "honed",  # legacy alias for finish
            "pattern_type": "veined",  # legacy alias for surface_pattern
            "visual_description": "Calacatta marble.",  # legacy alias for description
            "suitable_for": ["countertop"],  # legacy alias for applications
            "aesthetic": "luxury contemporary",  # legacy alias for style
            "ocr_text": ["KAI-100"],  # legacy alias for detected_text
        }
        va = vision_analysis_from_legacy_dict(d)
        assert va is not None
        assert va.material_type == "marble slab"
        assert "calacatta white" in va.colors
        assert va.finish == "honed"
        assert va.surface_pattern == "veined"
        assert va.style == "luxury contemporary"
        assert "KAI-100" in va.detected_text

    def test_rejects_error_envelope(self) -> None:
        d = {"error": "OOM", "message": "Out of memory"}
        assert vision_analysis_from_legacy_dict(d) is None

    def test_rejects_missing_material_type(self) -> None:
        d = {"colors": ["white"], "finish": "matte"}
        assert vision_analysis_from_legacy_dict(d) is None

    def test_rejects_non_dict(self) -> None:
        assert vision_analysis_from_legacy_dict(None) is None
        assert vision_analysis_from_legacy_dict("string") is None
        assert vision_analysis_from_legacy_dict(["list"]) is None


# ────────────────────────────────────────────────────────────────────────
# Stage 4.7 rollup helpers — case-folded grouping with original-case output
# ────────────────────────────────────────────────────────────────────────


class TestCaseFoldedHelpers:
    def test_case_fold_key_normalizes(self) -> None:
        assert _case_fold_key("  Matte  ") == "matte"
        assert _case_fold_key("MATTE") == "matte"
        assert _case_fold_key("matte") == "matte"

    def test_most_common_pretty_picks_majority_winner(self) -> None:
        # "Matte" appears 3x, "Glossy" 2x → "Matte" wins.
        winner = _most_common_pretty(["Matte", "matte", "Matte", "Glossy", "glossy"])
        assert winner is not None
        assert winner.lower() == "matte"

    def test_most_common_pretty_preserves_original_case(self) -> None:
        # Critical: this is what fixed the "matte"-everywhere bug.
        winner = _most_common_pretty(["Matte", "Matte", "matte"])
        assert winner == "Matte"

    def test_most_common_pretty_returns_none_on_empty(self) -> None:
        assert _most_common_pretty([]) is None

    def test_dedupe_pretty_unions_case_insensitively(self) -> None:
        result = _dedupe_pretty(["Matte", "matte", "Glossy", "GLOSSY", "Satin"], cap=10)
        # Three unique values regardless of case.
        assert len(result) == 3

    def test_dedupe_pretty_respects_cap(self) -> None:
        result = _dedupe_pretty(["a", "b", "c", "d", "e"], cap=3)
        assert len(result) == 3


# ────────────────────────────────────────────────────────────────────────
# Stage 4.7 rollup — schema-aligned reads + legacy fallbacks
# ────────────────────────────────────────────────────────────────────────


class TestRollupSchemaAlignment:
    """Pre-2026-05-04 the rollup read legacy field names that the current
    schema doesn't emit. These tests pin the schema-aligned reads."""

    def _row(self, va_dict: dict) -> dict:
        return {"vision_analysis": va_dict}

    def test_reads_surface_pattern_from_schema(self) -> None:
        rows = [
            self._row({"material_type": "tile", "surface_pattern": "Herringbone"}),
            self._row({"material_type": "tile", "surface_pattern": "Herringbone"}),
        ]
        out = _rollup_vision_analysis(rows)
        assert out.get("pattern") == "Herringbone"
        assert out.get("patterns") == ["Herringbone"]

    def test_falls_back_to_legacy_pattern_key(self) -> None:
        # Pre-2026-05-04 rows used `pattern` (the schema renamed to surface_pattern).
        rows = [self._row({"material_type": "tile", "pattern": "chevron"})]
        out = _rollup_vision_analysis(rows)
        assert out.get("pattern") == "chevron"

    def test_reads_textures_list_from_schema(self) -> None:
        rows = [
            self._row({
                "material_type": "stone",
                "textures": ["veined", "polished"],
            }),
            self._row({
                "material_type": "stone",
                "textures": ["veined"],
            }),
        ]
        out = _rollup_vision_analysis(rows)
        assert out.get("texture") == "veined"  # majority winner
        assert "veined" in out.get("textures", [])
        assert "polished" in out.get("textures", [])

    def test_falls_back_to_legacy_singular_texture(self) -> None:
        rows = [self._row({"material_type": "wood", "texture": "wood grain"})]
        out = _rollup_vision_analysis(rows)
        assert out.get("texture") == "wood grain"

    def test_reads_style_from_schema(self) -> None:
        rows = [
            self._row({"material_type": "tile", "style": "Scandinavian"}),
            self._row({"material_type": "tile", "style": "Scandinavian"}),
        ]
        out = _rollup_vision_analysis(rows)
        # Writes both new + legacy keys for backwards-compat.
        assert out.get("style") == "Scandinavian"
        assert out.get("design_style") == "Scandinavian"

    def test_falls_back_to_legacy_design_style(self) -> None:
        rows = [self._row({"material_type": "tile", "design_style": "industrial"})]
        out = _rollup_vision_analysis(rows)
        assert out.get("design_style") == "industrial"
        assert out.get("style") == "industrial"

    def test_reads_category_subcategory(self) -> None:
        rows = [
            self._row({
                "material_type": "tile",
                "category": "flooring",
                "subcategory": "porcelain tile",
            }),
        ]
        out = _rollup_vision_analysis(rows)
        assert out.get("category") == "flooring"
        assert out.get("subcategory") == "porcelain tile"
        # Subcategory is mirrored to legacy material_subtype for modal compat.
        assert out.get("material_subtype") == "porcelain tile"

    def test_reads_detected_text(self) -> None:
        rows = [
            self._row({
                "material_type": "tile",
                "detected_text": ["AB-1234", "Made in Italy"],
            }),
            self._row({
                "material_type": "tile",
                "detected_text": ["AB-1234", "60x60"],
            }),
        ]
        out = _rollup_vision_analysis(rows)
        # Union, deduped (order by frequency).
        dt = out.get("detected_text", [])
        assert "AB-1234" in dt
        assert "Made in Italy" in dt
        assert "60x60" in dt

    def test_aggregates_confidence_as_mean(self) -> None:
        rows = [
            self._row({"material_type": "tile", "confidence": 0.9}),
            self._row({"material_type": "tile", "confidence": 0.8}),
            self._row({"material_type": "tile", "confidence": 0.7}),
        ]
        out = _rollup_vision_analysis(rows)
        # Mean = 0.8 (rounded to 3 decimals).
        assert out.get("vision_confidence") == pytest.approx(0.8, abs=0.001)

    def test_picks_longest_description(self) -> None:
        rows = [
            self._row({"material_type": "tile", "description": "Short."}),
            self._row({"material_type": "tile", "description": "A much longer description with detail."}),
        ]
        out = _rollup_vision_analysis(rows)
        assert out.get("vision_description") == "A much longer description with detail."

    def test_returns_empty_on_empty_input(self) -> None:
        assert _rollup_vision_analysis([]) == {}

    def test_preserves_finish_case(self) -> None:
        # Critical regression: pre-2026-05-04 the rollup .lower()-ed finish
        # so the modal saw "matte" instead of "Matte".
        rows = [
            self._row({"material_type": "tile", "finish": "Matte"}),
            self._row({"material_type": "tile", "finish": "Matte"}),
            self._row({"material_type": "tile", "finish": "matte"}),
        ]
        out = _rollup_vision_analysis(rows)
        # Capitalized appears more often → wins, original case kept.
        assert out.get("finish") == "Matte"

    def test_preserves_color_case(self) -> None:
        rows = [
            self._row({"material_type": "tile", "colors": ["Warm White", "Grey"]}),
            self._row({"material_type": "tile", "colors": ["Warm White"]}),
        ]
        out = _rollup_vision_analysis(rows)
        colors = out.get("appearance_colors", [])
        # Original case preserved on output.
        assert "Warm White" in colors


class TestRollupDoesntCrash:
    """Defensive — the rollup is called with whatever shape the DB carries.
    Past audits found rows with missing fields, malformed structures, etc.
    The rollup must not raise on any of these."""

    def test_handles_missing_vision_analysis_field(self) -> None:
        rows = [{}, {"vision_analysis": None}, {"unrelated": "data"}]
        # Should not raise, just return empty.
        out = _rollup_vision_analysis(rows)
        assert isinstance(out, dict)

    def test_handles_string_vision_analysis(self) -> None:
        # If something stored a JSON string instead of dict.
        rows = [{"vision_analysis": "not a dict"}]
        out = _rollup_vision_analysis(rows)
        assert isinstance(out, dict)

    def test_handles_invalid_confidence(self) -> None:
        rows = [
            {"vision_analysis": {"material_type": "tile", "confidence": "not a number"}},
            {"vision_analysis": {"material_type": "tile", "confidence": 1.5}},  # out of range
            {"vision_analysis": {"material_type": "tile", "confidence": 0.85}},
        ]
        out = _rollup_vision_analysis(rows)
        # Only the valid 0.85 should be averaged in — out of range and string are dropped.
        assert out.get("vision_confidence") == pytest.approx(0.85, abs=0.001)


# ────────────────────────────────────────────────────────────────────────
# Schema version pin — bump catches anyone forgetting to update consumers
# ────────────────────────────────────────────────────────────────────────


def test_schema_version_is_set() -> None:
    """If someone bumps SCHEMA_VERSION, this test reminds them to update
    the backfill (so it picks up stale rows under the new version) and
    the consumers that care about version-gated behavior."""
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 2  # v2 introduced the aspect serializers
