"""
Vision provider provenance for material image analysis.

Tracks WHICH code path produced the `vision_analysis` JSON on a document_image
row, so we can distinguish "primary path succeeded" from "safety net rescued
the image" in stats and dashboards.

Only QWEN and CLAUDE_FALLBACK are persisted to `document_images.vision_provider`
— enforced by the `check_vision_provider_values` CHECK constraint. SKIPPED and
FAILED are in-memory return values from `_analyze_material_image` describing
code paths where no vision_analysis was produced (so no row is written) and
exist for stats / logging purposes only.
"""

from enum import Enum


class VisionProvider(str, Enum):
    """Provenance of the `vision_analysis` JSON on document_images."""

    QWEN = "qwen"
    """Qwen3-VL primary path produced a valid vision_analysis JSON."""

    CLAUDE_FALLBACK = "claude_fallback"
    """Qwen failed (or returned unparseable JSON); Claude Sonnet 4.7 produced it."""

    SKIPPED = "skipped"
    """In-memory only — analysis was deliberately skipped (prompt not loaded)."""

    FAILED = "failed"
    """In-memory only — both Qwen and Claude failed; no vision_analysis written."""

    @classmethod
    def persistable(cls) -> "frozenset[VisionProvider]":
        """Values that may be written to `document_images.vision_provider`.

        Anything outside this set will be rejected by the DB CHECK constraint.
        """
        return frozenset({cls.QWEN, cls.CLAUDE_FALLBACK})

    def is_persistable(self) -> bool:
        """True if this provider value can be safely persisted to the DB."""
        return self in self.__class__.persistable()
