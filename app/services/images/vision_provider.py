"""
Vision provider provenance for material image analysis.

Tracks WHICH code path produced the `vision_analysis` JSON on a document_image
row, so we can distinguish "primary path succeeded" from "safety net rescued
the image" in stats and dashboards.

Post-Qwen-removal (2026-05-01) the only persistable producer is CLAUDE.
QWEN and CLAUDE_FALLBACK are kept as legacy enum members so historical rows
in `document_images.vision_provider` still validate — but new rows must
write CLAUDE.
"""

from enum import Enum


class VisionProvider(str, Enum):
    """Provenance of the `vision_analysis` JSON on document_images."""

    CLAUDE = "claude"
    """Claude Opus 4.7 (the sole vision producer post-2026-05-01)."""

    QWEN = "qwen"
    """Legacy: pre-2026-05-01 rows produced by the removed Qwen3-VL path."""

    CLAUDE_FALLBACK = "claude_fallback"
    """Legacy: pre-2026-05-01 rescue from Qwen by Claude. New writes use CLAUDE."""

    SKIPPED = "skipped"
    """In-memory only — analysis was deliberately skipped (prompt not loaded)."""

    FAILED = "failed"
    """In-memory only — Claude failed; no vision_analysis written."""

    @classmethod
    def persistable(cls) -> "frozenset[VisionProvider]":
        """Values that may be written to `document_images.vision_provider`.

        Anything outside this set will be rejected by the DB CHECK constraint.
        """
        return frozenset({cls.CLAUDE, cls.QWEN, cls.CLAUDE_FALLBACK})

    def is_persistable(self) -> bool:
        """True if this provider value can be safely persisted to the DB."""
        return self in self.__class__.persistable()
