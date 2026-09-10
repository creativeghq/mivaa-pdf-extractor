"""Vision provider provenance for material image analysis."""

from enum import Enum


class VisionProvider(str, Enum):
    """Provenance of the `vision_analysis` JSON on document_images."""

    CLAUDE = "claude"
    """Claude Opus — the primary vision producer."""

    CLAUDE_FALLBACK = "claude_fallback"
    """A Claude retry that rescued an image the primary call failed on."""

    SKIPPED = "skipped"
    """In-memory only — analysis was deliberately skipped (prompt not loaded)."""

    FAILED = "failed"
    """In-memory only — Claude failed; no vision_analysis written."""

    @classmethod
    def persistable(cls) -> "frozenset[VisionProvider]":
        """Values that may be written to `document_images.vision_provider`.

        Anything outside this set will be rejected by the DB CHECK constraint.
        """
        return frozenset({cls.CLAUDE, cls.CLAUDE_FALLBACK})

    def is_persistable(self) -> bool:
        """True if this provider value can be safely persisted to the DB."""
        return self in self.__class__.persistable()
