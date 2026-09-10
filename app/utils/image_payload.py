"""Normalize an inbound image payload to raw base64."""

from __future__ import annotations

from typing import Optional

# A data URL's payload always follows this marker; anything before it is metadata.
_MARKER = "base64,"


def normalize_base64_image(payload: Optional[str]) -> Optional[str]:
    """Return the bare base64 payload of `payload`, or `None`."""
    if not payload:
        return payload

    stripped = payload.strip()
    if stripped.startswith("data:"):
        _, marker, encoded = stripped.partition(_MARKER)
        # No marker → a non-base64 data URL (e.g. `data:image/svg+xml,<svg…>`). Return it
        # unchanged so the caller's decode fails loudly rather than on a half-parsed string.
        if not marker:
            return stripped
        return encoded.strip()

    return stripped
