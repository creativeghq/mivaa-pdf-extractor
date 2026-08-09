"""Normalize an inbound image payload to raw base64.

Browsers produce data URLs. `FileReader.readAsDataURL` is how the search page encodes an
upload, so `image_base64` arrives as `data:image/jpeg;base64,/9j/4AA…` while every decode site
in this codebase assumed the bare payload.

That mismatch does NOT raise, which is the entire problem:

    base64.b64decode("data:image/jpeg;base64,/9j/…")

`b64decode` defaults to `validate=False`, so it silently drops characters outside the base64
alphabet and keeps the rest — and `d,a,t,a,i,m,a,g,e,j,p,e,g,b,a,s,e,6,4` and `/` are all IN
that alphabet. The prefix is folded into the stream, shifting every byte after it. You get
bytes, not an exception: ~175 bytes of noise where 160 bytes of JPEG should be.

Downstream, PIL fails to open the noise, the caller catches it, and the search quietly falls
back to embedding the *query text* instead — which on the search page is the uploaded file's
NAME. Five of that page's seven modes require an image and none of them ever looked at one.

So: normalize once, at the boundary, rather than at each decode site.
"""

from __future__ import annotations

from typing import Optional

# A data URL's payload always follows this marker; anything before it is metadata.
_MARKER = "base64,"


def normalize_base64_image(payload: Optional[str]) -> Optional[str]:
    """Return the bare base64 payload of `payload`, or `None`.

    Accepts a data URL (`data:image/png;base64,AAA…`) or an already-bare base64 string, and
    is idempotent — safe to apply at more than one layer, which matters because callers cannot
    always tell which form they were handed.

    Deliberately does NOT validate or decode: callers decode, and a decode error there is a
    real signal about the image. Swallowing it here would rebuild the silent fallback this
    exists to remove.
    """
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
