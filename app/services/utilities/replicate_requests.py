"""Which Replicate endpoint a model reference goes to. Stdlib only, so the unit test can import it."""

import re
from typing import Any, Dict, Tuple

_VERSION_HASH = re.compile(r"^[0-9a-f]{64}$")

PREDICTIONS_URL = "https://api.replicate.com/v1/predictions"


def replicate_create_request(model_ref: str, inputs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (url, json body) for creating a prediction.

    A pinned VERSION (a 64-hex hash, bare or after ``owner/name:``) goes to ``/v1/predictions``
    with ``{"version", "input"}``. A model SLUG goes to ``/v1/models/{slug}/predictions`` and runs
    the latest version. A slug sent as ``version`` is refused with a 422 — which every inpaint and
    SAM branch in sam_routes did until 2026-09-13, so none of them had ever run.
    """
    ref = model_ref.strip()
    version = ref.rsplit(":", 1)[1] if ":" in ref else ref
    if _VERSION_HASH.match(version):
        return PREDICTIONS_URL, {"version": version, "input": inputs}
    if not re.match(r"^[\w.-]+/[\w.-]+$", ref):
        raise ValueError(f"not a Replicate model slug or version: {model_ref!r}")
    return f"https://api.replicate.com/v1/models/{ref}/predictions", {"input": inputs}
