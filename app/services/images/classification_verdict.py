"""One definition of "did the classifier give a verdict" and of what a verdict writes."""

from datetime import datetime
from typing import Any, Dict, Optional

#: Columns a verdict may write. `classification` is NOT a column of document_images.
VERDICT_COLUMNS = frozenset({"confidence", "category", "image_type", "metadata"})

_IMAGE_TYPES = frozenset({"PRODUCT_IMAGE", "MIXED", "DECORATIVE", "TECHNICAL_DIAGRAM"})


def classification_failed(result: Optional[Dict[str, Any]]) -> bool:
    """True when `result` is an outage or a malformed reply, never a judgement about the image."""
    if not result:
        return True
    model = str(result.get("model") or "")
    reason = str(result.get("reason") or "")
    return bool(
        "error" in result
        or "_failed" in model
        or "_empty_response" in model
        or "_invalid_response" in model
        or "_no_tool_use" in model
        or reason.startswith("API error")
        or reason.startswith("Error:")
        or reason == "No tool_use block returned"
        or reason == "API key missing"
    )


def failed_result(model: str, reason: str) -> Dict[str, Any]:
    """The shape every classifier failure returns, so `classification_failed` recognises it."""
    return {
        "is_material": False,
        "confidence": 0.0,
        "reason": reason,
        "model": f"{model}_failed",
        "error": reason,
    }


def verdict_update(
    metadata: Optional[Dict[str, Any]],
    result: Dict[str, Any],
    by: str,
) -> Dict[str, Any]:
    """The `document_images` update for a real verdict; raises on a failed one."""
    if classification_failed(result):
        raise ValueError("a failed classification must not be written as a verdict")
    is_material = bool(result.get("is_material", False))
    category = result.get("classification")
    meta = dict(metadata or {})
    meta["ai_classification"] = {
        "is_material": is_material,
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "model": result.get("model"),
        "classification": category,
        "classification_pending": False,
        "reclassified_at": datetime.utcnow().isoformat(),
        "reclassified_by": by,
    }
    update: Dict[str, Any] = {
        "confidence": result.get("confidence") or 0.0,
        "category": "product" if is_material else "general",
        "metadata": meta,
    }
    if category in _IMAGE_TYPES:
        update["image_type"] = category
    return update


MIN_CLASSIFY_SIDE_PX = 100


def below_minimum_size(img_data: Dict[str, Any]) -> bool:
    """True when the extracted image is known to be smaller than a product photo can be."""
    try:
        width, height = float(img_data.get("width") or 0), float(img_data.get("height") or 0)
    except (TypeError, ValueError):
        return False
    if width <= 0 or height <= 0:
        return False
    return max(width, height) < MIN_CLASSIFY_SIDE_PX
