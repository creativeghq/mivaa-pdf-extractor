"""One derivation for "what vector represents this query's <aspect>?" (#277)."""

from __future__ import annotations

import base64 as _b64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The canonical aspect list. Must mirror ASPECT_SERIALIZERS' keys — not asserted at import,
# because importing app.models.vision_analysis here would drag the model layer into every
# consumer of this module. tests/unit/test_aspect_query.py holds the two in agreement instead.
ASPECTS: Tuple[str, ...] = ("color", "texture", "style", "material")

# Cap on a fetched query image. An unbounded read of a user-supplied URL is a memory DoS
# regardless of how safe the host is (invariant 7's size cap).
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


async def _resolve_image_base64(query_image: str) -> Tuple[Optional[str], Optional[str]]:
    """`(base64, error)` from a data URL, an https URL, or a raw base64 string.

    The https branch goes through `safe_fetch_bytes`, which validates the scheme and
    resolved address of every redirect hop — a permitted external host can 302 straight
    into link-local metadata, so validating the first URL only is not enough
    (invariant 7) — and caps the body while streaming.
    """
    if query_image.startswith("data:"):
        from app.utils.image_payload import normalize_base64_image

        encoded = normalize_base64_image(query_image)
        # normalize_base64_image returns the input unchanged when there is no `base64,`
        # marker (e.g. `data:image/svg+xml,<svg…>`), so that case still reads as a data URL.
        if not encoded or encoded.startswith("data:"):
            return None, "Malformed data URL (no base64 payload)"
        return encoded, None

    if query_image.startswith("http"):
        # This was the correct hand-rolled copy, and it had still drifted: the cap was
        # checked AFTER `resp.content` had pulled the whole body into memory, so it
        # rejected an oversized image only once the cost had been paid. `safe_fetch_bytes`
        # aborts mid-stream. Same reason escapeHtml has one owner — two implementations of
        # one rule end up at two strengths, and you find out which is weaker afterwards.
        from app.utils.ssrf_guard import SSRFError, safe_fetch_bytes

        try:
            fetched = await safe_fetch_bytes(query_image, max_bytes=_MAX_IMAGE_BYTES)
        except SSRFError as e:
            return None, f"query_image URL rejected: {e}"
        except Exception as e:
            return None, f"Failed to fetch query_image: {e}"

        if not fetched.ok:
            return None, f"Failed to fetch query_image (HTTP {fetched.status_code})"
        return _b64.b64encode(fetched.content).decode(), None

    # Assume raw base64.
    return query_image, None


async def analyze_query_image(query_image: str) -> Tuple[Optional[Any], Optional[str]]:
    """`(VisionAnalysis, error)` — ONE Anthropic vision call, schema-locked via tool_use.

    Forced `tool_choice` with no JSON-parse fallback, matching the ingestion path: a repaired
    or salvaged analysis would produce aspect text that never existed in the image.
    """
    from app.models.vision_analysis import (
        VisionAnalysis, VISION_ANALYSIS_TOOL, VISION_MAX_TOKENS,
        vision_call_extra_kwargs,
    )
    from app.config import get_settings

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return None, "ANTHROPIC_API_KEY not configured"

    image_b64, err = await _resolve_image_base64(query_image)
    if err:
        return None, err

    # Through the tracked helper, not a raw POST (#33 item 2). The forced tool_choice
    # was already right; what a hand-rolled POST costs is the cost record — this call is
    # OPUS vision, the most expensive model in the roster, and none of it reached
    # `ai_usage_logs`. `call_with_tool` also raises `ToolCallNotReturned` where this
    # returned a string, so a missing tool block is typed rather than stringly-compared.
    try:
        from app.services.core.claude_tool_call import call_with_tool

        result = await call_with_tool(
            task="aspect_query_vision_analysis",
            model=get_settings().anthropic_model_validation,
            max_tokens=VISION_MAX_TOKENS,
            extra_kwargs=vision_call_extra_kwargs(),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": image_b64,
                    }},
                    {"type": "text", "text": (
                        "Use the emit_vision_analysis tool to return a "
                        "structured catalog-grade material analysis for this image."
                    )},
                ],
            }],
            tool=VISION_ANALYSIS_TOOL,
        )
        return VisionAnalysis(**result.data), None
    except Exception as e:
        return None, f"Anthropic vision_analysis call failed: {e}"


async def _embed(text: str) -> Tuple[Optional[List[float]], Optional[str]]:
    """Voyage 1024D, `input_type="query"`. No second embedder to fall through to: the aspect
    collections ARE Voyage space, and a same-dimension vector from another model would score
    against them confidently and meaninglessly. On a Voyage outage this fails explicitly."""
    from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

    try:
        vec = await RealEmbeddingsService()._generate_text_embedding(text=text, input_type="query")
    except Exception as e:
        return None, f"Voyage embed failed: {e}"
    if not vec or len(vec) != 1024:
        return None, (
            f"Voyage returned wrong-dim embedding (len={len(vec) if vec else 0}, expected 1024)"
        )
    return vec, None


async def aspect_query_embedding(
    aspect: str,
    query_image: Optional[str] = None,
    query_text: Optional[str] = None,
) -> Tuple[Optional[List[float]], Optional[str], Optional[str]]:
    """Single aspect → `(embedding, source_text, error)`. Exactly one of embedding/error is set.

    `query_image` wins over `query_text`: an image is grounded in actual visible material,
    where the text is whatever the user happened to type.
    """
    from app.models.vision_analysis import ASPECT_SERIALIZERS

    if not query_image and not query_text:
        return None, None, "Provide either query_image or query_text"

    serializer = ASPECT_SERIALIZERS.get(aspect)
    if not serializer:
        return None, None, f"Unknown aspect: {aspect}"

    if query_image:
        va, err = await analyze_query_image(query_image)
        if err:
            return None, None, err
        source_text = serializer(va)
        if not source_text:
            return None, None, (
                f"VisionAnalysis from query_image had no {aspect} content (e.g. empty colors[])"
            )
    else:
        source_text = (query_text or "").strip()
        if not source_text:
            return None, None, "query_text is empty"

    vec, err = await _embed(source_text)
    if err:
        return None, None, err
    return vec, source_text, None


async def image_query_vectors(
    query_image: str,
    channels: Tuple[str, ...] = ASPECTS + ("understanding",),
) -> Tuple[Dict[str, List[float]], Dict[str, str], Optional[str]]:
    """Every text-space query vector an IMAGE can supply, from ONE vision call."""
    from app.models.vision_analysis import (
        ASPECT_SERIALIZERS,
        serialize_vision_analysis_to_text,
    )

    va, err = await analyze_query_image(query_image)
    if err:
        return {}, {}, err

    serializers = dict(ASPECT_SERIALIZERS)
    serializers["understanding"] = serialize_vision_analysis_to_text

    embeddings: Dict[str, List[float]] = {}
    source_texts: Dict[str, str] = {}
    for channel in channels:
        serializer = serializers.get(channel)
        if not serializer:
            continue
        text = serializer(va)
        if not text:
            logger.debug("⏭️ Query channel '%s' skipped — image yielded no source text", channel)
            continue
        vec, embed_err = await _embed(text)
        if embed_err:
            # One channel failing to embed must not take the others down with it.
            logger.warning("⚠️ Query channel '%s' embed failed: %s", channel, embed_err)
            continue
        embeddings[channel] = vec
        source_texts[channel] = text

    logger.info(
        "🎨 Query image → %d/%d query vectors (%s)",
        len(embeddings), len(channels), ", ".join(sorted(embeddings)) or "none",
    )
    return embeddings, source_texts, None
