"""One derivation for "what vector represents this query's <aspect>?" (#277).

The four per-aspect collections (`image_{color,texture,style,material}_embeddings`) hold
**Voyage 1024D embeddings of vision-analysis TEXT** — never image vectors. A row's vector is
`voyage(ASPECT_SERIALIZERS[aspect](VisionAnalysis))`, built at ingest.

So a query only lands in the same space by reproducing that exact chain. Two callers need it
and they must not each grow their own copy:

  * the four `/api/search/by-<aspect>` endpoints — one aspect, one query
  * `rag_service.multi_vector_search` — all four aspects, one image

Before this module the second caller had no derivation at all: it queried every aspect
collection with the *understanding embedding of the query text*. That is correct when the user
typed words, and silently wrong when they did not — the search page requires an image for its
aspect modes and sends the image's FILENAME as the query, so `image_texture_embeddings` was
being searched with `voyage("IMG_2831.jpg")` at 0.55 of the total ranking weight.

Nothing raised: a filename embeds to a perfectly valid 1024D vector, cosine-compares against
every row, and returns confident nonsense. Same failure family as a wrong-space vector — the
shape is right, the meaning is absent.

Vision runs ONCE per image here, not once per aspect: a VisionAnalysis carries all four
aspects' source fields, so four serializers share one Claude call and differ only in the four
cheap Voyage embeds that follow.
"""

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
        #
        # This also tightens the scheme: `assert_safe_url`'s default allowed plaintext
        # http, while invariant 7 and `image_download_service` both say https-only. A
        # plaintext query_image now comes back as "blocked url scheme: 'http'" rather
        # than being fetched. That is a deliberate narrowing, not a side effect.
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
    from app.models.vision_analysis import VisionAnalysis, VISION_ANALYSIS_TOOL

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
            model="claude-opus-4-8",
            max_tokens=4096,
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
    """Every text-space query vector an IMAGE can supply, from ONE vision call.

    Returns `(embeddings, source_texts, error)` keyed by channel — the four aspects plus
    `understanding`. Each mirrors exactly what ingestion embedded for that collection:
    `ASPECT_SERIALIZERS[aspect](va)` for the aspects, `serialize_vision_analysis_to_text(va)`
    for understanding. Same serializer, same model, same space.

    Understanding is included because it is the single heaviest channel in the balanced
    profile (18%) and it was being fed the caller's query TEXT. When that text is a filename
    — which is what the search page sends for its image modes — 18% of the ranking came from
    `voyage("IMG_2831.jpg")`. The vision analysis needed for the aspects already describes
    the whole image, so filling this channel from it costs one extra Voyage embed and no
    extra Claude call.

    `error` is set only when the whole derivation failed (no vision analysis). A per-channel
    miss is normal and NOT an error — an image with no discernible pattern legitimately
    yields no texture string — so that channel is simply absent. Callers must treat a missing
    key as "no query vector for this channel" and skip it rather than substituting another
    channel's vector, which would answer a different question with full confidence.
    """
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
