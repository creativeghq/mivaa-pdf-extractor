"""
Material Segmentation Service

Detects distinct material zones in 3D rendered images using Anthropic
Claude Opus vision. Returns bounding boxes + metadata per zone so the
frontend can crop and send each crop to the existing RAG image search
endpoint.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from app.services.utilities.prompt_registry import load_prompt

logger = logging.getLogger(__name__)

# The segmentation prompt lives in the `prompts` table:
#   prompt_type = 'agent', category = 'segmentation', is_active = true
#
# It used to ALSO live here, as a 9,119-character DEFAULT_SEGMENT_PROMPT that _get_prompt
# returned whenever the DB read raised — logged at DEBUG. An admin editing this prompt in the
# UI would have seen it save and changed nothing, forever, while the system reported perfect
# health. Deleted in #347 phase 3P: there is no fallback, and a missing row is an error.


class SegmentationService:
    """Detects material zones in 3D renders via Anthropic Claude Opus.

    Claude Opus is the only path, and quality on this task is excellent
    (22 well-described zones on a bathroom render in ~40s with catalog-grade
    material names). Do NOT add a second vision provider here without a
    measured quality/cost case AND a health probe that fails loudly - the
    last attempt silently 404'd on every call for months while quietly
    falling through to Anthropic, so nothing ever looked broken.
    """

    def __init__(self):
        import os
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    async def _get_prompt(self) -> str:
        """Fetch the active segmentation prompt. Raises if it is not configured.

        DB record: prompt_type='agent', category='segmentation', is_active=true.
        """
        return await load_prompt("agent", "segmentation")

    async def segment_image(self, image_base64: str) -> List[Dict[str, Any]]:
        """
        Detect material zones in a 3D render via Claude Opus.

        Prompt is loaded from the `prompts` table (category='segmentation'). There is no
        fallback: without the prompt this cannot segment, and pretending otherwise is how the
        old hardcoded default stayed live and invisible.

        Args:
            image_base64: Base64-encoded image (no data URI prefix)

        Returns:
            List of zone dicts: label, material_type, finish, dominant_color, bbox, confidence
        """
        start = time.time()
        prompt = await self._get_prompt()

        if not self.anthropic_api_key:
            raise RuntimeError("Segmentation backend not configured — set ANTHROPIC_API_KEY")

        zones = await self._segment_with_anthropic(image_base64, prompt)
        elapsed = round((time.time() - start) * 1000)
        logger.info(f"✅ Segmentation (Anthropic): {len(zones)} zones in {elapsed}ms")
        return zones

    @staticmethod
    def _detect_media_type(image_base64: str) -> str:
        """Detect image media type from magic bytes in base64 data."""
        import base64 as _b64
        try:
            header = _b64.b64decode(image_base64[:24] + "==")[:12]
            if header[:3] == b"\xff\xd8\xff":
                return "image/jpeg"
            if header[:4] == b"\x89PNG":
                return "image/png"
            if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                return "image/webp"
            if header[:4] in (b"GIF8", b"GIF9"):
                return "image/gif"
        except Exception:
            pass
        return "image/jpeg"  # safe fallback

    #: Forced-tool schema (#32 + #33 item 2). This call was BOTH a raw Anthropic POST
    #: (so an OPUS segmentation logged no cost anywhere) and a markdown-fence stripper
    #: with a truncation-recovery parser underneath it. The recovery code is the tell:
    #: it exists because the model ran out of max_tokens mid-array and someone had to
    #: rebuild the JSON by walking brace depth. A forced tool removes the need to guess.
    #:
    #: Only `bbox` is required per zone — `_validate_zones` fills every other default,
    #: and requiring them would push the model into inventing labels for a zone it is
    #: unsure about.
    SEGMENTATION_TOOL = {
        "name": "emit_material_zones",
        "description": "Return the material zones detected in this image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bbox": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"}, "y": {"type": "number"},
                                    "w": {"type": "number"}, "h": {"type": "number"},
                                },
                                "required": ["x", "y", "w", "h"],
                            },
                            "label": {"type": "string"},
                            "material_type": {"type": "string"},
                            "finish": {"type": "string"},
                            "dominant_color": {"type": "string"},
                            "zone_intent": {
                                "type": "string",
                                "enum": ["surface", "full_object", "upholstery", "sub_element"],
                            },
                            "search_query": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["bbox"],
                    },
                },
            },
            "required": ["zones"],
        },
    }

    async def _segment_with_anthropic(self, image_base64: str, prompt: str) -> List[Dict[str, Any]]:
        """Call Anthropic claude-opus-4-8 for segmentation, schema-locked via tool_use."""
        from app.services.core.claude_tool_call import call_with_tool, ToolCallNotReturned

        media_type = self._detect_media_type(image_base64)
        try:
            result = await call_with_tool(
                task="image_segmentation",
                model="claude-opus-4-8",
                max_tokens=16384,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                tool=self.SEGMENTATION_TOOL,
            )
        except ToolCallNotReturned as e:
            logger.warning(f"segmentation returned no usable tool block: {e}")
            return []
        return self._validate_zones(result.data.get("zones") or [])

    def _validate_zones(self, zones: List[Any]) -> List[Dict[str, Any]]:
        """Clamp and default the zones a forced tool call returned.

        This is what survives of `_parse_zones`. The fence-stripping and the
        brace-walking truncation recovery are gone with the free-form contract that
        needed them — but the clamping stays: a schema can say `number`, it cannot say
        "between 0 and 1", and a bbox outside the image is still a bug.
        """
        validated = []
        for i, zone in enumerate(zones):
            if not isinstance(zone, dict):
                continue
            bbox = zone.get("bbox", {})
            if not all(k in bbox for k in ("x", "y", "w", "h")):
                logger.debug(f"Zone {i} skipped — invalid bbox: {bbox}")
                continue
            # Clamp bbox to [0, 1]
            zone["bbox"] = {
                "x": max(0.0, min(1.0, float(bbox["x"]))),
                "y": max(0.0, min(1.0, float(bbox["y"]))),
                "w": max(0.01, min(1.0, float(bbox["w"]))),
                "h": max(0.01, min(1.0, float(bbox["h"]))),
            }
            zone["confidence"] = max(0.0, min(1.0, float(zone.get("confidence", 0.5))))
            zone.setdefault("label", f"zone_{i}")
            zone.setdefault("material_type", "unknown")
            zone.setdefault("finish", "unknown")
            zone.setdefault("dominant_color", "#888888")
            zone.setdefault("zone_intent", "surface")
            # Validate zone_intent value
            if zone["zone_intent"] not in ("surface", "full_object", "upholstery", "sub_element"):
                zone["zone_intent"] = "surface"
            # search_query is optional — frontend falls back to material_type + finish if absent
            zone.setdefault("search_query", "")
            validated.append(zone)

        return validated


_instance: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    global _instance
    if _instance is None:
        _instance = SegmentationService()
    return _instance
