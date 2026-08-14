"""
Material Segmentation Service

Detects distinct material zones in 3D rendered images using Anthropic
Claude Opus 4.7 vision. Returns bounding boxes + metadata per zone so the
frontend can crop and send each crop to the existing RAG image search
endpoint.
"""

import json
import logging
import re
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

    Claude Opus 4.7 is the only path, and quality on this task is excellent
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
        Detect material zones in a 3D render via Claude Opus 4.7.

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

    async def _segment_with_anthropic(self, image_base64: str, prompt: str) -> List[Dict[str, Any]]:
        """Call Anthropic claude-opus-4-8 for segmentation."""
        import httpx
        media_type = self._detect_media_type(image_base64)
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-opus-4-8",
                    "max_tokens": 16384,
                    "messages": [
                        {
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
                        }
                    ],
                },
            )
            resp.raise_for_status()
            content = resp.json()["content"][0]["text"].strip()
            return self._parse_zones(content)

    def _parse_zones(self, content: str) -> List[Dict[str, Any]]:
        """Extract and validate zone list from model response."""
        # Strip markdown code fences if present
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("```").strip()

        zones = self._extract_json_array(content)
        if zones is None:
            logger.warning(f"No JSON array recovered from response (len={len(content)}): {content[:300]}…{content[-200:] if len(content) > 500 else ''}")
            return []

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

    @staticmethod
    def _extract_json_array(content: str) -> Optional[List[Any]]:
        """Parse a JSON array from model output, recovering from common truncations.

        Recovery is deliberately limited to the most common failure mode: the model
        ran out of max_tokens mid-object, so the response opens with ``[`` but never
        closes. We rebuild the array by walking the string and tracking brace depth
        outside of strings, taking everything up to the last complete top-level
        ``}`` and re-wrapping with ``]``. Returns None only if no usable prefix exists.
        """
        if not content:
            return None
        start = content.find("[")
        if start < 0:
            return None

        # Fast path: well-formed array.
        match = re.search(r"\[.*\]", content[start:], re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass  # fall through to recovery

        # Recovery path: scan for the last complete top-level object in the array.
        depth = 0
        in_string = False
        escape = False
        last_complete = -1
        for i in range(start + 1, len(content)):
            ch = content[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_complete = i

        if last_complete < 0:
            return None

        recovered = content[start:last_complete + 1] + "]"
        try:
            parsed = json.loads(recovered)
            if isinstance(parsed, list):
                logger.info(f"Recovered truncated JSON array: kept {len(parsed)} complete objects (response len={len(content)})")
                return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON recovery failed: {e}")
        return None


_instance: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    global _instance
    if _instance is None:
        _instance = SegmentationService()
    return _instance
