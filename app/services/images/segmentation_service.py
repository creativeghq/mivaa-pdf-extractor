"""
Material Segmentation Service

Detects distinct material zones in 3D rendered images using Qwen3-VL.
Returns bounding boxes + metadata per zone so the frontend can crop and
send each crop to the existing RAG image search endpoint.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.core.ai_client_service import get_ai_client_service
from app.services.embeddings.qwen_endpoint_manager import QwenEndpointManager

logger = logging.getLogger(__name__)

SEGMENT_PROMPT = """You are a material detection expert analyzing a 3D architectural rendering.

Identify every distinct material surface in this image (floor, wall, ceiling, countertop, furniture upholstery, curtains, cabinet doors, etc.).

For EACH surface return a JSON object. Respond with ONLY a valid JSON array — no explanation, no markdown, no code fences.

Rules:
- bbox coordinates are RELATIVE (0.0–1.0), where (0,0) is top-left and (1,1) is bottom-right
- Skip surfaces smaller than 3% of the image area
- Use specific material names (e.g. "oak wood" not just "wood", "Carrara marble" not just "stone")
- confidence: your certainty that the material identification is correct (0.0–1.0)

Required fields per zone:
{
  "label": "descriptive surface name, e.g. floor, left wall, kitchen island countertop",
  "material_type": "specific material, e.g. polished concrete, brushed brass, linen",
  "finish": "surface finish, e.g. matte, glossy, satin, textured, rough",
  "dominant_color": "#rrggbb hex of the most prominent color",
  "bbox": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.3},
  "confidence": 0.85
}

Return ONLY the JSON array. Example format:
[
  {"label": "floor", "material_type": "white oak hardwood", "finish": "satin", "dominant_color": "#c8a97a", "bbox": {"x": 0.0, "y": 0.6, "w": 1.0, "h": 0.4}, "confidence": 0.93},
  {"label": "back wall", "material_type": "plaster", "finish": "matte", "dominant_color": "#f5f0eb", "bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.6}, "confidence": 0.88}
]"""


class SegmentationService:
    """Detects material zones in 3D renders via Claude Haiku (primary) or Qwen3-VL (fallback)."""

    def __init__(self):
        import os
        settings = get_settings()
        qwen_config = settings.get_qwen_config()

        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.qwen_endpoint_url: str = qwen_config["endpoint_url"]
        self.qwen_endpoint_token: str = qwen_config["endpoint_token"]
        self.qwen_manager = QwenEndpointManager(
            endpoint_url=self.qwen_endpoint_url,
            endpoint_name=qwen_config["endpoint_name"],
            namespace=qwen_config["namespace"],
            endpoint_token=self.qwen_endpoint_token,
            enabled=qwen_config["enabled"],
        )

    async def segment_image(self, image_base64: str) -> List[Dict[str, Any]]:
        """
        Detect material zones in a 3D render.

        Uses Claude Haiku as primary (always available, instant).
        Falls back to HF Qwen3-VL if Anthropic fails.

        Args:
            image_base64: Base64-encoded image (no data URI prefix)

        Returns:
            List of zone dicts: label, material_type, finish, dominant_color, bbox, confidence
        """
        start = time.time()

        # Primary: Anthropic claude-haiku — always available, no warmup delay
        if self.anthropic_api_key:
            try:
                zones = await self._segment_with_anthropic(image_base64)
                elapsed = round((time.time() - start) * 1000)
                logger.info(f"✅ Segmentation (Anthropic): {len(zones)} zones in {elapsed}ms")
                return zones
            except Exception as e:
                logger.warning(f"Anthropic segmentation failed, trying Qwen fallback: {e}")

        # Fallback: HF Qwen3-VL (only if endpoint is already running — no blocking resume)
        if self.qwen_endpoint_token:
            try:
                zones = await self._segment_with_qwen(image_base64)
                elapsed = round((time.time() - start) * 1000)
                logger.info(f"✅ Segmentation (Qwen): {len(zones)} zones in {elapsed}ms")
                return zones
            except Exception as e:
                logger.error(f"Qwen segmentation also failed: {e}")
                raise

        raise RuntimeError("No segmentation backend available — configure ANTHROPIC_API_KEY")

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

    async def _segment_with_anthropic(self, image_base64: str) -> List[Dict[str, Any]]:
        """Call Anthropic claude-haiku-4-5 for segmentation."""
        import httpx
        media_type = self._detect_media_type(image_base64)
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 2048,
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
                                {"type": "text", "text": SEGMENT_PROMPT},
                            ],
                        }
                    ],
                },
            )
            resp.raise_for_status()
            content = resp.json()["content"][0]["text"].strip()
            return self._parse_zones(content)

    async def _segment_with_qwen(self, image_base64: str) -> List[Dict[str, Any]]:
        """Call HF Qwen3-VL endpoint (only if already running — no blocking resume)."""
        ai_service = get_ai_client_service()
        response = await ai_service.httpx.post(
            self.qwen_endpoint_url,
            headers={
                "Authorization": f"Bearer {self.qwen_endpoint_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SEGMENT_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return self._parse_zones(content)

    def _parse_zones(self, content: str) -> List[Dict[str, Any]]:
        """Extract and validate zone list from Qwen response."""
        # Strip markdown code fences if present
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("```").strip()

        # Find JSON array
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            logger.warning(f"No JSON array found in Qwen response: {content[:200]}")
            return []

        try:
            zones = json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e} — content: {content[:200]}")
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
            validated.append(zone)

        return validated


_instance: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    global _instance
    if _instance is None:
        _instance = SegmentationService()
    return _instance
