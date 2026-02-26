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
    """Detects material zones in 3D renders via Qwen3-VL."""

    def __init__(self):
        settings = get_settings()
        qwen_config = settings.get_qwen_config()

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

        Args:
            image_base64: Base64-encoded image (no data URI prefix)

        Returns:
            List of zone dicts: label, material_type, finish, dominant_color, bbox, confidence
        """
        start = time.time()

        if not self.qwen_endpoint_token:
            raise ValueError("HUGGINGFACE_API_KEY not configured — cannot call Qwen3-VL")

        # Resume endpoint if paused
        if not self.qwen_manager.resume_if_needed():
            raise RuntimeError("Qwen endpoint unavailable — cannot perform segmentation")

        ai_service = get_ai_client_service()
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
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
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_base64}"
                                        },
                                    },
                                ],
                            }
                        ],
                        "max_tokens": 2048,
                        "temperature": 0.1,
                        "top_p": 0.9,
                    },
                )

                if response.status_code in [500, 503] and attempt < max_retries:
                    import asyncio
                    logger.warning(f"Qwen returned {response.status_code}, retry {attempt}/{max_retries}")
                    await asyncio.sleep(2 ** attempt)
                    continue

                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                zones = self._parse_zones(content)

                elapsed = round((time.time() - start) * 1000)
                logger.info(f"✅ Segmentation: {len(zones)} zones detected in {elapsed}ms")
                self.qwen_manager.mark_used()
                return zones

            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Segmentation failed after {max_retries} attempts: {e}")
                    raise
                import asyncio
                await asyncio.sleep(2 ** attempt)

        return []

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
