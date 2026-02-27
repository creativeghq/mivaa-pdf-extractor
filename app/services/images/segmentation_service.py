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

# Default prompt — used when no 'segmentation' agent prompt exists in the DB.
# To override: insert a row into the `prompts` table with
#   prompt_type = 'agent', category = 'segmentation', is_active = true
DEFAULT_SEGMENT_PROMPT = """You are a precision material-identification expert analyzing a 3D architectural rendering for catalog-based material matching.

Identify every distinct material surface (floor, wall, ceiling, countertop, cabinetry, upholstery, curtains, trims, tiles, rugs, decorative panels, etc.).

For EACH surface return a JSON object. Respond ONLY with a valid JSON array — no explanation, no markdown, no code fences.

Rules:
- bbox: RELATIVE coordinates (0.0–1.0), (0,0) = top-left, (1,1) = bottom-right
- Skip surfaces smaller than 2% of the image area
- material_type MUST be catalog-grade specific — this text drives database search matching:
  • "brushed stainless steel" not "metal"
  • "Calacatta marble" or "Nero Marquina marble" not just "marble"
  • "herringbone white oak engineered wood" not "wood floor"
  • "large-format porcelain tile" not "tile"
  • "bouclé wool upholstery" not "fabric"
  • "fluted oak veneer panel" not "wood panel"
  • "woven rattan" not "natural material"
- Note texture/pattern layouts when visible: chevron, herringbone, stacked bond, running bond, mosaic, grid, ribbed, fluted
- dominant_color: use the mid-tone hex (not the specular highlight or deepest shadow)
- confidence: how certain you are the material identification is correct (0.0–1.0); use lower values when lighting or render style obscures texture detail
- search_query: a single rich sentence optimised for multi-vector catalog search. It MUST cover all 6 dimensions so it matches every embedding type used to index the product catalog:
  1. MATERIAL — specific name and subtype (e.g. "Calacatta marble slab")
  2. TEXTURE — surface pattern, grain, weave or layout (e.g. "veined natural stone texture")
  3. COLOR — descriptive name + tone, not hex (e.g. "warm white with grey veining")
  4. FINISH — surface treatment (e.g. "honed matte finish")
  5. STYLE — aesthetic category (e.g. "luxury contemporary")
  6. APPLICATION — where the surface lives (e.g. "kitchen countertop surface")

Required fields per zone:
{
  "label": "descriptive surface name, e.g. kitchen island countertop, accent left wall, main floor",
  "material_type": "catalog-grade specific material name including pattern if applicable",
  "finish": "matte | glossy | satin | brushed | honed | polished | textured | rough | patinated | lacquered",
  "dominant_color": "#rrggbb",
  "bbox": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.3},
  "confidence": 0.85,
  "search_query": "rich sentence covering material, texture, color, finish, style, application"
}

Return ONLY the JSON array. Example:
[
  {"label": "main floor", "material_type": "herringbone white oak engineered wood", "finish": "satin", "dominant_color": "#c8a97a", "bbox": {"x": 0.0, "y": 0.6, "w": 1.0, "h": 0.4}, "confidence": 0.93, "search_query": "herringbone white oak engineered hardwood flooring, warm honey-brown tone, satin finish, natural wood grain with diagonal chevron pattern, Scandinavian contemporary style, residential floor surface"},
  {"label": "back wall", "material_type": "smooth white gypsum plaster", "finish": "matte", "dominant_color": "#f5f0eb", "bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.6}, "confidence": 0.88, "search_query": "smooth white gypsum plaster wall, off-white warm tone, matte finish, flat uniform texture, minimalist contemporary style, interior wall cladding"},
  {"label": "kitchen island countertop", "material_type": "Calacatta marble", "finish": "honed", "dominant_color": "#e8e4de", "bbox": {"x": 0.3, "y": 0.45, "w": 0.4, "h": 0.15}, "confidence": 0.91, "search_query": "Calacatta marble countertop slab, warm white with grey veining, honed matte finish, natural stone veined texture, luxury contemporary style, kitchen countertop surface"}
]"""


class SegmentationService:
    """Detects material zones in 3D renders via Qwen3-VL (primary) or Claude Sonnet (fallback)."""

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

    async def _get_prompt(self) -> str:
        """
        Fetch the active segmentation prompt from the `prompts` table.
        Falls back to DEFAULT_SEGMENT_PROMPT if no DB record is found.

        DB record: prompt_type='agent', category='segmentation', is_active=true
        The `content` field holds the prompt text.
        """
        try:
            from app.services.utilities.unified_prompt_service import UnifiedPromptService
            svc = UnifiedPromptService()
            prompts = await svc.get_agent_prompts(category="segmentation")
            if prompts:
                content = prompts[0].get("content") or prompts[0].get("prompt_text")
                if content:
                    logger.debug("Segmentation prompt loaded from DB")
                    return content
        except Exception as e:
            logger.debug(f"Could not load segmentation prompt from DB, using default: {e}")
        return DEFAULT_SEGMENT_PROMPT

    async def segment_image(self, image_base64: str) -> List[Dict[str, Any]]:
        """
        Detect material zones in a 3D render.

        Primary:  HF Qwen3-VL — resume_if_needed handles cold start (~60-90s first call).
        Fallback: Anthropic claude-sonnet-4-6 — always available, no warmup.

        Prompt is loaded dynamically from the `prompts` table (category='segmentation').
        Falls back to DEFAULT_SEGMENT_PROMPT if no DB record exists.

        Args:
            image_base64: Base64-encoded image (no data URI prefix)

        Returns:
            List of zone dicts: label, material_type, finish, dominant_color, bbox, confidence
        """
        import asyncio
        start = time.time()

        prompt = await self._get_prompt()

        # Primary: HF Qwen3-VL
        # resume_if_needed() is synchronous — runs in a thread to keep async loop free.
        # First call when paused: ~60-90s warmup. Subsequent calls while warm: instant.
        if self.qwen_endpoint_token:
            try:
                logger.info("Resuming Qwen endpoint if needed (may take ~60-90s on cold start)...")
                resumed = await asyncio.to_thread(self.qwen_manager.resume_if_needed)
                if resumed:
                    zones = await self._segment_with_qwen(image_base64, prompt)
                    self.qwen_manager.mark_used()
                    elapsed = round((time.time() - start) * 1000)
                    logger.info(f"✅ Segmentation (Qwen): {len(zones)} zones in {elapsed}ms")
                    return zones
                logger.warning("Qwen endpoint could not be resumed, falling back to Anthropic")
            except Exception as e:
                logger.warning(f"Qwen segmentation failed, falling back to Anthropic: {e}")

        # Fallback: Anthropic claude-sonnet-4-6
        if self.anthropic_api_key:
            try:
                zones = await self._segment_with_anthropic(image_base64, prompt)
                elapsed = round((time.time() - start) * 1000)
                logger.info(f"✅ Segmentation (Anthropic fallback): {len(zones)} zones in {elapsed}ms")
                return zones
            except Exception as e:
                logger.error(f"Anthropic segmentation also failed: {e}")
                raise

        raise RuntimeError("No segmentation backend available — configure ANTHROPIC_API_KEY or HF endpoint")

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
        """Call Anthropic claude-sonnet-4-6 for segmentation."""
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
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 4096,
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

    async def _segment_with_qwen(self, image_base64: str, prompt: str) -> List[Dict[str, Any]]:
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
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 0.9,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return self._parse_zones(content)

    def _parse_zones(self, content: str) -> List[Dict[str, Any]]:
        """Extract and validate zone list from model response."""
        # Strip markdown code fences if present
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("```").strip()

        # Find JSON array
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            logger.warning(f"No JSON array found in response: {content[:200]}")
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
