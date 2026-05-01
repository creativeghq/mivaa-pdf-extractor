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

logger = logging.getLogger(__name__)

# Default prompt — used when no 'segmentation' agent prompt exists in the DB.
# To override: insert a row into the `prompts` table with
#   prompt_type = 'agent', category = 'segmentation', is_active = true
DEFAULT_SEGMENT_PROMPT = """You are a precision material-identification expert analyzing a 3D architectural rendering for catalog-based material matching.

Identify every distinct material surface AND structural sub-element. Be EXHAUSTIVE — this list drives the editor's "replace material" workflow, so a missing zone means the user cannot edit that element. Walk the image methodically: floor, ceiling, every wall plane, every piece of furniture (frames AND upholstery separately), every fixture, every fitting, every visible hardware element.

CATEGORIES TO COVER (non-exhaustive — emit a separate zone for each one you see):
- Architectural surfaces: floors, ceilings, walls (each visible plane separately), columns, beams, soffits, niches, skirting boards, ceiling cornices, door panels, door frames, window frames, window glass, window sills.
- Tile/stone work: wall tiles, floor tiles, mosaic feature walls, splashbacks, shower walls, tub surrounds — emit each tiled plane as its own zone even if the tile pattern is the same.
- Bathroom-specific: glass shower screens / partitions / enclosures (glass IS a material — emit it), shower trays, bathtubs, vanity tops, vanity cabinets, sink basins, toilet, bidet, mirror surfaces, mirror frames, towel rails, robe hooks, shelf supports, shower heads, taps/faucets, faucet handles, drain covers.
- Kitchen-specific: countertops, splashbacks, upper cabinets, lower cabinets, kickboards, cooker hood, hob, oven door, sink, tap, cabinet handles/pulls, open-shelf brackets, integrated appliance fronts.
- Soft furnishings: rugs, curtains, blinds, cushions, throws, headboards (frame vs fabric separately), bed bases, pillows.
- Furniture: every distinct piece — sofa, chair, table, cabinet, bookshelf — AND each visually distinct sub-component (legs, frame, doors, drawer fronts, handles, glass tops).
- Lighting: pendant lights, wall sconces, floor lamps, lamp shades, lamp bases.
- Decorative: artwork, picture frames, vases, planters, plants, books, sculptures.

DO NOT lump items together. If a wall has tile on one side and paint on another, that's TWO zones. If a vanity has a stone top and wood drawers, that's TWO zones (plus handles as a third sub_element).

Rules:
- bbox: RELATIVE coordinates (0.0–1.0), (0,0) = top-left, (1,1) = bottom-right
- Skip surfaces smaller than 1% of the image area EXCEPT sub_element / hardware / fixture zones (handles, taps, hooks, drains, shower heads can be small — always include them if visually distinct)
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

- zone_intent: classify each zone as exactly one of:
  • "surface"     — floor, wall, ceiling, countertop, backsplash, cladding, tile fields, glass shower screens, mirror surfaces — material/texture to replace
  • "full_object" — sofa, chair, rug, curtain, table, cabinet, lamp, bathtub, toilet, sink, vanity body — entire product could be swapped
  • "upholstery"  — identifiable fabric/leather cover of furniture (seat cushion, sofa back panel, headboard fabric, padded vanity bench) — separate from the frame
  • "sub_element" — table legs, chair frame/base, door handles, cabinet hardware, window frame, skirting board, shelf brackets, taps/faucets, faucet handles, shower heads, towel rails, robe hooks, drain covers, light switches, cabinet pulls — finish/colour change only

Sub-element detection rule: If legs, frame, or hardware are visually distinct from the main object body, emit them as a SEPARATE zone entry with zone_intent="sub_element". Example: a dining table with visible metal legs MUST produce TWO zones — one for the table top (full_object) and one for the table legs (sub_element). Same for a chair with a distinct wooden frame vs upholstered seat.

Required fields per zone:
{
  "label": "descriptive surface name, e.g. kitchen island countertop, accent left wall, main floor, dining table legs",
  "material_type": "catalog-grade specific material name including pattern if applicable",
  "finish": "matte | glossy | satin | brushed | honed | polished | textured | rough | patinated | lacquered",
  "dominant_color": "#rrggbb",
  "bbox": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.3},
  "confidence": 0.85,
  "zone_intent": "surface | full_object | upholstery | sub_element",
  "search_query": "rich sentence covering material, texture, color, finish, style, application"
}

Return ONLY the JSON array. Example:
[
  {"label": "main floor", "material_type": "herringbone white oak engineered wood", "finish": "satin", "dominant_color": "#c8a97a", "bbox": {"x": 0.0, "y": 0.6, "w": 1.0, "h": 0.4}, "confidence": 0.93, "zone_intent": "surface", "search_query": "herringbone white oak engineered hardwood flooring, warm honey-brown tone, satin finish, natural wood grain with diagonal chevron pattern, Scandinavian contemporary style, residential floor surface"},
  {"label": "back wall", "material_type": "smooth white gypsum plaster", "finish": "matte", "dominant_color": "#f5f0eb", "bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.6}, "confidence": 0.88, "zone_intent": "surface", "search_query": "smooth white gypsum plaster wall, off-white warm tone, matte finish, flat uniform texture, minimalist contemporary style, interior wall cladding"},
  {"label": "dining sofa", "material_type": "bouclé wool upholstery natural white", "finish": "textured", "dominant_color": "#f0ece6", "bbox": {"x": 0.1, "y": 0.3, "w": 0.5, "h": 0.4}, "confidence": 0.89, "zone_intent": "full_object", "search_query": "bouclé wool sofa upholstery, natural warm white, textured loop pile fabric, contemporary lounge seating, Scandinavian interior style"},
  {"label": "dining table top", "material_type": "Calacatta marble", "finish": "honed", "dominant_color": "#e8e4de", "bbox": {"x": 0.3, "y": 0.45, "w": 0.4, "h": 0.1}, "confidence": 0.91, "zone_intent": "full_object", "search_query": "Calacatta marble table top slab, warm white with grey veining, honed matte finish, natural stone veined texture, luxury contemporary style, dining table surface"},
  {"label": "dining table legs", "material_type": "brushed brass metal", "finish": "brushed", "dominant_color": "#c9a84c", "bbox": {"x": 0.32, "y": 0.52, "w": 0.36, "h": 0.12}, "confidence": 0.85, "zone_intent": "sub_element", "search_query": "brushed brass metal table legs, warm gold tone, brushed satin finish, cylindrical metal legs, luxury contemporary style, dining furniture hardware"},
  {"label": "shower glass screen", "material_type": "frameless tempered safety glass", "finish": "polished", "dominant_color": "#dce6ea", "bbox": {"x": 0.55, "y": 0.15, "w": 0.4, "h": 0.7}, "confidence": 0.9, "zone_intent": "surface", "search_query": "frameless tempered safety glass shower screen, transparent clear glass with subtle blue-grey tint, polished smooth finish, vertical flat panel, contemporary minimalist bathroom style, walk-in shower partition"},
  {"label": "shower wall tile field", "material_type": "large-format porcelain tile", "finish": "matte", "dominant_color": "#d6cfc4", "bbox": {"x": 0.6, "y": 0.0, "w": 0.4, "h": 0.85}, "confidence": 0.88, "zone_intent": "surface", "search_query": "large-format porcelain tile wall cladding, warm beige stone-look tone, matte natural finish, subtle veining texture in stacked-bond layout, contemporary spa bathroom style, wet-area shower wall"},
  {"label": "wall-mounted basin tap", "material_type": "brushed nickel chrome alloy", "finish": "brushed", "dominant_color": "#9aa0a4", "bbox": {"x": 0.21, "y": 0.42, "w": 0.06, "h": 0.08}, "confidence": 0.86, "zone_intent": "sub_element", "search_query": "wall-mounted basin tap, brushed nickel finish on chrome alloy body, satin brushed surface, single-lever design with rectangular spout, contemporary minimalist bathroom style, lavatory faucet hardware"}
]"""


class SegmentationService:
    """Detects material zones in 3D renders via Anthropic Claude Opus.

    Note (2026-05-01): we used to attempt Qwen3-VL on an HF endpoint as a
    speed-optimised primary, but the configured endpoint serves a text-only
    model (Qwen3.6-35B-A3B-FP8) so every Qwen call 404'd in 0.7s and fell
    through to Anthropic. Qwen has been removed entirely until/unless a real
    Qwen-VL endpoint is provisioned. Claude Opus 4.7 is the only path now,
    and quality on this task is already excellent (22 well-described zones
    on a bathroom render in ~40s with catalog-grade material names).
    """

    def __init__(self):
        import os
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

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
        Detect material zones in a 3D render via Claude Opus 4.7.

        Prompt is loaded dynamically from the `prompts` table (category='segmentation').
        Falls back to DEFAULT_SEGMENT_PROMPT if no DB record exists.

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
        """Call Anthropic claude-opus-4-7 for segmentation."""
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
                    "model": "claude-opus-4-7",
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
