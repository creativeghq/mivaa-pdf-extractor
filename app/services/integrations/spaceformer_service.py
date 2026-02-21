"""
Spaceformer Spatial Analysis Service

AI-powered spatial reasoning using Claude Vision for room layout optimization,
material placement, and accessibility analysis.
"""

import logging
import time
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.services.core.ai_client_service import get_ai_client_service
from app.services.core.ai_call_logger import AICallLogger
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class SpaceformerService:
    """Service for AI-powered spatial analysis using Claude Vision"""

    def __init__(self):
        """Initialize Spaceformer service"""
        self.ai_service = get_ai_client_service()
        self.ai_logger = AICallLogger()
        self.supabase = get_supabase_client()

    async def analyze_space(
        self,
        image_url: Optional[str] = None,
        image_data: Optional[str] = None,
        room_type: str = "general",
        room_dimensions: Optional[Dict[str, float]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        analysis_type: str = "full",
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive spatial analysis using Claude Vision.

        Args:
            image_url: URL of room image
            image_data: Base64 encoded image data
            room_type: Type of room (bedroom, kitchen, bathroom, etc.)
            room_dimensions: Room dimensions {width, height, depth}
            user_preferences: User preferences for analysis
            constraints: Analysis constraints
            analysis_type: Type of analysis (full, layout, materials, accessibility)
            user_id: User ID for logging
            workspace_id: Workspace ID for logging

        Returns:
            Dict containing comprehensive spatial analysis results
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())

        try:
            logger.info(f"Starting spatial analysis {analysis_id} for room_type: {room_type}")

            # Build analysis prompt based on analysis type
            prompt = self._build_analysis_prompt(
                room_type=room_type,
                room_dimensions=room_dimensions,
                user_preferences=user_preferences,
                constraints=constraints,
                analysis_type=analysis_type
            )

            # Call Claude Vision API
            result = await self._analyze_with_claude(
                image_url=image_url,
                image_data=image_data,
                prompt=prompt,
                analysis_id=analysis_id,
                user_id=user_id
            )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            result["processing_time_ms"] = processing_time_ms
            result["analysis_id"] = analysis_id

            # Save analysis to database
            await self._save_analysis(
                analysis_id=analysis_id,
                user_id=user_id,
                workspace_id=workspace_id,
                room_type=room_type,
                analysis_type=analysis_type,
                result=result
            )

            logger.info(f"Spatial analysis {analysis_id} completed in {processing_time_ms}ms")
            return result

        except Exception as e:
            logger.error(f"Spatial analysis {analysis_id} failed: {e}")
            raise

    def _build_analysis_prompt(
        self,
        room_type: str,
        room_dimensions: Optional[Dict[str, float]],
        user_preferences: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]],
        analysis_type: str
    ) -> str:
        """Build comprehensive per-item analysis prompt for Claude Vision"""

        base_prompt = f"""You are an expert interior design analyst with deep knowledge of furniture, materials, dimensions, and product sourcing. Analyze this room image with extreme detail.

Room Type: {room_type}
"""

        if room_dimensions:
            base_prompt += f"""Known Room Dimensions: {room_dimensions['width']}m wide × {room_dimensions['depth']}m deep × {room_dimensions['height']}m high
"""

        if user_preferences:
            base_prompt += f"""User Preferences: {json.dumps(user_preferences, indent=2)}
"""

        base_prompt += """
Your task is to produce TWO levels of analysis:

LEVEL 1 — ROOM-LEVEL ANALYSIS
Analyse the overall room: its type, estimated dimensions, architectural style, surfaces (floor, walls, ceiling), lighting, and condition.

LEVEL 2 — ITEM-BY-ITEM INVENTORY
Identify and analyse EVERY visible item in the room individually — every piece of furniture, every light fixture, every decorative object, every rug, every plant, every cushion, every artwork. Do not group items. List each one separately.

For each item provide:
- Precise name (e.g. "3-seat sofa", not just "sofa")
- Category + subcategory
- Where it sits in the room (plain English description)
- Estimated real-world dimensions in centimetres (width × height × depth). Use visual cues, proportions, standard sizes as reference. State your confidence (low/medium/high).
- Primary colour and any secondary colours
- Material and finish (e.g. "solid oak with matte lacquer", "chrome-plated steel", "linen upholstery")
- Style (e.g. mid-century modern, Scandinavian, industrial, contemporary, traditional)
- Visible condition (excellent / good / fair / worn)
- Quantity (if multiple identical items)
- 5–8 product search keywords optimised for finding this exact item in a product catalogue
- Any notable features, brand clues, or distinguishing details

"""

        base_prompt += """
Respond with VALID JSON in this exact format:
{
  "success": true,
  "room_analysis": {
    "room_type": "living room",
    "estimated_dimensions": {
      "width_m": 5.5,
      "length_m": 7.0,
      "height_m": 2.7,
      "total_area_sqm": 38.5
    },
    "architectural_style": "contemporary",
    "lighting": {
      "natural_light": "high|medium|low",
      "artificial_light": "description of light fixtures",
      "light_direction": "e.g. south-facing, north-east"
    },
    "flooring": {
      "type": "hardwood|tile|carpet|concrete|laminate|vinyl|stone|other",
      "color": "color description",
      "pattern": "e.g. straight planks, herringbone, solid, geometric",
      "condition": "excellent|good|fair|worn",
      "estimated_area_sqm": 0.0
    },
    "walls": {
      "primary_color": "color",
      "material": "painted drywall|brick|concrete|wood panelling|wallpaper|stone|other",
      "finish": "matte|satin|gloss|textured|other",
      "special_features": ["e.g. feature wall", "built-in shelving", "wainscoting"]
    },
    "ceiling": {
      "height_m": 0.0,
      "type": "flat|vaulted|coffered|exposed beams|other",
      "color": "color",
      "special_features": []
    },
    "windows": {
      "count": 0,
      "type": "description",
      "estimated_total_area_sqm": 0.0,
      "glazing": "single|double|unknown"
    },
    "doors": {
      "count": 0,
      "types": ["hinged", "sliding", "pocket"]
    },
    "overall_condition": "excellent|good|fair|poor",
    "crowding_level": "sparse|balanced|moderate|cluttered",
    "color_palette": ["dominant color 1", "color 2", "color 3", "accent color"],
    "style_summary": "brief overall style description"
  },
  "detected_items": [
    {
      "id": "item_001",
      "name": "3-seat sofa",
      "category": "furniture",
      "subcategory": "seating",
      "position_description": "centre of room facing the TV wall",
      "estimated_dimensions": {
        "width_cm": 220,
        "height_cm": 85,
        "depth_cm": 95,
        "seat_height_cm": 42,
        "dimension_confidence": "medium"
      },
      "appearance": {
        "primary_color": "light grey",
        "secondary_colors": ["charcoal piping"],
        "material": "woven fabric upholstery",
        "frame_material": "solid wood",
        "finish": "matte",
        "pattern": "solid",
        "texture": "smooth weave"
      },
      "style": "contemporary",
      "condition": "excellent",
      "quantity": 1,
      "confidence_score": 0.92,
      "product_search_keywords": [
        "3-seat sofa light grey fabric",
        "contemporary fabric sofa 220cm",
        "low profile modern sofa",
        "grey woven upholstery sofa",
        "Scandinavian living room sofa"
      ],
      "notable_features": "Low profile with clean lines, solid wooden legs, appears to have removable cushion covers",
      "estimated_price_range": "mid-range"
    }
  ],
  "spatial_features": [
    {
      "type": "window|door|structural_column|fireplace|built-in",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "dimensions": {"width": 0.0, "height": 0.0, "depth": 0.0},
      "importance": 0.0,
      "accessibility_rating": 0.0
    }
  ],
  "layout_suggestions": [
    {
      "item_type": "description",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "rotation": 0,
      "reasoning": "explanation",
      "confidence": 0.0,
      "alternative_positions": []
    }
  ],
  "material_placements": [
    {
      "surface": "floor|wall|ceiling|specific surface name",
      "current_material": "what is there now",
      "recommended_material": "recommended replacement or enhancement",
      "surface_area": 0.0,
      "application_method": "tile|paint|wallpaper|plank|panel|other",
      "confidence": 0.0,
      "reasoning": "why this recommendation"
    }
  ],
  "accessibility_analysis": {
    "compliance_score": 0.0,
    "accessibility_features": [],
    "recommendations": [],
    "barrier_free_paths": [],
    "ada_compliance": false
  },
  "flow_optimization": {
    "traffic_patterns": [],
    "bottlenecks": [],
    "efficiency_score": 0.0,
    "suggested_improvements": []
  },
  "reasoning_explanation": "Comprehensive explanation of the room and all findings",
  "confidence_score": 0.0
}

CRITICAL RULES:
- detected_items MUST include every single visible item. Aim for completeness — if you can see it, list it.
- Dimensions must be realistic estimates in centimetres, not placeholder zeros.
- product_search_keywords must be specific enough to find this exact product in a catalogue.
- Respond ONLY with valid JSON. No markdown, no code blocks, just pure JSON."""

        return base_prompt

    async def _analyze_with_claude(
        self,
        image_url: Optional[str],
        image_data: Optional[str],
        prompt: str,
        analysis_id: str,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze image with Claude Vision"""
        start_time = time.time()

        try:
            client = self.ai_service.anthropic

            # Build content array
            content = []

            # Add image (URL or base64)
            if image_url:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image_url
                    }
                })
            elif image_data:
                # Handle base64 data
                detected_media_type = "image/jpeg"
                if image_data.startswith("data:image"):
                    # Extract media type and base64 data from data URL
                    header, image_data = image_data.split(",", 1)
                    detected_media_type = header.split(":")[1].split(";")[0]

                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": detected_media_type,
                        "data": image_data
                    }
                })

            # Add prompt
            content.append({
                "type": "text",
                "text": prompt
            })

            logger.info(f"Calling Claude Vision for spatial analysis {analysis_id}")

            # Call Claude API
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )

            # Parse response
            response_text = response.content[0].text

            # Log AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_claude_call(
                task="spatial_analysis",
                model="claude-sonnet-4-20250514",
                response=response,
                latency_ms=latency_ms,
                confidence_score=0.9,
                confidence_breakdown={},
                action="use_ai_result",
                job_id=analysis_id,
                user_id=user_id
            )

            # Parse JSON response — Claude sometimes wraps in markdown despite instructions
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                import re as _re
                result = None

                # 1. Try ```json ... ``` or ``` ... ``` code blocks
                code_block = _re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
                if code_block:
                    try:
                        result = json.loads(code_block.group(1).strip())
                        logger.debug("Extracted JSON from markdown code block")
                    except json.JSONDecodeError:
                        pass

                # 2. Last resort: find the outermost { ... } object in the full text
                if result is None:
                    obj_match = _re.search(r'\{[\s\S]*\}', response_text)
                    if obj_match:
                        try:
                            result = json.loads(obj_match.group(0))
                            logger.debug("Extracted JSON object from mixed-text response")
                        except json.JSONDecodeError:
                            pass

                if result is None:
                    logger.error(f"Claude returned non-JSON. Preview: {response_text[:300]}")
                    raise RuntimeError(f"AI returned non-JSON response: {response_text[:200]}")

            logger.info(f"Claude Vision analysis completed for {analysis_id}")
            return result

        except Exception as e:
            logger.error(f"Claude Vision analysis failed: {e}")
            raise

    async def _save_analysis(
        self,
        analysis_id: str,
        user_id: Optional[str],
        workspace_id: Optional[str],
        room_type: str,
        analysis_type: str,
        result: Dict[str, Any]
    ) -> None:
        """Save analysis results to database"""
        try:
            # Save to spatial_analysis table
            data = {
                "id": analysis_id,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "room_type": room_type,
                "analysis_type": analysis_type,
                "room_analysis": result.get("room_analysis", {}),
                "detected_items": result.get("detected_items", []),
                "spatial_features": result.get("spatial_features", []),
                "layout_suggestions": result.get("layout_suggestions", []),
                "material_placements": result.get("material_placements", []),
                "accessibility_analysis": result.get("accessibility_analysis", {}),
                "flow_optimization": result.get("flow_optimization", {}),
                "reasoning_explanation": result.get("reasoning_explanation", ""),
                "confidence_score": result.get("confidence_score", 0.0),
                "processing_time_ms": result.get("processing_time_ms", 0),
                "created_at": datetime.utcnow().isoformat()
            }

            response = self.supabase.client.table("spatial_analysis").insert(data).execute()
            logger.info(f"Saved spatial analysis {analysis_id} to database")

        except Exception as e:
            logger.warning(f"Failed to save spatial analysis to database: {e}")
            # Don't fail the request if database save fails


# Dependency injection
def get_spaceformer_service() -> SpaceformerService:
    """Get Spaceformer service instance"""
    return SpaceformerService()



