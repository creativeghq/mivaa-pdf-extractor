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

from app.services.ai_client_service import get_ai_client_service
from app.services.ai_call_logger import AICallLogger
from app.services.supabase_client import get_supabase_client

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
        """Build comprehensive analysis prompt for Claude Vision"""

        base_prompt = f"""Analyze this {room_type} image and provide comprehensive spatial analysis.

Room Type: {room_type}
"""

        if room_dimensions:
            base_prompt += f"""Room Dimensions: {room_dimensions['width']}m × {room_dimensions['depth']}m × {room_dimensions['height']}m
"""

        if user_preferences:
            base_prompt += f"""User Preferences: {json.dumps(user_preferences, indent=2)}
"""

        if constraints:
            base_prompt += f"""Constraints: {json.dumps(constraints, indent=2)}
"""

        # Add analysis-specific instructions
        if analysis_type == "full":
            analysis_instructions = """
Provide COMPLETE analysis including:
1. Spatial Features (windows, doors, furniture, fixtures)
2. Layout Suggestions (furniture placement, room organization)
3. Material Placements (flooring, walls, surfaces)
4. Accessibility Analysis (ADA compliance, barrier-free paths)
5. Flow Optimization (traffic patterns, bottlenecks)
"""
        elif analysis_type == "layout":
            analysis_instructions = """
Focus on LAYOUT OPTIMIZATION:
1. Spatial Features (existing furniture, fixtures)
2. Layout Suggestions (optimal furniture placement)
3. Flow Optimization (traffic patterns)
"""
        elif analysis_type == "materials":
            analysis_instructions = """
Focus on MATERIAL RECOMMENDATIONS:
1. Surface Analysis (walls, floors, ceilings)
2. Material Placements (optimal material selection)
3. Application Methods (installation recommendations)
"""
        elif analysis_type == "accessibility":
            analysis_instructions = """
Focus on ACCESSIBILITY COMPLIANCE:
1. Accessibility Features (ramps, grab bars, clearances)
2. Barrier-Free Paths (wheelchair accessibility)
3. ADA Compliance (compliance score and recommendations)
"""
        else:
            analysis_instructions = "Provide general spatial analysis."

        base_prompt += analysis_instructions

        # Add JSON response format
        base_prompt += """

Respond with VALID JSON in this exact format:
{
  "success": true,
  "spatial_features": [
    {
      "type": "window|door|furniture|fixture",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "dimensions": {"width": 0.0, "height": 0.0, "depth": 0.0},
      "importance": 0.0-1.0,
      "accessibility_rating": 0.0-1.0
    }
  ],
  "layout_suggestions": [
    {
      "item_type": "sofa|table|bed|etc",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "rotation": 0-360,
      "reasoning": "explanation",
      "confidence": 0.0-1.0,
      "alternative_positions": [{"x": 0.0, "y": 0.0, "z": 0.0}]
    }
  ],
  "material_placements": [
    {
      "material_id": "material_catalog_id",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "surface_area": 0.0,
      "application_method": "tile|paint|wallpaper|etc",
      "confidence": 0.0-1.0,
      "reasoning": "explanation"
    }
  ],
  "accessibility_analysis": {
    "compliance_score": 0.0-1.0,
    "accessibility_features": ["ramp", "grab_bar", "wide_doorway"],
    "recommendations": ["add grab bars", "widen doorway"],
    "barrier_free_paths": [
      {
        "start": {"x": 0.0, "y": 0.0},
        "end": {"x": 0.0, "y": 0.0},
        "width": 0.0
      }
    ],
    "ada_compliance": true|false
  },
  "flow_optimization": {
    "traffic_patterns": [
      {
        "path": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        "frequency": 0.0-1.0,
        "purpose": "entry|exit|circulation"
      }
    ],
    "bottlenecks": [
      {
        "position": {"x": 0.0, "y": 0.0},
        "severity": 0.0-1.0,
        "recommendation": "explanation"
      }
    ],
    "efficiency_score": 0.0-1.0,
    "suggested_improvements": ["widen pathway", "relocate furniture"]
  },
  "reasoning_explanation": "Detailed explanation of all recommendations",
  "confidence_score": 0.0-1.0
}

IMPORTANT: Respond ONLY with valid JSON. No markdown, no code blocks, just pure JSON."""

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
                if image_data.startswith("data:image"):
                    # Extract base64 part
                    image_data = image_data.split(",")[1]

                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
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
                max_tokens=8000,
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

            # Parse JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                    result = json.loads(response_text)
                else:
                    raise ValueError(f"Invalid JSON response from Claude: {response_text[:200]}")

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



