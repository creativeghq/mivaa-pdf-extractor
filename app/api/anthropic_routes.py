"""
Anthropic Claude Integration API endpoints.

This module provides image validation and product enrichment using Anthropic Claude:
- Image validation with Claude 3.5 Sonnet Vision
- Product enrichment with Claude 3.5 Sonnet
- Batch processing for both operations
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..services.supabase_client import get_supabase_client, SupabaseClient
from ..config import get_settings
import anthropic

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/anthropic", tags=["Anthropic Claude"])

# Get settings
settings = get_settings()

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


# ============================================================================
# Request/Response Models
# ============================================================================

class ImageValidationRequest(BaseModel):
    image_id: str
    image_url: str
    product_groups: Optional[List[str]] = Field(default=None)
    workspace_id: str


class ImageValidationResponse(BaseModel):
    image_id: str
    validation_status: str  # valid, invalid, needs_review
    quality_score: float
    product_associations: List[Dict[str, Any]]
    issues: List[str]
    recommendations: List[str]
    processing_time_ms: float


class ProductEnrichmentRequest(BaseModel):
    chunk_id: str
    chunk_content: str
    workspace_id: str


class ProductEnrichmentResponse(BaseModel):
    chunk_id: str
    enrichment_status: str  # enriched, partial, failed
    product_name: str
    product_category: str
    product_description: str
    specifications: Dict[str, Any]
    related_products: List[str]
    confidence_score: float
    processing_time_ms: float


# ============================================================================
# Image Validation Endpoints
# ============================================================================

@router.post("/images/validate", response_model=ImageValidationResponse)
async def validate_image_with_claude(
    request: ImageValidationRequest,
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> ImageValidationResponse:
    """
    Validate an image using Claude 3.5 Sonnet Vision.
    
    Analyzes image quality, content, and matches to product groups.
    """
    try:
        import time
        start_time = time.time()

        # Build Claude Vision prompt
        product_groups_context = ""
        if request.product_groups and len(request.product_groups) > 0:
            product_groups_context = f"\n\nProduct Groups to match against:\n" + "\n".join(
                [f"- {g}" for g in request.product_groups]
            )

        prompt = f"""You are an expert material and product analyst. Analyze this image and provide:

1. **Quality Assessment**: Rate the image quality (0-1 scale) considering clarity, lighting, composition
2. **Content Analysis**: Describe what you see in the image
3. **Material Identification**: Identify any materials visible
4. **Product Associations**: Match the image to relevant product groups{product_groups_context}
5. **Issues**: List any quality issues or concerns
6. **Recommendations**: Suggest improvements for better product matching

Respond in JSON format:
{{
  "quality_score": <number 0-1>,
  "content_description": "<description>",
  "materials_identified": ["<material1>", "<material2>"],
  "product_associations": [
    {{
      "product_group": "<group>",
      "confidence": <0-1>,
      "reasoning": "<why this matches>"
    }}
  ],
  "issues": ["<issue1>", "<issue2>"],
  "recommendations": ["<recommendation1>", "<recommendation2>"]
}}"""

        # Call Claude Vision API
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": request.image_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        # Parse response
        response_text = response.content[0].text
        import json
        analysis_result = json.loads(response_text)

        # Determine validation status
        quality_score = analysis_result.get("quality_score", 0)
        validation_status = (
            "valid" if quality_score >= 0.7
            else "needs_review" if quality_score >= 0.5
            else "invalid"
        )

        # Store validation result
        validation_record = {
            "image_id": request.image_id,
            "workspace_id": request.workspace_id,
            "validation_status": validation_status,
            "quality_score": quality_score,
            "issues": analysis_result.get("issues", []),
            "recommendations": analysis_result.get("recommendations", []),
            "validated_at": datetime.utcnow().isoformat(),
            "metadata": {
                "content_description": analysis_result.get("content_description"),
                "materials_identified": analysis_result.get("materials_identified"),
                "model_used": "claude-3-5-sonnet-20241022",
            },
        }

        supabase.client.table("image_validations").insert(validation_record).execute()

        processing_time = (time.time() - start_time) * 1000

        return ImageValidationResponse(
            image_id=request.image_id,
            validation_status=validation_status,
            quality_score=quality_score,
            product_associations=analysis_result.get("product_associations", []),
            issues=analysis_result.get("issues", []),
            recommendations=analysis_result.get("recommendations", []),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image validation failed: {str(e)}"
        )


# ============================================================================
# Product Enrichment Endpoints
# ============================================================================

@router.post("/products/enrich", response_model=ProductEnrichmentResponse)
async def enrich_product_with_claude(
    request: ProductEnrichmentRequest,
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> ProductEnrichmentResponse:
    """
    Enrich product data using Claude 3.5 Sonnet.
    
    Generates descriptions, extracts specifications, and identifies related products.
    """
    try:
        import time
        start_time = time.time()

        prompt = f"""You are an expert product analyst and technical writer. Analyze this product content and provide comprehensive enrichment:

CONTENT TO ANALYZE:
{request.chunk_content}

Provide enrichment in JSON format:
{{
  "product_name": "<primary product name>",
  "product_category": "<category>",
  "product_description": "<1-2 sentence summary>",
  "specifications": {{
    "<spec_name>": "<value>",
    "<spec_name>": "<value>"
  }},
  "related_products": ["<related_product_1>", "<related_product_2>"],
  "confidence_score": <0-1>,
  "key_features": ["<feature1>", "<feature2>"],
  "use_cases": ["<use_case1>", "<use_case2>"]
}}

Focus on:
1. Accurate product identification
2. Clear, professional descriptions
3. Comprehensive specifications
4. Related products that complement this one
5. High confidence only if information is clear"""

        # Call Claude API
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        # Parse response
        response_text = response.content[0].text
        import json
        enrichment_data = json.loads(response_text)

        # Determine enrichment status
        confidence_score = enrichment_data.get("confidence_score", 0)
        enrichment_status = (
            "enriched" if confidence_score >= 0.7
            else "partial" if confidence_score >= 0.4
            else "failed"
        )

        # Store enrichment result
        enrichment_record = {
            "chunk_id": request.chunk_id,
            "workspace_id": request.workspace_id,
            "enrichment_status": enrichment_status,
            "product_name": enrichment_data.get("product_name", ""),
            "product_category": enrichment_data.get("product_category", ""),
            "product_description": enrichment_data.get("product_description", ""),
            "specifications": enrichment_data.get("specifications", {}),
            "related_products": enrichment_data.get("related_products", []),
            "confidence_score": confidence_score,
            "enrichment_score": confidence_score,
            "enriched_at": datetime.utcnow().isoformat(),
            "metadata": {
                "key_features": enrichment_data.get("key_features"),
                "use_cases": enrichment_data.get("use_cases"),
                "model_used": "claude-3-5-sonnet-20241022",
            },
        }

        supabase.client.table("product_enrichments").insert(enrichment_record).execute()

        processing_time = (time.time() - start_time) * 1000

        return ProductEnrichmentResponse(
            chunk_id=request.chunk_id,
            enrichment_status=enrichment_status,
            product_name=enrichment_data.get("product_name", ""),
            product_category=enrichment_data.get("product_category", ""),
            product_description=enrichment_data.get("product_description", ""),
            specifications=enrichment_data.get("specifications", {}),
            related_products=enrichment_data.get("related_products", []),
            confidence_score=confidence_score,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Product enrichment failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Product enrichment failed: {str(e)}"
        )


# ============================================================================
# Test Endpoints
# ============================================================================

@router.post("/test/claude-integration")
async def test_claude_integration():
    """Test Claude Vision API integration."""
    try:
        import time
        start_time = time.time()

        # Test image (1x1 pixel JPEG)
        test_image_base64 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A"

        # Build prompt for material analysis
        prompt = """Analyze this image and provide material properties in JSON format:
{
  "material_type": "<type>",
  "color": "<color>",
  "texture": "<texture>",
  "confidence": <0-1>
}"""

        # Call Claude Vision API
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": test_image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        # Parse response
        response_text = response.content[0].text
        processing_time = (time.time() - start_time) * 1000

        return {
            "success": True,
            "claude_response": response_text,
            "processing_time_ms": processing_time,
            "api_key_available": bool(settings.anthropic_api_key),
            "model_used": "claude-3-5-sonnet-20241022"
        }

    except Exception as e:
        logger.error(f"Claude integration test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "api_key_available": bool(settings.anthropic_api_key)
        }

