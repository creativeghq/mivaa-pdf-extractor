"""
Category Prototype Management API

Endpoints for populating and managing material category prototypes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime
import logging
from openai import AsyncOpenAI

from app.services.supabase_client import get_supabase_client
from app.services.ai_client_service import get_ai_client_service
from app.config import Settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/category-prototypes", tags=["Category Prototypes"])

# Initialize settings and use centralized AI client service
settings = Settings()
ai_service = get_ai_client_service()
openai_client = ai_service.openai_async

# Category prototype descriptions
CATEGORY_PROTOTYPES: Dict[str, List[str]] = {
    "ceramic_tile": [
        "Ceramic tiles with glazed surface for interior floor and wall applications",
        "Porcelain ceramic tiles with uniform texture and color patterns",
        "Glazed ceramic tiles with smooth finish for residential and commercial use"
    ],
    "porcelain_tile": [
        "High-density porcelain tiles with low water absorption for durability",
        "Porcelain stoneware tiles with through-body color and texture",
        "Vitrified porcelain tiles suitable for high-traffic areas and outdoor use"
    ],
    "marble": [
        "Natural marble stone with distinctive veining and luxurious appearance",
        "Polished marble slabs with high-gloss mirror finish",
        "Honed marble with matte smooth surface for elegant interiors"
    ],
    "granite": [
        "Hard granite stone with speckled crystalline patterns",
        "Polished granite with durable scratch-resistant surface",
        "Natural granite with unique mineral composition and color variations"
    ],
    "wood": [
        "Natural hardwood and softwood timber materials",
        "Solid wood with natural grain patterns and textures",
        "Engineered wood products and veneers"
    ],
    "glass": [
        "Transparent or translucent glass materials",
        "Tempered and laminated safety glass",
        "Decorative glass with colored or textured finishes"
    ],
    "concrete": [
        "Cement-based concrete with aggregate composition",
        "Polished or exposed aggregate concrete finishes",
        "Reinforced concrete for structural applications"
    ],
    "terrazzo": [
        "Composite terrazzo with marble chips in cement or resin matrix",
        "Polished terrazzo with decorative aggregate patterns",
        "Traditional or epoxy terrazzo flooring systems"
    ],
    "quartz": [
        "Engineered quartz surfaces with resin-bound crushed quartz",
        "Non-porous quartz countertops with consistent patterns",
        "Durable quartz composite with stain-resistant properties"
    ],
    "vinyl": [
        "Resilient vinyl flooring with printed patterns",
        "Luxury vinyl tile (LVT) with realistic textures",
        "Waterproof vinyl planks for residential and commercial use"
    ],
}


class PrototypePopulationResponse(BaseModel):
    success: bool
    message: str
    categories_processed: int
    categories_succeeded: int
    categories_failed: int
    details: List[Dict[str, Any]]


async def generate_clip_text_embedding(texts: List[str]) -> List[float]:
    """Generate CLIP text embedding by averaging embeddings of multiple descriptions"""
    try:
        # Generate embeddings for all texts at once
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            dimensions=512
        )

        # Extract embeddings
        all_embeddings = [item.embedding for item in response.data]

        # Average all embeddings
        avg_embedding = np.mean(all_embeddings, axis=0).tolist()
        return avg_embedding

    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise


async def update_category_prototype(category_key: str, descriptions: List[str]) -> Dict[str, Any]:
    """Update a category with prototype descriptions and embedding"""
    try:
        logger.info(f"Processing category: {category_key}")
        
        # Generate embedding
        embedding = await generate_clip_text_embedding(descriptions)
        
        # Update database
        supabase_client = get_supabase_client()
        result = supabase_client.client.table('material_categories').update({
            'prototype_descriptions': descriptions,
            'text_embedding_512': embedding,
            'prototype_updated_at': datetime.utcnow().isoformat()
        }).eq('category_key', category_key).execute()

        if result.data:
            logger.info(f"✅ Updated {category_key} with {len(descriptions)} descriptions")
            return {
                "category_key": category_key,
                "success": True,
                "descriptions_count": len(descriptions),
                "message": f"Successfully updated {category_key}"
            }
        else:
            logger.warning(f"Category {category_key} not found in database")
            return {
                "category_key": category_key,
                "success": False,
                "message": f"Category {category_key} not found in database"
            }
            
    except Exception as e:
        logger.error(f"Error updating {category_key}: {str(e)}")
        return {
            "category_key": category_key,
            "success": False,
            "message": f"Error: {str(e)}"
        }


@router.post("/populate", response_model=PrototypePopulationResponse)
async def populate_category_prototypes(background_tasks: BackgroundTasks):
    """
    Populate all material categories with prototype descriptions and embeddings
    
    This endpoint:
    1. Generates CLIP text embeddings for predefined category descriptions
    2. Updates material_categories table with prototypes
    3. Returns summary of operation
    """
    logger.info("🚀 Starting category prototype population...")
    
    details = []
    success_count = 0
    fail_count = 0
    
    for category_key, descriptions in CATEGORY_PROTOTYPES.items():
        result = await update_category_prototype(category_key, descriptions)
        details.append(result)
        
        if result['success']:
            success_count += 1
        else:
            fail_count += 1
    
    return PrototypePopulationResponse(
        success=success_count > 0,
        message=f"Processed {len(CATEGORY_PROTOTYPES)} categories: {success_count} succeeded, {fail_count} failed",
        categories_processed=len(CATEGORY_PROTOTYPES),
        categories_succeeded=success_count,
        categories_failed=fail_count,
        details=details
    )


@router.get("/verify")
async def verify_prototypes():
    """
    Verify that prototypes were populated correctly
    """
    try:
        supabase_client = get_supabase_client()
        result = supabase_client.client.table('material_categories').select(
            'category_key, prototype_descriptions, prototype_updated_at'
        ).not_.is_('text_embedding_512', 'null').execute()

        if result.data:
            return {
                "success": True,
                "count": len(result.data),
                "categories": [
                    {
                        "category_key": item['category_key'],
                        "descriptions_count": len(item['prototype_descriptions']),
                        "updated_at": item['prototype_updated_at']
                    }
                    for item in result.data
                ]
            }
        else:
            return {
                "success": False,
                "count": 0,
                "message": "No categories with prototypes found"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

