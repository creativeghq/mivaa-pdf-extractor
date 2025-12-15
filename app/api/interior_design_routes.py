"""
Interior Design Generation API
Generates interior design images and saves progress to database
Frontend polls database for real-time updates
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import json
import os
import httpx
from datetime import datetime
import uuid
from app.database.connection import get_db_connection

router = APIRouter(prefix="/api", tags=["interior-design"])

# Model configurations
# Text-to-Image Models (for prompts without reference images)
TEXT_TO_IMAGE_MODELS = [
    {"id": "flux-dev", "name": "FLUX.1-dev", "provider": "replicate", "capability": "text-to-image"},
    {"id": "flux-schnell", "name": "FLUX.1-schnell", "provider": "replicate", "capability": "text-to-image"},
    {"id": "sdxl", "name": "SDXL", "provider": "replicate", "capability": "text-to-image"},
    {"id": "playground-v2.5", "name": "Playground v2.5", "provider": "replicate", "capability": "text-to-image"},
    {"id": "stable-diffusion-3", "name": "Stable Diffusion 3", "provider": "replicate", "capability": "text-to-image"},
    {"id": "kandinsky-2.2", "name": "Kandinsky 2.2", "provider": "replicate", "capability": "text-to-image"},
    {"id": "proteus-v0.2", "name": "Proteus v0.2", "provider": "replicate", "capability": "text-to-image"},
]

# Image-to-Image Models (for interior design transformation with reference images)
IMAGE_TO_IMAGE_MODELS = [
    {"id": "jschoormans/comfyui-interior-remodel", "name": "ComfyUI Interior Remodel", "provider": "replicate", "capability": "image-to-image", "status": "working"},
    {"id": "julian-at/interiorly-gen1-dev", "name": "Interiorly Gen1 Dev", "provider": "replicate", "capability": "image-to-image", "status": "working"},
    {"id": "davisbrown/designer-architecture", "name": "Designer Architecture", "provider": "replicate", "capability": "image-to-image", "status": "working"},
    {"id": "erayyavuz/interior-ai", "name": "Interior AI", "provider": "replicate", "capability": "image-to-image", "status": "failing"},
    {"id": "jschoormans/interior-v2", "name": "Interior V2", "provider": "replicate", "capability": "image-to-image", "status": "failing"},
    {"id": "adirik/interior-design", "name": "Adirik Interior Design", "provider": "replicate", "capability": "image-to-image", "status": "failing"},
    {"id": "rocketdigitalai/interior-design-sdxl", "name": "Interior Design SDXL", "provider": "replicate", "capability": "image-to-image", "status": "failing"},
]

# Combined list - use all models by default
ALL_MODELS = TEXT_TO_IMAGE_MODELS + IMAGE_TO_IMAGE_MODELS

class InteriorRequest(BaseModel):
    prompt: str = Field(..., description="Interior design description")
    image: Optional[str] = Field(None, description="Reference image URL for image-to-image generation")
    room_type: Optional[str] = Field(None, description="Type of room (living_room, bedroom, etc.)")
    style: Optional[str] = Field(None, description="Design style (modern, minimalist, etc.)")
    models: Optional[List[str]] = Field(None, description="Specific model IDs to use, or None for all models")
    user_id: str = Field(..., description="User ID")
    workspace_id: Optional[str] = Field(None, description="Workspace ID")
    width: int = Field(1024, description="Image width")
    height: int = Field(1024, description="Image height")


async def generate_with_replicate(model: dict, prompt: str, width: int, height: int, image_url: Optional[str], api_token: str) -> str:
    """Generate image using Replicate API"""

    # Build input based on model capability
    input_data = {
        "prompt": prompt,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "num_outputs": 1,
    }

    # Add dimensions for text-to-image
    if model["capability"] == "text-to-image":
        input_data["width"] = width
        input_data["height"] = height

    # Add image for image-to-image
    if model["capability"] == "image-to-image" and image_url:
        input_data["image"] = image_url
        input_data["strength"] = 0.8

    # Determine model identifier (use full model ID for image-to-image, version for text-to-image)
    model_identifier = model.get("version") if model.get("version") else model["id"]

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Create prediction
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            },
            json={
                "version": model_identifier,
                "input": input_data
            }
        )
        
        if response.status_code != 201:
            raise Exception(f"Replicate API error: {response.text}")
        
        result = response.json()
        prediction_id = result["id"]
        
        # Poll for completion
        max_attempts = 60
        for _ in range(max_attempts):
            status_response = await client.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={"Authorization": f"Bearer {api_token}"}
            )
            
            if status_response.status_code == 200:
                result = status_response.json()
                
                if result["status"] == "succeeded":
                    # Extract image URL
                    output = result.get("output")
                    if isinstance(output, list):
                        return output[0]
                    return output
                
                elif result["status"] == "failed":
                    raise Exception(f"Generation failed: {result.get('error', 'Unknown error')}")
            
            await asyncio.sleep(2)
        
        raise Exception("Generation timed out")


async def process_generation_background(job_id: str, request: InteriorRequest, models_to_use: List[dict], enhanced_prompt: str):
    """Background task to process all models and update database"""

    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        # Update job with error
        async with get_db_connection() as conn:
            await conn.execute(
                "UPDATE generation_3d SET generation_status = 'failed', error_message = $1 WHERE id = $2",
                "REPLICATE_API_TOKEN not configured", job_id
            )
        return

    total = len(models_to_use)

    for idx, model in enumerate(models_to_use):
        try:
            # Generate image
            image_url = await generate_with_replicate(
                model,
                enhanced_prompt,
                request.width,
                request.height,
                request.image,
                replicate_token
            )

            # Update database with success
            progress = int((idx + 1) / total * 100)
            async with get_db_connection() as conn:
                # Get current metadata
                row = await conn.fetchrow("SELECT metadata FROM generation_3d WHERE id = $1", job_id)
                metadata = json.loads(row['metadata']) if row and row['metadata'] else {}
                models_results = metadata.get('models_results', [])

                # Update the specific model result
                for mr in models_results:
                    if mr['model_id'] == model['id']:
                        mr['status'] = 'completed'
                        mr['image_urls'] = [image_url]
                        mr['completed_at'] = datetime.utcnow().isoformat()
                        break

                # Update metadata
                metadata['models_results'] = models_results

                # Update job
                await conn.execute(
                    """UPDATE generation_3d
                       SET metadata = $1, progress_percentage = $2, updated_at = NOW()
                       WHERE id = $3""",
                    json.dumps(metadata), progress, job_id
                )

        except Exception as e:
            # Update database with error
            async with get_db_connection() as conn:
                row = await conn.fetchrow("SELECT metadata FROM generation_3d WHERE id = $1", job_id)
                metadata = json.loads(row['metadata']) if row and row['metadata'] else {}
                models_results = metadata.get('models_results', [])

                for mr in models_results:
                    if mr['model_id'] == model['id']:
                        mr['status'] = 'failed'
                        mr['error'] = str(e)
                        mr['completed_at'] = datetime.utcnow().isoformat()
                        break

                metadata['models_results'] = models_results

                await conn.execute(
                    "UPDATE generation_3d SET metadata = $1, updated_at = NOW() WHERE id = $2",
                    json.dumps(metadata), job_id
                )

    # Mark job as complete
    async with get_db_connection() as conn:
        await conn.execute(
            "UPDATE generation_3d SET generation_status = 'completed', completed_at = NOW() WHERE id = $1",
            job_id
        )


@router.post("/interior")
async def create_interior_design(request: InteriorRequest):
    """
    Generate interior design images using multiple AI models.
    Creates job in database and processes in background.
    Frontend polls database for updates.
    """
    # Determine which models to use based on request type
    if request.models:
        # User specified specific models
        models_to_use = [m for m in ALL_MODELS if m["id"] in request.models]
    elif request.image:
        # Image-to-image: Use only working image-to-image models
        models_to_use = [m for m in IMAGE_TO_IMAGE_MODELS if m.get("status") != "failing"]
    else:
        # Text-to-image: Use all text-to-image models
        models_to_use = TEXT_TO_IMAGE_MODELS

    # Enhance prompt
    enhanced_prompt = request.prompt
    if request.room_type:
        enhanced_prompt = f"{request.room_type} - {enhanced_prompt}"
    if request.style:
        enhanced_prompt = f"{request.style} style - {enhanced_prompt}"

    # Create job in database
    job_id = str(uuid.uuid4())

    # Prepare models_results structure
    models_results = [{
        "model_id": m["id"],
        "model_name": m["name"],
        "provider": m["provider"],
        "capability": m["capability"],
        "status": "pending",
        "image_urls": []
    } for m in models_to_use]

    async with get_db_connection() as conn:
        await conn.execute(
            """INSERT INTO generation_3d (
                id, user_id, workspace_id, generation_name, generation_type,
                generation_status, progress_percentage,
                input_data, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
            job_id,
            request.user_id,
            request.workspace_id,
            f"Interior Design - {request.room_type or 'general'}",
            'interior_design',
            'processing',
            0,
            json.dumps({
                "prompt": request.prompt,
                "room_type": request.room_type,
                "style": request.style,
                "enhanced_prompt": enhanced_prompt,
                "request_type": 'image-to-image' if request.image else 'text-to-image',
                "reference_image": request.image
            }),
            json.dumps({
                "models_queue": [{"id": m["id"], "name": m["name"], "provider": m["provider"]} for m in models_to_use],
                "models_results": models_results,
                "workflow_status": "generating"
            })
        )

    # Start background processing
    asyncio.create_task(process_generation_background(job_id, request, models_to_use, enhanced_prompt))

    # Return job info immediately
    return JSONResponse({
        "success": True,
        "job_id": job_id,
        "model_count": len(models_to_use),
        "models": [{"id": m["id"], "name": m["name"]} for m in models_to_use],
        "message": f"Started generating {len(models_to_use)} interior design variations"
    })

