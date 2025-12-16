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


async def generate_with_replicate(model: dict, prompt: str, width: int, height: int, image_url: Optional[str], api_token: str, max_retries: int = 3) -> str:
    """Generate image using Replicate API with retry logic"""

    for attempt in range(max_retries):
        try:
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

            # Determine model identifier
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

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"⚠️ [{model['name']}] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ [{model['name']}] All {max_retries} attempts failed")
                raise


async def atomic_update_model_result(job_id: str, model_id: str, success: bool, image_url: Optional[str] = None, error: Optional[str] = None):
    """Atomically update a single model result using PostgreSQL RPC"""
    model_result = {
        "model_id": model_id,
        "status": "completed" if success else "failed",
        "generated_at": datetime.utcnow().isoformat()
    }

    if success and image_url:
        model_result["image_urls"] = [image_url]
    elif not success and error:
        model_result["error"] = error
        model_result["image_urls"] = []

    async with get_db_connection() as conn:
        try:
            # Try using PostgreSQL RPC function for atomic update
            result = await conn.fetchrow(
                "SELECT * FROM update_model_result($1, $2, $3)",
                job_id, model_id, json.dumps(model_result)
            )
            if result:
                print(f"📊 Progress: {result['progress_percentage']}% ({result['completed_models']}/{result['total_models']} models)")
        except Exception as e:
            # Fallback to direct update if RPC doesn't exist
            if "does not exist" in str(e):
                print(f"⚠️  RPC function not found, using fallback")
                await atomic_update_model_result_fallback(job_id, model_id, success, image_url, error)
            else:
                raise


async def atomic_update_model_result_fallback(job_id: str, model_id: str, success: bool, image_url: Optional[str] = None, error: Optional[str] = None):
    """Fallback method - still has race condition but better than nothing"""
    async with get_db_connection() as conn:
        row = await conn.fetchrow("SELECT models_results, models_queue FROM generation_3d WHERE id = $1", job_id)

        if not row:
            return

        models_results = row['models_results'] if row['models_results'] else {}
        models_queue = row['models_queue'] if row['models_queue'] else []

        # Update result
        model_result = {
            "model_id": model_id,
            "status": "completed" if success else "failed",
            "generated_at": datetime.utcnow().isoformat()
        }

        if success and image_url:
            model_result["image_urls"] = [image_url]
        elif not success and error:
            model_result["error"] = error
            model_result["image_urls"] = []

        models_results[model_id] = model_result

        # Calculate progress
        total = len(models_queue)
        completed = len(models_results)
        progress = int((completed / total) * 100) if total > 0 else 0

        await conn.execute(
            """UPDATE generation_3d
               SET models_results = $1, progress_percentage = $2,
                   generation_status = $3, updated_at = NOW(),
                   completed_at = CASE WHEN $3 = 'completed' THEN NOW() ELSE completed_at END
               WHERE id = $4""",
            json.dumps(models_results), progress,
            'completed' if progress >= 100 else 'processing', job_id
        )


async def process_generation_background(job_id: str, request: InteriorRequest, models_to_use: List[dict], enhanced_prompt: str):
    """Background task with parallel processing, retry logic, timeout, and error handling"""

    try:
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if not replicate_token:
            async with get_db_connection() as conn:
                await conn.execute(
                    "UPDATE generation_3d SET generation_status = 'failed', error_message = $1 WHERE id = $2",
                    "REPLICATE_API_TOKEN not configured", job_id
                )
            return

        # Semaphore to limit concurrent requests (3 at a time)
        semaphore = asyncio.Semaphore(3)

        async def process_one_model(model: dict):
            async with semaphore:
                try:
                    print(f"🎨 Starting generation for {model['name']}")
                    image_url = await generate_with_replicate(
                        model, enhanced_prompt, request.width, request.height,
                        request.image, replicate_token, max_retries=3
                    )
                    print(f"✅ {model['name']} completed successfully")
                    await atomic_update_model_result(job_id, model['id'], True, image_url)
                except Exception as e:
                    print(f"❌ {model['name']} failed: {e}")
                    await atomic_update_model_result(job_id, model['id'], False, error=str(e))

        # Process all models in parallel with 10-minute timeout
        tasks = [process_one_model(m) for m in models_to_use]
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=600  # 10 minutes
        )

        print(f"✅ Job {job_id} completed")

    except asyncio.TimeoutError:
        print(f"⏰ Job {job_id} timed out after 10 minutes")
        async with get_db_connection() as conn:
            await conn.execute(
                "UPDATE generation_3d SET generation_status = 'failed', error_message = $1, completed_at = NOW() WHERE id = $2",
                "Job timed out after 10 minutes", job_id
            )
    except Exception as e:
        print(f"❌ FATAL ERROR in job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        async with get_db_connection() as conn:
            await conn.execute(
                "UPDATE generation_3d SET generation_status = 'failed', error_message = $1, completed_at = NOW() WHERE id = $2",
                f"Background task error: {str(e)}", job_id
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

    # Prepare models_queue (list of model IDs)
    models_queue = [{"id": m["id"], "name": m["name"], "provider": m["provider"]} for m in models_to_use]

    # Determine request type
    request_type = 'image-to-image' if request.image else 'text-to-image'

    async with get_db_connection() as conn:
        await conn.execute(
            """INSERT INTO generation_3d (
                id, user_id, workspace_id, prompt, room_type, style,
                generation_status, progress_percentage, request_type,
                models_queue, models_results, workflow_status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
            job_id,
            request.user_id,
            request.workspace_id,
            enhanced_prompt,
            request.room_type,
            request.style,
            'processing',
            0,
            request_type,
            json.dumps(models_queue),  # models_queue as JSONB array
            json.dumps({}),  # models_results starts empty (dict)
            'generating'  # workflow_status
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

