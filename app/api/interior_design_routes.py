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
import io
from app.services.supabase_client import get_supabase_client

router = APIRouter(prefix="/api", tags=["interior-design"])

# Model configurations
# Text-to-Image Models (for prompts without reference images)
TEXT_TO_IMAGE_MODELS = [
    # Hugging Face Models (Pro plan - all models available)
    {"id": "flux-schnell", "name": "FLUX.1-schnell", "provider": "huggingface", "hf_model": "black-forest-labs/FLUX.1-schnell", "capability": "text-to-image"},
    {"id": "sdxl", "name": "SDXL", "provider": "huggingface", "hf_model": "stabilityai/stable-diffusion-xl-base-1.0", "capability": "text-to-image"},
    {"id": "sd-2.1", "name": "Stable Diffusion 2.1", "provider": "huggingface", "hf_model": "stabilityai/stable-diffusion-2-1", "capability": "text-to-image"},

    # Replicate Models - Text-to-Image
    {"id": "flux-dev", "name": "FLUX.1-dev", "provider": "replicate", "model": "black-forest-labs/flux-dev", "capability": "text-to-image"},
    {"id": "playground-v2.5", "name": "Playground v2.5", "provider": "replicate", "model": "playgroundai/playground-v2.5-1024px-aesthetic", "version": "a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24", "capability": "text-to-image"},
    {"id": "sd3", "name": "Stable Diffusion 3", "provider": "replicate", "model": "stability-ai/stable-diffusion-3", "capability": "text-to-image"},
]

# Image-to-Image Models (for interior design transformation with reference images)
IMAGE_TO_IMAGE_MODELS = [
    # Working models - recommended for production
    {"id": "comfyui-interior-remodel", "name": "ComfyUI Interior Remodel", "provider": "replicate", "model": "jschoormans/comfyui-interior-remodel", "version": "2a360362540e1f6cfe59c9db4aa8aa9059233d40e638aae0cdeb6b41f3d0dcce", "capability": "image-to-image", "status": "working"},
    {"id": "interiorly-gen1-dev", "name": "Interiorly Gen1 Dev", "provider": "replicate", "model": "julian-at/interiorly-gen1-dev", "version": "5e3080d1b308e80197b32f0ce638daa8a329d0cf42068739723d8259e44b445e", "capability": "image-to-image", "status": "working"},
    {"id": "designer-architecture", "name": "Designer Architecture", "provider": "replicate", "model": "davisbrown/designer-architecture", "version": "0d6f0893b05f14500ce03e45f54290cbffb907d14db49699f2823d0fd35def46", "capability": "image-to-image", "status": "working"},

    # Additional models - may have issues but available for testing
    {"id": "interior-ai", "name": "Interior AI", "provider": "replicate", "model": "erayyavuz/interior-ai", "version": "e299c531485aac511610a878ef44b554381355de5ee032d109fcae5352f39fa9", "capability": "image-to-image", "status": "experimental"},
    {"id": "interior-v2", "name": "Interior V2", "provider": "replicate", "model": "jschoormans/interior-v2", "version": "8372bd24c6011ea957a0861f0146671eed615e375f038c13259c1882e3c8bac7", "capability": "image-to-image", "status": "experimental"},
    {"id": "adirik-interior-design", "name": "Adirik Interior Design", "provider": "replicate", "model": "adirik/interior-design", "version": "76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38", "capability": "image-to-image", "status": "experimental"},
    {"id": "interior-design-sdxl", "name": "Interior Design SDXL", "provider": "replicate", "model": "rocketdigitalai/interior-design-sdxl", "version": "a3c091059a25590ce2d5ea13651fab63f447f21760e50c358d4b850e844f59ee", "capability": "image-to-image", "status": "experimental"},
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

            # Determine model identifier - use version if available, otherwise use model name
            # Replicate API accepts either "version" (hash) or "model" (owner/name format)
            prediction_payload = {"input": input_data}

            if model.get("version"):
                prediction_payload["version"] = model["version"]
            elif model.get("model"):
                # Use model name format (e.g., "black-forest-labs/flux-dev")
                prediction_payload["version"] = model["model"]
            else:
                raise Exception(f"Model {model['name']} missing both 'version' and 'model' fields")

            async with httpx.AsyncClient(timeout=300.0) as client:
                # Create prediction
                response = await client.post(
                    "https://api.replicate.com/v1/predictions",
                    headers={
                        "Authorization": f"Bearer {api_token}",
                        "Content-Type": "application/json"
                    },
                    json=prediction_payload
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


async def generate_with_huggingface(model: dict, prompt: str, width: int, height: int, api_token: str, max_retries: int = 3) -> str:
    """Generate image using Hugging Face Inference API with retry logic

    Uses the Inference Providers API which routes to available providers.
    Docs: https://huggingface.co/docs/inference-providers/en/index
    """

    hf_model = model.get("hf_model")
    if not hf_model:
        raise Exception(f"No Hugging Face model specified for {model['name']}")

    for attempt in range(max_retries):
        try:
            # Use Hugging Face Inference Providers API
            # This API automatically routes to available providers (fal, replicate, etc.)
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Build form data for text-to-image endpoint
                # The Inference Providers API uses a simpler format
                response = await client.post(
                    f"https://api-inference.huggingface.co/models/{hf_model}",
                    headers={
                        "Authorization": f"Bearer {api_token}",
                    },
                    json={"inputs": prompt}
                )

                if response.status_code == 503:
                    # Model is loading, wait and retry
                    try:
                        error_data = response.json()
                        wait_time = error_data.get("estimated_time", 20)
                        print(f"⏳ [{model['name']}] Model loading, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    except:
                        # If we can't parse the error, just wait and retry
                        await asyncio.sleep(20)
                        continue

                if response.status_code != 200:
                    raise Exception(f"Hugging Face API error ({response.status_code}): {response.text}")

                # The response is the image bytes directly
                import base64
                image_data = response.content
                image_base64 = base64.b64encode(image_data).decode('utf-8')

                # Return data URL (you may want to upload to S3/Supabase Storage instead)
                return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⚠️ [{model['name']}] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ [{model['name']}] All {max_retries} attempts failed")
                raise


async def download_and_upload_to_supabase(image_url: str, job_id: str, model_id: str) -> str:
    """Download image from temporary URL and upload to Supabase Storage

    Args:
        image_url: Temporary URL from Replicate/HuggingFace
        job_id: Generation job ID
        model_id: Model identifier

    Returns:
        Public URL of the uploaded image in Supabase Storage
    """
    try:
        # Download image from temporary URL
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(image_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download image: {response.status_code}")

            image_data = response.content

            # Determine file extension from URL or content-type
            content_type = response.headers.get('content-type', 'image/webp')
            if 'png' in content_type or image_url.endswith('.png'):
                ext = 'png'
            elif 'jpeg' in content_type or 'jpg' in content_type or image_url.endswith(('.jpg', '.jpeg')):
                ext = 'jpg'
            else:
                ext = 'webp'

            # Create unique filename
            filename = f"{job_id}/{model_id}_{uuid.uuid4().hex[:8]}.{ext}"

            # Upload to Supabase Storage
            supabase = get_supabase_client()
            result = supabase.client.storage.from_('designer-assets').upload(
                filename,
                image_data,
                file_options={"content-type": content_type}
            )

            # Get public URL
            public_url = supabase.client.storage.from_('designer-assets').get_public_url(filename)

            print(f"✅ Uploaded image to Supabase Storage: {public_url}")
            return public_url

    except Exception as e:
        print(f"❌ Error uploading to Supabase Storage: {e}")
        # Return original URL as fallback
        return image_url


async def atomic_update_model_result(job_id: str, model_id: str, success: bool, image_url: Optional[str] = None, error: Optional[str] = None):
    """Atomically update a single model result using Supabase RPC"""
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

    try:
        # Use Supabase RPC function for atomic update
        supabase = get_supabase_client()
        result = supabase.client.rpc(
            'update_model_result',
            {
                'p_job_id': job_id,
                'p_model_id': model_id,
                'p_model_result': model_result
            }
        ).execute()

        if result.data:
            data = result.data[0] if isinstance(result.data, list) else result.data
            print(f"📊 Progress: {data.get('progress_percentage', 0)}% ({data.get('completed_models', 0)}/{data.get('total_models', 0)} models)")
    except Exception as e:
        print(f"❌ Error updating model result: {e}")
        raise


async def process_generation_background(job_id: str, request: InteriorRequest, models_to_use: List[dict], enhanced_prompt: str):
    """Background task with parallel processing, retry logic, timeout, and error handling"""

    try:
        # Get API tokens (try both common environment variable names)
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGING_FACE_ACCESS_TOKEN")

        # Check if we have at least one API token
        has_replicate = replicate_token is not None
        has_huggingface = huggingface_token is not None

        if not has_replicate and not has_huggingface:
            supabase = get_supabase_client()
            supabase.client.table('generation_3d').update({
                'generation_status': 'failed',
                'error_message': 'No API tokens configured (need REPLICATE_API_TOKEN or HUGGINGFACE_API_TOKEN)'
            }).eq('id', job_id).execute()
            return

        # Semaphore to limit concurrent requests (3 at a time)
        semaphore = asyncio.Semaphore(3)

        async def process_one_model(model: dict):
            async with semaphore:
                try:
                    print(f"🎨 Starting generation for {model['name']} (provider: {model['provider']})")

                    # Route to correct provider
                    if model['provider'] == 'huggingface':
                        if not huggingface_token:
                            raise Exception("HUGGINGFACE_API_TOKEN not configured")
                        temp_image_url = await generate_with_huggingface(
                            model, enhanced_prompt, request.width, request.height,
                            huggingface_token, max_retries=3
                        )
                    elif model['provider'] == 'replicate':
                        if not replicate_token:
                            raise Exception("REPLICATE_API_TOKEN not configured")
                        temp_image_url = await generate_with_replicate(
                            model, enhanced_prompt, request.width, request.height,
                            request.image, replicate_token, max_retries=3
                        )
                    else:
                        raise Exception(f"Unknown provider: {model['provider']}")

                    print(f"✅ {model['name']} generation completed, uploading to Supabase Storage...")

                    # Download and upload to Supabase Storage for permanent storage
                    permanent_url = await download_and_upload_to_supabase(temp_image_url, job_id, model['id'])

                    print(f"✅ {model['name']} completed successfully with permanent URL")
                    await atomic_update_model_result(job_id, model['id'], True, permanent_url)
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
        supabase = get_supabase_client()
        supabase.client.table('generation_3d').update({
            'generation_status': 'failed',
            'error_message': 'Job timed out after 10 minutes',
            'completed_at': datetime.utcnow().isoformat()
        }).eq('id', job_id).execute()
    except Exception as e:
        print(f"❌ FATAL ERROR in job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        supabase = get_supabase_client()
        supabase.client.table('generation_3d').update({
            'generation_status': 'failed',
            'error_message': f"Background task error: {str(e)}",
            'completed_at': datetime.utcnow().isoformat()
        }).eq('id', job_id).execute()


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

    # Determine request type (use underscores to match DB constraint)
    request_type = 'image_to_image' if request.image else 'text_to_image'

    # Insert job into database using Supabase
    supabase = get_supabase_client()
    supabase.client.table('generation_3d').insert({
        'id': job_id,
        'user_id': request.user_id,
        'workspace_id': request.workspace_id,
        'prompt': enhanced_prompt,
        'room_type': request.room_type,
        'style': request.style,
        'generation_status': 'processing',
        'progress_percentage': 0,
        'request_type': request_type,
        'models_queue': models_queue,  # Supabase handles JSONB automatically
        'models_results': {},  # Empty dict
        'workflow_status': 'processing'  # Use valid constraint value
    }).execute()

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

