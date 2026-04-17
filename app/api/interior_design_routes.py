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
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.credits_integration_service import get_credits_service
from app.schemas.api_responses import InteriorDesignResponse

router = APIRouter(prefix="/api", tags=["Interior Design"])

# Room type → readable English name
_ROOM_NAMES: dict = {
    "living_room": "living room", "bedroom": "bedroom", "kitchen": "kitchen",
    "bathroom": "bathroom", "dining_room": "dining room", "home_office": "home office",
    "hallway": "hallway", "studio": "studio apartment", "outdoor": "outdoor terrace",
    "kids_room": "children's room", "basement": "basement lounge",
}

# Style → descriptive vocabulary for image generation models
_STYLE_VOCAB: dict = {
    "modern": "modern, clean lines, sleek surfaces, contemporary furniture, neutral palette",
    "minimalist": "minimalist, ultra-clean, negative space, uncluttered, monochromatic tones",
    "scandinavian": "Scandinavian, Nordic, light oak wood, white walls, cozy hygge atmosphere",
    "industrial": "industrial loft, exposed concrete, raw steel accents, warehouse aesthetic",
    "luxury": "luxury, high-end finishes, marble surfaces, gold accents, designer furniture, opulent",
    "bohemian": "bohemian, eclectic layered textiles, warm earth tones, plants, woven accents",
    "traditional": "traditional, classic rich wood tones, ornate mouldings, symmetrical layout",
    "mediterranean": "Mediterranean, terracotta tiles, arched details, warm plaster walls, natural stone",
    "japandi": "Japandi, wabi-sabi, natural wood, muted pale palette, zen minimalism",
    "art_deco": "Art Deco, geometric patterns, brass accents, velvet upholstery, dramatic lighting",
    "rustic": "rustic, reclaimed wood, exposed beams, stone fireplace, warm cozy atmosphere",
    "coastal": "coastal, light airy, sandy tones, rattan furniture, linen textiles, sea-glass tones",
}


def _build_generation_prompt(
    prompt: str,
    room_type: Optional[str],
    style: Optional[str],
    is_image_to_image: bool,
) -> str:
    """
    Build a rich, model-optimised generation prompt from user inputs.

    Text-to-image: full descriptive sentence + quality boosters.
    Image-to-image: concise style directive (avoids overriding the reference image).
    """
    room = _ROOM_NAMES.get(room_type or "", room_type or "interior space")
    style_name = style or "contemporary"
    style_tags = _STYLE_VOCAB.get(style or "", style_name)

    if is_image_to_image:
        # Transformation prompts should be directive and concise so the model
        # focuses on style transfer rather than ignoring the reference image.
        parts = [
            f"{style_name} style redesign of a {room},",
            prompt.rstrip(".") + ",",
            style_tags + ",",
            "professional interior design, high quality rendering",
        ]
    else:
        # Text-to-image benefits from rich natural language + quality anchors.
        parts = [
            f"Professional interior design photograph of a beautifully designed {style_name} {room},",
            prompt.rstrip(".") + ",",
            style_tags + ",",
            "soft natural and ambient lighting, photorealistic render,",
            "architectural photography, wide-angle lens, sharp focus, high detail, 8K resolution",
        ]
    return " ".join(parts)

# Model configurations - All Replicate models
# Text-to-Image Models (for prompts without reference images)
TEXT_TO_IMAGE_MODELS = [
    {"id": "flux-2-pro", "name": "FLUX.2 Pro", "provider": "replicate", "model": "black-forest-labs/flux-2-pro", "capability": "text-to-image", "cost_per_generation": 0.05},
    {"id": "playground-v2.5", "name": "Playground v2.5", "provider": "replicate", "model": "playgroundai/playground-v2.5-1024px-aesthetic", "version": "a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24", "capability": "text-to-image", "cost_per_generation": 0.01, "input_schema": "playground_v25"},
    {"id": "sd3", "name": "Stable Diffusion 3", "provider": "replicate", "model": "stability-ai/stable-diffusion-3", "capability": "text-to-image", "cost_per_generation": 0.055},
]

# Image-to-Image Models (for interior design transformation with reference images)
IMAGE_TO_IMAGE_MODELS = [
    {"id": "comfyui-interior-remodel", "name": "ComfyUI Interior Remodel", "provider": "replicate", "model": "jschoormans/comfyui-interior-remodel", "version": "2a360362540e1f6cfe59c9db4aa8aa9059233d40e638aae0cdeb6b41f3d0dcce", "capability": "image-to-image", "status": "working", "cost_per_generation": 0.02, "input_schema": "comfyui_interior"},
    {"id": "interiorly-gen1-dev", "name": "Interiorly Gen1 Dev", "provider": "replicate", "model": "julian-at/interiorly-gen1-dev", "version": "5e3080d1b308e80197b32f0ce638daa8a329d0cf42068739723d8259e44b445e", "capability": "image-to-image", "status": "working", "cost_per_generation": 0.015,
     "input_schema": "flux_lora_interior"},
    {"id": "designer-architecture", "name": "Designer Architecture", "provider": "replicate", "model": "davisbrown/designer-architecture", "version": "0d6f0893b05f14500ce03e45f54290cbffb907d14db49699f2823d0fd35def46", "capability": "image-to-image", "status": "working", "cost_per_generation": 0.018},
    # Restored: was returning 404 because no version hash — versioned endpoint now used (confirmed live 2026-03-22)
    {"id": "interior-v2", "name": "Interior V2", "provider": "replicate",
     "model": "jschoormans/interior-v2",
     "version": "8372bd24c6011ea957a0861f0146671eed615e375f038c13259c1882e3c8bac7",
     "capability": "image-to-image", "status": "working", "cost_per_generation": 0.02,
     "input_schema": "interior_v2"},
    # Restored: was returning 404 because no version hash — versioned endpoint now used (confirmed live 2026-03-22)
    {"id": "adirik-interior-design", "name": "Adirik Interior Design", "provider": "replicate",
     "model": "adirik/interior-design",
     "version": "76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38",
     "capability": "image-to-image", "status": "working", "cost_per_generation": 0.02,
     "input_schema": "adirik_interior"},
    # Restored: was timing out with 50 steps — reduced to 30 (model still live, image param is "input" not "image")
    {"id": "erayyavuz-interior-ai", "name": "Interior AI", "provider": "replicate",
     "model": "erayyavuz/interior-ai",
     "version": "e299c531485aac511610a878ef44b554381355de5ee032d109fcae5352f39fa9",
     "capability": "image-to-image", "status": "working", "cost_per_generation": 0.02,
     "input_schema": "interior_ai"},
    # Flux LoRA interior models
    {"id": "interor-2", "name": "Interior 2 (Flux)", "provider": "replicate", "model": "doobls-ai/interor-2",
     "version": "91f2ef63c76a73d2ec4c67cf7b2a9672e074046cf4fde1d98e46a5829f7ea68b",
     "capability": "image-to-image", "status": "working", "cost_per_generation": 0.014,
     "input_schema": "flux_lora_interior"},
    {"id": "colourful-interiors", "name": "Colourful Interiors (Flux)", "provider": "replicate", "model": "rihan-a/colourful_interiors",
     "version": "ba0425bc2e4bebafa8bd918519fdf3b5a022969a6a7c8ba0746b807bb5b541a3",
     "capability": "image-to-image", "status": "working", "cost_per_generation": 0.014,
     "input_schema": "flux_lora_interior", "trigger_word": "INTR"},
    # SD-based img2img models
    {"id": "stable-interiors-v2-pb", "name": "Stable Interiors V2", "provider": "replicate", "model": "pointblack/stable-interiors-v2",
     "version": "569b1bd6e4df6c9c900ad932d4a3a9f05585fac957dc6bc627aa1654853a97b5",
     "capability": "image-to-image", "status": "working", "cost_per_generation": 0.011,
     "input_schema": "stable_interiors"},
    {"id": "stable-interiors-v2-yz", "name": "Stable Interiors V2 (Fast)", "provider": "replicate", "model": "youzu/stable-interiors-v2",
     "version": "4836eb257a4fb8b87bac9eacbef9292ee8e1a497398ab96207067403a4be2daf",
     "capability": "image-to-image", "status": "working", "cost_per_generation": 0.011,
     "input_schema": "stable_interiors"},
    # SDXL interior — requires reference image for depth/ControlNet
    {"id": "interior-design-sdxl", "name": "Interior Design SDXL", "provider": "replicate",
     "model": "rocketdigitalai/interior-design-sdxl",
     "version": "a3c091059a25590ce2d5ea13651fab63f447f21760e50c358d4b850e844f59ee",
     "capability": "image-to-image", "status": "working", "cost_per_generation": 0.14,
     "input_schema": "sdxl_interior"},
]

# Combined list
ALL_MODELS = TEXT_TO_IMAGE_MODELS + IMAGE_TO_IMAGE_MODELS

class InteriorRequest(BaseModel):
    prompt: str = Field(..., description="Interior design description")
    image: Optional[str] = Field(None, description="Reference image URL for image-to-image generation")
    room_type: Optional[str] = Field(None, description="Type of room (living_room, bedroom, etc.)")
    style: Optional[str] = Field(None, description="Design style (modern, minimalist, etc.)")
    models: Optional[List[str]] = Field(None, description="Specific model IDs to use, or None for all models")
    exclude_models: Optional[List[str]] = Field(None, description="Model IDs to exclude from generation")
    user_id: str = Field(..., description="User ID")
    workspace_id: Optional[str] = Field(None, description="Workspace ID")
    width: int = Field(1024, description="Image width")
    height: int = Field(1024, description="Image height")


_VIRTUAL_STAGING_ROOM_MAP: dict = {
    "living_room": "Living Room", "bedroom": "Bedroom", "kitchen": "Kitchen",
    "bathroom": "Bathroom", "dining_room": "Dining Room", "home_office": "Office",
    "outdoor": "Garden", "hallway": "Living Room", "studio": "Living Room",
    "kids_room": "Bedroom", "basement": "Living Room",
}

_VIRTUAL_STAGING_STYLE_MAP: dict = {
    "modern": "Modern", "minimalist": "Modern", "scandinavian": "Scandinavian",
    "industrial": "Urban Industrial", "luxury": "Transitional Luxury",
    "bohemian": "Modern Organic", "traditional": "Traditional",
    "mediterranean": "Modern Organic", "japandi": "Scandinavian Oasis",
    "art_deco": "Mid-Century Modern", "rustic": "Farmhouse", "coastal": "Coastal",
}


def _build_model_input(
    model: dict,
    prompt: str,
    width: int,
    height: int,
    image_url: Optional[str],
    room_type: Optional[str] = None,
    style: Optional[str] = None,
    api_token: Optional[str] = None,
) -> dict:
    """
    Build model-specific input parameters.

    Each Replicate model has its own schema — sending unsupported params causes 422 errors.
    We use the 'input_schema' key on the model config to select the right builder.
    Models without an explicit schema fall back to the safe generic builder.
    """
    schema = model.get("input_schema", "generic")

    if schema == "comfyui_interior":
        # jschoormans/comfyui-interior-remodel: ComfyUI workflow — only accepts image + prompt.
        # Sending standard SD params (strength, guidance_scale, num_inference_steps) causes 422.
        if not image_url:
            raise ValueError("comfyui-interior-remodel requires a reference image")
        return {"image": image_url, "prompt": prompt}

    if schema == "adirik_interior":
        # adirik/interior-design: uses prompt_strength (not strength), no num_outputs
        data: dict = {"prompt": prompt, "num_inference_steps": 25, "guidance_scale": 7.5, "prompt_strength": 0.8}
        if image_url:
            data["image"] = image_url
        return data

    if schema == "interior_ai":
        # erayyavuz/interior-ai: image param is 'input' (not 'image'), supports strength/guidance/steps
        # Reduced to 30 steps (was 50) to avoid Replicate polling timeout
        data = {"prompt": prompt, "num_inference_steps": 30, "guidance_scale": 7.5, "strength": 0.8}
        if image_url:
            data["input"] = image_url  # NOTE: 'input', not 'image'
        return data

    if schema == "interior_v2":
        # jschoormans/interior-v2: minimal schema, only prompt + image
        data = {"prompt": prompt}
        if image_url:
            data["image"] = image_url
        return data

    if schema == "flux_lora_interior":
        # julian-at/interiorly-gen1-dev, doobls-ai/interor-2, rihan-a/colourful_interiors
        # Flux LoRA base — prompt_strength for img2img, aspect_ratio for text-to-image
        trigger = model.get("trigger_word", "")
        final_prompt = f"{trigger} {prompt}" if trigger and trigger not in prompt else prompt
        data = {"prompt": final_prompt, "num_inference_steps": 28, "guidance_scale": 3}
        if image_url:
            data["image"] = image_url
            data["prompt_strength"] = 0.8
        else:
            data["aspect_ratio"] = "16:9"
        return data

    if schema == "stable_interiors":
        # pointblack/stable-interiors-v2, youzu/stable-interiors-v2
        # SD-based img2img: requires image; skip gracefully when none provided
        if not image_url:
            raise ValueError(f"{model['name']} requires a reference image")
        return {"prompt": prompt, "image": image_url, "num_inference_steps": 50, "guidance_scale": 15, "prompt_strength": 0.8}

    if schema == "virtual_staging":
        # proplabs/virtual-staging: structured room + furniture_style enums, requires api_key as input
        room = _VIRTUAL_STAGING_ROOM_MAP.get(room_type or "", "Living Room")
        furniture_style = _VIRTUAL_STAGING_STYLE_MAP.get(style or "", "Default (AI decides)")
        data = {
            "image": image_url,
            "room": room,
            "furniture_style": furniture_style,
        }
        if api_token:
            data["replicate_api_key"] = api_token
        return data

    if schema == "sdxl_interior":
        # rocketdigitalai/interior-design-sdxl: requires image for depth/ControlNet
        if not image_url:
            raise ValueError(f"{model['name']} requires a reference image")
        return {
            "prompt": prompt,
            "image": image_url,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "depth_strength": 0.8,
            "promax_strength": 0.8,
            "refiner_strength": 0.4,
        }

    if schema == "playground_v25":
        # playgroundai/playground-v2.5-1024px-aesthetic
        # Recommended guidance_scale is 3.0 (much lower than SD defaults) — higher values
        # cause harsh edges and over-saturation. Negative prompt softens the output.
        data = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_outputs": 1,
            "num_inference_steps": 50,
            "guidance_scale": 3.0,
            "scheduler": "DPMSolver++",
            "negative_prompt": "dark, harsh shadows, heavy contrast, gritty, low quality, blurry, overexposed, sharp hard edges, ugly",
        }
        return data

    # Generic fallback — used by working models that do support these standard params
    data = {"prompt": prompt, "num_inference_steps": 25, "guidance_scale": 7.5}
    if model["capability"] == "text-to-image":
        data["width"] = width
        data["height"] = height
        data["num_outputs"] = 1
    if model["capability"] == "image-to-image" and image_url:
        data["image"] = image_url
        data["strength"] = 0.8
    return data


async def generate_with_replicate(model: dict, prompt: str, width: int, height: int, image_url: Optional[str], api_token: str, max_retries: int = 3, room_type: Optional[str] = None, style: Optional[str] = None) -> str:
    """Generate image using Replicate API with retry logic"""

    for attempt in range(max_retries):
        try:
            input_data = _build_model_input(model, prompt, width, height, image_url, room_type, style, api_token)

            # Determine Replicate API endpoint and payload format:
            # - If model has a version hash → POST /v1/predictions with {"version": "<hash>", "input": ...}
            # - If model only has owner/name → POST /v1/models/{owner}/{name}/predictions with {"input": ...}
            model_name = model.get("model", "")
            version_hash = model.get("version", "")

            if not model_name and not version_hash:
                raise Exception(f"Model {model['name']} missing both 'version' and 'model' fields")

            async with httpx.AsyncClient(timeout=300.0) as client:
                if version_hash:
                    prediction_payload = {"version": version_hash, "input": input_data}
                    url = "https://api.replicate.com/v1/predictions"
                else:
                    # Use the model-specific endpoint for latest version
                    prediction_payload = {"input": input_data}
                    url = f"https://api.replicate.com/v1/models/{model_name}/predictions"

                # Create prediction
                response = await client.post(
                    url,
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

                # Poll for completion — 90 attempts × 2s = 180s max
                # erayyavuz/interior-ai takes ~65s compute + Replicate queue time
                max_attempts = 90
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

        except ValueError:
            # Config/validation error — no point retrying (e.g. missing required image)
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"⚠️ [{model['name']}] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ [{model['name']}] All {max_retries} attempts failed")
                raise


async def generate_with_gemini_edge(
    prompt: str,
    room_type: Optional[str],
    style: Optional[str],
    image_url: Optional[str],
    user_id: str,
    workspace_id: Optional[str],
    model_tier: str = "fast",
) -> str:
    """
    Call the generate-interior-gemini Supabase edge function.
    Returns the permanent public image URL.
    """
    supabase_url = os.getenv("SUPABASE_URL", "")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url or not service_role_key:
        raise Exception("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not configured")

    body: dict = {
        "mode": "image-edit" if image_url else "text-to-image",
        "prompt": prompt,
        "room_type": room_type,
        "style": style,
        "model_tier": model_tier,
        "user_id": user_id,
        "workspace_id": workspace_id,
    }
    if image_url:
        body["reference_image_url"] = image_url

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{supabase_url}/functions/v1/generate-interior-gemini",
            headers={"Authorization": f"Bearer {service_role_key}", "apikey": service_role_key, "Content-Type": "application/json"},
            json=body,
        )
        if resp.status_code != 200:
            raise Exception(f"Gemini edge function error {resp.status_code}: {resp.text}")
        result = resp.json()
        if not result.get("success"):
            raise Exception(result.get("error", "Gemini generation failed"))
        return result["image_url"]


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
            result = supabase.client.storage.from_('generation-images').upload(
                filename,
                image_data,
                file_options={"content-type": content_type}
            )

            # Get public URL
            public_url = supabase.client.storage.from_('generation-images').get_public_url(filename)

            print(f"✅ Uploaded image to Supabase Storage: {public_url}")
            return public_url

    except Exception as e:
        print(f"❌ Error uploading to Supabase Storage: {e}")
        # Return original URL as fallback
        return image_url


async def atomic_update_model_result(job_id: str, model_id: str, success: bool, image_url: Optional[str] = None, cost: float = 0.0, error: Optional[str] = None):
    """Atomically update a single model result using Supabase RPC"""
    model_result = {
        "model_id": model_id,
        "status": "completed" if success else "failed",
        "generated_at": datetime.utcnow().isoformat(),
        "cost": cost
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
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if not replicate_token:
            for m in models_to_use:
                await atomic_update_model_result(job_id, m['id'], False, None, 0.0, "REPLICATE_API_TOKEN not configured")
            return

        # Semaphore to limit concurrent requests (3 at a time)
        semaphore = asyncio.Semaphore(3)

        async def process_one_model(model: dict):
            async with semaphore:
                try:
                    print(f"🎨 Starting generation for {model['name']}")

                    # All models in the grid are Replicate
                    temp_image_url = await generate_with_replicate(
                        model, enhanced_prompt, request.width, request.height,
                        request.image, replicate_token, max_retries=3,
                        room_type=request.room_type, style=request.style,
                    )
                    print(f"✅ {model['name']} generation completed, uploading to Supabase Storage...")
                    permanent_url = await download_and_upload_to_supabase(temp_image_url, job_id, model['id'])

                    cost = 0.0
                    credits_service = get_credits_service()
                    debit_result = await credits_service.debit_credits_for_replicate(
                        user_id=request.user_id,
                        workspace_id=request.workspace_id,
                        operation_type="interior_design",
                        model_name=model['id'],
                        num_generations=1,
                        job_id=job_id,
                        metadata={
                            'room_type': request.room_type,
                            'style': request.style,
                            'model_display_name': model['name'],
                        }
                    )
                    cost = debit_result.get('billed_cost_usd', model.get('cost_per_generation', 0.0))
                    if debit_result.get('success'):
                        print(f"✅ {model['name']} completed + credits debited (${cost:.3f}, {debit_result.get('credits_debited', 0):.1f} credits)")
                    else:
                        print(f"⚠️ {model['name']} completed but credit debit failed: {debit_result.get('error')}")

                    await atomic_update_model_result(job_id, model['id'], True, permanent_url, cost, None)
                except Exception as e:
                    print(f"❌ {model['name']} failed: {e}")
                    await atomic_update_model_result(job_id, model['id'], False, None, 0.0, str(e))

        # Process all models in parallel with 10-minute timeout
        tasks = [process_one_model(m) for m in models_to_use]
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=600  # 10 minutes
        )

        print(f"✅ Job {job_id} completed (credits debited per-model via shared layer)")

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


@router.post("/interior", responses={200: {"model": InteriorDesignResponse}})
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
        # Image-to-image: all working image-to-image models
        # (Gemini is handled separately by the agent via generate_gemini tool)
        models_to_use = [m for m in IMAGE_TO_IMAGE_MODELS if m.get("status") != "failing"]
    else:
        # Text-to-image: all text-to-image models
        models_to_use = list(TEXT_TO_IMAGE_MODELS)

    # Apply exclusions (e.g. gemini-interior excluded when generate_gemini tool handles it separately)
    if request.exclude_models:
        models_to_use = [m for m in models_to_use if m["id"] not in request.exclude_models]

    # Build rich generation prompt
    enhanced_prompt = _build_generation_prompt(
        prompt=request.prompt,
        room_type=request.room_type,
        style=request.style,
        is_image_to_image=bool(request.image),
    )

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

    # Start background processing — attach an error callback so task failures
    # are logged and the job status is updated rather than silently disappearing.
    def _on_generation_done(task: asyncio.Task) -> None:
        exc = task.exception() if not task.cancelled() else None
        if exc:
            import logging as _logging
            _logging.getLogger(__name__).error(
                f"[interior_design] Background generation task for job {job_id} failed: {exc}",
                exc_info=exc,
            )

    task = asyncio.create_task(
        process_generation_background(job_id, request, models_to_use, enhanced_prompt)
    )
    task.add_done_callback(_on_generation_done)

    # Return job info immediately
    return JSONResponse({
        "success": True,
        "job_id": job_id,
        "model_count": len(models_to_use),
        "models": [{"id": m["id"], "name": m["name"]} for m in models_to_use],
        "message": f"Started generating {len(models_to_use)} interior design variations"
    })


