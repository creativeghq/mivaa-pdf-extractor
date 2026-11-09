"""The interior generation roster and its per-model input adapters. Stdlib only, so the
unit tier (pytest, no app dependencies) can load it by path."""
from typing import Optional


# Model configurations. The `gemini_edge` rows are ONE upstream (generate-interior-gemini) reached
# with a `model_tier` that selects the model: pro -> gemini-3-pro-image, grok -> grok-aurora.
# Every Replicate row takes reference images, so the same row serves both rosters.
_REPLICATE_2026_11 = [
    {"id": "nano-banana-2.1", "name": "Nano Banana 2.1", "provider": "replicate",
     "model": "google/nano-banana-2.1", "cost_per_generation": 0.05, "input_schema": "nano_banana"},
    {"id": "gpt-image-2.6", "name": "GPT Image 2.6", "provider": "replicate",
     "model": "openai/gpt-image-2.6-sunburst", "cost_per_generation": 0.09, "input_schema": "gpt_image"},
    {"id": "seedream-5-pro", "name": "Seedream 5 Pro", "provider": "replicate",
     "model": "bytedance/seedream-5-pro", "cost_per_generation": 0.09, "input_schema": "seedream5"},
    {"id": "krea-3", "name": "Krea 3", "provider": "replicate",
     "model": "krea/krea-3", "cost_per_generation": 0.08, "input_schema": "krea3"},
    {"id": "flux-3-image", "name": "FLUX 3", "provider": "replicate",
     "model": "black-forest-labs/flux-3-image", "cost_per_generation": 0.10, "input_schema": "flux3"},
]


def _edge_rows(capability: str) -> list:
    return [
        {"id": "gemini-interior-pro", "registry_id": "gemini-3-pro-image", "name": "Gemini 3 Pro", "provider": "gemini",
         "route": "gemini_edge", "capability": capability, "cost_per_generation": 0.0, "model_tier": "pro"},
        {"id": "grok-aurora-interior", "registry_id": "xai-aurora", "name": "Grok Aurora", "provider": "xai",
         "route": "gemini_edge", "capability": capability, "cost_per_generation": 0.0, "model_tier": "grok"},
    ]


TEXT_TO_IMAGE_MODELS = _edge_rows("text-to-image") + [{**m, "capability": "text-to-image"} for m in _REPLICATE_2026_11]
IMAGE_TO_IMAGE_MODELS = _edge_rows("image-to-image") + [{**m, "capability": "image-to-image"} for m in _REPLICATE_2026_11]

# Combined list
ALL_MODELS = TEXT_TO_IMAGE_MODELS + IMAGE_TO_IMAGE_MODELS


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

    Each Replicate model has its own schema, selected by the roster row's `input_schema`.
    """
    schema = model.get("input_schema", "generic")

    # The 2026-11 roster: every model takes reference images, so one row serves both directions.
    # With a room photo the aspect follows the photo — a re-cropped picture is not the same room.
    if schema == "nano_banana":
        data = {"prompt": prompt, "resolution": "2K", "output_format": "jpg",
                "aspect_ratio": "match_input_image" if image_url else "16:9"}
        if image_url:
            data["image_input"] = [image_url]
        return data

    if schema == "seedream5":
        data = {"prompt": prompt, "size": "2K", "output_format": "jpeg",
                "aspect_ratio": "match_input_image" if image_url else "16:9"}
        if image_url:
            data["image_input"] = [image_url]
        return data

    if schema == "gpt_image":
        # `quality` must be explicit: the default `auto` bills at the $0.25 tier.
        data = {"prompt": prompt, "quality": "medium", "complexity": "high", "output_format": "jpeg",
                "aspect_ratio": "auto" if image_url else "16:9"}
        if image_url:
            data["input_images"] = [image_url]
        return data

    if schema == "krea3":
        data = {"prompt": prompt, "resolution": "2k", "thinking_mode": "high",
                "aspect_ratio": "auto" if image_url else "16:9"}
        if image_url:
            data["images"] = [image_url]
        return data

    if schema == "flux3":
        # Web grounding is ON by default upstream; a spec render must not pull in a search hit.
        data = {"prompt": prompt, "resolution": "2k", "grounding": False, "output_format": "jpg",
                "aspect_ratio": "auto" if image_url else "16:9"}
        if image_url:
            data["images"] = [image_url]
        return data

    # An unknown schema is a config error: a guessed payload is how unknown keys get
    # silently dropped and a "success" ignores the room photo.
    raise ValueError(f"No input adapter for {model['name']} (input_schema={schema!r})")
