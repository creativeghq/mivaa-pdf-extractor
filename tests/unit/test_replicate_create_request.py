"""A model slug and a pinned version go to different Replicate endpoints, and a slug is never sent as a version."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SAM_ROUTES = ROOT / "app" / "api" / "sam_routes.py"

# Loaded by path: importing the package would pull in supabase, which CI does not install.
_spec = importlib.util.spec_from_file_location("_replicate_requests_probe", ROOT / "app" / "services" / "utilities" / "replicate_requests.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PREDICTIONS_URL = _mod.PREDICTIONS_URL
replicate_create_request = _mod.replicate_create_request

HASH = "95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"


def test_a_bare_version_hash_goes_to_predictions():
    url, body = replicate_create_request(HASH, {"a": 1})
    assert url == PREDICTIONS_URL
    assert body == {"version": HASH, "input": {"a": 1}}


def test_an_owner_name_colon_hash_sends_only_the_hash():
    url, body = replicate_create_request(f"stability-ai/stable-diffusion-inpainting:{HASH}", {})
    assert url == PREDICTIONS_URL
    assert body["version"] == HASH


def test_a_slug_goes_to_the_model_endpoint_with_no_version_field():
    for slug in ("ali-vilab/anydoor", "meta/sam-2", "black-forest-labs/flux-fill-pro"):
        url, body = replicate_create_request(slug, {"x": "y"})
        assert url == f"https://api.replicate.com/v1/models/{slug}/predictions"
        assert body == {"input": {"x": "y"}}
        assert "version" not in body


def test_garbage_is_refused_rather_than_posted():
    with pytest.raises(ValueError):
        replicate_create_request("not a model", {})


def _blank_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


def test_sam_routes_never_posts_a_slug_as_a_version():
    src = _blank_comments(SAM_ROUTES.read_text(encoding="utf-8"))
    assert '"version": "meta/sam-2"' not in src
    assert '"version": _ANYDOOR' not in src
    assert '"model": model_id' not in src
    # Every create goes through the one helper, which resolves a slug to its version BEFORE
    # posting (a failed create spends the account's create budget) and waits out one 429.
    assert src.count("_create_prediction(") >= 3
    assert "latest_version" in src
    helper = src[src.index("async def _create_prediction"):src.index("async def _whole_image_mask_data_url")]
    assert helper.index("_resolve_version(") < helper.index("client.post(")
    assert "resp.status_code == 429" in helper
    assert "retry_after" in helper


def test_sam2_is_called_as_the_automatic_mask_generator_it_is():
    src = _blank_comments(SAM_ROUTES.read_text(encoding="utf-8"))
    assert '"image": image_url' in src
    assert "input_image" not in src and "box_x1\"" not in src
    # The box is applied HERE, to the masks the model returns, not sent to a prompt it does not have.
    assert "individual_masks" in src
    assert "_mask_for_box(" in src


def test_the_inpainted_image_is_uploaded_through_the_real_client():
    src = _blank_comments(SAM_ROUTES.read_text(encoding="utf-8"))
    upload = src[src.index("async def _upload_to_storage"):]
    assert "get_supabase_client().client" in upload
    assert "get_supabase_client()\n" not in upload.split("storage.from_")[0]


def test_anydoor_is_called_with_the_names_its_schema_declares():
    src = _blank_comments(SAM_ROUTES.read_text(encoding="utf-8"))
    for key in ("bg_image_path", "bg_mask_path", "reference_image_path", "reference_image_mask"):
        assert f'"{key}"' in src, key
    for stale in ("bg_image\"", "bg_mask\"", "ref_image\"", "num_steps"):
        assert stale not in src, stale
