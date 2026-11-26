"""Generated furniture is a SAM 3D Objects GLB mesh (fal), with no splat path and no other generator.

The source checks run in CI (pytest only, the app cannot be imported there); the behaviour checks run
wherever the app stack is installed.
"""

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _blank_comments():
    spec = importlib.util.spec_from_file_location("comment_budget", _ROOT / "scripts" / "comment_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.blank_comments


_SRC = _blank_comments()((_ROOT / "app" / "api" / "object3d_routes.py").read_text(encoding="utf-8"))


def test_no_idol_or_splat_path_left():
    for gone in ("idol", "to_spz", "clean_splat", ".spz", "modal"):
        assert gone not in _SRC.lower(), gone


def test_calls_sam3d_on_fal_and_stores_glb_only():
    assert 'FAL_ENDPOINT = "fal-ai/sam-3/3d-objects"' in _SRC
    assert 'out_path.endswith(".glb")' in _SRC
    assert '"model/gltf-binary"' in _SRC


def test_every_run_is_priced_and_logged():
    run = _SRC[_SRC.index("async def _run("):]
    assert run.index("unit_cost_usd()") < run.index("run_fal(")
    assert run.count("_log_run(") == 2


def test_failure_refunds_and_marks_failed_without_a_fallback():
    handler = _SRC[_SRC.index('logger.warning("object3d %s %s failed'):]
    assert "_refund(" in handler and '"failed"' in handler
    assert not re.search(r"\bfallback\b", _SRC, re.IGNORECASE)


def _routes():
    return pytest.importorskip("app.api.object3d_routes")


def test_mask_wins_over_prompt():
    body = _routes().fal_input("https://x/img.png", "https://x/mask.png", "sofa")
    assert body["mask_urls"] == ["https://x/mask.png"]
    assert "prompt" not in body
    assert body["export_textured_glb"] is True


def test_prompt_used_without_mask():
    assert _routes().fal_input("https://x/img.png", None, " bed ")["prompt"] == "bed"


def test_neither_mask_nor_prompt_is_refused():
    with pytest.raises(ValueError):
        _routes().fal_input("https://x/img.png", None, "  ")


@pytest.mark.parametrize("url,ok", [
    ("https://v3b.fal.media/files/a/model.glb", True),
    ("https://fal.media/files/a/model.glb", True),
    ("http://v3b.fal.media/files/a/model.glb", False),
    ("https://evil.example/fal.media/model.glb", False),
    ("https://fal.media.evil.example/model.glb", False),
])
def test_only_fal_https_files_are_downloaded(url, ok):
    assert _routes().is_fal_file_url(url) is ok


def test_only_a_product_model_takes_a_stated_size():
    assert '**(stated_size(body.size_m) if body.target == "product" else {})' in _SRC
