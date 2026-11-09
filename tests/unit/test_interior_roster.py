"""Every interior roster row builds a real payload, in both directions."""
import importlib.util
from pathlib import Path

import pytest

# CI installs pytest only, so the module is loaded by path rather than through `app`.
_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("interior_roster_under_test", _ROOT / "app/services/generation/interior_roster.py")
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)

REPLICATE_ROWS = [m for m in r.ALL_MODELS if m.get("provider") == "replicate"]
ROOM = "https://example.com/room.jpg"


@pytest.mark.parametrize("model", REPLICATE_ROWS, ids=lambda m: f"{m['id']}:{m['capability']}")
def test_every_replicate_row_has_an_adapter(model):
    image = ROOM if model["capability"] == "image-to-image" else None
    payload = r._build_model_input(model, "spec", 1024, 1024, image)
    assert payload["prompt"] == "spec"
    if image:
        # The room must reach the model under the key its schema reads, never be dropped.
        assert ROOM in str(payload), f"{model['id']} drops the room photo"


def test_an_unknown_schema_is_refused_not_guessed():
    with pytest.raises(ValueError):
        r._build_model_input({"name": "x", "input_schema": "nope", "capability": "text-to-image"}, "p", 1, 1, None)


def test_gpt_image_never_bills_the_auto_tier():
    row = next(m for m in REPLICATE_ROWS if m["input_schema"] == "gpt_image")
    assert r._build_model_input(row, "p", 1, 1, None)["quality"] != "auto"


def test_flux3_never_grounds_a_spec_render_on_the_web():
    row = next(m for m in REPLICATE_ROWS if m["input_schema"] == "flux3")
    assert r._build_model_input(row, "p", 1, 1, None)["grounding"] is False
