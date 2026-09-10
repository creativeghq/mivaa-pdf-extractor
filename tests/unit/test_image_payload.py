"""Guard: a browser data URL must be normalized before it reaches base64.b64decode."""

import base64
import importlib.util
from pathlib import Path

import pytest

# Loaded by path: app/utils/__init__.py is cheap, but keeping this import-free means the guard
# runs in CI with nothing installed but pytest.
_MODULE = Path(__file__).resolve().parents[2] / "app" / "utils" / "image_payload.py"
_spec = importlib.util.spec_from_file_location("image_payload_under_test", _MODULE)
image_payload = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(image_payload)

normalize_base64_image = image_payload.normalize_base64_image

# A real 1x1 JPEG, as a browser's FileReader.readAsDataURL would hand it over.
RAW_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIs"
    "IxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAA"
    "AAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
)
DATA_URL = f"data:image/jpeg;base64,{RAW_B64}"


def test_the_naive_decode_is_silently_wrong_not_loud():
    """The premise. If this ever starts raising, the silent-fallback bug could not recur."""
    correct = base64.b64decode(RAW_B64)

    naive = base64.b64decode(DATA_URL)  # must NOT raise — that is the trap

    assert naive != correct, "b64decode on a data URL somehow matched; premise broken"
    assert len(naive) != len(correct), (
        "the prefix must shift the stream — if lengths match, re-derive this guard"
    )


def test_a_data_url_normalizes_to_exactly_the_raw_payload():
    assert normalize_base64_image(DATA_URL) == RAW_B64
    assert base64.b64decode(normalize_base64_image(DATA_URL)) == base64.b64decode(RAW_B64)


def test_the_normalized_payload_decodes_to_a_real_jpeg():
    """End of the chain: the bytes PIL would receive must actually be an image."""
    decoded = base64.b64decode(normalize_base64_image(DATA_URL))
    assert decoded[:3] == b"\xff\xd8\xff", "not a JPEG SOI marker — normalization is wrong"


def test_bare_base64_passes_through_untouched():
    assert normalize_base64_image(RAW_B64) == RAW_B64


def test_is_idempotent():
    """Callers cannot always tell which form they were handed, so double-application is
    expected and must not corrupt the payload."""
    once = normalize_base64_image(DATA_URL)
    assert normalize_base64_image(once) == once


@pytest.mark.parametrize("mime", ["image/png", "image/webp", "image/jpeg", "application/pdf"])
def test_any_mime_type_is_handled(mime):
    assert normalize_base64_image(f"data:{mime};base64,{RAW_B64}") == RAW_B64


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_input_is_returned_unchanged(empty):
    """`None` means 'no image' and must stay `None` — not become an empty string that a
    caller then treats as a present-but-broken payload."""
    assert normalize_base64_image(empty) == empty


def test_a_non_base64_data_url_is_left_for_the_caller_to_reject():
    """
    `data:image/svg+xml,<svg…>` has no `base64,` marker. Returning a half-parsed string would
    rebuild the silent-corruption failure one layer over; the caller's decode must fail loudly.
    """
    svg = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'/>"
    assert normalize_base64_image(svg) == svg


def test_surrounding_whitespace_does_not_defeat_the_prefix_check():
    assert normalize_base64_image(f"  {DATA_URL}  ") == RAW_B64
