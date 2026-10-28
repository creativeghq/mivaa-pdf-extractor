import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[2] / "app" / "services" / "pdf" / "embedded_image_format.py"
_spec = importlib.util.spec_from_file_location("embedded_image_format", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
to_readable_image = _module.to_readable_image


def _must_not_decode():
    raise AssertionError("a format Claude can read must not be re-encoded")


def test_formats_claude_reads_pass_through_untouched():
    for ext in ("png", "jpg", "JPEG", "webp", "gif"):
        assert to_readable_image(b"raw", ext, _must_not_decode) == (b"raw", ext)


def test_jpeg2000_and_other_unreadable_formats_become_png():
    for ext in ("jpx", "jp2", "jb2"):
        assert to_readable_image(b"raw", ext, lambda: b"png-bytes") == (b"png-bytes", "png")


def test_media_type_is_read_from_the_bytes():
    sniff = _module.sniff_media_type
    assert sniff(bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"....") == "image/png"
    assert sniff(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"....") == "image/jpeg"
    assert sniff(b"GIF89a....") == "image/gif"
    assert sniff(b"RIFF" + bytes(4) + b"WEBPVP8 ") == "image/webp"
