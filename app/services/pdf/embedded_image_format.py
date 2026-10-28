READABLE_EXTS = frozenset({"png", "jpg", "jpeg", "webp", "gif"})
_PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def to_readable_image(image_bytes, image_ext, decode_to_png):
    """Re-encode an embedded image Claude cannot open (JPEG 2000, JBIG2, ...) as PNG.

    `decode_to_png` is only called for those formats and returns PNG bytes.
    """
    if image_ext.lower() in READABLE_EXTS:
        return image_bytes, image_ext
    return decode_to_png(), "png"


def sniff_media_type(image_bytes):
    """The media type Claude must be told, read from the bytes (never from the URL)."""
    if image_bytes.startswith(_PNG_SIGNATURE):
        return "image/png"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"
