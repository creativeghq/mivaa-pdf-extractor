READABLE_EXTS = frozenset({"png", "jpg", "jpeg", "webp", "gif"})


def to_readable_image(image_bytes, image_ext, decode_to_png):
    """Re-encode an embedded image Claude cannot open (JPEG 2000, JBIG2, ...) as PNG.

    `decode_to_png` is only called for those formats and returns PNG bytes.
    """
    if image_ext.lower() in READABLE_EXTS:
        return image_bytes, image_ext
    return decode_to_png(), "png"
