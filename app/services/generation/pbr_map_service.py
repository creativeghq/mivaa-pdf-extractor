"""
Derive PBR texture maps (normal / roughness / metalness) from a single albedo image.

WHY DETERMINISTIC RATHER THAN A MODEL
-------------------------------------
Audit #310 item 2 listed three options for this — an SVBRDF model on Modal, a hosted service, or
Replicate. All three carry a per-image inference cost, an availability dependency, and a cold
start. None of them is needed to clear the actual bar.

The bar is not "physically accurate BRDF recovery". It is "stop putting a flat photograph on a 3D
plane". `ARPreviewModal` already wires `normalMap` / `roughnessMap` / `metalnessMap` and simply
never receives anything, so a material currently reads as a sticker: no surface relief, no
specular variation, uniformly matte under every light. Sobel-derived relief and variance-derived
roughness fix precisely that, cost nothing per image, and are deterministic — the same input
always produces the same maps, so a re-run never silently changes a product's appearance.

Everything here uses libraries this backend already ships (Pillow, numpy, scipy). No new
dependency, no new secret, no new upstream that can be down.

If an ML SVBRDF is wanted later it replaces `derive_pbr_maps` behind the same signature and the
same `products.metadata.pbr_maps` contract — the consumer does not change.

WHAT THE MAPS ACTUALLY MEAN HERE
--------------------------------
normal     Surface relief. Luminance is treated as a height field; the normal is its gradient.
           This is an approximation and it is wrong in a specific, known way: dark PAINT reads as
           depth exactly like a dark GROOVE does. For tile, stone, wood and fabric — where tone
           variation really is mostly relief — it is close enough to be a large improvement. For a
           printed graphic it will invent relief that is not there.

roughness  Local contrast. A polished surface is locally smooth, so low variance -> low roughness
           (sharper reflections); a honed or textured surface is locally busy -> high roughness.

metalness  Flat zero. Architectural materials — porcelain, stone, timber, textile — are
           dielectrics. Guessing metalness from a photograph is unreliable and the failure mode is
           ugly (a mirror-finish floor), so this deliberately does not guess. A genuinely metallic
           product needs an explicit override, not an inference.
"""
from __future__ import annotations

import io
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

# Maps are downscaled before derivation. Beyond ~1024px the extra detail is below what the AR
# plane resolves, and the arrays are what dominate memory here.
MAX_EDGE = 1024
# Relief strength. Higher exaggerates the surface; 2.0 reads as a material rather than a relief
# map of a photograph, which is what larger values look like.
NORMAL_STRENGTH = 2.0
# Neighbourhood for the local-variance roughness estimate, in pixels.
ROUGHNESS_WINDOW = 5


def _load_rgb(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    # Flatten alpha onto white: a transparent PNG otherwise derives relief from its own
    # cutout edge, which produces a hard rectangular ridge around the whole tile.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _fit(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / float(max(w, h))
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luma, 0..1. Perceptual weighting matters — a plain channel mean makes blue
    grout read as deeper than it is."""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0


def _normal_map(lum: np.ndarray, strength: float = NORMAL_STRENGTH) -> np.ndarray:
    """Tangent-space normal map from a height field, encoded the way GL/three.js expects."""
    # Blur first: without it, JPEG ringing and sensor noise become high-frequency "relief" and
    # the surface reads as sandpaper regardless of the material.
    height = ndimage.gaussian_filter(lum, sigma=1.0)
    # Sobel gradients. Negated so that BRIGHT reads as RAISED, which is the convention that makes
    # grout lines sink rather than stand proud.
    dx = -ndimage.sobel(height, axis=1) * strength
    dy = -ndimage.sobel(height, axis=0) * strength
    dz = np.ones_like(height)
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    # 0..1 -> 0..255 with the standard [-1,1] -> [0,1] remap. Z stays in the upper half, which is
    # why a flat area comes out the familiar lavender (128,128,255).
    out = np.stack([
        (dx / norm) * 0.5 + 0.5,
        (dy / norm) * 0.5 + 0.5,
        (dz / norm) * 0.5 + 0.5,
    ], axis=-1)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _roughness_map(lum: np.ndarray, window: int = ROUGHNESS_WINDOW) -> np.ndarray:
    """Local standard deviation, normalised. Busy -> rough, smooth -> glossy."""
    mean = ndimage.uniform_filter(lum, size=window)
    sq = ndimage.uniform_filter(lum * lum, size=window)
    # Clamp before sqrt: floating-point error makes this marginally negative on flat regions.
    var = np.clip(sq - mean * mean, 0.0, None)
    std = np.sqrt(var)
    peak = float(std.max())
    if peak <= 1e-6:
        # A perfectly flat image has no texture to infer from. Mid-roughness is the honest
        # answer — neither mirror nor chalk — rather than 0, which would render it as a mirror.
        norm = np.full_like(std, 0.5)
    else:
        norm = std / peak
    # Floor at 0.25: nothing in a materials catalogue is a perfect mirror, and near-zero
    # roughness makes the AR plane reflect the environment map like chrome.
    norm = 0.25 + norm * 0.65
    return (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)


def _encode(arr: np.ndarray, mode: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def derive_pbr_maps(image_bytes: bytes) -> Optional[Dict[str, Tuple[bytes, str]]]:
    """Derive normal / roughness / metalness PNGs from one albedo image.

    Returns {kind: (png_bytes, content_type)} or None when the input is not a usable image.
    Returning None rather than raising keeps a single unreadable product image from failing a
    batch — the caller records that this product has no maps and moves on.
    """
    try:
        img = _fit(_load_rgb(image_bytes))
    except Exception as e:  # noqa: BLE001 — a corrupt or unsupported upload is data, not a bug
        logger.warning("[pbr] could not read image: %s", e)
        return None

    rgb = np.asarray(img, dtype=np.float32)
    lum = _luminance(rgb)

    normal = _normal_map(lum)
    roughness = _roughness_map(lum)
    # Dielectric. See the module docstring — this deliberately does not guess.
    metalness = np.zeros(lum.shape, dtype=np.uint8)

    return {
        "normal": (_encode(normal, "RGB"), "image/png"),
        "roughness": (_encode(roughness, "L"), "image/png"),
        "metalness": (_encode(metalness, "L"), "image/png"),
    }
