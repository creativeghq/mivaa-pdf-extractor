"""Pixel-level decisions for image sourcing: is this website image the same picture as a
low-resolution PDF crop, is a crop plain or patterned, and did a repair invent detail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

# A website candidate is the same picture when it clears these, and clears the runner-up by MARGIN.
# True swatch matches measured 0.96-0.98 at the comparison blur; a different stone of the same tone 0.77.
GLOBAL_MIN_SSIM = 0.85
GLOBAL_MAX_DELTA_E = 7.0
GLOBAL_MAX_ASPECT_GAP = 0.18
PART_MIN_CORRELATION = 0.72
# Print vs web colour drift on a true photo match measured dE 8.6; another colourway 28.8-33.3.
PART_MAX_DELTA_E = 14.0
MATCH_MARGIN = 0.04
PLAIN_MAX_DELTA_E = 4.0
# A repair must still be the source picture once shrunk back to the source's real size.
REPAIR_MIN_SSIM = 0.6


@dataclass
class MatchScore:
    score: float
    mode: str  # "whole" | "part" | "none"
    detail: str = ""


def _rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def _orient(a: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Rotate `a` to the same landscape/portrait orientation as `like` (swatches are printed rotated)."""
    if (a.shape[1] >= a.shape[0]) != (like.shape[1] >= like.shape[0]):
        return np.ascontiguousarray(np.rot90(a))
    return a


def _ssim(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = cv2.GaussianBlur(x, (7, 7), 1.5), cv2.GaussianBlur(y, (7, 7), 1.5)
    vx = cv2.GaussianBlur(x * x, (7, 7), 1.5) - mx * mx
    vy = cv2.GaussianBlur(y * y, (7, 7), 1.5) - my * my
    cxy = cv2.GaussianBlur(x * y, (7, 7), 1.5) - mx * my
    m = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(m.mean())


def _mean_delta_e(x: np.ndarray, y: np.ndarray) -> float:
    lx = cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float64)
    ly = cv2.cvtColor(y, cv2.COLOR_RGB2LAB).astype(np.float64)
    # OpenCV 8-bit Lab: L in 0..255 for 0..100, a/b offset by 128.
    lx[..., 0] *= 100 / 255
    ly[..., 0] *= 100 / 255
    lx[..., 1:] -= 128
    ly[..., 1:] -= 128
    return float(np.sqrt(((lx - ly) ** 2).sum(axis=2)).mean())


def _trim_border(a: np.ndarray) -> np.ndarray:
    """Drop near-white page margin left around a crop."""
    mask = (a < 245).any(axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return a
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def whole_image_score(crop: Image.Image, candidate: Image.Image) -> MatchScore:
    """Same framing: compare both at the crop's (low) detail level so resolution itself is not penalised."""
    c = _trim_border(_rgb(crop))
    k = _orient(_rgb(candidate), c)
    ac, ak = c.shape[1] / c.shape[0], k.shape[1] / k.shape[0]
    gap = abs(ac - ak) / max(ac, ak)
    if gap > GLOBAL_MAX_ASPECT_GAP:
        return MatchScore(0.0, "none", f"aspect {ac:.2f} vs {ak:.2f}")
    w = 96
    h = max(8, round(w / ac))
    cs = cv2.GaussianBlur(cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 1.2)
    ks = cv2.GaussianBlur(cv2.resize(k, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 1.2)
    s = _ssim(cv2.cvtColor(cs, cv2.COLOR_RGB2GRAY), cv2.cvtColor(ks, cv2.COLOR_RGB2GRAY))
    de = _mean_delta_e(cs, ks)
    if s < GLOBAL_MIN_SSIM or de > GLOBAL_MAX_DELTA_E:
        return MatchScore(0.0, "none", f"ssim {s:.2f} dE {de:.1f}")
    return MatchScore(s * (1 - de / (2 * GLOBAL_MAX_DELTA_E)), "whole", f"ssim {s:.2f} dE {de:.1f}")


def part_score(crop: Image.Image, candidate: Image.Image) -> MatchScore:
    """Different framing: the smaller picture is a region of the larger one (a PDF crop of a wider
    room shot, or one tile face of a multi-face swatch). Multi-scale normalised correlation."""
    c = _trim_border(_rgb(crop))
    k = _rgb(candidate)
    best, detail = 0.0, ""
    for small, large, label in ((c, k, "crop-in-site"), (k, c, "site-in-crop")):
        lw = 360
        Lc = cv2.resize(large, (lw, max(8, round(large.shape[0] * lw / large.shape[1]))), interpolation=cv2.INTER_AREA)
        L = cv2.GaussianBlur(cv2.cvtColor(Lc, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
        for frac in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.97, 1.0):
            if label == "site-in-crop" and frac < 0.85:
                continue  # a face of a stacked swatch spans the crop; a small patch of a decor is not a tile
            sw = round(lw * frac)
            sh = round(small.shape[0] * sw / small.shape[1])
            if sh < 16 or sw < 16 or sh > L.shape[0] or sw > L.shape[1]:
                continue
            Sc = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_AREA)
            S = cv2.cvtColor(Sc, cv2.COLOR_RGB2GRAY)
            if float(S.std()) < 4.0:
                continue  # a flat patch correlates with anything; plain tiles are judged whole
            res = cv2.matchTemplate(L, cv2.GaussianBlur(S, (0, 0), 1.0), cv2.TM_CCOEFF_NORMED)
            _, r, _, (x, y) = cv2.minMaxLoc(res)
            if r <= best:
                continue
            # Correlation is colour-blind: the same artwork in another colourway (Peonies Warm vs
            # Cold) lines up perfectly, so the matched region must also be the same colour.
            de = _mean_delta_e(cv2.GaussianBlur(Sc, (0, 0), 1.2), cv2.GaussianBlur(Lc[y:y + sh, x:x + sw], (0, 0), 1.2))
            if de > PART_MAX_DELTA_E:
                continue
            best, detail = float(r), f"{label} at {frac:.2f} r={r:.2f} dE {de:.1f}"
    if best < PART_MIN_CORRELATION:
        return MatchScore(0.0, "none", detail)
    return MatchScore(best, "part", detail)


def match_score(crop: Image.Image, candidate: Image.Image) -> MatchScore:
    w = whole_image_score(crop, candidate)
    if not is_patterned(crop):
        # A plain tile has nothing to correlate (a flat patch "appears" in any wall of any room
        # photo), and neighbouring colourways sit close: Spatula Grey vs Mint measured dE 5.3.
        # True plain matches measured dE 2.0-3.0, so only a near-identical colour counts.
        de = float(w.detail.rsplit("dE ", 1)[-1]) if "dE " in w.detail else 99.0
        return w if w.score > 0 and de <= PLAIN_MAX_DELTA_E else MatchScore(0.0, "none", w.detail)
    p = part_score(crop, candidate)
    return w if w.score >= p.score else p


def faces_in_crop(scores: Sequence[Tuple[str, MatchScore]]) -> List[Tuple[str, MatchScore]]:
    """A swatch printed as several stacked tile faces is matched by each face file the site has."""
    return [s for s in scores if s[1].mode == "part" and s[1].detail.startswith("site-in-crop")]


def pick_match(scores: Sequence[Tuple[str, MatchScore]]) -> Optional[Tuple[str, MatchScore]]:
    """The best candidate, only when it is unambiguous: an equal runner-up means we cannot tell which."""
    ranked = sorted((s for s in scores if s[1].score > 0), key=lambda s: s[1].score, reverse=True)
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1].score - ranked[1][1].score < MATCH_MARGIN:
        return None
    return ranked[0]


def is_patterned(img: Image.Image) -> bool:
    """Decors, photos and veined stone carry structure a generative repair can follow; a plain
    tile does not, and a model fills it with an invented grid."""
    a = _rgb(img)
    g = cv2.cvtColor(cv2.resize(a, (256, max(8, round(256 * a.shape[0] / a.shape[1])))), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.5), 40, 110)
    colour_spread = float(cv2.cvtColor(a, cv2.COLOR_RGB2LAB)[..., 1:].astype(np.float64).std())
    return float((edges > 0).mean()) > 0.03 or colour_spread > 6.0


def classical_cleanup(img: Image.Image) -> Image.Image:
    """Remove JPEG blocking without inventing anything, at the source's own size: non-local-means,
    edge-preserving smoothing on lightness, stronger smoothing on colour, faint grain so it does
    not look waxy. Never enlarged: an upscale adds no detail, only smoothing."""
    a = cv2.cvtColor(_rgb(img), cv2.COLOR_RGB2BGR)
    a = cv2.fastNlMeansDenoisingColored(a, None, 7, 10, 7, 21)
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    l, ca, cb = cv2.split(lab)
    l = cv2.bilateralFilter(l, 9, 30, 9)
    ca, cb = cv2.GaussianBlur(ca, (0, 0), 3), cv2.GaussianBlur(cb, (0, 0), 3)
    a = cv2.cvtColor(cv2.merge([l, ca, cb]), cv2.COLOR_LAB2BGR)
    noise = np.random.default_rng(0).normal(0, 2.0, a.shape)
    a = np.clip(a.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))


def repair_is_faithful(source: Image.Image, repaired: Image.Image, native_px: Optional[List[int]]) -> Tuple[bool, float]:
    """Shrink the repair back to the source's real resolution: if it is no longer the same picture,
    the model redrew it."""
    s = _trim_border(_rgb(source))
    r = _rgb(repaired)
    w = min(int(native_px[0]) if native_px else s.shape[1], 256)
    w = max(w, 48)
    h = max(8, round(w * s.shape[0] / s.shape[1]))
    sg = cv2.cvtColor(cv2.resize(s, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float64)
    rg = cv2.cvtColor(cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float64)
    if sg.std() < 8:
        # Near-flat source: correlation is noise, so judge it on similarity alone.
        v = _ssim(sg, rg)
    else:
        # Zero-mean correlation ignores brightness/contrast and asks only whether the structure lines up.
        a, b = sg - sg.mean(), rg - rg.mean()
        v = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) or 1.0))
    return v >= REPAIR_MIN_SSIM, round(v, 3)


#: A render whose real content covers less of its canvas than this is a scale drawing of a format.
SCALE_RENDER_MAX_SHARE = 0.3
#: It is redundant when a same-colour picture of the product holds at least this much more content.
SCALE_RENDER_MIN_GAIN = 2.0
SCALE_RENDER_MAX_DELTA_E = 6.0


@dataclass
class ContentProfile:
    area: int
    share: float
    lab: Tuple[float, float, float]


def content_box(img: Image.Image) -> np.ndarray:
    """The part of the picture that is not its flat background."""
    a = _rgb(img)
    bg = np.median(np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]), axis=0)
    mask = np.abs(a.astype(np.int32) - bg.astype(np.int32)).sum(axis=2) > 30
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return a
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def content_profile(img: Image.Image) -> ContentProfile:
    """How big the content is, and what colour."""
    h, w = img.height, img.width
    box = content_box(img)
    lab = cv2.cvtColor(box, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64).mean(axis=0)
    # OpenCV 8-bit Lab to CIE units, as in _mean_delta_e.
    lab = (lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128)
    area = int(box.shape[0] * box.shape[1])
    return ContentProfile(area, area / float(h * w), (float(lab[0]), float(lab[1]), float(lab[2])))


def _same_surface(small: np.ndarray, large: np.ndarray) -> bool:
    """The small render's content matches a centre crop of the larger content cut to its shape."""
    aspect = small.shape[1] / small.shape[0]
    h, w = large.shape[:2]
    if w / h > aspect:
        cw = max(1, int(h * aspect))
        crop = large[:, (w - cw) // 2:(w - cw) // 2 + cw]
    else:
        ch = max(1, int(w / aspect))
        crop = large[(h - ch) // 2:(h - ch) // 2 + ch]
    return match_score(Image.fromarray(small), Image.fromarray(np.ascontiguousarray(crop))).mode != "none"


def redundant_scale_renders(images: Sequence[Tuple[str, Image.Image]]) -> Dict[str, str]:
    """Scale drawing -> the larger picture that visibly shows the same surface. No match, no entry."""
    profiles = [(k, content_profile(im), content_box(im)) for k, im in images]
    out: Dict[str, str] = {}
    for key, p, pbox in profiles:
        if p.share >= SCALE_RENDER_MAX_SHARE:
            continue
        for other, q, qbox in sorted(profiles, key=lambda t: t[1].area, reverse=True):
            if other != key and other not in out and q.area >= SCALE_RENDER_MIN_GAIN * p.area \
                    and float(np.linalg.norm(np.subtract(p.lab, q.lab))) <= SCALE_RENDER_MAX_DELTA_E \
                    and _same_surface(pbox, qbox):
                out[key] = other
                break
    return out


#: The same photo saved at two resolutions: same shape, same structure, same colour.
RESIZED_COPY_MAX_ASPECT_GAP = 0.03
RESIZED_COPY_MIN_SSIM = 0.95
RESIZED_COPY_MAX_DELTA_E = 3.0


def is_resized_copy(a: Image.Image, b: Image.Image) -> bool:
    ra, rb = a.width / a.height, b.width / b.height
    if abs(ra - rb) / max(ra, rb) > RESIZED_COPY_MAX_ASPECT_GAP:
        return False
    w, h = 96, max(8, round(96 / ra))
    sa = cv2.resize(_rgb(a), (w, h), interpolation=cv2.INTER_AREA)
    sb = cv2.resize(_rgb(b), (w, h), interpolation=cv2.INTER_AREA)
    ssim = _ssim(cv2.cvtColor(sa, cv2.COLOR_RGB2GRAY), cv2.cvtColor(sb, cv2.COLOR_RGB2GRAY))
    return ssim >= RESIZED_COPY_MIN_SSIM and _mean_delta_e(sa, sb) <= RESIZED_COPY_MAX_DELTA_E


def redundant_resized_copies(images: Sequence[Tuple[str, Image.Image]]) -> Dict[str, str]:
    """Smaller copy -> the larger copy of the same photo it duplicates."""
    ordered = sorted(images, key=lambda kv: kv[1].width * kv[1].height, reverse=True)
    kept: List[Tuple[str, Image.Image]] = []
    out: Dict[str, str] = {}
    for key, img in ordered:
        twin = next((k for k, im in kept if is_resized_copy(img, im)), None)
        if twin:
            out[key] = twin
        else:
            kept.append((key, img))
    return out
