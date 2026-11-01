"""Image sourcing (#525): find the manufacturer page, match a low-resolution crop to the same
picture, never to a look-alike, and never accept a repair that redrew the image."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.utils"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
if not hasattr(sys.modules.get("app.utils.ssrf_guard"), "safe_fetch_bytes"):
    _load("app.utils.ssrf_guard", "app/utils/ssrf_guard.py")
_site = _load("manufacturer_site_under_test", "app/services/images/manufacturer_site.py")
candidate_pages, name_tokens, page_score, parse_page = _site.candidate_pages, _site.name_tokens, _site.page_score, _site.parse_page

PAGE = """
<html><head><meta property="og:image" content="/upload/serie/52/cover.jpg"></head><body>
<a href="/it/collezioni-ceramiche-euro/52-spatula-30x90">Spatula 30x90</a>
<a href="/it/collezioni-ceramiche-euro/40-spatula">Spatula</a>
<a href="/it/collezioni-ceramiche-euro/53-alchemy">Alchemy</a>
<a href="https://other.example.com/x">elsewhere</a>
<a href="upload/colori_serie/immagine/318/SPATULA BONE 30X90.jpg">bone</a>
<img src="upload/colori_serie/preview/318/thumbnail/SPATULA BONE 30X90.jpg">
<img data-src="/upload/serie/ambienti/52/AMB_SPATULA.png" src="data:image/gif;base64,R0lG">
<img src="/img/logo-white-big.png">
<img srcset="/a/small.jpg 400w, /a/large.jpg 1600w">
</body></html>
"""


def test_page_links_stay_on_the_site_and_images_skip_thumbnails_and_logos():
    links, images = parse_page(PAGE, "https://www.cereuro.it/it")
    assert all(u.startswith("https://www.cereuro.it/") for u, _ in links)
    assert "https://www.cereuro.it/upload/colori_serie/immagine/318/SPATULA BONE 30X90.jpg" in images
    assert "https://www.cereuro.it/upload/serie/ambienti/52/AMB_SPATULA.png" in images
    assert "https://www.cereuro.it/a/large.jpg" in images
    assert not any("thumbnail" in u or "logo" in u or u.startswith("data:") for u in images)


def test_relative_paths_follow_base_href_and_a_catalogue_pdf_is_not_a_page():
    html = ('<base href="/"><a href="it/collezioni/52-spatula-30x90">Spatula</a>'
            '<a href="upload/cataloghi/allegati/59/EURO_Cat_SPATULA.pdf">Spatula catalogue</a>'
            '<a href="upload/colori/318/SPATULA BONE.jpg">bone</a>')
    links, images = parse_page(html, "https://cereuro.it/it/collezioni/52-spatula-30x90")
    assert [u for u, _ in links] == ["https://cereuro.it/it/collezioni/52-spatula-30x90"]
    assert images == ["https://cereuro.it/upload/colori/318/SPATULA BONE.jpg"]
    _, www_images = parse_page('<img src="https://www.cereuro.it/upload/x/tile.jpg">', "https://cereuro.it/it")
    assert www_images == ["https://www.cereuro.it/upload/x/tile.jpg"]


def test_the_size_in_the_product_name_picks_the_right_collection_page():
    links, _ = parse_page(PAGE, "https://www.cereuro.it/it")
    assert name_tokens("SPATULA (30x90 wall tile)") == (["spatula"], ["30x90"])
    assert candidate_pages(links, "SPATULA (30x90 wall tile)") == ["https://www.cereuro.it/it/collezioni-ceramiche-euro/52-spatula-30x90"]
    assert page_score("https://x/53-alchemy", "Alchemy", ["spatula"], []) == 0.0


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

_matching = _load("image_matching_under_test", "app/services/images/image_matching.py")
classical_cleanup, is_patterned, match_score = _matching.classical_cleanup, _matching.is_patterned, _matching.match_score
pick_match, repair_is_faithful = _matching.pick_match, _matching.repair_is_faithful


def _texture(seed, size=(300, 100), tint=(140, 136, 128)):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.normal(0, 18, (size[1], size[0])), (0, 0), 6)
    base = base / (base.std() or 1) * 35
    rgb = np.clip(np.stack([base + t for t in tint], axis=2), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _degrade(img, factor=6):
    small = img.resize((img.width // factor, img.height // factor), Image.BILINEAR)
    return small.resize(img.size, Image.NEAREST)


def test_a_degraded_crop_matches_its_own_original_and_not_a_different_colour():
    original = _texture(1)
    other_colour = _texture(1, tint=(110, 95, 60))
    crop = _degrade(original)
    scores = [("same", match_score(crop, original)), ("other", match_score(crop, other_colour))]
    picked = pick_match(scores)
    assert picked is not None and picked[0] == "same"


def test_two_equally_good_candidates_are_ambiguous_and_match_nothing():
    original = _texture(2)
    crop = _degrade(original)
    assert pick_match([("a", match_score(crop, original)), ("b", match_score(crop, original))]) is None


def test_plain_tile_is_not_sent_to_a_generative_repair():
    plain = Image.new("RGB", (300, 100), (190, 186, 180))
    assert not is_patterned(plain)


def test_a_repair_that_redraws_the_picture_is_rejected_and_the_clean_up_is_accepted():
    source = _degrade(_texture(3))
    assert repair_is_faithful(source, classical_cleanup(source), [50, 16])[0]
    assert not repair_is_faithful(source, _texture(99), [50, 16])[0]



def test_neighbouring_plain_colourways_do_not_match_and_a_plain_tile_has_no_region_match():
    grey = Image.new("RGB", (300, 100), (168, 168, 160))
    mint = Image.new("RGB", (300, 100), (160, 170, 156))
    assert match_score(grey, grey.copy()).score > 0
    assert match_score(grey, mint).score == 0
    room = _texture(5, size=(600, 400))
    room.paste(grey, (100, 100))
    assert match_score(grey, room).score == 0
