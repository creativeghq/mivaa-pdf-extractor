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


def _hit(page, img, w=1500, h=1000):
    return {"url": page, "imageUrl": img, "imageWidth": w, "imageHeight": h}


def test_search_hits_stay_on_the_maker_site_and_go_to_the_series_the_page_names():
    up = "https://keros.com/en/wp-content/uploads/sites/2/2024/09/"
    hits = [
        _hit("https://keros.com/en/series/wood/", up + "WOOD_5-en-jw.jpg"),
        _hit("https://keros.com/en/producto/wood-oak-3/", up + "WOOD-OAK-en-ab.jpg"),
        _hit("https://keros.com/en/producto/wood-espiga-oak/", up + "WOOD-ESPIGA-OAK-en-LY.jpg"),
        _hit("https://keros.com/en/producto/woodland-oak/", up + "WOODLAND-en-x.jpg"),
        _hit("https://matcarrelage.com/gb/705826-wood-30x60-keros.html", "https://matcarrelage.com/wood.jpg"),
        _hit("https://keros.com/en/series/wood/", "https://keros.com/en/logo-keros.png"),
    ]
    got = [img for img, _ in _site.search_hits_for(hits, "https://keros.com/", "Wood", ["Wood", "Wood Espiga", "Ardenza"])]
    assert got == [up + "WOOD_5-en-jw.jpg", up + "WOOD-OAK-en-ab.jpg"]
    espiga = [img for img, _ in _site.search_hits_for(hits, "https://keros.com/", "Wood Espiga", ["Wood", "Wood Espiga"])]
    assert espiga == [up + "WOOD-ESPIGA-OAK-en-LY.jpg"]


def test_a_wordpress_size_copy_collapses_to_the_largest_and_accents_fold():
    up = "https://keros.com/en/wp-content/uploads/sites/2/2023/01/"
    hits = [
        _hit("https://keros.com/en/series/tiber/", up + "TIBER-en-Gt-1024x348.jpg", 1024, 348),
        _hit("https://keros.com/en/series/tiber/", up + "TIBER-en-Gt.jpg", 3000, 1020),
    ]
    assert _site.search_hits_for(hits, "https://keros.com/", "Tíber") == [(up + "TIBER-en-Gt.jpg", "https://keros.com/en/series/tiber/")]


def test_a_page_slug_that_spells_the_product_confirms_its_page():
    base = "https://www.cereuro.it/it/collezioni-ceramiche-euro/"
    assert _site.page_names_product(base + "51-docks", "DOCKS")
    assert _site.page_names_product(base + "30-out-door", "OUTDOOR")
    assert _site.page_names_product(base + "52-spatula-30x90", "SPATULA (30x90 wall tile)")
    assert _site.page_names_product(base + "41-spatula", "SPATULA")


def test_a_sibling_series_never_takes_the_other_page():
    base = "https://www.cereuro.it/it/collezioni-ceramiche-euro/"
    assert not _site.page_names_product(base + "52-spatula-30x90", "SPATULA")
    assert not _site.page_names_product(base + "41-spatula", "SPATULA (30x90 wall tile)")
    assert not _site.page_names_product(base + "42-sinfonie-dautore", "SINFONIE")


def test_the_page_finder_takes_the_page_whose_slug_spells_the_product():
    base = "https://www.cereuro.it/it/collezioni-ceramiche-euro/"
    links = [(base + "?5", "Collezioni"), (base + "30-out-door", ""), (base + "2-area", "AREA")]
    assert _site.candidate_pages(links, "OUTDOOR") == [base + "30-out-door"]


def test_the_classical_cleanup_never_enlarges():
    pytest.importorskip("cv2")
    from PIL import Image as _Image
    src = _Image.new("RGB", (51, 51), (90, 90, 95))
    assert classical_cleanup(src).size == (51, 51)


def test_a_repair_is_stored_at_most_the_audit_limit_over_its_real_pixels():
    pytest.importorskip("cv2")
    from PIL import Image as _Image
    big = _Image.new("RGB", (595, 300), (90, 90, 95))
    assert _matching.within_upscale(big, [150, 76], 2.0).size == (300, 151)
    assert _matching.within_upscale(big, [400, 200], 2.0).size == (595, 300)
    assert _matching.within_upscale(big, None, 2.0) is big


def test_the_sourcing_repair_caps_its_size_before_storing_it():
    budget = _load("comment_budget_for_sourcing", "scripts/comment_budget.py")
    src = budget.blank_comments((_ROOT / "app/services/images/image_sourcing.py").read_text(encoding="utf-8"))
    body = src[src.index("async def _repair"):src.index("async def run_product")]
    assert body.index("within_upscale(restored, native, UPSCALE_LIMIT)") < body.index("await self._store(")


def _render(canvas, content, colour, bg=(255, 255, 255)):
    from PIL import Image as _Image
    im = _Image.new("RGB", canvas, bg)
    tile = _Image.new("RGB", content, colour)
    im.paste(tile, ((canvas[0] - content[0]) // 2, (canvas[1] - content[1]) // 2))
    return im


def test_a_small_format_drawn_on_a_blank_canvas_is_dropped_for_the_full_surface():
    pytest.importorskip("cv2")
    black = (60, 60, 62)
    images = [
        ("10x30", _render((500, 500), (125, 41), black)),
        ("60x60", _render((500, 500), (250, 250), black)),
        ("120x120", _render((500, 250), (500, 250), black)),
    ]
    assert _matching.redundant_scale_renders(images) == {"10x30": "120x120", "60x60": "120x120"}


def test_a_different_colour_is_never_dropped_for_another():
    pytest.importorskip("cv2")
    images = [
        ("white_small", _render((500, 500), (125, 125), (235, 232, 226))),
        ("black_full", _render((500, 250), (500, 250), (60, 60, 62))),
    ]
    assert _matching.redundant_scale_renders(images) == {}


def test_a_render_is_kept_when_the_larger_picture_shows_a_different_surface_of_the_same_colour():
    pytest.importorskip("cv2")
    import numpy as np
    from PIL import Image as _Image
    stripes = np.zeros((250, 500, 3), dtype="uint8")
    for x in range(0, 500, 40):
        stripes[:, x:x + 20] = (40, 40, 41)
        stripes[:, x + 20:x + 40] = (80, 80, 83)
    images = [
        ("flat_small", _render((500, 500), (125, 125), (60, 60, 62))),
        ("striped_full", _Image.fromarray(stripes)),
    ]
    assert _matching.redundant_scale_renders(images) == {}


def test_two_renders_on_blank_canvases_are_not_copies_when_their_surfaces_differ():
    pytest.importorskip("cv2")
    images = [("light", _render((500, 500), (64, 64), (200, 198, 195))), ("dark", _render((500, 500), (64, 64), (60, 60, 62)))]
    assert _matching.redundant_resized_copies(images) == {}


def test_the_same_photo_at_two_resolutions_keeps_only_the_larger():
    pytest.importorskip("cv2")
    import numpy as np
    from PIL import Image as _Image
    from PIL import ImageFilter
    rng = np.random.default_rng(3)
    # Photo-like: smooth structure survives a resize the way a real picture does; pixel noise does not.
    full = _Image.fromarray((rng.random((250, 500, 3)) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(6))
    small = full.resize((250, 125))
    other = _Image.fromarray((rng.random((250, 500, 3)) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(6))
    assert _matching.redundant_resized_copies([("full", full), ("small", small), ("other", other)]) == {"small": "full"}


def test_a_small_image_gives_way_to_the_gallery_only_when_a_gallery_photo_matches_it():
    import re
    spec = importlib.util.spec_from_file_location("comment_budget", _ROOT / "scripts" / "comment_budget.py")
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    src = budget.blank_comments((_ROOT / "app/services/images/image_sourcing.py").read_text(encoding="utf-8"))
    assert src.count('"superseded_by_manufacturer_gallery"') == 1, "one place retires for the gallery"
    helper = src[src.index("async def _give_way_to_gallery"):]
    assert helper.index("if self.dry_run:") < helper.index("self._supersede("), "a dry run records, never retires"
    calls = [m.start() for m in re.finditer(r"await self\._give_way_to_gallery\(", src)]
    assert len(calls) == 2
    for at in calls:
        block = src[max(0, at - 400):at]
        assert "await self._gallery_twin(" in block or "await self._gallery_covers(" in block
        assert "if twin and replaceable(" in block or "if replaceable(" in block, "same colour and no smaller, every time"
    twin = src[src.index("async def _gallery_twin"):]
    twin = twin[:twin.index("\n    async def ")]
    assert "pick_match(" in twin and "match_score(" in twin
    covers = src[src.index("async def _gallery_covers"):]
    covers = covers[:covers.index("\n    def ")]
    assert "except Exception" in covers and "return {}" in covers, "a failed vision check must cover nothing"


def test_a_gallery_photo_replaces_an_image_only_in_the_same_colour_and_no_smaller():
    pytest.importorskip("cv2")
    from PIL import Image as _Image
    pink, peach = (236, 190, 190), (226, 180, 150)
    swatch = _Image.new("RGB", (600, 600), pink)
    assert _matching.gallery_can_replace(swatch, _Image.new("RGB", (900, 450), pink), 600, smoothed=False)
    assert not _matching.gallery_can_replace(swatch, _Image.new("RGB", (900, 450), peach), 600, smoothed=False)
    assert not _matching.gallery_can_replace(swatch, _render((500, 500), (120, 120), pink), 600, smoothed=False)


def test_a_smoothed_repair_needs_a_gallery_photo_half_its_size():
    pytest.importorskip("cv2")
    from PIL import Image as _Image
    grey = (180, 180, 182)
    smear = _Image.new("RGB", (400, 110), grey)
    assert _matching.gallery_can_replace(smear, _render((500, 500), (250, 62), grey), 400, smoothed=True)
    assert not _matching.gallery_can_replace(smear, _render((500, 500), (150, 40), grey), 400, smoothed=True)


def test_an_image_made_from_a_crop_is_filed_under_the_crops_own_document():
    budget = _load("comment_budget_for_sourcing_docs", "scripts/comment_budget.py")
    src = budget.blank_comments((_ROOT / "app/services/images/image_sourcing.py").read_text(encoding="utf-8"))
    assert '"document_id": document_id,' in src and '"document_id": self.document_id,' not in src
    assert 'cls, crop.get("document_id"))' in src and "verdict, crop_doc)" in src


def test_the_gemini_redo_stops_when_gemini_cannot_run_and_passes_the_fidelity_gate_first():
    budget = _load("comment_budget_for_redo", "scripts/comment_budget.py")
    src = budget.blank_comments((_ROOT / "app/services/images/image_sourcing.py").read_text(encoding="utf-8"))
    body = src[src.index("async def redo_repairs_with_gemini"):src.index("async def run_image_sourcing")]
    stop = body.index('out["failed_to_run"] = ')
    assert "break" in body[stop:stop + 90]
    assert body.index("repair_is_faithful(") < body.index("s._store(") < body.index("s._supersede(")
    assert body.index("is_patterned(") < body.index("s._gemini_restore(")


def test_a_picture_the_series_page_shares_with_another_page_is_page_furniture():
    # Izida: one Cerdisa tile and one blog photo rode on every series page and landed on ~95 series each.
    up = "https://www.izida1894.gr/wp-content/uploads/2025/"
    greco = [up + "08/126cc11_06.jpg", up + "10/cerdisa_deep_powder_60x120-900x450.webp", up + "05/amb-04-rodeno-haya-900x450.jpg"]
    home = ["https://izida1894.gr/wp-content/uploads/2025/10/cerdisa_deep_powder_60x120.webp"]
    athens = [up + "05/amb-04-rodeno-haya-900x450.jpg", up + "08/9032b11_01.jpg"]
    shared = _site.page_furniture(greco, [home, athens], ["Greco", "126CC11"])
    assert {k.rsplit("/", 1)[-1] for k in shared} == {"cerdisa_deep_powder_60x120.webp", "amb-04-rodeno-haya.jpg"}


def test_a_file_named_for_the_series_survives_a_neighbour_showing_it_as_related():
    up = "https://www.izida1894.gr/wp-content/uploads/2025/08/"
    own = [up + "126cc11_06.jpg", up + "greco-ivory-60x60-amb.jpg"]
    neighbour = [up + "126cc11_06-300x300.jpg", up + "greco-ivory-60x60-amb-600x600.jpg"]
    assert _site.page_furniture(own, [neighbour], ["Greco", "126CC11"]) == set()
    assert len(_site.page_furniture(own, [neighbour], [])) == 2


def test_reference_pages_are_siblings_under_the_same_parent():
    links = [("https://x.gr/en/product/greco/", ""), ("https://x.gr/en/product/athens/", ""),
             ("https://x.gr/en/about/", ""), ("https://x.gr/en/product/arena/", "")]
    assert _site.reference_pages(links, "https://x.gr/en/product/greco/") == ["https://x.gr/en/product/athens/",
                                                                             "https://x.gr/en/product/arena/"]
