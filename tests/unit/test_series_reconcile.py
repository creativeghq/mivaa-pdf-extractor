"""A series keeps only what its sources print about it; Keros Saboya is the case."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("series_reconcile_under_test", _ROOT / "app/services/discovery/series_reconcile.py")
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)

SABOYA_PAGES = {
    162: "SABOYA\n33x33 cm / 13\"x13\"\nRedstone Crema\n33x33 cm · Antislip M188\nSaboya Crema\n33x33",
    257: "Saboya Gris  $ 25 \\times 50 $ cm | Saboya Olas  $ 25 \\times 50 $ cm | Saboya Marengo  $ 33 \\times 33 $ cm",
    4: "CÓRDOBA 176 SABOYA 256\nTRAVERTINO 124",
    5: "PACKING | Saboya 33X33 14 | Calma 30x60 8 | Reality 25x50 12 | Austral 25x25 20 | Alberca 25x25 20",
}
SITE = ["https://keros.com/en/wp-content/uploads/sites/2/2022/03/SABOYA-OLAS-BEIGE-en-t6VTk4ZH48UTAHpw.jpg",
        "https://keros.com/en/wp-content/uploads/sites/2/2022/03/SABOYA-BEIGE-en-yb9OLIraGDViJqXe.jpg",
        "https://keros.com/en/wp-content/uploads/sites/2/2022/03/SABOYA-en-EJs6aMzg3Fmi8Ufw.jpg"]


def test_a_series_owns_every_page_printing_its_name_beside_a_size_and_not_the_index():
    names = ["Saboya", "Calma", "Reality", "Austral", "Alberca"]
    assert set(sr.series_pages(SABOYA_PAGES, "Saboya", names)) == {162, 257}


def test_the_makers_photo_names_give_the_variants_without_the_file_hash():
    assert sr.site_variant_names(SITE, "Saboya") == ["beige", "olas beige"]


def test_only_values_the_sources_print_survive_and_a_maths_size_is_a_size():
    corpus = "\n".join(sr.normalize_text(t) for t in SABOYA_PAGES.values())
    kept = sr.verify({"colours": ["Crema", "Gris", "Marengo", "Beige", "Blanco"], "decors": ["Olas"], "finishes": [],
                      "formats": ["33x33", "25x50", "60x60"], "uses": ["floor", "roof"]}, corpus, ["beige", "olas beige"])
    assert kept["colours"] == ["crema", "gris", "marengo", "beige"]
    assert kept["formats"] == ["33x33", "25x50"] and kept["decors"] == ["olas"] and kept["uses"] == ["floor"]


def test_an_attribute_naming_another_series_is_a_leak():
    attrs = {"slip_resistance": ["antislip m188 (redstone series)", "c3"], "color": ["crema"]}
    assert sr.leaked_values(attrs, ["Redstone", "Wood"]) == {"slip_resistance": ["antislip m188 (redstone series)"]}


def test_the_description_is_built_from_kept_fields_and_the_makers_text_follows_it():
    line = sr.factual_line("Saboya", "porcelain", ["floor", "wall"], ["33x33", "25x50"], ["crema", "gris"], ["olas"])
    assert line == "Saboya: porcelain floor and wall tile, in 33x33 and 25x50 cm, in Crema and Gris, with an Olas decor."
    assert sr.compose_description(line, " Saboya evokes stone. ") == line + "\n\nSaboya evokes stone."
    assert sr.factual_line("Peldaño", None, ["stair"], [], [], []) == "Peldaño: tile."
    assert sr.REFUSAL.search("The source text contains only format and finish listings for the Saboya 33x33 tile")


def test_sizes_single_letters_and_language_codes_in_a_photo_name_are_not_variants():
    urls = ["https://keros.com/x/LIVORNO-30X90-G-en-AbCdEfGhIjKlMnOp.jpg", "https://keros.com/x/GALAXY-596X596-en-AbCdEfGhIjKlMnOp.jpg",
            "https://keros.com/x/TECHNO-SQUARE-EN.jpg", "https://keros.com/x/LEGNO-OAK-en-AbCdEfGhIjKlMnOp.jpg"]
    assert sr.site_variant_names(urls, "Livorno") == [] and sr.site_variant_names(urls, "Galaxy") == []
    assert sr.site_variant_names(urls, "Techno") == ["square"] and sr.site_variant_names(urls, "Legno") == ["oak"]


def test_a_one_letter_slip_in_a_long_colour_name_is_the_same_colour_and_a_short_one_is_not():
    assert sr.one_slip("natutal", "natural") and sr.one_slip("natrual", "natural") and sr.one_slip("naturl", "natural")
    assert not sr.one_slip("gris", "grey") and not sr.one_slip("bone", "bona") and not sr.one_slip("natural", "neutral")


IZIDA_PAGES = {
    41: "MARBLE\nArena\nCOLORS\nTECHNICAL SPECIFICATIONS\nPACKAGING DETAILS\nSIZES AND CODES\n\n\nSOFT LIGHT\nsize m2/box\n60 × 120 cm",
    42: "",
    15: "Index Arena 36 Calacata 38 Thasos 64 Athens 66 Atlanta 67",
}


def test_a_heading_with_its_sizes_further_down_falls_back_to_the_discovered_range():
    names = ["Arena", "Calacata", "Thasos", "Athens", "Atlanta"]
    assert sr.series_pages(IZIDA_PAGES, "Arena", names) == {}
    assert set(sr.own_pages(IZIDA_PAGES, "Arena", names, [41, 42])) == {41}


def test_the_fallback_never_takes_an_index_page_or_a_page_outside_the_range():
    names = ["Arena", "Calacata", "Thasos", "Athens", "Atlanta"]
    assert set(sr.own_pages(IZIDA_PAGES, "Arena", names, [15, 41])) == {41}
    assert sr.own_pages(IZIDA_PAGES, "Arena", names, []) == {}
