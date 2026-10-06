"""Guard: an imported product carries what a marketplace feed actually reads.

The feed queries FROM `product_prices` and reads `category_id`, `metadata.image_url`, `mpn`
and `barcode`. The XML import wrote none of them - price and category stopped in metadata,
the image went to document_images, and MPN had no mapping target at all. Each of those is a
200 with a short listing, so nothing raises.

Source-based: CI installs pytest and nothing else, so nothing here imports `app`.
"""

import ast
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
_CATEGORY_UNITS = _APP / "services" / "products" / "category_units.py"
_IMPORT_SERVICE = _APP / "services" / "integrations" / "data_import_service.py"


def _load(path, name):
    """Import WITHOUT `app` - `app.services.__init__` reaches for a Supabase client."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cu = _load(_CATEGORY_UNITS, "category_units")


def _blank_comments(src: str) -> str:
    """A marker satisfied by a comment is not satisfied. Docstrings too."""
    tree = ast.parse(src)
    spans = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                spans.add(ln)
    out = []
    for i, line in enumerate(src.split("\n"), start=1):
        if i in spans or line.lstrip().startswith("#"):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


_SERVICE_SRC = _blank_comments(_IMPORT_SERVICE.read_text(encoding="utf-8"))

CATEGORY_IDS = {"tiles": "id-tiles", "lighting": "id-lighting"}
VOCAB = {"porcelain_tile": "tiles", "pendant": "lighting"}


class TestCategoryResolvesToAnId:
    def test_a_category_key_resolves(self):
        assert _cu.resolve_category_id("tiles", CATEGORY_IDS, VOCAB) == "id-tiles"

    def test_a_fine_vocabulary_value_resolves_through_its_owner(self):
        assert _cu.resolve_category_id("porcelain_tile", CATEGORY_IDS, VOCAB) == "id-tiles"
        assert _cu.resolve_category_id("PENDANT ", CATEGORY_IDS, VOCAB) == "id-lighting"

    def test_an_unknown_value_is_none_and_never_a_neighbour(self):
        # None leaves category_id unset, which the feed reports as a gap. A guess files the
        # product under a category it does not belong to, and nothing says so.
        assert _cu.resolve_category_id("scaffolding_hire", CATEGORY_IDS, VOCAB) is None
        assert _cu.resolve_category_id(None, CATEGORY_IDS, VOCAB) is None
        assert _cu.resolve_category_id("   ", CATEGORY_IDS, VOCAB) is None

    def test_the_unit_resolver_is_unchanged_by_the_shared_registry(self):
        assert _cu.resolve_default_unit("tiles", {"tiles": "m2"}, VOCAB) == "m2"
        assert _cu.resolve_default_unit("porcelain_tile", {"tiles": "m2"}, VOCAB) == "m2"

    def test_load_category_units_still_returns_exactly_two_maps(self):
        # stage_4_products unpacks two; a third breaks it the moment the module loads.
        assert _cu.load_category_units(object()) == ({}, {})

    def test_the_registry_loader_fails_soft_to_three_empty_maps(self):
        assert _cu.load_category_registry(object()) == ({}, {}, {})


class TestTheImportWritesWhatTheFeedReads:
    def test_it_writes_a_product_prices_row(self):
        # Without one the product is invisible to the feed AND the storefront, however
        # complete it is: both start their query from product_prices.
        assert "table('product_prices')" in _SERVICE_SRC
        assert "_upsert_import_price" in _SERVICE_SRC

    def test_the_price_row_upserts_on_the_unique_key(self):
        # A second row for one product makes the feed emit it twice under one id, which
        # Skroutz and BestPrice both reject.
        assert "on_conflict='workspace_id,product_id,variant_key'" in _SERVICE_SRC

    def test_an_import_never_publishes_to_the_storefront(self):
        # Publishing to a public shop and to a marketplace is a decision, not a side effect
        # of loading a supplier file.
        assert "storefront_published" not in _SERVICE_SRC

    def test_the_category_lands_on_the_column_not_only_in_metadata(self):
        assert "product_record['category_id']" in _SERVICE_SRC
        assert "_category_id_for(" in _SERVICE_SRC

    def test_the_hero_image_lands_where_every_public_surface_looks(self):
        # imageFromMetadata reads metadata; _link_images_to_product writes document_images,
        # which no feed or storefront queries.
        assert "product_metadata['image_url']" in _SERVICE_SRC

    def test_mpn_and_barcode_reach_their_columns(self):
        # Both are mandatory on Skroutz and BestPrice, and a product missing one is dropped
        # silently, per product.
        assert "('mpn', 'mpn'), ('barcode', 'barcode')" in _SERVICE_SRC

    def test_the_price_is_parsed_rather_than_cast(self):
        # Supplier feeds carry "1.299,00 EUR"; float() on that raises and loses the product.
        assert "parse_price" in _SERVICE_SRC
