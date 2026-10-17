"""#494 registry cleanup: scaled and category-scoped aliases, per-category roles, the grout map,
and no extraction module writing a retired field name. The real FieldRegistry loads from a fake
Supabase client; CI installs pytest only, so app modules are stubbed and loaded by path.
"""
import asyncio
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_META = _ROOT / "app" / "services" / "metadata"


class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, rows):
        self.rows, self.filters = rows, []

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self.filters.append((col, val))
        return self

    def execute(self):
        rows = self.rows
        for col, val in self.filters:
            rows = [r for r in rows if r.get(col) == val]
        return _Resp(list(rows))


class _Fake:
    def __init__(self, tables):
        self.tables, self.client = tables, self

    def table(self, name):
        return _Query(self.tables.get(name, []))


def _field(name, **kw):
    row = {"field_name": name, "status": "active", "section": "dimensions", "role": "descriptive",
           "destination": "metadata", "canonicalize": False, "field_type": "text",
           "applies_to_categories": None, "aliases": [], "sensitivity": "public"}
    row.update(kw)
    return row


FIELDS = [
    _field("lead_time_days", field_type="number", aliases=["lead_time_weeks*7"]),
    _field("thickness_mm", field_type="number", applies_to_categories=["tiles", "insulation"],
           role="identity", role_by_category={"insulation": "descriptive"},
           aliases=["thickness", "thickness_inch*25.4"]),
    _field("pipe_wall_thickness_mm", field_type="number", applies_to_categories=["plumbing"],
           aliases=["thickness_mm"]),
    _field("grade", applies_to_categories=["insulation"]),
    _field("wood_grade", applies_to_categories=["wood"], aliases=["grade"]),
    _field("v_rating", applies_to_categories=["tiles"], aliases=["shade_variation"]),
    _field("color_variation", applies_to_categories=["wood"], aliases=["shade_variation"]),
    _field("grout_color_codes", applies_to_categories=["tiles"],
           aliases=["grout_mapei", "grout_kerakoll", "grout_details"]),
    _field("color", aliases=["available_colors", "colors"]),
    _field("application", aliases=["applications"]),
    _field("width_mm", field_type="number", aliases=["width", "width_cm*10"]),
    _field("sku", destination="column"),
    _field("mpn", destination="column", aliases=["product_code"]),
    _field("country_of_origin", destination="column"),
]
CATEGORIES = [
    {"category_key": k, "controlled_vocab": [], "vocab_aliases": [], "is_fallback": k == "tiles",
     "is_active": True, "sort_order": i}
    for i, k in enumerate(["tiles", "wood", "insulation", "plumbing"])
]


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env():
    fake = _Fake({"material_metadata_fields": FIELDS, "material_categories": CATEGORIES,
                  "material_field_sections": [{"section_key": "dimensions", "display_order": 1}]})
    names = ["app", "app.services", "app.services.core", "app.services.metadata",
             "app.services.core.supabase_client", "app.services.metadata.field_registry",
             "app.services.metadata.metadata_shape", "app.services.metadata.metadata_normalizer"]
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for pkg in names[:4]:
            sys.modules[pkg] = _module(pkg)
        sys.modules["app.services.core.supabase_client"] = _module(
            "app.services.core.supabase_client", get_supabase_client=lambda: fake)
        registry_mod = _load("app.services.metadata.field_registry", _META / "field_registry.py")
        _load("app.services.metadata.metadata_shape", _META / "metadata_shape.py")
        normalizer = _load("app.services.metadata.metadata_normalizer", _META / "metadata_normalizer.py")
        asyncio.run(registry_mod.field_registry.ensure_loaded(force=True))
        yield types.SimpleNamespace(registry=registry_mod.field_registry, normalizer=normalizer)
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_a_scaled_alias_resolves_with_its_factor_and_is_invisible_to_plain_renames(env):
    assert env.registry.resolve_field("lead_time_weeks") == ("lead_time_days", 7.0)
    assert env.registry.resolve_field_key("lead_time_weeks") is None
    assert env.registry.resolve_field("thickness_inch") == ("thickness_mm", 25.4)
    assert env.registry.resolve_field_key("thickness") == "thickness_mm"


def test_a_category_scoped_alias_wins_only_inside_its_category(env):
    assert env.registry.resolve_field_key("grade", "wood") == "wood_grade"
    assert env.registry.resolve_field_key("grade", "insulation") == "grade"
    assert env.registry.resolve_field_key("grade") == "grade"
    assert env.registry.resolve_field_key("thickness_mm", "plumbing") == "pipe_wall_thickness_mm"
    assert env.registry.resolve_field_key("thickness_mm", "tiles") == "thickness_mm"


def test_one_retired_name_split_by_category_is_ambiguous_without_one(env):
    assert env.registry.resolve_field_key("shade_variation", "tiles") == "v_rating"
    assert env.registry.resolve_field_key("shade_variation", "wood") == "color_variation"
    assert env.registry.resolve_field_key("shade_variation") is None


def test_role_is_decided_per_category(env):
    assert env.registry.is_identity("thickness_mm", "tiles") is True
    assert env.registry.is_identity("thickness_mm", "insulation") is False
    assert "thickness_mm" in env.registry.identity_fields("tiles")
    assert "thickness_mm" not in env.registry.identity_fields("insulation")


def test_the_normalizer_converts_a_scaled_alias_and_keeps_an_unconvertible_value(env):
    out = env.normalizer.normalize_metadata(
        {"commercial": {"lead_time_weeks": "3 weeks"}, "dimensions": {"thickness_inch": "abc"}})
    assert out["commercial"] == {"lead_time_days": 21}
    assert out["dimensions"] == {"thickness_inch": "abc"}


def test_the_normalizer_uses_the_product_category_for_a_scoped_alias(env):
    out = env.normalizer.normalize_metadata({"performance": {"grade": "A"}}, product_category="wood")
    assert out["performance"] == {"wood_grade": "A"}


def test_grout_brand_keys_fold_into_one_brand_map(env):
    out = env.normalizer.normalize_metadata({"commercial": {
        "grout_mapei": {"value": "Ultracolor", "product": "130"},
        "grout_kerakoll": "Fugabella 04",
        "grout_details": [{"supplier": "mapei", "code": "130"}],
    }})
    assert out["commercial"]["grout_color_codes"] == {
        "mapei": "Ultracolor 130", "kerakoll": "Fugabella 04",
        "details": [{"supplier": "mapei", "code": "130"}],
    }


# Retired registry names no extraction writer may emit. Generic words and other-domain names
# (the vision_analysis schema's material_type / surface_pattern, warehouse product_code) are out.
RETIRED = (
    "available_colors primary_color application_areas recommended_use usage_type suitable_rooms "
    "origin_country warranty_years energy_efficiency voc_level care_instructions "
    "maintenance_frequency group_of_companies lead_time_weeks body_material primary_material "
    "grout_isomat grout_kerakoll grout_mapei grout_technica grout_suppliers grout_details "
    "thickness_inch surface_finish finish_type finish_sheen stone_surface_treatment surface_texture "
    "shade_variation traffic_level thermal_shock water_absorption tile_subtype heat_resistance "
    "coefficient_of_expansion stone_hardness installation_format moisture_content "
    "wear_layer_thickness top_layer_thickness underlay_required wood_installation_method "
    "wood_surface_treatment wood_treatment bevel_edge planks_per_box heat_output_watts "
    "heat_output_btu flow_connection_size upholstery_material fabric_options table_height_cm "
    "seat_height_cm seat_depth_cm arm_height_cm drop_cm roll_width_cm pattern_repeat_cm edge_profiles "
    "stone_density light_color noise_class_db mounting_method fixing_type sqft_per_box sqft_per_pallet "
    "weight_per_box_lb weight_per_pallet_lb electric_element_watts water_content_litres "
    "elongation_percent dimension_unit"
).split()

SCANNED = [
    _META / "metadata_normalizer.py", _META / "dynamic_metadata_extractor.py",
    _META / "table_metadata_extractor.py", _META / "field_registry.py",
    _ROOT / "app/services/products/product_spec_vision_extractor.py",
    _ROOT / "app/services/products/product_spec_extractor_v2.py",
    _ROOT / "app/services/knowledge/catalog_legend_extractor_v2.py",
    _ROOT / "app/services/integrations/product_identity_service.py",
    _ROOT / "app/api/pdf_processing/product_processor.py",
    _ROOT / "app/services/search/rag_service.py",
    _ROOT / "app/api/pdf_processing/stage_4_products.py",
    _ROOT / "app/services/products/product_creation_service.py",
]


def test_no_extraction_module_names_a_retired_field():
    sys.path.insert(0, str(_ROOT / "scripts"))
    try:
        from comment_budget import blank_comments
    finally:
        sys.path.pop(0)
    pattern = re.compile(r"\b(" + "|".join(RETIRED) + r")\b")
    offenders = []
    for path in SCANNED:
        code = blank_comments(path.read_text(encoding="utf-8"))
        offenders += [f"{path.name}: {m}" for m in sorted(set(pattern.findall(code)))]
    assert offenders == []


def test_a_labelled_or_roman_grade_matches_a_numeric_option():
    validation = _load("_field_validation_494", _META / "field_validation.py")
    spec = types.SimpleNamespace(field_type="dropdown", dropdown_options=("0", "1", "2", "3", "4", "5"),
                                 validation_rules=None)
    for raw in ("PEI 4", "PEI IV", "4"):
        verdict = validation.validate_value(spec, raw)
        assert (verdict.status, verdict.value) == ("ok", "4"), raw
    rcd = types.SimpleNamespace(field_type="dropdown", dropdown_options=("AC", "A", "F", "B", "B+"),
                                validation_rules=None)
    assert validation.validate_value(rcd, "B").value == "B"
    assert validation.validate_value(rcd, "b+").value == "B+"


_STAGE4_FNS = ("_is_empty_value", "_category_of", "_flatten_for_canonicalization",
               "_route_column_fields", "_extract_fields_from_chunk_text", "_registry_key",
               "_merge_enriched_fields_into_metadata", "_iso_country")


@pytest.fixture
def stage4(env):
    """Stage 4's pure helpers, lifted out by AST: the module itself needs the full app."""
    import ast
    tree = ast.parse((_ROOT / "app/api/pdf_processing/stage_4_products.py").read_text(encoding="utf-8"))
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _STAGE4_FNS]
    keep += [n for n in tree.body if isinstance(n, ast.AnnAssign)
             and getattr(n.target, "id", None) == "_COLUMN_FOR_FIELD"]
    keep += [n for n in tree.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) in ("_COLUMN_FORM", "_ISO_COUNTRY_EXCEPTIONS") for t in n.targets)]
    from typing import Any, Callable, Dict, List, Optional
    ns = {"Callable": Callable, "field_registry": env.registry, "scale_value": env.normalizer.scale_value, "_re": re, "re": re,
          "Any": Any, "Dict": Dict, "List": List, "Optional": Optional}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "stage_4_products", "exec"), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in (*_STAGE4_FNS, "_COLUMN_FOR_FIELD")})


def test_product_code_reaches_mpn_through_the_registry_alias(stage4):
    assert "product_code" not in stage4._COLUMN_FOR_FIELD
    meta = {"product_code": "A-1", "material_category": "tiles"}
    assert stage4._route_column_fields(meta) == {"mpn": "A-1"}
    assert "product_code" not in meta
    meta = {"product_code": "A-1", "mpn": "M-9"}
    assert stage4._route_column_fields(meta) == {"mpn": "M-9"}


def test_a_country_name_stays_in_metadata_instead_of_failing_the_column_check(stage4):
    meta = {"country_of_origin": "it "}
    assert stage4._route_column_fields(meta) == {"country_of_origin": "IT"}
    meta = {"country_of_origin": "Germany"}
    assert stage4._route_column_fields(meta) == {}
    assert meta["country_of_origin"] == "Germany"


def test_flatten_places_aliases_under_their_field_converted_and_category_scoped(stage4):
    flat = stage4._flatten_for_canonicalization({
        "material_category": "wood", "available_colors": ["oak"],
        "dimensions": {"width_cm": "12"}, "performance": {"grade": "A"},
    })
    assert flat["color"] == ["oak"]
    assert flat["width_mm"] == 120
    assert flat["wood_grade"] == "A"
    assert stage4._flatten_for_canonicalization({"dimensions": {"width_cm": "wide"}}).get("width_mm") is None
    assert stage4._flatten_for_canonicalization({"color": "red", "colors": ["blue"]})["color"] == "red"


def test_chunk_grout_lands_in_one_grout_color_codes_map(stage4):
    cands = stage4._extract_fields_from_chunk_text("100 Mapei and 43 Kerakoll. MAPEI | ULTRACOLOR PLUS")
    grout = cands["grout_color_codes"]
    assert grout["suppliers"] == ["KERAKOLL", "MAPEI"]
    assert grout["mapei"] == "ULTRACOLOR PLUS"
    assert not [k for k in cands if k.startswith("grout_") and k != "grout_color_codes"]
    merged, _filled = stage4._merge_enriched_fields_into_metadata(
        {"material_category": "tiles"}, cands, {"application": ["floor"]})
    assert merged["grout_color_codes"] == grout
    assert merged["application"] == ["floor"]
    assert not [k for k in merged if k.startswith("grout_") and k != "grout_color_codes"]


def test_a_scalar_already_holding_a_container_key_is_kept_not_crashed_on(stage4):
    merged, _filled = stage4._merge_enriched_fields_into_metadata(
        {"appearance": "matte white", "material_category": "tiles"},
        {}, {"appearance_colors": ["white"]},
    )
    assert merged["appearance"] == "matte white"
    assert merged["_extraction_metadata"]["appearance._not_nested"] == ["colors_from_vision"] \
        or "_extraction_metadata" not in merged


def test_the_eu_and_uk_country_conventions_map_to_iso(stage4):
    assert stage4._route_column_fields({"country_of_origin": "EL"}) == {"country_of_origin": "GR"}
    assert stage4._route_column_fields({"country_of_origin": "uk"}) == {"country_of_origin": "GB"}


def test_the_country_column_takes_the_vocabulary_answer_not_the_regex(stage4):
    known = {"italy": "IT", "ελλάδα": "GR"}
    resolve = lambda v: known.get(v.strip().lower())  # noqa: E731
    assert stage4._route_column_fields({"country_of_origin": "Ελλάδα"}, country=resolve) == {"country_of_origin": "GR"}
    meta = {"country_of_origin": "EU"}
    assert stage4._route_column_fields(meta, country=resolve) == {}, "EU is two letters, not a country"
    assert meta["country_of_origin"] == "EU"
