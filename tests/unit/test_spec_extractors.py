"""Spec extraction: flags by COLUMN, per-page failures reported, registry-mapped vision output,
and a property extractor with no silent rule-based stand-in (#494 HIGH items).

CI installs pytest only, so fitz / PIL / sentry_sdk and the app modules are stubbed and the
modules under test are loaded by path.
"""
import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTS = _ROOT / "app" / "services" / "products"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Spec:
    def __init__(self, name, section, categories=None):
        self.name, self.section, self.categories = name, section, categories

    def applies_to(self, key):
        return self.categories is None or key in self.categories


class _Registry:
    is_loaded = True
    fields = {"lumens": _Spec("lumens", "performance", {"lighting"}),
              "slip_resistance": _Spec("slip_resistance", "performance", {"tiles"})}

    async def ensure_loaded(self, **_kw):
        return None

    def resolve_field_key(self, key):
        return key if key in self.fields else None

    def resolve_field(self, key, _category=None):
        return (key, 1.0) if key in self.fields else None

    def spec_for(self, key):
        return self.fields.get(key)

    def resolve_category(self, value):
        return {"pendant_light": "lighting", "lighting": "lighting", "tiles": "tiles"}.get(value)

    def priority_fields_prompt(self, _cat):
        return "FIELDS"


@pytest.fixture
def mods():
    stubs = {
        "fitz": types.SimpleNamespace(open=None, Document=object),
        "PIL": types.SimpleNamespace(Image=None),
        "sentry_sdk": types.SimpleNamespace(),
        "app": types.ModuleType("app"), "app.services": types.ModuleType("app.services"),
        "app.services.core": types.ModuleType("app.services.core"),
        "app.services.metadata": types.ModuleType("app.services.metadata"),
        "app.services.utilities": types.ModuleType("app.services.utilities"),
        "app.services.products": types.ModuleType("app.services.products"),
        "app.services.core.anthropic_error_reporter": types.SimpleNamespace(
            report_anthropic_failure=lambda *a, **k: None),
        "app.services.metadata.field_registry": types.SimpleNamespace(field_registry=_Registry()),
        "app.services.metadata.metadata_normalizer": types.SimpleNamespace(
            scale_value=lambda v, f: v * f),
        "app.services.utilities.prompt_registry": types.SimpleNamespace(
            get_cached=lambda *a, **k: "Spec for {product_name}. {category_fields}",
            prefetch=None, load_prompt=None, PromptNotConfigured=RuntimeError),
        "app.services.core.supabase_client": types.SimpleNamespace(get_supabase_client=lambda: None),
        "app.config": types.SimpleNamespace(
            get_settings=lambda: types.SimpleNamespace(default_workspace_id="ws")),
    }
    stubs["PIL"].Image = types.SimpleNamespace()
    sys.modules.setdefault("PIL.Image", stubs["PIL"].Image)
    saved = {n: sys.modules.get(n) for n in list(stubs) + ["app.services.core.claude_tool_call"]}
    try:
        sys.modules.update(stubs)
        tool_call = _load("app.services.core.claude_tool_call",
                          _ROOT / "app/services/core/claude_tool_call.py")
        psve = _load("_psve_under_test", _PRODUCTS / "product_spec_vision_extractor.py")
        v2 = _load("_spec_v2_under_test", _PRODUCTS / "product_spec_extractor_v2.py")
        emp = _load("_emp_under_test", _PRODUCTS / "enhanced_material_property_extractor.py")
        yield types.SimpleNamespace(psve=psve, v2=v2, emp=emp, tool_call=tool_call)
    finally:
        for n in ("_psve_under_test", "_spec_v2_under_test", "_emp_under_test"):
            sys.modules.pop(n, None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _span(text, x0, x1, cy):
    return {"text": text, "x0": x0, "x1": x1, "y0": cy - 4, "y1": cy + 4,
            "cx": (x0 + x1) / 2, "cy": cy}


def test_a_flag_bullet_is_assigned_by_the_column_it_sits_under(mods):
    """A row marking ONLY gloss has one bullet; bullet order would have called it MATT."""
    header = [_span("MATT", 200, 230, 100), _span("GLOSS", 240, 275, 100),
              _span("SHOWER", 280, 320, 92), _span("WALL", 285, 315, 104)]
    row = [_span("NOVA", 10, 50, 200), _span("•", 255, 260, 200),
           _span("12", 400, 410, 200), _span("1.44", 420, 440, 200), _span("30", 450, 460, 200)]
    out = mods.v2._extract_values_positional(row, "NOVA", header + row)
    assert out.get("finish_gloss") is True
    assert "finish_matt" not in out


def test_without_header_geometry_no_flag_is_guessed(mods):
    row = [_span("NOVA", 10, 50, 200), _span("•", 255, 260, 200),
           _span("12", 400, 410, 200), _span("1.44", 420, 440, 200), _span("30", 450, 460, 200)]
    out = mods.v2._extract_values_positional(row, "NOVA", row)
    assert not any(k.startswith("finish_") for k in out)
    assert out.get("_flags_unmapped") == 1


def test_the_spec_tool_requires_a_boolean_page_contains_target(mods):
    schema = mods.psve.SPEC_VISION_TOOL["input_schema"]
    assert schema["properties"]["page_contains_target"] == {"type": "boolean"}
    assert "page_contains_target" in schema["required"]
    assert schema["additionalProperties"] is True


def test_per_page_failures_are_counted_and_all_failed_is_failed(mods, monkeypatch, tmp_path):
    pdf = tmp_path / "c.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(mods.psve, "_resolve_pdf_pages_for_product", lambda *a, **k: [0, 1])
    monkeypatch.setattr(mods.psve, "_render_page_under_limit", lambda *a, **k: b"img")
    monkeypatch.setattr(mods.psve, "_call_claude_vision", lambda *a, **k: None)
    out = mods.psve.extract_specs_from_pdf_pages(str(pdf), [1, 2], "NOVA")
    assert out["_status"] == "failed" and len(out["_page_failures"]) == 2

    replies = iter([None, {"page_contains_target": True, "lumens": 800}])
    monkeypatch.setattr(mods.psve, "_call_claude_vision", lambda *a, **k: next(replies))
    out = mods.psve.extract_specs_from_pdf_pages(str(pdf), [1, 2], "NOVA")
    assert out["_status"] == "degraded" and out["lumens"] == 800


def test_vision_keys_map_through_the_categorys_registry_fields(mods):
    out = mods.psve.map_vision_specs_to_product_metadata(
        {"lumens": 800, "slip_resistance": "R10", "beam_colour": "warm"}, "pendant_light")
    assert out["performance"] == {"lumens": 800, "slip_resistance": "R10"}
    assert out["unregistered_spec_fields"] == {"beam_colour": "warm"}


def test_a_field_outside_the_category_is_kept_flagged_not_mapped(mods):
    out = mods.psve.map_vision_specs_to_product_metadata({"lumens": 800}, "tiles")
    assert "lumens" not in out.get("performance", {})
    assert out["unregistered_spec_fields"] == {"lumens": 800}


def test_tier_b_all_pages_failed_is_a_tier_error(mods, monkeypatch):
    monkeypatch.setattr(mods.psve, "extract_specs_from_pdf_pages",
                        lambda **k: {"_status": "failed", "_page_failures": [{"page": 0}]})
    sys.modules["app.services.products"].product_spec_vision_extractor = mods.psve
    out = mods.v2._tier_b_opus("x.pdf", [0], "NOVA")
    assert "_tier_error" in out


def test_property_extraction_has_no_rule_based_fallback(mods):
    src = (_PRODUCTS / "enhanced_material_property_extractor.py").read_text(encoding="utf-8")
    assert "_rule_based_extraction" not in src
    assert "call_with_schema" in src


def test_a_failed_property_call_is_a_failed_result(mods):
    ex = mods.emp.EnhancedMaterialPropertyExtractor(workspace_id="ws")
    ex._db_prompt = "system"

    async def _boom(*_a, **_k):
        raise mods.tool_call.ToolCallNotReturned("refused")

    ex._extract_with_claude = _boom
    res = asyncio.run(ex.extract("R10 tile", "NOVA"))
    assert res.status == "failed" and res.method == "failed"
    assert res.properties == {} and res.coverage_pct == 0.0


def test_per_category_confidence_is_kept(mods):
    ex = mods.emp.EnhancedMaterialPropertyExtractor(workspace_id="ws")
    ex._db_prompt = "system"

    async def _reply(*_a, **_k):
        return {"rows": [{"field": "rValue", "text": "R10", "numbers": [], "confidence": 0.8}]}

    ex._extract_with_claude = _reply
    res = asyncio.run(ex.extract("R10 tile", "NOVA"))
    assert res.properties == {"slipSafetyRatings": {"rValue": ["R10"]}}
    assert res.category_confidence == {"slipSafetyRatings": 0.8}
    assert res.status == "ok"


def test_the_property_schema_is_closed_and_strict(mods):
    schema = mods.emp._properties_output_config()["format"]["schema"]
    assert schema["additionalProperties"] is False
    row = schema["properties"]["rows"]["items"]
    assert row["additionalProperties"] is False and set(row["required"]) == set(row["properties"])
    # Strict output refuses >16 union parameters and an over-large grammar (both 400s).
    assert "anyOf" not in repr(schema) and "'type': [" not in repr(schema)
    assert len(repr(schema)) < 2000


def test_property_rows_decode_to_functional_metadata(mods):
    ex = mods.emp.EnhancedMaterialPropertyExtractor(workspace_id="ws")
    ex._db_prompt = "system"

    def row(_group, field, text="", numbers=(), conf=0.7):
        return {"field": field, "text": text, "numbers": list(numbers), "confidence": conf}

    async def _reply(*_a, **_k):
        return {"rows": [
            row("mechanicalPropertiesExtended", "breakingStrength", numbers=[1300]),
            row("mechanicalPropertiesExtended", "peiRating", numbers=[4], conf=0.9),
            row("waterMoistureResistance", "frostResistance", "yes", conf=0.6),
            row("waterMoistureResistance", "moldMildewResistant", ""),
            row("slipSafetyRatings", "rValue", "R10"),
            row("slipSafetyRatings", "rValue", "R11"),
        ]}

    ex._extract_with_claude = _reply
    res = asyncio.run(ex.extract("tile", "NOVA"))
    assert res.properties["mechanicalPropertiesExtended"] == {"breakingStrength": 1300, "peiRating": [4]}
    assert res.properties["waterMoistureResistance"] == {"frostResistance": True}
    assert res.properties["slipSafetyRatings"] == {"rValue": ["R10", "R11"]}
    assert res.category_confidence["mechanicalPropertiesExtended"] == 0.9
