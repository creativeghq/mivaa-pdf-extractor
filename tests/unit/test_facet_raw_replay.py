"""attributes_raw keeps every key, list facets re-resolve per raw, and the replay can be forced.

#494 comment 4 point 6 and the facet findings: `attributes_raw` held only canonicalizable
keys, so a field approved later never replayed; reuse handed every value of a list facet the
FIRST canonical; a translation failure dropped Greek values while the run reported ok.
Modules are loaded by path with stubbed dependencies (CI installs pytest only).
"""
import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_FACETS = _ROOT / "app" / "services" / "facets"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def facets():
    names = ["app", "app.services", "app.services.metadata", "app.services.facets",
             "app.services.core", "app.services.core.supabase_client",
             "app.services.metadata.field_registry", "app.services.facets.facet_translator",
             "app.services.facets.facet_whitelist", "app.services.facets.facet_canonicalizer",
             "app.services.integrations", "app.services.integrations.platform_secret_resolver"]
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for pkg in ("app", "app.services", "app.services.metadata", "app.services.facets",
                    "app.services.core", "app.services.integrations"):
            sys.modules[pkg] = types.ModuleType(pkg)
        reg = types.ModuleType("app.services.metadata.field_registry")

        class _Registry:
            is_loaded = True

            async def ensure_loaded(self, **_kw):
                return None

            @staticmethod
            def is_canonicalizable(key):
                return key in {"color", "finish"}

        reg.field_registry = _Registry()
        sys.modules["app.services.metadata.field_registry"] = reg
        sys.modules["app.services.core.supabase_client"] = types.SimpleNamespace(
            get_supabase_client=lambda: None)
        secrets = types.ModuleType("app.services.integrations.platform_secret_resolver")
        secrets.resolve_secret = lambda key: types.SimpleNamespace(value=None, source="missing")
        sys.modules["app.services.integrations.platform_secret_resolver"] = secrets

        translator = _load("app.services.facets.facet_translator", _FACETS / "facet_translator.py")
        _load("app.services.facets.facet_whitelist", _FACETS / "facet_whitelist.py")
        canon = _load("app.services.facets.facet_canonicalizer", _FACETS / "facet_canonicalizer.py")
        sys.modules["app.services.facets"].canonicalize_product_attributes = \
            canon.canonicalize_product_attributes
        recanon = _load("_facet_recanon_under_test", _FACETS / "facet_recanonicalization.py")
        yield types.SimpleNamespace(translator=translator, canon=canon, recanon=recanon)
    finally:
        sys.modules.pop("_facet_recanon_under_test", None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_attributes_raw_keeps_keys_that_are_not_canonicalizable(facets):
    raw = facets.canon.collect_raw_attributes({
        "color": ["White", "Sand"], "lumens": 800, "beam_angle": "24°",
        "sku": "ABC-1", "_extraction_metadata": {"x": 1}, "grout": {"a": "b"},
        "notes": "x" * 400,
    })
    assert raw == {"color": ["White", "Sand"], "lumens": ["800"], "beam_angle": ["24°"]}


def test_a_list_facet_is_never_reused_from_its_first_canonical(facets):
    c = facets.canon.FacetCanonicalizer(None, None)
    assert c._lookup_prior_canonical({"color": ["white", "beige"]}, "color") is None
    assert c._lookup_prior_canonical({"finish": "matte"}, "finish") == "matte"


def test_reuse_requires_the_raw_to_be_the_only_recorded_raw(facets):
    c = facets.canon.FacetCanonicalizer(None, None)

    async def _existing(_pid):
        return {"finish": "matte"}, {"finish": ["Matt", "Satin"]}

    seen = {}

    async def _fresh(fresh, **_kw):
        seen["fresh"] = fresh
        return [], False

    c._load_existing = _existing
    c._resolve_fresh = _fresh
    asyncio.run(c.canonicalize_product({"finish": "Matt"}, source="t", product_id="p"))
    assert [f for f, _r, _n in seen["fresh"]] == ["finish"], (
        "two raws recorded for a scalar facet: the stored canonical may belong to the other one")


def test_reuse_can_be_switched_off(facets):
    c = facets.canon.FacetCanonicalizer(None, None)

    async def _existing(_pid):
        return {"finish": "matte"}, {"finish": ["Matt"]}

    seen = {}

    async def _fresh(fresh, **_kw):
        seen["fresh"] = fresh
        return [], False

    c._load_existing = _existing
    c._resolve_fresh = _fresh
    asyncio.run(c.canonicalize_product({"finish": "Matt"}, source="t", product_id="p",
                                       reuse_existing=False))
    assert len(seen["fresh"]) == 1


def test_an_untranslatable_batch_is_degraded_not_ok(facets):
    out, degraded = asyncio.run(
        facets.translator.translate_facet_values_with_status([("color", "Λευκό"), ("color", "White")]))
    assert degraded is True, "the key is unresolved, so the Greek value is missing, not absent"
    assert out == {("color", "White"): "white"}


def test_the_replay_merges_now_canonicalizable_metadata_over_attributes_raw(facets):
    product = {
        "attributes_raw": {"color": ["White"], "lumens": ["800"]},
        "metadata": {"finish": "Matt", "color": ["Sand"], "lumens": 900, "_x": 1},
    }
    src = facets.recanon._replay_source(product)
    assert src == {"color": ["White", "Sand"], "lumens": ["800"], "finish": ["Matt"]}


def test_force_mode_exists_and_disables_reuse(facets):
    src = (_FACETS / "facet_recanonicalization.py").read_text(encoding="utf-8")
    assert "force: bool = False" in src
    assert "reuse_existing=not force" in src
    assert "q.lte('facet_canonicalization_version', target_version)" in src
