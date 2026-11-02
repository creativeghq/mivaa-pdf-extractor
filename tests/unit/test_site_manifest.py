"""A series stands in for its crops only on a confirmed gallery of the maker's own photos."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("site_manifest_under_test", _ROOT / "app/services/discovery/site_manifest.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)


def _product(status, n):
    return {"metadata": {"site_manifest": {"status": status, "photos": [f"https://keros.com/p{i}.jpg" for i in range(n)]}}}


def test_only_a_found_gallery_of_enough_photos_counts():
    assert len(sm.gallery(_product("found", 3))) == 3
    assert sm.gallery(_product("found", 2)) == []
    assert sm.gallery(_product("not_found", 5)) == []
    assert sm.gallery({"metadata": {}}) == []


def test_the_manifest_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("SITE_MANIFEST_ENABLED", "false")
    assert not sm.enabled()
