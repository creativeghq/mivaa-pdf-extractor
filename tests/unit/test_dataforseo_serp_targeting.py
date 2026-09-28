"""Guard: a tracked keyword's device and city reach DataForSEO.

`device` was stored per keyword and never sent, so every mobile keyword was checked
against the desktop SERP.
"""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MOD = _ROOT / "app" / "services" / "integrations" / "dataforseo_serp_targeting.py"
_CLIENT = _ROOT / "app" / "services" / "integrations" / "dataforseo_unified_client.py"
_ROUTES = _ROOT / "app" / "api" / "seo_agent_routes.py"


def _load():
    spec = importlib.util.spec_from_file_location("dataforseo_serp_targeting", _MOD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t = _load()


def _task(**kw):
    base = dict(keyword="πλακάκια", country_location=2300, language_code="el", depth=100, paa_depth=1)
    base.update(kw)
    return t.organic_task(**base)


class TestDevice:
    def test_mobile_is_sent(self):
        assert _task(device="mobile")["device"] == "mobile"

    def test_default_is_desktop(self):
        assert _task()["device"] == "desktop"

    def test_unknown_device_is_refused_not_defaulted(self):
        with pytest.raises(ValueError):
            _task(device="tablet")


class TestLocation:
    def test_country_level_when_no_city(self):
        assert _task()["location_code"] == 2300

    def test_city_replaces_the_country(self):
        assert _task(location_code=1012019)["location_code"] == 1012019
        assert _task(location_code="1012019")["location_code"] == 1012019

    @pytest.mark.parametrize("bad", [0, -5, "athens", "12.5", True])
    def test_bad_city_is_refused(self, bad):
        with pytest.raises(ValueError):
            _task(location_code=bad)

    def test_locations_path_needs_an_iso_country(self):
        assert t.locations_path("GR") == "/serp/google/locations/gr"
        for bad in ("", None, "GRC", "../x"):
            with pytest.raises(ValueError):
                t.locations_path(bad)


def test_the_client_and_dispatcher_use_it():
    client = _CLIENT.read_text(encoding="utf-8")
    assert "serp_targeting.organic_task(" in client
    assert "device=device, location_code=location_code" in client
    assert '"serp_google_locations"' in _ROUTES.read_text(encoding="utf-8")
