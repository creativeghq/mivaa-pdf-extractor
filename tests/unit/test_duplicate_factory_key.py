"""Duplicate detection keys on the maker NAME, never on the repr of the nested `factory` object."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("numpy")

_PATH = Path(__file__).resolve().parents[2] / "app" / "services" / "search" / "duplicate_detection_service.py"


@pytest.fixture()
def service_cls():
    stubs = {
        "app.services.core.supabase_client": {"SupabaseClient": object},
        "app.utils.text_similarity": {
            "calculate_string_similarity": lambda *_a, **_k: 0.0,
            "calculate_text_similarity": lambda *_a, **_k: 0.0,
        },
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        for name, attrs in stubs.items():
            mod = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(mod, key, value)
            sys.modules[name] = mod
        spec = importlib.util.spec_from_file_location("_dup_under_test", _PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module.DuplicateDetectionService
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _factory(service_cls, metadata):
    return service_cls._extract_factory_info(object.__new__(service_cls), {"metadata": metadata})


def test_two_products_of_one_maker_match_however_complete_their_factory_object(service_cls):
    sparse = {"factory_name": "Marazzi", "factory": {"factory_name": "Marazzi"}}
    full = {"factory": {"factory_name": "Marazzi", "city": "Sassuolo", "phone": "+39"}}
    assert _factory(service_cls, sparse) == _factory(service_cls, full) == "marazzi"


def test_a_factory_object_without_a_name_is_not_an_identity(service_cls):
    assert _factory(service_cls, {"factory": {"city": "Sassuolo"}}) is None
