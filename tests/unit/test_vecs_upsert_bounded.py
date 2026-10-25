"""VECS writes share one engine's worth of the pooler: capped writers, retried while it is full."""

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "app/services/embeddings/vecs_service.py"


def _load():
    vecs = types.ModuleType("vecs")
    vecs.Client = object
    collection = types.ModuleType("vecs.collection")
    collection.Collection = object
    collection.IndexMethod = object
    weights = types.ModuleType("app.services.search.weight_profiles")
    weights.image_only_weights = lambda *a, **k: {}
    for name in ("app", "app.services", "app.services.search"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vecs"] = vecs
    sys.modules["vecs.collection"] = collection
    sys.modules["app.services.search.weight_profiles"] = weights

    spec = importlib.util.spec_from_file_location("vecs_service_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Collection:
    def __init__(self, fail_times=0, message="FATAL: (EMAXCONNSESSION) max clients reached", hold=0.0):
        self.calls = 0
        self.fail_times = fail_times
        self.message = message
        self.hold = hold
        self.live = 0
        self.peak = 0
        self._lock = threading.Lock()

    def upsert(self, records):
        with self._lock:
            self.calls += 1
            self.live += 1
            self.peak = max(self.peak, self.live)
            n = self.calls
        try:
            time.sleep(self.hold)
            if n <= self.fail_times:
                raise RuntimeError(self.message)
        finally:
            with self._lock:
                self.live -= 1


def test_a_full_pooler_is_retried_until_it_clears():
    mod = _load()
    col = _Collection(fail_times=2)
    mod._upsert_bounded(col, records=[("a", [0.0], {})])
    assert col.calls == 3


def test_any_other_error_is_raised_at_once():
    mod = _load()
    col = _Collection(fail_times=5, message="relation does not exist")
    try:
        mod._upsert_bounded(col, records=[])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
    assert col.calls == 1


def test_writers_never_exceed_the_cap():
    mod = _load()
    col = _Collection(hold=0.05)
    threads = [
        threading.Thread(target=mod._upsert_bounded, args=(col,), kwargs={"records": []})
        for _ in range(24)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert col.calls == 24
    assert col.peak <= 8
