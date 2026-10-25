"""Threads that ask for the VECS client or one collection at the same instant share ONE of each."""

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "app/services/embeddings/vecs_service.py"


def _load(created):
    class _Collection:
        def create_index(self, **kw):
            pass

    class _Client:
        def get_or_create_collection(self, name, dimension):
            time.sleep(0.05)
            created["collections"] += 1
            return _Collection()

    def create_client(_conn):
        time.sleep(0.05)
        created["clients"] += 1
        return _Client()

    vecs = types.ModuleType("vecs")
    vecs.Client = object
    vecs.create_client = create_client
    collection = types.ModuleType("vecs.collection")
    collection.Collection = object
    collection.IndexMethod = types.SimpleNamespace(hnsw="hnsw", ivfflat="ivfflat")
    weights = types.ModuleType("app.services.search.weight_profiles")
    weights.image_only_weights = lambda *a, **k: {}
    for name in ("app", "app.services", "app.services.search"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vecs"] = vecs
    sys.modules["vecs.collection"] = collection
    sys.modules["app.services.search.weight_profiles"] = weights

    spec = importlib.util.spec_from_file_location("vecs_service_race_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    svc = mod.VecsService()
    svc._get_connection_string = lambda: "postgresql://unused"
    return svc


def _hammer(fn, n=12):
    gate = threading.Barrier(n)

    def run():
        gate.wait()
        fn()

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_a_burst_of_threads_builds_one_client():
    created = {"clients": 0, "collections": 0}
    svc = _load(created)
    _hammer(svc._get_client)
    assert created["clients"] == 1


def test_a_burst_of_threads_creates_one_collection_and_one_client():
    created = {"clients": 0, "collections": 0}
    svc = _load(created)
    _hammer(lambda: svc.get_or_create_collection("image_slig_embeddings", 768))
    assert created["clients"] == 1
    assert created["collections"] == 1


def test_different_collections_are_each_created_once():
    created = {"clients": 0, "collections": 0}
    svc = _load(created)
    names = ["a", "b", "c"]
    _hammer(lambda: [svc.get_or_create_collection(n, 8) for n in names])
    assert created["collections"] == 3 and created["clients"] == 1
