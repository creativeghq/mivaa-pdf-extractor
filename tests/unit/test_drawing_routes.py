"""The drawing routes: trusted-service gate, storage-path confinement, size limit, response shape."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("numpy")
fitz = pytest.importorskip("pymupdf")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ROUTE = Path(__file__).resolve().parents[2] / "app" / "api" / "drawing_routes.py"


def _load_drawings():
    """Load the package by path: other unit tests replace `app` with stub modules."""
    name = "app.services.drawings"
    pkg = Path(__file__).resolve().parents[2] / "app" / "services" / "drawings"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", "") == str(pkg / "__init__.py"):
        return existing
    for key in [k for k in sys.modules if k.startswith(name + ".")]:
        del sys.modules[key]
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Storage:
    def __init__(self, files):
        self.files = files
        self.calls = []

    def from_(self, bucket):
        def download(path):
            self.calls.append((bucket, path))
            return self.files.get(path)
        return types.SimpleNamespace(download=download)


@pytest.fixture
def client(monkeypatch):
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)
    page.insert_text((100, 100), "SCALE 1:50")
    pdf = doc.tobytes()
    storage = _Storage({"project-docs/p1/plan.pdf": pdf, "project-docs/p1/big.pdf": b"x" * (60 * 1024 * 1024 + 1)})

    async def require_trusted_service(request: Request):
        if request.headers.get("x-test-trusted") != "1":
            raise HTTPException(status_code=403, detail="no")
        return None

    deps = types.ModuleType("app.dependencies")
    deps.require_trusted_service = require_trusted_service
    sb = types.ModuleType("app.services.core.supabase_client")
    sb.SupabaseClient = object
    sb.get_supabase_client = lambda: types.SimpleNamespace(client=types.SimpleNamespace(storage=storage))
    monkeypatch.setitem(sys.modules, "app.dependencies", deps)
    monkeypatch.setitem(sys.modules, "app.services.core.supabase_client", sb)
    _load_drawings()

    spec = importlib.util.spec_from_file_location("_drawing_routes_under_test", ROUTE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.router)
    return TestClient(app), storage


def _body(path="project-docs/p1/plan.pdf", bucket="pdf-documents"):
    return {"storage_bucket": bucket, "storage_object_path": path}


def test_untrusted_caller_refused(client):
    tc, storage = client
    assert tc.post("/api/internal/drawings/analyze", json=_body()).status_code == 403
    assert storage.calls == []


@pytest.mark.parametrize("bucket,path", [
    ("generation-images", "project-docs/p1/plan.pdf"),
    ("pdf-documents", "inbox/t1/plan.pdf"),
    ("pdf-documents", "project-docs/../inbox/plan.pdf"),
])
def test_path_confinement(client, bucket, path):
    tc, storage = client
    r = tc.post("/api/internal/drawings/analyze", json=_body(path, bucket), headers={"x-test-trusted": "1"})
    assert r.status_code == 403
    assert storage.calls == []


def test_size_limit(client):
    tc, _ = client
    r = tc.post("/api/internal/drawings/analyze", json=_body("project-docs/p1/big.pdf"), headers={"x-test-trusted": "1"})
    assert r.status_code == 413


def test_analyze_and_find_similar(client):
    tc, _ = client
    r = tc.post("/api/internal/drawings/analyze", json=_body(), headers={"x-test-trusted": "1"})
    assert r.status_code == 200
    body = r.json()
    assert body["page_count"] == 1 and body["pages"][0]["scale"]["ratio"] == 50
    r = tc.post(
        "/api/internal/drawings/find-similar",
        json={**_body(), "page": 1, "bbox": [0, 0, 1, 1]},
        headers={"x-test-trusted": "1"},
    )
    assert r.status_code == 200
    assert r.json() == {"matches": [], "example_items": 0, "reason": "not_a_vector_page"}
    r = tc.post(
        "/api/internal/drawings/find-similar",
        json={**_body(), "page": 3, "bbox": [0, 0, 1, 1]},
        headers={"x-test-trusted": "1"},
    )
    assert r.status_code == 422
