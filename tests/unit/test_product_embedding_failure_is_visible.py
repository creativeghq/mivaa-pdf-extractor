"""Guard: a product that failed to embed must not look like one nobody has embedded yet.

WHY THIS EXISTS (#348)
----------------------
``ProductEnrichmentService._create_product_embedding`` answered three different
questions with the same ``{}``:

  * the embedder raised,
  * the embedder ran and produced nothing,
  * there was nothing on the product worth embedding.

The row afterwards was identical in all three cases *and* identical to a product the
enrichment had never touched. That is pipeline convention 1 -- explicit failure
markers, not empty returns -- and it is the reason a retry sweep could not tell which
products to pick up. ``generate_all_embeddings`` already distinguishes the cases via
``success`` / ``error``; this service was throwing that away with
``.get("embeddings", {})`` and re-flattening it to ``{}`` in the except branch.

Underneath the reported symptom sat the thing that made it permanent: the vectors were
generated and then **discarded**. ``entity_id="temp"`` carried a comment saying it would
be updated with the product id; nothing updated it, nothing persisted the result, and
``_store_enrichment_results`` documented that embeddings live elsewhere. So a *successful*
enrichment also left ``text_embedding_1024`` NULL. A marker that says "failed" is only
worth writing if "ok" means something different on the row, which is why the storage half
is pinned here too.

WHAT IS PINNED
--------------
  1. Every exit from ``_create_product_embedding`` carries a ``status``.
  2. ``ok`` requires vectors; a raise, ``success: False``, an empty map and empty input
     each produce ``failed`` with a distinguishable ``reason``.
  3. A landed text vector reaches ``products.text_embedding_1024``.
  4. A vector of the wrong width is REFUSED and downgraded to ``failed`` -- same shape,
     different space is the one thing the column cannot detect for itself.
  5. Failure stamps ``metadata.embedding_failure`` -- the SAME key, shape and consumer
     as the Stage 0 marker, so ``text_embedding_backfill`` already sweeps it.
  6. Success CLEARS a prior marker, so the sweep stops re-picking the row.
  7. The write MERGES ``metadata`` / ``properties``. It used to replace both objects
     wholesale, which erased ``metadata.facet_canonicalization`` and the layout
     provenance -- i.e. the marker convention this fix joins.
  8. ``_find_related_products`` calls an RPC that EXISTS. ``search_similar_products``
     is not a function in this database and never has been, so the call 404'd into the
     warning handler and the answer was a permanent ``[]``.

CI CONSTRAINTS (see test_workspace_resolution.py -- this cost a deploy once)
---------------------------------------------------------------------------
CI installs pytest and nothing else: no supabase, no httpx, no pytest-asyncio. The
module under test imports all three transitively, so it is loaded BY PATH with its four
app-level dependencies stubbed in ``sys.modules``, and the stubs are removed again
afterwards -- a permanently registered bare ``app`` package breaks every test that
collects later. Coroutines are driven with ``asyncio.run`` for want of the marker.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _ROOT / "app" / "services" / "products" / "product_enrichment_service.py"


def _load_service_module():
    """Import the module by path with its heavy app imports stubbed.

    sys.modules is restored on the way out. Registering a bare ``app`` package and
    leaving it there is what made test_table_extraction.py hostile to its neighbours.
    """
    stub_names = [
        "app",
        "app.services",
        "app.services.embeddings",
        "app.services.embeddings.real_embeddings_service",
        "app.services.ai_validation",
        "app.services.ai_validation.real_quality_scoring_service",
        "app.services.core",
        "app.services.core.ai_client_service",
        "app.services.utilities",
        "app.services.utilities.prompt_registry",
    ]
    saved = {name: sys.modules.get(name) for name in stub_names}
    try:
        for name in stub_names:
            mod = ModuleType(name)
            mod.__path__ = []  # make it a package so submodule imports resolve
            sys.modules[name] = mod

        class _StubEmbeddings:
            def __init__(self, supabase_client=None, config=None):
                pass

        class _StubQuality:
            def __init__(self, supabase_client=None):
                pass

            def calculate_product_quality_score(self, product_data):
                return 0.5, {"stub": True}

        sys.modules["app.services.embeddings.real_embeddings_service"].RealEmbeddingsService = _StubEmbeddings
        sys.modules["app.services.ai_validation.real_quality_scoring_service"].RealQualityScoringService = _StubQuality
        sys.modules["app.services.core.ai_client_service"].get_ai_client_service = (
            lambda: SimpleNamespace(anthropic=None)
        )
        sys.modules["app.services.utilities.prompt_registry"].load_prompt = None
        sys.modules["app.services.utilities.prompt_registry"].render = None

        spec = importlib.util.spec_from_file_location(
            "product_enrichment_service_under_test", _MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


_svc = _load_service_module()
ProductEnrichmentService = _svc.ProductEnrichmentService

PRODUCT_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ID = "22222222-2222-2222-2222-222222222222"
WORKSPACE_ID = "99999999-9999-9999-9999-999999999999"


# ── Fake Supabase ────────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, rows, updates, unreadable):
        self._rows = rows
        self._updates = updates
        self._unreadable = unreadable
        self._op = None
        self._row_id = None

    def select(self, _cols):
        self._op = ("select", None)
        return self

    def update(self, payload):
        self._op = ("update", payload)
        return self

    def eq(self, _col, value):
        self._row_id = value
        return self

    def maybe_single(self):
        return self

    def execute(self):
        kind, payload = self._op
        if kind == "select":
            if self._unreadable:
                raise RuntimeError("PostgREST unavailable")
            row = self._rows.get(self._row_id)
            return _Resp(dict(row) if row else None)
        self._updates.append(payload)
        row = self._rows.setdefault(self._row_id, {})
        row.update(payload)
        return _Resp([dict(row)])


class _FakeRpc:
    def __init__(self, result, raises):
        self._result = result
        self._raises = raises

    def execute(self):
        if self._raises:
            raise RuntimeError(self._raises)
        return _Resp(self._result)


class _FakeClient:
    def __init__(self, rows=None, rpc_result=None, rpc_raises=None, unreadable=False):
        self.rows = rows or {}
        self.updates = []
        self.rpc_calls = []
        self._rpc_result = rpc_result if rpc_result is not None else []
        self._rpc_raises = rpc_raises
        self._unreadable = unreadable

    def table(self, _name):
        return _FakeTable(self.rows, self.updates, self._unreadable)

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        return _FakeRpc(self._rpc_result, self._rpc_raises)


def _service(**client_kwargs):
    client = _FakeClient(**client_kwargs)
    service = ProductEnrichmentService(SimpleNamespace(client=client))
    return service, client


def _stub_embedder(service, *, returns=None, raises=None):
    """Swap the class the service instantiates inside _create_product_embedding."""

    class _Stub:
        def __init__(self, supabase_client=None):
            pass

        async def generate_all_embeddings(self, **kwargs):
            _Stub.last_kwargs = kwargs
            if raises:
                raise RuntimeError(raises)
            return returns

    _Stub.last_kwargs = None
    _svc.RealEmbeddingsService = _Stub
    return _Stub


def _vec(n=1024, fill=0.01):
    return [fill] * n


def _ok_payload(vector=None):
    return {
        "success": True,
        "embeddings": {"text_1024": vector if vector is not None else _vec()},
        "metadata": {"model_versions": {"text": "voyage-4"}},
    }


# ── 1-2. every exit carries a status, and only vectors earn "ok" ─────────────────


def test_exception_reports_failed_not_empty():
    service, _ = _service()
    _stub_embedder(service, raises="voyage down")
    result = asyncio.run(
        service._create_product_embedding("Tile", "A tile", {"materials": ["stone"]})
    )
    assert result["status"] == "failed"
    assert result["reason"] == "exception"
    assert "voyage down" in result["error"]
    assert result["embeddings"] == {}


def test_service_reporting_failure_reports_failed():
    service, _ = _service()
    _stub_embedder(service, returns={"success": False, "error": "no_vectors_generated"})
    result = asyncio.run(service._create_product_embedding("Tile", "A tile", {}))
    assert result["status"] == "failed"
    assert result["reason"] == "no_vectors_generated"


def test_success_with_no_vectors_reports_failed():
    """`success: True` alongside an empty map is the exact case the embedder's own
    comment says callers must be able to see."""
    service, _ = _service()
    _stub_embedder(service, returns={"success": True, "embeddings": {}})
    result = asyncio.run(service._create_product_embedding("Tile", "A tile", {}))
    assert result["status"] == "failed"


def test_nothing_to_embed_is_its_own_reason_and_never_calls_the_embedder():
    """The embedding text is an f-string with its own labels, so it is never empty --
    embedding ". . Materials: . Colors:" buys a paid vector of the template."""
    service, _ = _service()
    stub = _stub_embedder(service, returns=_ok_payload())
    result = asyncio.run(service._create_product_embedding("", "", {}))
    assert result["status"] == "failed"
    assert result["reason"] == "no_embeddable_text"
    assert stub.last_kwargs is None


def test_vectors_report_ok_with_provenance_and_attribution():
    service, _ = _service()
    stub = _stub_embedder(service, returns=_ok_payload())
    result = asyncio.run(
        service._create_product_embedding(
            "Tile", "A tile", {}, product_id=PRODUCT_ID, workspace_id=WORKSPACE_ID
        )
    )
    assert result["status"] == "ok"
    assert result["model_versions"]["text"] == "voyage-4"
    # The ids are what attribute the Voyage spend; entity_id was the literal "temp".
    assert stub.last_kwargs["product_id"] == PRODUCT_ID
    assert stub.last_kwargs["workspace_id"] == WORKSPACE_ID
    assert stub.last_kwargs["entity_id"] == PRODUCT_ID


def test_every_return_is_an_envelope_with_a_status():
    """Structural backstop for exits a future edit might add."""
    import ast

    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_create_product_embedding"
    )
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "the method returns nothing at all"
    for node in returns:
        assert isinstance(node.value, ast.Dict), ast.unparse(node)
        keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        assert "status" in keys, f"return without a status: {ast.unparse(node)}"


# ── 3-4. the vector reaches the column, or it is refused ─────────────────────────


def _enrichment(embedding, **extra):
    payload = {
        "product_id": PRODUCT_ID,
        "material_properties": {},
        "enhanced_description": "A tile",
        "related_images": [],
        "related_products": [],
        "enrichment_timestamp": "2026-08-20T00:00:00",
        "embedding": embedding,
    }
    payload.update(extra)
    return payload


def _store(service, embedding, rows=None):
    return asyncio.run(service._store_enrichment_results(PRODUCT_ID, _enrichment(embedding)))


def test_landed_vector_is_written_to_the_column():
    service, client = _service(rows={PRODUCT_ID: {"metadata": {}, "properties": {}}})
    _store(
        service,
        {"status": "ok", "embeddings": {"text_1024": _vec()}, "model_versions": {"text": "voyage-4"}},
    )
    written = client.updates[-1]
    assert written["text_embedding_1024"].startswith("[")
    assert written["text_embedding_1024_model"] == "voyage-4"
    assert written["text_embedding_schema_version"] == 1
    assert "embedding_failure" not in written["metadata"]


def test_wrong_width_vector_is_refused_and_marked_failed():
    """768D in a 1024D column is the same shape in a different space -- Postgres,
    HNSW and cosine all accept it and rank it confidently."""
    service, client = _service(rows={PRODUCT_ID: {"metadata": {}, "properties": {}}})
    _store(service, {"status": "ok", "embeddings": {"text_1024": _vec(768)}})
    written = client.updates[-1]
    assert "text_embedding_1024" not in written
    assert written["metadata"]["embedding_failure"]["reason"] == "dim_mismatch_768"


def test_vectors_without_the_text_one_still_count_as_failed():
    service, client = _service(rows={PRODUCT_ID: {"metadata": {}, "properties": {}}})
    _store(service, {"status": "ok", "embeddings": {"color_aspect_1024": _vec()}})
    assert client.updates[-1]["metadata"]["embedding_failure"]["reason"] == "no_text_vector"


# ── 5-6. the marker, and its clearing ────────────────────────────────────────────


def test_failure_stamps_the_stage_0_marker_key():
    service, client = _service(rows={PRODUCT_ID: {"metadata": {}, "properties": {}}})
    _store(service, {"status": "failed", "embeddings": {}, "reason": "exception", "error": "boom"})
    marker = client.updates[-1]["metadata"]["embedding_failure"]
    # Same key text_embedding_backfill already sweeps and clears. A second vocabulary
    # would need a second sweep.
    assert marker["stage"] == "product_enrichment"
    assert marker["reason"] == "exception"
    assert marker["error"] == "boom"
    assert marker["failed_at"]


def test_success_clears_a_prior_marker():
    service, client = _service(
        rows={
            PRODUCT_ID: {
                "metadata": {"embedding_failure": {"stage": "stage_0_discovery"}},
                "properties": {},
            }
        }
    )
    _store(service, {"status": "ok", "embeddings": {"text_1024": _vec()}})
    metadata = client.updates[-1]["metadata"]
    assert "embedding_failure" not in metadata
    assert metadata["embedding_failure_resolved"]["stage"] == "stage_0_discovery"
    assert metadata["embedding_failure_resolved"]["resolved_by"] == "product_enrichment"


# ── 7. the write must not erase the marker convention it joins ───────────────────


def test_existing_metadata_and_properties_survive_the_write():
    service, client = _service(
        rows={
            PRODUCT_ID: {
                "metadata": {
                    "facet_canonicalization": {"status": "degraded"},
                    "extracted_from": "layout_analysis",
                },
                "properties": {"page_number": 4, "bounding_box": [1, 2, 3, 4]},
            }
        }
    )
    _store(service, {"status": "ok", "embeddings": {"text_1024": _vec()}})
    written = client.updates[-1]
    assert written["metadata"]["facet_canonicalization"] == {"status": "degraded"}
    assert written["metadata"]["extracted_from"] == "layout_analysis"
    assert written["metadata"]["enriched"] is True
    assert written["properties"]["page_number"] == 4


def test_failed_property_extraction_does_not_wipe_properties():
    """`_extract_material_properties_from_images` returns {} on failure -- replacing
    the column with it turned one broken image analysis into lost provenance."""
    service, client = _service(
        rows={PRODUCT_ID: {"metadata": {}, "properties": {"page_number": 4}}}
    )
    asyncio.run(
        service._store_enrichment_results(
            PRODUCT_ID,
            _enrichment({"status": "ok", "embeddings": {"text_1024": _vec()}}, material_properties={}),
        )
    )
    assert client.updates[-1]["properties"]["page_number"] == 4


def test_unreadable_row_writes_neither_jsonb_column():
    """Without the current value a merge is impossible, and writing would DESTROY
    markers rather than add one. Losing the marker is the lesser failure."""
    service, client = _service(rows={PRODUCT_ID: {"metadata": {}, "properties": {}}}, unreadable=True)
    _store(service, {"status": "failed", "embeddings": {}, "reason": "exception"})
    written = client.updates[-1]
    assert "metadata" not in written
    assert "properties" not in written
    assert written["quality_score"] == 0.5


# ── 8. the related-products RPC has to exist ─────────────────────────────────────


def test_related_products_calls_the_rpc_that_exists():
    service, client = _service(
        rpc_result=[{"product_id": PRODUCT_ID}, {"product_id": OTHER_ID}]
    )
    found = asyncio.run(
        service._find_related_products(
            product_id=PRODUCT_ID, query_vector=_vec(), workspace_id=WORKSPACE_ID
        )
    )
    name, params = client.rpc_calls[-1]
    assert name == "search_products_by_embedding"
    assert set(params) == {"query_embedding", "p_workspace_id", "p_limit"}
    # It returns `product_id`, not `id`, and has no exclude argument.
    assert found == [OTHER_ID]


def test_no_vector_means_no_rpc_call():
    service, client = _service()
    found = asyncio.run(
        service._find_related_products(
            product_id=PRODUCT_ID, query_vector=None, workspace_id=WORKSPACE_ID
        )
    )
    assert found == []
    assert client.rpc_calls == []


def test_the_rpc_that_never_existed_is_not_referenced_anywhere():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1] if source.count('"""') >= 2 else source
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("*"):
            continue
        assert "'search_similar_products'" not in stripped
        assert '"search_similar_products"' not in stripped
