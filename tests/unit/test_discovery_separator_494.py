"""#494 M2: discovery, separator, OCR icons, legends, tables, boundaries.

Behaviour tests load the modules by path with their `app.*` imports stubbed, because
`app.services.__init__` reaches for a Supabase client the unit environment lacks.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
_PDS = _APP / "services" / "discovery" / "product_discovery_service.py"
_STAGE0 = _APP / "api" / "pdf_processing" / "stage_0_discovery.py"
_STAGE1 = _APP / "api" / "pdf_processing" / "stage_1_layout_precompute.py"
_STAGE2 = _APP / "api" / "pdf_processing" / "stage_2_chunking.py"
_OCR = _APP / "services" / "pdf" / "ocr_service.py"
_LEGEND = _APP / "services" / "knowledge" / "catalog_legend_extractor_v2.py"
_REGION = _APP / "services" / "pdf" / "region_crop_extractor.py"
_LAYOUT = _APP / "services" / "discovery" / "catalog_layout_analyzer.py"
_TABLE = _APP / "services" / "pdf" / "table_extraction.py"
_BOUNDARY = _APP / "services" / "ai_validation" / "boundary_detector.py"


def _load_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_blank = _load_path(_ROOT / "scripts" / "comment_budget.py", "_cb_494").blank_comments


def _code(path: Path) -> str:
    return _blank(path.read_text(encoding="utf-8"))


def _stub(monkeypatch, name: str, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules or not getattr(sys.modules[parent], "_m2_stub", False):
            pkg = types.ModuleType(parent)
            pkg.__path__ = []
            pkg._m2_stub = True
            monkeypatch.setitem(sys.modules, parent, pkg)
    mod = types.ModuleType(name)
    mod._m2_stub = True
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


class _Registry:
    is_loaded = False


@pytest.fixture
def pds(monkeypatch):
    class _Settings:
        default_workspace_id = "ws"
        anthropic_model_validation = "claude-opus-5"

    _stub(monkeypatch, "app.schemas.jobs", ProcessingStage=object)
    _stub(monkeypatch, "app.services.core.ai_call_logger", AICallLogger=object)
    _stub(monkeypatch, "app.services.metadata.dynamic_metadata_extractor", DynamicMetadataExtractor=object)
    _stub(monkeypatch, "app.services.metadata.metadata_shape", flatten_extracted_metadata=lambda d: d)
    _stub(monkeypatch, "app.services.metadata.field_registry", field_registry=_Registry())
    _stub(monkeypatch, "app.services.utilities.prompt_templates", get_prompt_template_from_db=None)
    _stub(monkeypatch, "app.config", get_settings=lambda: _Settings())
    _stub(
        monkeypatch, "app.utils.pdf_to_images",
        analyze_pdf_layout=None, get_physical_page_text=None, PDFLayoutAnalysis=object,
        PDFToImagesConverter=None,
    )
    mod = _load_path(_PDS, "_pds_494")
    svc = mod.ProductDiscoveryService.__new__(mod.ProductDiscoveryService)
    svc.logger = __import__("logging").getLogger("pds_test")
    svc.model = "claude"
    svc.tracker = None
    return mod, svc


# ── separator: page ranges ──────────────────────────────────────────────────

def test_supplied_page_range_is_not_overwritten(pds):
    mod, svc = pds
    a = mod.ProductInfo(name="A", page_range=[3, 4, 5], metadata={"_start_page": 3})
    b = mod.ProductInfo(name="B", page_range=[], metadata={"_start_page": 5})
    svc._calculate_page_ranges_from_start_pages([a, b], 40)
    assert a.page_range == [3, 4, 5]


def test_last_product_runs_to_the_document_end_not_start_plus_ten(pds):
    mod, svc = pds
    a = mod.ProductInfo(name="A", page_range=[], metadata={"_start_page": 2})
    b = mod.ProductInfo(name="B", page_range=[], metadata={"_start_page": 5})
    svc._calculate_page_ranges_from_start_pages([a, b], 40)
    assert a.page_range == [2, 3, 4]
    assert b.page_range[0] == 5 and b.page_range[-1] == 40


def test_last_product_stops_before_the_next_non_product_section(pds):
    mod, svc = pds
    b = mod.ProductInfo(name="B", page_range=[], metadata={"_start_page": 5})
    svc._calculate_page_ranges_from_start_pages([b], 40, {30: "legend", 3: "toc"})
    assert b.page_range == list(range(5, 30))


def test_two_products_on_one_page_keep_that_page(pds):
    mod, svc = pds
    a = mod.ProductInfo(name="A", page_range=[], metadata={"_start_page": 4})
    b = mod.ProductInfo(name="B", page_range=[], metadata={"_start_page": 4})
    svc._calculate_page_ranges_from_start_pages([a, b], 10)
    assert 4 in a.page_range and 4 in b.page_range


def test_toc_is_judged_by_content_not_by_sharing(pds):
    _, svc = pds
    names = ["Alba", "Brera", "Corso", "Duomo", "Etna", "Fiume"]
    pages = {
        2: "contents\nalba 12\nbrera 14\ncorso 16\nduomo 18\netna 20\nfiume 22",
        12: "alba\n60x60 matt\nbrera teaser",
        14: "brera\ncorso\nshared spread of two ranges",
    }
    toc = svc._detect_toc_pages(pages, names)
    assert toc == {2}


def test_toc_classification_marks_a_page(pds):
    _, svc = pds
    assert svc._detect_toc_pages({7: "x"}, ["a", "b"], {7: "Index"}) == {7}


def test_fallback_detection_finds_a_product_in_the_first_pages(pds):
    _, svc = pds
    pages = {1: "cover", 2: "alba\nporcelain 60x60", 3: "brera\nmore", 4: "brera detail"}
    found = svc._detect_product_pages_optimized(pages, "Alba", 4, ["Alba", "Brera"])
    assert found and found[0] == 2


# ── discovery batches ───────────────────────────────────────────────────────

def test_every_batch_is_read_and_catalog_factory_survives(pds):
    mod, svc = pds
    calls = []

    async def build(text, *a, **k):
        return text

    async def discover(prompt, job_id=None):
        calls.append(prompt)
        n = len(calls)
        if n == 2:
            return {"products": [], "catalog_factory": "Maker", "catalog_factory_group": "Group"}
        if n == 5:
            return {"products": [{"name": "Late", "start_page": 90}], "page_classification": {"88": "index"}}
        return {"products": []}

    svc._build_index_scan_prompt = build
    svc._discover_with_claude = discover
    text = "x" * 450_000
    cat = asyncio.run(svc._iterative_batch_discovery(text, 100, ["products"], None, "ws", True))
    assert len(calls) == 5
    assert [p.name for p in cat.products] == ["Late"]
    assert cat.catalog_factory == "Maker" and cat.catalog_factory_group == "Group"
    assert cat.content_classification == {88: "index"}
    assert cat.discovery_coverage["complete"] is True
    assert cat.discovery_status == "ok"


def test_discovery_metadata_carries_no_synthetic_defaults(pds):
    _, svc = pds
    cat = svc._parse_discovery_results({"products": [{"name": "A", "start_page": 3}]}, 10, ["products"])
    assert cat.products[0].metadata == {"_start_page": 3}


# ── vision retry ────────────────────────────────────────────────────────────

def _vision_stubs(monkeypatch, response=None, raises=None):
    class Conv:
        def __init__(self, **k):
            pass

        def convert_pdf_to_images(self, **k):
            return [(1, "b64")]

    sys.modules["app.utils.pdf_to_images"].PDFToImagesConverter = Conv

    async def call(**k):
        if raises:
            raise raises
        return response

    _stub(monkeypatch, "app.services.core.claude_helper", tracked_claude_call_async=call)


def _prompt_ok(svc):
    async def build(**k):
        return "prompt"
    svc._build_discovery_prompt = build


def test_a_failed_vision_call_is_a_failure_not_zero_products(pds, monkeypatch):
    mod, svc = pds
    _vision_stubs(monkeypatch, raises=RuntimeError("credit"))
    _prompt_ok(svc)
    with pytest.raises(mod.VisionRetryFailed):
        asyncio.run(svc._vision_retry_discovery("f.pdf", 3, ["products"], None, "ws", True, None))


def test_a_refused_vision_call_is_a_failure(pds, monkeypatch):
    mod, svc = pds
    _vision_stubs(monkeypatch, response=types.SimpleNamespace(stop_reason="refusal", content=[]))
    _prompt_ok(svc)
    with pytest.raises(mod.VisionRetryFailed):
        asyncio.run(svc._vision_retry_discovery("f.pdf", 3, ["products"], None, "ws", True, None))


def test_a_clean_empty_vision_answer_is_a_verdict(pds, monkeypatch):
    mod, svc = pds
    block = types.SimpleNamespace(type="tool_use", input={"products": []})
    _vision_stubs(monkeypatch, response=types.SimpleNamespace(stop_reason="tool_use", content=[block]))
    _prompt_ok(svc)
    cat = asyncio.run(svc._vision_retry_discovery("f.pdf", 3, ["products"], None, "ws", True, None))
    assert cat.products == []


def test_discovery_builder_fences_the_user_request():
    code = _code(_PDS)
    start = code.index("async def _build_discovery_prompt(")
    body = code[start:code.index("async def _build_index_scan_prompt(")]
    assert '_fence_user_request(agent_prompt)' in body


def test_stage_0_fails_a_zero_product_job_without_a_verdict():
    code = _code(_STAGE0)
    at = code.index('_disc_status != "ok"')
    window = code[at:at + 2000]
    assert "if not catalog.products:" in window and "raise RuntimeError(" in window
    assert at < code.index("'catalog_cache': {"), "the marker check must run before the catalog is cached"


# ── layout precompute / chunking ────────────────────────────────────────────

def test_ocr_circuit_breaker_latches_on_zero_ocr_successes():
    code = _code(_STAGE1)
    body = code[code.index("async def _bounded("):code.index("await asyncio.gather(*[_bounded")]
    assert '_state["ocr_ok"] == 0' in body
    assert '_state["persisted"] == 0' not in body


def test_chunking_reports_failed_pages():
    code = _code(_STAGE2)
    assert code.count("'failed_pages': failed_pages") >= 2
    assert "'chunking_status': 'partial' if (failed_pages or extraction_failed) else 'ok'" in code


# ── OCR icon metadata ───────────────────────────────────────────────────────

def test_icon_metadata_is_schema_constrained_async_and_cheap():
    code = _code(_OCR)
    body = code[code.index("async def extract_icon_metadata("):code.index("def get_endpoint_stats(")]
    assert "call_with_schema(" in body
    assert "tracked_claude_call(" not in body, "a sync model call blocks the event loop"
    assert "anthropic_model_classification" in body
    assert "claude-opus" not in body
    assert "re.search" not in body, "regex salvage of the reply is back"
    assert "ICON_OCR_OPEN" in body, "OCR tokens are not fenced as data"
    assert "return []" in body and body.count("return []") == 1, (
        "only a clean no-text OCR may return []; every failure must raise"
    )
    assert "_icon_field_names(category)" in body


# ── legends ─────────────────────────────────────────────────────────────────

@pytest.fixture
def legend(monkeypatch):
    pytest.importorskip("fitz")
    _stub(monkeypatch, "sentry_sdk", capture_exception=lambda *a, **k: None)
    _stub(monkeypatch, "app.services.core.anthropic_error_reporter", report_anthropic_failure=lambda *a, **k: None)
    _stub(monkeypatch, "app.services.utilities.prompt_registry", load_prompt=None)
    _stub(monkeypatch, "app.utils.tenancy", assert_document_in_workspace=lambda *a: None)
    return _load_path(_LEGEND, "_legend_494")


def test_legend_honours_applies_globally(legend):
    certs = legend._global_certifications("certification", {"certifications": [
        {"name": "ISO 9001", "applies_globally": True},
        {"name": "Line-X mark", "applies_globally": False},
    ]})
    assert certs == ["ISO 9001"]
    icons = {"icons": [{"category": "certification", "code": "CE"}]}
    assert legend._global_certifications("icons", {**icons, "applies_globally": False}) == []
    assert legend._global_certifications("icons", {**icons, "applies_globally": True}) == ["CE"]


def test_legend_icon_schema_accepts_certification(legend):
    schema = legend._legend_output_config("icons")["format"]["schema"]
    enum = schema["properties"]["icons"]["items"]["properties"]["category"]["enum"]
    assert "certification" in enum
    for t in legend.CATEGORY_BY_TYPE:
        s = legend._legend_output_config(t)["format"]["schema"]
        assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])


def test_legend_extracted_at_only_when_every_page_succeeded():
    code = _code(_LEGEND)
    assert '"extracted_at": None if failed_pages else now' in code
    assert "call_with_schema(" in code and "json.loads" not in code
    assert '"unknown"' not in code, "unknown pages are run through every prompt again"


# ── region crops / layout / tables / boundaries ─────────────────────────────

def test_region_crop_cache_read_failure_raises():
    code = _code(_REGION)
    body = code[code.index("async def _load_layout_regions_for_physical_pages("):]
    at = body.index("except Exception as exc:")
    assert "raise RegionCropCacheReadFailed(" in body[at:at + 400]


def test_layout_analyzer_regroups_cache_text_onto_sheets(monkeypatch):
    pytest.importorskip("fitz")
    _stub(monkeypatch, "sentry_sdk")
    layout = types.SimpleNamespace(physical_to_pdf_map={1: (0, "single"), 2: (1, "left"), 3: (1, "right")})
    _stub(monkeypatch, "app.utils.pdf_to_images", analyze_pdf_layout=lambda p: layout)
    mod = _load_path(_LAYOUT, "_layout_494")
    out = mod._cache_text_by_sheet("f.pdf", {1: "cover", 2: "left page", 3: "right page"}, 2)
    assert out == {0: "cover", 1: "left page\n\nright page"}


def test_table_spans_are_expanded():
    mod = _load_path(_TABLE, "_table_494")
    grid = mod.parse_html_table(
        "<table><tr><th rowspan=2>Code</th><th colspan=2>Size</th></tr>"
        "<tr><th>W</th><th>H</th></tr><tr><td>A1</td><td>60</td><td>120</td></tr></table>"
    )
    assert grid[0] == ["Code", "Size", "Size"]
    assert grid[1] == ["Code", "W", "H"]
    assert grid[2] == ["A1", "60", "120"]


def test_boundary_detector_consumes_the_vector_and_fails_without_one(monkeypatch):
    pytest.importorskip("numpy")
    _stub(monkeypatch, "app.services.embeddings.real_embeddings_service", RealEmbeddingsService=object)
    _stub(monkeypatch, "app.services.core.ai_call_logger", AICallLogger=object)
    mod = _load_path(_BOUNDARY, "_boundary_494")
    det = mod.BoundaryDetector.__new__(mod.BoundaryDetector)

    class Svc:
        def __init__(self, ok):
            self.ok = ok

        async def generate_text_embedding(self, query, **k):
            return {"success": True, "embedding": [1.0, 0.0]} if self.ok else {"success": False, "error": "x"}

    det.embeddings_service = Svc(True)
    vecs = asyncio.run(det._get_embeddings_batch(["a", "b"], None))
    assert [list(v) for v in vecs] == [[1.0, 0.0], [1.0, 0.0]]
    det.embeddings_service = Svc(False)
    with pytest.raises(mod.BoundaryEmbeddingFailed):
        asyncio.run(det._get_embeddings_batch(["a"], None))
