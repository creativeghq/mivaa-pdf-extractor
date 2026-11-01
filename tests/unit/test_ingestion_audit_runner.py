import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sent = []


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.services", "app.services.pdf"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
flow_stub = types.ModuleType("app.services.flow_events")
flow_stub.emit_flow_event = lambda event_type, data, url, key: sent.append((event_type, data)) or True
sys.modules["app.services.flow_events"] = flow_stub
_load("app.services.pdf.ocr_completeness", "app/services/pdf/ocr_completeness.py")
_load("app.services.pdf.text_layer_extractor", "app/services/pdf/text_layer_extractor.py")
_load("app.services.pdf.ingestion_audit", "app/services/pdf/ingestion_audit.py")
_load("app.services.pdf.catalogue_tables", "app/services/pdf/catalogue_tables.py")
runner = _load("ingestion_audit_runner_under_test", "app/services/pdf/ingestion_audit_runner.py")

fitz = pytest.importorskip("fitz")


def make_pdf(codes):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i in range(30):
        page.insert_text((40, 40 + i * 12), "radiator heat output data sheet line", fontsize=8)
    for i, code in enumerate(codes):
        page.insert_text((40, 440 + i * 14), f"{code} {100 + i} {200 + i}", fontsize=9)
    return doc.tobytes()


CODES = [f"80{i:05d}" for i in range(12)]


def collected(codes_saved):
    tables = [{"id": "t1", "product_id": "p1", "page_number": 1, "headers": ["Code", "A", "B"], "extractor": "pymupdf-text", "metadata": {"title": "T"},
               "table_data": {"rows": [[c, str(100 + i), str(200 + i)] for i, c in enumerate(codes_saved)]}}]
    text = " ".join(f"{c} {100 + i} {200 + i}" for i, c in enumerate(codes_saved))
    chunks = [{"id": "c1", "product_id": "p1", "content": text, "has_text_embedding": True, "metadata": {"source": "table_block", "table_id": "t1", "page_number": 1}}]
    layout = [{"page_number": 1, "layout_elements": [{"text_content": text, "metadata": {}}], "analysis_metadata": {"pdf_idx": 0, "position": "single", "route": "text_layer"}}]
    return {"products": [{"id": "p1", "name": "P"}], "layout": layout, "chunks": chunks, "tables": tables, "embedded": {"p1"}, "images": {"p1": 1}}


def test_a_table_stored_under_another_page_number_still_counts_for_the_pages_its_product_owns():
    data = collected(CODES)
    data["products"] = [{"id": "p1", "name": "P", "metadata": {"page_range": [1]}}]
    data["tables"][0]["page_number"] = 99
    data["chunks"][0]["metadata"] = {"source": "table_block", "table_id": "t1", "page_number": 99}
    data["layout"][0]["layout_elements"] = []
    result = runner.build_audit({"id": "d"}, data, make_pdf(CODES), "post_embed")
    assert result.metrics["pages_with_missing_numbers"] == 0, result.as_dict()
    data["products"] = [{"id": "p1", "name": "P", "metadata": {"page_range": [5]}}]
    assert runner.build_audit({"id": "d"}, data, make_pdf(CODES), "post_embed").metrics["pages_with_missing_numbers"] == 1


def test_a_page_whose_numbers_are_all_stored_passes():
    result = runner.build_audit({"id": "d"}, collected(CODES), make_pdf(CODES), "post_embed")
    assert result.status == "pass", result.as_dict()
    assert result.metrics["code_coverage"] == 1.0


def test_codes_printed_in_the_pdf_but_not_stored_fail_the_audit():
    result = runner.build_audit({"id": "d"}, collected(CODES[:6]), make_pdf(CODES), "post_embed")
    assert result.status == "fail"
    assert {f.code for f in result.findings} & {"codes_missing", "numbers_missing"}


def test_without_a_source_file_the_result_is_unverifiable_never_a_pass():
    result = runner.build_audit({"id": "d"}, collected(CODES), None, "post_embed")
    assert result.status == "unverifiable"


class Query:
    def __init__(self, db, name):
        self.db, self.name, self.filters = db, name, {}
        self.op, self.payload = "select", None

    def select(self, *_):
        return self

    def insert(self, payload):
        self.op, self.payload = "insert", payload
        return self

    def eq(self, col, val):
        self.filters[col] = val
        return self

    @property
    def not_(self):
        return self

    def __getattr__(self, name):
        return lambda *a, **k: self

    def execute(self):
        if self.op == "insert":
            self.db.inserted.append((self.name, self.payload))
            return types.SimpleNamespace(data=[{"id": "audit-1"}])
        if self.name == "documents":
            if self.db.broken:
                raise RuntimeError("database unavailable")
            return types.SimpleNamespace(data=[{"id": "d", "workspace_id": "w", "filename": "cat.pdf", "storage_bucket": None, "storage_object_path": None}])
        if self.name == "roles":
            return types.SimpleNamespace(data=[{"id": "r"}])
        if self.name == "user_profiles":
            return types.SimpleNamespace(data=[{"user_id": "u1"}, {"user_id": "u2"}])
        return types.SimpleNamespace(data=[])


class Db:
    def __init__(self, broken=False):
        self.inserted, self.broken = [], broken
        self.client = self
        self.storage = self

    def table(self, name):
        return Query(self, name)

    def rpc(self, name, args):
        self.inserted.append((name, args))
        return types.SimpleNamespace(execute=lambda: types.SimpleNamespace(data=None))


SETTINGS = types.SimpleNamespace(supabase_url="https://x", supabase_service_role_key="k")


def run(db, **kw):
    sent.clear()
    return asyncio.run(runner.run_ingestion_audit("d", db, SETTINGS, **kw))


def test_a_document_with_no_source_is_stored_as_unverifiable_and_admins_are_told():
    db = Db()
    out = run(db, job_id="job-1")
    assert out["status"] == "unverifiable" and out["audit_id"] == "audit-1" and out["notified"] == 2
    stored = [p for n, p in db.inserted if n == "ingestion_audits"][0]
    assert stored["status"] == "unverifiable" and stored["document_id"] == "d" and stored["job_id"] == "job-1"
    assert [e[0] for e in sent] == ["ingestion_audit_failed", "ingestion_audit_failed"]
    assert sent[0][1]["document_id"] == "d" and sent[0][1]["action_url"].startswith("/admin/")
    assert any(n == "append_stage_history" for n, _ in db.inserted) and any(n == "merge_background_job_metadata" for n, _ in db.inserted)


def test_an_audit_that_crashes_is_recorded_as_audit_error_not_as_passed():
    db = Db(broken=True)
    out = run(db)
    assert out["status"] == "audit_error"
    assert out["findings"][0]["code"] == "audit_crashed"


def test_a_dry_run_stores_and_notifies_nobody():
    db = Db()
    out = run(db, persist=False, notify=False)
    assert out["audit_id"] is None and out["notified"] == 0
    assert not [1 for n, _ in db.inserted if n == "ingestion_audits"] and sent == []
