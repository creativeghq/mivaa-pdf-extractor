"""Cancelling a running job stops it, and a deploy does not kill one (#513)."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("job_heartbeat_under_test", _ROOT / "app/services/tracking/job_heartbeat.py")
hb = importlib.util.module_from_spec(_spec)
sys.modules["job_heartbeat_under_test"] = hb
_spec.loader.exec_module(hb)
budget_spec = importlib.util.spec_from_file_location("comment_budget_for_lifecycle", _ROOT / "scripts/comment_budget.py")
budget = importlib.util.module_from_spec(budget_spec)
sys.modules["comment_budget_for_lifecycle"] = budget
budget_spec.loader.exec_module(budget)


def _code(relative):
    return budget.blank_comments((_ROOT / relative).read_text(encoding="utf-8"))


class _Rows:
    def __init__(self, statuses):
        self.statuses = statuses

    def table(self, _):
        return self

    def select(self, *_):
        return self

    def eq(self, *_):
        return self

    def limit(self, *_):
        return self

    def execute(self):
        return types.SimpleNamespace(data=[{"status": self.statuses.pop(0)}])


def test_a_cancelled_row_stops_the_task_running_the_job():
    async def run():
        db = types.SimpleNamespace(client=_Rows(["processing", "cancelled"]))
        work = asyncio.create_task(asyncio.sleep(30))
        await hb.cancel_when_job_cancelled("j", db, work, interval_seconds=0)
        with pytest.raises(asyncio.CancelledError):
            await work

    asyncio.run(run())


def test_a_read_that_fails_does_not_stop_the_job():
    class Flaky(_Rows):
        def execute(self):
            status = self.statuses.pop(0)
            if status == "boom":
                raise RuntimeError("db down")
            return types.SimpleNamespace(data=[{"status": status}])

    async def run():
        db = types.SimpleNamespace(client=Flaky(["boom", "processing", "cancelled"]))
        work = asyncio.create_task(asyncio.sleep(30))
        await hb.cancel_when_job_cancelled("j", db, work, interval_seconds=0)
        assert db.client.statuses == []
        work.cancel()

    asyncio.run(run())


def test_the_orchestrator_watches_for_a_cancel_and_stops_watching_when_it_ends():
    src = _code("app/api/rag_routes.py")
    body = src[src.index("async def process_document_with_discovery"):]
    assert "cancel_when_job_cancelled(job_id, supabase, asyncio.current_task())" in body
    assert body.index("cancel_when_job_cancelled(job_id") < body.index("cancel_watch.cancel()")


def test_a_deploy_defers_its_restart_while_an_ingestion_runs():
    workflow = (_ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    drain = workflow[workflow.index("pause-for-deploy?max_wait_seconds=120"):]
    assert drain.index("restart_when_idle.sh") < drain.index("sudo systemctl restart mivaa-pdf-extractor")
    script = (_ROOT / "scripts/deploy/restart_when_idle.sh").read_text(encoding="utf-8")
    assert script.index("pause-for-deploy") < script.rindex('systemctl restart "$SERVICE"')
    assert "MAX_WAIT" in script and "resume-from-deploy" in script


def test_a_parked_job_tells_the_operators():
    src = _code("app/api/rag_routes.py")
    parked = src[src.index("if stage_0_result.get(\"awaiting_review\"):"):]
    assert parked.index("notify_review_waiting") < parked.index("return")
    review = _code("app/services/discovery/discovery_review.py")
    assert 'emit_flow_event("ingestion_review_waiting"' in review


def test_a_table_copied_from_another_pdf_is_not_checked_against_this_one():
    src = _code("app/services/pdf/ingestion_audit_runner.py")
    assert "sourced_elsewhere" in src and "sourced_elsewhere" in _code("app/services/pdf/ingestion_audit.py")


def test_a_merge_moves_records_in_one_transaction_and_never_deletes_a_product_itself():
    src = _code("app/services/products/product_merge_service.py")
    assert "rpc('merge_products_into'" in src and "rpc('drop_catalog_product'" in src
    assert "table('products').delete()" not in src
    body = src[src.index("async def merge_products"):]
    assert body.index("_merge_product_data(") < body.index("merge_products_into") < body.index("table('products').update(")


def test_a_job_whose_row_is_gone_stops_too():
    class Gone(_Rows):
        def execute(self):
            return types.SimpleNamespace(data=[])

    async def run():
        work = asyncio.create_task(asyncio.sleep(30))
        await hb.cancel_when_job_cancelled("j", types.SimpleNamespace(client=Gone([])), work, interval_seconds=0)
        with pytest.raises(asyncio.CancelledError):
            await work

    asyncio.run(run())


def test_merged_fields_are_only_what_changes_and_survive_empty_and_nested_values():
    pkg = {n: types.ModuleType(n) for n in ("app.services.core.supabase_client",)}
    pkg["app.services.core.supabase_client"].SupabaseClient = object
    for name, module in pkg.items():
        sys.modules.setdefault(name, module)
    for name in ("app", "app.services", "app.services.core"):
        sys.modules.setdefault(name, types.ModuleType(name))
    spec = importlib.util.spec_from_file_location("product_merge_under_test", _ROOT / "app/services/products/product_merge_service.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    svc = pm.ProductMergeService.__new__(pm.ProductMergeService)
    target = {"id": "t", "name": "T", "description": "", "metadata": None, "attributes": {"color": ["gris"]}, "workspace_id": "w"}
    source = {"id": "s", "name": "S", "description": "From source", "attributes": {"color": ["gris", "beige"]},
              "metadata": {"colour_options": [{"name": "Gris"}], "page_range": [1]}}
    out = asyncio.run(svc._merge_product_data(target, [source]))
    assert set(out) == {"description", "metadata", "attributes", "updated_at"}
    assert out["description"] == "From source"
    assert out["attributes"]["color"] == ["gris", "beige"]
    assert out["metadata"]["colour_options"] == [{"name": "Gris"}]
    assert out["metadata"]["merged_from"] == [{"id": "s", "name": "S"}]
