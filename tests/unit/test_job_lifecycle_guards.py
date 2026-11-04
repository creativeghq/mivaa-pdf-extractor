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
