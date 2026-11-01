"""The unattended image backfills wait for a document's ingestion to settle; explicit repairs do not."""
import ast
import asyncio
from pathlib import Path

import pytest

from app.services.embeddings.ingestion_settling import (
    ACTIVE_JOB_STATUSES,
    outside_documents_clause,
    settling_document_ids,
)

pytestmark = pytest.mark.unit

EMB = Path(__file__).resolve().parents[2] / "app" / "services" / "embeddings"


class _Query:
    def __init__(self, rows, seen):
        self.rows, self.seen = rows, seen

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.seen.append((name, args))
            return self
        return call

    @property
    def not_(self):
        self.seen.append(("not_", ()))
        return self

    def execute(self):
        return type("R", (), {"data": self.rows})()


class _Client:
    def __init__(self, rows):
        self.seen = []
        self.rows = rows

    def table(self, name):
        self.seen.append(("table", (name,)))
        return _Query(self.rows, self.seen)


def test_active_and_recent_jobs_settle_their_documents():
    client = _Client([{"document_id": "b"}, {"document_id": "a"}, {"document_id": "a"}, {"document_id": None}])
    ids = asyncio.run(settling_document_ids(client, hours=24))
    assert ids == ["a", "b"]
    (or_filter,) = [args[0] for name, args in client.seen if name == "or_"]
    for status in ("processing", "pending", "interrupted"):
        assert status in ACTIVE_JOB_STATUSES and status in or_filter
    assert "updated_at.gte." in or_filter


def test_clause_keeps_images_without_a_document():
    assert outside_documents_clause(["a", "b"]) == "document_id.is.null,document_id.not.in.(a,b)"
    assert "not.in" not in outside_documents_clause([])


@pytest.mark.parametrize("module,func", [
    ("understanding_backfill.py", "_fetch_stale_images"),
    ("understanding_backfill.py", "backfill_understanding_embeddings"),
    ("classification_backfill.py", "_fetch_pending_images"),
    ("classification_backfill.py", "backfill_pending_classifications"),
    ("aspect_backfill.py", "_fetch_stale_aspect_images"),
    ("aspect_backfill.py", "backfill_aspect_embeddings"),
])
def test_an_explicit_repair_is_never_deferred_by_default(module, func):
    tree = ast.parse((EMB / module).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == func)
    args = node.args.args + node.args.kwonlyargs
    defaults = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):], node.args.defaults))
    assert "skip_settling_documents" in [a.arg for a in args], f"{func} lost the settling switch"
    assert ast.literal_eval(defaults["skip_settling_documents"]) is False
