"""The unattended image backfills wait for a document's ingestion to settle; explicit repairs do not."""
import ast
import asyncio
from pathlib import Path

import pytest

from app.services.embeddings.ingestion_settling import (
    outside_documents_clause,
    settling_document_ids,
)

pytestmark = pytest.mark.unit

EMB = Path(__file__).resolve().parents[2] / "app" / "services" / "embeddings"


class _Client:
    def __init__(self, rows):
        self.calls = []
        self.rows = rows

    def rpc(self, name, params):
        self.calls.append((name, params))
        rows = self.rows
        return type("Q", (), {"execute": lambda _self: type("R", (), {"data": rows})()})()


def test_the_rule_is_the_sql_function():
    client = _Client(["b", "a", "a", None])
    assert asyncio.run(settling_document_ids(client)) == ["a", "b"]
    assert client.calls == [("ingestion_settling_document_ids", {})]


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
