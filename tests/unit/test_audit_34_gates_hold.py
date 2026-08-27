"""Guards for the mivaa#34 job-management fixes (M19-1, M19-2, M19-3).

All three are LATENT — every ingestion table holds zero rows — which is exactly why
they are worth pinning now: they become live on the first production ingestion, and a
guard written before the data exists is a guard written without pressure.

#34 is the most precisely-scoped issue in this series. It also CORRECTS earlier passes:
seven routes reported as cross-tenant IDORs are not, because `require_rag_resource_access`
resolves the owning workspace from the resource itself. Several cases below therefore pin
things that are already RIGHT, so a later tidy-up cannot quietly undo them.

10 of the 11 cases were watched to fail against the pre-fix tree. The one that passes
both ways is `test_restart_still_uses_its_compare_and_swap` — a stays-as-it-is guard over
the pattern M19-3 was fixed BY, named so it is not mistaken for coverage.

One place the issue's own fix does not work, and the correction is recorded in the code:
it says to swap all FOUR weakly-gated routes to `require_rag_resource_access`. That gate
raises 400 when there is no resource id in the path or query, so on the LIST route it
would reject every call. A list needs a predicate, not a resource lookup.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
RAG_ROUTES = APP / "api" / "rag_routes.py"

#: The three id-bearing routes that authenticated without authorizing.
OWNERSHIP_GATED = [
    "get_job_status",
    "get_job_full_status",
    "get_job_checkpoints",
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _decorators(src: str, name: str) -> str:
    node = _node(src, name)
    return " ".join(ast.get_source_segment(src, d) or "" for d in node.decorator_list)


def _body(src: str, name: str) -> str:
    return _strip_comments(ast.get_source_segment(src, _node(src, name)) or "")


# ───────────────────────────────────────────────────────────────────────────
# M19-1 — authenticate is not authorize
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("func", OWNERSHIP_GATED)
def test_the_job_read_routes_verify_ownership_not_just_a_token(func: str):
    """`verify_internal_access` admits "ANY valid platform token, including an end
    user's" — its own words. These three then read the job by id alone, so any token
    plus a job id returned that job's metadata, stage, recovery history, checkpoint
    payloads and memory state."""
    src = _read(RAG_ROUTES)
    decorators = _decorators(src, func)
    assert "require_rag_resource_access" in decorators, (
        f"{func} is back on a gate that authenticates without authorizing (#34 M19-1)"
    )
    assert "verify_internal_access" not in decorators, (
        f"{func} still carries verify_internal_access alongside the ownership gate — "
        "two gates where one is weaker is the weaker one's semantics"
    )


def test_the_list_route_uses_a_predicate_because_the_gate_cannot_help_it():
    """The issue said to swap all four. `require_rag_resource_access` raises 400 without
    a resource id ("A list endpoint with no resource id would otherwise return rows
    across all tenants"), so on this route it would reject every call. A list needs a
    filter."""
    src = _read(RAG_ROUTES)
    assert "require_rag_resource_access" not in _decorators(src, "list_jobs"), (
        "list_jobs was swapped to the resource gate, which 400s when there is no id — "
        "every call to this endpoint now fails"
    )
    body = _body(src, "list_jobs")
    assert ".eq('workspace_id', caller_workspace)" in body, (
        "list_jobs no longer filters by the caller's workspace, so any valid token "
        "lists every tenant's jobs (#34 M19-1)"
    )


def test_the_list_route_refuses_a_caller_with_neither_workspace_nor_cron_secret():
    """The unfiltered path is for auto-recovery, which has to see the whole estate and
    is not a user. Anything else with no workspace must be refused, not served
    platform-wide."""
    body = _body(_read(RAG_ROUTES), "list_jobs")
    assert "x-cron-secret" in body and "HTTP_403_FORBIDDEN" in body, (
        "a caller with no workspace context and no cron secret is no longer refused, so "
        "the unfiltered branch is reachable by an ordinary token again"
    )


# ───────────────────────────────────────────────────────────────────────────
# M19-3 — reprocess deletes; it must not delete under a running job
# ───────────────────────────────────────────────────────────────────────────

def test_reprocess_refuses_while_a_job_is_in_flight():
    """Reprocess deletes products, chunks, images, VECS embeddings, tile storage and
    document metadata. Issued mid-ingestion it pulls the running job's outputs out from
    under it, and the running job carries on writing into the wreckage."""
    body = _body(_read(RAG_ROUTES), "reprocess_document")
    assert '"pending", "processing"' in body or "'pending', 'processing'" in body, (
        "reprocess no longer checks whether a job for this document is in flight "
        "(#34 M19-3)"
    )
    assert "HTTP_409_CONFLICT" in body, "the in-flight case no longer refuses"


def test_the_guard_runs_before_anything_is_deleted():
    """The ORDER is the fix. A check after the first delete is not a check."""
    body = _body(_read(RAG_ROUTES), "reprocess_document")
    guard_at = body.index("HTTP_409_CONFLICT")
    delete_at = body.find(".delete()")
    assert delete_at == -1 or guard_at < delete_at, (
        "the in-flight guard now runs after a delete — the outputs it exists to protect "
        "are already gone"
    )


def test_the_status_is_actually_selected():
    """A guard reading a column the query never fetched is a guard that always passes."""
    body = _body(_read(RAG_ROUTES), "reprocess_document")
    assert '"id, status, metadata"' in body, (
        "the previous-job read no longer selects `status`, so the in-flight check reads "
        "None and never fires"
    )


def test_restart_still_uses_its_compare_and_swap():
    """Recorded because it is the pattern M19-3 was fixed BY, and it is correct: a
    status-scoped CAS that refuses to restart a job already processing. If this is ever
    loosened, the reprocess guard becomes the only one left."""
    body = _body(_read(RAG_ROUTES), "restart_job_from_checkpoint")
    assert "'pending', 'interrupted'" in body or '"pending", "interrupted"' in body, (
        "restart's status-scoped compare-and-swap is gone"
    )


# ───────────────────────────────────────────────────────────────────────────
# M19-2 — a control that does nothing is worse than no control
# ───────────────────────────────────────────────────────────────────────────

def test_include_embeddings_false_actually_withholds_the_embedding():
    """The base query is `select('*')`, so `text_embedding` was in every row regardless
    and the flag only layered derived fields on top. A caller explicitly asking not to
    receive embeddings received them anyway."""
    body = _body(_read(RAG_ROUTES), "get_chunks")
    assert "chunk.pop('text_embedding', None)" in body, (
        "the raw column is no longer removed, so include_embeddings=false returns it "
        "anyway (#34 M19-2)"
    )


def test_has_embedding_is_reported_either_way():
    """"Does this chunk have one" is the question an `include_embeddings=false` caller
    is usually asking, and answering it costs nothing."""
    body = _body(_read(RAG_ROUTES), "get_chunks")
    at_pop = body.index("chunk.pop('text_embedding'")
    at_flag = body.index("chunk['has_embedding']")
    at_if = body.index("if include_embeddings:")
    assert at_pop < at_flag < at_if, (
        "has_embedding moved inside the include_embeddings branch, so the cheap answer "
        "is now withheld along with the expensive one"
    )
