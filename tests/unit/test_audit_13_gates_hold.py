"""Guards for the mivaa#13 fixes (MI-1 … MI-6) — the /api/internal authorization surface.

THE ONE IDEA THIS FILE PROTECTS
-------------------------------
`verify_internal_access` proves AUTHENTICATION and says so in its own docstring:
it admits any valid platform token, *including an end user's*. Seven `/api/internal`
mutation routes used it as their only gate and then took the object id straight from
the path. MIVAA runs every query as service role, so there is no RLS behind those
routes — an authenticated user naming another tenant's document, product or job got
a cross-tenant mutation.

The durable fix is a fork, and which branch a route takes depends on a fact about the
world rather than about the code:

  * a route with NO user caller gets `require_trusted_service` (cron secret / `mk_`
    platform key / service role only);
  * a route WITH a user caller keeps the permissive gate and checks that the caller
    owns each id it was handed.

Getting that backwards breaks production in one direction and leaves the hole open in
the other, which already happened once while fixing this: `document-extraction-status`
and `classify-images` looked callerless to a grep of the edge functions, and a wider
sweep found `DocumentHealthPanel` calling the first through `mivaaApiClient` with the
operator's own JWT. So the caller inventory is pinned below as data, not as a habit.

Static, over source text: CI installs pytest alone, so nothing here imports `app`.
That means these prove the SHAPE of the fix, not its runtime behaviour.

Every case was watched to FAIL against the pre-fix tree.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
INTERNAL = APP / "api" / "internal_routes.py"
ADMIN = APP / "api" / "admin.py"
DEPS = APP / "dependencies.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _route(path: Path, decorator_path: str) -> str:
    """Decorator + handler source for the route mounted at `decorator_path`."""
    src = _read(path)
    tree = ast.parse(src)
    lines = src.split("\n")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec.func if isinstance(dec, ast.Call) else dec
            if not (isinstance(call, ast.Attribute) and call.attr in
                    ("get", "post", "put", "patch", "delete")):
                continue
            if isinstance(dec, ast.Call) and dec.args and \
                    getattr(dec.args[0], "value", None) == decorator_path:
                return "\n".join(lines[dec.lineno - 1: node.end_lineno])
    raise AssertionError(f"route {decorator_path} not found in {path.name} — re-point this guard")


# ───────────────────────────────────────────────────────────────────────────
# The fork itself
# ───────────────────────────────────────────────────────────────────────────

def test_the_trusted_only_gate_exists_and_refuses_a_user_token():
    """`require_trusted_service` is the half of the fix that `verify_internal_access`
    deliberately does not do. If it stops rejecting ordinary user tokens it becomes
    an alias for the gate it was introduced to replace."""
    src = _strip_comments(_read(DEPS))
    assert "async def require_trusted_service" in src, "the trusted-only gate is gone"
    assert "def is_trusted_service_caller" in src, "the trusted-caller predicate is gone"
    assert "403" in src or "HTTP_403_FORBIDDEN" in src, (
        "require_trusted_service no longer refuses anything"
    )
    # The predicate must key on what a token IS, not merely on it being present.
    for marker in ('"service"', '"role"', '"aud"'):
        assert marker in src, (
            f"is_trusted_service_caller no longer inspects {marker} — it cannot tell "
            "the platform key or a service-role token from an end user's JWT"
        )


def test_ownership_assertions_return_404_not_403():
    """A distinct status for 'exists but not yours' is an id oracle (invariant 1)."""
    src = _read(DEPS)
    body_start = src.index("async def _assert_row_in_caller_workspace")
    body = _strip_comments(src[body_start: body_start + 2000])
    assert "status_code=404" in body, "the ownership assertion no longer 404s"
    assert "status_code=403" not in body, (
        "the ownership assertion 403s, which confirms the id exists to an attacker "
        "probing for it"
    )


def test_an_unowned_row_is_not_everyones_row():
    """`if _job_ws:` skipped the check entirely for a null-workspace job, so any
    authenticated caller could reset one."""
    src = _read(DEPS)
    start = src.index("async def _assert_row_in_caller_workspace")
    assert "if not owner_ws:" in src[start: start + 2000], (
        "a row with no workspace no longer falls through to a refusal"
    )
    reset = _strip_comments(_route(INTERNAL, "/reset-job/{job_id}"))
    assert "is_trusted_service_caller" in reset, (
        "reset-job no longer distinguishes a trusted caller, so a null-workspace job "
        "is resettable by anyone again"
    )


# ───────────────────────────────────────────────────────────────────────────
# MI-1 — the seven, split by whether they have a real user caller
# ───────────────────────────────────────────────────────────────────────────
#
# Pinned as DATA because the split is a fact about callers, not about this repo.
# Re-verify with a sweep of src/, supabase/functions/, api/, scripts/ and .github/
# before moving a route between the lists.

TRUSTED_ONLY = [
    "/classify-images/{job_id}",          # only ref: scripts/security-smoke.sh, expects rejection
    "/upload-images/{job_id}",            # no caller
    "/create-relationships/{job_id}",     # no caller
    "/enrich-existing-product/{product_id}",  # no caller; mutates one product
]

OWNERSHIP_CHECKED = [
    # route path, the assertion it must make
    ("/validate-pipeline/{job_id}", "assert_job_in_workspace"),        # AsyncJobQueueMonitor
    ("/run-catalog-knowledge/{document_id}", "assert_document_in_workspace"),  # mivaaApiClient
    ("/document-extraction-status/{document_id}", "assert_document_in_workspace"),  # DocumentHealthPanel
]


@pytest.mark.parametrize("path", TRUSTED_ONLY)
def test_callerless_internal_routes_admit_only_the_platform(path: str):
    body = _strip_comments(_route(INTERNAL, path))
    assert "require_trusted_service" in body, (
        f"{path} is back on a gate that admits any authenticated user. It has no user "
        "caller, so there is nothing to lose by refusing one — and MIVAA has no RLS "
        "behind it."
    )


@pytest.mark.parametrize("path, assertion", OWNERSHIP_CHECKED)
def test_user_callable_internal_routes_check_the_id_they_were_handed(path: str, assertion: str):
    body = _strip_comments(_route(INTERNAL, path))
    assert assertion in body, (
        f"{path} no longer verifies the caller owns the id in its path. This route HAS "
        "a real user caller (the Admin UI sends the operator's own JWT), so it cannot "
        f"use require_trusted_service — it needs {assertion}."
    )
    assert "require_trusted_service" not in body, (
        f"{path} was given the trusted-only gate, which 403s its real user caller. "
        "Ownership check, not a stricter gate."
    )


# ───────────────────────────────────────────────────────────────────────────
# MI-2 — the other id in the request
# ───────────────────────────────────────────────────────────────────────────

def test_every_workspace_authorized_route_also_reconciles_its_document():
    """`authorize_rag_workspace(claims, body.workspace_id)` proves the caller may act
    in the workspace they NAMED. It says nothing about the document id in the same
    body. Sixth confirmed instance of the two-unchecked-ids class."""
    src = _strip_comments(_read(INTERNAL))
    authorized = src.count("authorize_rag_workspace(claims, request.workspace_id)")
    reconciled = src.count("assert_document_belongs_to(request.document_id, request.workspace_id)")
    assert authorized > 0, "re-point this guard: no authorize_rag_workspace calls found"
    assert reconciled >= authorized, (
        f"{authorized} route(s) authorize a body-supplied workspace but only "
        f"{reconciled} reconcile the document id against it — name your own workspace "
        "and another tenant's document and the difference is a cross-tenant write"
    )


# ───────────────────────────────────────────────────────────────────────────
# MI-3 — the job writes
# ───────────────────────────────────────────────────────────────────────────

def test_no_background_job_write_is_keyed_on_a_body_supplied_id_alone():
    """regenerate-image-embeddings scoped the IMAGES correctly and then wrote
    background_jobs by id alone, so another tenant's job could be marked
    processing/completed/failed with attacker-controlled progress metadata."""
    src = _strip_comments(_read(INTERNAL))
    bare = re.findall(r"\.eq\('id', request\.job_id\)\.execute\(\)", src)
    assert not bare, (
        f"{len(bare)} background_jobs write(s) keyed on request.job_id with no "
        "workspace filter"
    )
    body = _strip_comments(_route(INTERNAL, "/regenerate-image-embeddings"))
    assert "assert_job_belongs_to" in body, (
        "the route no longer reconciles the body-supplied job_id against the workspace"
    )


# ───────────────────────────────────────────────────────────────────────────
# MI-4 / MI-5 — admin.py
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/jobs", "/jobs/{job_id}", "/jobs/{job_id}/products"])
def test_job_readers_confine_a_non_trusted_caller(path: str):
    """Despite the /api/jobs path these are not admin-only. Without a workspace
    filter they returned another tenant's filenames, document ids and errors."""
    body = _strip_comments(_route(ADMIN, path))
    assert ("caller_workspace_or_none" in body) or ("assert_job_readable" in body), (
        f"{path} reads jobs with no tenant confinement — verify_internal_access admits "
        "any valid platform token, so this is readable by any authenticated user"
    )


@pytest.mark.parametrize("path", [
    "/admin/drain-status", "/jobs/health", "/system/health", "/packages/status",
])
def test_diagnostics_carry_a_route_level_gate(path: str):
    """Invariant 5 wants the gate at the route as well as the middleware — #361
    showed gateway path normalisation can reach past the middleware."""
    body = _route(ADMIN, path)
    assert "verify_internal_access" in body, (
        f"{path} has no route-level gate; it relies entirely on the middleware"
    )


# ───────────────────────────────────────────────────────────────────────────
# MI-6 — success is derived, not asserted
# ───────────────────────────────────────────────────────────────────────────

def test_no_route_claims_success_while_returning_errors():
    """`success=True` beside a populated errors[] is the platform's own silent-zero
    shape: the dashboard records a clean backfill over rows that failed."""
    src = _read(INTERNAL)
    offenders = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        seg = ast.get_source_segment(src, node.value) or ""
        if "success=True" in seg and "errors" in seg:
            offenders.append(node.lineno)
    assert not offenders, (
        "return(s) at line(s) " + ", ".join(map(str, offenders)) +
        " assert success while carrying errors — derive it (`success=not errors`)"
    )
