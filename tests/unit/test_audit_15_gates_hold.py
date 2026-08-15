"""Guards for the fixes in issue #15.

One test per defect that a typecheck cannot see and no integrity probe can reach —
which, in this repo, is most of them. Source/AST based so they run in about a second and
therefore actually run.

Scoping note, since #15's headline finding was about exactly this: where a check CAN be
expressed over the whole tree it is (`test_no_kb_route_reads_by_id_without_ownership`
walks every route in the file rather than naming the five that were broken). Where it
genuinely pins one call site, the test says so in its own docstring instead of implying
breadth it does not have.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

_KB = ROOT / "app" / "api" / "knowledge_base.py"
_TRACKER = ROOT / "app" / "services" / "tracking" / "progress_tracker.py"
_CLEANUP = ROOT / "app" / "services" / "utilities" / "cleanup_service.py"
_VECS = ROOT / "app" / "services" / "embeddings" / "vecs_service.py"
_EMB = ROOT / "app" / "services" / "embeddings" / "real_embeddings_service.py"
_MAIN = ROOT / "app" / "main.py"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_prose(src: str) -> str:
    """Drop docstrings and comments before matching on source text.

    Guards in this repo explain the bug they prevent in prose, and prose about a 403 is
    not a 403 — `test_paid_route_metering` carries the same helper for the same reason.
    """
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _func(path: Path, name: str) -> str:
    src = _src(path)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{name}() not found in {path.name} — renamed or removed")


# ───────────────────── MV2-11 / the live KB BOLA ──────────────────────


def test_management_routes_is_gone():
    """The shadowing router, third instance of the shape, deleted like the other two.

    It held 10 handlers with no gate of any kind — including `DELETE /documents/jobs/{id}`
    — mounted into rag_routes' namespace, so they were unreachable BUT still wrote the
    published OpenAPI (last-write-wins per path) and were one include-order change away
    from being live.
    """
    assert not (ROOT / "app" / "api" / "documents" / "management_routes.py").exists(), (
        "management_routes.py is back. Routes under /api/rag belong in rag_routes.py, "
        "next to the gates."
    )
    assert "management_router" not in _src(_MAIN).replace("#", "\n#").split("\n#")[0], (
        "main.py imports/includes management_router again"
    )


def test_no_kb_route_reads_by_id_without_ownership():
    """EVERY /api/kb handler taking an id must prove the caller's workspace owns it.

    Scoped to the whole file, not to the five routes that were broken: `GET`, `PATCH`,
    `DELETE /documents/{doc_id}`, `GET /documents/{doc_id}/attachments` and
    `GET /products/{product_id}/documents` all took an id off the path and queried it
    with the service-role client, no workspace predicate anywhere. MIVAA has no RLS
    backstop, so any authenticated member of ANY workspace could read, edit or delete
    another tenant's KB document by id. `/api/kb` is a prefix no other router declares,
    so — unlike management_routes — these were live.
    """
    src = _src(_KB)
    tree = ast.parse(src)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = [ast.unparse(d) for d in node.decorator_list]
        if not any(d.startswith("router.") for d in decorators):
            continue
        # Routes that take an id off the PATH are the ones at risk.
        takes_id = any(
            a.arg.endswith("_id") or a.arg == "doc_id" for a in node.args.args
        )
        if not takes_id:
            continue
        body = ast.get_source_segment(src, node) or ""
        binds = (
            "_load_own_doc" in body
            or "_assert_own_workspace" in body
            or "resolve_workspace_id" in body
            or 'eq("workspace_id"' in body
            or "eq('workspace_id'" in body
        )
        if not binds:
            offenders.append(node.name)

    assert not offenders, (
        f"these /api/kb handlers take an id and never bind it to the caller's "
        f"workspace: {offenders}. Use _load_own_doc() — 404, never 403, so the "
        "endpoint is not an id-existence oracle."
    )


def test_kb_ownership_check_returns_404_not_403():
    body = _strip_prose(_func(_KB, "_load_own_doc"))
    assert "404" in body, "the ownership mismatch must 404"
    assert "403" not in body, (
        "403 on an ownership mismatch confirms the id exists — that is an enumeration "
        "oracle (invariant 1 says 404)"
    )


# ───────────────── MV2-12 / MV2-13 — trust fields and paired ids ─────────────────


def test_search_cannot_declare_itself_admin():
    """`is_admin_caller` was a REQUEST BODY field feeding kb_match_docs(include_private).

    Any member could send one boolean and read the workspace's private KB. The field is
    deleted rather than defaulted-and-ignored: one the OpenAPI still advertises is one a
    caller still sends, and the next person to wire it up re-opens the hole.
    """
    src = _src(_KB)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SearchKBRequest":
            fields = {t.target.id for t in node.body if isinstance(t, ast.AnnAssign)}
            assert "is_admin_caller" not in fields, (
                "is_admin_caller is back on the request model — a caller cannot be "
                "allowed to assert its own admin status (invariant 8)"
            )
            assert "allowed_access_levels" not in fields, (
                "allowed_access_levels is back on the request model; it was documented "
                "as 'Overrides category access gating', which is precisely the problem"
            )
            break
    else:
        raise AssertionError("SearchKBRequest not found")

    body = _func(_KB, "search_kb_documents")
    assert "_caller_is_workspace_admin" in body, (
        "the search route no longer derives admin status server-side"
    )
    assert "request.is_admin_caller" not in body


def test_admin_status_is_read_from_workspace_members():
    """Not from the JWT `role` claim / `permissions`.

    A real Supabase token carries role="authenticated", which falls through
    `UserRole(...)` to MEMBER with an empty permissions list — deriving admin from it
    would make every admin a member, which is the silent-zero shape pointed at an access
    gate. `workspace_members.role` is what the platform means by admin-of-a-workspace.
    """
    body = _func(_KB, "_caller_is_workspace_admin")
    assert "workspace_members" in body
    assert "owner" in body and "admin" in body
    assert "return False" in body, "the failure path must fail CLOSED"


def test_kb_attachment_checks_both_ids_against_the_workspace():
    """Eighth instance of 'two ids each individually valid, never checked against each other'."""
    body = _func(_KB, "attach_document_to_product")
    assert "_load_own_doc" in body, "document_id is not proven to be in this workspace"
    assert "products" in body and "request.product_id" in body, (
        "product_id is not proven to be in this workspace"
    )


# ─────────────────────── MV2-15 — the 100%-failure endpoint ───────────────────────


def test_from_pdf_does_not_call_the_route_function():
    """`create_kb_document(create_request, supabase_client)` passed a `Depends` OBJECT.

    Called directly with two positional args, `current_user` keeps its default — the
    `Depends(get_current_user)` marker — which then reached `resolve_workspace_id`.
    FastAPI only resolves dependencies for requests it ROUTES. The endpoint failed 100%
    of the time inside a `try` that reported it as a generic 500.
    """
    body = _func(_KB, "create_kb_document_from_pdf")
    assert "_upsert_kb_document" in body, "should call the shared helper"
    assert not re.search(r"\bcreate_kb_document\(", body), (
        "the route function is being invoked directly again — its Depends defaults will "
        "arrive unresolved"
    )


def test_no_route_function_is_called_directly_anywhere_in_kb():
    """Generalised: no handler in this file may call another handler.

    The specific bug is one instance of a class the file can host again anywhere.
    """
    src = _src(_KB)
    tree = ast.parse(src)
    handlers = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(ast.unparse(d).startswith("router.") for d in n.decorator_list)
    }
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id in handlers:
                    offenders.append(f"{node.name}() calls handler {sub.func.id}()")
    assert not offenders, (
        "a route handler is being called as a plain function; its Depends() defaults "
        f"arrive as marker objects, not resolved values: {offenders}"
    )


# ───────────────────────── MV2-1 — the unreconciled count ─────────────────────────


def test_sync_counts_assigns_every_count_it_queries():
    """`actual_embeddings` was queried, LOGGED, and never assigned.

    Three of four counts were reconciled; the fourth was also the only one with no
    mismatch warning, so the single unreconciled field was the single field with no
    drift signal — and `_sync_to_database()` on the next line wrote the stale counter.
    The platform's dominant documented failure, inside the function built to prevent it.
    """
    body = _func(_TRACKER, "sync_counts_from_database")
    tree = ast.parse(textwrap_dedent(body))

    queried = {
        t.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name) and t.id.startswith("actual_")
    }
    assert queried, "no actual_* counts found — this guard has gone blind"

    for name in sorted(queried):
        # Each queried count must be READ again (assigned onto self, compared, or both).
        uses = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
        ]
        assigned_to_self = re.search(rf"self\.\w+\s*=\s*{name}\b", body)
        assert assigned_to_self, (
            f"{name} is queried but never assigned to the tracker. Logging it is not "
            "reconciling it — that is exactly how this metric sat at zero forever."
        )
        assert len(uses) >= 2, f"{name} is barely used; check it is compared AND assigned"

    # Every reconciled count gets a drift warning, not just three of them.
    assert body.count("mismatch") >= 5, (
        "a count without a mismatch warning drifts silently — that asymmetry was the bug"
    )


def textwrap_dedent(s: str) -> str:
    import textwrap

    return textwrap.dedent(s)


# ──────────────────── MV2-2 / MV2-3 — atomic writes, real boundaries ────────────────────


def test_progress_and_history_are_one_write():
    body = _func(_TRACKER, "_sync_to_database_inner")
    assert "update_job_progress_and_append_history" in body, (
        "progress and stage_history are being written separately again — a crash "
        "between the halves leaves them disagreeing (pipeline convention 3)"
    )
    assert "'append_stage_history'" not in body, (
        "the two-call pattern is back: a separate append wrapped in a bare except that "
        "only logs makes the audit record the best-effort one"
    )


def test_the_stage_boundary_event_is_not_optional():
    """It used to be `if stage:` where `stage` came from an Optional parameter.

    `start_processing()` moved the job to DOWNLOADING through a bare
    `_sync_to_database()`, so the first stage of every job emitted no `in_progress` at
    all (pipeline convention 9).
    """
    body = _func(_TRACKER, "_sync_to_database_inner")
    assert "effective_stage = stage or" in body, (
        "the stage no longer falls back to the tracker's own current_stage, so a caller "
        "omitting stage_name silently skips its history event again"
    )


def test_terminal_event_names_the_stage_that_ended():
    """`current_stage` is set to COMPLETED/FAILED BEFORE the closing event is appended.

    Reading it there produced `{"stage": "failed", "status": "failed"}` — an event that
    names no real stage and cannot answer which one broke.
    """
    body = _func(_TRACKER, "_append_terminal_event")
    assert "_last_real_stage" in body, (
        "the terminal event is reading current_stage again, which by then is the "
        "synthetic terminal stage"
    )


# ─────────────────────────── MV2-4 / MV2-5 / MV2-6 — the janitor ───────────────────────────


def test_document_row_is_deleted_before_its_storage():
    """Orphaned OBJECT self-heals via the GC cron; orphaned POINTER is a permanent 404.

    The old order (storage first, row second, storage failure appended to errors and
    execution CONTINUING) put the race in the direction that does not heal.
    """
    body = _func(_CLEANUP, "delete_job_completely")
    # Restrict to the full-wipe path: the preserve path is checked separately below.
    tail = body[body.index("Checkpoints now live on"):]
    row_at = tail.index("table('documents')")
    storage_at = tail.index("self.cleanup_document_storage(")
    assert row_at < storage_at, (
        "storage is being deleted before the documents row again — a crash in that "
        "window leaves a live row pointing at objects that no longer exist, and nothing "
        "in the platform sweeps a dangling pointer"
    )


def test_preserve_outputs_does_not_delete_the_tiles_it_preserves():
    """`preserve_outputs=True` kept document_images rows and wiped the bytes they point at.

    Those rows carry storage_bucket='pdf-tiles' and a path under
    `extracted/{document_id}/` — the exact prefix this mode was deleting — and the
    docstring said storage was preserved while the code deleted it. An operator picking
    the safe-sounding mode got the destructive one.
    """
    body = _func(_CLEANUP, "delete_job_completely")
    preserve = body[body.index("if preserve_outputs:"):body.index("Checkpoints now live on")]
    assert "source_only=True" in preserve, (
        "preserve_outputs is deleting derived tiles again; it may only reap the source PDF"
    )


def test_temp_reaper_checks_liveness_not_just_age():
    """A 1-hour age threshold with no live-job check, run on every job completion.

    Age is not a reference check. A slow OCR pass crossing the hour mark had its working
    directory removed underneath it — by a sweep running for a DIFFERENT job.
    """
    body = _func(_CLEANUP, "_clean_old_files")
    assert "live_ids" in body, "the age-based reap no longer consults live jobs"
    assert "skipped_live" in body, (
        "the reaper must report what it did NOT delete; a janitor that only reports "
        "deletions reads identically whether it is working or eating live work"
    )

    resolver = _func(_CLEANUP, "live_job_ids")
    assert "return None" in resolver, (
        "liveness failure must return None (unknown), never an empty set — empty means "
        "'nothing is running, delete freely', the opposite conclusion to draw from a "
        "failed query"
    )


# ─────────────────────── MV2-7 / MV2-8 — vector spaces ───────────────────────


def test_page_search_requires_and_checks_provenance():
    """The one collection in a different latent space, guarded only by a docstring.

    voyage-4 and voyage-multimodal are both 1024D, so a wrong-space query is ACCEPTED
    and ranked rather than raising. A guard on the producer cannot see that; the hazard
    is at the boundary.
    """
    src = _src(_VECS)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "search_page_embeddings":
            kwonly = {a.arg for a in node.args.kwonlyargs}
            assert "embedding_model" in kwonly, (
                "embedding_model must be KEYWORD-ONLY and required, so a new caller "
                "physically cannot omit the space its vector lives in"
            )
            defaults = dict(zip(node.args.kwonlyargs, node.args.kw_defaults))
            assert defaults.get("embedding_model") is None, (
                "embedding_model has a default again — a default is an assumption, and "
                "the assumption is what broke"
            )
            break
    else:
        raise AssertionError("search_page_embeddings not found")

    body = _func(_VECS, "search_page_embeddings")
    assert "_is_page_space_model" in body, "the provenance is taken but never checked"


def test_aspect_collections_reject_non_1024d():
    """Auto-detecting the dimension kept the REMOVED `*_slig_768` space writable.

    `dimension=len(query_embedding)` would create and query a 768D aspect collection in
    any fresh or partially-migrated environment, and set the presence flags for it.
    """
    src = _src(_VECS)
    assert "ASPECT_DIMENSION = 1024" in src
    assert "Auto-detect: 768" not in src, "the auto-detect branch is back"
    assert not re.search(r"dimension=len\(query_embedding\)", src), (
        "aspect collection dimension is being derived from the input again — that is "
        "not a contract, it is whatever the caller happened to send"
    )


# ──────────────────────── MV2-9 / MV2-14 — cost attribution ────────────────────────


def test_every_ai_log_call_in_the_embeddings_service_carries_a_payer():
    """`workspace_id` was threaded; `user_id` never was, though AICallLogger accepts it.

    Same defect a85f8a5 fixed once, in a third location. Walks every logging call in the
    file rather than naming the three that were broken.
    """
    src = _src(_EMB)
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        if f.attr not in ("log_ai_call", "log_time_based_call"):
            continue
        kwargs = {k.arg for k in node.keywords}
        missing = {"workspace_id", "user_id"} - kwargs
        if missing:
            offenders.append(f"line {node.lineno}: missing {sorted(missing)}")
    assert not offenders, (
        "AI cost logged with no tenant and/or no payer — invisible to per-workspace "
        f"cost views and unmatchable by ai_usage_logs' own RLS policy: {offenders}"
    )


def test_slig_failures_are_logged_not_just_successes():
    """Three GPU calls were made and paid for before the give-up path returned nothing.

    A misconfigured endpoint burned real money that appeared nowhere — the spend was
    invisible precisely when it was pure waste.
    """
    body = _func(_EMB, "_generate_siglip_embedding")
    assert body.count("log_time_based_call") >= 3, (
        "the SLIG failure paths (dim-mismatch give-up, exception) are not logging their "
        "spend again"
    )
    assert "fallback_failed" in body, (
        "a failed embed must not be logged as a successful one"
    )

    visual = _func(_EMB, "generate_visual_embedding")
    assert "log_time_based_call" in visual, (
        "generate_visual_embedding hits a billed GPU endpoint and must log it; it is a "
        "search entry point, so it runs far more often than ingestion"
    )


def test_kb_embedding_spend_is_attributed():
    """Every KB embedding call landed in ai_usage_logs with workspace_id NULL."""
    src = _src(_KB)
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute) or f.attr != "generate_all_embeddings":
            continue
        kwargs = {k.arg for k in node.keywords}
        missing = {"workspace_id", "user_id"} - kwargs
        if missing:
            offenders.append(f"line {node.lineno}: missing {sorted(missing)}")
    assert not offenders, f"unattributed paid embedding calls in knowledge_base.py: {offenders}"


def test_voyage_retries_are_distinguishable_from_a_clean_call():
    """Up to 3 rate-limit retries collapsed into one log row (pipeline convention 4)."""
    src = _src(_EMB)
    assert src.count("rate_limit_retries") >= 2, (
        "the throttle count is no longer recorded on both Voyage paths — a clean first "
        "call and a success after three throttled attempts are the same row again"
    )
    assert src.count("throttled_ms") >= 4
