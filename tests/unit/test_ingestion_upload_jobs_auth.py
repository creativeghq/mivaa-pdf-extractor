"""#494 M1: upload, job lifecycle, auth and the search route.

Pure helpers are exercised by extracting their source (the `app` package cannot be
imported under the CI's pytest-only install); everything else is a guard over code with
comments blanked, so a comment naming a fix never satisfies it.
"""

import ast
import importlib.util
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_ROOT = Path(__file__).resolve().parents[2]


def _load_by_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUDGET = _load_by_path(_ROOT / "scripts" / "comment_budget.py", "comment_budget_for_m1")


def _code(rel: str) -> str:
    return _BUDGET.blank_comments((_ROOT / rel).read_text(encoding="utf-8"))


def _function_source(rel: str, name: str) -> str:
    source = (_ROOT / rel).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in {rel}")


def _extract(rel: str, names: List[str], namespace: Dict[str, Any]) -> Dict[str, Any]:
    ns = {"Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "Iterable": Iterable,
          "datetime": datetime, "timezone": timezone, **namespace}
    for name in names:
        exec(_function_source(rel, name), ns)  # noqa: S102 - our own source, not input
    return ns


RAG = "app/api/rag_routes.py"
JWT = "app/middleware/jwt_auth.py"
DEPS = "app/dependencies.py"
TRACKER = "app/services/tracking/progress_tracker.py"
CHECKPOINT = "app/services/tracking/checkpoint_recovery_service.py"
RECOVERY = "app/services/tracking/job_recovery_service.py"
PRODUCTS = "app/api/products.py"


def _route_body(rel: str, name: str) -> str:
    return _BUDGET.blank_comments(_function_source(rel, name))


# ── capability rule ──────────────────────────────────────────────────────────

_cap = _extract("app/utils/tenancy.py", ["capability_is_held"], {})["capability_is_held"]
_MATRIX = [{"role": "owner", "capability": "workspace.manage"},
           {"role": "member", "capability": "workspace.read"}]


def test_membership_is_not_a_capability():
    assert not _cap(member_role="client", capability="workspace.manage",
                    matrix_rows=_MATRIX, is_platform_operator=False)
    assert not _cap(member_role="member", capability="catalog.import",
                    matrix_rows=_MATRIX, is_platform_operator=False)


def test_matrix_and_operator_grant():
    assert _cap(member_role="owner", capability="workspace.manage",
                matrix_rows=_MATRIX, is_platform_operator=False)
    assert _cap(member_role="member", capability="catalog.import",
                matrix_rows=_MATRIX, is_platform_operator=True)


def test_operator_still_needs_membership_of_the_target():
    assert not _cap(member_role=None, capability="catalog.import",
                    matrix_rows=_MATRIX, is_platform_operator=True)


# ── workspace binding (B12) ──────────────────────────────────────────────────

def test_jwt_never_reads_user_metadata():
    assert "user_metadata" not in _code(JWT)


def test_workspace_context_honours_and_checks_the_header():
    body = _route_body(DEPS, "get_workspace_context")
    assert 'request.headers.get("X-Workspace-Id")' in body
    assert "_validate_workspace_access(user_id, requested)" in body
    assert "HTTP_404_NOT_FOUND" in body


def test_capability_gates_on_job_routes():
    code = _code(RAG)
    for route, cap in (
        ('"/jobs/{job_id}/restart"', "catalog.import"),
        ('"/documents/{document_id}/reprocess"', "catalog.import"),
        ('"/documents/jobs/{job_id}"', "workspace.manage"),
    ):
        line = next(l for l in code.splitlines() if route in l and "@router." in l)
        assert f'require_rag_resource_capability("{cap}")' in line, route
    resume = code[code.index('"/documents/job/{job_id}/resume"'):]
    assert 'require_rag_resource_capability("catalog.import")' in resume[:300]


def test_upload_requires_catalog_import_and_bills_the_caller():
    body = _route_body(RAG, "upload_document")
    assert 'require_workspace_capability(workspace_context.user_id, workspace_id, "catalog.import")' in body
    assert "create_document_upload_job" in body
    assert '"p_user_id": upload_user_id' in body
    assert ".table('documents').insert(" not in body
    assert ".table('background_jobs').insert(" not in body


def test_unowned_rows_are_refused():
    body = _route_body(RAG, "_authorize_rag_resource")
    assert "not resource_ws" in body


# ── URL upload (B3) + storage path (B14) ─────────────────────────────────────

def test_url_upload_reads_the_whole_body():
    body = _route_body(RAG, "upload_document")
    assert "safe_fetch_bytes(" in body
    assert ".content.read(" not in body


def test_storage_object_path_and_filename_are_decoded():
    body = _route_body(RAG, "upload_document")
    assert re.search(r"storage_object_path = _unquote\(", body)
    assert re.search(r"filename = unquote\(", body)


# ── resume / restart / reprocess ─────────────────────────────────────────────

class _Stage(str, Enum):
    INITIALIZED = "initialized"
    CHUNKS_CREATED = "chunks_created"
    PRODUCTS_CREATED = "products_created"


_ck = _extract(CHECKPOINT, ["_is_resumable_stage", "select_resumable_checkpoint"],
               {"ProcessingStage": _Stage})
select_resumable_checkpoint = _ck["select_resumable_checkpoint"]

_HISTORY = [
    {"stage": "initialized", "status": "completed", "completed_at": "t0", "data": {"a": 1}},
    {"stage": "chunks_created", "status": "completed", "completed_at": "t1", "data": {"chunks_created": 4}},
    {"stage": "stage_4_products", "status": "in_progress", "completed_at": "t2"},
    {"stage": "failed", "status": "failed", "completed_at": "t3"},
]


def test_resume_skips_progress_and_terminal_events():
    cp = select_resumable_checkpoint("j", None, _HISTORY)
    assert cp["stage"] == "chunks_created"
    assert cp["checkpoint_data"] == {"chunks_created": 4}


def test_last_checkpoint_column_wins():
    cp = select_resumable_checkpoint("j", {"stage": "initialized", "created_at": "c"}, _HISTORY)
    assert cp["stage"] == "initialized"
    assert cp["created_at"] == "c"


def test_nothing_resumable_is_none():
    assert select_resumable_checkpoint("j", {"stage": "bogus"}, _HISTORY[2:]) is None


class _Settings:
    anthropic_model_validation = "model-x"


_rd = _extract(RAG, ["resume_dispatch_params"], {"get_settings": lambda: _Settings()})
resume_dispatch_params = _rd["resume_dispatch_params"]


def test_resume_restores_extract_only_and_categories():
    params = resume_dispatch_params(
        {"metadata": {"categories": ["extract_only"], "extract_only_mode": True}}, {}
    )
    assert params["extract_only_mode"] is True
    assert params["extract_categories"] == ["extract_only"]


def test_resume_falls_back_to_the_document_upload_params():
    params = resume_dispatch_params({"metadata": {}}, {"metadata": {"categories": ["logos"]}})
    assert params["extract_categories"] == ["logos"]
    assert params["extract_only_mode"] is False


def test_restart_releases_the_claim_on_every_refusal():
    body = _route_body(RAG, "restart_job_from_checkpoint")
    prep = body.index("_prepare_resume(")
    assert body.index("_release_resume_claim(", prep) > prep
    assert body.count("_release_resume_claim(") >= 2
    assert "ProcessingStage(" not in body


def test_upload_persists_what_resume_reads():
    body = _route_body(RAG, "upload_document")
    for key in ('"extract_only_mode": extract_only_mode', '"categories": category_list',
                '"chunk_size": chunk_size'):
        assert key in body, key


def test_reprocess_inserts_the_job_before_deleting():
    body = _route_body(RAG, "reprocess_document")
    insert = body.index('table("background_jobs").insert(')
    delete = body.index('table("products")')
    assert insert < delete
    assert '"discovery_model": discovery_model' not in body
    assert '"extract_categories": extract_categories' not in body


def test_auto_restart_hands_off_to_the_resume_path():
    body = _route_body(CHECKPOINT, "auto_restart_stuck_job")
    assert '"status": "interrupted"' in body
    assert '"status": "pending"' not in body


def test_startup_interrupt_is_scoped_to_this_service():
    body = _route_body(RECOVERY, "mark_all_processing_as_interrupted")
    assert '.eq("status", "processing")' in body
    assert '.in_("job_type"' in body
    assert "last_heartbeat" in body
    assert '"pending"' not in body


# ── terminal job state ───────────────────────────────────────────────────────

def test_complete_and_fail_merge_metadata_and_raise():
    for name in ("complete_job", "fail_job"):
        body = _route_body(TRACKER, name)
        assert "_write_terminal_state(" in body, name
        assert "'metadata':" not in body, name
        assert "raise JobStatePersistenceError" in body, name
    writer = _route_body(TRACKER, "_write_terminal_state")
    assert "merge_background_job_metadata" in writer


# ── stuck-job analyzer ───────────────────────────────────────────────────────

_parse_utc = _extract("app/services/tracking/stuck_job_analyzer.py", ["parse_utc"], {})["parse_utc"]


def test_naive_and_aware_timestamps_compare():
    naive = _parse_utc("2026-10-17T10:00:00")
    aware = _parse_utc("2026-10-17T12:00:00+02:00")
    assert (aware - naive).total_seconds() == 0
    assert _parse_utc("not a date") is None


# ── search route ─────────────────────────────────────────────────────────────

_sf = _extract(RAG, ["search_failure"], {})["search_failure"]


def test_a_failed_search_is_not_an_empty_one():
    assert _sf({"results": [], "error": "vecs down"}) == "vecs down"
    assert _sf({"results": [{"id": 1}], "error": "one source"}) is None
    assert _sf({"results": []}) is None


def test_search_route_raises_and_redacts():
    body = _route_body(RAG, "search_documents")
    assert "search_failure(results)" in body
    assert "HTTP_503_SERVICE_UNAVAILABLE" in body
    assert body.index("redact_internal_fields(") < body.index("return SearchResponse(")
    assert "redact_internal_product_fields" in _route_body(RAG, "redact_internal_fields")


# ── manual product create ────────────────────────────────────────────────────

def test_manual_create_does_not_swallow_its_writes():
    body = _route_body(PRODUCTS, "create_manual_product")
    assert "logger.warning" not in body
    assert "failed_writes" in body
    assert body.count("refund_credits(") >= 3
    assert 'require_workspace_capability(ctx.user_id, ws, "pricing.manage")' in body
