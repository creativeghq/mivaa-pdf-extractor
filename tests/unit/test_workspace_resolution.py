"""Behavioral tests for the workspace-binding rule (BOLA, invariant 1, #250).

Real tests, not a source scan — `app.auth.workspace_resolution` is deliberately free
of third-party imports so this can exercise the actual function against a stubbed
membership check.

The two branches that matter, and why getting either wrong is severe:

  - Reject the service key's requested workspace -> every edge function breaks at
    once, because the platform identity is not tenant-scoped and the workspace can
    only come from the request.
  - Accept an end user's requested workspace without a membership check -> any logged
    in user reads any tenant's data. That is the live shape on the excluded prefixes
    (`/api/rag`, `/api/internal`, `/api/interior`), where the middleware never runs.

TWO CI CONSTRAINTS, both learned the hard way — this file broke the deploy once:

  1. CI installs pytest and NOTHING else (`deploy.yml`: `pip install pytest==7.4.3`).
     No fastapi, no pytest-asyncio. A third-party import here makes the module
     uncollectable and takes the ENTIRE suite down with it, not just this file.
  2. Therefore no `@pytest.mark.asyncio`. Without the plugin that marker is
     unregistered, and pytest.ini sets `--strict-markers`, so it is a hard collection
     error. Coroutines are driven with `asyncio.run`, which is what the rest of this
     suite does (see test_page_embeddings.py).
"""

import asyncio
import importlib.util
import re
from pathlib import Path

import pytest

# Loaded by path, not by `import app.auth...`, because tests/unit/test_table_extraction.py
# registers a bare ModuleType("app") in sys.modules to avoid executing the real package —
# and it collects first, so `app` is no longer a package by the time this module imports.
_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _ROOT / "app" / "auth" / "workspace_resolution.py"
_spec = importlib.util.spec_from_file_location("workspace_resolution_under_test", _MODULE_PATH)
_wr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wr)

SERVICE_SUBJECT = _wr.SERVICE_SUBJECT
is_service_caller = _wr.is_service_caller
_resolve = _wr.resolve_workspace_id
WorkspaceAccessDenied = _wr.WorkspaceAccessDenied

OWN = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
OPERATOR = "99999999-9999-9999-9999-999999999999"

USER = {"sub": "user-abc", "workspace_id": OPERATOR}
SERVICE = {"sub": SERVICE_SUBJECT, "service": "mivaa", "workspace_id": OPERATOR}


def resolve(*args, **kwargs):
    """Drive the coroutine without pytest-asyncio (see module docstring)."""
    return asyncio.run(_resolve(*args, **kwargs))


def member_of(*allowed):
    """Stub membership check that records what it was asked."""
    calls = []

    async def _is_member(user_id, workspace_id):
        calls.append((user_id, workspace_id))
        return workspace_id in allowed

    _is_member.calls = calls
    return _is_member


# --------------------------------------------------------------------------
# Service caller — the trusted channel
# --------------------------------------------------------------------------

def test_service_key_may_name_any_workspace():
    """The edge function already bound this to a verified user JWT on its side."""
    check = member_of()  # would deny everything
    assert resolve(SERVICE, OTHER, is_member=check) == OTHER
    assert check.calls == [], "the service key must not be membership-checked"


def test_service_key_with_no_request_value_falls_back_to_its_own():
    assert resolve(SERVICE, None, is_member=member_of()) == OPERATOR


def test_service_detection_cannot_be_forged_by_a_user_token():
    # A Supabase token is signed by Supabase; `service` is set only by
    # _validate_simple_api_key. Guard the predicate anyway.
    assert is_service_caller(SERVICE) is True
    assert is_service_caller(USER) is False
    assert is_service_caller({}) is False
    assert is_service_caller({"user_metadata": {"service": "mivaa"}}) is False


# --------------------------------------------------------------------------
# End user — the untrusted channel
# --------------------------------------------------------------------------

def test_user_requesting_a_workspace_they_belong_to_is_allowed():
    check = member_of(OWN)
    assert resolve(USER, OWN, is_member=check) == OWN
    assert check.calls == [("user-abc", OWN)]


def test_user_requesting_a_foreign_workspace_is_refused():
    with pytest.raises(WorkspaceAccessDenied) as exc:
        resolve(USER, OTHER, is_member=member_of(OWN))
    assert exc.value.workspace_id == OTHER


def test_the_refusal_is_a_404_not_a_403():
    """A 403 confirms the workspace exists, which enumerates ids. Pinned because it
    is the kind of detail a later 'clearer error message' change quietly reverses."""
    assert WorkspaceAccessDenied.status_code == 404


def test_the_boundary_maps_the_refusal_to_that_status():
    """The rule raises a plain exception; app.dependencies turns it into HTTP. That
    file imports Supabase and cannot be imported here, so read it as text — an
    untranslated exception would surface as a 500, not a 404."""
    src = (_ROOT / "app" / "dependencies.py").read_text(encoding="utf-8")
    assert "except WorkspaceAccessDenied" in src, (
        "app.dependencies.resolve_workspace_id no longer catches WorkspaceAccessDenied — "
        "a denied workspace would escape as an unhandled 500."
    )
    assert re.search(r"status_code=denied\.status_code", src), (
        "the boundary must use WorkspaceAccessDenied.status_code, not a hardcoded literal "
        "that can drift from the reason recorded next to it."
    )


def test_user_with_no_requested_workspace_keeps_the_established_one():
    check = member_of(OWN)
    assert resolve(USER, None, is_member=check) == OPERATOR
    assert check.calls == [], "nothing was requested, so nothing needs checking"


def test_a_middleware_validated_workspace_wins_over_the_token_default():
    """`_validate_supabase_jwt` defaults a user's workspace to the operator's, so the
    token value is not intent. X-Workspace-Id (already membership-checked) is."""
    assert resolve(USER, None, is_member=member_of(), validated_workspace_id=OWN) == OWN


def test_requesting_exactly_the_validated_workspace_needs_no_recheck():
    check = member_of()
    assert resolve(USER, OWN, is_member=check, validated_workspace_id=OWN) == OWN
    assert check.calls == [], "already validated upstream; do not pay for it twice"


@pytest.mark.parametrize("blank", ["", "   ", "None", "null", None])
def test_blank_request_values_are_not_treated_as_a_workspace(blank):
    """A literal 'None' string reaching a filter is how an unscoped query happens."""
    check = member_of()
    assert resolve(USER, blank, is_member=check) == OPERATOR
    assert check.calls == []


def test_a_user_with_no_subject_cannot_pass_a_foreign_workspace():
    with pytest.raises(WorkspaceAccessDenied):
        resolve({"workspace_id": OPERATOR}, OTHER, is_member=member_of(OTHER))
