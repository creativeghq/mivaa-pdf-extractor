"""Behavioral tests for the workspace-binding rule (BOLA, invariant 1, #250).

Real tests, not a source scan — `app.auth.workspace_resolution` is deliberately free
of service imports so this can exercise the actual function against a stubbed
membership check.

The two branches that matter, and why getting either wrong is severe:

  - Reject the service key's requested workspace -> every edge function breaks at
    once, because the platform identity is not tenant-scoped and the workspace can
    only come from the request.
  - Accept an end user's requested workspace without a membership check -> any logged
    in user reads any tenant's data. That is the live shape on the excluded prefixes
    (`/api/rag`, `/api/internal`, `/api/interior`), where the middleware never runs.
"""

import importlib.util
from pathlib import Path

import pytest
from fastapi import HTTPException

# Loaded by path, not by `import app.auth...`, because tests/unit/test_table_extraction.py
# registers a bare ModuleType("app") in sys.modules to avoid executing the real package —
# and it collects first, so `app` is no longer a package by the time this module imports.
# The repo idiom for that is a by-path load, and this module has no service imports to
# resolve anyway.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "app" / "auth" / "workspace_resolution.py"
_spec = importlib.util.spec_from_file_location("workspace_resolution_under_test", _MODULE_PATH)
_wr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wr)

SERVICE_SUBJECT = _wr.SERVICE_SUBJECT
is_service_caller = _wr.is_service_caller
resolve_workspace_id = _wr.resolve_workspace_id

OWN = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
OPERATOR = "99999999-9999-9999-9999-999999999999"

USER = {"sub": "user-abc", "workspace_id": OPERATOR}
SERVICE = {"sub": SERVICE_SUBJECT, "service": "mivaa", "workspace_id": OPERATOR}


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

@pytest.mark.asyncio
async def test_service_key_may_name_any_workspace():
    """The edge function already bound this to a verified user JWT on its side."""
    check = member_of()  # would deny everything
    got = await resolve_workspace_id(SERVICE, OTHER, is_member=check)
    assert got == OTHER
    assert check.calls == [], "the service key must not be membership-checked"


@pytest.mark.asyncio
async def test_service_key_with_no_request_value_falls_back_to_its_own():
    assert await resolve_workspace_id(SERVICE, None, is_member=member_of()) == OPERATOR


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

@pytest.mark.asyncio
async def test_user_requesting_a_workspace_they_belong_to_is_allowed():
    check = member_of(OWN)
    assert await resolve_workspace_id(USER, OWN, is_member=check) == OWN
    assert check.calls == [("user-abc", OWN)]


@pytest.mark.asyncio
async def test_user_requesting_a_foreign_workspace_gets_404():
    """404, not 403 — a 403 confirms the id exists and enumerates workspaces."""
    with pytest.raises(HTTPException) as exc:
        await resolve_workspace_id(USER, OTHER, is_member=member_of(OWN))
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_user_with_no_requested_workspace_keeps_the_established_one():
    check = member_of(OWN)
    assert await resolve_workspace_id(USER, None, is_member=check) == OPERATOR
    assert check.calls == [], "nothing was requested, so nothing needs checking"


@pytest.mark.asyncio
async def test_a_middleware_validated_workspace_wins_over_the_token_default():
    """`_validate_supabase_jwt` defaults a user's workspace to the operator's, so the
    token value is not intent. X-Workspace-Id (already membership-checked) is."""
    got = await resolve_workspace_id(
        USER, None, is_member=member_of(), validated_workspace_id=OWN
    )
    assert got == OWN


@pytest.mark.asyncio
async def test_requesting_exactly_the_validated_workspace_needs_no_recheck():
    check = member_of()
    got = await resolve_workspace_id(
        USER, OWN, is_member=check, validated_workspace_id=OWN
    )
    assert got == OWN
    assert check.calls == [], "already validated upstream; do not pay for it twice"


@pytest.mark.parametrize("blank", ["", "   ", "None", "null", None])
@pytest.mark.asyncio
async def test_blank_request_values_are_not_treated_as_a_workspace(blank):
    """A literal 'None' string reaching a filter is how an unscoped query happens."""
    check = member_of()
    assert await resolve_workspace_id(USER, blank, is_member=check) == OPERATOR
    assert check.calls == []


@pytest.mark.asyncio
async def test_a_user_with_no_subject_cannot_pass_a_foreign_workspace():
    with pytest.raises(HTTPException) as exc:
        await resolve_workspace_id(
            {"workspace_id": OPERATOR}, OTHER, is_member=member_of(OTHER)
        )
    assert exc.value.status_code == 404
