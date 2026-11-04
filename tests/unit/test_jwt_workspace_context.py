import asyncio
from types import SimpleNamespace

import pytest

jwt_auth = pytest.importorskip("app.middleware.jwt_auth")

PLATFORM, MINE, OTHER, FOREIGN = "platform-ws", "ws-a", "ws-b", "ws-foreign"


def _context(memberships, requested=None):
    m = jwt_auth.JWTAuthMiddleware.__new__(jwt_auth.JWTAuthMiddleware)
    m.settings = SimpleNamespace(environment="production")
    m._is_test_user = lambda claims: False

    async def validate(user_id, workspace_id):
        return workspace_id in memberships

    async def active(user_id):
        return sorted(memberships)

    m._validate_workspace_access = validate
    m._active_workspace_ids = active
    claims = {"sub": "u", "workspace_id": PLATFORM, "role": "authenticated"}
    return asyncio.run(m._extract_workspace_context(claims, requested))


def test_a_user_of_several_workspaces_reaches_the_one_they_named():
    assert _context({MINE, OTHER}, MINE).workspace_id == MINE


def test_naming_a_workspace_you_do_not_belong_to_grants_nothing():
    assert _context({MINE, OTHER}, FOREIGN) is None
    assert _context({MINE}, FOREIGN).workspace_id == MINE


def test_several_workspaces_and_no_choice_stays_refused():
    assert _context({MINE, OTHER}) is None


def test_one_membership_is_unambiguous():
    assert _context({MINE}).workspace_id == MINE
