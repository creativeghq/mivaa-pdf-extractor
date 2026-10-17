"""X-Acting-User is honoured only for MIVAA's own service identity (#496 on the platform repo).

Source-based, comments blanked: MIVAA's CI installs pytest only, so the app cannot be imported.
"""

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _blank_comments():
    spec = importlib.util.spec_from_file_location("comment_budget", _ROOT / "scripts" / "comment_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.blank_comments


_SRC = _blank_comments()((_ROOT / "app" / "dependencies.py").read_text(encoding="utf-8"))


def _get_current_user_body() -> str:
    start = _SRC.index("async def get_current_user(")
    end = _SRC.index("\nasync def ", start + 1)
    return _SRC[start:end]


def test_the_actor_header_is_read_only_inside_the_service_branch():
    body = _get_current_user_body()
    assert re.search(r'if acting and is_service_caller\(claims\):', body), \
        "X-Acting-User must be honoured ONLY when the caller is MIVAA's service identity"
    header_reads = [m.start() for m in re.finditer(r'X-Acting-User', body)]
    gate = body.index("if acting and is_service_caller(claims):")
    assert header_reads and all(pos < gate for pos in header_reads[:1])
    assert "uuid.UUID(acting)" in body[gate:], "a malformed actor must be refused, not passed through"


def test_the_override_returns_a_plain_user_not_a_service_caller():
    body = _get_current_user_body()
    branch = body[body.index("if acting and is_service_caller(claims):"):]
    returned = re.search(r'return \{[^}]*\}', branch).group(0)
    assert not re.search(r'"service"\s*:', returned) and "material-kai-platform" not in returned
    assert '"sub": acting_id' in returned
