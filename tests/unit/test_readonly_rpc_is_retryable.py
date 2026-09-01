"""Guard: a read-only RPC is retryable, and a permanent rejection is not retried.

WHY THIS EXISTS
---------------
PostgREST maps insert, upsert AND rpc onto POST, so the central retry patch in
`app.services.core.supabase_client` cannot tell a SELECT from a credit debit and refuses
to repeat any of them. That rule is correct. Its cost landed on the crons: they call
PostgREST once an hour, Supabase closes the pooled keep-alive connection in between, and
the first call of the tick dies with "Server disconnected". When that call is the
`..._due` RPC deciding what to work on, the ENTIRE tick is skipped — while pg_cron records
`succeeded`, because all it did was queue the HTTP request. Ten of the twelve Sentry
issues of this shape in the 17 days to 2026-09-01 were exactly that (MIVAA-5KH and kin).

Three separate defects were fixed together, and each is pinned below:

  1. A read-only RPC now goes over GET (`read_rpc`), which lands in the retry patch's
     idempotent set. There is deliberately NO list of "safe function names" here to drift:
     PostgREST refuses GET on a VOLATILE function with 405 / SQLSTATE 25006 ("cannot
     execute UPDATE in a read-only transaction"), verified against the live database on
     2026-09-01. `pg_proc.provolatile` stays the single source of truth. What CAN silently
     go wrong is the argument shapes, so that is what is checked here.

  2. `should_retry_exception` consulted the message TEXT before any structured error code.
     A CHECK violation quotes the FAILING ROW back at you, so a row that merely mentioned
     a connection or a timeout was read as a transient network fault — that is how a
     permanent 23514 on `agent_run_logs_level_check` was reported as "PostgREST transient
     failure" (MIVAA-5JV). The code now decides first.

  3. The retry log interpolated the raw exception. A Cloudflare 522 for a Supabase origin
     timeout is ~6 KB of HTML inside the APIError, so that page became the Sentry ISSUE
     TITLE (MIVAA-5K3) and a 6 KB `system_logs` row per retry decision.

Source/stdlib based: imports no app package and touches no DB, so it runs in MIVAA's CI
(which installs pytest and nothing else) in about a second.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"

_NL = chr(10)


def _load(name, relpath):
    """Import a stdlib-only app module by path, without importing the `app` package."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


readonly_rpc = _load("_readonly_rpc", "app/utils/readonly_rpc.py")
retry_helper = _load("_retry_helper", "app/utils/retry_helper.py")


class _FakeAPIError(Exception):
    """Shaped like postgrest's APIError: a dict payload plus a `.code` attribute."""

    def __init__(self, payload):
        super().__init__(payload)
        self.code = payload.get("code")


# ── 1. Which argument shapes may ride in a query string ─────────────────────────────

def test_scalar_arguments_are_url_safe():
    assert readonly_rpc.read_rpc_param_error("f", {"p_limit": 50}) is None
    assert readonly_rpc.read_rpc_param_error("f", {"p_query": "tile", "p_limit": 24}) is None
    assert readonly_rpc.read_rpc_param_error("f", None) is None
    assert readonly_rpc.read_rpc_param_error("f", {}) is None


def test_none_is_refused_because_get_turns_it_into_an_empty_string():
    # Over POST this is SQL NULL; over GET httpx sends "" and PostgREST fails the cast.
    # `read_document_chunk_span` passes a routinely-None p_product_id, which is exactly
    # why it was left on POST.
    reason = readonly_rpc.read_rpc_param_error("f", {"p_product_id": None})
    assert reason is not None
    assert "p_product_id" in reason


def test_array_arguments_are_refused():
    # A Postgres array argument must arrive as the literal {a,b}; httpx repeats the key.
    for value in ([1, 2], (1, 2), {"a": 1}, {1, 2}):
        reason = readonly_rpc.read_rpc_param_error("f", {"p_ids": value})
        assert reason is not None, value
        assert "p_ids" in reason


def test_overlong_arguments_are_refused_before_cloudflare_says_414():
    reason = readonly_rpc.read_rpc_param_error("f", {"p_q": "x" * 5000})
    assert reason is not None
    assert str(readonly_rpc.MAX_QUERY_STRING_LENGTH) in reason


# ── 2. A structured error code outranks the message text ────────────────────────────

@pytest.mark.parametrize("code,expected", [
    ("23514", False),   # CHECK violation — the MIVAA-5JV shape
    ("23505", False),   # unique violation
    ("42501", False),   # RLS / permission denied
    ("22P02", False),   # invalid text representation
    ("PGRST116", False),
    ("08006", True),    # connection failure
    ("40001", True),    # serialization failure
    ("57014", True),    # query canceled
    ("522", True),      # Cloudflare: origin timed out
    ("503", True),
    ("404", False),
    ("", None),
])
def test_error_code_verdicts(code, expected):
    assert retry_helper.classify_error_code(code) is expected


def test_a_permanent_violation_is_not_retried_even_when_the_row_mentions_a_connection():
    """The regression that motivated the reordering (MIVAA-5JV).

    The words come from the DATA, not from the fault: a CHECK violation quotes the failing
    row, and this platform logs agent runs whose messages are about network errors.
    """
    exc = _FakeAPIError({
        "code": "23514",
        "message": 'new row for relation "agent_run_logs" violates check constraint'
                   ' "agent_run_logs_level_check"',
        "details": 'Failing row contains (.., "connection reset by peer, timed out", ..)',
    })
    assert retry_helper.should_retry_exception(exc) is False


def test_the_idle_keepalive_drop_is_still_retryable():
    """The failure this whole change exists to survive.

    httpx.RemoteProtocolError is NOT an httpx.NetworkError, so before this it was caught
    only by the accident of "disconnected" containing the substring "connect".
    """
    httpx = pytest.importorskip("httpx")
    exc = httpx.RemoteProtocolError("Server disconnected without sending a response.")
    assert not isinstance(exc, httpx.NetworkError)
    assert retry_helper.should_retry_exception(exc) is True


def test_a_gateway_timeout_is_retryable_by_code_not_by_reading_the_page():
    exc = _FakeAPIError({"code": 522, "message": "JSON could not be generated",
                         "details": "<!DOCTYPE html><html><title>522</title></html>"})
    assert retry_helper.should_retry_exception(exc) is True


# ── 3. An error page never becomes the log line ─────────────────────────────────────

def test_describe_exception_drops_the_html_page_and_keeps_the_diagnosis():
    page = "<!DOCTYPE html>" + "<div>filler</div>" * 500
    exc = _FakeAPIError({"code": 522, "message": "JSON could not be generated",
                         "details": page})
    assert len(str(exc)) > 4000

    brief = retry_helper.describe_exception(exc)
    assert len(brief) < 400
    assert "<div>" not in brief
    assert "JSON could not be generated" in brief
    assert "522" in brief
    assert brief.startswith("_FakeAPIError[522]:")


def test_describe_exception_is_single_line():
    exc = ValueError("first line" + _NL + "second line")
    assert _NL not in retry_helper.describe_exception(exc)


# ── 4. Every read_rpc call site passes arguments that survive a query string ────────

def _read_rpc_calls():
    """(path, lineno, func_name, dict_node | None) for every read_rpc(...) call in app/."""
    found = []
    for path in sorted(_APP.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Name) and fn.id == "read_rpc"):
                continue
            name = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                name = node.args[1].value
            params = node.args[2] if len(node.args) >= 3 else None
            found.append((path, node.lineno, name, params))
    return found


def test_read_rpc_is_actually_used():
    calls = _read_rpc_calls()
    assert len(calls) >= 5, (
        "read_rpc has no call sites — the cron reads went back to POST and lost their retry"
    )


def test_every_read_rpc_call_site_passes_url_safe_literals():
    """The runtime check inside `read_rpc` raises; this one fails the BUILD instead.

    A cron path only exercises its failure mode on the tick after an idle gap, so a bad
    argument shape added here would sit unnoticed until the next disconnect.
    """
    problems = []
    for path, lineno, name, params in _read_rpc_calls():
        rel = path.relative_to(_ROOT)
        if params is None:
            continue
        if not isinstance(params, ast.Dict):
            problems.append(
                f"{rel}:{lineno} read_rpc({name}) params are not a dict literal,"
                f" so this test cannot verify them"
            )
            continue
        for key, value in zip(params.keys, params.values):
            label = key.value if isinstance(key, ast.Constant) else ast.dump(key)
            if isinstance(value, ast.Constant) and value.value is None:
                problems.append(
                    f"{rel}:{lineno} read_rpc({name}) argument {label!r} is None; a query"
                    f" string sends that as an empty string and PostgREST fails the cast"
                )
            if isinstance(value, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                problems.append(
                    f"{rel}:{lineno} read_rpc({name}) argument {label!r} is a collection;"
                    f" httpx repeats the key instead of sending a Postgres array literal"
                )
    assert not problems, (
        "read_rpc call sites that cannot survive a query string:" + _NL + _NL.join(problems)
    )


# ── 5. The mechanism the fix depends on ─────────────────────────────────────────────

def test_the_retry_patch_still_treats_get_as_idempotent():
    """`read_rpc` buys retry by making the request a GET.

    If the patch stops trusting GET, every one of those reads silently goes back to being
    un-retryable — with no error anywhere, which is how this cost ten Sentry issues.
    """
    source = (_APP / "services" / "core" / "supabase_client.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_IDEMPOTENT_METHODS" for t in node.targets)
    ]
    assert literals, "_IDEMPOTENT_METHODS is gone from the retry patch"
    methods = {n.value for n in ast.walk(literals[0].value) if isinstance(n, ast.Constant)}
    assert "GET" in methods
    assert "POST" not in methods, "POST must stay non-idempotent — that is the whole guard"


def test_read_rpc_sends_get():
    source = (_APP / "services" / "core" / "supabase_client.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "read_rpc")
    rpc_calls = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "rpc"]
    assert len(rpc_calls) == 1
    kwargs = {k.arg: k.value for k in rpc_calls[0].keywords}
    assert "get" in kwargs, "read_rpc must pass get=True or it is just an ordinary POST"
    assert isinstance(kwargs["get"], ast.Constant) and kwargs["get"].value is True


def test_read_rpc_validates_before_calling():
    """The check has to run BEFORE the request is built, or it is not a check."""
    source = (_APP / "services" / "core" / "supabase_client.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "read_rpc")
    validate_line = min(n.lineno for n in ast.walk(fn)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == "read_rpc_param_error")
    rpc_line = min(n.lineno for n in ast.walk(fn)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "rpc")
    assert validate_line < rpc_line
