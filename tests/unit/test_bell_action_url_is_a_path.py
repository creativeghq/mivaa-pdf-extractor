"""
A bell notification's `action_url` is a PATH; the absolute URL belongs to the email.

THE FAILURE THIS PREVENTS
-------------------------
`JobDigestDispatcher` built ONE url and gave it to both channels:

    action_url = self._build_action_url(...)      # https://app.materialshub.gr/agent-hub?…
    await self._send_bell(..., action_url=action_url)
    await self._send_email(..., action_url=action_url)

Correct for the email. Fatal for the bell. The frontend bell hands
`user_notifications.action_url` to react-router's `navigate()`, which reads ANY string as a
PATH — so the stored URL became the path `/https://app.materialshub.gr/agent-hub`, matched no
route, and landed on the 404 catch-all. Nineteen digests shipped before anyone clicked one, and
nothing could see it: the row is well-formed, the URL is valid, the notification is the right
notification.

Same shape as the silent-zero rule in CLAUDE.md — a wrong answer that is a perfectly valid value.

WHY SOURCE-BASED
----------------
MIVAA's CI installs pytest and nothing else, so this module cannot be imported. It also should
not need to be: the defect is visible in the call, not in a run.
"""
import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = _ROOT / "app/modules/job_research_notifications/service.py"


def _tree() -> ast.Module:
    return ast.parse(_SERVICE.read_text(encoding="utf-8"))


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name}() is gone from {_SERVICE.name} — this guard no longer guards anything")


def _kwarg(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _calls_to(tree: ast.Module, method: str):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
        ):
            yield node


def test_the_path_builder_never_prefixes_an_origin():
    """`_build_action_path` returns a path. If it learns about PUBLIC_APP_URL again, it is a URL."""
    fn = _func(_tree(), "_build_action_path")
    src = ast.get_source_segment(_SERVICE.read_text(encoding="utf-8"), fn) or ""
    assert "PUBLIC_APP_URL" not in src, (
        "_build_action_path() reads PUBLIC_APP_URL — it is building an absolute URL again. "
        "The bell needs the path; _absolute() is where the origin goes."
    )
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "_build_action_path() returns nothing"
    for r in returns:
        # Every return is a path literal, or a literal concatenated with a query string.
        leftmost = r.value
        while isinstance(leftmost, ast.BinOp):
            leftmost = leftmost.left
        assert isinstance(leftmost, ast.Constant) and isinstance(leftmost.value, str), (
            "every return of _build_action_path() must start with a string literal so this guard "
            "can read it"
        )
        assert leftmost.value.startswith("/"), (
            f"_build_action_path() returns {leftmost.value!r} — the bell feeds this to navigate(), "
            "which treats it as a path. It must start with '/'."
        )


def test_every_bell_send_is_given_a_path_not_a_url():
    """`_send_bell(action_url=…)` is never handed the value `_absolute()` produced."""
    tree = _tree()
    sends = list(_calls_to(tree, "_send_bell"))
    assert sends, "_send_bell() is no longer called — this guard no longer guards anything"
    for call in sends:
        value = _kwarg(call, "action_url")
        assert value is not None, "_send_bell() must be called with an explicit action_url="
        assert isinstance(value, ast.Name), (
            "action_url= on the bell should be a plain local built from _build_action_path()"
        )
        assert value.id != "action_url", (
            f"line {call.lineno}: the bell is being handed `action_url`, which is the ABSOLUTE "
            "email CTA. Pass the path (`action_path`) — navigate() reads an absolute URL as a "
            "path and 404s."
        )


def test_the_email_still_gets_the_absolute_url():
    """The other half: an email CTA must NOT become a bare path, which resolves nowhere in a mail client."""
    tree = _tree()
    sends = list(_calls_to(tree, "_send_email"))
    assert sends, "_send_email() is no longer called — this guard no longer guards anything"
    for call in sends:
        value = _kwarg(call, "action_url")
        assert isinstance(value, ast.Name) and value.id == "action_url", (
            f"line {call.lineno}: the email must be sent the absolute URL built by _absolute()"
        )
    # And that local is in fact built by _absolute(), at every site that defines it.
    absolutes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "action_url" for t in n.targets)
    ]
    assert absolutes, "nothing assigns `action_url` any more"
    for assign in absolutes:
        assert (
            isinstance(assign.value, ast.Call)
            and isinstance(assign.value.func, ast.Attribute)
            and assign.value.func.attr == "_absolute"
        ), f"line {assign.lineno}: `action_url` must come from self._absolute(action_path)"
