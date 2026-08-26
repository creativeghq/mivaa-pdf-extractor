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


def test_the_seeded_prompt_never_contains_an_internal_identifier():
    """A `?q=` seed is rendered by AgentHub as the USER'S OWN message.

    The fallback used to seed `Show me today's findings for tracked_job_id <uuid>`, so clicking
    the digest showed the owner a raw uuid attributed to themselves — and spent a whole agent turn
    re-fetching findings the digest already had in hand. Name the search the way its owner named
    it, or (better, and what the code now does) link to the conversation the findings were posted
    into.
    """
    src = _SERVICE.read_text(encoding="utf-8")
    fn = _func(_tree(), "_build_action_path")
    # CODE only. The docstring names the old prompt on purpose — recording the defect is not
    # committing it, and a guard that cannot tell those apart makes the record unwritable.
    stmts = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                            and isinstance(fn.body[0].value, ast.Constant)
                            and isinstance(fn.body[0].value.value, str)) else fn.body
    body = "\n".join(ast.get_source_segment(src, s) or "" for s in stmts)
    assert "tracked_job_id" not in body, (
        "_build_action_path() puts `tracked_job_id` into text the user reads. Use the search's "
        "label."
    )
    assert "label" in body, "_build_action_path()'s fallback should name the search by its label"
    # And no id is interpolated into the seed by any other route.
    for node in ast.walk(fn):
        if isinstance(node, ast.JoinedStr):
            rendered = ast.get_source_segment(src, node) or ""
            assert "_id" not in rendered, f"an identifier is being formatted into a seed: {rendered}"


def test_the_digest_opens_a_conversation_rather_than_skipping_the_chat_post():
    """`source_conversation_id` NULL meant the findings were posted NOWHERE a person looks.

    It is set only when the search was created from a chat turn that already had a conversation.
    Everything else left it NULL, and NULL made `_dispatch_for_user` `continue` past
    `_post_findings_to_chat` — so the product of the whole feature never reached a chat, and the
    bell fell back to a seeded prompt. Measured 2026-08-26: 1 of 1 tracked_jobs had it NULL.
    """
    tree = _tree()
    ensure = _func(tree, "_ensure_digest_conversation")
    src = _SERVICE.read_text(encoding="utf-8")
    ensure_src = ast.get_source_segment(src, ensure) or ""
    assert "agent_chat_conversations" in ensure_src, "it must actually create the conversation"
    assert "source_conversation_id" in ensure_src, "and stamp it back, so every digest lands in the same thread"

    # Both dispatch paths resolve it before building the link.
    #
    # The CALL, via the AST — not the NAME in the text. The first version of this checked
    # `"_ensure_digest_conversation" in <function source>` and passed with the call deleted,
    # because a comment two lines further down mentions the helper by name. Same failure as the
    # OrderLinkPicker guard in docs/prevention-coverage.md: something else in the file answered
    # on the call site's behalf.
    for fname in ("_dispatch_for_user", "dispatch_burst_if_warranted"):
        fn_node = _func(tree, fname)
        calls = [
            n for n in ast.walk(fn_node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_ensure_digest_conversation"
        ]
        assert calls, (
            f"{fname}() builds the notification link without CALLING _ensure_digest_conversation "
            "— the link will point at a seeded prompt instead of at the findings"
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
