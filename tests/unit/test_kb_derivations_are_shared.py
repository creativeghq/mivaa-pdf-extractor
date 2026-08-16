"""Guard: one derivation each for the KB query vector, the caller, and the read scope.

The bug this exists to stop
---------------------------
There are two KB search endpoints and both are needed — `/api/kb/search` (admin UI,
whole-doc `kb_docs`, plus full-text and hybrid) and `/api/rag/search/knowledge-base`
(the `knowledge_base_search` agent tool, section-level `kb_doc_chunks`). Different
corpora, different consumers. What they are NOT allowed to do is each keep a private
copy of the parts that must agree.

They did, and it drifted exactly the way this codebase's other copy-paste pairs drift:

    rag_routes        entity_type="query"    → Voyage input_type="query"   ✅
    knowledge_base    entity_type="search"   → falls through to "document" ❌

`input_type = "query" if entity_type == "query" else "document"` is one exact-string
equality, so any near-miss silently produces a DOCUMENT vector and compares it against
document-side vectors. Same model, same 1024 dimensions, same column — no exception, no
zero, no failed probe, just quietly worse ranking. `rag_routes` had already FOUND and
FIXED this, with a comment naming it. The fix did not reach the copy, because nothing
connected them.

Same shape as `escapeHtml` drifting to three different strengths in the parent repo, and
as the money quantity that was derived five times. The rule there is the rule here: one
derivation, many consumers, each formatting the result differently.

What is deliberately NOT pinned
-------------------------------
The corpus, the similarity threshold (0.5 vs a measured 0.4), the response shape, and
which search modes exist. Those are real differences between two real endpoints, and
forcing them into one function would be the opposite mistake.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
KB_ACCESS = APP / "services" / "kb" / "kb_access.py"

#: The one module allowed to decide how a KB query becomes a vector.
_OWNER = "app/services/kb/kb_access.py"

#: `entity_type` values that mean "this text is a SEARCH QUERY". Anything in here, used
#: outside the owner module, is a second derivation of the input_type decision.
_QUERY_ISH = re.compile(
    r"entity_type\s*=\s*[\"'](query|search|kb_query|user_query|search_query)[\"']"
)


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def _code_only(src: str) -> str:
    """Blank out comments and docstrings, PRESERVING line numbers.

    Guards in this repo explain the bug they prevent in prose, so the fixed code sits
    next to a comment quoting the broken code. A text scanner that reads the comment
    reports the fix as the defect — this file did exactly that on its first run, and
    `test_paid_route_metering` carries the same helper for the same reason.

    Newlines are kept so reported line numbers still point at real source.
    """

    def _blank(match: re.Match) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    src = re.sub(r'"""[\s\S]*?"""', _blank, src)
    src = re.sub(r"'''[\s\S]*?'''", _blank, src)
    src = re.sub(r"(?m)#.*$", _blank, src)
    return src


def _app_files() -> list[Path]:
    return sorted(APP.rglob("*.py"))


def test_the_owner_module_exists_and_pins_input_type():
    assert KB_ACCESS.exists(), (
        "app/services/kb/kb_access.py is gone — the KB derivations have been scattered "
        "back into the routes"
    )
    src = KB_ACCESS.read_text(encoding="utf-8")
    assert '_QUERY_ENTITY_TYPE = "query"' in src, (
        "the query entity_type is no longer pinned in one place; the whole point is "
        "that this exact string is decided once"
    )
    assert "_QUERY_ENTITY_TYPE" in src.split("async def kb_query_vector", 1)[1], (
        "kb_query_vector no longer uses the pinned constant"
    )


def test_only_the_owner_module_builds_a_kb_query_vector():
    """No file outside kb_access may pass a search-ish entity_type.

    Full-tree, not a declared list of the two files that were wrong — #15's headline
    finding was that this repo's guards scan lists while claiming to cover classes, and
    every defect found landed in the gap.
    """
    offenders = []
    scanned = 0
    for path in _app_files():
        rel = _rel(path)
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:
            continue
        scanned += 1
        if rel == _OWNER:
            continue
        for match in _QUERY_ISH.finditer(_code_only(src)):
            lineno = src[: match.start()].count("\n") + 1
            offenders.append(f"{rel}:{lineno} — {match.group(0)}")

    assert scanned > 100, f"only {scanned} files scanned; the walk is broken"
    assert not offenders, (
        "a KB query vector is being built outside app/services/kb/kb_access.py:\n"
        + "\n".join(f"  {o}" for o in offenders)
        + "\n\nCall kb_query_vector() instead. The input_type decision is one exact "
        "string comparison, and a second copy of it is how one endpoint ended up "
        "embedding its query as a document."
    )


def test_the_access_scope_has_one_definition():
    """`resolve_kb_access_scope` may be bound per corpus, but not reimplemented.

    A read-by-id endpoint that re-derives this slightly differently is a BOLA hole —
    the caller supplies the kb_doc_id.
    """
    definitions = []
    for path in _app_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.lstrip("_") != "resolve_kb_access_scope":
                continue
            body = ast.unparse(node)
            # A thin binding that delegates is fine; a reimplementation is not.
            if "resolve_kb_access_scope(" in body.split("\n", 1)[1]:
                continue
            definitions.append(f"{_rel(path)}:{node.lineno}")

    assert definitions == [f"{_OWNER}:{_scope_lineno()}"], (
        f"KB access scope is implemented in {definitions}. It must be defined once, in "
        f"{_OWNER}; per-corpus differences belong in an ARGUMENT (per_doc_agent_gate), "
        "not in a second function."
    )


def _scope_lineno() -> int:
    tree = ast.parse(KB_ACCESS.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_kb_access_scope":
            return node.lineno
    raise AssertionError("resolve_kb_access_scope not found in the owner module")


def test_per_doc_agent_gate_is_required_and_never_defaulted():
    """There is no value that is right for both corpora, so there is no safe default.

    `kb_match_doc_chunks` enforces category access_level AND per-doc allowed_agents
    inside the RPC, so an agent may read a `visibility='private'` doc there — private
    means "not published to the public KB website", not "hidden from agents".
    `kb_match_docs` enforces neither, so `include_private` is the only thing between a
    non-admin and private content and must track admin-ness. Passing True from the
    second corpus silently opens every private doc in the workspace to any member.
    """
    tree = ast.parse(KB_ACCESS.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_kb_access_scope":
            kwonly = [a.arg for a in node.args.kwonlyargs]
            assert "per_doc_agent_gate" in kwonly, (
                "per_doc_agent_gate must be KEYWORD-ONLY so a positional argument "
                "cannot land on it by accident"
            )
            idx = kwonly.index("per_doc_agent_gate")
            assert node.args.kw_defaults[idx] is None, (
                "per_doc_agent_gate has a default — whichever value you chose is wrong "
                "for one of the two corpora, and the caller who forgets it is the one "
                "who gets the wrong one"
            )
            return
    raise AssertionError("resolve_kb_access_scope not found")


def test_no_route_takes_its_kb_caller_from_the_request_body():
    """`caller` decides access levels AND include_private. It cannot come from the body.

    `PriceLookupDrawer` sends `caller: 'admin'` from the frontend through
    mivaa-gateway, and the gateway deliberately forwards the END USER's JWT for
    `/api/rag/*` (so MIVAA enforces ownership). So `request.caller or "agent"` honoured
    an admin claim made on an ordinary user token — the same defect as MV2-12, one door
    along. `resolve_kb_caller` honours a platform service credential, lets any caller
    narrow, and clamps a widening request.
    """
    offenders = []
    for path in _app_files():
        rel = _rel(path)
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in re.finditer(r"caller\s*=\s*request\.caller", _code_only(src)):
            lineno = src[: match.start()].count("\n") + 1
            offenders.append(f"{rel}:{lineno}")

    assert not offenders, (
        "KB caller taken straight from the request body at: "
        + ", ".join(offenders)
        + ". Use `await resolve_kb_caller(supabase, claims, request.caller, "
        "workspace_id)` — a body may narrow the scope, never widen it."
    )


def test_the_detector_catches_the_shape_it_was_built_for():
    """Guard the guard — an empty result from a detector never shown to fire proves nothing."""
    assert _QUERY_ISH.search('entity_type="search",'), "the original bug would not be caught"
    assert _QUERY_ISH.search("entity_type='query',"), "the sibling's value would not be caught"
    assert not _QUERY_ISH.search('entity_type="kb_doc",'), (
        "indexing calls must NOT be flagged — documents are embedded in document mode, "
        "which is correct"
    )
