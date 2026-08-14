"""
Every prompt comes from the database, and no code fallback may reappear (#347 phase 3P).

THE FAILURE THIS PREVENTS
-------------------------
`segmentation_service._get_prompt` read the `prompts` table, caught every exception, logged at
DEBUG, and returned a 9,119-character DEFAULT_SEGMENT_PROMPT. An admin editing that prompt in
/admin/ai-configs would have seen it save and changed nothing, forever, while every health
signal stayed green. The prompt in the database and the prompt sent to the model were different
documents and nothing in the system could tell you that.

`enhanced_material_property_extractor` had the one-line version: `self._db_prompt or
EXTRACTION_SYSTEM_PROMPT`.

Both are the same shape as the silent-zero rule in CLAUDE.md — a wrong answer that is a
perfectly valid value, so no typecheck and no integrity probe can see it.

WHY SOURCE-BASED
----------------
Runs in CI in ~2s with no database and no credentials.
"""
import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
_REGISTRY = _APP / "services/utilities/prompt_registry.py"

#: A file that reaches a model.
_LLM_MARKERS = (
    "tracked_claude_call_async", "messages.create", "anthropic.com/v1/messages",
    "claude_helper", "_call_claude",
)

#: Instruction-shaped text. Used only to describe a literal we ALREADY know is long and lives in
#: a file that calls a model — never as the sole test, because prompts do not reliably look like
#: anything (the edge-side count was wrong twice for exactly that reason).
_HINTS = ("you are", "respond", "return json", "extract", "analyze", "analyse",
          "classify", "json object", "rules:")

_MIN_PROMPT_CHARS = 220


def _sources():
    for path in sorted(_APP.rglob("*.py")):
        yield path, path.read_text(encoding="utf-8")


def _prompt_literals(tree):
    """Long instruction-shaped literals, excluding docstrings and f-string innards."""
    docstrings, nested = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                docstrings.add(id(node.body[0].value))
        if isinstance(node, ast.JoinedStr):
            nested.update(id(sub) for sub in ast.walk(node) if sub is not node)

    for node in ast.walk(tree):
        if id(node) in docstrings or id(node) in nested:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(v.value for v in node.values
                           if isinstance(v, ast.Constant) and isinstance(v.value, str))
        else:
            continue
        if len(text) < _MIN_PROMPT_CHARS:
            continue
        if sum(h in text.lower() for h in _HINTS) < 2:
            continue
        yield node, text


def test_no_prompt_is_hardcoded():
    offenders = []
    for path, src in _sources():
        if not any(m in src for m in _LLM_MARKERS):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:  # pragma: no cover — the syntax gate catches this first
            continue
        for node, text in _prompt_literals(tree):
            offenders.append(
                f"{path.relative_to(_ROOT).as_posix()}:{node.lineno} "
                f"({len(text)} chars) {' '.join(text.split())[:60]!r}"
            )

    assert offenders == [], (
        "a prompt is hardcoded in a file that calls a model:\n  " + "\n  ".join(offenders)
        + "\n\nPrompts live in the `prompts` table (#347 phase 3P). Load with "
          "`prompt_registry.load_prompt(...)`, or `get_cached(...)` at a sync site whose async "
          "entry point called `prefetch(...)`. Seed the row before deleting the literal."
    )


def test_no_loader_falls_back_to_code():
    """No `except: return DEFAULT`, no `db_prompt or CONSTANT`, anywhere near a prompt."""
    offenders = []
    for path, src in _sources():
        if "prompt" not in src.lower() or path == _REGISTRY:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:  # pragma: no cover
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.unparse(fn)
            if "prompt" not in body.lower():
                continue
            for node in ast.walk(fn):
                # `something_prompt or SOME_CONSTANT`
                if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
                    rendered = ast.unparse(node)
                    if "prompt" in rendered.lower() and any(
                        isinstance(v, ast.Name) and v.id.isupper() for v in node.values
                    ):
                        offenders.append(
                            f"{path.relative_to(_ROOT).as_posix()}::{fn.name} -> {rendered[:70]}")
                # `except ...: return SOME_CONSTANT`
                if isinstance(node, ast.ExceptHandler):
                    for ret in ast.walk(node):
                        if (isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
                                and ret.value.id.isupper()):
                            offenders.append(
                                f"{path.relative_to(_ROOT).as_posix()}::{fn.name} -> "
                                f"except: return {ret.value.id}")

    assert offenders == [], (
        "a prompt path falls back to a code constant:\n  " + "\n  ".join(offenders)
        + "\n\nThis is the segmentation_service bug: the fallback is invisible when it fires, so "
          "an admin's edit silently does nothing forever. A missing prompt must raise."
    )


def test_the_registry_separates_missing_from_unreachable():
    """`PromptNotConfigured` and `PromptStoreUnavailable` must stay distinct.

    The six loaders this replaced all returned `None` for both, so no caller could tell "add the
    row" from "the database is down" — which is how an outage looked like a misconfiguration for
    as long as nobody looked.
    """
    src = _REGISTRY.read_text(encoding="utf-8")
    tree = ast.parse(src)

    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert {"PromptNotConfigured", "PromptStoreUnavailable"} <= classes, (
        f"the registry no longer defines both prompt failure types: {sorted(classes)}")

    resolver = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "_resolve_blocking")
    raised = {
        r.exc.func.id for r in ast.walk(resolver)
        if isinstance(r, ast.Raise) and isinstance(r.exc, ast.Call)
        and isinstance(r.exc.func, ast.Name)
    }
    assert raised == {"PromptNotConfigured", "PromptStoreUnavailable"}, (
        f"_resolve_blocking must raise exactly those two, not {sorted(raised)} — collapsing them "
        f"is the bug this replaced.")

    # It must never answer with a default instead of raising.
    returns = [r for r in ast.walk(resolver)
               if isinstance(r, ast.Return) and r.value is not None]
    assert all("rows[0]" in ast.unparse(r) for r in returns), (
        "_resolve_blocking returns something that is not a row from the prompts table — there is "
        "no fallback by design.")


def test_active_prompts_only():
    """Resolution must filter `is_active`.

    It did not before: `_load_prompt_from_database` matched on workspace/type/stage/category and
    never checked the flag, so deactivating a prompt in the admin UI changed nothing about what
    the pipeline actually sent.
    """
    src = _REGISTRY.read_text(encoding="utf-8")
    select = src[src.index("def _select("):src.index("def _resolve_blocking(")]
    assert '.eq("is_active", True)' in select, (
        "the prompt query stopped filtering is_active — deactivating a prompt in /admin/ai-configs "
        "would silently keep serving it.")
