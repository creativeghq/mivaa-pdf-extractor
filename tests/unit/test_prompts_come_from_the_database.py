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
#: A file is only scanned if it makes an outbound model call. This list used to be
#: ANTHROPIC-ONLY, so `perplexity_price_search_service` — which posts to
#: api.perplexity.ai — was invisible to the whole guard while holding a 2,000-char
#: system prompt that decides what price data enters the platform (audit #14 MV-2).
#: Vision is Anthropic-only; TEXT generation is not, and the rule is about prompts.
_LLM_MARKERS = (
    # Anthropic
    "tracked_claude_call_async", "messages.create", "anthropic.com/v1/messages",
    "claude_helper", "_call_claude",
    # everyone else this platform actually calls
    "api.perplexity.ai", "api.openai.com", "api.voyageai.com",
    "api.replicate.com", "generativelanguage.googleapis.com",
    # Firecrawl's JSON format runs an LLM extraction on the page, so a
    # `.scrape(..., extraction_prompt=...)` IS a model call — the prompt is just
    # executed on someone else's infrastructure. Five marketplace adapters and the
    # price verifier reach a model ONLY this way and carried no provider string at
    # all, so the sweep skipped their files entirely (#33 item 3).
    "firecrawl", "FirecrawlClient", "get_firecrawl_client",
)

#: Argument names and payload keys that hand text to a model.
_MODEL_ARGS = frozenset({
    "messages", "system", "prompt", "input", "contents",
    # Firecrawl's extraction prompt. Same contract as `prompt`: free text handed to a
    # model, and a site that changes its markup is exactly when it needs editing
    # without a deploy.
    "extraction_prompt",
})

_MIN_PROMPT_CHARS = 220


def _sources():
    for path in sorted(_APP.rglob("*.py")):
        yield path, path.read_text(encoding="utf-8")


def _model_payload_regions(tree):
    """AST subtrees that hand text to a model.

    Two shapes, because the payload is not always inline at the call:
      * a Call with a `messages=` / `system=` / `prompt=` keyword, or `json={...}`
        whose dict has one of those keys;
      * ANY dict literal carrying one of those keys — which is how
        `perplexity_price_search_service` does it: `body = {"messages": [...]}` is
        built first and passed as `json=body` several lines later, so the literal
        never appears inside the call node at all.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if {k.arg for k in node.keywords if k.arg} & _MODEL_ARGS:
                yield node
                continue
            for kw in node.keywords:
                if kw.arg == "json" and isinstance(kw.value, ast.Dict):
                    keys = {
                        k.value for k in kw.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }
                    if keys & _MODEL_ARGS:
                        yield node
                        break
        elif isinstance(node, ast.Dict):
            keys = {
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            if keys & _MODEL_ARGS:
                yield node


def _prompt_literals(tree):
    """Long literals that REACH a model call. Structural, not keyword-based.

    The rule used to be `length >= 220 AND >= 2 words from a hint list`, and the
    hints were the only discriminator after length — which this file's own docstring
    forbids, in those words, because "prompts do not reliably look like anything".
    It let through `rag_service`'s "You are an expert document analyst. Answer the
    following question based ONLY on the provided context." — 376 chars, in a file
    with four LLM markers — because "document analyst" does not contain "analyze"
    and one hint is not two (audit #14 MV-2).

    A literal now offends if it is long and either sits inside a model payload, or is
    bound to a NAME that appears inside one. The name hop is what reaches across a
    function boundary: Perplexity's system prompt is assigned in `_build_messages`
    and consumed in the caller's `body["messages"]`.

    This is deliberately not "every long literal in a file that calls a model" —
    that flags 56 strings including a base64 test fixture, an OpenAPI description and
    an HTML page, and would need an allowlist big enough to rot.
    """
    candidates = {}
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
        # A prompt is prose. This drops base64 blobs and minified payloads without
        # asking what WORDS they contain — the base64 JPEG fixture in
        # anthropic_routes is 378 chars and reaches a model payload, and it is not
        # a prompt. Structural, so it cannot become a keyword list by drift.
        if len(text.split()) < 10:
            continue
        candidates[id(node)] = (node, text)

    if not candidates:
        return

    regions = list(_model_payload_regions(tree))
    if not regions:
        return

    # Names (and attribute names) referenced anywhere inside a model payload.
    reachable = set()
    for region in regions:
        for sub in ast.walk(region):
            if isinstance(sub, ast.Name):
                reachable.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                reachable.add(sub.attr)

    offending = {}
    for region in regions:
        for sub in ast.walk(region):
            if id(sub) in candidates:
                offending[id(sub)] = candidates[id(sub)]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if id(node.value) not in candidates:
            continue
        for target in node.targets:
            name = (target.id if isinstance(target, ast.Name)
                    else target.attr if isinstance(target, ast.Attribute) else None)
            if name and name in reachable:
                offending[id(node.value)] = candidates[id(node.value)]

    for node, text in offending.values():
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


def test_required_prompts_matches_every_literal_call_site():
    """`REQUIRED_PROMPTS` must list every prompt this service loads with a literal key.

    The list is DECLARED, not derived, because a running service cannot AST-scan itself to build
    its own health check. A declared list is exactly the kind that rots, so this walks every
    `load_prompt(...)` / `get_cached(...)` call in app/ and compares both ways.

    Without it, /health would confidently report "all required prompts present" while a call site
    loaded a key nobody was checking for — the same silent-zero shape the probe exists to prevent.
    """
    import re

    registry_src = _REGISTRY.read_text(encoding="utf-8")
    block = registry_src[
        registry_src.index("REQUIRED_PROMPTS"):registry_src.index("async def check_required_prompts")
    ]
    declared = {
        (t, c, st or None)
        for t, c, st in re.findall(
            r'\(\s*"([\w-]+)"\s*,\s*"([\w-]+)"\s*,\s*(?:"([\w-]+)"|None)\s*\)', block
        )
    }

    call_re = re.compile(r'(?:load_prompt|get_cached)\(\s*"([\w-]+)"\s*,\s*"([\w-]+)"(?:\s*,\s*stage\s*=\s*"([\w-]+)")?')
    called = set()
    for path, src in _sources():
        if path == _REGISTRY:
            continue
        for m in call_re.finditer(src):
            called.add((m.group(1), m.group(2), m.group(3)))

    undeclared = sorted(called - declared)
    assert undeclared == [], (
        "these prompt keys are loaded by code but missing from REQUIRED_PROMPTS, so /health "
        "would report every prompt present while these could be absent: "
        + "; ".join(f"{t}/{st or '-'}/{c}" for t, c, st in undeclared)
    )

    # Keys loaded with a VARIABLE category, which a literal scan cannot see. The six catalog
    # legend prompts go through `load_prompt("extraction", CATEGORY_BY_TYPE[legend_type],
    # stage="discovery")`. They are genuinely required — /health must check them — so they are
    # declared and excluded from the stale check rather than deleted.
    #
    # SHRINK-ONLY. An entry here is a claim that some call site loads this key dynamically; if
    # that call site goes, the entry must go with it.
    dynamically_keyed = {
        ("extraction", "legend_care", "discovery"),
        ("extraction", "legend_certifications", "discovery"),
        ("extraction", "legend_icons", "discovery"),
        ("extraction", "legend_installation", "discovery"),
        ("extraction", "legend_regulations", "discovery"),
        ("extraction", "legend_sustainability", "discovery"),
    }
    legend_src = (_APP / "services/knowledge/catalog_legend_extractor_v2.py").read_text(
        encoding="utf-8")
    assert "CATEGORY_BY_TYPE" in legend_src and "load_prompt(" in legend_src, (
        "the legend extractor no longer loads its prompts dynamically — the "
        "`dynamically_keyed` exemptions above are stale and must be removed."
    )

    stale = sorted(declared - called - dynamically_keyed)
    assert stale == [], (
        "REQUIRED_PROMPTS lists keys nothing loads any more — /health would fail a deploy over "
        "a prompt no code needs: "
        + "; ".join(f"{t}/{st or '-'}/{c}" for t, c, st in stale)
    )
