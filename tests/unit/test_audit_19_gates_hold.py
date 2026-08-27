"""Guards for the mivaa#19 audit fixes (M6-1, M6-6, M6-7).

One file per audit, matching `test_audit_18_gates_hold.py`.

Static, not runtime: CI installs pytest alone (`deploy.yml`) and these unit tests
import nothing from `app`, so each case parses source instead. That constrains what
can be checked — a guard here proves the SHAPE is gone, not that the replacement
behaves. It especially cannot check the DATA half of M6-1: whether the four
`prompt_type='search'` rows still carry the `subcategory` they were backfilled with
lives in the database, not in this repository. `test_the_subtype_filter_is_still_applied`
is the closest a source test can get, and it says so.

13 of the 19 cases were watched to FAIL against the pre-fix source. The other six are
stays-as-they-are guards and pass both ways, named here so they are not mistaken for
coverage: the four `test_each_method_still_reaches_its_llm_helper` cases (deleting the
branch is only correct while the call survives), `test_the_subtype_filter_is_still_applied`
(the WRONG repair for M6-1(b) would have been to drop the filter), and
`test_every_wrapper_is_declared_required[search_result_enrichment]` (that one wrapper
was already registered).

NOT covered here, deliberately: M6-2 (the scheduled-refetch SSRF), M6-3, M6-4 and
M6-5 are untouched by this batch.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

SERVICE = APP / "services" / "search" / "search_prompt_service.py"
RAG_ROUTES = APP / "api" / "rag_routes.py"
REGISTRY = APP / "services" / "utilities" / "prompt_registry.py"

PUBLIC_METHODS = ["enhance_query", "format_results", "filter_results", "enrich_results"]

#: The four wrappers this service loads. All four must be declared in
#: `REQUIRED_PROMPTS`, or a missing row is discovered at 2am instead of at deploy.
SEARCH_WRAPPERS = [
    "search_query_enhancement",
    "search_result_formatting",
    "search_result_filtering",
    "search_result_enrichment",
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Drop comments and docstrings so prose about the old bug is not read as code."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _source_of(src: str, name: str) -> str:
    return ast.get_source_segment(src, _node(src, name)) or ""


# ───────────────────────────────────────────────────────────────────────────
# M6-1 (a) — the attribute that was never assigned
# ───────────────────────────────────────────────────────────────────────────

def test_the_undefined_llm_client_attribute_is_gone():
    """`self.llm_client` was assigned nowhere on the class or anywhere in the file, so
    four `if self.llm_client:` branches raised AttributeError on the CONDITION. That
    put the `_simple_*` fallback out of reach too — the whole call fell into a broad
    handler that returned the unenhanced input."""
    body = _strip_comments(_read(SERVICE))
    assert "self.llm_client" not in body, (
        "`self.llm_client` is back in the code. Nothing assigns it, so every branch "
        "testing it raises (audit #19 M6-1)."
    )


@pytest.mark.parametrize("method", PUBLIC_METHODS)
def test_each_method_still_reaches_its_llm_helper(method: str):
    """Deleting the branch is only correct because each `_apply_llm_*` falls back to
    its `_simple_*` twin in its own handler. If the call disappears, the feature is
    dead again — just more quietly than before."""
    body = _strip_comments(_source_of(_read(SERVICE), method))
    assert "_apply_llm_" in body, f"{method} no longer calls its LLM helper at all"


# ───────────────────────────────────────────────────────────────────────────
# M6-1 (b) — the lookup that matched zero of nine rows, silently
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", PUBLIC_METHODS)
def test_a_zero_match_lookup_says_so(method: str):
    """Returning early on an empty lookup is correct; doing it SILENTLY is how nine
    actively-maintained prompt rows went nine months without ever being read.

    `ops.silent_zero` cannot see this one — search enhancement produces no metric at
    all — so the log line is the only signal that exists."""
    body = _strip_comments(_source_of(_read(SERVICE), method))
    assert "_no_prompt_configured" in body, (
        f"{method} returns early on a zero-match prompt lookup without saying so. An "
        "admin's edit then changes nothing, forever, with every health signal green."
    )


def test_the_no_prompt_warning_is_a_warning():
    """INFO would be dropped by the DB log sink's noise filter; WARNING and above never
    is. The whole point is that this reaches `system_logs`."""
    body = _strip_comments(_source_of(_read(SERVICE), "_no_prompt_configured"))
    assert "logger.warning" in body, (
        "the zero-match notice dropped below WARNING — the DB log sink filters "
        "sub-WARNING records, so it would reach no queryable sink"
    )


def test_the_subtype_filter_is_still_applied():
    """The other repair for M6-1(b) would have been to DROP the subtype filter. That is
    wrong: nine rows carry `prompt_type='search'` and only four map to the four
    subtypes, so an unfiltered lookup would let `enhance_query` pick "RAG Document
    Analysis" as its enhancement prompt.

    The data half — those four rows carrying the backfilled `subcategory` — lives in the
    database and cannot be checked from here."""
    body = _strip_comments(_source_of(_read(SERVICE), "get_active_prompts"))
    assert "prompt_subtype" in body, (
        "get_active_prompts no longer passes the subtype through. Without it every "
        "method picks the first active `search` prompt regardless of what it is for."
    )


# ───────────────────────────────────────────────────────────────────────────
# M6-6 — the Claude calls dropped both ids
# ───────────────────────────────────────────────────────────────────────────

def test_every_claude_call_here_carries_both_ids():
    """The public methods receive a workspace, the private `_apply_llm_*` ones did not,
    and every `tracked_claude_call_async` omitted both — so the spend landed
    unattributed. Masked by M6-1 while the calls never executed, which is why the two
    had to be fixed together."""
    tree = ast.parse(_read(SERVICE))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "tracked_claude_call_async"
    ]
    assert len(calls) == 4, f"expected 4 Claude calls in this service, found {len(calls)}"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        task = next(
            (ast.literal_eval(kw.value) for kw in call.keywords
             if kw.arg == "task" and isinstance(kw.value, ast.Constant)),
            "<unnamed>",
        )
        assert "user_id" in kwargs, f"the {task} Claude call omits user_id"
        assert "workspace_id" in kwargs, f"the {task} Claude call omits workspace_id"


def test_the_service_accepts_the_ids_and_the_caller_supplies_them():
    """A service cannot invent a user id. The construction site in `rag_routes` has both
    in scope and now passes them."""
    init = _node(_read(SERVICE), "__init__")
    names = [a.arg for a in init.args.args] + [a.arg for a in init.args.kwonlyargs]
    assert "workspace_id" in names and "user_id" in names, (
        "SearchPromptService no longer accepts the ids, so its Claude calls have "
        "nothing to attribute the spend to"
    )

    rag = _strip_comments(_read(RAG_ROUTES))
    at = rag.find("SearchPromptService(")
    assert at != -1, "rag_routes no longer constructs SearchPromptService"
    window = rag[at:at + 400]
    assert "workspace_id=" in window and "user_id=" in window, (
        "rag_routes constructs SearchPromptService without passing the ids it holds"
    )


# ───────────────────────────────────────────────────────────────────────────
# M6-7 — prompts hardcoded in a file whose job is loading them from the DB
# ───────────────────────────────────────────────────────────────────────────

def test_no_instruction_text_is_built_in_code():
    """Three wrappers were assembled from f-strings in the one file whose entire purpose
    is loading admin-configured prompts from the database. The existing guard could not
    see them: it matches Anthropic-specific markers and a 2-of-9 keyword heuristic."""
    src = _strip_comments(_read(SERVICE))
    for banned in ("Return ONLY the JSON array", "Return ONLY the enhanced query text"):
        assert banned not in src, (
            f"instruction text {banned!r} is built in code again — it belongs in the "
            "prompts table, where an admin can change it without a deploy"
        )


def test_all_four_wrappers_load_from_the_registry():
    src = _strip_comments(_read(SERVICE))
    for category in SEARCH_WRAPPERS:
        assert f'"{category}"' in src, (
            f"the {category} wrapper is no longer loaded through load_prompt"
        )


@pytest.mark.parametrize("category", SEARCH_WRAPPERS)
def test_every_wrapper_is_declared_required(category: str):
    """No-fallback means a missing row stops the work cold. `check_required_prompts()`
    is what turns that into a deploy-time failure instead of a 2am one."""
    src = _read(REGISTRY)
    assert f'("extraction", "{category}", "search")' in src, (
        f"{category} is missing from REQUIRED_PROMPTS, so a missing row would only be "
        "discovered when a search runs"
    )
