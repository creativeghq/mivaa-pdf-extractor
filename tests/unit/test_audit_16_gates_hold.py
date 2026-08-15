"""Guards for the mivaa#16 audit fixes (M3-2 … M3-15).

One file per audit, matching `test_audit_12_gates_hold.py`. M3-1 has its own
file because its guard is substantial; everything else lives here.

Static, not runtime: CI installs pytest alone (`deploy.yml`) and these unit
tests import nothing from `app`, so each case parses source instead. That
constrains what can be checked - a guard here proves the SHAPE is gone, not
that the replacement behaves. Where a case can only assert an absence, it says
so rather than implying more.

Every case below was watched to FAIL against the pre-fix source before being
committed. A guard nobody has seen fire is a guard that might be asserting
nothing.
"""

import ast
import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / "app"

CLIENT = APP / "services" / "core" / "supabase_client.py"
LOGGER = APP / "services" / "core" / "ai_call_logger.py"
PRODUCTS = APP / "services" / "products" / "product_creation_service.py"
JOBS = APP / "services" / "integrations" / "job_search_service.py"
UNIFIED = APP / "services" / "search" / "unified_search_service.py"
VISUAL = APP / "services" / "search" / "material_visual_search_service.py"
CONFIG = APP / "config.py"

#: The platform workspace. It may appear exactly once, as the settings default.
PLATFORM_WS = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_read(path))


def _code_only(source: str) -> str:
    """Source with comments and string literals blanked out.

    A guard that greps raw text trips on the comment EXPLAINING the thing it
    forbids - which is exactly what happened when this file first ran: the
    removal note for the JSON salvage regex quoted the regex. Checking tokens
    instead keeps the guard about code.
    """
    import io
    import tokenize

    out = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok_type, tok_str, _, _, _ in tokens:
            if tok_type in (tokenize.COMMENT, tokenize.STRING):
                # Keep the line count, drop the content.
                out.append("\n" * tok_str.count("\n"))
            else:
                out.append(tok_str)
    except tokenize.TokenError:  # pragma: no cover - defensive
        return source
    return " ".join(out)


def _body_code(path: Path, name: str) -> str:
    """A function's source with its docstring and comments removed."""
    func = _function(path, name)
    segment = ast.get_source_segment(_read(path), func) or ""
    # Dedent so tokenize accepts it standalone.
    import textwrap

    return _code_only(textwrap.dedent(segment))


def _function(path: Path, name: str):
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path.name} - re-point this guard")


def _args(func) -> list:
    return [a.arg for a in func.args.args + func.args.kwonlyargs]


def _required_args(func) -> set:
    """Positional args with no default. kwonly handled separately."""
    positional = [a.arg for a in func.args.args]
    defaults = len(func.args.defaults)
    required = set(positional[: len(positional) - defaults] if defaults else positional)
    for arg, default in zip(func.args.kwonlyargs, func.args.kw_defaults):
        if default is None:
            required.add(arg.arg)
    return required


# ══════════════════════════════════════════════ M3-2 — the debit was SKIPPED


def test_no_ai_wrapper_skips_the_debit_without_recording_it():
    """`if user_id:` around the debit, with no else, was the whole defect.

    The UNBILLED markers only fire when the debit RAISES or FAILS, so with no
    principal nothing was recorded at all - unbilled revenue read as zero
    because the recording path was unreachable. Every branch that guards on
    user_id must now have an else that says something.
    """
    tree = _tree(LOGGER)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Name) and node.test.id == "user_id"):
            continue
        calls_debit = any(
            isinstance(n, ast.Attribute) and n.attr == "_debit"
            for n in ast.walk(node)
        )
        if calls_debit and not node.orelse:
            offenders.append(node.lineno)
    assert not offenders, (
        f"debit guarded by `if user_id:` with no else at line(s) {offenders} - "
        "a call with no principal must be RECORDED as unbilled, not skipped"
    )


def test_the_unbilled_reason_reaches_the_row():
    """Logging UNBILLED to stdout is not measuring it. It has to be persisted."""
    src = _read(LOGGER)
    assert '"unbilled_reason": unbilled_reason' in src, (
        "the ai_call_logs insert no longer carries unbilled_reason - without it "
        '"how much is the preflight absorbing" is a grep, not a query'
    )
    assert "_record_missing_principal" in src, (
        "the no-principal case must record a reason, not fall through"
    )


# ══════════════════════════════════ M3-3 — the primary log had no attribution


def test_ai_call_logs_insert_carries_user_and_workspace():
    """Scoped to log_ai_call, NOT the whole file.

    The first draft grepped the module and passed against the pre-fix source,
    because `user_id` / `workspace_id` were already present — in the
    ai_usage_logs MIRROR insert, which is precisely the distinction M3-3 is
    about. A guard that the bug passes is worse than no guard.
    """
    func = _function(LOGGER, "log_ai_call")
    entry = None
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            keys = {
                k.value for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            if "task" in keys and "model" in keys:
                entry = keys
                break
    assert entry is not None, (
        "could not find the ai_call_logs row literal in log_ai_call - re-point "
        "this guard"
    )
    for field in ("user_id", "workspace_id"):
        assert field in entry, (
            f"{field} missing from the ai_call_logs row - attribution existed "
            "only in the ai_usage_logs mirror, so anything grouping by tenant in "
            "the table actually named for AI calls found nothing. "
            f"Row keys: {sorted(entry)}"
        )


# ═══════════════════════════════════════ M3-4 — booting on the anon key


def test_the_service_role_key_is_required_not_preferred():
    src = _read(CLIENT)
    assert "settings.supabase_service_role_key or settings.supabase_anon_key" not in src, (
        "the client falls back to the anon key again - MIVAA has no RLS "
        "backstop, so that turns a missing env var into an empty platform "
        "rather than a refusal to start"
    )
    assert "SUPABASE_SERVICE_ROLE_KEY is required" in src, (
        "no explicit assertion that the service-role key is present"
    )


# ════════════════════════ M3-6 — helpers collapsed failure into emptiness


@pytest.mark.parametrize(
    "name",
    ["list_documents", "get_document_by_id", "get_document_images"],
)
def test_read_helpers_raise_rather_than_return_empty_on_failure(name):
    """Emptiness must stay unambiguous: empty means empty, never "it broke"."""
    func = _function(CLIENT, name)
    handlers = [n for n in ast.walk(func) if isinstance(n, ast.ExceptHandler)]
    assert handlers, f"{name} has no exception handler - re-point this guard"
    for handler in handlers:
        returns = [n for n in ast.walk(handler) if isinstance(n, ast.Return)]
        assert not returns, (
            f"{name} returns from an except handler, so a failed query is "
            "indistinguishable from a query that matched nothing"
        )
        raises = [n for n in ast.walk(handler) if isinstance(n, ast.Raise)]
        assert raises, f"{name}'s handler neither raises nor returns"


def test_the_knowledge_base_save_stops_when_the_parent_lookup_fails():
    src = _read(CLIENT)
    assert '⚠️ Document creation failed' not in src, (
        "the document lookup failure is a warning again - everything after it "
        "writes CHILD rows keyed on that document, which then land with no "
        "tenant at all and are counted as saved"
    )


# ══════════════════════════════ M3-5/M3-7 — two ids, never checked together


def test_product_creation_derives_the_workspace_from_the_document():
    src = _read(PRODUCTS)
    assert "_resolve_document_workspace" in src, (
        "products are written with a caller-supplied workspace again, with "
        "nothing reconciling it against the document the content came from"
    )
    for entry in ("create_products_from_chunks", "create_products_from_layout_candidates"):
        func = _function(PRODUCTS, entry)
        calls = [
            n for n in ast.walk(func)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_resolve_document_workspace"
        ]
        assert calls, f"{entry} does not resolve the workspace from its document"


def test_a_tenancy_refusal_is_not_swallowed_into_a_fallback():
    """The broad handlers around both entry points must let it through.

    Absorbed into the chunk-path fallback or a soft {"success": false}, a
    tenancy refusal reads as an unlucky document and gets retried.
    """
    for entry in ("create_products_from_chunks", "create_products_from_layout_candidates"):
        func = _function(PRODUCTS, entry)
        reraises = [
            h for h in ast.walk(func)
            if isinstance(h, ast.ExceptHandler)
            and isinstance(h.type, ast.Name)
            and h.type.id == "TenancyViolation"
        ]
        assert reraises, f"{entry} has no `except TenancyViolation: raise`"


def test_save_single_image_requires_a_workspace():
    func = _function(CLIENT, "save_single_image")
    assert "workspace_id" in _required_args(func), (
        "workspace_id is optional on save_single_image again - an optional "
        "tenant on a write is how document_images rows ended up with NULL, "
        "invisible to every scoped read afterwards"
    )


# ══════════════════════════════════════════════ M3-8 — success in a finally


def test_the_rss_poll_reports_its_real_outcome():
    src = _read(JOBS)
    assert "success=True,\n                markup_multiplier=1.0," not in src, (
        "success=True is hardcoded in the RSS finally again - a blocked URL, a "
        "dead host and a malformed feed all log identically to a healthy feed "
        "with no new jobs"
    )
    assert "success=failure is None" in src, (
        "the RSS log row no longer derives success from what actually happened"
    )


# ════════════════════════════════════════ M3-9 — Firecrawl bypassed the guard


def test_every_firecrawl_target_goes_through_the_ssrf_guard():
    """Guarded at the chokepoint, so a new call site inherits it."""
    func = _function(JOBS, "firecrawl_scrape")
    calls = [
        n for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "assert_safe_url"
    ]
    assert calls, (
        "firecrawl_scrape does not SSRF-check its target - fetching THROUGH a "
        "third party relocates the primitive, it does not remove it"
    )


# ═══════════════════════════════════ M3-10 — DataForSEO 200 != success


def test_the_dataforseo_serp_call_checks_the_envelope():
    src = _read(JOBS)
    assert "_assert_dataforseo_ok" in src, (
        "the SERP path no longer validates the DataForSEO body - a "
        "provider-side failure comes back 200 OK and iterates to "
        "hits_returned=0, success=True"
    )
    func = _function(JOBS, "_assert_dataforseo_ok")
    raises = [n for n in ast.walk(func) if isinstance(n, ast.Raise)]
    assert len(raises) >= 4, (
        "the envelope check should reject on the envelope code, tasks_error, "
        "an empty task list AND per-task codes; found "
        f"{len(raises)} raise(s)"
    )


# ═════════════════════════════ M3-11 — free-form JSON + regex salvage


def test_both_product_verdicts_use_forced_tool_choice():
    src = _read(PRODUCTS)
    for tool in ("STAGE1_CLASSIFICATION_TOOL", "STAGE2_ENRICHMENT_TOOL"):
        assert f'tools=[{tool}]' in src, f"{tool} is not passed to the model"
        assert f'tool_choice={{"type": "tool", "name": {tool}["name"]}}' in src, (
            f"{tool} is offered but not FORCED - the model can still answer in prose"
        )


def test_the_regex_json_salvage_is_gone_from_the_product_path():
    src = _code_only(_read(PRODUCTS))
    assert r"(\{.*\}|\[.*\])" not in src, (
        "the JSON salvage regex is back in the product path - invariant 9 "
        "requires a forced tool_choice for a verdict that drives a DB write"
    )
    assert "_extract_json_from_response" not in src, (
        "the salvage helper is back; leaving it in the class invites the next "
        "verdict to reach for it"
    )


def test_a_failed_classifier_does_not_fall_back_to_keyword_heuristics():
    """The old fallback promoted EVERY chunk on a classifier outage."""
    src = _body_code(PRODUCTS, "_read_stage1_results")
    assert "_is_valid_product_chunk" not in src, (
        "the keyword fallback is back in the Stage 1 reader - a classifier "
        "outage would again silently promote the whole document to candidates"
    )


# ════════════════════════════ M3-12 — degraded strategies reported success


def test_every_degrading_strategy_records_that_it_degraded():
    """Returning [] on failure is fine; doing it invisibly is not."""
    src = _read(UNIFIED)
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("_search_"):
            continue
        for handler in [n for n in ast.walk(node) if isinstance(n, ast.ExceptHandler)]:
            returns_empty = any(
                isinstance(n, ast.Return)
                and isinstance(n.value, ast.List)
                and not n.value.elts
                for n in ast.walk(handler)
            )
            if not returns_empty:
                continue
            records = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "_record_degradation"
                for n in ast.walk(handler)
            )
            if not records:
                offenders.append(f"{node.name}:{handler.lineno}")
    assert not offenders, (
        f"strategies return [] on failure without recording it: {offenders}. "
        'search() then reports success=True, so "all strategies threw" and '
        '"nothing matched" become the same response.'
    )


def test_search_does_not_claim_success_when_everything_degraded():
    src = _read(UNIFIED)
    assert "success=not all_failed" in src, (
        "search() hardcodes success=True again"
    )
    assert '"degraded_strategies": degraded' in src, (
        "degraded_strategies is not surfaced, so a consumer cannot tell "
        '"matched nothing" from "could not look"'
    )


def test_the_visual_search_handler_does_not_report_an_empty_success():
    src = _read(VISUAL)
    assert "# Return empty results instead of mock data — but NOT as a success." in src, (
        "the visual-search exception handler is back to success=True with "
        "zero results"
    )


# ═══════════════════════════ M3-13 — unattributed query-understanding calls


def test_query_understanding_goes_through_the_tracked_helper():
    src = _read(UNIFIED)
    assert "https://api.anthropic.com" not in src, (
        "a raw Anthropic endpoint is back in the search path - a bare httpx "
        "POST has no debit and no cost log, so the call is free as far as the "
        "platform can tell"
    )
    assert src.count("tracked_claude_call_async") >= 2, (
        "both the primary and the fallback query parse must use the tracked "
        "helper (pipeline convention 10)"
    )


# ══════════════════════════════ M3-14 — the platform UUID, 11 times over


def test_the_platform_workspace_uuid_appears_exactly_once():
    hits = []
    for path in APP.rglob("*.py"):
        for i, line in enumerate(_read(path).splitlines(), 1):
            if PLATFORM_WS in line:
                hits.append(f"{path.relative_to(APP)}:{i}")
    assert hits == ["config.py:26"], (
        f"the platform workspace UUID appears at {hits}. It has a config home; "
        "copies can disagree with it, and one of them used to be a ROUTE "
        "DEFAULT - omitting workspace_id silently targeted that tenant. "
        "Expected exactly one: the settings default in config.py."
    )


def test_no_route_model_defaults_a_workspace_to_the_platform_tenant():
    src = _read(APP / "api" / "products.py")
    assert re.search(r"workspace_id:\s*Optional\[str\]\s*=\s*Field\(\s*\n\s*default=None", src), (
        "ProductCreationRequest.workspace_id has a default again - a request "
        "that simply omits the field must bind to the caller's own workspace, "
        "not to whatever tenant the default names"
    )


# ═══════════════════════════════════ M3-15 — silent zeros in extraction


def test_spec_extraction_counts_failures_separately_from_calls():
    src = _read(APP / "api" / "pdf_processing" / "stage_4_products.py")
    assert '"spec_vision_failures": 0,' in src, (
        "no spec_vision_failures counter - failures incremented "
        "spec_vision_calls alongside successes, so a tier throwing on every "
        "product looked like a tier that ran fine and found nothing"
    )
    assert 'stats["spec_vision_failures"] += 1' in src, (
        "the failures counter exists but is never incremented"
    )


def test_extractor_tiers_mark_failure_rather_than_returning_bare_empty():
    src = _read(APP / "services" / "products" / "product_spec_extractor_v2.py")
    assert src.count("_tier_error") >= 2, (
        "tier failures return a bare {} again, which is indistinguishable "
        "from a page that genuinely had no specs on it"
    )


# ═══════════════════════════ follow-ups found while closing the audit ════════


ENRICHMENT = APP / "services" / "search" / "search_enrichment_service.py"
SUGGESTIONS = APP / "services" / "search" / "search_suggestions_service.py"


def test_enrichment_joins_cannot_bridge_workspaces():
    """The association tables carry no tenant of their own.

    `image_product_associations` and `chunk_image_relationships` have no
    workspace_id column, so a correctly-scoped image can still point at
    another tenant's product or chunk. The predicate has to land on the
    embedded row, as an inner join.
    """
    func = _function(ENRICHMENT, "enrich_image_results")
    assert "workspace_id" in _required_args(func), (
        "enrich_image_results no longer requires a workspace_id"
    )

    src = _read(ENRICHMENT)
    for embed, table in (("products!inner", "get_related_products"),
                         ("document_chunks!inner", "get_related_chunks")):
        assert embed in src, (
            f"{table} no longer inner-joins its embedded row - a plain embed "
            "nulls the child instead of dropping the association"
        )
    for predicate in ("products.workspace_id", "document_chunks.workspace_id"):
        assert predicate in src, (
            f"no {predicate} filter - the association row carries no tenant, so "
            "this is the only place the check can land"
        )


def test_autocomplete_scopes_the_one_tenant_table_it_reads():
    """products is tenant data; the other three sources are not.

    search_suggestions and trending_searches are global reference data and
    search_analytics is keyed by user, so only the products lookup needs a
    workspace — but it had none, and the route bound none either, so
    autocomplete surfaced every tenant's product names.
    """
    func = _function(SUGGESTIONS, "_get_product_matches")
    assert "workspace_id" in _args(func), (
        "_get_product_matches takes no workspace_id"
    )
    src = _read(SUGGESTIONS)
    assert '.eq("workspace_id", workspace_id)' in src, (
        "the products lookup in autocomplete is unscoped again"
    )
    entry = _function(SUGGESTIONS, "get_autocomplete_suggestions")
    assert "workspace_id" in _required_args(entry), (
        "get_autocomplete_suggestions must require the workspace it passes down"
    )


def test_no_user_supplied_ilike_pattern_is_left_unescaped():
    """A `%` typed into a search box must not match the whole table.

    Sweeps the app rather than naming files, so a new ilike is covered by
    default. Interpolating anything into a like/ilike pattern without
    escape_like() fails here.
    """
    offenders = []
    for path in APP.rglob("*.py"):
        tree = ast.parse(_read(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr in ("ilike", "like")):
                continue
            if len(node.args) < 2:
                continue
            pattern = node.args[1]
            if isinstance(pattern, ast.Constant):
                continue  # a literal pattern has no user input in it
            interpolations = []
            if isinstance(pattern, ast.JoinedStr):
                interpolations = [
                    p.value for p in pattern.values if isinstance(p, ast.FormattedValue)
                ]
            else:
                interpolations = [pattern]
            for expr in interpolations:
                escaped = (
                    isinstance(expr, ast.Call)
                    and isinstance(expr.func, ast.Name)
                    and expr.func.id == "escape_like"
                ) or (isinstance(expr, ast.Name) and "escaped" in expr.id.lower()) or (
                    isinstance(expr, ast.Name) and expr.id == "like_pattern"
                )
                if not escaped:
                    offenders.append(
                        f"{path.relative_to(APP)}:{node.lineno}"
                    )
    assert not offenders, (
        f"unescaped user input in a like/ilike pattern at: {sorted(set(offenders))}. "
        "Wrap the term in escape_like() from app.utils.postgrest_filters."
    )
