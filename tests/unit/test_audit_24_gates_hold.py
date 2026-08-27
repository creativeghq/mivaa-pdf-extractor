"""Guards for the mivaa#24 audit fixes (M11-1 … M11-6), plus the #26 M13-1 half.

One file per audit, matching `test_audit_18_gates_hold.py`.

Static, not runtime: CI installs pytest alone (`deploy.yml`) and these unit tests
import nothing from `app`, so each case parses source instead. That constrains what
can be checked — a guard here proves the SHAPE is gone, not that the replacement
behaves.

Every case below was watched to FAIL against the pre-fix source before being
committed. A guard nobody has seen fire is a guard that might be asserting nothing.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

AGENTS = APP / "api" / "agent_routes.py"
LOGS = APP / "api" / "logs_routes.py"
LINKING = APP / "api" / "admin_linking.py"
CATALOG = APP / "api" / "catalog_routes.py"

#: Columns the two agent handlers referenced that do NOT exist on `products` —
#: confirmed against the live schema (#26 M13-1). They live in the `attributes` jsonb
#: now. PostgREST rejects the whole request on any one of them, which is why the
#: unscoped write in M11-2 was latent rather than live.
DROPPED_PRODUCT_COLUMNS = ("material_type", "tags", "image_url", "search_keywords")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Drop comments and docstrings so prose about the old bug is not read as code."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _source_of(src: str, name: str) -> str:
    return ast.get_source_segment(src, _node(src, name)) or ""


def _model_fields(src: str, name: str) -> list:
    """Declared attribute names of a pydantic model — AST, not text, so a docstring
    explaining why a field is absent is not read as the field being present."""
    return [
        stmt.target.id
        for stmt in _node(src, name).body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]


# ───────────────────────────────────────────────────────────────────────────
# M11-1 — the internal key guard failed OPEN when the key was unset
# ───────────────────────────────────────────────────────────────────────────

def test_the_agent_key_guard_fails_closed():
    """`if expected_key and authorization != ...` short-circuits to no check at all
    when MIVAA_API_KEY is unset, on the route that spends AI credits and writes to
    `products`. The correct fail-closed form was already in two sibling files."""
    src = _strip_comments(_read(AGENTS))
    assert "_require_internal_key" in src, "the fail-closed key guard is gone"

    body = _strip_comments(_source_of(_read(AGENTS), "_require_internal_key"))
    assert re.search(r"if not expected_key", body), (
        "the guard no longer rejects an UNSET key. That is the whole bug: a missing "
        "env var must not become an open door."
    )
    assert not re.search(r"if\s+expected_key\s+and\b", body), (
        "the `if expected_key and ...` short-circuit is back — an unset key runs no "
        "check at all (audit #24 M11-1)"
    )


def test_the_run_route_still_calls_the_guard():
    """A correct guard nothing invokes is the same as no guard."""
    body = _strip_comments(_source_of(_read(AGENTS), "run_agent"))
    assert "_require_internal_key(" in body, (
        "POST /api/agents/run no longer checks the internal key"
    )


# ───────────────────────────────────────────────────────────────────────────
# M11-2 + #26 M13-1 — a global product mutation behind a stale-column error
# ───────────────────────────────────────────────────────────────────────────

def test_every_products_chain_in_the_agent_handlers_is_workspace_scoped():
    """Both handlers selected the first `batch_size` products PLATFORM-WIDE and
    updated them by product id alone, so one tenant's agent run could rewrite
    another tenant's gold layer."""
    src = _strip_comments(_read(AGENTS))
    chains = re.findall(r'table\("products"\)(.*?)\.execute\(\)', src, re.DOTALL)
    assert chains, "the agent handlers no longer touch products — have they moved?"
    for chain in chains:
        assert '.eq("workspace_id"' in chain, (
            f"a products chain in agent_routes carries no workspace predicate:\n{chain.strip()[:300]}"
        )


@pytest.mark.parametrize("column", DROPPED_PRODUCT_COLUMNS)
def test_no_dropped_product_column_is_used_as_a_column(column: str):
    """These are facet keys inside `attributes` jsonb, not columns. Naming one in a
    select, a filter or an update dict makes PostgREST reject the whole request.

    Prompt text and the facet allowlists legitimately contain the same words, so this
    checks the places that are COLUMN positions, not every occurrence."""
    src = _strip_comments(_read(AGENTS))

    for sel in re.findall(r'\.select\((["\'])(.*?)\1', src, re.DOTALL):
        assert column not in sel[1], (
            f'`{column}` appears in a .select(...) column list: "{sel[1][:120]}" — it is '
            "not a column on products (#26 M13-1)"
        )
    for filt in re.findall(r'\.(?:eq|is_|not_\.is_|or_)\((["\'])(.*?)\1', src, re.DOTALL):
        assert column not in filt[1], (
            f'`{column}` is used as a PostgREST filter target: "{filt[1][:120]}"'
        )
    assert not re.search(rf'update\["{column}"\]', src), (
        f"`{column}` is assigned into a products update dict"
    )


def test_the_workspace_is_derived_from_stored_rows_not_the_request_body():
    """Invariant 1. `AgentRunRequest` has no `workspace_id` field on purpose: there is
    then nothing for a handler to trust. An unattributed run must fail, not widen to
    every workspace."""
    src = _read(AGENTS)
    assert "workspace_id" not in _model_fields(src, "AgentRunRequest"), (
        "AgentRunRequest grew a workspace_id field — a body-supplied workspace is a "
        "claim, not a fact (invariant 1)"
    )

    resolver = _strip_comments(_source_of(src, "_resolve_run_workspace"))
    assert 'table("agent_runs")' in resolver, (
        "_resolve_run_workspace no longer reads the run row"
    )
    assert "raise ValueError" in resolver, (
        "_resolve_run_workspace no longer fails on a run with no workspace. Falling "
        "through means running platform-wide."
    )


@pytest.mark.parametrize("handler", ["handle_product_enrichment", "handle_material_tagger"])
def test_both_handlers_take_a_workspace(handler: str):
    node = _node(_read(AGENTS), handler)
    names = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
    assert "workspace_id" in names, f"{handler} no longer takes workspace_id"


def test_a_degraded_canonicalization_does_not_blank_the_facets():
    """`CanonicalizedAttributes.status == 'degraded'` means the canonical map is not
    trustworthy, not that the product has no facets. Writing it anyway produces a
    product byte-identical to one with nothing to canonicalize — the silent-zero shape
    the flag exists to prevent."""
    src = _strip_comments(_source_of(_read(AGENTS), "_canonicalize_into_attributes"))
    assert '"degraded"' in src or "'degraded'" in src, (
        "the degraded status is no longer checked before writing attributes"
    )
    assert "return None" in src, "the degraded path no longer refuses to write"


def test_existing_attributes_are_merged_not_replaced():
    """`CanonicalizedAttributes.attributes` is built from THIS run's resolutions alone.
    Writing it wholesale erases every facet the product already had."""
    src = _strip_comments(_source_of(_read(AGENTS), "_canonicalize_into_attributes"))
    assert 'product.get("attributes")' in src, (
        "the merge no longer reads the product's existing attributes"
    )


# ───────────────────────────────────────────────────────────────────────────
# M11-3 — agent runs were readable by id alone
# ───────────────────────────────────────────────────────────────────────────

def test_run_status_is_gated_and_tenant_scoped():
    """A valid run UUID returned another tenant's agent inputs, outputs, errors and
    model metadata. `agent_runs` carries `workspace_id`; the predicate was simply
    absent."""
    body = _strip_comments(_source_of(_read(AGENTS), "get_run_status"))
    assert "Depends(get_workspace_context)" in body, "get_run_status declares no gate"
    assert '.eq("workspace_id"' in body, (
        "get_run_status reads agent_runs by id alone again"
    )
    assert '.select("*")' not in body, (
        "select(\"*\") is back — project the columns the caller needs"
    )


# ───────────────────────────────────────────────────────────────────────────
# M11-4 — forged admin log entries attributed to another user
# ───────────────────────────────────────────────────────────────────────────

def test_the_frontend_log_route_cannot_be_told_who_the_user_is():
    """It accepted a body-supplied `user_id`, so any caller could write a CRITICAL
    entry attributed to somebody else. The route stays OPEN deliberately — the browser
    logger sends no Authorization header and swallows failures, so gating it would
    switch frontend error reporting off silently — but open is not the same as
    credulous."""
    assert "user_id" not in _model_fields(_read(LOGS), "FrontendLogRequest"), (
        "FrontendLogRequest accepts user_id from the body again (audit #24 M11-4)"
    )

    handler = _strip_comments(_source_of(_read(LOGS), "log_frontend_error"))
    assert "_attributed_user_id" in handler, (
        "the log entry no longer derives user_id from the token"
    )
    assert "log_request.user_id" not in handler


def test_the_frontend_log_level_is_an_enum():
    """`level` had no enum, so `level.upper()` wrote whatever was sent straight into
    `system_logs.level`."""
    model = _source_of(_read(LOGS), "FrontendLogRequest")
    assert "Literal[" in model and '"CRITICAL"' in model, (
        "FrontendLogRequest.level is no longer constrained to the known levels"
    )


@pytest.mark.parametrize("field", ["message", "logger_name", "url", "user_agent"])
def test_every_frontend_log_field_is_bounded(field: str):
    """Unbounded fields on the one unauthenticated write in this repository make it a
    write amplifier into a table whose only defence is a TTL."""
    model = _source_of(_read(LOGS), "FrontendLogRequest")
    line = next((ln for ln in model.splitlines() if ln.strip().startswith(f"{field}:")), None)
    assert line, f"FrontendLogRequest.{field} is gone"
    assert "max_length=" in line, f"FrontendLogRequest.{field} declares no maximum length"


def test_the_free_form_context_is_size_bounded():
    """`context` is an arbitrary dict, so its SIZE is the only thing that can be
    bounded — and it has to be, or it is the amplifier the other caps removed."""
    model = _source_of(_read(LOGS), "FrontendLogRequest")
    assert "MAX_CONTEXT_BYTES" in model, "the context size bound is gone"


# ───────────────────────────────────────────────────────────────────────────
# M11-5 — an admin-NAMED route with no admin check
# ───────────────────────────────────────────────────────────────────────────

def test_the_linking_route_requires_admin_and_a_workspace():
    """It mutates the silver-to-gold boundary and had no `require_admin`, no user
    dependency and no workspace predicate on any of its three reads."""
    src = _read(LINKING)
    assert "Depends(require_admin)" in src, (
        "/api/admin/linking/link-chunks-to-products is admin-named and admin-nothing-else again"
    )
    body = _strip_comments(_source_of(src, "link_chunks_to_products"))
    assert "Depends(get_workspace_context)" in body

    stripped = _strip_comments(src)
    for table in ("documents", "document_chunks", "products"):
        chains = re.findall(rf"table\('{table}'\)(.*?)\.execute\(\)", stripped, re.DOTALL)
        assert chains, f"{table} is no longer read here"
        for chain in chains:
            assert ".eq('workspace_id'" in chain, (
                f"the {table} read in admin_linking carries no workspace predicate:\n{chain.strip()[:200]}"
            )


# ───────────────────────────────────────────────────────────────────────────
# M11-6 — nothing bounded the raster before rendering it
# ───────────────────────────────────────────────────────────────────────────

def test_the_render_is_bounded_before_get_pixmap_allocates():
    """`page_no` and `dpi` were validated; page dimensions were not. A small
    compressed PDF can declare enormous ones, and the render size is page_rect x
    dpi/72 — so the check has to happen BEFORE `get_pixmap`, not after."""
    src = _strip_comments(_read(CATALOG))
    assert "_assert_renderable" in src, "the pre-render size bound is gone"
    guard_at = src.index("_assert_renderable(page")
    pixmap_at = src.index("get_pixmap(")
    assert guard_at < pixmap_at, (
        "the size check now runs AFTER get_pixmap — the allocation it is supposed to "
        "prevent has already happened"
    )
    assert "MAX_RASTER_PIXELS" in src and "MAX_RASTER_EDGE" in src
