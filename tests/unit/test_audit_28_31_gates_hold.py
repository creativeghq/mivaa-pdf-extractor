"""Guards for the mivaa#28 (prompt registry) and #31 M17-1 (agent-run helpers) fixes.

Both audits found the same shape, which is why they are held in one file: **the route
did the security work and the layer below discarded it.** Anyone reading the route would
conclude the endpoint was safe.

    #28 M14-2   the route calls `resolve_workspace_id`, says so in a comment, and passes
                the result to a service method that ACCEPTS the parameter and never
                applies it — so the endpoint returned every active prompt on the platform.
    #31 M17-1   `create_background_agent` records `workspace_id` correctly, and every
                update path then filters on id alone.

Static analysis, so these pin the SHAPE. 12 of the 14 cases were watched to fail against
the pre-fix source; the other two are stays-as-they-are guards that pass both ways, named
so they are not mistaken for coverage:
`test_the_prompt_list_still_shows_platform_defaults` (a bare `.eq()` here would be a
different bug — a correct filter that hides half the data) and
`test_the_routers_are_still_included_without_a_blanket_dependency` (so the per-route
gates are not first made redundant and then removed).

NOT covered: #28 M14-5 onward and #31 M17-2/M17-3 are untouched by this batch.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

PROMPT_SERVICE = APP / "services" / "utilities" / "admin_prompt_service.py"
PROMPT_ROUTES = APP / "api" / "admin_prompts.py"
TEMPLATE_ROUTES = APP / "api" / "prompt_templates.py"
AGENT_RUNS = APP / "services" / "integrations" / "job_agent_runs.py"
JOB_SERVICE = APP / "services" / "integrations" / "job_research_service.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
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
# #28 M14-1 — editing a prompt raised before the audit AND before the update
# ───────────────────────────────────────────────────────────────────────────

def test_the_audit_entry_reads_the_column_that_exists():
    """`prompt_template` is not a column on `prompts`; the text is in `prompt_text`,
    which the update twelve lines below already used. Bracket access, so it RAISED —
    and it sat before both the audit entry and the update, so editing an existing
    prompt had never worked and left no audit trail precisely because the audit call is
    what raised."""
    body = _strip_comments(_source_of(_read(PROMPT_SERVICE), "update_prompt"))
    assert "current['prompt_template']" not in body, (
        "the audit entry reads `prompt_template` again — that column does not exist, and "
        "this is bracket access so it raises (#28 M14-1)"
    )
    assert "old_prompt=current.get('prompt_text')" in body, (
        "the audit entry no longer reads the prompt text from `prompt_text`"
    )


# ───────────────────────────────────────────────────────────────────────────
# #28 M14-2 / M14-3 — the workspace the route derived must survive the handoff
# ───────────────────────────────────────────────────────────────────────────

def test_the_prompt_list_applies_the_workspace_it_is_given():
    """The parameter was accepted and ignored. `select('*')`, service role, no RLS
    backstop — every active prompt row on the platform."""
    body = _strip_comments(_source_of(_read(PROMPT_SERVICE), "get_prompts"))
    assert "workspace_scope(workspace_id)" in body, (
        "get_prompts no longer filters by workspace, so it returns every tenant's "
        "prompts to whoever asks (#28 M14-2)"
    )


def test_the_prompt_list_still_shows_platform_defaults():
    """`workspace_scope` rather than a bare equality, on purpose: platform defaults live
    in the global workspace and an admin is meant to see them beside their own. A plain
    `.eq()` here would be a different bug — a correct filter that hides half the data."""
    body = _strip_comments(_source_of(_read(PROMPT_SERVICE), "get_prompts"))
    assert ".eq('workspace_id'" not in body, (
        "the filter narrowed to the tenant's own rows, so platform default prompts "
        "vanished from the admin list"
    )


def test_prompt_history_is_resolved_inside_a_workspace():
    """History shows every revision, so it discloses MORE than the current text, and it
    was readable by prompt id alone. `prompt_history` has no workspace of its own, so
    the parent prompt is what gets resolved."""
    node = _node(_read(PROMPT_SERVICE), "get_prompt_history")
    names = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
    assert "workspace_id" in names, "get_prompt_history takes no workspace again"

    body = _strip_comments(_source_of(_read(PROMPT_SERVICE), "get_prompt_history"))
    assert "table('prompts')" in body, (
        "the parent prompt is no longer resolved, so the history is keyed on an "
        "unverified id (#28 M14-3)"
    )
    assert "return []" in body, (
        "an id outside the workspace no longer reads as empty"
    )


def test_the_history_route_binds_the_workspace_to_the_token():
    body = _strip_comments(_source_of(_read(PROMPT_ROUTES), "get_prompt_history"))
    assert "resolve_workspace_id(" in body, (
        "the history route accepts a workspace without binding it to the caller "
        "(invariant 1)"
    )
    assert "workspace_id=workspace_id" in body, (
        "the route resolves a workspace and then does not pass it down — which is the "
        "exact defect M14-2 was"
    )


# ───────────────────────────────────────────────────────────────────────────
# #28 M14-4 — these routes edit the text driving every model call
# ───────────────────────────────────────────────────────────────────────────

WRITE_ROUTES = [
    (PROMPT_ROUTES, "update_prompt"),
    (TEMPLATE_ROUTES, "create_prompt_template"),
    (TEMPLATE_ROUTES, "update_prompt_template"),
    (TEMPLATE_ROUTES, "delete_prompt_template"),
]


@pytest.mark.parametrize(("path", "func"), WRITE_ROUTES, ids=lambda v: getattr(v, "name", v))
def test_prompt_writes_require_admin(path: Path, func: str):
    """They gated on `get_current_user` alone, and `main.py` includes both routers with
    no dependency — so any authenticated workspace member could rewrite the prompts that
    drive every model call on the platform."""
    src = _read(path)
    node = _node(src, func)
    decorators = " ".join(ast.get_source_segment(src, d) or "" for d in node.decorator_list)
    assert "require_admin" in decorators, (
        f"{path.name}::{func} no longer requires admin (#28 M14-4)"
    )


def test_the_routers_are_still_included_without_a_blanket_dependency():
    """Recorded so the per-route gates above are not silently made redundant AND then
    removed. If a blanket dependency is ever added at inclusion, this fails and the
    decision gets made deliberately rather than by two half-changes."""
    main = _strip_comments(_read(APP / "main.py"))
    for line in main.splitlines():
        if "include_router(admin_prompts_router" in line or "include_router(prompt_templates_router" in line:
            assert "dependencies" not in line, (
                "a router-level dependency appeared; fold the per-route require_admin "
                "into it deliberately rather than leaving both"
            )


# ───────────────────────────────────────────────────────────────────────────
# #31 M17-1 — mutating live rows by bare id
# ───────────────────────────────────────────────────────────────────────────

def test_toggling_an_agent_requires_a_workspace_and_refuses_without_one():
    """Disabling is the worst mutation here because it is SILENT: a background agent
    that stops running looks exactly like one with nothing to do."""
    body = _strip_comments(_source_of(_read(AGENT_RUNS), "_set_agent_enabled"))
    assert ".eq(\"workspace_id\", workspace_id)" in body, (
        "the enabled toggle filters on id alone again (#31 M17-1)"
    )
    assert "if not workspace_id:" in body and "return" in body, (
        "a missing workspace no longer refuses — an unscoped toggle reaches every "
        "tenant's agents"
    )


def test_every_background_agent_update_carries_a_workspace():
    """Derived from the file, not a list: a NEW mutator here fails on the day it is
    written."""
    src = _strip_comments(_read(AGENT_RUNS))
    chains = re.findall(r'table\("background_agents"\)\s*\n?\s*\.update\((.*?)\.execute\(\)', src, re.DOTALL)
    assert chains, "background_agents is no longer updated here — has this moved?"
    for chain in chains:
        assert '.eq("workspace_id"' in chain, (
            f"a background_agents update carries no workspace predicate:\n{chain.strip()[:240]}"
        )


def test_the_agent_behind_a_run_is_checked_against_that_run_s_workspace():
    """Two ids that are never checked against each other is the shape this audit keeps
    finding: `agent_id` came off the run row and the update then trusted it alone."""
    for fn in ("complete_run", "fail_run"):
        body = _strip_comments(_source_of(_read(AGENT_RUNS), fn))
        assert '"agent_id, workspace_id"' in body, (
            f"{fn} no longer reads the run's own workspace before updating its agent"
        )
        assert 'bg_id.get("workspace_id")' in body, (
            f"{fn} updates the agent without requiring the run to have a workspace"
        )


def test_the_caller_supplies_the_workspace_it_already_holds():
    """A required-in-spirit parameter that nobody passes is a fix that never runs."""
    body = _strip_comments(_read(JOB_SERVICE))
    at = body.find("deactivate_background_agent(")
    assert at != -1, "job_research_service no longer deactivates the mirrored agent"
    assert "workspace_id" in body[at:at + 240], (
        "deactivate is called without the workspace, so the helper refuses and the "
        "agent silently stays enabled"
    )
