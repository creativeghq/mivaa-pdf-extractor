"""Guards for the #31 M17-1 (agent-run helpers) fixes. The #28 prompt-editing routes were deleted."""

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

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
