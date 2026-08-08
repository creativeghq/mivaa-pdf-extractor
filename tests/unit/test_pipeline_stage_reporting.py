"""
Guards for pipeline stage reporting and for the lint gate that protects it.

Six internal pipeline endpoints opened with `tracker = JobTracker(job_id)`.
`JobTracker` stopped existing on 2025-11-22, when commit 5ee4538 fixed a
crash-on-boot by swapping the IMPORT to `ProgressTracker` and left every call site
alone. The result was worse than the crash it replaced: the module imported fine and
all six endpoints raised NameError on their first line, were swallowed by
`except Exception`, and answered 500 having done nothing — for eight months.

No test could have caught that, because nothing calls those endpoints. A linter
catches it in one second. That is why the gate guarded here exists.

Source-based on purpose: imports neither `app` nor a DB, so it runs in CI in ~1s.
"""

import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_INTERNAL_ROUTES = _ROOT / "app" / "api" / "internal_routes.py"
_GATE_SCRIPT = _ROOT / "scripts" / "check_ruff_gate.py"
_BASELINE = _ROOT / ".github" / "ruff-baseline.json"
_DEPLOY = _ROOT / ".github" / "workflows" / "deploy.yml"

_SOURCE = _INTERNAL_ROUTES.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _handler_region() -> str:
    """Everything after the router is declared — i.e. the endpoints, not the helper."""
    return _SOURCE[_SOURCE.index("router = APIRouter("):]


def test_jobtracker_is_gone_from_the_handlers():
    assert "JobTracker(" not in _handler_region(), (
        "JobTracker is being constructed again — it is defined nowhere in this repo "
        "and raises NameError on the first line of the endpoint"
    )


def test_every_stage_key_used_is_mapped():
    """A key absent from _PIPELINE_STAGES makes report_stage log and return, so the
    stage silently stops being reported while the endpoint still returns 200 — the
    same shape as the bug this replaced, one layer down."""
    mapped = set()
    for node in ast.walk(_TREE):
        # Annotated (`_PIPELINE_STAGES: Dict[...] = {...}`) or plain assignment.
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign)
            else node.targets if isinstance(node, ast.Assign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == "_PIPELINE_STAGES" for t in targets):
            if isinstance(node.value, ast.Dict):
                mapped = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    assert mapped, "_PIPELINE_STAGES not found"

    used = set()
    for node in ast.walk(_TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "report_stage"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        ):
            used.add(node.args[1].value)

    assert used, "no report_stage calls found — the handlers stopped reporting stages"
    assert used <= mapped, f"stage key(s) not in _PIPELINE_STAGES: {sorted(used - mapped)}"


def test_report_stage_never_fabricates_a_tracker():
    """The fallback must NOT construct a ProgressTracker. A fresh one carries
    total_pages=0 and zeroed counters, and `_sync_to_database` writes all of them back
    to background_jobs — overwriting real progress with a confident-looking zero."""
    for node in ast.walk(_TREE):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "report_stage":
            fn = node
            break
    else:
        raise AssertionError("report_stage() not found")

    # Compare CODE only. The docstring names the construction it forbids and the
    # comments explain why, so a raw substring check matches the explanation rather
    # than the behaviour — the same trap this file's siblings hit.
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    src = "\n".join(ast.unparse(n) for n in body)

    assert "ProgressTracker(" not in src, (
        "report_stage constructs a ProgressTracker; a fabricated tracker syncs "
        "total_pages=0 and zeroed counters over real progress"
    )
    assert "get_tracker(" in src, "the live in-process tracker is no longer preferred"
    assert "append_stage_history" in src, "the no-tracker path stopped recording anything"


def test_lint_gate_exists_and_holds_undefined_names_at_zero():
    assert _GATE_SCRIPT.exists(), "the ruff gate script is gone"
    gate = _GATE_SCRIPT.read_text(encoding="utf-8")
    assert '"F821"' in gate and "ZERO_TOLERANCE" in gate, (
        "F821 dropped out of the zero-tolerance set — it is the entire bug class "
        "(JobTracker, vecs_service, four missing imports) this gate was added for"
    )

    assert '"F811"' in gate, (
        "F811 dropped out of the zero-tolerance set — a redefinition silently "
        "discards the earlier binding; three of this repo's ten were live bugs"
    )

    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))["counts"]
    for rule in ("F821", "F811"):
        assert rule not in baseline, (
            f"{rule} was moved into the ratchet baseline. It is enforced at zero; "
            f"putting it in the baseline makes new occurrences landable again."
        )

    # The ratchet must not be quietly loosened either.
    assert baseline.get("F401", 0) <= 371 and baseline.get("F841", 0) <= 53, (
        "a baseline count was raised instead of ratcheted down — fix the findings"
    )


def test_lint_gate_is_wired_into_ci_with_a_pinned_ruff():
    """An unpinned ruff silently re-baselines the gate on the next release: counts
    move for reasons nobody introduced, the build fails, and the gate gets deleted
    rather than debugged."""
    deploy = _DEPLOY.read_text(encoding="utf-8")
    assert "check_ruff_gate.py" in deploy, "the lint gate is not wired into CI"
    assert "ruff==" in deploy, "ruff is installed unpinned in CI"
