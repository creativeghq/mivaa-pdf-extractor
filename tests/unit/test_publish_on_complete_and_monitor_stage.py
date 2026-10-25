"""publish_on_complete activates only a clean job's own drafts; the monitor knows the processing stage."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ROUTES = _ROOT / "app/api/rag_routes.py"
_MONITOR = _ROOT / "app/services/tracking/job_progress_monitor.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str):
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def _chain(node: ast.AST) -> str:
    return ast.unparse(node)


def test_publishing_touches_only_this_jobs_drafts_and_only_sets_active():
    body = _chain(_function(_tree(_ROUTES), "_publish_job_products"))
    assert ".update({'status': 'active'})" in body
    assert ".eq('source_job_id', job_id)" in body
    assert ".eq('status', 'draft')" in body


def test_the_publish_call_is_gated_on_the_flag_and_on_no_failed_product():
    fn = _function(_tree(_ROUTES), "process_document_with_discovery")
    gates = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and _chain(n.test) == "publish_on_complete"
    ]
    assert len(gates) == 1
    inner = gates[0].body[0]
    assert isinstance(inner, ast.If) and _chain(inner.test) == "products_failed == 0"
    assert "_publish_job_products" in _chain(inner)


def test_the_upload_publishes_by_default_but_a_resumed_or_older_job_never_does():
    tree = _tree(_ROUTES)
    upload = _function(tree, "upload_document")
    defaults = {a.arg: d for a, d in zip(upload.args.args[-len(upload.args.defaults):], upload.args.defaults)}
    assert _chain(defaults["publish_on_complete"]).startswith("Form(True")
    runner = _function(tree, "process_document_with_discovery")
    names = [a.arg for a in runner.args.args]
    default = runner.args.defaults[names.index("publish_on_complete") - (len(names) - len(runner.args.defaults))]
    assert _chain(default) == "False"
    resumed = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _chain(n.func) == "job_meta.get"
        and len(n.args) == 2
        and isinstance(n.args[0], ast.Constant) and n.args[0].value == "publish_on_complete"
        and isinstance(n.args[1], ast.Constant) and n.args[1].value is False
    ]
    assert resumed, "a resumed job must not publish unless it was started with the option"


def test_the_monitor_gets_a_processing_stage_with_its_own_timeout():
    assert '"product_processing"' in _MONITOR.read_text(encoding="utf-8")
    routes = _function(_tree(_ROUTES), "process_document_with_discovery")
    stages = [
        n.args[0].value for n in ast.walk(routes)
        if isinstance(n, ast.Call) and _chain(n.func) == "progress_monitor.update_stage"
        and n.args and isinstance(n.args[0], ast.Constant)
    ]
    assert stages.index("product_discovery") < stages.index("product_processing") < stages.index("field_propagation")
