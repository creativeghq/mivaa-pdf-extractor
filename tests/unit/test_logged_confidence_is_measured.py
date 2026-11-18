"""
Guard: a confidence written to the AI-call logs is MEASURED, or it is absent.

`extraction.confidence_uninformative` found ~30 operation types logging only 0.9/0.95
on success and 0.0 on failure: helpers defaulted to a constant and call sites passed
literals. A threshold on that is a constant; `action`/`error` already record failure.
"""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"

LOGGING_CALLS = {
    "log_ai_call",
    "log_claude_call",
    "log_replicate_call",
    "log_time_based_call",
    "tracked_claude_call_async",
    "tracked_claude_call",
    "tracked_claude_call_sync",
}

DEFAULT_HOLDERS = {
    "services/core/claude_helper.py",
    "services/core/claude_tool_call.py",
}


def _is_literal_number(node: ast.AST) -> bool:
    """A number, or arithmetic / a conditional built only from numbers."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.BinOp):
        return _is_literal_number(node.left) and _is_literal_number(node.right)
    if isinstance(node, ast.UnaryOp):
        return _is_literal_number(node.operand)
    if isinstance(node, ast.IfExp):
        return _is_literal_number(node.body) and _is_literal_number(node.orelse)
    return False


def _call_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _trees():
    for path in APP.rglob("*.py"):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_no_logging_call_is_handed_a_constant_confidence():
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in LOGGING_CALLS:
                continue
            for kw in node.keywords:
                if kw.arg == "confidence_score" and _is_literal_number(kw.value):
                    offenders.append(f"{path.relative_to(APP)}:{node.lineno} {_call_name(node)}")
    assert not offenders, (
        "Pass a measured confidence (or a callable reading the reply) or None — "
        "never a literal:\n" + "\n".join(offenders)
    )


def test_logger_parameters_default_to_no_confidence():
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in LOGGING_CALLS:
                continue
            args = node.args.args + node.args.kwonlyargs
            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults)
            defaults += list(node.args.kw_defaults)
            for arg, default in zip(args, defaults):
                if arg.arg == "confidence_score" and default is not None and _is_literal_number(default):
                    offenders.append(f"{path.relative_to(APP)}:{node.lineno} {node.name}")
    assert not offenders, "confidence_score must default to None:\n" + "\n".join(offenders)


def test_the_unmeasured_default_is_none():
    for rel in DEFAULT_HOLDERS:
        tree = ast.parse((APP / rel).read_text(encoding="utf-8"))
        values = [
            n.value for n in ast.walk(tree)
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            and any(isinstance(t, ast.Name) and t.id == "_DEFAULT_CONFIDENCE"
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))
        ]
        assert values, f"{rel} no longer declares _DEFAULT_CONFIDENCE"
        for v in values:
            assert isinstance(v, ast.Constant) and v.value is None, (
                f"{rel}: _DEFAULT_CONFIDENCE must be None — an unmeasured confidence is absent"
            )
