"""A refresh that finds nothing still has to say when the next one is due."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MENTIONS = ROOT / "app" / "services" / "integrations" / "tracked_mentions_service.py"

CADENCE_RPC = "update_tracked_mention_cadence"


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
    return None


def _constants(node) -> set:
    return {
        n.value for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


@pytest.fixture(scope="module")
def src() -> str:
    return _strip_comments(_read(MENTIONS))


def test_the_cadence_update_lives_inside_the_stamp(src: str) -> None:
    stamp = _node(src, "_stamp_refresh")
    assert stamp is not None, "_stamp_refresh must exist"
    assert CADENCE_RPC in _constants(stamp), (
        "the cadence update must be part of stamping a refresh, not a separate step a "
        "caller can return before"
    )


def _rpc_call_lines(src: str, rpc_name: str) -> list:
    return [
        n.lineno for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "rpc"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == rpc_name
    ]


def test_no_other_site_advances_the_cadence(src: str) -> None:
    # Two call sites means one can be forgotten, which is exactly how the no-hits return
    # left next_check_at frozen and the subject billed on every tick.
    calls = _rpc_call_lines(src, CADENCE_RPC)
    assert len(calls) == 1, (
        f"{CADENCE_RPC} must be invoked from exactly one place (_stamp_refresh); "
        f"found calls on lines {calls}"
    )
    stamp = _node(src, "_stamp_refresh")
    assert stamp.lineno < calls[0] <= max(
        n.lineno for n in ast.walk(stamp) if hasattr(n, "lineno")
    ), "the one call site must be inside _stamp_refresh"


def test_every_stamp_declares_the_velocity(src: str) -> None:
    calls = [
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_stamp_refresh"
    ]
    assert calls, "_stamp_refresh must be called"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "velocity_pct" in kwargs, (
            f"_stamp_refresh call on line {call.lineno} omits velocity_pct — the cadence "
            "would fall back to 0 and read as a stable subject regardless of its volume"
        )


def test_a_failed_cadence_rpc_still_moves_next_check_at(src: str) -> None:
    fallback = _node(src, "_set_next_check_fallback")
    assert fallback is not None, "a fallback must exist"
    assert "next_check_at" in _constants(fallback), (
        "a subject left with a past next_check_at is charged on every cron tick, so the "
        "fallback must actually write the field"
    )
    stamp = _node(src, "_stamp_refresh")
    assert "_set_next_check_fallback" in {
        n.func.attr for n in ast.walk(stamp)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }, "_stamp_refresh must fall back when the cadence RPC fails"
