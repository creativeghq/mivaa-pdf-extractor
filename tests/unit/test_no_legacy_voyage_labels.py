"""
Guard: no code names a Voyage model as a literal outside the settings and the price table.

A literal model name in code is a provenance stamp that can lie: eleven stamped
"voyage-3" on voyage-4-large vectors, and "voyage-4" stayed on chunks after they moved to
voyage-context-4. Stamp the setting (`get_settings().voyage_*_model`) or the service's
`_last_provider` instead. OpenAPI example payloads are documentation and are skipped.
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_APP = Path(__file__).resolve().parents[2] / "app"

_VOYAGE_MODEL = re.compile(r"^voyage-[a-z0-9.\-]+$")
_DECLARATIONS = {_APP / "config.py", _APP / "config" / "ai_pricing.py"}
_EXAMPLE_NAMES = {"json_schema_extra", "schema_extra", "example", "examples"}


def _example_nodes(tree: ast.AST) -> set:
    skip = set()
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in _EXAMPLE_NAMES for t in node.targets):
            value = node.value
        elif isinstance(node, ast.keyword) and node.arg in _EXAMPLE_NAMES:
            value = node.value
        if value is not None:
            skip.update(id(n) for n in ast.walk(value))
    return skip


def offenders_in(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _example_nodes(tree)
    return [
        f"{path.relative_to(_APP.parent)}:{n.lineno} {n.value!r}"
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and _VOYAGE_MODEL.match(n.value) and id(n) not in skip
    ]


def test_no_voyage_model_literal_outside_declarations():
    offenders = [o for p in _APP.rglob("*.py") if p not in _DECLARATIONS for o in offenders_in(p)]
    assert not offenders, "Voyage model named as a literal:\n" + "\n".join(offenders)
