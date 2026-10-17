"""
Guard: no code names a Voyage model as a literal outside the modules that DECLARE models.

A literal model name in code is a provenance stamp that can lie: eleven stamped
"voyage-3" on voyage-4-large vectors, and "voyage-4" stayed on chunks after they moved to
voyage-context-4. Stamp the setting (`get_settings().voyage_*_model`) or the service's
`_last_provider` instead.
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_APP = Path(__file__).resolve().parents[2] / "app"

_VOYAGE_MODEL = re.compile(r"^voyage-[a-z0-9.\-]+$")

#: Where a model name is a declaration, not a stamp: settings, prices, OpenAPI examples.
_DECLARATIONS = (_APP / "config.py", _APP / "config" / "ai_pricing.py", _APP / "schemas", _APP / "models")


def _declares(path: Path) -> bool:
    return any(path == d or d in path.parents for d in _DECLARATIONS)


def test_no_voyage_model_literal_outside_declarations():
    offenders = []
    for path in _APP.rglob("*.py"):
        if _declares(path):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and _VOYAGE_MODEL.match(node.value):
                offenders.append(f"{path.relative_to(_APP.parent)}:{node.lineno} {node.value!r}")
    assert not offenders, "Voyage model named as a literal:\n" + "\n".join(offenders)
