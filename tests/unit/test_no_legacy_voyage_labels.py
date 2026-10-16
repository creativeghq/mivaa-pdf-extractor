"""
Guard: no code names a pre-4 Voyage model as a literal.

Every vector this service writes comes from the voyage-4 series, so a literal
`"voyage-3"` can only be a provenance stamp that lies about it. Eleven did,
including two that stamped every backfilled aspect row. Stamp
`get_settings().voyage_document_model` (or the service's `_last_provider`).
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_APP = Path(__file__).resolve().parents[2] / "app"

_LEGACY = re.compile(r"^voyage-(3|3\.5|3-large|3-lite|3\.5-lite|context-3|code-3|multimodal-3)$")

#: The pricing table prices historical usage rows, so it must keep their names.
_ALLOWED = {_APP / "config" / "ai_pricing.py"}


def test_no_legacy_voyage_model_literal():
    offenders = []
    for path in _APP.rglob("*.py"):
        if path in _ALLOWED:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and _LEGACY.match(node.value):
                offenders.append(f"{path.relative_to(_APP.parent)}:{node.lineno} {node.value!r}")
    assert not offenders, "Legacy Voyage model named as a literal:\n" + "\n".join(offenders)
