"""`call_with_schema` must send `output_config.format` in the shape the Messages API accepts.

Every builder names its schema (`format.name`) for logging; the API rejects that key with
a 400, which failed every structured call in the first real ingest (#494).
"""
import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "app" / "services" / "core" / "claude_tool_call.py"


def _load():
    spec = importlib.util.spec_from_file_location("_ctc_under_test", _SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves its module by name
    spec.loader.exec_module(module)
    return module


def test_format_name_is_a_label_never_sent(monkeypatch):
    ctc = _load()
    sent = {}

    async def fake_tracked(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(usage=None)

    fake_helper = types.ModuleType("app.services.core.claude_helper")
    fake_helper.tracked_claude_call_async = fake_tracked
    monkeypatch.setitem(sys.modules, "app.services.core.claude_helper", fake_helper)
    labels = []
    monkeypatch.setattr(
        ctc, "extract_structured_output",
        lambda response, label: labels.append(label) or {"ok": 1},
    )

    config = {"format": {"type": "json_schema", "name": "emit_vision_analysis",
                         "schema": {"type": "object", "additionalProperties": False}}}
    asyncio.run(ctc.call_with_schema(
        task="t", model="m", messages=[], output_config=config,
        extra_kwargs={"output_config": {"effort": "low"}},
    ))

    sent_config = sent["extra_kwargs"]["output_config"]
    assert "name" not in sent_config["format"]
    assert sent_config["format"]["type"] == "json_schema" and "schema" in sent_config["format"]
    assert sent_config["effort"] == "low"
    assert labels == ["emit_vision_analysis"]
    assert config["format"]["name"] == "emit_vision_analysis", "caller's config must not be mutated"
