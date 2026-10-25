"""call_with_schema asks once more after a non-JSON reply, and never after a refusal or a cut-off."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "app/services/core/claude_tool_call.py"


def _load(replies):
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return replies[len(calls) - 1]

    helper = types.ModuleType("app.services.core.claude_helper")
    helper.tracked_claude_call_async = fake_call
    for name in ("app", "app.services", "app.services.core"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["app.services.core.claude_helper"] = helper

    spec = importlib.util.spec_from_file_location("claude_tool_call_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, calls


def _reply(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def _run(mod):
    return asyncio.run(
        mod.call_with_schema(
            task="t", model="m", messages=[], output_config={"format": {"name": "n", "schema": {}}}
        )
    )


def test_second_attempt_rescues_a_non_json_reply():
    mod, calls = _load([_reply("{'a': 1}"), _reply('{"a": 1}')])
    assert _run(mod).data == {"a": 1}
    assert len(calls) == 2


def test_two_non_json_replies_still_fail_and_say_how_the_reply_began():
    mod, calls = _load([_reply("{'a': 1}"), _reply("{'a': 2}")])
    try:
        _run(mod)
    except mod.UnparseableReply as e:
        assert "{'a': 2}" in str(e)
    else:
        raise AssertionError("expected UnparseableReply")
    assert len(calls) == 2


def test_a_cut_off_reply_is_not_retried():
    mod, calls = _load([_reply('{"a":', stop_reason="max_tokens")])
    try:
        _run(mod)
    except mod.ToolCallNotReturned:
        pass
    else:
        raise AssertionError("expected ToolCallNotReturned")
    assert len(calls) == 1
