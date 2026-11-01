"""Every Claude request is rewritten into the form its model accepts."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PATH = Path(__file__).resolve().parents[2] / "app" / "services" / "core" / "claude_models.py"
_spec = importlib.util.spec_from_file_location("claude_models", _PATH)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


def _forced(model: str, **extra):
    return {
        "model": model, "max_tokens": 300, "messages": [],
        "tools": [{"name": "record"}], "tool_choice": {"type": "tool", "name": "record"}, **extra,
    }


@pytest.mark.parametrize("model", [cm.OPUS, cm.SONNET])
def test_forced_tool_becomes_auto_with_an_instruction(model: str) -> None:
    out = cm.shape_request(_forced(model, system="Extract."))
    assert out["tool_choice"] == {"type": "auto"}
    assert "`record` tool" in out["system"]


def test_haiku_keeps_its_forced_tool_and_runs_without_thinking() -> None:
    out = cm.shape_request(_forced(cm.HAIKU))
    assert out["tool_choice"] == {"type": "tool", "name": "record"}
    assert out["thinking"] == {"type": "disabled"}


def test_a_converted_forced_call_does_not_spend_a_small_budget_thinking() -> None:
    assert cm.shape_request(_forced(cm.SONNET))["thinking"] == {"type": "between_tools"}
    assert cm.shape_request(_forced(cm.OPUS))["output_config"] == {"effort": "low"}


def test_disabled_thinking_maps_to_what_each_model_accepts() -> None:
    off = {"type": "disabled"}
    opus = cm.shape_request({"model": cm.OPUS, "thinking": off})
    assert "thinking" not in opus and opus["output_config"]["effort"] == "low"
    assert cm.shape_request({"model": cm.SONNET, "thinking": off})["thinking"] == {"type": "between_tools"}
    xhigh = cm.shape_request({"model": cm.SONNET, "thinking": off, "output_config": {"effort": "xhigh"}})
    assert "thinking" not in xhigh


def test_a_caller_chosen_thinking_setting_is_left_alone() -> None:
    adaptive = {"type": "adaptive"}
    assert cm.shape_request({"model": cm.HAIKU, "thinking": adaptive})["thinking"] == adaptive


def test_block_system_prompt_gets_the_instruction_appended_as_a_block() -> None:
    system = [{"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}]
    out = cm.shape_request(_forced(cm.OPUS, system=system))
    assert out["system"][0] == system[0] and len(out["system"]) == 2


def test_only_the_previous_haiku_accepts_sampling() -> None:
    assert cm.accepts_sampling_params("claude-haiku-4-5")
    for model in (cm.OPUS, cm.SONNET, cm.HAIKU):
        assert not cm.accepts_sampling_params(model)
