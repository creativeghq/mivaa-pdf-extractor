"""What each Claude model accepts. Import-free so source-reading guard tests can load it."""

from __future__ import annotations

from typing import Any, Dict, Optional

OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"

#: `tool_choice` `any`/`tool` is a 400 on these.
_REJECTS_FORCED_TOOL_CHOICE = (
    "claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1",
)
#: Thinking cannot be disabled; effort is the only control.
_ALWAYS_THINKS = ("claude-opus-5-5",)
#: `disabled` is a 400; `between_tools` is the no-thinking setting.
_BETWEEN_TOOLS = ("claude-sonnet-5-5",)
#: Thinks by default where its predecessor did not, so an unset `thinking` is turned off.
_THINKING_OFF_UNLESS_ASKED = ("claude-haiku-5-5",)
#: The only models still accepting `temperature` / `top_p` / `top_k`.
_ACCEPTS_SAMPLING = ("claude-haiku-4-5",)

_THINKING_OFF_EFFORTS = (None, "low", "medium", "high")

THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def _is(model: str, family: tuple) -> bool:
    return any((model or "").startswith(m) for m in family)


def rejects_forced_tool_choice(model: str) -> bool:
    return _is(model, _REJECTS_FORCED_TOOL_CHOICE)


def accepts_sampling_params(model: str) -> bool:
    return _is(model, _ACCEPTS_SAMPLING)


def _append_system(system: Any, text: str) -> Any:
    if not system:
        return text
    if isinstance(system, list):
        return [*system, {"type": "text", "text": text}]
    return f"{system}\n\n{text}"


def shape_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a Messages payload into the form its model accepts. Mutates and returns it.

    A forced tool becomes `auto` plus an instruction naming the tool, so the caller's
    "no tool_use block" check still decides whether the call worked.
    """
    model = str(payload.get("model") or "")
    output_config = dict(payload.get("output_config") or {})
    effort: Optional[str] = output_config.get("effort")
    thinking = payload.get("thinking")

    if thinking is None and _is(model, _THINKING_OFF_UNLESS_ASKED) and effort in _THINKING_OFF_EFFORTS:
        payload["thinking"] = {"type": "disabled"}
    elif isinstance(thinking, dict) and thinking.get("type") == "disabled":
        if _is(model, _ALWAYS_THINKS):
            payload.pop("thinking")
            payload["output_config"] = {**output_config, "effort": effort or "low"}
        elif _is(model, _BETWEEN_TOOLS):
            if effort in _THINKING_OFF_EFFORTS:
                payload["thinking"] = {"type": "between_tools"}
            else:
                payload.pop("thinking")

    tool_choice = payload.get("tool_choice")
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") in ("any", "tool")
        and rejects_forced_tool_choice(model)
    ):
        # A forced call never thought first; budgets sized for that would be spent on thinking.
        if thinking is None and _is(model, _BETWEEN_TOOLS) and effort in _THINKING_OFF_EFFORTS:
            payload["thinking"] = {"type": "between_tools"}
        elif thinking is None and _is(model, _ALWAYS_THINKS) and effort is None:
            payload["output_config"] = {**output_config, "effort": "low"}
        name = tool_choice.get("name")
        auto: Dict[str, Any] = {"type": "auto"}
        if "disable_parallel_tool_use" in tool_choice:
            auto["disable_parallel_tool_use"] = tool_choice["disable_parallel_tool_use"]
        payload["tool_choice"] = auto
        payload["system"] = _append_system(
            payload.get("system"),
            f"Answer by calling the `{name}` tool. Do not reply in text."
            if name else "Answer by calling one of the provided tools. Do not reply in text.",
        )
    return payload
