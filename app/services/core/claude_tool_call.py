"""One forced tool call. The model returns a structured object or it fails.

WHAT IT REPLACES
----------------
34 parsers across 22 files ask a model for JSON in a prompt and then repair the reply
when it is not JSON — 15 of them by stripping markdown fences (#32). The fence-strip is
the tell: it is an admission that the free-form contract does not hold, written by
someone who had already seen it break.

Every one of those sites has the same three properties, and forcing the tool call
removes all three at once:

  1. a prompt ASKING for JSON — the model may still write prose around it
  2. a parser that REPAIRS the reply — regex, fence-stripping, first-`{...}` matching
  3. a failure path that degrades to None or a skip — so a broken reply is
     indistinguishable from a legitimate "nothing found"

With `tool_choice` forced, the model cannot return prose, so there is nothing to repair,
and an absent tool block is a broken API contract rather than an ambiguous result.

WHY IT IS SHARED
----------------
The correct shape already existed in FOUR files — `document_classifier`,
`consensus_validator`, `chunk_type_classification_service`, and the vision path — each
with its own hand-copied "find the tool_use block, raise if absent" loop. This session
has now unified a debit rule that had seven copies (#30 M16-1) and a PDF bounds check
that had three (#35 M20-2). This is the same argument, made before the drift rather
than after it.

INVARIANT 9 AND THE HONEST SCOPE
--------------------------------
The platform rule requires forced tool-calling for "a classifier whose verdict drives a
DB write or alert". Not all 34 sites are that: a parser feeding a search ranking is a
robustness concern, one feeding an INSERT is the invariant. Both benefit; only the
second is mandatory. The guard file ratchets the total down rather than claiming all 34
are equally urgent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallResult:
    """The tool input, AND the token usage that produced it.

    Usage is returned rather than dropped because the first caller migrated
    (`agent_routes`) accumulates it into `agent_runs.input_tokens` / `output_tokens`.
    A helper that returned only the parsed object would have quietly zeroed those
    counters on every migrated site — a plausible zero that nothing raises on, which is
    the exact shape this codebase keeps being audited for. Making it part of the return
    type means the next migration cannot forget.

    The cost itself is already logged by `tracked_claude_call_async`; these are for the
    caller's own bookkeeping.
    """

    data: Dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


class ToolCallNotReturned(RuntimeError):
    """The model returned no tool_use block for a FORCED tool_choice.

    A distinct type on purpose. This is not "the model was unsure" and not "nothing
    matched" — it is the API contract breaking, and it is retryable. Collapsing it into
    None is what made the free-form parsers unable to tell a broken reply from an empty
    one.
    """


def extract_tool_input(response: Any, tool_name: str) -> Dict[str, Any]:
    """The tool input from a forced tool call, or raise.

    Never returns a default. `tool_choice` was forced, so an absent block means the
    contract broke, and guessing would be worse than admitting it.
    """
    for block in (getattr(response, "content", None) or []):
        if getattr(block, "type", None) == "tool_use":
            if getattr(block, "name", tool_name) != tool_name:
                continue
            tool_input = getattr(block, "input", None)
            if isinstance(tool_input, dict):
                return tool_input
            raise ToolCallNotReturned(
                f"tool_use block for {tool_name!r} carried {type(tool_input).__name__}, "
                "not an object"
            )
    raise ToolCallNotReturned(
        f"no tool_use block for {tool_name!r} despite a forced tool_choice"
    )


async def call_with_tool(
    *,
    task: str,
    model: str,
    messages: List[Dict[str, Any]],
    tool: Dict[str, Any],
    max_tokens: int = 1024,
    system: Optional[str] = None,
    job_id: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    product_id: Optional[str] = None,
    image_id: Optional[str] = None,
    required: Optional[List[str]] = None,
) -> ToolCallResult:
    """Call Claude with `tool` forced, and return its validated input plus usage.

    Goes through `tracked_claude_call_async` so the cost lands in `ai_usage_logs`
    automatically (pipeline convention 10) — a hand-rolled httpx POST here would be
    invisible to every cost view, which is the shape #30 measured.

    `required` names keys the caller cannot proceed without. The tool schema should
    already declare them, but a schema is a request and this is a check: a model that
    omits one has not answered the question, and finding that out here beats finding it
    out from a KeyError three frames down.
    """
    from app.services.core.claude_helper import tracked_claude_call_async

    response = await tracked_claude_call_async(
        task=task,
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        job_id=job_id,
        user_id=user_id,
        workspace_id=workspace_id,
        product_id=product_id,
        image_id=image_id,
        extra_kwargs={
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool["name"]},
        },
    )

    tool_input = extract_tool_input(response, tool["name"])

    missing = [k for k in (required or []) if tool_input.get(k) in (None, "")]
    if missing:
        raise ToolCallNotReturned(
            f"{tool['name']} returned no value for {missing} — the reply is structurally "
            "valid but does not answer the question"
        )

    usage = getattr(response, "usage", None)
    return ToolCallResult(
        data=tool_input,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )
