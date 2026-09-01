"""
Centralised Claude call wrapper.

Single sanctioned entry point for ALL Claude messages.create calls in the mivaa
backend. Wraps the API call + logging in one atomic operation so it is
impossible to make an untracked Claude call.

Architecture (post-2026-05-23 SDK removal):
- Calls go through `_call_anthropic_async` / `_call_anthropic_sync`, both of
  which use `httpx` against the public `POST /v1/messages` endpoint.
- The `anthropic-sdk-python` package is NOT a dependency. Standardising on
  httpx eliminates the SDK-pin failure mode (job_classifier/job_keyword
  workarounds existed because the SDK pin was too old to accept `tools`;
  the Stage 3 vision tool_use fix tripped on the same trap).
- `ClaudeResponse` mimics the SDK's response shape (.content[].type/.text/
  .input/.id/.name, .usage.input_tokens, .usage.output_tokens, .model,
  .stop_reason) so existing call sites parse the response unchanged.
- AICallLogger reads the same attributes (.usage.* and .content[0].text) —
  no logger changes needed.

Why this exists:
- Before this helper, ~25 Claude call sites bypassed the AICallLogger.
- Tokens, costs, and credits were silently uncounted for those calls — real
  spend was ~3x what the dashboard reported.

Rules:
- Every Claude call MUST go through tracked_claude_call() (sync) or
  tracked_claude_call_async() (async).
- Direct calls via the `anthropic` SDK package are no longer possible
  (the package is removed). The `get_ai_client_service().anthropic[_async]`
  property returns a shim whose `.messages.create(...)` proxies here.
- user_id is optional but strongly preferred. If absent, the call is logged
  to ai_call_logs (cost tracking) but credits are not debited.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import httpx

from app.config import get_settings
from app.services.core.ai_call_logger import AICallLogger
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

#: What we record when the caller did not measure a confidence. Named rather than
#: inlined because it is now used in three places and a fallback that disagrees with
#: the documented default is the kind of drift this module exists to prevent.
_DEFAULT_CONFIDENCE = 0.9


def _resolve_confidence(confidence: Union[float, Callable[[Any], float]], response: Any) -> float:
    """A float, or a callable applied to the response once it has arrived.

    Six call sites log the confidence the MODEL reported inside its own reply
    (`result['confidence']`, `result['confidence_score']`). That value does not exist
    at call time, so before this the only way to migrate them off a hand-written
    `log_claude_call` was to downgrade a measured confidence to the default — trading
    an invisible-cost bug for a quietly-wrong quality signal.

    It runs inside a try on purpose: the call has COMPLETED and Anthropic has already
    billed for it. A bookkeeping helper must never be the reason a paid-for result is
    lost, so a callable that raises costs us the confidence, not the response.
    """
    if not callable(confidence):
        return float(confidence)
    try:
        return float(confidence(response))
    except Exception:
        logger.warning(
            "confidence callable raised; recording the default instead",
            exc_info=True,
        )
        return _DEFAULT_CONFIDENCE


_DEFAULT_CONFIDENCE_BREAKDOWN: Dict[str, float] = {
    "model_confidence": 0.9,
    "completeness": 0.9,
    "consistency": 0.9,
    "validation": 0.9,
}

# Models where the Anthropic API now rejects the `temperature` parameter
# (status: deprecated → invalid_request_error 400). Callers can keep passing
# temperature; we silently drop it for these models so the call still succeeds.
#: Models that ACCEPT sampling parameters (`temperature` / `top_p` / `top_k`).
#:
#: This is an ALLOWLIST on purpose. It used to be a denylist
#: (`_MODELS_WITHOUT_TEMPERATURE`) naming only `claude-opus-4-8` and
#: `claude-opus-4-6`, so every model absent from it was sent `temperature` —
#: including `claude-opus-5`, `claude-opus-4-7`, `claude-sonnet-5` and
#: `claude-fable-5`, which REMOVED sampling params and reject them with a hard
#: 400.
#:
#: A denylist fails OPEN here, and `_build_payload` defaults temperature to 0.0
#: so nobody has to opt in: pointing `anthropic_model_validation` at
#: `claude-opus-5` would have 400'd every call through this helper at once —
#: vision, product discovery, stage 4, the document classifier, chunk-type
#: classification, ~20 sites — and none of them would have looked wrong until
#: they ran.
#:
#: Inverted, an unknown model runs at the provider's default sampling. That is a
#: difference nobody will notice, which is the correct way for this to fail.
_MODELS_WITH_TEMPERATURE = (
    "claude-haiku-4-5",
)


def _model_supports_temperature(model: str) -> bool:
    return any(model.startswith(m) for m in _MODELS_WITH_TEMPERATURE)


# ─────────────────────────────────────────────────────────────────────────────
# Response shape (mirrors the anthropic SDK so call sites parse unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _ContentBlock:
    """One content block from a Claude response.

    SDK parity: `.type` is 'text' or 'tool_use'. Text blocks carry `.text`.
    Tool-use blocks carry `.id`, `.name`, and `.input` (a dict).
    """
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None


@dataclass
class ClaudeResponse:
    """Shape-compatible with anthropic SDK's Message response.

    Existing call sites read `.content[0].text`, `.content[i].type`,
    `.content[i].input`, `.usage.input_tokens`, `.usage.output_tokens`,
    `.model`, `.stop_reason` — all preserved.
    """
    id: str = ""
    type: str = "message"
    role: str = "assistant"
    content: List[_ContentBlock] = field(default_factory=list)
    model: str = ""
    stop_reason: Optional[str] = None
    usage: _Usage = field(default_factory=_Usage)


def _parse_anthropic_response(data: Dict[str, Any]) -> ClaudeResponse:
    blocks: List[_ContentBlock] = []
    for block in data.get("content") or []:
        blocks.append(_ContentBlock(
            type=block.get("type", "text"),
            text=block.get("text"),
            id=block.get("id"),
            name=block.get("name"),
            input=block.get("input"),
        ))
    usage = data.get("usage") or {}
    return ClaudeResponse(
        id=data.get("id", ""),
        type=data.get("type", "message"),
        role=data.get("role", "assistant"),
        content=blocks,
        model=data.get("model", ""),
        stop_reason=data.get("stop_reason"),
        usage=_Usage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
        ),
    )


def _build_payload(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float],
    system: Optional[str],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if temperature is not None and _model_supports_temperature(model):
        payload["temperature"] = temperature
    if system:
        payload["system"] = system
    if extra:
        payload.update(extra)
    return payload


def _request_headers() -> Dict[str, str]:
    settings = get_settings()
    api_key = getattr(settings, "anthropic_api_key", None) or ""
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured")
    return {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _request_timeout() -> float:
    """Match the SDK default + the gateway's 200s vision timeout."""
    settings = get_settings()
    return float(getattr(settings, "anthropic_timeout", 200.0) or 200.0)


async def _call_anthropic_async(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: Optional[float] = 0.0,
    # Forwarded verbatim into the request payload, so this is str OR the block list
    # form (`[{'type':'text','text':..., 'cache_control':...}]`). Two price-monitor
    # call sites depend on the block form for prompt caching; narrowing this to str
    # would be a type that lies about what the function accepts.
    system: Optional[Any] = None,
    **extra: Any,
) -> ClaudeResponse:
    """Raw async httpx call to Anthropic Messages API.

    Returns a ClaudeResponse with the same shape as the SDK's Message object.
    Pass tools / tool_choice via `**extra`.
    """
    payload = _build_payload(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, system=system, extra=extra,
    )
    headers = _request_headers()
    timeout = _request_timeout()

    # Use a per-call AsyncClient so we don't pin a connection across the
    # whole pipeline. ai_client_service.httpx is sized for parallel calls
    # but we want isolated retry/timeout semantics per Anthropic call.
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        resp = await client.post(_ANTHROPIC_API_URL, headers=headers, json=payload)
    if resp.status_code >= 400:
        body_preview = resp.text[:500]
        raise httpx.HTTPStatusError(
            f"Anthropic API {resp.status_code}: {body_preview}",
            request=resp.request,
            response=resp,
        )
    return _parse_anthropic_response(resp.json())


async def _stream_anthropic_async(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: Optional[float] = 0.0,
    system: Optional[Any] = None,
    **extra: Any,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream `POST /v1/messages`, yielding each decoded SSE event dict.

    Same request as `_call_anthropic_async` with `stream: true` added, so `tools` /
    `tool_choice` behave identically — a forced tool arrives as `content_block_start`
    (an empty `input`) followed by `input_json_delta` fragments of the input's JSON
    TEXT, which is what lets a caller act on the first array element without waiting
    for the last.

    Events are yielded raw. Assembling them into a `ClaudeResponse` is
    `tracked_claude_stream_async`'s job, and it must happen there rather than here
    because that is where the usage numbers have to reach the logger.
    """
    payload = _build_payload(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, system=system, extra=extra,
    )
    payload["stream"] = True
    headers = _request_headers()
    timeout = _request_timeout()

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        async with client.stream(
            "POST", _ANTHROPIC_API_URL, headers=headers, json=payload,
        ) as resp:
            if resp.status_code >= 400:
                # The body has not been read yet on a streamed response, and the
                # error detail is the only thing that makes a 400 actionable.
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise httpx.HTTPStatusError(
                    f"Anthropic API {resp.status_code}: {body}",
                    request=resp.request,
                    response=resp,
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue  # `event:` lines restate `data`'s own `type`
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    yield json.loads(raw)
                except ValueError:
                    logger.warning("unparseable SSE frame from Anthropic: %.200s", raw)


def _call_anthropic_sync(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: Optional[float] = 0.0,
    # Forwarded verbatim into the request payload, so this is str OR the block list
    # form (`[{'type':'text','text':..., 'cache_control':...}]`). Two price-monitor
    # call sites depend on the block form for prompt caching; narrowing this to str
    # would be a type that lies about what the function accepts.
    system: Optional[Any] = None,
    **extra: Any,
) -> ClaudeResponse:
    """Sync httpx variant. Used from non-async code paths."""
    payload = _build_payload(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, system=system, extra=extra,
    )
    headers = _request_headers()
    timeout = _request_timeout()
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        resp = client.post(_ANTHROPIC_API_URL, headers=headers, json=payload)
    if resp.status_code >= 400:
        body_preview = resp.text[:500]
        raise httpx.HTTPStatusError(
            f"Anthropic API {resp.status_code}: {body_preview}",
            request=resp.request,
            response=resp,
        )
    return _parse_anthropic_response(resp.json())


# ─────────────────────────────────────────────────────────────────────────────
# Internal job → user resolution (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_user_from_job(job_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """If we have a job_id but no user_id, look up the job owner.

    Background pipelines (PDF ingest, agent runs) carry a job_id but no
    explicit user_id. The job row tells us who triggered it, so we can still
    bill the right user for the AI call.
    """
    if not job_id:
        return None, None
    try:
        sb = get_supabase_client().client
        row = sb.table("background_jobs") \
                .select("user_id, workspace_id") \
                .eq("id", job_id).single().execute()
        if row.data:
            return row.data.get("user_id"), row.data.get("workspace_id")
    except Exception as exc:
        logger.warning(f"[claude_helper] Could not resolve user from job_id={job_id}: {exc}")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Public tracked-call API (signatures unchanged; backed by httpx now)
# ─────────────────────────────────────────────────────────────────────────────

#: A Claude call that raised still cost money — Anthropic bills a request whose
#: tokens were generated before the client timed out or the connection dropped.
#: Logging only the successes therefore makes ai_usage_logs quietly optimistic
#: AND removes the failure data you most want when a model starts erroring: the
#: task, the model, and when it began. Audit #12: neither tracked wrapper had a
#: try/except, so every failed call vanished without a row.
_FAILED_CONFIDENCE_BREAKDOWN = {
    "model_confidence": 0.0,
    "completeness": 0.0,
    "consistency": 0.0,
    "validation": 0.0,
}


async def _log_failed_claude_call_async(
    *, task: str, model: str, latency_ms: int, error: BaseException,
    job_id, user_id, workspace_id, product_id, image_id,
) -> None:
    """Best-effort failure row. Never raises: the caller is already failing and
    a logging problem must not replace the real exception."""
    try:
        await AICallLogger().log_ai_call(
            task=task,
            model=model,
            # Unknown, not zero-cost. The row exists to record that the call
            # happened and failed; cost attribution for a raised call is not
            # recoverable client-side.
            #
            # The comment above was already right and the DATA did not say it (#19
            # M6-5). A timeout or 5xx AFTER Anthropic accepted and billed the request
            # was written as cost=0.0 with nothing marking it, so the spend was
            # permanently indistinguishable from a free no-op — the silent-zero shape
            # applied to the failure ledger instead of the success one.
            # `unbilled_reason` is the column that already exists for exactly this:
            # NULL means billed, anything else names why it was not.
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            unbilled_reason="billable_attempt_failed",
            latency_ms=latency_ms,
            confidence_score=0.0,
            confidence_breakdown=_FAILED_CONFIDENCE_BREAKDOWN,
            action="call_failed",
            job_id=job_id,
            user_id=user_id,
            workspace_id=workspace_id,
            product_id=product_id,
            image_id=image_id,
            error_message=str(error)[:500],
        )
    except Exception as log_err:  # pragma: no cover - logging must never mask
        logger.warning(f"Could not log failed Claude call for task={task}: {log_err}")


def _log_failed_claude_call_sync(
    *, task: str, model: str, latency_ms: int, error: BaseException,
    job_id, user_id, workspace_id, product_id, image_id,
) -> None:
    """Sync twin. Dispatches the same coroutine the sync success path uses:
    fire-and-forget inside a running loop, asyncio.run outside one."""
    import asyncio

    try:
        coro = AICallLogger().log_ai_call(
            task=task,
            model=model,
            # See the async twin: cost is UNKNOWN here, not zero, and the marker is
            # what makes that recoverable (#19 M6-5).
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            unbilled_reason="billable_attempt_failed",
            latency_ms=latency_ms,
            confidence_score=0.0,
            confidence_breakdown=_FAILED_CONFIDENCE_BREAKDOWN,
            action="call_failed",
            job_id=job_id,
            user_id=user_id,
            workspace_id=workspace_id,
            product_id=product_id,
            image_id=image_id,
            error_message=str(error)[:500],
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)
    except Exception as log_err:  # pragma: no cover
        logger.warning(f"Could not log failed Claude call for task={task}: {log_err}")


async def tracked_claude_call_async(
    *,
    task: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    # Forwarded verbatim into the request payload, so this is str OR the block list
    # form (`[{'type':'text','text':..., 'cache_control':...}]`). Two price-monitor
    # call sites depend on the block form for prompt caching; narrowing this to str
    # would be a type that lies about what the function accepts.
    system: Optional[Any] = None,
    job_id: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    # A float, OR a callable applied to the response once it arrives — see
    # `_resolve_confidence`. The model's own reported confidence lives INSIDE the
    # reply and is therefore not knowable at call time.
    confidence_score: Union[float, Callable[[Any], float]] = _DEFAULT_CONFIDENCE,
    confidence_breakdown: Optional[Dict[str, float]] = None,
    # Diagnostic metadata forwarded verbatim to `log_claude_call` (e.g.
    # `{"prompt_length": ...}`). Added so migrating sites do not have to drop the field
    # to move off a hand-written logger — losing information is not a neutral cost of
    # consolidating, it is a reason people keep the copy.
    request_data: Optional[Dict[str, Any]] = None,
    action: str = "use_ai_result",
    extra_kwargs: Optional[Dict[str, Any]] = None,
    product_id: Optional[str] = None,
    image_id: Optional[str] = None,
    # Platform work with genuinely nobody to bill (health checks, cron probes).
    # Recorded as `system_initiated` rather than flagged as `no_principal`,
    # which is what keeps a deliberate gap distinguishable from a call that
    # simply lost its user_id on the way down (M3-2, #16).
    system_initiated: bool = False,
) -> ClaudeResponse:
    """Async Claude messages.create with automatic logging + credit debit.

    Returns a ClaudeResponse with the same shape as the SDK's Message.
    Pass tools / tool_choice via `extra_kwargs={'tools': [...], 'tool_choice': {...}}`.
    """
    start = time.time()
    try:
        response = await _call_anthropic_async(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **(extra_kwargs or {}),
        )
    except BaseException as call_err:
        # Record the failure, then re-raise unchanged — callers' behaviour is
        # untouched, they simply stop being invisible in ai_usage_logs.
        _uid, _wsid = user_id, workspace_id
        if not _uid and job_id:
            try:
                _uid, _ws = _resolve_user_from_job(job_id)
                _wsid = _wsid or _ws
            except Exception:
                pass
        await _log_failed_claude_call_async(
            task=task, model=model,
            latency_ms=int((time.time() - start) * 1000),
            error=call_err, job_id=job_id, user_id=_uid, workspace_id=_wsid,
            product_id=product_id, image_id=image_id,
        )
        raise
    latency_ms = int((time.time() - start) * 1000)

    # Auto-resolve user from job if not provided
    if not user_id and job_id:
        user_id, ws = _resolve_user_from_job(job_id)
        workspace_id = workspace_id or ws

    await AICallLogger().log_claude_call(
        task=task,
        model=model,
        response=response,
        latency_ms=latency_ms,
        confidence_score=_resolve_confidence(confidence_score, response),
        confidence_breakdown=confidence_breakdown or _DEFAULT_CONFIDENCE_BREAKDOWN,
        request_data=request_data,
        action=action,
        job_id=job_id,
        user_id=user_id,
        workspace_id=workspace_id,
        product_id=product_id,
        image_id=image_id,
        system_initiated=system_initiated,
    )
    return response


async def tracked_claude_stream_async(
    *,
    task: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: Optional[Any] = None,
    job_id: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    confidence_score: Union[float, Callable[[Any], float]] = _DEFAULT_CONFIDENCE,
    confidence_breakdown: Optional[Dict[str, float]] = None,
    request_data: Optional[Dict[str, Any]] = None,
    action: str = "use_ai_result",
    extra_kwargs: Optional[Dict[str, Any]] = None,
    product_id: Optional[str] = None,
    image_id: Optional[str] = None,
    system_initiated: bool = False,
) -> AsyncIterator[Dict[str, Any]]:
    """Streaming twin of `tracked_claude_call_async`, with the same logging contract.

    Yields, in order:

      ``{"type": "text", "text": str}``            — a text delta
      ``{"type": "input_json", "partial": str}``   — a fragment of a tool input's JSON
      ``{"type": "complete", "response": ClaudeResponse}`` — once, last

    The assembled `ClaudeResponse` is shape-identical to the non-streaming one, so
    `extract_tool_input`, `AICallLogger` and every existing reader work on it unchanged.

    Cost is logged after the final event and failures go through
    `_log_failed_claude_call_async`, exactly as the non-streaming path does. A streamed
    call that skipped this would be spend Anthropic bills and no cost view can see —
    the failure shape this module was written to close, reopened one API away.
    """
    start = time.time()
    blocks: Dict[int, _ContentBlock] = {}
    partials: Dict[int, List[str]] = {}
    response = ClaudeResponse(model=model)

    try:
        async for event in _stream_anthropic_async(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **(extra_kwargs or {}),
        ):
            etype = event.get("type")

            if etype == "message_start":
                message = event.get("message") or {}
                response.id = message.get("id", "")
                response.role = message.get("role", "assistant")
                response.model = message.get("model", model)
                usage = message.get("usage") or {}
                response.usage.input_tokens = int(usage.get("input_tokens", 0) or 0)
                response.usage.output_tokens = int(usage.get("output_tokens", 0) or 0)

            elif etype == "content_block_start":
                index = int(event.get("index", 0))
                block = event.get("content_block") or {}
                blocks[index] = _ContentBlock(
                    type=block.get("type", "text"),
                    text="" if block.get("type") == "text" else None,
                    id=block.get("id"),
                    name=block.get("name"),
                    input={} if block.get("type") == "tool_use" else None,
                )
                partials[index] = []

            elif etype == "content_block_delta":
                index = int(event.get("index", 0))
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    chunk = delta.get("text") or ""
                    block = blocks.get(index)
                    if block is not None:
                        block.text = (block.text or "") + chunk
                    yield {"type": "text", "text": chunk}
                elif delta.get("type") == "input_json_delta":
                    chunk = delta.get("partial_json") or ""
                    partials.setdefault(index, []).append(chunk)
                    yield {"type": "input_json", "partial": chunk}

            elif etype == "content_block_stop":
                index = int(event.get("index", 0))
                block = blocks.get(index)
                if block is not None and block.type == "tool_use":
                    text = "".join(partials.get(index) or [])
                    try:
                        block.input = json.loads(text) if text else {}
                    except ValueError:
                        # Left as None so `extract_tool_input` raises
                        # ToolCallNotReturned rather than handing back a plausible
                        # empty dict — a truncated reply is a broken contract, not
                        # an empty answer.
                        logger.warning(
                            "tool input for %r did not parse after %d chars",
                            block.name, len(text),
                        )
                        block.input = None

            elif etype == "message_delta":
                usage = event.get("usage") or {}
                if "output_tokens" in usage:
                    response.usage.output_tokens = int(usage.get("output_tokens") or 0)
                delta = event.get("delta") or {}
                if delta.get("stop_reason"):
                    response.stop_reason = delta["stop_reason"]

            elif etype == "error":
                detail = (event.get("error") or {}).get("message", "unknown")
                raise RuntimeError(f"Anthropic stream error: {detail}")

        response.content = [blocks[i] for i in sorted(blocks)]

    except BaseException as call_err:
        _uid, _wsid = user_id, workspace_id
        if not _uid and job_id:
            try:
                _uid, _ws = _resolve_user_from_job(job_id)
                _wsid = _wsid or _ws
            except Exception:
                pass
        await _log_failed_claude_call_async(
            task=task, model=model,
            latency_ms=int((time.time() - start) * 1000),
            error=call_err, job_id=job_id, user_id=_uid, workspace_id=_wsid,
            product_id=product_id, image_id=image_id,
        )
        raise

    latency_ms = int((time.time() - start) * 1000)

    if not user_id and job_id:
        user_id, ws = _resolve_user_from_job(job_id)
        workspace_id = workspace_id or ws

    await AICallLogger().log_claude_call(
        task=task,
        model=model,
        response=response,
        latency_ms=latency_ms,
        confidence_score=_resolve_confidence(confidence_score, response),
        confidence_breakdown=confidence_breakdown or _DEFAULT_CONFIDENCE_BREAKDOWN,
        request_data=request_data,
        action=action,
        job_id=job_id,
        user_id=user_id,
        workspace_id=workspace_id,
        product_id=product_id,
        image_id=image_id,
        system_initiated=system_initiated,
    )

    yield {"type": "complete", "response": response}


def tracked_claude_call(
    *,
    task: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    # Forwarded verbatim into the request payload, so this is str OR the block list
    # form (`[{'type':'text','text':..., 'cache_control':...}]`). Two price-monitor
    # call sites depend on the block form for prompt caching; narrowing this to str
    # would be a type that lies about what the function accepts.
    system: Optional[Any] = None,
    job_id: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    # A float, OR a callable applied to the response once it arrives — see
    # `_resolve_confidence`. The model's own reported confidence lives INSIDE the
    # reply and is therefore not knowable at call time.
    confidence_score: Union[float, Callable[[Any], float]] = _DEFAULT_CONFIDENCE,
    confidence_breakdown: Optional[Dict[str, float]] = None,
    # Diagnostic metadata forwarded verbatim to `log_claude_call` (e.g.
    # `{"prompt_length": ...}`). Added so migrating sites do not have to drop the field
    # to move off a hand-written logger — losing information is not a neutral cost of
    # consolidating, it is a reason people keep the copy.
    request_data: Optional[Dict[str, Any]] = None,
    action: str = "use_ai_result",
    extra_kwargs: Optional[Dict[str, Any]] = None,
    product_id: Optional[str] = None,
    image_id: Optional[str] = None,
    # Platform work with genuinely nobody to bill (health checks, cron probes).
    # Recorded as `system_initiated` rather than flagged as `no_principal`,
    # which is what keeps a deliberate gap distinguishable from a call that
    # simply lost its user_id on the way down (M3-2, #16).
    system_initiated: bool = False,
) -> ClaudeResponse:
    """Sync Claude messages.create with automatic logging + credit debit.

    Logging is fire-and-forget: we wrap the async logger call in asyncio.run
    only if we are not already in an event loop. Inside FastAPI handlers
    (which are async), prefer tracked_claude_call_async().
    """
    import asyncio

    start = time.time()
    try:
        response = _call_anthropic_sync(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **(extra_kwargs or {}),
        )
    except BaseException as call_err:
        # Same contract as the async twin: record, then re-raise unchanged.
        _uid, _wsid = user_id, workspace_id
        if not _uid and job_id:
            try:
                _uid, _ws = _resolve_user_from_job(job_id)
                _wsid = _wsid or _ws
            except Exception:
                pass
        _log_failed_claude_call_sync(
            task=task, model=model,
            latency_ms=int((time.time() - start) * 1000),
            error=call_err, job_id=job_id, user_id=_uid, workspace_id=_wsid,
            product_id=product_id, image_id=image_id,
        )
        raise
    latency_ms = int((time.time() - start) * 1000)

    if not user_id and job_id:
        user_id, ws = _resolve_user_from_job(job_id)
        workspace_id = workspace_id or ws

    log_coro = AICallLogger().log_claude_call(
        task=task,
        model=model,
        response=response,
        latency_ms=latency_ms,
        confidence_score=_resolve_confidence(confidence_score, response),
        confidence_breakdown=confidence_breakdown or _DEFAULT_CONFIDENCE_BREAKDOWN,
        request_data=request_data,
        action=action,
        job_id=job_id,
        user_id=user_id,
        workspace_id=workspace_id,
        product_id=product_id,
        image_id=image_id,
        system_initiated=system_initiated,
    )

    try:
        loop = asyncio.get_running_loop()
        # Inside an event loop — schedule but don't await (fire-and-forget)
        loop.create_task(log_coro)
    except RuntimeError:
        # No event loop — run synchronously
        asyncio.run(log_coro)

    return response
