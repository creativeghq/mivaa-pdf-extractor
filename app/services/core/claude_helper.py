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
from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings
from app.services.core.ai_call_logger import AICallLogger
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

_DEFAULT_CONFIDENCE_BREAKDOWN: Dict[str, float] = {
    "model_confidence": 0.9,
    "completeness": 0.9,
    "consistency": 0.9,
    "validation": 0.9,
}

# Models where the Anthropic API now rejects the `temperature` parameter
# (status: deprecated → invalid_request_error 400). Callers can keep passing
# temperature; we silently drop it for these models so the call still succeeds.
_MODELS_WITHOUT_TEMPERATURE = (
    "claude-opus-4-7",
    "claude-opus-4-6",
)


def _model_supports_temperature(model: str) -> bool:
    return not any(model.startswith(m) for m in _MODELS_WITHOUT_TEMPERATURE)


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
    system: Optional[str] = None,
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


def _call_anthropic_sync(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: Optional[float] = 0.0,
    system: Optional[str] = None,
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

async def tracked_claude_call_async(
    *,
    task: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: Optional[str] = None,
    job_id: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    confidence_score: float = 0.9,
    confidence_breakdown: Optional[Dict[str, float]] = None,
    action: str = "use_ai_result",
    extra_kwargs: Optional[Dict[str, Any]] = None,
    product_id: Optional[str] = None,
    image_id: Optional[str] = None,
) -> ClaudeResponse:
    """Async Claude messages.create with automatic logging + credit debit.

    Returns a ClaudeResponse with the same shape as the SDK's Message.
    Pass tools / tool_choice via `extra_kwargs={'tools': [...], 'tool_choice': {...}}`.
    """
    start = time.time()
    response = await _call_anthropic_async(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        **(extra_kwargs or {}),
    )
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
        confidence_score=confidence_score,
        confidence_breakdown=confidence_breakdown or _DEFAULT_CONFIDENCE_BREAKDOWN,
        action=action,
        job_id=job_id,
        user_id=user_id,
        workspace_id=workspace_id,
        product_id=product_id,
        image_id=image_id,
    )
    return response


def tracked_claude_call(
    *,
    task: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: Optional[str] = None,
    job_id: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    confidence_score: float = 0.9,
    confidence_breakdown: Optional[Dict[str, float]] = None,
    action: str = "use_ai_result",
    extra_kwargs: Optional[Dict[str, Any]] = None,
    product_id: Optional[str] = None,
    image_id: Optional[str] = None,
) -> ClaudeResponse:
    """Sync Claude messages.create with automatic logging + credit debit.

    Logging is fire-and-forget: we wrap the async logger call in asyncio.run
    only if we are not already in an event loop. Inside FastAPI handlers
    (which are async), prefer tracked_claude_call_async().
    """
    import asyncio

    start = time.time()
    response = _call_anthropic_sync(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        **(extra_kwargs or {}),
    )
    latency_ms = int((time.time() - start) * 1000)

    if not user_id and job_id:
        user_id, ws = _resolve_user_from_job(job_id)
        workspace_id = workspace_id or ws

    log_coro = AICallLogger().log_claude_call(
        task=task,
        model=model,
        response=response,
        latency_ms=latency_ms,
        confidence_score=confidence_score,
        confidence_breakdown=confidence_breakdown or _DEFAULT_CONFIDENCE_BREAKDOWN,
        action=action,
        job_id=job_id,
        user_id=user_id,
        workspace_id=workspace_id,
        product_id=product_id,
        image_id=image_id,
    )

    try:
        loop = asyncio.get_running_loop()
        # Inside an event loop — schedule but don't await (fire-and-forget)
        loop.create_task(log_coro)
    except RuntimeError:
        # No event loop — run synchronously
        asyncio.run(log_coro)

    return response
