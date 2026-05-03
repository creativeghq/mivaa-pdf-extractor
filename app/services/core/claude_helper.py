"""
Centralised Claude call wrapper.

Single sanctioned entry point for ALL Claude messages.create calls in the mivaa
backend. Wraps the API call + logging in one atomic operation so it is
impossible to make an untracked Claude call.

Why this exists:
- Before this helper, ~25 Claude call sites bypassed the AICallLogger.
- Tokens, costs, and credits were silently uncounted for those calls — real
  spend was ~3x what the dashboard reported.

Rules:
- Every Claude call MUST go through tracked_claude_call() (sync) or
  tracked_claude_call_async() (async).
- Direct calls to client.messages.create(...) anywhere outside this module
  are a regression — grep for them in CI.
- user_id is optional but strongly preferred. If absent, the call is logged
  to ai_call_logs (cost tracking) but credits are not debited.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.services.core.ai_call_logger import AICallLogger
from app.services.core.ai_client_service import get_ai_client_service
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

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


def _build_messages_kwargs(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    system: Optional[str],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if _model_supports_temperature(model):
        kwargs["temperature"] = temperature
    if system:
        kwargs["system"] = system
    if extra:
        kwargs.update(extra)
    return kwargs


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
):
    """Async Claude messages.create with automatic logging + credit debit.

    Returns the raw Anthropic response (same shape as client.messages.create).
    """
    start = time.time()
    ai = get_ai_client_service()
    client = ai.anthropic_async

    kwargs = _build_messages_kwargs(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, system=system, extra=extra_kwargs,
    )
    response = await client.messages.create(**kwargs)
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
):
    """Sync Claude messages.create with automatic logging + credit debit.

    Logging is fire-and-forget: we wrap the async logger call in asyncio.run
    only if we are not already in an event loop. Inside FastAPI handlers
    (which are async), prefer tracked_claude_call_async().
    """
    import asyncio

    start = time.time()
    ai = get_ai_client_service()
    client = ai.anthropic

    kwargs = _build_messages_kwargs(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, system=system, extra=extra_kwargs,
    )
    response = client.messages.create(**kwargs)
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
