"""
Uniform error reporting for Anthropic API failures from enrichment services.

Background
----------
Several enrichment services (catalog_legend_extractor_v2, catalog_knowledge_extractor,
product_spec_vision_extractor, product_description_writer, embedding_to_text_service)
wrap their Anthropic API calls in broad `try/except` blocks and log a WARNING
when the call fails, then return an empty / default result so the pipeline keeps
moving.

That swallows infrastructure problems. When the MIVAA ANTHROPIC_API_KEY hits
zero balance, every one of these services starts emitting `credit balance is
too low` warnings to the journal — but nothing reaches Sentry because warnings
aren't captured by default. The job reports `completed=100%` and the admin UI
shows green, while products land in the DB with empty descriptions and zero
spec rollup because Claude never actually produced the enrichment content.

This module gives those services a single, uniform way to surface Anthropic
failures to Sentry with structured tags so on-call sees them immediately.

Usage
-----
    from app.services.core.anthropic_error_reporter import report_anthropic_failure

    try:
        response = await self.client.messages.create(...)
    except Exception as e:
        report_anthropic_failure(
            e,
            service="catalog_knowledge_extractor",
            context={"document_id": document_id, "page_count": len(pages)},
        )
        logger.warning(f"catalog_knowledge_extractor: Claude call failed: {e}")
        return {}  # fallback

Tags set on the Sentry event
----------------------------
- `anthropic_failure`   : "credit_balance" | "rate_limit" | "other"
- `enrichment_service`  : service name passed in (used for grouping/alerting)
- `level`               : "error" for credit_balance (needs human action),
                          "warning" otherwise
"""

import logging
from typing import Any, Dict, Optional

import sentry_sdk

logger = logging.getLogger(__name__)


def _classify_anthropic_error(exc: BaseException) -> str:
    """Classify an Anthropic exception into a tag value.

    Returns one of: "credit_balance", "rate_limit", "auth", "other".
    """
    msg = str(exc).lower()

    # Credit balance / billing exhaustion — the reason we wrote this helper.
    # Anthropic returns 400 with {"error": {"type": "invalid_request_error",
    # "message": "Your credit balance is too low to access the Anthropic API..."}}
    if (
        "credit balance is too low" in msg
        or "credit balance too low" in msg
        or "plans & billing" in msg
        or "plans and billing" in msg
    ):
        return "credit_balance"

    # Rate-limit / quota — these are transient but still worth surfacing if
    # they happen in bursts.
    if (
        "rate_limit" in msg
        or "rate limit" in msg
        or "429" in msg
        or "insufficient_quota" in msg
    ):
        return "rate_limit"

    # Auth / permission
    if (
        "authentication_error" in msg
        or "permission_error" in msg
        or "invalid api key" in msg
        or "401" in msg
        or "403" in msg
    ):
        return "auth"

    return "other"


def report_anthropic_failure(
    exc: BaseException,
    service: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Record an Anthropic API failure in Sentry with uniform tags.

    Args:
        exc: the exception raised by the anthropic client (or httpx wrapper)
        service: short name of the calling service (e.g.
            ``"catalog_knowledge_extractor"``). Used as the
            ``enrichment_service`` tag so alerts and dashboards can group by it.
        context: optional dict of extra key/value pairs to attach as
            Sentry extras (e.g. document_id, page numbers, prompt version).

    Returns:
        The classification tag value (``"credit_balance"``, ``"rate_limit"``,
        ``"auth"`` or ``"other"``). Callers can switch on this to decide
        whether to retry, fall back, or abort.
    """
    classification = _classify_anthropic_error(exc)

    try:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("anthropic_failure", classification)
            scope.set_tag("enrichment_service", service)
            # credit_balance is an ops incident (someone must top up); everything
            # else is transient noise that still deserves visibility but shouldn't
            # page an engineer in the middle of the night.
            scope.level = "error" if classification == "credit_balance" else "warning"
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            sentry_sdk.capture_exception(exc)
    except Exception as sentry_err:
        # Never let Sentry reporting itself break the pipeline.
        logger.debug(
            "Failed to report Anthropic failure to Sentry (%s/%s): %s",
            service, classification, sentry_err,
        )

    return classification
