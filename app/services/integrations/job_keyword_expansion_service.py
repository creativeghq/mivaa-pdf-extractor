"""
Job Keyword Expansion — Haiku-driven keyword variant generation.

For "Product Manager" the user almost always wants to also catch "Senior
Product Manager", "PM", "Product Lead", "Principal PM", "Head of Product".
Manually listing every variant is friction; a one-shot Haiku call covers it
with high recall.

When called:
  - On first save (`JobResearchService.create()` runs this synchronously)
  - On `POST /track/{id}/regenerate-keywords` (manual refresh of the variants)

Cost: one Haiku tool-use call per tracked_job, ~$0.001. Cached on
`tracked_jobs.expanded_keywords` so we never re-run unless asked.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import httpx

from app.services.integrations import job_cost_logger as costs

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

logger = logging.getLogger(__name__)


_EXPAND_TOOL = {
    "name": "submit_expanded_keywords",
    "description": "Return job-search keyword variants the user likely also wants to match.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title_variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Alternate phrasings of the user's job titles (e.g. 'frontend developer' for 'frontend engineer'). Up to 8.",
            },
            "seniority_variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Same role at adjacent seniority levels matching the user's stated seniority (e.g. 'senior product manager' + 'lead product manager' for a senior search). Up to 6.",
            },
            "abbreviations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Common abbreviations or industry shorthand (e.g. 'PM' for product manager). Up to 4.",
            },
            "rejected_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Terms NOT to add because they'd dilute precision (e.g. for 'senior react developer' DO NOT add 'react native'). Surface this to the user.",
                "default": [],
            },
        },
        "required": ["title_variants", "seniority_variants", "abbreviations"],
    },
}


def _build_user_prompt(
    label: str,
    keywords: List[str],
    location: Optional[str],
    seniority: Optional[str],
    excluded_keywords: Optional[List[str]],
) -> str:
    return (
        f"User set up a job search:\n"
        f"  Label: {label}\n"
        f"  Keywords (any-of): {', '.join(keywords)}\n"
        f"  Seniority: {seniority or 'unspecified'}\n"
        f"  Location: {location or 'any'}\n"
        f"  Excluded terms: {', '.join(excluded_keywords or []) or '(none)'}\n\n"
        f"Generate keyword variants the user almost certainly also wants to match. "
        f"Be precise — variants should match the user's intent, not broaden into different "
        f"specializations. For example:\n"
        f"  'product manager' → also 'senior product manager', 'product lead', 'PM' "
        f"(NOT 'product marketing manager' — different role).\n"
        f"  'react developer' → also 'react engineer', 'frontend developer (React)', "
        f"'reactjs developer' (NOT 'react native developer' — mobile, different stack).\n"
        f"  'devops engineer' → also 'site reliability engineer', 'platform engineer', "
        f"'SRE', 'infrastructure engineer'.\n\n"
        f"Honor the seniority level. If the user said 'senior', do not add junior variants. "
        f"If they said 'any' or unspecified, include junior + mid + senior.\n\n"
        f"Honor excluded terms — never produce a variant that contains an excluded term.\n\n"
        f"Call submit_expanded_keywords with concise lowercase variants."
    )


async def expand_keywords(
    *,
    label: str,
    keywords: List[str],
    location: Optional[str] = None,
    seniority: Optional[str] = None,
    excluded_keywords: Optional[List[str]] = None,
    attribution: Optional[costs.CostAttribution] = None,
) -> dict:
    """Returns {expanded: [...], rejected: [...], raw: {title_variants, seniority_variants, abbreviations}}."""
    api_key = os.getenv("ANTHROPIC_API_KEY") or ""
    if not api_key:
        logger.info("job-keyword-expansion: ANTHROPIC_API_KEY missing — skipping")
        return {"expanded": [], "rejected": [], "raw": {}}

    started = time.time()
    in_tok = 0
    out_tok = 0
    success = True
    err: Optional[str] = None
    raw: dict = {}

    # Call Anthropic /v1/messages HTTP API directly to avoid SDK version drama.
    # The deployed anthropic-python is pinned to 0.23.1 (beta-era tool-use path);
    # the HTTP API has had stable tool_use since 2024-06 and doesn't care about
    # our SDK version.
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 600,
        "tools": [_EXPAND_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_expanded_keywords"},
        "messages": [{"role": "user", "content": _build_user_prompt(label, keywords, location, seniority, excluded_keywords)}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
            resp = await client.post(_ANTHROPIC_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        usage = data.get("usage") or {}
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        for block in (data.get("content") or []):
            if block.get("type") == "tool_use" and block.get("name") == "submit_expanded_keywords":
                raw = block.get("input") or {}
                break
    except Exception as e:
        success = False
        err = str(e)[:200]
        logger.warning(f"job-keyword-expansion haiku call failed: {err}")

    costs.log_haiku_call(
        attribution=attribution,
        operation="keyword_expansion",
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=int((time.time() - started) * 1000),
        success=success,
        error_message=err,
    )

    if not success or not raw:
        return {"expanded": [], "rejected": [], "raw": {}}

    # Normalize + dedupe + drop anything matching an excluded term
    excluded_lower = {(e or "").lower() for e in (excluded_keywords or []) if e}
    seen: set = set()
    expanded: list = []
    for bucket in ("title_variants", "seniority_variants", "abbreviations"):
        for term in (raw.get(bucket) or []):
            t = (term or "").strip().lower()
            if not t or t in seen:
                continue
            if any(ex in t for ex in excluded_lower):
                continue
            seen.add(t)
            expanded.append(t)

    rejected = [(r or "").strip() for r in (raw.get("rejected_terms") or []) if r]

    return {"expanded": expanded[:18], "rejected": rejected, "raw": raw}
