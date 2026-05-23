"""
Job Classifier — Haiku-based relevance gate.

For each candidate JobHit, classify against the user's keywords + filters into:
  - match        : title + description clearly hit the user's intent
  - tangential   : adjacent role (e.g. "React" listing for a "Vue" search)
  - mismatch     : entirely different field (drop)
  - unverifiable : page didn't load enough signal — keep but flag

Cost discipline:
  1. Rule shortcut first (deterministic). Drops ~60% of candidates before Haiku.
  2. 7d verdict cache keyed on sha1(content_hash + facets_hash). Repeat URLs
     across daily refreshes hit ~95% cache rate.
  3. Batched Haiku call (≤25 candidates per call) with tool-use response shape
     so we get a hard JSON guarantee (no regex recovery).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations import job_cost_logger as costs
from app.services.integrations.job_search_service import JobHit

logger = logging.getLogger(__name__)

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class JobFacets:
    keywords: List[str]
    excluded_keywords: List[str]
    location: Optional[str]
    remote_only: bool
    seniority: Optional[str]
    excluded_companies: List[str]
    preferred_companies: List[str]

    def hash(self) -> str:
        h = hashlib.sha1()
        h.update("|".join(sorted(self.keywords)).encode())
        h.update(b"||")
        h.update("|".join(sorted(self.excluded_keywords or [])).encode())
        h.update(b"||")
        h.update((self.location or "").lower().encode())
        h.update(b"||")
        h.update(b"R" if self.remote_only else b"N")
        h.update(b"||")
        h.update((self.seniority or "any").encode())
        h.update(b"||")
        h.update("|".join(sorted(self.excluded_companies or [])).encode())
        return h.hexdigest()


def _normalize(text: str) -> str:
    return (text or "").lower()


def rule_shortcut(facets: JobFacets, hit: JobHit) -> Optional[Tuple[str, str]]:
    """Deterministic verdict before Haiku. Returns (relevance, note) or None."""
    blob = " ".join(filter(None, [
        _normalize(hit.title), _normalize(hit.description_excerpt or ""),
        _normalize(hit.company or ""), _normalize(hit.location or ""),
    ]))
    if not blob.strip():
        return ("unverifiable", "no readable content from source")

    # excluded company → mismatch
    co_norm = _normalize(hit.company or "")
    for ex in facets.excluded_companies or []:
        if ex and _normalize(ex) in co_norm:
            return ("mismatch", f"excluded company: {ex}")

    # excluded keyword in title → mismatch
    title_norm = _normalize(hit.title or "")
    for ex in facets.excluded_keywords or []:
        if ex and _normalize(ex) in title_norm:
            return ("mismatch", f"excluded term: {ex}")

    # remote_only enforcement
    if facets.remote_only:
        loc = _normalize(hit.location or "")
        if hit.is_remote is False:
            return ("mismatch", "non-remote when remote_only=true")
        if hit.is_remote is None and "remote" not in loc and "remote" not in blob:
            # not certain — let Haiku decide
            pass

    # at least one keyword token in title or description → likely match
    keyword_hits = sum(1 for k in facets.keywords if k and _normalize(k) in blob)
    if keyword_hits == 0:
        return ("mismatch", "no keyword found in title/description")

    # All keywords appear AND title clearly matches → fast-path to match
    if keyword_hits == len(facets.keywords):
        first_kw_in_title = any(_normalize(k) in title_norm for k in facets.keywords if k)
        if first_kw_in_title:
            return ("match", "all keywords + title hit")

    return None  # ambiguous → Haiku


# ─── Verdict cache (7d TTL via DB column) ──────────────────────────────────

def _cache_key(content_hash: str, facets_hash: str) -> str:
    h = hashlib.sha1()
    h.update(content_hash.encode())
    h.update(b"|")
    h.update(facets_hash.encode())
    return h.hexdigest()


def _get_cached(content_hash: str, facets_hash: str) -> Optional[Dict[str, Any]]:
    try:
        sb = get_supabase_client().client
        key = _cache_key(content_hash, facets_hash)
        res = (
            sb.table("job_classifier_verdict_cache")
            .select("relevance, relevance_score, match_note, expires_at")
            .eq("cache_key", key)
            .maybe_single()
            .execute()
        )
        row = (res.data if res else None) or None
        if not row:
            return None
        # expires_at is server-side; the DB also runs a daily prune. Trust it.
        return row
    except Exception as e:
        logger.debug(f"job-classifier cache read failed: {e}")
        return None


def _put_cached(content_hash: str, facets_hash: str, verdict: Dict[str, Any]) -> None:
    try:
        sb = get_supabase_client().client
        sb.table("job_classifier_verdict_cache").upsert({
            "cache_key": _cache_key(content_hash, facets_hash),
            "relevance": verdict.get("relevance"),
            "relevance_score": verdict.get("relevance_score"),
            "match_note": (verdict.get("match_note") or "")[:240] or None,
        }, on_conflict="cache_key").execute()
    except Exception as e:
        logger.debug(f"job-classifier cache write failed: {e}")


# ─── Haiku batched classifier via Anthropic tool use ───────────────────────

_CLASSIFY_TOOL = {
    "name": "submit_classifications",
    "description": "Return one verdict per input job listing.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "relevance": {"type": "string", "enum": ["match", "tangential", "mismatch", "unverifiable"]},
                        "relevance_score": {"type": "number"},
                        "match_note": {"type": "string"},
                    },
                    "required": ["index", "relevance"],
                },
            },
        },
        "required": ["verdicts"],
    },
}


def _load_recent_corrections(tracked_job_id: Optional[str], limit: int = 5) -> List[Dict[str, Any]]:
    """v0.3: pull the most recent classifier corrections the user submitted for
    THIS tracked_job. Used as Haiku few-shot examples on the next refresh."""
    if not tracked_job_id:
        return []
    try:
        sb = get_supabase_client().client
        res = (
            sb.table("job_match_corrections")
            .select("original_relevance, corrected_relevance, reason, "
                    "job_listings(title, company)")
            .eq("tracked_job_id", tracked_job_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.debug(f"job-classifier: load corrections failed: {e}")
        return []


def _build_user_prompt(facets: JobFacets, batch: List[JobHit], corrections: Optional[List[Dict[str, Any]]] = None) -> str:
    facets_block = (
        f"USER IS LOOKING FOR:\n"
        f"  Keywords (any of): {', '.join(facets.keywords)}\n"
        f"  Excluded keywords: {', '.join(facets.excluded_keywords) or '(none)'}\n"
        f"  Location: {facets.location or '(any)'}\n"
        f"  Remote-only: {facets.remote_only}\n"
        f"  Seniority: {facets.seniority or 'any'}\n"
        f"  Excluded companies: {', '.join(facets.excluded_companies) or '(none)'}\n"
    )

    # v0.3: prepend the most recent few user corrections as concrete examples.
    # This is how the system "learns" the user's idiosyncratic preferences without
    # any model retraining — Haiku just sees "here's what the user said was wrong
    # last time, mirror that judgment."
    few_shot_block = ""
    if corrections:
        examples = []
        for c in corrections[:5]:
            inner = (c.get("job_listings") or {})
            t = inner.get("title") or ""
            co = inner.get("company") or ""
            orig = c.get("original_relevance") or "?"
            corr = c.get("corrected_relevance") or "?"
            reason = (c.get("reason") or "")[:140]
            examples.append(
                f"  - We classified '{t}' at '{co}' as {orig}; user corrected to {corr}"
                + (f" (reason: {reason})" if reason else "")
            )
        few_shot_block = (
            "RECENT USER CORRECTIONS (mirror this judgment style):\n"
            + "\n".join(examples) + "\n\n"
        )

    items = []
    for i, h in enumerate(batch):
        items.append(
            f"[{i}] title='{h.title or ''}' company='{h.company or ''}' "
            f"location='{h.location or ''}' "
            f"excerpt='{(h.description_excerpt or '')[:300]}'"
        )
    return (
        f"{facets_block}\n"
        f"{few_shot_block}"
        f"Classify each listing below. 'match' = clearly hits the user's intent. "
        f"'tangential' = same field but wrong specialization. "
        f"'mismatch' = entirely different role or excluded. "
        f"'unverifiable' = not enough signal to decide.\n\n"
        + "\n".join(items)
        + "\n\nCall submit_classifications with one verdict per listing."
    )


async def classify_batch(
    facets: JobFacets,
    hits: List[JobHit],
    *,
    attribution: costs.CostAttribution,
) -> List[Dict[str, Any]]:
    """Returns a list parallel to `hits`, each {relevance, relevance_score, match_note, classifier_cached}."""
    out: List[Dict[str, Any]] = [None] * len(hits)  # type: ignore
    facets_hash = facets.hash()
    to_call: List[int] = []

    # 1. Rule shortcut + cache lookup
    for i, h in enumerate(hits):
        shortcut = rule_shortcut(facets, h)
        if shortcut:
            out[i] = {
                "relevance": shortcut[0],
                "relevance_score": 1.0 if shortcut[0] == "match" else 0.0,
                "match_note": shortcut[1],
                "classifier_cached": False,
            }
            continue
        cached = _get_cached(h.content_hash, facets_hash)
        if cached:
            out[i] = {
                "relevance": cached.get("relevance"),
                "relevance_score": cached.get("relevance_score"),
                "match_note": cached.get("match_note"),
                "classifier_cached": True,
            }
            continue
        to_call.append(i)

    if not to_call:
        return out

    api_key = os.getenv("ANTHROPIC_API_KEY") or ""
    if not api_key:
        logger.warning("job-classifier: ANTHROPIC_API_KEY not configured — marking remaining as unverifiable")
        for i in to_call:
            out[i] = {"relevance": "unverifiable", "relevance_score": 0.0, "match_note": "classifier disabled", "classifier_cached": False}
        return out

    # 2. Batched Haiku tool-use via HTTP API (bypasses SDK version drama —
    #    deployed anthropic-python is pinned to 0.23.1 (beta-era), HTTP API has
    #    had stable tool_use since 2024-06).
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    BATCH = 25
    # v0.3: load few-shot corrections once per refresh (not per batch).
    corrections = _load_recent_corrections(getattr(attribution, "tracked_job_id", None))
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as http:
        for chunk_start in range(0, len(to_call), BATCH):
            chunk_indices = to_call[chunk_start:chunk_start + BATCH]
            batch_hits = [hits[i] for i in chunk_indices]
            started = time.time()
            in_tok = 0
            out_tok = 0
            success = True
            err: Optional[str] = None
            try:
                payload = {
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 2000,
                    "tools": [_CLASSIFY_TOOL],
                    "tool_choice": {"type": "tool", "name": "submit_classifications"},
                    "messages": [{"role": "user", "content": _build_user_prompt(facets, batch_hits, corrections)}],
                }
                resp = await http.post(_ANTHROPIC_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                in_tok = int(usage.get("input_tokens") or 0)
                out_tok = int(usage.get("output_tokens") or 0)
                verdicts: List[Dict[str, Any]] = []
                for block in (data.get("content") or []):
                    if block.get("type") == "tool_use" and block.get("name") == "submit_classifications":
                        verdicts = (block.get("input") or {}).get("verdicts", []) or []
                        break
                by_idx = {int(v.get("index", -1)): v for v in verdicts}
                for local_idx, global_idx in enumerate(chunk_indices):
                    v = by_idx.get(local_idx)
                    if not v:
                        out[global_idx] = {"relevance": "unverifiable", "relevance_score": 0.0, "match_note": "no verdict returned", "classifier_cached": False}
                        continue
                    verdict = {
                        "relevance": v.get("relevance") or "unverifiable",
                        "relevance_score": float(v.get("relevance_score") or 0.0),
                        "match_note": (v.get("match_note") or "")[:240] or None,
                        "classifier_cached": False,
                    }
                    out[global_idx] = verdict
                    _put_cached(hits[global_idx].content_hash, facets_hash, verdict)
            except Exception as e:
                success = False
                err = str(e)[:200]
                logger.warning(f"job-classifier haiku batch failed: {err}")
                for global_idx in chunk_indices:
                    out[global_idx] = {"relevance": "unverifiable", "relevance_score": 0.0, "match_note": f"classifier error: {err}", "classifier_cached": False}

            costs.log_haiku_call(
                attribution=attribution,
                operation="classifier",
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=int((time.time() - started) * 1000),
                success=success,
                error_message=err,
            )

    # Defensive fill
    for i, v in enumerate(out):
        if v is None:
            out[i] = {"relevance": "unverifiable", "relevance_score": 0.0, "match_note": None, "classifier_cached": False}
    return out
