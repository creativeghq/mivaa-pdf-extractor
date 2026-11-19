"""Described signals (#499): a plain-language signal compiled into rules, matched by a prefilter + AI judge.

Every model call is schema-constrained (invariant 9); the item being judged is fenced as data; a judgement
row is CLAIMED before the credit is charged, so a crash never re-charges; a match must quote the item.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.services.core.claude_models import HAIKU, SONNET
from app.services.integrations.described_signals_rules import (
    COMPILE_SCHEMA,
    JUDGE_SCHEMA,
    SOURCES,
    clean_spec,
    quote_is_in_item,
)
from app.services.core.claude_tool_call import ToolCallNotReturned, call_with_schema
from app.services.core.supabase_client import get_supabase_client
from app.services.utilities.prompt_registry import load_prompt, render
from app.utils.untrusted_content import as_untrusted_data

logger = logging.getLogger(__name__)

TASK_COMPILE = "signals.described.compile"
TASK_JUDGE = "signals.described.judge"
JUDGE_CRON_KEY = "signals-described-judge"

BACKTEST_DAYS = 30
BACKTEST_JUDGE_SAMPLE = 10
PER_SIGNAL_RUN_CAP = 25
EMBED_BATCH = 64


def _sb():
    return get_supabase_client().client


async def _rpc(fn: str, params: Dict[str, Any]) -> Any:
    return await asyncio.to_thread(lambda: _sb().rpc(fn, params).execute().data)


def _vec_literal(emb: List[float]) -> str:
    return "[" + ",".join(f"{float(x):.5g}" for x in emb) + "]"


async def _embed(text: str, *, workspace_id: Optional[str], user_id: Optional[str], input_type: str) -> Optional[List[float]]:
    from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

    res = await RealEmbeddingsService().generate_text_embedding(
        text[:8000], workspace_id=workspace_id, user_id=user_id, input_type=input_type)
    return res.get("embedding") if res.get("success") else None


async def compile_description(description: str, *, workspace_id: str, user_id: str) -> Dict[str, Any]:
    """Draft rules from a description. Raises ToolCallNotReturned/ValueError — never a guessed spec."""
    prompt = await load_prompt("classification", "signals_describe_compile", stage="signals")
    system = await load_prompt("classification", "signals_describe_compile", stage="signals", field="system_prompt")
    sources = "\n".join(f"- {k}: {v}" for k, v in SOURCES.items())
    result = await call_with_schema(
        task=TASK_COMPILE, model=SONNET, max_tokens=2000, system=system,
        workspace_id=workspace_id, user_id=user_id,
        messages=[{"role": "user", "content": render(prompt, description=description.strip(), sources=sources)}],
        output_config={"format": {"type": "json_schema", "schema": COMPILE_SCHEMA}},
    )
    data = result.data or {}
    return {"name": str(data.get("name") or "").strip()[:80], "spec": clean_spec(data)}


def _item_text(row: Dict[str, Any]) -> str:
    return f"Title: {row.get('title') or ''}\nURL: {row.get('url') or ''}\n\n{row.get('body_md') or ''}"


async def judge_item(spec: Dict[str, Any], row: Dict[str, Any], *, workspace_id: str, user_id: Optional[str]) -> Dict[str, Any]:
    """{status: match|no_match|failed, evidence_quote, reason, confidence, error}. A refusal is 'failed'."""
    try:
        prompt = await load_prompt("classification", "signals_describe_judge", stage="signals")
        system = await load_prompt("classification", "signals_describe_judge", stage="signals", field="system_prompt")
        item_text = _item_text(row)
        bullets = lambda xs: "\n".join(f"- {x}" for x in xs) or "- (none given)"  # noqa: E731
        result = await call_with_schema(
            task=TASK_JUDGE, model=HAIKU, max_tokens=500, system=system,
            workspace_id=workspace_id, user_id=user_id,
            messages=[{"role": "user", "content": render(
                prompt,
                judge_question=spec.get("judge_question") or "",
                positive_examples=bullets(spec.get("positive_examples") or []),
                negative_examples=bullets(spec.get("negative_examples") or []),
                item=as_untrusted_data(item_text, source=f"signal observation ({row.get('source')})"),
            )}],
            output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
    except (ToolCallNotReturned, Exception) as e:  # noqa: BLE001 — a failed verdict is recorded, never read as no-match
        return {"status": "failed", "error": f"{type(e).__name__}: {str(e)[:240]}"}

    data = result.data or {}
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        confidence = None
    quote = str(data.get("evidence_quote") or "").strip()[:300]
    reason = str(data.get("reason") or "").strip()[:400]
    if data.get("match") is True:
        if not quote_is_in_item(quote, item_text):
            return {"status": "failed", "error": "evidence_quote is not a passage of the item", "reason": reason}
        return {"status": "match", "evidence_quote": quote, "reason": reason, "confidence": confidence}
    return {"status": "no_match", "reason": reason, "confidence": confidence}


async def _candidates(workspace_id: str, signal_id: Optional[str], spec: Dict[str, Any], qvec: Optional[str],
                      crm_only: bool, days: int, limit: int) -> List[Dict[str, Any]]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return await _rpc("signals_described_candidates", {
        "p_workspace_id": workspace_id, "p_signal_id": signal_id, "p_spec": spec,
        "p_query_embedding": qvec, "p_crm_only": crm_only, "p_since": since, "p_limit": limit,
    }) or []


async def backtest(description: str, spec: Dict[str, Any], *, crm_only: bool, workspace_id: str,
                   user_id: str) -> Dict[str, Any]:
    """Free: prefilter over 30 days, judge a small sample, estimate the weekly judge volume."""
    spec = clean_spec(spec)
    emb = await _embed(description, workspace_id=workspace_id, user_id=user_id, input_type="query")
    qvec = _vec_literal(emb) if emb else None
    rows = await _candidates(workspace_id, None, spec, qvec, crm_only, BACKTEST_DAYS, 500)
    sample = rows[:BACKTEST_JUDGE_SAMPLE]
    verdicts = await asyncio.gather(*(judge_item(spec, r, workspace_id=workspace_id, user_id=user_id) for r in sample))
    judged = []
    for r, v in zip(sample, verdicts):
        judged.append({
            "title": r.get("title"), "url": r.get("url"), "source": r.get("source"),
            "observed_at": r.get("observed_at"), "company_domain": r.get("company_domain"),
            "keyword_hits": r.get("keyword_hits"), "similarity": r.get("similarity"), **v,
        })
    per_week = round(len(rows) / BACKTEST_DAYS * 7)
    return {
        "candidates_30d": len(rows),
        "candidates_capped": len(rows) >= 500,
        "embedding_used": qvec is not None,
        "judged": judged,
        "matches_in_sample": sum(1 for j in judged if j["status"] == "match"),
        "failed_in_sample": sum(1 for j in judged if j["status"] == "failed"),
        "estimate": {"items_per_week": per_week,
                     "credits_per_week": min(per_week, PER_SIGNAL_RUN_CAP * 24 * 7),
                     "credit_per_item": 1},
    }


async def _charge(workspace_id: str, user_id: Optional[str], subject: Dict[str, Any]) -> str:
    """'ok' | 'no_credits' | 'error'. Distinguishes an empty wallet (pause) from a transport fault (retry)."""
    try:
        data = await _rpc("cron_charge_workspace", {
            "p_workspace_id": workspace_id, "p_cron_key": JUDGE_CRON_KEY, "p_units": 1,
            "p_description": "Described signal: AI check of one item", "p_subject": subject,
        })
    except Exception as e:  # noqa: BLE001
        logger.warning("described signals: charge failed (%s); skipping, not pausing", e)
        return "error"
    row = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
    if not row:
        return "error"
    return "ok" if row.get("allowed", False) else "no_credits"


async def _refund(workspace_id: str, reason: str) -> bool:
    try:
        await _rpc("cron_refund_workspace", {
            "p_workspace_id": workspace_id, "p_cron_key": JUDGE_CRON_KEY, "p_amount": 1,
            "p_description": f"Described signal: refund, {reason}"[:200]})
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("described signals: refund failed (%s)", e)
        return False


async def embed_pending_observations(budget_s: float) -> Dict[str, int]:
    started = time.monotonic()
    done = failed = 0
    while time.monotonic() - started < budget_s:
        batch = await _rpc("claim_observations_to_embed", {"p_limit": EMBED_BATCH}) or []
        if not batch:
            break
        updates = []
        for row in batch:
            emb = await _embed(row["text_to_embed"], workspace_id=None, user_id=None, input_type="document")
            if emb:
                updates.append({"id": row["id"], "embedding": _vec_literal(emb), "error": None})
                done += 1
            else:
                updates.append({"id": row["id"], "embedding": None, "error": "embedding failed"})
                failed += 1
        await _rpc("set_observation_embeddings", {"p_rows": updates})
    return {"embedded": done, "embedding_failed": failed}


async def _run_one(sig: Dict[str, Any], deadline: float) -> Dict[str, Any]:
    ws, uid, sid = sig["workspace_id"], sig.get("created_by"), sig["id"]
    filt = sig.get("filter") or {}
    spec = filt.get("spec") or {}
    if sig.get("needs_embedding"):
        emb = await _embed(filt.get("description") or "", workspace_id=ws, user_id=uid, input_type="query")
        if emb:
            await _rpc("set_signal_description_embedding", {"p_signal_id": sid, "p_embedding": _vec_literal(emb)})
    qvec = await _rpc("signals_described_embedding", {"p_signal_id": sid})
    rows = await _candidates(ws, sid, spec, qvec, bool(filt.get("crm_only")), BACKTEST_DAYS, PER_SIGNAL_RUN_CAP)

    new_matches = judged = failed = 0
    paused: Optional[str] = None
    for row in rows:
        if time.monotonic() > deadline:
            break
        jid = await _rpc("claim_described_judgement", {
            "p_signal_id": sid, "p_spec_hash": sig["spec_hash"], "p_observation_id": row["id"]})
        if not jid:
            continue
        charge = await _charge(ws, uid, {"signal_id": sid, "observation_id": row["id"]})
        if charge != "ok":
            await _rpc("finish_described_judgement", {
                "p_judgement_id": jid, "p_status": "skipped_no_credits" if charge == "no_credits" else "failed",
                "p_confidence": None, "p_evidence_quote": None, "p_reason": None,
                "p_error": "insufficient credits" if charge == "no_credits" else "credit charge failed", "p_credits": 0})
            if charge == "no_credits":
                paused = "no_credits"
            break
        verdict = await judge_item(spec, row, workspace_id=ws, user_id=uid)
        judged += 1
        credits = 1
        if verdict["status"] == "failed":
            failed += 1
            credits = 0 if await _refund(ws, "the AI check failed") else 1
        matched = await _rpc("finish_described_judgement", {
            "p_judgement_id": jid, "p_status": verdict["status"], "p_confidence": verdict.get("confidence"),
            "p_evidence_quote": verdict.get("evidence_quote"), "p_reason": verdict.get("reason"),
            "p_error": verdict.get("error"), "p_credits": credits})
        new_matches += 1 if matched else 0

    await _rpc("record_described_signal_run", {"p_signal_id": sid, "p_new_matches": new_matches, "p_paused_reason": paused})
    return {"signal_id": sid, "candidates": len(rows), "judged": judged, "failed": failed,
            "new_matches": new_matches, "paused": paused}


async def run_described(limit: int = 20, budget_s: float = 110.0) -> Dict[str, Any]:
    """Hourly: embed the observation queue, then check every due described signal, inside the edge's 150s."""
    started = time.monotonic()
    deadline = started + budget_s
    embedding = await embed_pending_observations(budget_s=min(30.0, budget_s / 3))
    due = await _rpc("signals_described_due", {"p_limit": limit}) or []
    outcomes: List[Dict[str, Any]] = []
    for sig in due:
        if time.monotonic() > deadline:
            break
        try:
            outcomes.append(await _run_one(sig, deadline))
        except Exception as e:  # noqa: BLE001 — one signal failing must not stop the others; recorded below
            logger.error("described signal %s run failed: %s", sig.get("id"), e)
            outcomes.append({"signal_id": sig.get("id"), "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return {"embedding": embedding, "due": len(due), "ran": len(outcomes),
            "new_matches": sum(int(o.get("new_matches") or 0) for o in outcomes),
            "errors": sum(1 for o in outcomes if "error" in o), "outcomes": outcomes}

