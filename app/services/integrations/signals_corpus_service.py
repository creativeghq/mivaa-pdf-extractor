"""Signals corpus (#462): find each company's job board, then read every board on a due-gate.

Market-wide, not subject-scoped: nothing here belongs to a tracked search or a workspace.
Every fetch goes through the shared SSRF guard; board tokens come from scraped HTML.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.ats_boards import (
    ENUMERABLE,
    AtsShapeError,
    board_url,
    count_anchors,
    find_ats_accounts,
    find_careers_links,
    parse_board,
)
from app.services.integrations.tech_fingerprints import detect as detect_technologies
from app.utils.ssrf_guard import ResponseTooLarge, SSRFError, safe_fetch_bytes

logger = logging.getLogger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (compatible; MaterialKAI-Signals/1.0; +https://materialshub.gr)"}
_PAGE_MAX_BYTES = 3 * 1024 * 1024
_BOARD_MAX_BYTES = 32 * 1024 * 1024
_MAX_POSTINGS_PER_BOARD = 3000
_MIN_ANCHORS_TO_TRUST_A_NONE = 5
_DOMAIN_BUDGET_S = 35.0


def _sb():
    return get_supabase_client().client


async def _rpc(fn: str, params: Dict[str, Any]) -> Any:
    return await asyncio.to_thread(lambda: _sb().rpc(fn, params).execute().data)


async def _gather_within(coros: List[Any], budget_s: float) -> Tuple[List[Any], int]:
    """Run to a wall-clock budget. Unfinished work keeps its 1h claim lease and is retried."""
    if not coros:
        return [], 0
    tasks = [asyncio.ensure_future(c) for c in coros]
    done, pending = await asyncio.wait(tasks, timeout=budget_s)
    for t in pending:
        t.cancel()
    results = [t.exception() or t.result() for t in done if not t.cancelled()]
    return results, len(pending)


async def _get_page(client: httpx.AsyncClient, url: str) -> Tuple[Optional[str], str, int]:
    """(text or None when not 2xx, final url after redirects, status)."""
    res = await safe_fetch_bytes(url, max_bytes=_PAGE_MAX_BYTES, timeout=15.0,
                                 allow_schemes=("https", "http"), headers=_UA, client=client)
    text = res.content.decode("utf-8", errors="replace") if res.ok else None
    return text, res.final_url, res.status_code


async def resolve_domain(client: httpx.AsyncClient, domain: str,
                         fingerprints: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Read the homepage and up to two careers pages. Returns the record_company_ats_check args,
    plus `technologies` (slugs the homepage gives away) and `home_url`, which are not RPC args."""
    home = None
    last_error = None
    for url in (f"https://{domain}/", f"https://www.{domain}/"):
        try:
            home, base, status = await _get_page(client, url)
            if home is not None:
                break
            last_error = f"HTTP {status} at {base[:160]}"
        except (SSRFError, httpx.HTTPError) as e:
            last_error = f"{type(e).__name__}: {str(e)[:160]}"
    if home is None:
        return {"p_status": "unreachable", "p_careers_url": None, "p_error": last_error, "p_accounts": []}

    technologies = detect_technologies(home, fingerprints or [])
    accounts = find_ats_accounts(home)
    careers = find_careers_links(home, base)
    for url in careers[:2]:
        try:
            page, _, _ = await _get_page(client, url)
        except (SSRFError, httpx.HTTPError) as e:
            logger.info(f"signals resolve: careers page {url[:120]} failed: {e}")
            continue
        for acc in find_ats_accounts(page or ""):
            if acc not in accounts:
                accounts.append(acc)

    if accounts:
        status = "found"
    elif careers or count_anchors(home) >= _MIN_ANCHORS_TO_TRUST_A_NONE:
        status = "none"
    else:
        status = "unreadable"
    return {
        "p_status": status,
        "p_careers_url": careers[0] if careers else None,
        "p_error": None if status != "unreadable" else "homepage carried almost no links (JS-rendered?)",
        "p_accounts": [{"provider": p, "token": t} for p, t in accounts],
        "technologies": technologies,
        "home_url": base,
    }


async def run_resolution(limit: int = 8, budget_s: float = 45.0) -> Dict[str, Any]:
    due = await _rpc("claim_company_ats_checks", {"p_limit": limit}) or []
    fingerprints = (await _rpc("get_technology_fingerprints", {}) or []) if due else []
    tally: Dict[str, int] = {}
    accounts_written = 0
    technologies_written = 0
    sem = asyncio.Semaphore(8)

    async with httpx.AsyncClient(timeout=15.0) as client:
        async def one(row: Dict[str, Any]) -> None:
            nonlocal accounts_written, technologies_written
            domain = row["domain"]
            async with sem:
                try:
                    args = await asyncio.wait_for(resolve_domain(client, domain, fingerprints), _DOMAIN_BUDGET_S)
                except asyncio.TimeoutError:
                    args = {"p_status": "unreachable", "p_careers_url": None,
                            "p_error": f"timed out after {_DOMAIN_BUDGET_S:.0f}s", "p_accounts": []}
                except Exception as e:  # noqa: BLE001 — recorded on the row, never swallowed
                    args = {"p_status": "unreachable", "p_careers_url": None,
                            "p_error": f"{type(e).__name__}: {str(e)[:160]}", "p_accounts": []}
            technologies = args.pop("technologies", None) or []
            home_url = args.pop("home_url", None)
            n = await _rpc("record_company_ats_check", {"p_domain": domain, **args})
            accounts_written += int(n or 0)
            if technologies:
                t = await _rpc("record_company_technologies", {
                    "p_domain": domain, "p_slugs": technologies, "p_evidence_url": home_url})
                technologies_written += int(t or 0)
            tally[args["p_status"]] = tally.get(args["p_status"], 0) + 1

        results, timed_out = await _gather_within([one(r) for r in due], budget_s)
    errors = [f"{type(r).__name__}: {str(r)[:160]}" for r in results if isinstance(r, BaseException)]
    for e in errors:
        logger.error(f"signals resolve: recording a check failed: {e}")
    return {"claimed": len(due), "by_status": tally, "accounts_written": accounts_written,
            "technologies_written": technologies_written, "fingerprints": len(fingerprints),
            "timed_out": timed_out, "record_errors": errors}


async def enumerate_account(client: httpx.AsyncClient, provider: str, token: str,
                            etag: Optional[str], lock: asyncio.Semaphore) -> Dict[str, Any]:
    if provider not in ENUMERABLE:
        raise ValueError(f"{provider} is not enumerable")
    url = board_url(provider, token)
    headers = dict(_UA)
    headers["Accept"] = "application/rss+xml" if provider == "teamtailor" else "application/json"
    if etag:
        headers["If-None-Match"] = etag
    async with lock:
        res = await safe_fetch_bytes(url, max_bytes=_BOARD_MAX_BYTES, timeout=45.0,
                                     headers=headers, client=client)
    if res.status_code == 304:
        return await _rpc("ingest_market_job_postings", {
            "p_provider": provider, "p_token": token, "p_company_name": None,
            "p_postings": None, "p_etag": None, "p_not_modified": True})
    if not res.ok:
        raise _BoardHttpError(res.status_code)
    company, postings = parse_board(provider, res.content)
    if len(postings) > _MAX_POSTINGS_PER_BOARD:
        logger.warning(f"signals enumerate {provider}:{token}: {len(postings)} postings, "
                       f"kept the first {_MAX_POSTINGS_PER_BOARD}")
        postings = postings[:_MAX_POSTINGS_PER_BOARD]
    return await _rpc("ingest_market_job_postings", {
        "p_provider": provider, "p_token": token, "p_company_name": company,
        "p_postings": postings, "p_etag": res.etag or None, "p_not_modified": False})


class _BoardHttpError(Exception):
    def __init__(self, status: int):
        super().__init__(f"board returned HTTP {status}")
        self.status = status


async def run_enumeration(limit: int = 10, budget_s: float = 75.0) -> Dict[str, Any]:
    due = await _rpc("claim_ats_accounts_due", {"p_limit": limit}) or []
    outcomes: List[Dict[str, Any]] = []
    # One request in flight per provider: these are embed APIs, not bulk feeds. Built per
    # run so no semaphore outlives the event loop it was bound to.
    locks = {p: asyncio.Semaphore(3 if p == "teamtailor" else 1) for p in ENUMERABLE}

    async with httpx.AsyncClient(timeout=45.0) as client:
        async def one(row: Dict[str, Any]) -> None:
            provider, token = row["provider"], row["token"]
            try:
                result = await enumerate_account(client, provider, token, row.get("etag"),
                                                 locks[provider])
                outcomes.append({"board": f"{provider}:{token}", **(result or {})})
                return
            except _BoardHttpError as e:
                code, err = e.status, str(e)
            except AtsShapeError as e:
                code, err = None, f"shape: {e}"
            except ResponseTooLarge as e:
                code, err = 413, f"too large: {str(e)[:200]}"
            except (SSRFError, httpx.HTTPError, ValueError) as e:
                code, err = None, f"{type(e).__name__}: {str(e)[:200]}"
            status = await _rpc("record_ats_enumeration_failure", {
                "p_provider": provider, "p_token": token, "p_status_code": code, "p_error": err})
            logger.warning(f"signals enumerate {provider}:{token} failed ({err}); account now {status}")
            outcomes.append({"board": f"{provider}:{token}", "error": err, "account_status": status})

        results, timed_out = await _gather_within([one(r) for r in due], budget_s)
    errors = [f"{type(r).__name__}: {str(r)[:160]}" for r in results if isinstance(r, BaseException)]
    for e in errors:
        logger.error(f"signals enumerate: recording an outcome failed: {e}")
    return {
        "claimed": len(due),
        "timed_out": timed_out,
        "record_errors": errors,
        "failed": sum(1 for o in outcomes if "error" in o),
        "new_postings": sum(int(o.get("new") or 0) for o in outcomes),
        "closed_postings": sum(int(o.get("closed") or 0) for o in outcomes),
        "outcomes": outcomes,
    }
