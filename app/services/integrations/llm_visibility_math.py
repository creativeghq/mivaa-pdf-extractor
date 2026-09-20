"""LLM-visibility derivations — pure functions over probe rows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

__all__ = [
    "citation_domain",
    "anthropic_search_tool_type",
    "anthropic_citation_urls",
    "anthropic_search_failure",
    "gemini_citation_urls",
    "domain_is_ours",
    "dedupe_urls",
    "sentiment_rollup",
    "citation_rollup",
    "group_by_run",
    "trend_from_rows",
    "share_of_voice_from_rows",
    "answered_rows",
    "visibility_rollup",
]


# ────────────────────────────────────────────────────────────────────────────
# Citations
# ────────────────────────────────────────────────────────────────────────────

def citation_domain(url: str) -> str:
    """Host of a cited URL, lowercased, `www.` stripped. Deliberately NOT a public-
    suffix parse — the AI Overview path compares bare hosts the same way."""
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        netloc = urlparse(raw).netloc or urlparse("//" + raw).netloc
    except Exception:
        return ""
    host = (netloc or "").split("@")[-1].split(":")[0].strip().lower()
    # The `//` retry above exists so a bare `example.com` parses, but it also happily
    # accepts any prose at all - `urlparse("//not a url").netloc` is "not a url".
    # A host has no whitespace and at least one dot.
    if not host or " " in host or "." not in host:
        return ""
    return host[4:] if host.startswith("www.") else host


#: Web search tool versions. Dynamic filtering is rejected by anything older than the
#: 4.6 generation, and the basic variant is what those take instead.
_SEARCH_TOOL_DYNAMIC = "web_search_20260209"
_SEARCH_TOOL_BASIC = "web_search_20250305"

#: Families that took dynamic filtering from 4.6 on. Haiku is absent deliberately:
#: 4.5 does not accept it, and a later Haiku has to be checked before it is assumed.
_DYNAMIC_SEARCH_FAMILIES = frozenset({"opus", "sonnet", "fable", "mythos"})
_DYNAMIC_SEARCH_MIN = (4, 6)


def _model_generation(model: str) -> Tuple[Optional[str], Tuple[int, int]]:
    """(family, (major, minor)) from a Claude id. `claude-haiku-4-5-20251001` is
    (haiku, (4, 5)) — the trailing date is not a version part."""
    parts = (model or "").strip().lower().split("-")
    if len(parts) < 3 or parts[0] != "claude":
        return None, (0, 0)
    family = parts[1]
    nums: List[int] = []
    for p in parts[2:]:
        if not p.isdigit() or len(p) > 2:
            break
        nums.append(int(p))
    if not nums:
        return family, (0, 0)
    return family, (nums[0], nums[1] if len(nums) > 1 else 0)


def anthropic_search_tool_type(model: str) -> str:
    """The `web_search` tool version `model` accepts.

    Derived from the generation rather than a list of ids: a list goes stale the day a
    model ships, and the failure is a 400 recorded as a provider error rather than an
    answer. An id we cannot parse gets the basic variant, which every model accepts.
    """
    family, version = _model_generation(model)
    if family in _DYNAMIC_SEARCH_FAMILIES and version >= _DYNAMIC_SEARCH_MIN:
        return _SEARCH_TOOL_DYNAMIC
    return _SEARCH_TOOL_BASIC


def anthropic_citation_urls(blocks: Any) -> List[str]:
    """Sources behind a Claude answer, from the Messages API content blocks.

    Read from two places because they answer different questions: a
    `web_search_tool_result` is every page the model READ, and a `citations` entry on
    a text block is a page it actually LEANED ON. A failed search returns the same
    block type with `content` as an error OBJECT rather than a list, so an
    unconditional iteration silently walks the error's keys instead.
    """
    urls: List[str] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "web_search_tool_result":
            results = block.get("content")
            if not isinstance(results, list):
                continue
            for r in results:
                if isinstance(r, dict) and r.get("url"):
                    urls.append(str(r["url"]))
        elif kind == "text":
            for c in block.get("citations") or []:
                if isinstance(c, dict) and c.get("url"):
                    urls.append(str(c["url"]))
    return urls


def anthropic_search_failure(blocks: Any) -> Optional[str]:
    """The reason the web search produced nothing, or None when it worked.

    A `web_search_tool_result` whose `content` is an error OBJECT rather than a list is
    a search that did not happen. The model still answers - from memory - so the call
    looks successful and the probe records "answered, no sources", which is precisely
    the unknown-rendered-as-zero shape. Only reported when NO search succeeded:
    `max_uses_exceeded` after two good searches is a stop, not a failure.
    """
    errors: List[str] = []
    succeeded = False
    for block in blocks or []:
        if not isinstance(block, dict) or block.get("type") != "web_search_tool_result":
            continue
        content = block.get("content")
        if isinstance(content, list):
            succeeded = True
        elif isinstance(content, dict):
            errors.append(str(content.get("error_code") or content.get("type") or "unknown"))
    if succeeded or not errors:
        return None
    return f"web_search failed: {', '.join(sorted(set(errors)))}"


def gemini_citation_urls(candidate: Any) -> List[str]:
    """Sources behind a grounded Gemini answer.

    `groundingChunks[].web.uri` is a Google REDIRECT, not the publisher — storing it
    makes every citation read as vertexaisearch.cloud.google.com and no answer can
    ever cite the subject's own domain. The publisher is in `.title`, which grounding
    returns as a bare host; the redirect is kept only when it is not.
    """
    if not isinstance(candidate, dict):
        return []
    meta = candidate.get("groundingMetadata")
    if not isinstance(meta, dict):
        return []
    urls: List[str] = []
    for chunk in meta.get("groundingChunks") or []:
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if not isinstance(web, dict):
            continue
        host = citation_domain(str(web.get("title") or ""))
        urls.append(f"https://{host}" if host else str(web.get("uri") or ""))
    return urls


def domain_is_ours(cited_url: str, homepage_domain: Optional[str]) -> bool:
    """True when a cited URL sits on the subject's own domain or a subdomain of it.

    `blog.brand.com` is the brand. `notbrand.com` is not, and neither is
    `brand.com.evil.net` — which a substring test (the obvious implementation) gets
    wrong in the direction that invents citations we never earned.
    """
    home = (homepage_domain or "").strip().lower()
    if home.startswith("www."):
        home = home[4:]
    cited = citation_domain(cited_url)
    if not cited or not home:
        return False
    return cited == home or cited.endswith("." + home)


def dedupe_urls(urls: List[str], *, limit: int = 20) -> List[str]:
    """Order-preserving dedupe. Native and extracted citations overlap constantly."""
    seen: set = set()
    out: List[str] = []
    for u in urls or []:
        v = (u or "").strip()
        if not v or not v.lower().startswith(("http://", "https://")):
            continue
        key = v.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v[:600])
        if len(out) >= limit:
            break
    return out


# ────────────────────────────────────────────────────────────────────────────
# Rollups
# ────────────────────────────────────────────────────────────────────────────

def sentiment_rollup(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sentiment over the probes where the subject was ACTUALLY MENTIONED.

    Rolling every row in would average in the extractor's `neutral` default for
    answers that never named the subject, which drags every score toward neutral in
    proportion to how invisible the brand is — the exact opposite of the signal.
    """
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in rows:
        if not r.get("mentioned"):
            continue
        key = r.get("sentiment") or "neutral"
        if key in counts:
            counts[key] += 1
    total = sum(counts.values())
    return {
        **counts,
        "basis_probes": total,
        # -1.0 (all negative) .. +1.0 (all positive). None when never mentioned —
        # "no opinion recorded" is not the same fact as "the opinion was neutral".
        "score": ((counts["positive"] - counts["negative"]) / total) if total else None,
    }


def citation_rollup(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Citation counts, cited domains, and the ghost-citation tally."""
    with_citations = brand_cited = ghost = undecidable = 0
    domains: Dict[str, int] = {}
    for r in rows:
        urls = r.get("cited_urls") or []
        if not urls:
            continue
        with_citations += 1
        for u in urls:
            d = citation_domain(u)
            if d:
                domains[d] = domains.get(d, 0) + 1
        flag = r.get("brand_cited")
        if flag is None:
            # homepage_domain was never configured, so nothing here can be judged.
            undecidable += 1
        elif flag:
            brand_cited += 1
            if not r.get("mentioned"):
                ghost += 1
    return {
        "probes_with_citations": with_citations,
        "brand_cited": brand_cited,
        # Our page was the SOURCE and the brand was never named in the answer.
        "ghost_citations": ghost,
        # Surfaced rather than folded into 0 so the UI can say WHY it is zero.
        "undecidable_no_homepage_domain": undecidable,
        "top_cited_domains": sorted(domains.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }


def position_rollup(rows: List[Dict[str, Any]]) -> Tuple[List[int], Optional[float]]:
    """Ranks the subject actually held, and their mean. Empty → None, never 0."""
    positions = [int(r["position"]) for r in rows if r.get("mentioned") and r.get("position")]
    return positions, (sum(positions) / len(positions)) if positions else None


def answered_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Probes that actually came back. A row with an `error` asked nothing."""
    return [r for r in rows if not r.get("error")]


def visibility_rollup(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mention counts and share of voice over a set of probe rows."""
    answered = answered_rows(rows)
    mentioned = sum(1 for r in answered if r.get("mentioned"))
    _, avg_position = position_rollup(answered)
    return {
        "probes": len(rows),
        "answered": len(answered),
        "failed": len(rows) - len(answered),
        "mentioned": mentioned,
        "share_of_voice": (mentioned / len(answered)) if answered else None,
        "avg_position": avg_position,
        # A first error is worth surfacing verbatim — "HTTP 429" tells the reader
        # what to fix, where "no verdict" alone does not.
        "sample_error": next((str(r.get("error")) for r in rows if r.get("error")), None),
    }


# ────────────────────────────────────────────────────────────────────────────
# Windowed series
# ────────────────────────────────────────────────────────────────────────────

def group_by_run(rows: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Probe rows bucketed by run, oldest run first.

    The run IS the measurement bucket — every row in one carries the same template
    set against the same models, so runs are comparable to each other in a way an
    arbitrary calendar bucket is not.
    """
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(row.get("probe_run_id") or "", []).append(row)
    return sorted(
        by_run.items(),
        key=lambda kv: min((r.get("run_at") or "") for r in kv[1]),
    )


def trend_from_rows(
    rows: List[Dict[str, Any]], *, days: int, truncated: bool = False,
) -> Dict[str, Any]:
    """Share of voice, average rank, sentiment and citations PER RUN over a window.

    The history was always in `llm_mention_probes`; nothing read across runs, so the
    product could say where the brand stands today and never whether that is better
    or worse than last week.
    """
    if not rows:
        return {"present": False, "days": days, "truncated": truncated, "points": []}

    points: List[Dict[str, Any]] = []
    previous_models: Optional[Tuple[str, ...]] = None
    for probe_run_id, run_rows in group_by_run(rows):
        _, avg_position = position_rollup(run_rows)
        mentioned = sum(1 for r in run_rows if r.get("mentioned"))
        citations = citation_rollup(run_rows)
        # WHICH INSTRUMENT took this measurement.
        #
        # A subject can be moved between probe tiers (#349 A7), and a frontier run and a
        # cheap run are not two readings of the same thing — different models, different
        # answers, a different share of voice. Plotting them on one line and calling the
        # step a trend is the wrong-number-that-is-a-valid-number shape. The model set is
        # already recorded per row, so the break is derivable rather than asserted.
        models = tuple(sorted({(r.get("model") or "") for r in run_rows if r.get("model")}))
        comparable = previous_models is None or models == previous_models
        points.append({
            "probe_run_id": probe_run_id,
            "run_at": min((r.get("run_at") or "") for r in run_rows),
            "total_probes": len(run_rows),
            "mentioned": mentioned,
            "share_of_voice": (mentioned / len(run_rows)) if run_rows else 0.0,
            "avg_position": avg_position,
            "sentiment_score": sentiment_rollup(run_rows)["score"],
            "ghost_citations": citations["ghost_citations"],
            "brand_cited": citations["brand_cited"],
            "models": list(models),
            # False on the FIRST run measured with a different set — the point where a
            # reader must stop reading the line as continuous.
            "comparable_with_previous": comparable,
        })
        previous_models = models

    first, last = points[0], points[-1]
    # A window that changed instruments mid-way has no single answer to "better or worse",
    # so it does not get given one.
    model_changed = any(not p["comparable_with_previous"] for p in points)
    return {
        "present": True,
        "days": days,
        "truncated": truncated,
        "points": points,
        # The one number a person actually asks for: better or worse than when the
        # window opened. `None` where there is nothing to compare against, never 0.
        "model_set_changed": model_changed,
        "change": {
            "share_of_voice": (
                None if model_changed
                else last["share_of_voice"] - first["share_of_voice"]
            ),
            "avg_position": (
                last["avg_position"] - first["avg_position"]
                if not model_changed
                and last["avg_position"] is not None and first["avg_position"] is not None
                else None
            ),
            "runs_compared": len(points),
            # Named so a caller rendering a dash knows WHY it is a dash. "No data" and
            # "the question is not answerable over this window" are different facts.
            "not_comparable_reason": (
                "the probe model set changed inside this window" if model_changed else None
            ),
        },
    }


def share_of_voice_from_rows(
    rows: List[Dict[str, Any]], *, subject_label: str, days: int, truncated: bool = False,
) -> Dict[str, Any]:
    """Share of voice: the SUBJECT against the competitors, bucketed per run.

    Named share-of-voice since it shipped, this counted competitor mentions only and
    left the subject out entirely — a competitor tally, in which the one brand the
    page belongs to had no share at all.
    """
    label = (subject_label or "").strip() or "This subject"
    buckets: List[Dict[str, Any]] = []
    overall: Dict[str, int] = {}
    overall_probes = overall_subject = 0

    for probe_run_id, run_rows in group_by_run(rows):
        counts: Dict[str, int] = {}
        for r in run_rows:
            for c in r.get("competitors_mentioned") or []:
                cn = (c or "").strip()
                if cn:
                    counts[cn] = counts.get(cn, 0) + 1
        subject_count = sum(1 for r in run_rows if r.get("mentioned"))
        # Share is over every NAMED brand in the run, ours included — that is what
        # makes it a share rather than two unrelated tallies side by side.
        named_total = subject_count + sum(counts.values())
        buckets.append({
            "probe_run_id": probe_run_id,
            "run_at": min((r.get("run_at") or "") for r in run_rows),
            "total_probes": len(run_rows),
            "subject_mentions": subject_count,
            "share_of_named_brands": (subject_count / named_total) if named_total else 0.0,
            "competitor_mentions": [
                {"name": k, "count": v}
                for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            ],
        })
        overall_probes += len(run_rows)
        overall_subject += subject_count
        for k, v in counts.items():
            overall[k] = overall.get(k, 0) + v

    named_total = overall_subject + sum(overall.values())
    return {
        "days": days,
        "truncated": truncated,
        "subject_label": label,
        "buckets": buckets,
        "totals": {
            "probes": overall_probes,
            "subject_mentions": overall_subject,
            "subject_share_of_named_brands": (overall_subject / named_total) if named_total else 0.0,
            "subject_share_of_probes": (overall_subject / overall_probes) if overall_probes else 0.0,
            "competitor_mentions": [
                {"name": k, "count": v}
                for k, v in sorted(overall.items(), key=lambda kv: kv[1], reverse=True)[:20]
            ],
        },
    }
