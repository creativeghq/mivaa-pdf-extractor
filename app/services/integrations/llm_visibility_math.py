"""
LLM-visibility derivations — pure functions over probe rows.

Split out of `llm_mention_probe_service` for two reasons, in this order:

1. These are DERIVATIONS. Share of voice, average rank, a sentiment score, a ghost-
   citation count and a trend are all read off the same `llm_mention_probes` rows,
   and every one of them is a number that is wrong-but-valid when the arithmetic
   drifts. Keeping them in one module means there is one place each is computed.
2. The service they came from cannot be imported without a Supabase client, and this
   repo's CI installs pytest and nothing else — so a rollup living there is a rollup
   no test can exercise. Everything here imports stdlib only, deliberately.

Nothing in this module touches the network or the database. Callers fetch rows and
pass them in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

__all__ = [
    "citation_domain",
    "domain_is_ours",
    "dedupe_urls",
    "sentiment_rollup",
    "citation_rollup",
    "group_by_run",
    "trend_from_rows",
    "share_of_voice_from_rows",
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
