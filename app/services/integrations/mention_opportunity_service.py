"""
Mention Opportunity Service — surfaces actionable content + outreach
opportunities for a tracked subject.

Inputs (cheap, already paid for):
  - The subject's existing mention_history rows (DB read)
  - DataForSEO Labs Related Keywords / Keyword Suggestions APIs
  - DataForSEO SERP People-Also-Ask extraction (already in our SERP responses)

Output: a list of typed opportunities the user can act on:
  - trending_topic       — recurring theme across recent mentions worth writing about
  - outlet_pitch         — outlet that's been actively covering this subject area
  - keyword_opportunity  — high-volume related keyword to target with content
  - pao_question         — "People Also Ask" question to answer in a post
  - author_relationship  — author who's mentioned the subject 2+ times (warm contact)
  - sentiment_response   — negative-sentiment mention worth addressing publicly

Cost discipline:
  - Default path: DB aggregation only (free) + 1 DataForSEO Labs call (~$0.001)
  - Optional Haiku narrative summarization (`use_llm_summary=True`) for
    polished prose around each opportunity (~$0.005-0.015 per call)

The endpoint is read-only — it doesn't mutate state, doesn't trigger refreshes,
and doesn't write to mention_history. It's a pure analysis pass over existing data.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.mention_identity_service import (
    SubjectFacets, normalize_text,
)
from app.services.integrations.mention_cost_logger import (
    CostAttribution, log_dataforseo_labs_call, log_dataforseo_serp_call,
    log_haiku_call, recompute_lifetime_cost,
)

logger = logging.getLogger(__name__)


DATAFORSEO_LABS_RELATED = "https://api.dataforseo.com/v3/dataforseo_labs/google/related_keywords/live"
DATAFORSEO_LABS_SUGGESTIONS = "https://api.dataforseo.com/v3/dataforseo_labs/google/keyword_suggestions/live"
DATAFORSEO_SERP_ORGANIC = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Stopwords for n-gram trending — keep tight so we don't over-prune signal.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "to", "of", "in", "on", "by",
    "at", "as", "with", "from", "is", "are", "be", "this", "that", "these",
    "those", "it", "its", "if", "than", "then", "so", "not", "no", "we",
    "they", "their", "our", "your", "his", "her", "you", "us", "them",
    "into", "out", "up", "down", "over", "under", "via",
    # numerals / weak words
    "new", "best", "top", "all", "more",
}


# ────────────────────────────────────────────────────────────────────────────
# DTOs
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Opportunity:
    type: str                      # 'trending_topic' | 'outlet_pitch' | ...
    title: str                     # short headline (1 line)
    rationale: str                 # 1-3 sentences explaining why this is an opportunity
    suggested_action: str          # what the user should do with it
    priority_score: float          # 0..1
    source: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "title": self.title,
            "rationale": self.rationale,
            "suggested_action": self.suggested_action,
            "priority_score": round(self.priority_score, 3),
            "source": self.source,
            "metadata": self.metadata,
        }


# ────────────────────────────────────────────────────────────────────────────
# Service
# ────────────────────────────────────────────────────────────────────────────

class MentionOpportunityService:
    def __init__(self) -> None:
        self.supabase = get_supabase_client()
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY") or ""
        self.dataforseo_b64 = os.getenv("DATAFORSEO_BASE64") or ""

    async def generate(
        self,
        *,
        tracked_mention_id: str,
        types: Optional[List[str]] = None,
        days: int = 30,
        limit_per_type: int = 5,
        use_llm_summary: bool = False,
        attribution: Optional[CostAttribution] = None,
    ) -> Dict[str, Any]:
        """
        Generate opportunities for a tracked subject.

        Two categories of opportunity types:

        Subject-driven (work on a fresh subject with zero mention history):
          - keyword_opportunity   — DataForSEO Labs Related Keywords on subject_label
          - pao_question          — DataForSEO SERP "People Also Ask" block
          - ai_overview           — Google's generative AI Overview answer; brand-mention check
          - featured_snippet      — current position-0 snippet to outrank
          - related_search        — Google's "Searches related to" block
          - competitor_ranking    — top organic pages currently ranking for the subject

        Mention-derived (require existing mention_history; skip silently when 0):
          - trending_topic        — bigram analysis over recent mention titles/excerpts
          - outlet_pitch          — outlets that have covered the subject
          - author_relationship   — authors who covered the subject 2+ times
          - sentiment_response    — recent negative-sentiment mentions

        types: subset of all 10 names; default = all
        """
        types = types or [
            # Subject-driven (always work)
            "keyword_opportunity", "pao_question", "ai_overview",
            "featured_snippet", "related_search", "competitor_ranking",
            # Mention-derived (require mention history)
            "trending_topic", "outlet_pitch",
            "author_relationship", "sentiment_response",
        ]
        start = time.time()

        # Load subject + recent mentions
        subject = self._load_subject(tracked_mention_id)
        if not subject:
            return {
                "tracked_mention_id": tracked_mention_id,
                "opportunities": [],
                "errors": {"subject": "not found"},
            }
        mentions = self._load_mentions(tracked_mention_id, days=days)

        # If caller didn't pre-build attribution, derive it from the subject row
        # so cost logging still works.
        if attribution is None:
            attribution = CostAttribution(
                user_id=subject.get("user_id"),
                workspace_id=subject.get("workspace_id"),
                tracked_mention_id=tracked_mention_id,
                product_id=subject.get("product_id"),
                api_key_id=subject.get("api_key_id"),
            )

        # Build opportunities per type
        opportunities: List[Opportunity] = []
        errors: Dict[str, str] = {}

        if "trending_topic" in types:
            try:
                opportunities.extend(self._trending_topics(mentions, subject, limit_per_type))
            except Exception as e:
                errors["trending_topic"] = str(e)[:200]

        if "outlet_pitch" in types:
            try:
                opportunities.extend(self._outlet_pitches(mentions, subject, limit_per_type))
            except Exception as e:
                errors["outlet_pitch"] = str(e)[:200]

        if "author_relationship" in types:
            try:
                opportunities.extend(self._author_relationships(mentions, subject, limit_per_type))
            except Exception as e:
                errors["author_relationship"] = str(e)[:200]

        if "sentiment_response" in types:
            try:
                opportunities.extend(self._sentiment_responses(mentions, subject, limit_per_type))
            except Exception as e:
                errors["sentiment_response"] = str(e)[:200]

        if "keyword_opportunity" in types:
            try:
                kw_ops = await self._keyword_opportunities(subject, limit_per_type, attribution)
                opportunities.extend(kw_ops)
            except Exception as e:
                errors["keyword_opportunity"] = str(e)[:200]
                logger.warning(f"opportunity: keyword fetch failed: {e}")

        # SERP-derived signals share one DataForSEO call. If any of these
        # types are requested, fetch the SERP response once and extract each
        # block. Cost: ≤ 3 SERP calls (fallback chain) regardless of how
        # many of the 5 signal types are requested.
        serp_types = {
            "pao_question", "ai_overview", "featured_snippet",
            "related_search", "competitor_ranking",
        } & set(types)
        if serp_types:
            try:
                serp_ops = await self._serp_signals(
                    subject=subject,
                    types_wanted=serp_types,
                    limit=limit_per_type,
                    attribution=attribution,
                )
                for t, ops in serp_ops.items():
                    opportunities.extend(ops)
            except Exception as e:
                errors["serp_signals"] = str(e)[:200]
                logger.warning(f"opportunity: SERP signals fetch failed: {e}")

        # Sort by priority desc
        opportunities.sort(key=lambda o: o.priority_score, reverse=True)

        # Optional LLM polish: rewrite rationales + suggested_actions in better prose
        if use_llm_summary and opportunities and self.anthropic_key:
            try:
                opportunities = await self._polish_with_haiku(opportunities, subject, attribution)
            except Exception as e:
                errors["llm_summary"] = str(e)[:200]
                logger.warning(f"opportunity: Haiku polish failed: {e}")

        # Roll up lifetime cost (Layer C) so total_billed_usd on the row stays current
        recompute_lifetime_cost(tracked_mention_id=tracked_mention_id)

        return {
            "tracked_mention_id": tracked_mention_id,
            "subject_label": subject.get("subject_label"),
            "days": days,
            "mention_count": len(mentions),
            "opportunities": [o.to_dict() for o in opportunities],
            "errors": errors,
            "latency_ms": int((time.time() - start) * 1000),
        }

    # ───── DB loaders ─────

    def _load_subject(self, tracked_mention_id: str) -> Optional[Dict[str, Any]]:
        try:
            r = (
                self.supabase.client.table("tracked_mentions")
                .select(
                    "id, subject_label, brand_name, aliases, "
                    "language_codes, country_codes, subject_facets, "
                    "user_id, workspace_id, product_id, api_key_id"
                )
                .eq("id", tracked_mention_id)
                .maybe_single()
                .execute()
            )
            return r.data if r else None
        except Exception as e:
            logger.warning(f"opportunity: subject load failed: {e}")
            return None

    def _load_mentions(self, tracked_mention_id: str, *, days: int) -> List[Dict[str, Any]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        try:
            r = (
                self.supabase.client.table("mention_history")
                .select(
                    "id, title, excerpt, outlet_domain, outlet_name, outlet_type, "
                    "author, published_at, sentiment, relevance, source, url"
                )
                .eq("tracked_mention_id", tracked_mention_id)
                .gte("discovered_at", cutoff)
                .order("published_at", desc=True)
                .limit(500)
                .execute()
            )
            return r.data or []
        except Exception as e:
            logger.warning(f"opportunity: mentions load failed: {e}")
            return []

    # ───── 1. Trending topics (n-gram analysis on titles + excerpts) ─────

    def _trending_topics(
        self, mentions: List[Dict[str, Any]], subject: Dict[str, Any], limit: int,
    ) -> List[Opportunity]:
        # Subject token blocklist — don't surface the subject's own name as a "topic"
        subject_tokens = set()
        for s in [subject.get("subject_label"), subject.get("brand_name"),
                  *(subject.get("aliases") or [])]:
            for w in (normalize_text(s or "")).split():
                if w:
                    subject_tokens.add(w)

        bigrams: Counter = Counter()
        bigram_to_mentions: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for m in mentions:
            text = " ".join(filter(None, [m.get("title"), m.get("excerpt")]))
            words = [
                w for w in normalize_text(text).split()
                if w and w not in _STOPWORDS and not w.isdigit() and len(w) >= 3
            ]
            for i in range(len(words) - 1):
                pair = (words[i], words[i + 1])
                if pair[0] in subject_tokens and pair[1] in subject_tokens:
                    continue
                bigrams[pair] += 1
                bigram_to_mentions[pair].append(m["id"])

        out: List[Opportunity] = []
        for (a, b), count in bigrams.most_common(limit * 3):
            if count < 2:
                break  # need at least 2 mentions to be a "trend"
            phrase = f"{a} {b}"
            mention_ids = bigram_to_mentions[(a, b)][:5]
            sample_titles = [m["title"] for m in mentions if m["id"] in mention_ids][:3]
            out.append(Opportunity(
                type="trending_topic",
                title=f"\"{phrase}\" — {count} recent mentions",
                rationale=(
                    f"This phrase appears in {count} recent mentions of "
                    f"\"{subject.get('subject_label')}\". Sample headlines: "
                    + "; ".join(sample_titles[:2])
                ),
                suggested_action=(
                    f"Write a post or article that engages directly with the "
                    f"\"{phrase}\" theme. Cite the press coverage, take a position, "
                    "or add your perspective."
                ),
                priority_score=min(1.0, count / 10.0),
                source={"mention_ids": mention_ids, "phrase": phrase, "count": count},
            ))
            if len(out) >= limit:
                break
        return out

    # ───── 2. Outlet pitches (frequency + relationship signal) ─────

    def _outlet_pitches(
        self, mentions: List[Dict[str, Any]], subject: Dict[str, Any], limit: int,
    ) -> List[Opportunity]:
        outlet_counts: Counter = Counter()
        outlet_meta: Dict[str, Dict[str, Any]] = {}
        for m in mentions:
            domain = (m.get("outlet_domain") or "").lower()
            if not domain:
                continue
            outlet_counts[domain] += 1
            outlet_meta.setdefault(domain, {
                "name": m.get("outlet_name") or domain,
                "type": m.get("outlet_type"),
                "sample_url": m.get("url"),
            })

        out: List[Opportunity] = []
        for domain, count in outlet_counts.most_common(limit):
            meta = outlet_meta.get(domain, {})
            if count >= 3:
                rationale = (
                    f"{meta.get('name', domain)} has covered \"{subject.get('subject_label')}\" "
                    f"{count} times in the discovery window. They're a warm contact — "
                    "more likely to publish your pitch than a cold outreach."
                )
                action = (
                    f"Pitch {meta.get('name', domain)} a follow-up or angle they haven't covered yet. "
                    "Reference the prior coverage in your pitch to anchor relevance."
                )
                priority = min(1.0, 0.5 + count / 10.0)
            else:
                rationale = (
                    f"{meta.get('name', domain)} has covered the subject {count} time(s). "
                    "Worth considering as a pitch target."
                )
                action = f"Send a thoughtful pitch to {meta.get('name', domain)} for a feature."
                priority = 0.3 + (count * 0.1)
            out.append(Opportunity(
                type="outlet_pitch",
                title=meta.get("name") or domain,
                rationale=rationale,
                suggested_action=action,
                priority_score=priority,
                source={"outlet_domain": domain, "mention_count": count,
                        "sample_url": meta.get("sample_url")},
                metadata={"outlet_type": meta.get("type")},
            ))
        return out

    # ───── 3. Author relationships ─────

    def _author_relationships(
        self, mentions: List[Dict[str, Any]], subject: Dict[str, Any], limit: int,
    ) -> List[Opportunity]:
        author_counts: Counter = Counter()
        author_meta: Dict[str, Dict[str, Any]] = {}
        for m in mentions:
            author = (m.get("author") or "").strip()
            if not author or author.lower() in ("staff", "editor", "admin", "unknown"):
                continue
            key = author.lower()
            author_counts[key] += 1
            author_meta.setdefault(key, {
                "display_name": author,
                "outlet_domain": m.get("outlet_domain"),
                "sample_url": m.get("url"),
            })

        out: List[Opportunity] = []
        for key, count in author_counts.most_common(limit):
            if count < 2:
                continue
            meta = author_meta[key]
            out.append(Opportunity(
                type="author_relationship",
                title=meta["display_name"],
                rationale=(
                    f"{meta['display_name']} has written about \"{subject.get('subject_label')}\" "
                    f"{count} times in the last window — at {meta.get('outlet_domain')}. "
                    "Established relationship; high-conversion pitch target."
                ),
                suggested_action=(
                    f"Reach out to {meta['display_name']} directly with a fresh angle. "
                    "Reference your past appearance in their writing."
                ),
                priority_score=min(1.0, 0.4 + count / 6.0),
                source={"author": meta["display_name"],
                        "outlet_domain": meta.get("outlet_domain"),
                        "mention_count": count,
                        "sample_url": meta.get("sample_url")},
            ))
        return out

    # ───── 4. Sentiment response (negative mentions to address) ─────

    def _sentiment_responses(
        self, mentions: List[Dict[str, Any]], subject: Dict[str, Any], limit: int,
    ) -> List[Opportunity]:
        negs = [
            m for m in mentions
            if (m.get("sentiment") == "negative")
            and (m.get("relevance") in ("exact", "tangential", None))
        ]
        out: List[Opportunity] = []
        for m in negs[:limit]:
            out.append(Opportunity(
                type="sentiment_response",
                title=m.get("title") or "Negative mention",
                rationale=(
                    f"{m.get('outlet_name') or m.get('outlet_domain')} published a "
                    "negative-sentiment piece. Active reputation-management moment."
                ),
                suggested_action=(
                    "Decide: respond publicly with clarification, write a post addressing "
                    "the underlying concern, or reach out to the author privately. Don't ignore."
                ),
                priority_score=0.85,  # always near top — reputation is urgent
                source={
                    "mention_id": m["id"],
                    "url": m.get("url"),
                    "outlet_domain": m.get("outlet_domain"),
                    "published_at": m.get("published_at"),
                },
            ))
        return out

    # ───── 5. Keyword opportunities (DataForSEO Labs) ─────

    def _fallback_seeds(self, subject: Dict[str, Any]) -> List[str]:
        """Order of seeds to try when keyword-research APIs return 0 items.

        Niche multi-word product SKUs often have no search volume in
        DataForSEO's database, so the API returns empty for the literal
        label. To still produce useful results, we try ONLY seeds the
        caller explicitly provided:

          1. subject_label    (always tried first)
          2. brand_name       (when set on the row)
          3. aliases[*]       (when supplied)

        We do NOT autonomously split the label into individual words —
        that decomposes the input string in ways the caller didn't
        request and risks producing unrelated keyword data (a niche SKU's
        last word is often not a meaningful brand or category). To get
        that breadth, the caller supplies their own variants in `aliases`
        or sets `auto_expand_aliases: true` at create time.
        """
        seeds: List[str] = []
        seen: set = set()

        def _add(s: Optional[str]) -> None:
            if not s:
                return
            v = s.strip()
            if not v or len(v) < 3:
                return
            key = normalize_text(v)
            if key in seen:
                return
            seen.add(key)
            seeds.append(v)

        _add(subject.get("subject_label"))
        _add(subject.get("brand_name"))
        for a in (subject.get("aliases") or []):
            _add(a)
        return seeds

    async def _keyword_opportunities(
        self, subject: Dict[str, Any], limit: int,
        attribution: Optional[CostAttribution] = None,
    ) -> List[Opportunity]:
        if not self.dataforseo_b64:
            return []
        seeds = self._fallback_seeds(subject)
        if not seeds:
            return []
        country_code = (subject.get("country_codes") or [None])[0]
        language_code = (subject.get("language_codes") or ["en"])[0].lower()

        # Try seeds in order until one returns enough items. Cap at 3 calls so
        # the worst case is 3× DataForSEO Labs (~$0.003) — still well under
        # the 2-credit charge.
        items: List[Dict[str, Any]] = []
        used_seed = seeds[0]
        for seed in seeds[:3]:
            body = [{
                "keyword": seed,
                "location_code": _country_to_dfs_location(country_code),
                "language_code": language_code,
                "limit": max(20, limit * 4),
                "include_serp_info": False,
                "include_seed_keyword": False,
            }]
            call_start = time.time()
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(
                        DATAFORSEO_LABS_RELATED,
                        headers={"Authorization": f"Basic {self.dataforseo_b64}",
                                 "Content-Type": "application/json"},
                        json=body,
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as e:
                logger.warning(f"opportunity: DataForSEO related-keywords '{seed}' failed: {e}")
                log_dataforseo_labs_call(
                    attribution=attribution, seed_keyword=seed, items_returned=0,
                    latency_ms=int((time.time() - call_start) * 1000),
                    success=False, error_message=str(e),
                )
                continue

            this_round: List[Dict[str, Any]] = []
            for task in (data.get("tasks") or []):
                for r in (task.get("result") or []):
                    for it in (r.get("items") or []):
                        kw_data = (it.get("keyword_data") or {})
                        kw = kw_data.get("keyword") or it.get("keyword")
                        info = kw_data.get("keyword_info") or {}
                        if not kw:
                            continue
                        this_round.append({
                            "keyword": kw,
                            "search_volume": info.get("search_volume") or 0,
                            "competition": info.get("competition") or "",
                            "cpc": info.get("cpc") or 0.0,
                        })

            log_dataforseo_labs_call(
                attribution=attribution, seed_keyword=seed,
                items_returned=len(this_round),
                latency_ms=int((time.time() - call_start) * 1000), success=True,
            )

            if this_round:
                items = this_round
                used_seed = seed
                break  # found a seed with data, stop fallback

        if not items:
            return []

        # Rank by search volume desc
        items.sort(key=lambda x: -(x.get("search_volume") or 0))
        items = items[:limit]

        # Note used_seed in source so partners can see WHICH seed produced the
        # opportunities — important when fallback kicked in (e.g. brand_name
        # was used because subject_label had no data).
        seed_was_fallback = (used_seed != seeds[0]) if seeds else False

        out: List[Opportunity] = []
        for item in items:
            volume = item.get("search_volume") or 0
            if volume < 10:
                continue
            out.append(Opportunity(
                type="keyword_opportunity",
                title=item["keyword"],
                rationale=(
                    f"\"{item['keyword']}\" gets {volume:,} monthly searches in "
                    f"{country_code or 'your target market'}. Related to "
                    f"\"{used_seed}\". Potential SEO content angle."
                ),
                suggested_action=(
                    f"Write a piece optimized for \"{item['keyword']}\". Anchor it to "
                    "your brand's expertise on the topic."
                ),
                priority_score=min(1.0, 0.3 + (volume / 5000.0)),
                source={"keyword": item["keyword"], "search_volume": volume,
                        "competition": item.get("competition"), "cpc": item.get("cpc"),
                        "seed_used": used_seed, "fallback": seed_was_fallback},
            ))
        return out

    # ───── SERP signals (PAA + AI Overview + Featured Snippet + Related + Top Organic) ─────

    async def _serp_signals(
        self, *, subject: Dict[str, Any],
        types_wanted: set,  # subset of {pao_question, ai_overview, featured_snippet, related_search, competitor_ranking}
        limit: int,
        attribution: Optional[CostAttribution] = None,
    ) -> Dict[str, List[Opportunity]]:
        """One DataForSEO SERP call → up to 5 different opportunity types extracted
        from the response. Same fallback-seed chain as keyword opportunities.

        Returns a dict keyed by opportunity type. Caller filters by `types_wanted`
        to control which signal types to surface; we always parse all blocks since
        the parsing cost is trivial vs the SERP call cost.
        """
        if not self.dataforseo_b64:
            return {}
        seeds = self._fallback_seeds(subject)
        if not seeds:
            return {}
        country_code = (subject.get("country_codes") or [None])[0]
        language_code = (subject.get("language_codes") or ["en"])[0].lower()

        # Fallback chain: try up to 3 seeds. Switch seeds when the current one
        # returns no signal blocks at all (no PAA, no AI Overview, no Featured
        # Snippet, no related searches). Brand/product names with very low
        # search volume often have empty SERP feature blocks.
        used_seed = seeds[0]
        blocks: Dict[str, Any] = {
            "pao": [], "ai_overview": None, "featured_snippet": None,
            "related_searches": [], "organic": [],
        }

        for seed in seeds[:3]:
            body = [{
                "keyword": seed,
                "location_code": _country_to_dfs_location(country_code),
                "language_code": language_code,
                "depth": 30,
                "people_also_ask_click_depth": 1,
            }]
            call_start = time.time()
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(
                        DATAFORSEO_SERP_ORGANIC,
                        headers={"Authorization": f"Basic {self.dataforseo_b64}",
                                 "Content-Type": "application/json"},
                        json=body,
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as e:
                logger.warning(f"opportunity: DataForSEO SERP '{seed}' failed: {e}")
                log_dataforseo_serp_call(
                    attribution=attribution, operation="serp_signals",
                    query=seed, items_returned=0,
                    latency_ms=int((time.time() - call_start) * 1000),
                    success=False, error_message=str(e),
                )
                continue

            this_round = self._parse_serp_blocks(data, limit=limit)
            log_dataforseo_serp_call(
                attribution=attribution, operation="serp_signals",
                query=seed,
                items_returned=(
                    len(this_round["pao"])
                    + (1 if this_round["ai_overview"] else 0)
                    + (1 if this_round["featured_snippet"] else 0)
                    + len(this_round["related_searches"])
                    + len(this_round["organic"])
                ),
                latency_ms=int((time.time() - call_start) * 1000), success=True,
            )

            # Stop falling back as soon as we get any signal at all
            has_signal = (
                this_round["pao"]
                or this_round["ai_overview"]
                or this_round["featured_snippet"]
                or this_round["related_searches"]
                or this_round["organic"]
            )
            if has_signal:
                blocks = this_round
                used_seed = seed
                break

        seed_was_fallback = (used_seed != seeds[0])

        # Build opportunity lists per requested type
        out: Dict[str, List[Opportunity]] = {}
        if "pao_question" in types_wanted:
            out["pao_question"] = self._build_pao_opps(
                blocks["pao"], used_seed, seed_was_fallback, limit,
            )
        if "ai_overview" in types_wanted:
            out["ai_overview"] = self._build_ai_overview_opps(
                blocks["ai_overview"], subject, used_seed, seed_was_fallback,
            )
        if "featured_snippet" in types_wanted:
            out["featured_snippet"] = self._build_featured_snippet_opps(
                blocks["featured_snippet"], used_seed, seed_was_fallback,
            )
        if "related_search" in types_wanted:
            out["related_search"] = self._build_related_search_opps(
                blocks["related_searches"], used_seed, seed_was_fallback, limit,
            )
        if "competitor_ranking" in types_wanted:
            out["competitor_ranking"] = self._build_competitor_ranking_opps(
                blocks["organic"], subject, used_seed, seed_was_fallback, limit,
            )
        return out

    def _parse_serp_blocks(self, data: Dict[str, Any], *, limit: int) -> Dict[str, Any]:
        """Walk the DataForSEO SERP response and extract every block type we care
        about into normalized dicts. Single pass through the items array."""
        result: Dict[str, Any] = {
            "pao": [], "ai_overview": None, "featured_snippet": None,
            "related_searches": [], "organic": [],
        }
        seen_q: set = set()
        seen_related: set = set()

        for task in (data.get("tasks") or []):
            for r in (task.get("result") or []):
                for item in (r.get("items") or []):
                    itype = (item.get("type") or "").lower()

                    if itype == "people_also_ask":
                        for q in (item.get("items") or []):
                            title = (q.get("title") or "").strip()
                            if not title:
                                continue
                            key = normalize_text(title)
                            if key in seen_q:
                                continue
                            seen_q.add(key)
                            result["pao"].append({
                                "title": title,
                                "expanded": q.get("expanded_element") or [],
                            })
                            if len(result["pao"]) >= limit * 2:
                                break

                    elif itype == "ai_overview" and result["ai_overview"] is None:
                        # AI Overview = Google's generative answer at the top
                        # of the SERP. May contain text + cited references.
                        ai_text_parts: List[str] = []
                        for sub in (item.get("items") or []):
                            t = (sub.get("text") or sub.get("description") or "").strip()
                            if t:
                                ai_text_parts.append(t)
                        ai_text = " ".join(ai_text_parts)[:1500]
                        references: List[Dict[str, Any]] = []
                        for ref in (item.get("references") or []):
                            references.append({
                                "title": (ref.get("title") or "")[:200],
                                "url": ref.get("url") or "",
                                "domain": ref.get("domain") or "",
                                "source": ref.get("source") or "",
                            })
                        result["ai_overview"] = {
                            "text": ai_text,
                            "references": references[:10],
                        }

                    elif itype == "featured_snippet" and result["featured_snippet"] is None:
                        result["featured_snippet"] = {
                            "title": (item.get("title") or "")[:200],
                            "description": (item.get("description") or "")[:400],
                            "url": item.get("url") or "",
                            "domain": item.get("domain") or "",
                        }

                    elif itype == "related_searches":
                        for term in (item.get("items") or []):
                            t = (term or "").strip() if isinstance(term, str) else (term.get("title") or "").strip()
                            if not t:
                                continue
                            key = normalize_text(t)
                            if key in seen_related:
                                continue
                            seen_related.add(key)
                            result["related_searches"].append(t)
                            if len(result["related_searches"]) >= limit * 2:
                                break

                    elif itype == "organic":
                        # Top organic results — keep in rank order, capped
                        if len(result["organic"]) < 10:
                            result["organic"].append({
                                "rank": item.get("rank_absolute") or item.get("rank_group"),
                                "title": (item.get("title") or "")[:200],
                                "description": (item.get("description") or "")[:400],
                                "url": item.get("url") or "",
                                "domain": item.get("domain") or "",
                            })
        return result

    def _build_pao_opps(
        self, pao_items: List[Dict[str, Any]], used_seed: str,
        seed_was_fallback: bool, limit: int,
    ) -> List[Opportunity]:
        out: List[Opportunity] = []
        for q in pao_items[:limit]:
            answer_snippet = ""
            for exp in (q.get("expanded") or []):
                ds = exp.get("description") or exp.get("answer")
                if ds:
                    answer_snippet = (ds or "")[:240]
                    break
            out.append(Opportunity(
                type="pao_question",
                title=q["title"],
                rationale=(
                    f"Real Google searchers are asking this when they search \"{used_seed}\". "
                    "Sourced from Google's People Also Ask block."
                    + (f" Current top answer snippet: \"{answer_snippet}\"" if answer_snippet else "")
                ),
                suggested_action=(
                    "Write a focused FAQ-style post or article section answering this exact question. "
                    "Optimize the H2 to match the question text — Google often pulls these straight into "
                    "PAA blocks, giving you free SERP real estate."
                ),
                priority_score=0.6,
                source={"question": q["title"], "seed_keyword": used_seed,
                        "fallback": seed_was_fallback},
            ))
        return out

    def _build_ai_overview_opps(
        self, ai_overview: Optional[Dict[str, Any]],
        subject: Dict[str, Any], used_seed: str, seed_was_fallback: bool,
    ) -> List[Opportunity]:
        if not ai_overview or not ai_overview.get("text"):
            return []
        ai_text = ai_overview["text"]
        references = ai_overview.get("references") or []

        # Brand-mention check: does the AI Overview text or any reference
        # mention the subject? Compare normalized strings.
        nt_ai = normalize_text(ai_text)
        nt_refs = " ".join(
            normalize_text(r.get("title") or "") + " " + normalize_text(r.get("domain") or "")
            for r in references
        )
        haystack = nt_ai + " " + nt_refs

        candidates: List[str] = []
        for s in [subject.get("subject_label"), subject.get("brand_name"),
                  *(subject.get("aliases") or [])]:
            if s and s.strip():
                candidates.append(s.strip())
        # Dedup normalized candidates
        seen: set = set()
        unique_candidates: List[str] = []
        for c in candidates:
            n = normalize_text(c)
            if n and n not in seen:
                seen.add(n)
                unique_candidates.append(c)

        brand_mentioned = any(normalize_text(c) in haystack for c in unique_candidates if normalize_text(c))

        # Find competitor brands that DID get cited (from reference domains/titles)
        cited_domains = [r.get("domain") for r in references if r.get("domain")]
        cited_titles = [r.get("title") for r in references if r.get("title")]

        if brand_mentioned:
            return [Opportunity(
                type="ai_overview",
                title=f"Google's AI Overview cites {subject.get('subject_label') or used_seed}",
                rationale=(
                    f"For the search \"{used_seed}\", Google's generative AI Overview "
                    f"includes your subject. The AI says: \"{ai_text[:280]}{'…' if len(ai_text) > 280 else ''}\""
                    + (f" Cited references: {', '.join(cited_domains[:5])}" if cited_domains else "")
                ),
                suggested_action=(
                    "If the AI's framing is correct, amplify it in your own content to reinforce. "
                    "If it's incomplete or wrong, write authoritative content that targets the cited "
                    "URLs' position — Google regenerates the AI Overview as new content gets indexed."
                ),
                priority_score=0.95,
                source={
                    "ai_overview_text": ai_text,
                    "references": references,
                    "brand_mentioned": True,
                    "cited_domains": cited_domains,
                    "seed_keyword": used_seed,
                    "fallback": seed_was_fallback,
                },
            )]
        else:
            return [Opportunity(
                type="ai_overview",
                title=f"Google's AI Overview does NOT cite {subject.get('subject_label') or used_seed}",
                rationale=(
                    f"For \"{used_seed}\", Google's generative AI answer does not mention your subject. "
                    f"It cites these sources instead: {', '.join(cited_domains[:5]) if cited_domains else '(no references shown)'}. "
                    f"AI text: \"{ai_text[:240]}{'…' if len(ai_text) > 240 else ''}\""
                ),
                suggested_action=(
                    "Generative Engine Optimization (GEO) opportunity: study what the cited sources say, "
                    "write content that more authoritatively answers the same query intent, and target "
                    "those domains to displace them. Also pitch the cited outlets directly — getting "
                    "linked from them feeds the next AI Overview regeneration."
                ),
                priority_score=0.95,
                source={
                    "ai_overview_text": ai_text,
                    "references": references,
                    "brand_mentioned": False,
                    "cited_domains": cited_domains,
                    "cited_titles": cited_titles,
                    "seed_keyword": used_seed,
                    "fallback": seed_was_fallback,
                },
            )]

    def _build_featured_snippet_opps(
        self, snippet: Optional[Dict[str, Any]],
        used_seed: str, seed_was_fallback: bool,
    ) -> List[Opportunity]:
        if not snippet or not (snippet.get("title") or snippet.get("description")):
            return []
        return [Opportunity(
            type="featured_snippet",
            title=f"Position-0 snippet held by {snippet.get('domain') or 'unknown'}",
            rationale=(
                f"For \"{used_seed}\", Google's featured snippet (position 0) is currently held by "
                f"{snippet.get('domain') or 'a competitor'}: \"{(snippet.get('description') or snippet.get('title') or '')[:240]}\". "
                "Featured snippets get the largest CTR share above the standard organic results."
            ),
            suggested_action=(
                "Write a piece that answers the underlying question more directly and concisely. "
                "Aim for a 40–60 word answer in a single paragraph immediately after a matching H2. "
                "Outranking the snippet's source on the underlying query is the typical way to take it."
            ),
            priority_score=0.85,
            source={
                "current_url": snippet.get("url"),
                "current_domain": snippet.get("domain"),
                "current_title": snippet.get("title"),
                "current_description": snippet.get("description"),
                "seed_keyword": used_seed,
                "fallback": seed_was_fallback,
            },
        )]

    def _build_related_search_opps(
        self, related: List[str], used_seed: str,
        seed_was_fallback: bool, limit: int,
    ) -> List[Opportunity]:
        out: List[Opportunity] = []
        for term in related[:limit]:
            out.append(Opportunity(
                type="related_search",
                title=term,
                rationale=(
                    f"Google's \"Searches related to {used_seed}\" block surfaces this term, "
                    "meaning real users searching your subject also search for this. "
                    "Direct intent overlap — different from the keyword-volume signal."
                ),
                suggested_action=(
                    f"Write a piece optimized for \"{term}\" and cross-link to your existing content "
                    "on the parent subject. Google itself is telling you these queries cluster together "
                    "in user intent."
                ),
                priority_score=0.5,
                source={"related_term": term, "seed_keyword": used_seed,
                        "fallback": seed_was_fallback},
            ))
        return out

    def _build_competitor_ranking_opps(
        self, organic: List[Dict[str, Any]],
        subject: Dict[str, Any], used_seed: str, seed_was_fallback: bool,
        limit: int,
    ) -> List[Opportunity]:
        """Return top-ranked organic results for the seed keyword as-is.

        We do NOT auto-skip the brand's own domain. The same tracked subject
        gets used in two opposite ways:
          - In-house team tracking their own brand → wants competitor cards only.
          - Third-party analyst / distributor / competitor tracking the brand →
            wants the brand's own ranking too (signals SEO ownership).
        Implicit filtering would silently break the second use case. Callers
        who want to filter their own domain can use `mention_excluded_urls`
        (the existing exclude mechanism takes a `domain` field).
        """
        out: List[Opportunity] = []
        kept = 0
        for item in organic:
            if kept >= limit:
                break
            domain = (item.get("domain") or "").lower()
            if not domain:
                continue
            out.append(Opportunity(
                type="competitor_ranking",
                title=f"#{item.get('rank') or '?'} — {domain}",
                rationale=(
                    f"For \"{used_seed}\", Google ranks {domain} at position {item.get('rank') or '?'}: "
                    f"\"{item.get('title') or ''}\" — {(item.get('description') or '')[:160]}. "
                    "These are the pages currently capturing organic traffic for the keyword."
                ),
                suggested_action=(
                    f"Audit the page at {item.get('url') or domain}: what intent does it serve, what "
                    "questions does it answer, what depth/structure does it use. Write content that "
                    "matches the same intent more authoritatively to outrank it — or, if it's the "
                    "tracked brand's own domain, treat this as a baseline for their current SEO position."
                ),
                priority_score=max(0.3, 1.0 - (kept * 0.1)),  # rank 1 = highest priority
                source={
                    "rank": item.get("rank"),
                    "url": item.get("url"),
                    "domain": domain,
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "seed_keyword": used_seed,
                    "fallback": seed_was_fallback,
                },
            ))
            kept += 1
        return out

    # ───── Optional Haiku polish ─────

    async def _polish_with_haiku(
        self, ops: List[Opportunity], subject: Dict[str, Any],
        attribution: Optional[CostAttribution] = None,
    ) -> List[Opportunity]:
        if not ops or not self.anthropic_key:
            return ops
        # Send up to 12 opportunities at once, ask Haiku to rewrite rationale
        # and suggested_action in tighter, more actionable language.
        batch = ops[:12]
        prompt = f"""Polish each opportunity below into more actionable, concrete language.
Return ONLY a JSON array, same length, each entry: {{"rationale": "...", "suggested_action": "..."}}.
Keep rationale ≤ 240 chars, suggested_action ≤ 160 chars. No prose, no markdown.

Subject: {subject.get('subject_label')}
Brand: {subject.get('brand_name') or '(none)'}

Opportunities:
{json.dumps([{
    "type": o.type, "title": o.title,
    "rationale": o.rationale, "suggested_action": o.suggested_action,
} for o in batch], ensure_ascii=False)}
"""
        call_start = time.time()
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    ANTHROPIC_API,
                    headers={
                        "x-api-key": self.anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": HAIKU_MODEL,
                        "max_tokens": 2000,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            blocks = data.get("content") or []
            text = "\n".join([b.get("text", "") for b in blocks if b.get("type") == "text"])
            usage = data.get("usage") or {}
            log_haiku_call(
                attribution=attribution, operation="opportunity_polish",
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                latency_ms=int((time.time() - call_start) * 1000),
                success=True,
            )
            arr = self._parse_json_array(text)
            for i, polished in enumerate(arr[:len(batch)]):
                if isinstance(polished, dict):
                    if polished.get("rationale"):
                        batch[i].rationale = polished["rationale"][:240]
                    if polished.get("suggested_action"):
                        batch[i].suggested_action = polished["suggested_action"][:160]
        except Exception as e:
            logger.warning(f"opportunity: Haiku polish failed: {e}")
            log_haiku_call(
                attribution=attribution, operation="opportunity_polish",
                input_tokens=0, output_tokens=0,
                latency_ms=int((time.time() - call_start) * 1000),
                success=False, error_message=str(e),
            )
        return ops

    def _parse_json_array(self, text: str) -> List[Any]:
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        start = s.find("[")
        if start == -1:
            return []
        depth = 0
        end = start
        for i, ch in enumerate(s[start:], start=start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        return json.loads(s[start:end + 1])


# ────────────────────────────────────────────────────────────────────────────
# Helpers (mirrors mention_search_service._country_to_dfs_location)
# ────────────────────────────────────────────────────────────────────────────

def _country_to_dfs_location(code: Optional[str]) -> int:
    if not code:
        return 2840
    return {
        "US": 2840, "GB": 2826, "DE": 2276, "FR": 2250, "IT": 2380, "ES": 2724,
        "GR": 2300, "NL": 2528, "BE": 2056, "AT": 2040, "CH": 2756, "PT": 2620,
        "IE": 2372, "CA": 2124, "AU": 2036, "PL": 2616, "SE": 2752, "DK": 2208,
        "NO": 2578, "FI": 2246, "TR": 2792, "BG": 2100, "RO": 2642, "CY": 2196,
    }.get(code.upper(), 2840)


_service: Optional[MentionOpportunityService] = None


def get_mention_opportunity_service() -> MentionOpportunityService:
    global _service
    if _service is None:
        _service = MentionOpportunityService()
    return _service
