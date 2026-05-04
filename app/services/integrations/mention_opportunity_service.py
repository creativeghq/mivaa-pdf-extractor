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
          - pao_question          — DataForSEO SERP "People Also Ask" on subject_label

        Mention-derived (require existing mention_history; skip silently when 0):
          - trending_topic        — bigram analysis over recent mention titles/excerpts
          - outlet_pitch          — outlets that have covered the subject
          - author_relationship   — authors who covered the subject 2+ times
          - sentiment_response    — recent negative-sentiment mentions

        types: subset of all 6 names; default = all
        """
        types = types or [
            "trending_topic", "outlet_pitch", "keyword_opportunity",
            "pao_question", "author_relationship", "sentiment_response",
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

        if "pao_question" in types:
            try:
                pao_ops = await self._pao_questions(subject, limit_per_type, attribution)
                opportunities.extend(pao_ops)
            except Exception as e:
                errors["pao_question"] = str(e)[:200]
                logger.warning(f"opportunity: pao fetch failed: {e}")

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

    async def _keyword_opportunities(
        self, subject: Dict[str, Any], limit: int,
        attribution: Optional[CostAttribution] = None,
    ) -> List[Opportunity]:
        if not self.dataforseo_b64:
            return []
        seed = subject.get("subject_label") or subject.get("brand_name") or ""
        if not seed:
            return []
        country_code = (subject.get("country_codes") or [None])[0]
        language_code = (subject.get("language_codes") or ["en"])[0].lower()

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
            logger.warning(f"opportunity: DataForSEO related-keywords failed: {e}")
            log_dataforseo_labs_call(
                attribution=attribution, seed_keyword=seed, items_returned=0,
                latency_ms=int((time.time() - call_start) * 1000),
                success=False, error_message=str(e),
            )
            return []

        items: List[Dict[str, Any]] = []
        for task in (data.get("tasks") or []):
            for r in (task.get("result") or []):
                for it in (r.get("items") or []):
                    kw_data = (it.get("keyword_data") or {})
                    kw = kw_data.get("keyword") or it.get("keyword")
                    info = kw_data.get("keyword_info") or {}
                    if not kw:
                        continue
                    items.append({
                        "keyword": kw,
                        "search_volume": info.get("search_volume") or 0,
                        "competition": info.get("competition") or "",
                        "cpc": info.get("cpc") or 0.0,
                    })

        # Rank by search volume desc
        items.sort(key=lambda x: -(x.get("search_volume") or 0))
        log_dataforseo_labs_call(
            attribution=attribution, seed_keyword=seed,
            items_returned=len(items),
            latency_ms=int((time.time() - call_start) * 1000), success=True,
        )
        items = items[:limit]

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
                    f"{country_code or 'your target market'}. Related to your subject. "
                    "Potential SEO content angle."
                ),
                suggested_action=(
                    f"Write a piece optimized for \"{item['keyword']}\". Anchor it to "
                    "your brand's expertise on the topic."
                ),
                priority_score=min(1.0, 0.3 + (volume / 5000.0)),
                source={"keyword": item["keyword"], "search_volume": volume,
                        "competition": item.get("competition"), "cpc": item.get("cpc")},
            ))
        return out

    # ───── 6. People-Also-Ask questions (subject-driven, no mention dependency) ─────

    async def _pao_questions(
        self, subject: Dict[str, Any], limit: int,
        attribution: Optional[CostAttribution] = None,
    ) -> List[Opportunity]:
        """Pull "People Also Ask" questions directly from Google SERP via the
        DataForSEO Advanced SERP endpoint. This is purely subject-driven —
        works regardless of how much mention history exists. Each PAO entry
        is a real search query Google's data shows readers asking, which is
        exactly what content readers want answered."""
        if not self.dataforseo_b64:
            return []
        seed = subject.get("subject_label") or subject.get("brand_name") or ""
        if not seed:
            return []
        country_code = (subject.get("country_codes") or [None])[0]
        language_code = (subject.get("language_codes") or ["en"])[0].lower()

        body = [{
            "keyword": seed,
            "location_code": _country_to_dfs_location(country_code),
            "language_code": language_code,
            "depth": 30,  # SERP depth — PAO blocks usually appear in top 10
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
            logger.warning(f"opportunity: DataForSEO SERP PAO failed: {e}")
            log_dataforseo_serp_call(
                attribution=attribution, operation="pao_question",
                query=seed, items_returned=0,
                latency_ms=int((time.time() - call_start) * 1000),
                success=False, error_message=str(e),
            )
            return []

        # Walk the SERP items to find people_also_ask blocks. Each block has
        # `items` containing the questions and (sometimes) clicked answers.
        questions: List[Dict[str, Any]] = []
        seen_q: set = set()
        for task in (data.get("tasks") or []):
            for r in (task.get("result") or []):
                for item in (r.get("items") or []):
                    if (item.get("type") or "").lower() != "people_also_ask":
                        continue
                    for q in (item.get("items") or []):
                        title = (q.get("title") or "").strip()
                        if not title:
                            continue
                        key = normalize_text(title)
                        if key in seen_q:
                            continue
                        seen_q.add(key)
                        questions.append({
                            "title": title,
                            "expanded": q.get("expanded_element") or [],
                        })
                        if len(questions) >= limit * 2:
                            break

        log_dataforseo_serp_call(
            attribution=attribution, operation="pao_question",
            query=seed, items_returned=len(questions),
            latency_ms=int((time.time() - call_start) * 1000), success=True,
        )

        out: List[Opportunity] = []
        for q in questions[:limit]:
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
                    f"Real Google searchers are asking this when they search \"{seed}\". "
                    "Sourced from Google's People Also Ask block."
                ) + (f" Current top answer snippet: \"{answer_snippet}\"" if answer_snippet else ""),
                suggested_action=(
                    "Write a focused FAQ-style post or article section answering this exact question. "
                    "Optimize the H2 to match the question text — Google often pulls these straight into "
                    "PAA blocks, giving you free SERP real estate."
                ),
                priority_score=0.6,
                source={"question": q["title"], "seed_keyword": seed},
            ))
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
