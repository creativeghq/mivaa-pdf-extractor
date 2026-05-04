"""
Mention identity verification + sentiment classification.

Mirror of `product_identity_service.py` for the mention-monitoring path.
Pipeline:

  1. Decompose subject into facets ONCE (brand, model, type, must-have tokens,
     alias variants). Cached on tracked_mentions.subject_facets to skip repeat
     Haiku calls.
  2. Pre-filter candidates that can't be a real mention (the alias must appear
     in title|excerpt|first 200 chars of body) before paying classifier cost.
  3. Verdict cache lookup keyed on sha1(content_hash || subject_facets_hash) —
     repeat URLs across daily refreshes hit cache.
  4. Haiku 4.5 batch-classifies misses — relevance + sentiment + match_kind.
  5. Persist verdict to cache (7d TTL).

Cost discipline:
  - Rule-based pre-filter drops obvious mismatches before Haiku.
  - Up to 50 candidates per Haiku call.
  - Verdict cache hits cost zero credits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
# Use the dated form when calling Anthropic's HTTP API directly — the bare
# alias `claude-haiku-4-5` 400s. `product_identity_service.py` uses the same.
HAIKU_MODEL = "claude-haiku-4-5-20251001"

CLASSIFIER_BATCH_SIZE = 50
CACHE_TTL_DAYS = 7


# ────────────────────────────────────────────────────────────────────────────
# Greek↔Latin lookalikes + accent normalization (reused pattern)
# ────────────────────────────────────────────────────────────────────────────

_GREEK_TO_LATIN: Dict[str, str] = {
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "α": "a", "β": "b", "ε": "e", "ζ": "z", "η": "h", "ι": "i", "κ": "k",
    "μ": "m", "ν": "n", "ο": "o", "ρ": "p", "τ": "t", "υ": "y", "χ": "x",
}


def _strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    mapped = "".join(_GREEK_TO_LATIN.get(ch, ch) for ch in text)
    return " ".join(_strip_accents(mapped).lower().split())


# ────────────────────────────────────────────────────────────────────────────
# Subject facets
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SubjectFacets:
    """Decomposed subject — drives matching decisions across the pipeline."""
    label: str
    aliases: List[str] = field(default_factory=list)
    brand: Optional[str] = None
    product_type: Optional[str] = None
    must_have_tokens: List[str] = field(default_factory=list)
    competitor_brands: List[str] = field(default_factory=list)
    language_codes: List[str] = field(default_factory=lambda: ["en"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "aliases": self.aliases,
            "brand": self.brand,
            "product_type": self.product_type,
            "must_have_tokens": self.must_have_tokens,
            "competitor_brands": self.competitor_brands,
            "language_codes": self.language_codes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubjectFacets":
        return cls(
            label=data.get("label") or "",
            aliases=list(data.get("aliases") or []),
            brand=data.get("brand"),
            product_type=data.get("product_type"),
            must_have_tokens=list(data.get("must_have_tokens") or []),
            competitor_brands=list(data.get("competitor_brands") or []),
            language_codes=list(data.get("language_codes") or ["en"]),
        )

    def all_aliases(self) -> List[str]:
        """All strings that count as a hit when present in text."""
        seen = set()
        out: List[str] = []
        for s in [self.label, *self.aliases]:
            n = normalize_text(s)
            if n and n not in seen:
                seen.add(n)
                out.append(s)
        return out


def _facets_hash(facets: SubjectFacets) -> str:
    payload = json.dumps({
        "label": normalize_text(facets.label),
        "brand": normalize_text(facets.brand or ""),
        "type": normalize_text(facets.product_type or ""),
        "must": sorted([normalize_text(t) for t in facets.must_have_tokens]),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def content_hash(*, url: str, title: Optional[str], body: Optional[str]) -> str:
    """Hash the *content* (not URL) so syndicated copies dedupe."""
    base = "\n".join([
        (title or "").strip(),
        (body or "").strip()[:1500],
    ])
    if not base.strip():
        base = url.strip()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def alias_present(text: str, facets: SubjectFacets) -> bool:
    """Cheap deterministic check: does at least one alias appear in text?

    Single-word aliases use substring match. Multi-word aliases (e.g. brand +
    model phrases like "ORABELLA PRECIOSA") require ALL constituent words to
    be present — order and adjacency don't matter. This is critical for
    real-world coverage where news/blog articles split the words across a
    sentence ("The Orabella line by Preciosa...", "Preciosa's Orabella
    collection", etc.).

    Skips ultra-short tokens (≤2 chars) to avoid false positives from things
    like "6.1" matching every "6.1mm" reference. Numeric model codes still
    work because the multi-word path matches them as one of several tokens.
    """
    if not text:
        return False
    nt = normalize_text(text)
    for a in facets.all_aliases():
        n = normalize_text(a)
        if not n:
            continue
        words = [w for w in n.split() if len(w) >= 3]
        if not words:
            # All tokens too short — fall back to strict substring match
            if n in nt:
                return True
            continue
        if len(words) == 1:
            # Single-word alias: substring match (cheap)
            if words[0] in nt:
                return True
        else:
            # Multi-word alias: ALL words must appear, any order
            if all(w in nt for w in words):
                return True
    return False


# ────────────────────────────────────────────────────────────────────────────
# Facet extraction (Haiku, cached)
# ────────────────────────────────────────────────────────────────────────────

class MentionIdentityService:
    """Stateful only in the sense of holding the supabase client + http client."""

    def __init__(self) -> None:
        self.supabase = get_supabase_client()
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or ""

    # ───── Facet extraction ─────

    async def extract_facets(
        self,
        *,
        subject_label: str,
        subject_type: str,
        aliases_seed: Optional[List[str]] = None,
        brand_hint: Optional[str] = None,
        product_type_hint: Optional[str] = None,
        cached: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
    ) -> SubjectFacets:
        """
        Decompose the subject into facets.

        Default behavior (use_llm=False): build deterministic facets from the
        inputs only. No Haiku call. Discovery searches use the label + any
        aliases the caller supplied — nothing more, nothing less.

        Opt-in behavior (use_llm=True): run Haiku once to expand the label
        into per-word aliases, infer brand/product_type, and surface
        competitor brands. Broader recall, higher cost, dependency on
        Anthropic. Caller persists the result.

        If `cached` is provided (from tracked_mentions.subject_facets) we
        round-trip via from_dict regardless of use_llm.
        """
        if cached:
            try:
                return SubjectFacets.from_dict(cached)
            except Exception as e:
                logger.warning(f"mention-identity: cached facets parse failed, regenerating: {e}")

        # Deterministic path — default
        if not use_llm or not self.api_key:
            return SubjectFacets(
                label=subject_label,
                aliases=list(aliases_seed or []),
                brand=brand_hint,
                product_type=product_type_hint,
                must_have_tokens=[t for t in [brand_hint, subject_label] if t],
            )

        # One Haiku call — extract aliases, must-have tokens, competitor brands.
        prompt = self._facet_prompt(
            subject_label=subject_label,
            subject_type=subject_type,
            aliases_seed=aliases_seed or [],
            brand_hint=brand_hint,
            product_type_hint=product_type_hint,
        )
        try:
            text = await self._haiku_completion(prompt, max_tokens=600)
        except Exception as e:
            logger.warning(f"mention-identity: facet Haiku call failed: {e}")
            return SubjectFacets(
                label=subject_label,
                aliases=list(aliases_seed or []),
                brand=brand_hint,
                product_type=product_type_hint,
                must_have_tokens=[t for t in [brand_hint, subject_label] if t],
            )

        try:
            payload = self._parse_json_block(text)
            facets = SubjectFacets(
                label=subject_label,
                aliases=list(payload.get("aliases") or aliases_seed or []),
                brand=payload.get("brand") or brand_hint,
                product_type=payload.get("product_type") or product_type_hint,
                must_have_tokens=list(payload.get("must_have_tokens") or []),
                competitor_brands=list(payload.get("competitor_brands") or []),
                language_codes=list(payload.get("language_codes") or ["en"]),
            )
            if not facets.must_have_tokens:
                facets.must_have_tokens = [t for t in [facets.brand, subject_label] if t]
            return facets
        except Exception as e:
            logger.warning(f"mention-identity: facet parse failed: {e}")
            return SubjectFacets(
                label=subject_label,
                aliases=list(aliases_seed or []),
                brand=brand_hint,
                product_type=product_type_hint,
                must_have_tokens=[t for t in [brand_hint, subject_label] if t],
            )

    # ───── Rule-based pre-filter ─────

    def rule_prefilter(self, *, candidate: Dict[str, Any], facets: SubjectFacets) -> Optional[Dict[str, Any]]:
        """Return a verdict when unambiguous. Otherwise None (caller defers to Haiku)."""
        text = " ".join([
            candidate.get("title") or "",
            candidate.get("excerpt") or "",
            (candidate.get("body_md") or "")[:600],
        ])
        if not alias_present(text, facets):
            return {
                "relevance": "mismatch",
                "sentiment": "neutral",
                "relevance_score": 0.95,
                "match_note": "alias not present in title/excerpt",
            }
        return None

    # ───── Verdict cache lookup + write ─────

    def cache_lookup(self, *, content_hash_str: str, facets_hash_str: str) -> Optional[Dict[str, Any]]:
        cache_key = hashlib.sha1(f"{content_hash_str}|{facets_hash_str}".encode("utf-8")).hexdigest()
        try:
            resp = (
                self.supabase.client.table("mention_classifier_verdict_cache")
                .select("sentiment, sentiment_score, relevance, relevance_score, match_note, expires_at")
                .eq("cache_key", cache_key)
                .gte("expires_at", datetime.now(timezone.utc).isoformat())
                .maybe_single()
                .execute()
            )
            data = resp.data if resp else None
            return data
        except Exception as e:
            logger.debug(f"mention-identity: cache lookup miss: {e}")
            return None

    def cache_write(
        self,
        *,
        content_hash_str: str,
        facets_hash_str: str,
        verdict: Dict[str, Any],
    ) -> None:
        cache_key = hashlib.sha1(f"{content_hash_str}|{facets_hash_str}".encode("utf-8")).hexdigest()
        try:
            self.supabase.client.table("mention_classifier_verdict_cache").upsert({
                "cache_key": cache_key,
                "content_hash": content_hash_str,
                "subject_facets_hash": facets_hash_str,
                "sentiment": verdict.get("sentiment"),
                "sentiment_score": verdict.get("sentiment_score"),
                "relevance": verdict.get("relevance"),
                "relevance_score": verdict.get("relevance_score"),
                "match_note": verdict.get("match_note"),
            }, on_conflict="cache_key").execute()
        except Exception as e:
            logger.debug(f"mention-identity: cache write failed (non-fatal): {e}")

    # ───── Batched classifier ─────

    async def classify_batch(
        self,
        *,
        candidates: List[Dict[str, Any]],
        facets: SubjectFacets,
    ) -> List[Dict[str, Any]]:
        """
        Classify each candidate. Returns parallel list of verdicts:
          { relevance, relevance_score, sentiment, sentiment_score, match_note, classifier_cached }
        """
        if not candidates:
            return []

        facets_hash_str = _facets_hash(facets)
        verdicts: List[Optional[Dict[str, Any]]] = [None] * len(candidates)
        misses: List[Tuple[int, Dict[str, Any], str]] = []

        for i, c in enumerate(candidates):
            # Rule prefilter
            rv = self.rule_prefilter(candidate=c, facets=facets)
            if rv is not None:
                verdicts[i] = {**rv, "classifier_cached": False}
                continue

            ch = c.get("content_hash") or content_hash(
                url=c.get("url") or "",
                title=c.get("title"),
                body=c.get("body_md") or c.get("excerpt"),
            )
            cached = self.cache_lookup(content_hash_str=ch, facets_hash_str=facets_hash_str)
            if cached:
                verdicts[i] = {
                    "relevance": cached.get("relevance"),
                    "relevance_score": cached.get("relevance_score"),
                    "sentiment": cached.get("sentiment"),
                    "sentiment_score": cached.get("sentiment_score"),
                    "match_note": cached.get("match_note"),
                    "classifier_cached": True,
                }
                continue
            misses.append((i, c, ch))

        # Few-shot examples from recent corrections
        few_shot = self._recent_corrections(limit=5)

        for chunk_start in range(0, len(misses), CLASSIFIER_BATCH_SIZE):
            chunk = misses[chunk_start:chunk_start + CLASSIFIER_BATCH_SIZE]
            try:
                results = await self._classify_chunk(
                    candidates=[item[1] for item in chunk],
                    facets=facets,
                    few_shot=few_shot,
                )
            except Exception as e:
                logger.warning(f"mention-identity: classifier batch failed: {e}")
                results = [
                    {"relevance": "unverifiable", "sentiment": "neutral",
                     "relevance_score": 0.5, "sentiment_score": 0.0,
                     "match_note": f"classifier failed: {str(e)[:100]}"}
                    for _ in chunk
                ]

            for (idx, _, ch), v in zip(chunk, results):
                verdict = {**v, "classifier_cached": False}
                verdicts[idx] = verdict
                # Write to cache for next time
                self.cache_write(
                    content_hash_str=ch,
                    facets_hash_str=facets_hash_str,
                    verdict=verdict,
                )

        # Fill any holes with safe defaults
        out: List[Dict[str, Any]] = []
        for v in verdicts:
            if v is None:
                v = {
                    "relevance": "unverifiable",
                    "sentiment": "neutral",
                    "relevance_score": 0.0,
                    "sentiment_score": 0.0,
                    "match_note": "no verdict",
                    "classifier_cached": False,
                }
            out.append(v)
        return out

    # ───── Internal: anthropic completion ─────

    async def _haiku_completion(self, prompt: str, max_tokens: int = 1000) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{ANTHROPIC_API_BASE}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": HAIKU_MODEL,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            blocks = data.get("content") or []
            text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            return "\n".join(text_parts).strip()

    def _parse_json_block(self, text: str) -> Dict[str, Any]:
        # Strip ```json fences
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        # Find the first balanced JSON object
        start = s.find("{")
        if start == -1:
            raise ValueError("no JSON object found")
        depth = 0
        end = start
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        return json.loads(s[start:end + 1])

    def _facet_prompt(
        self, *, subject_label: str, subject_type: str,
        aliases_seed: List[str], brand_hint: Optional[str], product_type_hint: Optional[str],
    ) -> str:
        seed = "\n".join(f"  - {a}" for a in aliases_seed) if aliases_seed else "  (none)"
        return f"""You are extracting facets for a mention-tracking subject.

Subject label: {subject_label}
Subject type: {subject_type}   (one of: product | brand | keyword)
Brand hint: {brand_hint or '(none)'}
Product type hint: {product_type_hint or '(none)'}
Seed aliases:
{seed}

Return ONLY a JSON object with these keys:
- aliases: array of strings that should count as a hit when found in text. Include:
  * The original full label.
  * Each meaningful word in the label as a separate entry (e.g. label "ORABELLA PRECIOSA" → also include "ORABELLA" and "PRECIOSA" as separate aliases, since articles often mention only one).
  * Common reorderings (e.g. "PRECIOSA ORABELLA" if the words can swap).
  * Alternate spellings, abbreviations, model SKUs, brand variants.
  Skip generic words ("the", "tile", "collection"). Aim for 4-8 entries.
- brand: the brand string if applicable, else null.
- product_type: the noun-category if applicable (e.g. "porcelain tile", "console table"), else null.
- must_have_tokens: array of strings that MUST appear in a candidate page for it to count as a real mention. Usually = brand + model token. Keep tight (1-3 items).
- competitor_brands: array of brands that compete in the same space — used to detect "comparison" mentions.
- language_codes: array of ISO 639-1 codes the subject is likely discussed in. Default ["en"].

Be terse. No prose. Just JSON.
"""

    async def _classify_chunk(
        self, *,
        candidates: List[Dict[str, Any]],
        facets: SubjectFacets,
        few_shot: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return [
                {"relevance": "unverifiable", "sentiment": "neutral",
                 "relevance_score": 0.5, "sentiment_score": 0.0, "match_note": "no api key"}
                for _ in candidates
            ]

        items_json = json.dumps([
            {
                "i": i,
                "title": (c.get("title") or "")[:280],
                "excerpt": (c.get("excerpt") or "")[:300],
                "body": (c.get("body_md") or "")[:600],
                "url": c.get("url") or "",
                "outlet_domain": c.get("outlet_domain") or "",
            }
            for i, c in enumerate(candidates)
        ], ensure_ascii=False)

        few_shot_block = ""
        if few_shot:
            few_shot_block = (
                "\nRecent admin corrections (treat as ground truth):\n"
                + json.dumps(few_shot, ensure_ascii=False)
            )

        prompt = f"""Classify each candidate mention against the subject facets.

Subject facets:
{json.dumps(facets.to_dict(), ensure_ascii=False)}

Definitions:
- relevance: "exact" (clearly about THIS subject), "tangential" (mentions subject in passing), "mismatch" (different product/brand), "unverifiable" (cannot tell).
- sentiment: "positive" / "neutral" / "negative" — overall tone toward the subject. If subject is a passing mention, sentiment is for the surrounding context only.
- match_note: 1 short sentence (<=120 chars) explaining the verdict.

Candidates:
{items_json}
{few_shot_block}

Return ONLY a JSON array of length {len(candidates)}, in input order. Each entry:
{{"relevance": "...", "relevance_score": 0.0-1.0, "sentiment": "...", "sentiment_score": -1.0-1.0, "match_note": "..."}}
"""
        text = await self._haiku_completion(prompt, max_tokens=2000)
        # Parse top-level array
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        start = s.find("[")
        if start == -1:
            raise ValueError("no JSON array")
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
        arr = json.loads(s[start:end + 1])
        # Defensive: pad/truncate to expected length
        out: List[Dict[str, Any]] = []
        for i in range(len(candidates)):
            v = arr[i] if i < len(arr) else {}
            out.append({
                "relevance": v.get("relevance") or "unverifiable",
                "relevance_score": float(v.get("relevance_score") or 0.5),
                "sentiment": v.get("sentiment") or "neutral",
                "sentiment_score": float(v.get("sentiment_score") or 0.0),
                "match_note": (v.get("match_note") or "")[:200],
            })
        return out

    def _recent_corrections(self, *, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            resp = (
                self.supabase.client.table("mention_match_corrections")
                .select("title, original_relevance, corrected_relevance, original_sentiment, corrected_sentiment, correction_note")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception:
            return []


_service: Optional[MentionIdentityService] = None


def get_mention_identity_service() -> MentionIdentityService:
    global _service
    if _service is None:
        _service = MentionIdentityService()
    return _service
