"""
LLM Mention Probe Service — measures how subjects appear in AI answers.

Runs a fixed bank of probe templates against frontier models, then post-
processes responses with a Haiku tool-use call to extract:
  - mentioned: bool
  - position: rank (1-based) when listed
  - sentiment: positive | neutral | negative
  - competitors_mentioned: list of competitor names
  - context_snippet: the sentence containing the mention

Cost discipline:
  - Default 4 templates × 4 cheap models = 16 calls/subject/cycle
  - Only "cheap" tier of each provider (haiku, gpt-4o-mini, gemini-flash, sonar)
  - Frontier models OPT-IN via probe_template_overrides on the tracked_mentions row
  - Weekly cadence by default
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.mention_identity_service import (
    SubjectFacets, normalize_text,
)

logger = logging.getLogger(__name__)


ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
OPENAI_API = "https://api.openai.com/v1/chat/completions"
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models"
PERPLEXITY_API = "https://api.perplexity.ai/chat/completions"

# Cheap-tier IDs only — Haiku uses dated form because we hit Anthropic's HTTP API directly
HAIKU = "claude-haiku-4-5-20251001"
GPT4O_MINI = "gpt-4o-mini"
GEMINI_FLASH = "gemini-2.0-flash"
SONAR = "sonar"

# Token-cost in USD per 1K tokens (input, output) — for ai_usage_logs
COST_TABLE: Dict[str, Dict[str, float]] = {
    HAIKU: {"input": 0.001, "output": 0.005},
    "claude-haiku-4-5": {"input": 0.001, "output": 0.005},  # alias still tracked
    GPT4O_MINI: {"input": 0.00015, "output": 0.0006},
    GEMINI_FLASH: {"input": 0.00010, "output": 0.0004},
    SONAR: {"input": 0.0010, "output": 0.0010},
}


# ────────────────────────────────────────────────────────────────────────────
# Probe templates
# ────────────────────────────────────────────────────────────────────────────

def build_probes(facets: SubjectFacets) -> List[Dict[str, str]]:
    """4 probe templates per subject. Caller may add more via source_config."""
    label = facets.label
    ptype = facets.product_type or "products"
    brand = facets.brand or label
    competitor_str = ", ".join(facets.competitor_brands[:3]) if facets.competitor_brands else "leading alternatives"

    return [
        {
            "key": "generic_recommendation",
            "prompt": (
                f"What are the best {ptype} brands? "
                "Give a ranked list with one short reason per entry."
            ),
        },
        {
            "key": "use_case",
            "prompt": (
                f"Recommend 5 {ptype} for use in a high-traffic commercial space. "
                "Name them and briefly explain each."
            ),
        },
        {
            "key": "comparison",
            "prompt": (
                f"Compare {brand} with {competitor_str}. "
                "Cover product range, quality, and typical price tier."
            ),
        },
        {
            "key": "direct_lookup",
            "prompt": (
                f"Tell me about {label}. What do they make and what are they known for?"
            ),
        },
    ]


# ────────────────────────────────────────────────────────────────────────────
# Service
# ────────────────────────────────────────────────────────────────────────────

class LlmMentionProbeService:
    def __init__(self) -> None:
        self.supabase = get_supabase_client()
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY") or ""
        self.openai_key = os.getenv("OPENAI_API_KEY") or ""
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY") or ""
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY") or ""

    def enabled_models(self) -> List[str]:
        out: List[str] = []
        if self.anthropic_key:
            out.append(HAIKU)
        if self.openai_key:
            out.append(GPT4O_MINI)
        if self.gemini_key:
            out.append(GEMINI_FLASH)
        if self.perplexity_key:
            out.append(SONAR)
        return out

    async def probe(
        self, *,
        tracked_mention_id: str,
        facets: SubjectFacets,
        models: Optional[List[str]] = None,
        templates: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        models_to_use = models or self.enabled_models()
        if not models_to_use:
            return {"status": "no_models_enabled", "probes": [], "credits_used": 0}

        probes = templates or build_probes(facets)
        rows: List[Dict[str, Any]] = []
        total_cost = 0.0

        for p in probes:
            for model in models_to_use:
                try:
                    text, in_tok, out_tok, latency_ms, err = await self._call_model(
                        model=model, prompt=p["prompt"],
                    )
                except Exception as e:
                    text, in_tok, out_tok, latency_ms, err = ("", 0, 0, 0, str(e))

                cost = self._cost(model, in_tok, out_tok)
                total_cost += cost

                # Post-process: extract structured signal via Haiku tool use
                extraction = await self._extract(
                    response_text=text or "", facets=facets, model_used=model,
                )
                rows.append({
                    "tracked_mention_id": tracked_mention_id,
                    "probe_run_id": run_id,
                    "probe_template_key": p["key"],
                    "prompt_text": p["prompt"],
                    "model": model,
                    "response_text": (text or "")[:6000],
                    "mentioned": extraction.get("mentioned"),
                    "position": extraction.get("position"),
                    "sentiment": extraction.get("sentiment"),
                    "competitors_mentioned": extraction.get("competitors_mentioned") or [],
                    "context_snippet": extraction.get("context_snippet"),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                    "latency_ms": latency_ms,
                    "error": err,
                })

        try:
            for i in range(0, len(rows), 25):
                self.supabase.client.table("llm_mention_probes").insert(rows[i:i + 25]).execute()
        except Exception as e:
            logger.error(f"llm_probes insert failed: {e}")

        return {
            "status": "completed",
            "probe_run_id": run_id,
            "probe_count": len(rows),
            "models": models_to_use,
            "total_cost_usd": total_cost,
        }

    # ───── Visibility analytics ─────

    def visibility_snapshot(
        self, tracked_mention_id: str, *, run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Latest probe run aggregated → share-of-voice + position trend."""
        try:
            if run_id:
                q = (
                    self.supabase.client.table("llm_mention_probes")
                    .select("*")
                    .eq("tracked_mention_id", tracked_mention_id)
                    .eq("probe_run_id", run_id)
                )
            else:
                latest = (
                    self.supabase.client.table("llm_mention_probes")
                    .select("probe_run_id, run_at")
                    .eq("tracked_mention_id", tracked_mention_id)
                    .order("run_at", desc=True)
                    .limit(1)
                    .execute()
                )
                if not latest.data:
                    return {"present": False}
                run_id = (latest.data[0] or {}).get("probe_run_id")
                q = (
                    self.supabase.client.table("llm_mention_probes")
                    .select("*")
                    .eq("tracked_mention_id", tracked_mention_id)
                    .eq("probe_run_id", run_id)
                )
            r = q.execute()
            rows = r.data or []
        except Exception as e:
            logger.warning(f"visibility_snapshot read failed: {e}")
            return {"present": False, "error": str(e)}

        per_model: Dict[str, Dict[str, Any]] = {}
        competitors: Dict[str, int] = {}
        positions: List[int] = []
        for row in rows:
            m = row.get("model")
            d = per_model.setdefault(m, {"probes": 0, "mentioned": 0, "positions": []})
            d["probes"] += 1
            if row.get("mentioned"):
                d["mentioned"] += 1
                if row.get("position"):
                    d["positions"].append(int(row["position"]))
                    positions.append(int(row["position"]))
            for c in row.get("competitors_mentioned") or []:
                cn = (c or "").strip()
                if cn:
                    competitors[cn] = competitors.get(cn, 0) + 1

        total_probes = len(rows)
        total_mentioned = sum(1 for r in rows if r.get("mentioned"))
        return {
            "present": True,
            "probe_run_id": run_id,
            "total_probes": total_probes,
            "share_of_voice": (total_mentioned / total_probes) if total_probes else 0.0,
            "avg_position": (sum(positions) / len(positions)) if positions else None,
            "per_model": per_model,
            "top_competitors": sorted(competitors.items(), key=lambda kv: kv[1], reverse=True)[:10],
        }

    # ───── Internal: model calls ─────

    async def _call_model(
        self, *, model: str, prompt: str,
    ) -> tuple[str, int, int, int, Optional[str]]:
        start = time.time()
        try:
            if model == HAIKU:
                return await self._call_anthropic(prompt, model=HAIKU, start=start)
            if model == GPT4O_MINI:
                return await self._call_openai(prompt, model=GPT4O_MINI, start=start)
            if model == GEMINI_FLASH:
                return await self._call_gemini(prompt, model=GEMINI_FLASH, start=start)
            if model == SONAR:
                return await self._call_perplexity(prompt, model=SONAR, start=start)
            return "", 0, 0, 0, f"unsupported model {model}"
        except httpx.HTTPStatusError as e:
            return "", 0, 0, int((time.time() - start) * 1000), f"HTTP {e.response.status_code}"
        except Exception as e:
            return "", 0, 0, int((time.time() - start) * 1000), str(e)[:200]

    async def _call_anthropic(
        self, prompt: str, *, model: str, start: float,
    ) -> tuple[str, int, int, int, Optional[str]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                ANTHROPIC_API,
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            blocks = data.get("content") or []
            text = "\n".join([b.get("text", "") for b in blocks if b.get("type") == "text"]).strip()
            usage = data.get("usage") or {}
            return (
                text,
                int(usage.get("input_tokens") or 0),
                int(usage.get("output_tokens") or 0),
                int((time.time() - start) * 1000),
                None,
            )

    async def _call_openai(
        self, prompt: str, *, model: str, start: float,
    ) -> tuple[str, int, int, int, Optional[str]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                OPENAI_API,
                headers={"Authorization": f"Bearer {self.openai_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            u = data.get("usage") or {}
            return (
                text.strip(),
                int(u.get("prompt_tokens") or 0),
                int(u.get("completion_tokens") or 0),
                int((time.time() - start) * 1000),
                None,
            )

    async def _call_gemini(
        self, prompt: str, *, model: str, start: float,
    ) -> tuple[str, int, int, int, Optional[str]]:
        url = f"{GEMINI_API}/{model}:generateContent?key={self.gemini_key}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 800},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            cands = data.get("candidates") or []
            parts = ((cands[0] or {}).get("content") or {}).get("parts") if cands else []
            text = "".join(p.get("text", "") for p in (parts or []))
            u = data.get("usageMetadata") or {}
            return (
                text.strip(),
                int(u.get("promptTokenCount") or 0),
                int(u.get("candidatesTokenCount") or 0),
                int((time.time() - start) * 1000),
                None,
            )

    async def _call_perplexity(
        self, prompt: str, *, model: str, start: float,
    ) -> tuple[str, int, int, int, Optional[str]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                PERPLEXITY_API,
                headers={"Authorization": f"Bearer {self.perplexity_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            u = data.get("usage") or {}
            return (
                text.strip(),
                int(u.get("prompt_tokens") or 0),
                int(u.get("completion_tokens") or 0),
                int((time.time() - start) * 1000),
                None,
            )

    # ───── Internal: extraction (Haiku) ─────

    async def _extract(
        self, *, response_text: str, facets: SubjectFacets, model_used: str,
    ) -> Dict[str, Any]:
        """Use Haiku to extract structured signal from a model's free-text response.
        Falls back to deterministic parsing if Haiku is unavailable."""
        if not response_text or not response_text.strip():
            return {"mentioned": False, "position": None, "sentiment": "neutral",
                    "competitors_mentioned": [], "context_snippet": None}

        # Deterministic fallback
        if not self.anthropic_key:
            return self._extract_deterministic(response_text, facets)

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
                        "model": HAIKU,
                        "max_tokens": 500,
                        "tools": [{
                            "name": "record_mention",
                            "description": "Record whether and how the subject appears in the answer.",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "mentioned": {"type": "boolean"},
                                    "position": {"type": ["integer", "null"]},
                                    "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                                    "competitors_mentioned": {"type": "array", "items": {"type": "string"}},
                                    "context_snippet": {"type": ["string", "null"]},
                                },
                                "required": ["mentioned", "sentiment", "competitors_mentioned"],
                            },
                        }],
                        "tool_choice": {"type": "tool", "name": "record_mention"},
                        "messages": [{
                            "role": "user",
                            "content": (
                                f"Subject we are tracking: {facets.label}\n"
                                f"Aliases: {', '.join(facets.aliases[:5])}\n"
                                f"Brand (if any): {facets.brand or '(none)'}\n\n"
                                f"Model response to analyze:\n{response_text[:4000]}\n\n"
                                "Determine: was the subject mentioned? At what rank (1-based) in any "
                                "list? Sentiment of the surrounding context? Other brands mentioned in "
                                "the same answer (competitors). One short context snippet."
                            ),
                        }],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.debug(f"llm-probe extract Haiku failed: {e}")
            return self._extract_deterministic(response_text, facets)

        for block in (data.get("content") or []):
            if block.get("type") == "tool_use" and block.get("name") == "record_mention":
                inp = block.get("input") or {}
                return {
                    "mentioned": bool(inp.get("mentioned")),
                    "position": inp.get("position"),
                    "sentiment": inp.get("sentiment") or "neutral",
                    "competitors_mentioned": inp.get("competitors_mentioned") or [],
                    "context_snippet": (inp.get("context_snippet") or "")[:400] or None,
                }
        return self._extract_deterministic(response_text, facets)

    def _extract_deterministic(self, text: str, facets: SubjectFacets) -> Dict[str, Any]:
        nt = normalize_text(text)
        mentioned = any(normalize_text(a) in nt for a in facets.all_aliases())
        # Extract rank from numbered list
        position = None
        if mentioned:
            for line in text.splitlines():
                m = re.match(r"\s*(\d+)[.):]\s*(.+)", line)
                if m and any(normalize_text(a) in normalize_text(m.group(2)) for a in facets.all_aliases()):
                    try:
                        position = int(m.group(1))
                        break
                    except Exception:
                        pass
        # Naive sentiment: keyword scan
        pos_words = {"best", "excellent", "premium", "highly recommended", "top", "leader"}
        neg_words = {"avoid", "poor", "bad", "issue", "problem", "expensive"}
        sentiment = "neutral"
        if mentioned:
            score = sum(1 for w in pos_words if w in nt) - sum(1 for w in neg_words if w in nt)
            sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        return {
            "mentioned": mentioned,
            "position": position,
            "sentiment": sentiment,
            "competitors_mentioned": list(facets.competitor_brands)[:5],
            "context_snippet": None,
        }

    def _cost(self, model: str, in_tok: int, out_tok: int) -> float:
        prices = COST_TABLE.get(model)
        if not prices:
            return 0.0
        return (in_tok / 1000.0) * prices["input"] + (out_tok / 1000.0) * prices["output"]


_service: Optional[LlmMentionProbeService] = None


def get_llm_mention_probe_service() -> LlmMentionProbeService:
    global _service
    if _service is None:
        _service = LlmMentionProbeService()
    return _service
