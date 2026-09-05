"""
LLM Mention Probe Service — measures how subjects appear in AI answers.

Runs a fixed bank of probe templates against frontier models, then post-
processes responses with a Haiku tool-use call to extract:
  - mentioned: bool
  - position: rank (1-based) when listed
  - sentiment: positive | neutral | negative
  - competitors_mentioned: list of competitor names
  - context_snippet: the sentence containing the mention
  - cited_urls: the sources the answer pointed at (native for Sonar, extracted
    from the prose for the models that inline their links)

`cited_urls` + `brand_cited` are what make a GHOST CITATION visible: our page used
as a source while the brand is never named in the answer. That is invisible to a
mention count, and it is the single measurement this pipeline was missing.

Cost discipline:
  - Default 4 templates × the CHEAP tier = up to 12 calls/subject/cycle
  - Frontier models are OPT-IN per subject via `tracked_mentions.probe_tier`
  - Weekly cadence by default

The docstring used to promise the opt-in came through `probe_template_overrides`. That
name existed in this comment and nowhere else — no column, no code path, no caller. It is
`probe_tier` now, and it is real (#349 A7).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, NamedTuple, Optional

import httpx

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.llm_probe_templates import build_probes
from app.services.integrations.llm_visibility_math import (
    citation_rollup, dedupe_urls, domain_is_ours,
    sentiment_rollup, share_of_voice_from_rows, trend_from_rows,
    visibility_rollup,
)
from app.services.integrations.mention_identity_service import (
    SubjectFacets, normalize_text,
)
from app.services.integrations.mention_cost_logger import (
    CostAttribution, log_llm_probe_call, log_haiku_call,
    recompute_lifetime_cost,
)
from app.services.integrations.platform_secret_resolver import resolve_secret
from app.services.utilities.prompt_registry import load_prompt, render

logger = logging.getLogger(__name__)


ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models"
PERPLEXITY_API = "https://api.perplexity.ai/chat/completions"
# Chat completions ONLY. The 2026-08-23 removal of OpenAI was total (package, clients,
# embeddings fallback); ChatGPT came back on 2026-09-05 as a chat provider for this probe
# and nothing else — httpx, this one endpoint, this one file. tests/unit/
# test_no_fallback_embedder.py allowlists exactly that and still fails the build on the
# package import or the embeddings endpoint anywhere.
OPENAI_API = "https://api.openai.com/v1/chat/completions"

# Cheap tier — Haiku uses the dated form because we hit Anthropic's HTTP API directly
HAIKU = "claude-haiku-4-5-20251001"
GEMINI_FLASH = "gemini-2.0-flash"
SONAR = "sonar"
GPT_MINI = "gpt-5-mini"

# Frontier tier (#349 A7) — the models a person is actually answered by.
#
# Every one of these has a row in `ai_model_pricing`, which is deliberate: an unpriced
# model falls through to that config's conservative default, and a frontier model costed
# at a cheap-tier guess under-reports spend by more than an order of magnitude. If you add
# a model here, add its price row first.
#
# ChatGPT is the largest answer engine and was the one this probe could not ask between
# 2026-08-23 and 2026-09-05 (its earlier `gpt-4o-mini` had produced 212 rows and 212
# failures against an account that never worked). It is back as a deliberate provider
# decision: a working key under OPENAI_API_KEY, price rows for both models, and the
# roster route reporting "no key configured" rather than silently dropping it.
OPUS = "claude-opus-5"
GEMINI_PRO = "gemini-3.1-pro"
SONAR_PRO = "sonar-pro"
GPT = "gpt-5"

CHEAP_TIER = "cheap"
FRONTIER_TIER = "frontier"

#: Which models each tier asks for. What actually runs is this ∩ the configured keys.
TIER_MODELS: Dict[str, List[str]] = {
    CHEAP_TIER: [HAIKU, GPT_MINI, GEMINI_FLASH, SONAR],
    FRONTIER_TIER: [OPUS, GPT, GEMINI_PRO, SONAR_PRO],
}

# Token-cost in USD per 1K tokens (input, output) — for ai_usage_logs
COST_TABLE: Dict[str, Dict[str, float]] = {
    HAIKU: {"input": 0.001, "output": 0.005},
    "claude-haiku-4-5": {"input": 0.001, "output": 0.005},  # alias still tracked
    GEMINI_FLASH: {"input": 0.00010, "output": 0.0004},
    SONAR: {"input": 0.0010, "output": 0.0010},
    GPT_MINI: {"input": 0.00025, "output": 0.002},
    # Frontier. These are a LOCAL ESTIMATE for the `total_cost_usd` the probe reports back
    # to the caller; the number that reaches `ai_usage_logs` is resolved from
    # `ai_model_pricing` by `log_llm_probe_call`, which is the platform's single USD
    # source. Two numbers with two jobs — do not let this one become a second ledger.
    OPUS: {"input": 0.005, "output": 0.025},
    GEMINI_PRO: {"input": 0.00125, "output": 0.010},
    SONAR_PRO: {"input": 0.003, "output": 0.015},
    GPT: {"input": 0.00125, "output": 0.010},
}


# ────────────────────────────────────────────────────────────────────────────
# Citations
# ────────────────────────────────────────────────────────────────────────────

def _describe_http_error(e: httpx.HTTPStatusError) -> str:
    """`HTTP 429` on its own cannot tell a rate limit from an unfunded account.

    OpenAI answers both with 429 — `insufficient_quota` / `credit_balance_exhausted` is
    "add credits", a plain 429 is "slow down" — and 212 ChatGPT probes were once read as
    "assistants never mention us" on the strength of the bare number. The recorded error
    keeps the `HTTP <status>` prefix (what the rollup counts on) and carries the
    provider's own code and message, bounded, so the panel can say WHY there is no verdict.
    """
    detail = ""
    try:
        body = e.response.json()
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            code = err.get("code") or err.get("type") or ""
            detail = " ".join(str(x) for x in (code, err.get("message") or "") if x).strip()
        elif isinstance(err, str):
            detail = err
        elif isinstance(body, dict) and body.get("message"):
            detail = str(body["message"])
    except Exception:
        detail = (e.response.text or "").strip()
    head = f"HTTP {e.response.status_code}"
    return f"{head} {detail[:160]}".strip() if detail else head


class ModelReply(NamedTuple):
    """One answering model's reply. Was a bare 5-tuple until citations arrived —
    a NamedTuple so the next field does not silently shift every unpack."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    error: Optional[str]
    citations: List[str]


# ────────────────────────────────────────────────────────────────────────────
# Service
# ────────────────────────────────────────────────────────────────────────────

class LlmMentionProbeService:
    """Probe matrix runner.

    KEYS ARE RESOLVED, NOT CAPTURED. This class used to read `os.getenv` for all four
    providers in `__init__`, and `get_llm_mention_probe_service()` returns a module-level
    singleton — so the four keys were read once per worker process, from env only.

    Two consequences, both silent:

      - `platform_secrets` was never consulted. An admin pasting a key into
        /admin → Keys saved it correctly and this service could not see it, which is the
        same shape as the Zernio outage (env-only read of an admin-editable secret).
        `GEMINI_API_KEY` is empty in that table today and Gemini has produced ZERO probe
        rows since the feature shipped — not failures, none at all, because a provider
        with no key never enters `enabled_models()` and therefore never appears in the
        matrix it is advertised as part of.
      - Even a real env value was captured at first use, so a key added later needed a
        process restart to take effect.

    `resolve_secret` is env-first with a DB fallback and a 30s cache, so per-call
    resolution is both correct and cheap.
    """

    def __init__(self) -> None:
        self.supabase = get_supabase_client()

    @property
    def anthropic_key(self) -> str:
        return resolve_secret("ANTHROPIC_API_KEY").value or ""

    @property
    def gemini_key(self) -> str:
        # Three names for one credential, kept because deployments use all of them: the
        # Supabase edge runtime holds it as GOOGLE_GENERATIVE_AI_API_KEY (the AI SDK's
        # name), which is why Gemini ran for image generation on the edge while this
        # service, reading only the first two, listed it as "no key" (2026-09-05).
        return (
            resolve_secret("GEMINI_API_KEY").value
            or resolve_secret("GOOGLE_GENAI_API_KEY").value
            or resolve_secret("GOOGLE_GENERATIVE_AI_API_KEY").value
            or ""
        )

    @property
    def perplexity_key(self) -> str:
        return resolve_secret("PERPLEXITY_API_KEY").value or ""

    @property
    def openai_key(self) -> str:
        return resolve_secret("OPENAI_API_KEY").value or ""

    def key_sources(self) -> Dict[str, str]:
        """Where each provider's key came from — 'env', 'db', 'default' or 'missing'.

        Exposed so "why is this provider not in my probe matrix" is answerable without
        shell access to the host. A missing key is the commonest cause and the least
        visible: the provider simply does not appear.
        """
        gemini_source = "missing"
        for name in ("GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
            hit = resolve_secret(name)
            if hit.value:
                gemini_source = hit.source
                break
        return {
            "anthropic": resolve_secret("ANTHROPIC_API_KEY").source,
            "openai": resolve_secret("OPENAI_API_KEY").source,
            "gemini": gemini_source,
            "perplexity": resolve_secret("PERPLEXITY_API_KEY").source,
        }

    def roster(self) -> Dict[str, Any]:
        """Every model each tier asks for, and whether this deployment can run it.

        The probe matrix is `tier ∩ configured keys`, and a model dropped for a missing
        key leaves no row anywhere — the report simply has one assistant fewer, which
        reads as "we only measure Claude" rather than "Gemini's key is not set". This
        is the answer to that question, without shell access to the host.
        """
        sources = self.key_sources()
        provider_of = {
            HAIKU: "anthropic", OPUS: "anthropic",
            GPT_MINI: "openai", GPT: "openai",
            GEMINI_FLASH: "gemini", GEMINI_PRO: "gemini",
            SONAR: "perplexity", SONAR_PRO: "perplexity",
        }
        return {
            "tiers": {
                tier: [
                    {
                        "model": m,
                        "provider": provider_of.get(m, "unknown"),
                        "key_source": sources.get(provider_of.get(m, ""), "missing"),
                        "enabled": bool(self._key_for(m)),
                    }
                    for m in models
                ]
                for tier, models in TIER_MODELS.items()
            },
            "key_sources": sources,
        }

    def _key_for(self, model: str) -> str:
        """The credential a model needs. One place, so a new model cannot be added to a
        tier and then silently skipped because nothing knew which key it wanted."""
        if model in (HAIKU, OPUS):
            return self.anthropic_key
        if model in (GPT_MINI, GPT):
            return self.openai_key
        if model in (GEMINI_FLASH, GEMINI_PRO):
            return self.gemini_key
        if model in (SONAR, SONAR_PRO):
            return self.perplexity_key
        return ""

    def enabled_models(self, tier: str = CHEAP_TIER) -> List[str]:
        """The tier's models, minus the ones this deployment has no key for.

        An unknown tier falls back to CHEAP rather than to nothing: returning an empty
        list would make a typo look like "no models are configured", which is the same
        shape as a probe that ran and found nothing.
        """
        wanted = TIER_MODELS.get(tier) or TIER_MODELS[CHEAP_TIER]
        return [m for m in wanted if self._key_for(m)]

    async def probe(
        self, *,
        tracked_mention_id: str,
        facets: SubjectFacets,
        models: Optional[List[str]] = None,
        templates: Optional[List[Dict[str, str]]] = None,
        attribution: Optional[CostAttribution] = None,
        homepage_domain: Optional[str] = None,
        tier: str = CHEAP_TIER,
        custom_probes: Any = None,
        include_default_probes: bool = True,
    ) -> Dict[str, Any]:
        """Run the probe matrix and persist one row per (template × model).

        `homepage_domain` is the subject's own domain (the `homepage_domain` column
        on `tracked_mentions`, passed in rather than copied into `subject_facets` so
        there is one place it is edited). Without it `brand_cited` stays NULL —
        undecidable, not false.
        """
        run_id = str(uuid.uuid4())
        models_to_use = models or self.enabled_models(tier)
        if not models_to_use:
            return {"status": "no_models_enabled", "probes": [], "credits_used": 0,
                    "tier": tier}

        # What the tier ASKED for versus what it got. A frontier run that quietly drops to
        # one model because two keys are missing is still a run, still billed, and reads
        # exactly like a frontier run in the trend — so the gap is reported, not inferred.
        requested = TIER_MODELS.get(tier) or TIER_MODELS[CHEAP_TIER]
        unavailable = [m for m in requested if m not in models_to_use]

        # `custom_probes` comes from `tracked_mentions.source_config.custom_probes`.
        # Until now that key was documented and never read, so a merchant could save
        # their own questions and be probed with the stock four regardless.
        probes = templates or build_probes(
            facets,
            custom_probes=custom_probes,
            include_defaults=include_default_probes,
            site=homepage_domain,
        )
        rows: List[Dict[str, Any]] = []
        total_cost = 0.0

        for p in probes:
            for model in models_to_use:
                try:
                    reply = await self._call_model(model=model, prompt=p["prompt"])
                except Exception as e:
                    reply = ModelReply("", 0, 0, 0, str(e), [])

                cost = self._cost(model, reply.input_tokens, reply.output_tokens)
                total_cost += cost

                # Layer A: log every probe call with attribution
                log_llm_probe_call(
                    attribution=attribution, model=model,
                    input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                    latency_ms=reply.latency_ms,
                    success=reply.error is None, error_message=reply.error,
                )

                # Post-process: extract structured signal via Haiku tool use
                extraction = await self._extract(
                    response_text=reply.text or "", facets=facets, model_used=model,
                    attribution=attribution,
                )
                # Native citations first — Sonar returns a real list and the extractor
                # would only be re-reading the same links back out of the prose, worse.
                cited_urls = dedupe_urls(
                    [*reply.citations, *(extraction.get("cited_urls") or [])]
                )
                rows.append({
                    "tracked_mention_id": tracked_mention_id,
                    "probe_run_id": run_id,
                    "probe_template_key": p["key"],
                    "prompt_text": p["prompt"],
                    "model": model,
                    "response_text": (reply.text or "")[:6000],
                    "mentioned": extraction.get("mentioned"),
                    "position": extraction.get("position"),
                    "sentiment": extraction.get("sentiment"),
                    "competitors_mentioned": extraction.get("competitors_mentioned") or [],
                    "context_snippet": extraction.get("context_snippet"),
                    "cited_urls": cited_urls,
                    "brand_cited": (
                        any(domain_is_ours(u, homepage_domain) for u in cited_urls)
                        if homepage_domain else None
                    ),
                    "input_tokens": reply.input_tokens,
                    "output_tokens": reply.output_tokens,
                    "cost_usd": cost,
                    "latency_ms": reply.latency_ms,
                    "error": reply.error,
                })

        try:
            for i in range(0, len(rows), 25):
                self.supabase.client.table("llm_mention_probes").insert(rows[i:i + 25]).execute()
        except Exception as e:
            logger.error(f"llm_probes insert failed: {e}")

        # Layer C: roll up the new cost into total_billed_usd on the row
        recompute_lifetime_cost(tracked_mention_id=tracked_mention_id)

        failed = sum(1 for r in rows if r.get("error"))
        return {
            "status": "completed",
            "probe_run_id": run_id,
            "probe_count": len(rows),
            "tier": tier,
            "models": models_to_use,
            # Not cosmetic: `gpt-4o-mini` has produced 212 probe rows and 212 failures
            # since this feature shipped, and nothing ever said so. A run where every call
            # to a provider errored is a run with a smaller matrix than it claims.
            "models_unavailable": unavailable,
            # Paired with the above: which provider a missing model belongs to, and
            # whether its key was absent everywhere or merely absent from env.
            "key_sources": self.key_sources(),
            "failed_calls": failed,
            "total_cost_usd": total_cost,
        }

    # ───── Visibility analytics ─────

    def visibility_snapshot(
        self, tracked_mention_id: str, *, run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """ONE probe run, aggregated.

        This is a point measurement and nothing more - it used to claim "position
        trend" in its docstring while reading a single `probe_run_id`. The movement
        over time is `visibility_trend()`, which is a different read.
        """
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
        for row in rows:
            m = row.get("model")
            d = per_model.setdefault(m, {"probes": 0, "mentioned": 0, "positions": [], "samples": []})
            d["probes"] += 1
            if row.get("mentioned"):
                d["mentioned"] += 1
                if row.get("position"):
                    d["positions"].append(int(row["position"]))
            # Keep the raw probe answer (trimmed) so the UI can let users read exactly
            # what each model said, not just the aggregate counts. Capped per model.
            if len(d["samples"]) < 4:
                d["samples"].append({
                    "template": row.get("probe_template_key"),
                    "response": (row.get("response_text") or "")[:800],
                    "mentioned": bool(row.get("mentioned")),
                    "position": row.get("position"),
                    "sentiment": row.get("sentiment"),
                    "context_snippet": row.get("context_snippet"),
                    "cited_urls": row.get("cited_urls") or [],
                    "brand_cited": row.get("brand_cited"),
                })
            for c in row.get("competitors_mentioned") or []:
                cn = (c or "").strip()
                if cn:
                    competitors[cn] = competitors.get(cn, 0) + 1

        # Sentiment and citations are rolled up PER MODEL as well as overall: the
        # article's whole point is that the same brand reads differently on different
        # answer engines, and a single blended number hides exactly that.
        for m, d in per_model.items():
            model_rows = [r for r in rows if r.get("model") == m]
            d["sentiment"] = sentiment_rollup(model_rows)
            d["citations"] = citation_rollup(model_rows)
            # Per model matters more than overall here: one dead API key does not
            # mean the brand is invisible, it means that engine was never asked.
            roll = visibility_rollup(model_rows)
            d["answered"] = roll["answered"]
            d["failed"] = roll["failed"]
            d["share_of_voice"] = roll["share_of_voice"]
            d["status"] = "collector_failed" if roll["answered"] == 0 and roll["failed"] else "ok"
            d["error"] = roll["sample_error"] if d["status"] == "collector_failed" else None

        # Derived in `llm_visibility_math`, not inline: the denominator is probes
        # that ANSWERED. This was `total_mentioned / len(rows)`, which counts a
        # failed call as a probe that found nothing — so the 212 rate-limited
        # gpt-4o-mini rows in production reported that model at 0% share of voice
        # rather than as having no verdict at all.
        overall = visibility_rollup(rows)
        return {
            "present": True,
            "probe_run_id": run_id,
            "run_at": (rows[0] or {}).get("run_at") if rows else None,
            "total_probes": overall["probes"],
            "answered_probes": overall["answered"],
            "failed_probes": overall["failed"],
            "probe_error_sample": overall["sample_error"],
            # None means NO VERDICT. Never coerce it to 0 — that is a claim.
            "share_of_voice": overall["share_of_voice"],
            "avg_position": overall["avg_position"],
            "sentiment": sentiment_rollup(rows),
            "citations": citation_rollup(rows),
            "per_model": per_model,
            "top_competitors": sorted(competitors.items(), key=lambda kv: kv[1], reverse=True)[:10],
        }

    # ---- Windowed reads (A2 / A4) ----
    #
    # `MAX_WINDOW_ROWS` bounds a windowed read at ~8x a year of weekly 16-call runs.
    # Every read that hits it says so in `truncated` rather than quietly returning a
    # short answer: a capped aggregate that looks complete is the silent-zero shape.
    MAX_WINDOW_ROWS = 2000

    def _window_rows(
        self, tracked_mention_id: str, *, days: int,
    ) -> tuple[List[Dict[str, Any]], bool, Optional[str]]:
        """Every probe row for this subject inside `days`, newest first."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, days))).isoformat()
        try:
            r = (
                self.supabase.client.table("llm_mention_probes")
                .select(
                    "probe_run_id, run_at, model, mentioned, position, sentiment, "
                    "competitors_mentioned, cited_urls, brand_cited"
                )
                .eq("tracked_mention_id", tracked_mention_id)
                .gte("run_at", cutoff)
                .order("run_at", desc=True)
                .limit(self.MAX_WINDOW_ROWS)
                .execute()
            )
            rows = r.data or []
        except Exception as e:
            logger.warning(f"llm probe window read failed: {e}")
            return [], False, str(e)
        return rows, len(rows) >= self.MAX_WINDOW_ROWS, None

    def visibility_trend(self, tracked_mention_id: str, *, days: int = 90) -> Dict[str, Any]:
        """Visibility across probe RUNS. Fetch here, derive in llm_visibility_math."""
        rows, truncated, err = self._window_rows(tracked_mention_id, days=days)
        if err:
            return {"present": False, "error": err, "days": days, "points": []}
        return trend_from_rows(rows, days=days, truncated=truncated)

    def share_of_voice_series(
        self, tracked_mention_id: str, *, subject_label: str, days: int = 30,
    ) -> Dict[str, Any]:
        """The subject's share against its competitors, bucketed per run."""
        rows, truncated, err = self._window_rows(tracked_mention_id, days=days)
        if err:
            return {"tracked_mention_id": tracked_mention_id, "days": days,
                    "error": err, "buckets": [], "totals": None}
        out = share_of_voice_from_rows(
            rows, subject_label=subject_label, days=days, truncated=truncated,
        )
        out["tracked_mention_id"] = tracked_mention_id
        return out


    # ───── Internal: model calls ─────

    async def _call_model(self, *, model: str, prompt: str) -> ModelReply:
        start = time.time()
        try:
            if model in (HAIKU, OPUS):
                return await self._call_anthropic(prompt, model=model, start=start)
            if model in (GPT_MINI, GPT):
                return await self._call_openai(prompt, model=model, start=start)
            if model in (GEMINI_FLASH, GEMINI_PRO):
                return await self._call_gemini(prompt, model=model, start=start)
            if model in (SONAR, SONAR_PRO):
                return await self._call_perplexity(prompt, model=model, start=start)
            return ModelReply("", 0, 0, 0, f"unsupported model {model}", [])
        except httpx.HTTPStatusError as e:
            return ModelReply("", 0, 0, int((time.time() - start) * 1000),
                              _describe_http_error(e), [])
        except Exception as e:
            return ModelReply("", 0, 0, int((time.time() - start) * 1000), str(e)[:200], [])

    async def _call_anthropic(self, prompt: str, *, model: str, start: float) -> ModelReply:
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
            # No native citation channel — anything cited is inline in the prose and
            # comes back through the record_mention tool call instead.
            return ModelReply(
                text,
                int(usage.get("input_tokens") or 0),
                int(usage.get("output_tokens") or 0),
                int((time.time() - start) * 1000),
                None,
                [],
            )

    async def _call_openai(self, prompt: str, *, model: str, start: float) -> ModelReply:
        """ChatGPT, over plain HTTP. No package (the pin trap that removed the SDK), no
        embeddings (the fallback that removed the provider) — one endpoint, chat only."""
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                OPENAI_API,
                headers={"Authorization": f"Bearer {self.openai_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    # The gpt-5 family rejects `max_tokens`; this is the current name.
                    "max_completion_tokens": 800,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            u = data.get("usage") or {}
            # No native citation channel on plain chat completions — same as Anthropic.
            return ModelReply(
                text.strip(),
                int(u.get("prompt_tokens") or 0),
                int(u.get("completion_tokens") or 0),
                int((time.time() - start) * 1000),
                None,
                [],
            )

    async def _call_gemini(self, prompt: str, *, model: str, start: float) -> ModelReply:
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
            # Grounding is off for these probes, so `groundingMetadata` is absent and
            # there is nothing native to read — same as Anthropic/OpenAI above.
            return ModelReply(
                text.strip(),
                int(u.get("promptTokenCount") or 0),
                int(u.get("candidatesTokenCount") or 0),
                int((time.time() - start) * 1000),
                None,
                [],
            )

    async def _call_perplexity(self, prompt: str, *, model: str, start: float) -> ModelReply:
        """Sonar is the only probe model that answers WITH ITS SOURCES.

        Re-deriving them from the prose via the extractor would be a second, worse
        copy of a list the API already returns — so the native array is read here and
        the extractor's guesses are merged behind it.
        """
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
            # `citations` is the legacy flat list of URLs; `search_results` is the
            # current shape ({title, url, date}). Both are read — a Sonar version bump
            # that drops one would otherwise take citations to zero without an error.
            native: List[str] = [c for c in (data.get("citations") or []) if isinstance(c, str)]
            for sr in (data.get("search_results") or []):
                if isinstance(sr, dict) and sr.get("url"):
                    native.append(str(sr["url"]))
            return ModelReply(
                text.strip(),
                int(u.get("prompt_tokens") or 0),
                int(u.get("completion_tokens") or 0),
                int((time.time() - start) * 1000),
                None,
                dedupe_urls(native),
            )

    # ───── Internal: extraction (Haiku) ─────

    async def _extract(
        self, *, response_text: str, facets: SubjectFacets, model_used: str,
        attribution: Optional[CostAttribution] = None,
    ) -> Dict[str, Any]:
        """Use Haiku to extract structured signal from a model's free-text response.
        Falls back to deterministic parsing if Haiku is unavailable."""
        if not response_text or not response_text.strip():
            return {"mentioned": False, "position": None, "sentiment": "neutral",
                    "competitors_mentioned": [], "context_snippet": None,
                    "cited_urls": []}

        # Deterministic fallback
        if not self.anthropic_key:
            return self._extract_deterministic(response_text, facets)

        # Loaded before the request dict is assembled — the prompt sits several levels
        # deep inside it, where an await would be awkward (audit #14 MV-2).
        _MENTION_PROBE_PROMPT = await load_prompt(
            "classification", "llm_mention_probe", stage="mention_monitoring"
        )

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
                                    "cited_urls": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": (
                                            "Source URLs the answer points at — markdown links, bare "
                                            "URLs, footnote lists. Copy verbatim; never invent or "
                                            "complete one. Empty when the answer cites nothing."
                                        ),
                                    },
                                },
                                "required": ["mentioned", "sentiment", "competitors_mentioned"],
                            },
                        }],
                        "tool_choice": {"type": "tool", "name": "record_mention"},
                        "messages": [{
                            "role": "user",
                            "content": render(
                                _MENTION_PROBE_PROMPT,
                                label=facets.label,
                                aliases=", ".join(facets.aliases[:5]),
                                brand=facets.brand or "(none)",
                                response_text=response_text[:4000],
                            ),
                        }],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            usage = data.get("usage") or {}
            log_haiku_call(
                attribution=attribution, operation="llm_probe_extract",
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                latency_ms=int((time.time() - call_start) * 1000),
                success=True,
            )
        except Exception as e:
            logger.debug(f"llm-probe extract Haiku failed: {e}")
            log_haiku_call(
                attribution=attribution, operation="llm_probe_extract",
                input_tokens=0, output_tokens=0,
                latency_ms=int((time.time() - call_start) * 1000),
                success=False, error_message=str(e),
            )
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
                    "cited_urls": dedupe_urls(inp.get("cited_urls") or []),
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
            # Bare URL scrape. Cruder than the tool call but not a guess — a link that
            # is literally in the text is a link the answer cited.
            "cited_urls": dedupe_urls(re.findall(r"https?://[^\s)\]<>\"']+", text)),
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
