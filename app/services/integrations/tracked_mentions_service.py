"""
Tracked Mentions Service — CRUD + refresh orchestration.

Mirror of `tracked_queries_service.py` for mention-monitoring. The chokepoint
through which both flows (internal product/brand + external API consumers)
run their refreshes.

Pattern: same row schema, two routing modes.
  - Internal:  api_key_id IS NULL, product_id NOT NULL OR brand_name NOT NULL
  - External:  api_key_id NOT NULL, product_id NULL

Single `refresh()` chokepoint runs:
  1. Decompose subject → facets (cached on row).
  2. Search across enabled sources in parallel.
  3. Strip excluded URLs / promote pinned overrides.
  4. Classify (Haiku batched + verdict cache).
  5. Apply anomaly check (sentiment outlier vs trailing 7d).
  6. Persist mention_history rows with refresh_run_id.
  7. Update denormalized cache (mention_count, sentiment_avg, top_outlets).
  8. Update volatility cadence via update_tracked_mention_cadence RPC.
  9. Detect alerts and dispatch via the notifications module.
"""

from __future__ import annotations

import logging
import os
import statistics
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.mention_identity_service import (
    SubjectFacets, content_hash, get_mention_identity_service,
)
from app.services.integrations.mention_search_service import (
    MentionHit, canonicalize_url, domain_of, get_mention_search_service,
)

logger = logging.getLogger(__name__)


HEALTHY_CADENCE_HOURS = 24
SPIKE_THRESHOLD_RATIO = 2.0


class TrackedMentionsService:
    def __init__(self) -> None:
        self.supabase = get_supabase_client()
        self.identity = get_mention_identity_service()
        self.search = get_mention_search_service()

    # ───── CRUD ─────

    async def create(
        self,
        *,
        api_key_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        subject_type: str = "product",
        subject_label: str,
        product_id: Optional[str] = None,
        brand_name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        sources_enabled: Optional[Dict[str, bool]] = None,
        source_config: Optional[Dict[str, Any]] = None,
        language_codes: Optional[List[str]] = None,
        country_codes: Optional[List[str]] = None,
        refresh_interval_hours: int = HEALTHY_CADENCE_HOURS,
        alert_channels: Optional[List[str]] = None,
        alert_on_spike: Optional[bool] = None,
        alert_on_negative_sentiment: Optional[bool] = None,
        alert_on_new_outlet: Optional[bool] = None,
        alert_on_llm_visibility_change: Optional[bool] = None,
        alert_webhook_url: Optional[str] = None,
        auto_expand_aliases: bool = False,
        run_first_refresh: bool = True,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "subject_type": subject_type,
            "subject_label": subject_label,
            "product_id": product_id,
            "brand_name": brand_name,
            "api_key_id": api_key_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "aliases": aliases or [],
            "auto_expand_aliases": bool(auto_expand_aliases),
            "sources_enabled": sources_enabled or {
                "news": True, "blogs": True,
                "youtube": False, "rss": True, "llm": True,
            },
            "source_config": source_config or {},
            "language_codes": language_codes or ["en"],
            "country_codes": country_codes or [],
            "refresh_interval_hours": refresh_interval_hours,
            "alert_channels": alert_channels or ["bell"],
            "alert_on_spike": bool(alert_on_spike),
            "alert_on_negative_sentiment": bool(alert_on_negative_sentiment),
            "alert_on_new_outlet": bool(alert_on_new_outlet),
            "alert_on_llm_visibility_change": bool(alert_on_llm_visibility_change),
            "alert_webhook_url": alert_webhook_url,
            "is_active": True,
            "next_check_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            res = self.supabase.client.table("tracked_mentions").insert(row).execute()
            inserted = (res.data or [None])[0] or {}
        except Exception as e:
            logger.error(f"tracked_mentions.create insert failed: {e}")
            raise

        if run_first_refresh and inserted.get("id"):
            try:
                refresh_outcome = await self.refresh(inserted["id"], force=True)
                inserted["last_refresh"] = refresh_outcome
            except Exception as e:
                logger.warning(f"tracked_mentions.create first refresh failed: {e}")
        return inserted

    def get(self, tracked_mention_id: str) -> Optional[Dict[str, Any]]:
        try:
            r = (
                self.supabase.client.table("tracked_mentions")
                .select("*")
                .eq("id", tracked_mention_id)
                .maybe_single()
                .execute()
            )
            return r.data if r else None
        except Exception as e:
            logger.warning(f"tracked_mentions.get failed: {e}")
            return None

    def find_for_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        try:
            r = (
                self.supabase.client.table("tracked_mentions")
                .select("*")
                .eq("product_id", product_id)
                .is_("api_key_id", "null")
                .maybe_single()
                .execute()
            )
            return r.data if r else None
        except Exception as e:
            logger.debug(f"tracked_mentions.find_for_product miss: {e}")
            return None

    async def find_or_create_for_product(
        self, *,
        product_id: str,
        product_name: str,
        brand_name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        country_codes: Optional[List[str]] = None,
        auto_expand_aliases: bool = False,
        run_first_refresh: bool = True,
    ) -> Dict[str, Any]:
        existing = self.find_for_product(product_id)
        if existing:
            return existing
        return await self.create(
            subject_type="product",
            subject_label=product_name,
            product_id=product_id,
            brand_name=brand_name,
            aliases=aliases,
            auto_expand_aliases=auto_expand_aliases,
            user_id=user_id,
            workspace_id=workspace_id,
            country_codes=country_codes,
            run_first_refresh=run_first_refresh,
        )

    def list_internal(
        self, *, workspace_id: Optional[str] = None, include_inactive: bool = False, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        try:
            q = (
                self.supabase.client.table("tracked_mentions")
                .select("*")
                .is_("api_key_id", "null")
                .order("created_at", desc=True)
                .limit(limit)
            )
            if not include_inactive:
                q = q.eq("is_active", True)
            if workspace_id:
                q = q.eq("workspace_id", workspace_id)
            r = q.execute()
            return r.data or []
        except Exception as e:
            logger.warning(f"tracked_mentions.list_internal failed: {e}")
            return []

    def update(self, tracked_mention_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
        allowed = {
            "subject_label", "aliases", "sources_enabled", "source_config",
            "language_codes", "country_codes", "refresh_interval_hours",
            "alert_channels", "alert_on_spike", "alert_on_negative_sentiment",
            "alert_on_new_outlet", "alert_on_llm_visibility_change",
            "alert_webhook_url", "is_active", "auto_expand_aliases",
        }
        payload = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not payload:
            return self.get(tracked_mention_id)
        try:
            r = (
                self.supabase.client.table("tracked_mentions")
                .update(payload)
                .eq("id", tracked_mention_id)
                .execute()
            )
            return (r.data or [None])[0]
        except Exception as e:
            logger.warning(f"tracked_mentions.update failed: {e}")
            return None

    def deactivate(self, tracked_mention_id: str) -> bool:
        try:
            (
                self.supabase.client.table("tracked_mentions")
                .update({"is_active": False})
                .eq("id", tracked_mention_id)
                .execute()
            )
            return True
        except Exception as e:
            logger.warning(f"tracked_mentions.deactivate failed: {e}")
            return False

    # ───── Exclusions ─────

    def add_exclusion(
        self, tracked_mention_id: str, *,
        url: Optional[str] = None,
        domain: Optional[str] = None,
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not url and not domain:
            raise ValueError("either url or domain is required")
        row = {
            "tracked_mention_id": tracked_mention_id,
            "url": canonicalize_url(url) if url else None,
            "domain": domain.lower() if domain else None,
            "reason": reason,
            "excluded_by": user_id,
        }
        r = self.supabase.client.table("mention_excluded_urls").upsert(
            row, on_conflict="tracked_mention_id,url" if url else "tracked_mention_id,domain"
        ).execute()
        return (r.data or [None])[0] or row

    def list_exclusions(self, tracked_mention_id: str) -> List[Dict[str, Any]]:
        try:
            r = (
                self.supabase.client.table("mention_excluded_urls")
                .select("*")
                .eq("tracked_mention_id", tracked_mention_id)
                .order("excluded_at", desc=True)
                .execute()
            )
            return r.data or []
        except Exception:
            return []

    def remove_exclusion(self, tracked_mention_id: str, *, url: Optional[str] = None, domain: Optional[str] = None) -> int:
        q = self.supabase.client.table("mention_excluded_urls").delete().eq("tracked_mention_id", tracked_mention_id)
        if url:
            q = q.eq("url", canonicalize_url(url))
        if domain:
            q = q.eq("domain", domain.lower())
        try:
            r = q.execute()
            return len(r.data or [])
        except Exception:
            return 0

    def add_promoted_url(
        self, tracked_mention_id: str, *,
        url: str, override_relevance: str, reason: Optional[str], user_id: Optional[str],
    ) -> Dict[str, Any]:
        row = {
            "tracked_mention_id": tracked_mention_id,
            "url": canonicalize_url(url),
            "override_relevance": override_relevance,
            "reason": reason,
            "created_by": user_id,
        }
        r = (
            self.supabase.client.table("mention_promoted_urls")
            .upsert(row, on_conflict="tracked_mention_id,url")
            .execute()
        )
        return (r.data or [None])[0] or row

    # ───── Refresh chokepoint ─────

    async def refresh(self, tracked_mention_id: str, *, force: bool = False) -> Dict[str, Any]:
        row = self.get(tracked_mention_id)
        if not row:
            return {"status": "not_found", "credits_used": 0}
        if not row.get("is_active"):
            return {"status": "inactive", "credits_used": 0}
        if not force:
            next_check = row.get("next_check_at")
            if next_check:
                try:
                    nc = datetime.fromisoformat(str(next_check).replace("Z", "+00:00"))
                    if nc > datetime.now(timezone.utc):
                        return {"status": "throttled", "credits_used": 0,
                                "next_check_at": next_check}
                except Exception:
                    pass

        run_id = str(uuid.uuid4())

        # Facets — Haiku expansion is opt-in via auto_expand_aliases flag.
        # Default is deterministic (label + user aliases only, no LLM call,
        # no Anthropic dependency).
        cached_facets = row.get("subject_facets")
        use_llm = bool(row.get("auto_expand_aliases"))
        facets = await self.identity.extract_facets(
            subject_label=row["subject_label"],
            subject_type=row["subject_type"],
            aliases_seed=row.get("aliases") or [],
            brand_hint=row.get("brand_name"),
            cached=cached_facets,
            use_llm=use_llm,
        )
        if not cached_facets:
            try:
                self.supabase.client.table("tracked_mentions").update({
                    "subject_facets": facets.to_dict(),
                    "subject_facets_cached_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", tracked_mention_id).execute()
            except Exception as e:
                logger.warning(f"facet cache write failed: {e}")

        # Discovery
        sources_enabled = row.get("sources_enabled") or {}
        source_config = row.get("source_config") or {}
        country_codes = row.get("country_codes") or []
        is_first_refresh = not row.get("last_refreshed_at")
        result = await self.search.search(
            facets=facets,
            sources_enabled=sources_enabled,
            source_config=source_config,
            country_codes=country_codes,
            recency_days=30,
            force_full_discovery=is_first_refresh or force,
        )
        if not result.hits:
            self._stamp_refresh(
                tracked_mention_id, run_id=run_id, credits=result.credits_used,
                hits_count=0, sentiment_avg=None, top_outlets=[], errors=result.errors,
            )
            return {
                "status": "refreshed",
                "credits_used": result.credits_used,
                "hits_count": 0,
                "by_source": result.by_source,
                "errors": result.errors,
                "results": [],
                "refresh_run_id": run_id,
            }

        # Apply exclusions + promoted overrides
        exclusions = self.list_exclusions(tracked_mention_id)
        excluded_urls = {e.get("url") for e in exclusions if e.get("url")}
        excluded_domains = {e.get("domain") for e in exclusions if e.get("domain")}
        promoted = self._load_promoted(tracked_mention_id)

        filtered: List[MentionHit] = []
        for h in result.hits:
            curl = h.canonical_url()
            host = h.outlet_domain or domain_of(h.url)
            if curl in excluded_urls or (host and host in excluded_domains):
                continue
            filtered.append(h)

        # Classify
        candidates_for_classifier = [
            {
                "url": h.url,
                "title": h.title,
                "excerpt": h.excerpt,
                "body_md": h.body_md,
                "outlet_domain": h.outlet_domain,
                "content_hash": content_hash(url=h.url, title=h.title, body=h.body_md or h.excerpt),
            }
            for h in filtered
        ]
        verdicts = await self.identity.classify_batch(
            candidates=candidates_for_classifier, facets=facets,
        )

        # Persist + sentiment metrics
        rows_to_insert: List[Dict[str, Any]] = []
        sentiment_scores: List[float] = []
        outlet_counts: Dict[str, int] = {}
        positive = neutral = negative = 0
        for h, v in zip(filtered, verdicts):
            curl = h.canonical_url()
            ch = content_hash(url=h.url, title=h.title, body=h.body_md or h.excerpt)
            relevance = v.get("relevance") or "unverifiable"
            sentiment = v.get("sentiment") or "neutral"
            sscore = float(v.get("sentiment_score") or 0.0)
            # Promoted URL override (sticky admin override)
            if curl in promoted:
                relevance = promoted[curl]
            # Drop mismatches entirely (keep tangential + exact + unverifiable)
            if relevance == "mismatch":
                continue
            sentiment_scores.append(sscore)
            if sentiment == "positive":
                positive += 1
            elif sentiment == "negative":
                negative += 1
            else:
                neutral += 1
            host = h.outlet_domain or domain_of(h.url) or ""
            if host:
                outlet_counts[host] = outlet_counts.get(host, 0) + 1
            rows_to_insert.append({
                "tracked_mention_id": tracked_mention_id,
                "refresh_run_id": run_id,
                "url": h.url,
                "canonical_url": curl,
                "content_hash": ch,
                "outlet_domain": host,
                "outlet_name": h.outlet_name,
                "outlet_type": h.outlet_type,
                "title": h.title,
                "excerpt": (h.excerpt or "")[:1000] if h.excerpt else None,
                "body_md": (h.body_md or "")[:2000] if h.body_md else None,
                "language_code": h.language_code,
                "country_code": h.country_code,
                "author": h.author,
                "published_at": h.published_at,
                "discovered_at": datetime.now(timezone.utc).isoformat(),
                "sentiment": sentiment,
                "sentiment_score": sscore,
                "relevance": relevance,
                "relevance_score": v.get("relevance_score"),
                "match_note": v.get("match_note"),
                "engagement": h.engagement,
                "source": h.source,
                "classifier_cached": bool(v.get("classifier_cached")),
                "raw_payload": h.raw,
            })

        # Anomaly detection: sentiment outlier vs trailing 7d
        rows_to_insert = self._stamp_anomalies(tracked_mention_id, rows_to_insert)

        if rows_to_insert:
            try:
                # Insert in chunks to avoid payload limits
                for i in range(0, len(rows_to_insert), 50):
                    chunk = rows_to_insert[i:i + 50]
                    self.supabase.client.table("mention_history").insert(chunk).execute()
            except Exception as e:
                logger.error(f"mention_history insert failed: {e}")

        sentiment_avg = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
        top_outlets = sorted(outlet_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

        # Update denormalized cache + cadence
        velocity_pct = self._compute_velocity(tracked_mention_id, current_count=len(rows_to_insert))
        self._stamp_refresh(
            tracked_mention_id, run_id=run_id, credits=result.credits_used,
            hits_count=len(rows_to_insert), sentiment_avg=sentiment_avg,
            top_outlets=[{"domain": d, "count": c} for d, c in top_outlets],
            errors=result.errors,
        )
        try:
            self.supabase.client.rpc("update_tracked_mention_cadence", {
                "p_tracked_mention_id": tracked_mention_id,
                "p_velocity_pct_change": float(velocity_pct or 0.0),
            }).execute()
        except Exception as e:
            logger.warning(f"update_tracked_mention_cadence failed: {e}")

        # Alerts (best-effort)
        try:
            from app.modules.mention_monitoring_notifications.service import (
                get_mention_alert_dispatcher,
            )
            dispatcher = get_mention_alert_dispatcher()
            candidates = dispatcher.detect_after_refresh(
                tracked_mention_id=tracked_mention_id,
                new_rows=rows_to_insert,
            )
            dispatcher.dispatch(candidates)
        except Exception as e:
            logger.warning(f"mention alert dispatch failed: {e}")

        return {
            "status": "refreshed",
            "credits_used": result.credits_used,
            "hits_count": len(rows_to_insert),
            "by_source": result.by_source,
            "errors": result.errors,
            "refresh_run_id": run_id,
            "results": rows_to_insert,
            "sentiment_avg": sentiment_avg,
            "top_outlets": top_outlets,
        }

    # ───── Read APIs (history, summary, share-of-voice) ─────

    def latest_results(self, tracked_mention_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            # Latest run_id
            run = (
                self.supabase.client.table("mention_history")
                .select("refresh_run_id, discovered_at")
                .eq("tracked_mention_id", tracked_mention_id)
                .order("discovered_at", desc=True)
                .limit(1)
                .execute()
            )
            if not run.data:
                return []
            run_id = (run.data[0] or {}).get("refresh_run_id")
            if not run_id:
                return []
            r = (
                self.supabase.client.table("mention_history")
                .select("*")
                .eq("tracked_mention_id", tracked_mention_id)
                .eq("refresh_run_id", run_id)
                .order("published_at", desc=True)
                .limit(limit)
                .execute()
            )
            return r.data or []
        except Exception as e:
            logger.warning(f"latest_results failed: {e}")
            return []

    def history(
        self, tracked_mention_id: str, *,
        days: int = 30, limit: int = 500,
        sentiment: Optional[str] = None, outlet_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        try:
            q = (
                self.supabase.client.table("mention_history")
                .select("*")
                .eq("tracked_mention_id", tracked_mention_id)
                .gte("discovered_at", cutoff)
                .order("published_at", desc=True)
                .limit(limit)
            )
            if sentiment:
                q = q.eq("sentiment", sentiment)
            if outlet_type:
                q = q.eq("outlet_type", outlet_type)
            r = q.execute()
            return r.data or []
        except Exception as e:
            logger.warning(f"history failed: {e}")
            return []

    def summary(self, tracked_mention_id: str, *, days: int = 30) -> Dict[str, Any]:
        rows = self.history(tracked_mention_id, days=days, limit=2000)
        sentiments = [r.get("sentiment") for r in rows if r.get("sentiment")]
        outlets: Dict[str, int] = {}
        for r in rows:
            d = r.get("outlet_domain") or "unknown"
            outlets[d] = outlets.get(d, 0) + 1
        outlet_breakdown = sorted(outlets.items(), key=lambda kv: kv[1], reverse=True)[:15]
        sscores = [float(r.get("sentiment_score") or 0.0) for r in rows]
        return {
            "tracked_mention_id": tracked_mention_id,
            "days": days,
            "total_count": len(rows),
            "by_sentiment": {
                "positive": sentiments.count("positive"),
                "neutral": sentiments.count("neutral"),
                "negative": sentiments.count("negative"),
            },
            "sentiment_avg": (sum(sscores) / len(sscores)) if sscores else None,
            "top_outlets": [{"domain": d, "count": c} for d, c in outlet_breakdown],
            "latest_at": rows[0].get("discovered_at") if rows else None,
        }

    # ───── Helpers ─────

    def _load_promoted(self, tracked_mention_id: str) -> Dict[str, str]:
        try:
            r = (
                self.supabase.client.table("mention_promoted_urls")
                .select("url, override_relevance")
                .eq("tracked_mention_id", tracked_mention_id)
                .execute()
            )
            out: Dict[str, str] = {}
            for row in (r.data or []):
                if row.get("url"):
                    out[row["url"]] = row.get("override_relevance") or "exact"
            return out
        except Exception:
            return {}

    def _stamp_anomalies(
        self, tracked_mention_id: str, rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compare each row's sentiment_score against the trailing 7d median."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        try:
            r = (
                self.supabase.client.table("mention_history")
                .select("sentiment_score, outlet_domain")
                .eq("tracked_mention_id", tracked_mention_id)
                .gte("discovered_at", cutoff)
                .execute()
            )
            scores = [float(x.get("sentiment_score") or 0.0)
                      for x in (r.data or []) if x.get("sentiment_score") is not None]
        except Exception:
            scores = []

        if len(scores) < 5:
            return rows  # not enough history

        median = statistics.median(scores)
        try:
            stdev = statistics.stdev(scores)
        except statistics.StatisticsError:
            stdev = 0.0

        for row in rows:
            ss = row.get("sentiment_score")
            if ss is None or stdev == 0:
                continue
            z = abs(float(ss) - median) / stdev
            if z >= 2.5 and float(ss) < median:  # only flag negative outliers
                row["is_anomaly"] = True
                row["anomaly_reason"] = (
                    f"sentiment {ss:.2f} is {z:.1f}σ below 7d median {median:.2f}"
                )
        return rows

    def _compute_velocity(self, tracked_mention_id: str, *, current_count: int) -> Optional[float]:
        """Compare new count vs trailing 7d daily average. Returns % change."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        try:
            r = (
                self.supabase.client.table("mention_history")
                .select("id", count="exact")
                .eq("tracked_mention_id", tracked_mention_id)
                .gte("discovered_at", cutoff)
                .limit(1)
                .execute()
            )
            total_7d = r.count or 0
        except Exception:
            return None
        if total_7d <= 0:
            return 100.0 if current_count > 0 else 0.0
        avg_daily = total_7d / 7.0
        if avg_daily <= 0:
            return None
        return abs(current_count - avg_daily) / avg_daily * 100.0

    def _stamp_refresh(
        self, tracked_mention_id: str, *,
        run_id: str, credits: int, hits_count: int,
        sentiment_avg: Optional[float],
        top_outlets: List[Dict[str, Any]],
        errors: Dict[str, str],
    ) -> None:
        try:
            cur = self.get(tracked_mention_id) or {}
            total = (cur.get("total_credits_used") or 0) + (credits or 0)
            self.supabase.client.table("tracked_mentions").update({
                "last_refreshed_at": datetime.now(timezone.utc).isoformat(),
                "last_refresh_credits_used": credits,
                "total_credits_used": total,
                "current_mention_count_7d": self._count_window(tracked_mention_id, days=7),
                "current_mention_count_30d": self._count_window(tracked_mention_id, days=30),
                "current_sentiment_avg": sentiment_avg,
                "current_top_outlets": top_outlets,
                "current_metadata": {"errors": errors, "last_run_id": run_id},
                "current_snapshot_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", tracked_mention_id).execute()
        except Exception as e:
            logger.warning(f"stamp_refresh failed: {e}")

    def _count_window(self, tracked_mention_id: str, *, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        try:
            r = (
                self.supabase.client.table("mention_history")
                .select("id", count="exact")
                .eq("tracked_mention_id", tracked_mention_id)
                .gte("discovered_at", cutoff)
                .limit(1)
                .execute()
            )
            return r.count or 0
        except Exception:
            return 0


_service: Optional[TrackedMentionsService] = None


def get_tracked_mentions_service() -> TrackedMentionsService:
    global _service
    if _service is None:
        _service = TrackedMentionsService()
    return _service
