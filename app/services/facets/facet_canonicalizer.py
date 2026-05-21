"""
Auto-canonicalizing facet system.

Every ingest path (PDF Stage 4, XML supplier import, web scrape, background
agents, catalog candidate promote) routes raw facet values through this single
chokepoint before products.attributes is written.

Three layers:
  L0  Upstream LLM prompt rule ("return values in English") — implemented in the
      prompts themselves; this module assumes most values arrive in English but
      handles the residual non-English values gracefully via L2.
  L1  Deterministic string normalize (NFKC, lowercase, whitespace/separator
      collapse). Pure function, no I/O.
  L2  Voyage multilingual embedding + cosine cluster vs existing
      facet_canonical_values rows. Threshold 0.92 (cross-lingual auto-merge).

Threshold is locked at 0.92. products.attributes_raw is lossless — re-canonicalize
any time by replaying the raw arrays.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from app.services.facets.facet_whitelist import is_canonicalizable

logger = logging.getLogger(__name__)

_THRESHOLD = 0.92


# ─────────────────────────────────────────────────────────────────────────────
# L1: pure-function normalizer
# ─────────────────────────────────────────────────────────────────────────────

_SEPARATORS_RE = re.compile(r'[\s\-_/]+')
_MULTI_SPACE_RE = re.compile(r'\s+')


def normalize_string(value: Any) -> str:
    """L1 normalizer. Deterministic, locale-blind. NFKC → strip → lowercase →
    collapse separators (`-`, `_`, `/`, whitespace) to single space."""
    if value is None:
        return ""
    s = unicodedata.normalize('NFKC', str(value).strip().lower())
    s = _SEPARATORS_RE.sub(' ', s)
    s = _MULTI_SPACE_RE.sub(' ', s).strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FacetResolution:
    facet_key: str
    raw_value: str
    normalized: str
    canonical: str
    action: str           # 'exact_alias' | 'embedding_merge' | 'new'
    similarity: Optional[float]


@dataclass
class CanonicalizedAttributes:
    attributes: Dict[str, Any]              # facet -> canonical (str or list[str])
    attributes_raw: Dict[str, List[str]]    # facet -> raw values seen this call
    resolutions: List[FacetResolution]


# ─────────────────────────────────────────────────────────────────────────────
# Canonicalizer
# ─────────────────────────────────────────────────────────────────────────────

class FacetCanonicalizer:
    """Resolve raw facet values to canonical English via the resolve_facet_value
    SQL RPC. Batches Voyage embeddings per call so a product with N new values
    incurs one HTTP round-trip, not N."""

    def __init__(self, supabase_client: Any, embedding_service: Any):
        self._supabase = supabase_client
        self._embedder = embedding_service

    async def canonicalize_product(
        self,
        raw_metadata: Dict[str, Any],
        *,
        source: str,
        product_id: Optional[UUID] = None,
    ) -> CanonicalizedAttributes:
        pending = self._collect_pending(raw_metadata)
        if not pending:
            return CanonicalizedAttributes(attributes={}, attributes_raw={}, resolutions=[])

        attributes_raw = self._build_raw_map(pending)

        existing_by_facet = await self._fetch_existing(sorted({f for (f, _r, _n) in pending}))

        needs_embed: List[Tuple[str, str, str]] = []
        for facet, raw, norm in pending:
            if self._tier1_hit(existing_by_facet.get(facet, []), norm, raw) is None:
                needs_embed.append((facet, raw, norm))

        embeddings: Dict[Tuple[str, str, str], Optional[List[float]]] = {}
        if needs_embed:
            try:
                vectors = await self._embedder.generate_batch_embeddings(
                    [n for (_f, _r, n) in needs_embed],
                    dimensions=1024,
                    input_type="document",
                )
            except Exception as e:
                logger.warning(f"Voyage batch embed failed ({len(needs_embed)} values): {e}")
                vectors = [None] * len(needs_embed)
            for triple, vec in zip(needs_embed, vectors):
                embeddings[triple] = vec

        resolutions: List[FacetResolution] = []
        for facet, raw, norm in pending:
            emb = embeddings.get((facet, raw, norm))
            res = await self._rpc_resolve(facet, raw, norm, emb, source, product_id)
            if res is not None:
                resolutions.append(res)

        attributes = self._build_canonical_map(raw_metadata, resolutions)

        return CanonicalizedAttributes(
            attributes=attributes,
            attributes_raw=attributes_raw,
            resolutions=resolutions,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _collect_pending(self, raw_metadata: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        pending: List[Tuple[str, str, str]] = []
        for key, value in raw_metadata.items():
            if not is_canonicalizable(key) or value is None:
                continue
            raw_values = value if isinstance(value, list) else [value]
            for rv in raw_values:
                if rv is None:
                    continue
                raw_str = str(rv).strip()
                if not raw_str:
                    continue
                norm = normalize_string(raw_str)
                if not norm:
                    continue
                pending.append((key, raw_str, norm))
        return pending

    def _build_raw_map(self, pending: List[Tuple[str, str, str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for facet, raw, _norm in pending:
            bucket = out.setdefault(facet, [])
            if raw not in bucket:
                bucket.append(raw)
        return out

    def _build_canonical_map(
        self,
        raw_metadata: Dict[str, Any],
        resolutions: List[FacetResolution],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in resolutions:
            original = raw_metadata.get(r.facet_key)
            if isinstance(original, list):
                bucket = out.setdefault(r.facet_key, [])
                if r.canonical not in bucket:
                    bucket.append(r.canonical)
            else:
                out[r.facet_key] = r.canonical
        return out

    def _tier1_hit(
        self,
        rows: List[Dict[str, Any]],
        norm: str,
        raw: str,
    ) -> Optional[str]:
        for row in rows:
            if row.get('canonical_value') == norm:
                return row['canonical_value']
            aliases = row.get('aliases') or []
            if isinstance(aliases, list) and (norm in aliases or raw in aliases):
                return row['canonical_value']
        return None

    async def _fetch_existing(self, facet_keys: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not facet_keys:
            return {}
        try:
            resp = await asyncio.to_thread(
                lambda: self._supabase.client
                    .table('facet_canonical_values')
                    .select('facet_key, canonical_value, aliases')
                    .in_('facet_key', facet_keys)
                    .execute()
            )
            out: Dict[str, List[Dict[str, Any]]] = {}
            for row in (resp.data or []):
                out.setdefault(row['facet_key'], []).append(row)
            return out
        except Exception as e:
            logger.warning(f"facet_canonical_values prefetch failed: {e}")
            return {}

    async def _rpc_resolve(
        self,
        facet: str,
        raw: str,
        norm: str,
        embedding: Optional[List[float]],
        source: str,
        product_id: Optional[UUID],
    ) -> Optional[FacetResolution]:
        try:
            resp = await asyncio.to_thread(
                lambda: self._supabase.client.rpc('resolve_facet_value', {
                    'p_facet_key': facet,
                    'p_raw_value': raw,
                    'p_normalized': norm,
                    'p_embedding': embedding,
                    'p_threshold': _THRESHOLD,
                    'p_source': source,
                    'p_product_id': str(product_id) if product_id else None,
                }).execute()
            )
            data = resp.data
            row = data if isinstance(data, dict) else (data[0] if data else None)
            if not row:
                return None
            return FacetResolution(
                facet_key=facet,
                raw_value=raw,
                normalized=norm,
                canonical=row['canonical'],
                action=row['action'],
                similarity=row.get('similarity'),
            )
        except Exception as e:
            logger.warning(f"resolve_facet_value RPC failed for {facet}={raw!r}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper for ingest call sites
# ─────────────────────────────────────────────────────────────────────────────

async def canonicalize_product_attributes(
    supabase: Any,
    raw_metadata: Dict[str, Any],
    *,
    source: str,
    product_id: Optional[UUID] = None,
    embedding_service: Optional[Any] = None,
) -> CanonicalizedAttributes:
    """One-shot helper for ingest paths. Instantiates RealEmbeddingsService if
    one wasn't supplied. Failures degrade gracefully — returns an empty result
    rather than blocking the product insert."""
    try:
        if embedding_service is None:
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
            embedding_service = RealEmbeddingsService()
        canonicalizer = FacetCanonicalizer(supabase, embedding_service)
        return await canonicalizer.canonicalize_product(
            raw_metadata, source=source, product_id=product_id,
        )
    except Exception as e:
        logger.warning(f"canonicalize_product_attributes failed (source={source}): {e}")
        return CanonicalizedAttributes(attributes={}, attributes_raw={}, resolutions=[])
