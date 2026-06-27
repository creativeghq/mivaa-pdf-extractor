"""
Auto-canonicalizing facet system.

Every ingest path (PDF Stage 4, XML supplier import, web scrape, background
agents, catalog candidate promote) routes raw facet values through this single
chokepoint before products.attributes is written.

Pipeline:
  L0  Upstream LLM prompt rule ("return values in English") — implemented in the
      prompts themselves; this module assumes most values arrive in English but
      handles the residual non-English values via L0.5 below.
  L0.5 Haiku pretranslate (facet_translator) — only when the value contains
       non-ASCII characters. Batched per canonicalize_product call so the
       Haiku cost is one HTTP request per product regardless of N values.
       ASCII-only values bypass entirely.
  L1  Deterministic string normalize (NFKC, lowercase, whitespace/separator
      collapse). Pure function, no I/O.
  L2  Voyage multilingual embedding + cosine cluster vs existing
      facet_canonical_values rows. Threshold 0.92 (cross-lingual auto-merge).

Optimisations:
  * Diff-before-canonicalize: when canonicalising an existing product we read
    its current attributes_raw and skip pairs already seen, so re-ingest is
    near-free on stable products.
  * Batch RPC: resolve_facet_values_batch handles N values in one DB round-trip.

Threshold is locked at 0.92. products.attributes_raw is lossless — re-canonicalize
any time by replaying the raw arrays.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from app.services.facets.facet_translator import is_ascii_english, translate_facet_values
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
    canonical: Optional[str]
    action: str           # 'exact_alias' | 'embedding_merge' | 'new' | 'rejected_non_english' | 'reused_diff'
    similarity: Optional[float]


@dataclass
class CanonicalizedAttributes:
    attributes: Dict[str, Any]              # facet -> canonical (str or list[str])
    attributes_raw: Dict[str, List[str]]    # facet -> raw values seen (cumulative across runs)
    resolutions: List[FacetResolution]


# ─────────────────────────────────────────────────────────────────────────────
# Canonicalizer
# ─────────────────────────────────────────────────────────────────────────────

class FacetCanonicalizer:
    """Resolve raw facet values to canonical English via resolve_facet_values_batch.

    Single round-trip per product (after pretranslate + Voyage batch embed).
    """

    def __init__(self, supabase_client: Any, embedding_service: Any):
        self._supabase = supabase_client
        self._embedder = embedding_service

    async def canonicalize_product(
        self,
        raw_metadata: Dict[str, Any],
        *,
        source: str,
        product_id: Optional[UUID] = None,
        workspace_id: Optional[str] = None,
    ) -> CanonicalizedAttributes:
        # ── 1. Collect canonicalizable (facet, raw, norm) triples ────────────
        pending = self._collect_pending(raw_metadata)
        if not pending:
            return CanonicalizedAttributes(attributes={}, attributes_raw={}, resolutions=[])

        # ── 2. Diff against existing product if product_id supplied ──────────
        existing_attrs, existing_raw = await self._load_existing(product_id)

        # `reused`: (facet, raw, norm) we already canonicalized in a prior run.
        # Their canonical comes from existing_attrs; no RPC / embed / translate needed.
        reused: List[Tuple[str, str, str, str]] = []      # (facet, raw, norm, canonical)
        fresh: List[Tuple[str, str, str]] = []
        for facet, raw, norm in pending:
            prior_raws = existing_raw.get(facet, []) if existing_raw else []
            prior_canonical = self._lookup_prior_canonical(existing_attrs, facet)
            if raw in prior_raws and prior_canonical is not None:
                reused.append((facet, raw, norm, prior_canonical))
            else:
                fresh.append((facet, raw, norm))

        resolutions: List[FacetResolution] = [
            FacetResolution(facet, raw, norm, canon, 'reused_diff', None)
            for (facet, raw, norm, canon) in reused
        ]

        if fresh:
            resolutions.extend(await self._resolve_fresh(fresh, source=source, product_id=product_id, workspace_id=workspace_id))

        # ── 3. Build canonical + raw maps ────────────────────────────────────
        attributes = self._build_canonical_map(raw_metadata, resolutions)
        attributes_raw = self._merge_raw_maps(existing_raw, pending)

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

    async def _load_existing(
        self,
        product_id: Optional[UUID],
    ) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        if product_id is None:
            return {}, {}
        try:
            resp = await asyncio.to_thread(
                lambda: self._supabase.client
                    .table('products')
                    .select('attributes, attributes_raw')
                    .eq('id', str(product_id))
                    .limit(1)
                    .execute()
            )
            if not resp.data:
                return {}, {}
            row = resp.data[0]
            return (row.get('attributes') or {}), (row.get('attributes_raw') or {})
        except Exception as e:
            logger.warning(f"_load_existing failed for product_id={product_id}: {e}")
            return {}, {}

    def _lookup_prior_canonical(
        self,
        existing_attrs: Dict[str, Any],
        facet: str,
    ) -> Optional[str]:
        """For list-valued facets we can't know which canonical a specific raw
        produced — only that the raw was seen and the facet had at least one
        canonical. We return the first canonical so reuse remains stable.
        Callers that re-resolve from raw aliases would be more precise but
        require a per-alias map, which we don't store.
        """
        val = existing_attrs.get(facet)
        if val is None:
            return None
        if isinstance(val, list):
            return val[0] if val else None
        if isinstance(val, str):
            return val
        return None

    def _merge_raw_maps(
        self,
        existing_raw: Dict[str, List[str]],
        pending: List[Tuple[str, str, str]],
    ) -> Dict[str, List[str]]:
        merged: Dict[str, List[str]] = {}
        # Start from existing (preserve prior raw values across runs)
        for facet, vals in (existing_raw or {}).items():
            if isinstance(vals, list):
                merged[facet] = [v for v in vals if isinstance(v, str)]
        # Append new raws (dedupe)
        for facet, raw, _norm in pending:
            bucket = merged.setdefault(facet, [])
            if raw not in bucket:
                bucket.append(raw)
        return merged

    def _build_canonical_map(
        self,
        raw_metadata: Dict[str, Any],
        resolutions: List[FacetResolution],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in resolutions:
            if not r.canonical:
                continue
            original = raw_metadata.get(r.facet_key)
            if isinstance(original, list):
                bucket = out.setdefault(r.facet_key, [])
                if r.canonical not in bucket:
                    bucket.append(r.canonical)
            else:
                out[r.facet_key] = r.canonical
        return out

    async def _resolve_fresh(
        self,
        fresh: List[Tuple[str, str, str]],
        *,
        source: str,
        product_id: Optional[UUID],
        workspace_id: Optional[str] = None,
    ) -> List[FacetResolution]:
        # ── L0.5: pretranslate non-ASCII norms ────────────────────────────────
        translation_inputs = [(facet, raw) for (facet, raw, norm) in fresh if not is_ascii_english(norm)]
        translations: Dict[Tuple[str, str], str] = {}
        if translation_inputs:
            translations = await translate_facet_values(translation_inputs)

        # Build a working list (facet, raw, effective_normalized) post-pretranslate.
        # If pretranslate produced an English term for this (facet, raw), use it
        # as the normalized form. Otherwise keep the original — the RPC's
        # non-ASCII guard will reject it and log 'rejected_non_english'.
        working: List[Tuple[str, str, str]] = []
        for facet, raw, norm in fresh:
            if not is_ascii_english(norm):
                translated = translations.get((facet, raw))
                effective = normalize_string(translated) if translated else norm
            else:
                effective = norm
            working.append((facet, raw, effective))

        # ── Tier-1 local prefetch (skip embed for known aliases) ──────────────
        existing_by_facet = await self._fetch_existing(sorted({f for (f, _r, _n) in working}), workspace_id=workspace_id)
        needs_embed: List[Tuple[str, str, str]] = []
        for facet, raw, norm in working:
            if self._tier1_hit(existing_by_facet.get(facet, []), norm, raw) is None:
                # Also skip embedding for non-ASCII norms — the RPC will reject them,
                # so spending Voyage cost on them is pointless.
                if is_ascii_english(norm):
                    needs_embed.append((facet, raw, norm))

        # ── Tier-2 batch embed ────────────────────────────────────────────────
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

        # ── Single batch RPC call ─────────────────────────────────────────────
        items_payload: List[Dict[str, Any]] = []
        for facet, raw, norm in working:
            item: Dict[str, Any] = {
                'facet_key': facet,
                'raw': raw,
                'normalized': norm,
            }
            emb = embeddings.get((facet, raw, norm))
            if emb is not None:
                item['embedding'] = list(emb)
            items_payload.append(item)

        results = await self._rpc_batch(items_payload, source=source, product_id=product_id, workspace_id=workspace_id)

        return [
            FacetResolution(
                facet_key=r.get('facet_key', ''),
                raw_value=r.get('raw', ''),
                normalized=r.get('normalized', ''),
                canonical=r.get('canonical'),
                action=r.get('action', 'unknown'),
                similarity=r.get('similarity'),
            )
            for r in results
        ]

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

    async def _fetch_existing(self, facet_keys: List[str], workspace_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        if not facet_keys:
            return {}
        try:
            def _run():
                q = (self._supabase.client
                    .table('facet_canonical_values')
                    .select('facet_key, canonical_value, aliases')
                    .in_('facet_key', facet_keys))
                if workspace_id:
                    # Tier-1 prefetch scoped to own workspace + operator golden set.
                    q = q.or_(f'workspace_id.eq.{workspace_id},is_golden.is.true')
                return q.execute()
            resp = await asyncio.to_thread(_run)
            out: Dict[str, List[Dict[str, Any]]] = {}
            for row in (resp.data or []):
                out.setdefault(row['facet_key'], []).append(row)
            return out
        except Exception as e:
            logger.warning(f"facet_canonical_values prefetch failed: {e}")
            return {}

    async def _rpc_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        source: str,
        product_id: Optional[UUID],
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []
        try:
            resp = await asyncio.to_thread(
                lambda: self._supabase.client.rpc('resolve_facet_values_batch', {
                    'p_items': items,
                    'p_threshold': _THRESHOLD,
                    'p_source': source,
                    'p_product_id': str(product_id) if product_id else None,
                    'p_workspace_id': str(workspace_id) if workspace_id else None,
                }).execute()
            )
            data = resp.data
            if isinstance(data, list):
                return data
            if isinstance(data, str):
                try:
                    return json.loads(data)
                except Exception:
                    return []
            return []
        except Exception as e:
            logger.warning(f"resolve_facet_values_batch failed ({len(items)} items): {e}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper for ingest call sites
# ─────────────────────────────────────────────────────────────────────────────

async def resolve_query_term(
    supabase: Any,
    facet_key: str,
    raw_term: str,
    workspace_id: Optional[str] = None,
) -> str:
    """Query-side canonicalizer. L1 normalize → alias lookup against
    facet_canonical_values. Returns canonical English if known, else the
    L1-normalized form so the filter still matches whatever else is in the DB.

    No embedding cost — query terms either match an existing alias (cheap)
    or don't (free; we fall back to the normalized form). Uses three safe
    parameterised lookups instead of one f-string interpolated PostgREST
    `.or_()` filter so untrusted user terms (Greek words containing quotes,
    commas, brackets) can't break the query.
    """
    norm = normalize_string(raw_term)
    if not norm:
        return ""

    def _scoped_base():
        q = (supabase.client.table('facet_canonical_values')
                .select('canonical_value')
                .eq('facet_key', facet_key))
        if workspace_id:
            # Scope query-side resolution to own workspace + operator golden set.
            q = q.or_(f'workspace_id.eq.{workspace_id},is_golden.is.true')
        return q

    async def _lookup_one(filter_fn) -> Optional[str]:
        try:
            resp = await asyncio.to_thread(
                lambda: filter_fn(_scoped_base()).limit(1).execute()
            )
            if resp.data:
                return resp.data[0]['canonical_value']
        except Exception as e:
            logger.debug(f"resolve_query_term sub-lookup failed for {facet_key}={raw_term!r}: {e}")
        return None

    # Tier 1: exact canonical match (most common hit on stable catalogs)
    hit = await _lookup_one(lambda q: q.eq('canonical_value', norm))
    if hit:
        return hit
    # Tier 2: normalized form appears as an alias
    hit = await _lookup_one(lambda q: q.contains('aliases', [norm]))
    if hit:
        return hit
    # Tier 3: raw (pre-normalize) form appears as an alias — covers cases where
    # an upstream wrote the raw value verbatim (Greek-cased "Λευκό")
    if raw_term and raw_term != norm:
        hit = await _lookup_one(lambda q: q.contains('aliases', [raw_term]))
        if hit:
            return hit
    return norm


def collect_raw_attributes(raw_metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build the attributes_raw map from metadata WITHOUT any embedding/RPC.

    Pure-local fallback for ingest paths when canonicalization times out or
    errors: products.attributes_raw is the lossless contract that lets a later
    re-canonicalization pass replay — losing it means the product is
    permanently unfacetable without a full re-ingest. Mirrors
    FacetCanonicalizer._collect_pending's whitelist rules.
    """
    raw_map: Dict[str, List[str]] = {}
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
            bucket = raw_map.setdefault(key, [])
            if raw_str not in bucket:
                bucket.append(raw_str)
    return raw_map


async def canonicalize_product_attributes(
    supabase: Any,
    raw_metadata: Dict[str, Any],
    *,
    source: str,
    product_id: Optional[UUID] = None,
    embedding_service: Optional[Any] = None,
    workspace_id: Optional[str] = None,
) -> CanonicalizedAttributes:
    """One-shot helper for ingest paths. Instantiates RealEmbeddingsService if
    one wasn't supplied. Failures degrade gracefully — returns a result with
    empty canonical attributes but the LOSSLESS raw map preserved, so a later
    re-canonicalization pass can replay without re-ingesting."""
    try:
        if embedding_service is None:
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
            embedding_service = RealEmbeddingsService()
        canonicalizer = FacetCanonicalizer(supabase, embedding_service)
        return await canonicalizer.canonicalize_product(
            raw_metadata, source=source, product_id=product_id, workspace_id=workspace_id,
        )
    except Exception as e:
        logger.error(f"canonicalize_product_attributes failed (source={source}): {e}")
        return CanonicalizedAttributes(
            attributes={},
            attributes_raw=collect_raw_attributes(raw_metadata),
            resolutions=[],
        )
