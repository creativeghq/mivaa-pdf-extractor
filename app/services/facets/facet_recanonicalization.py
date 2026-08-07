"""Bulk facet re-canonicalization sweep (#316).

Why this exists
---------------
`products.attributes_raw` is the lossless, cumulative record of every raw facet value a
product has ever presented. It is kept precisely so the canonical layer can be REBUILT
without re-ingesting the source PDF — the Medallion rule in CLAUDE.md: rebuild gold from
silver, never re-derive from bronze.

Until now nothing replayed it in bulk. That left two classes of product permanently stale:

  1. **Outage-degraded rows.** When Voyage, the merge RPC, or the 30s budget failed during
     Stage 4, the product was written with EMPTY `attributes` and flagged in
     `metadata.facet_canonicalization.status`. The flag had no consumer, so those products
     stayed facet-less forever — invisible to every facet filter while looking, from the
     outside, exactly like a product that genuinely has no facets.
  2. **Everything already ingested**, whenever the canonicalization RULES change — the
     similarity threshold, a curated alias from `/lock`, a golden-set correction. None of it
     could reach an existing catalog.

Idempotence and resumability
----------------------------
`products.facet_canonicalization_version` is the cursor. A product is swept when its version
is below `target_version`; the write bumps it. So a run that dies halfway resumes exactly
where it stopped, and a completed run is a no-op until the target is raised.

Bump `CURRENT_FACET_CANONICALIZATION_VERSION` when you change canonicalization behaviour in a
way existing products should inherit, then sweep until `remaining` reaches 0.

The one rule that matters
-------------------------
**A degraded result never bumps the version.** Marking a product swept while its attributes
are still empty would convert a retryable outage into a permanent silent zero, and it would do
so while reporting success — the exact failure shape CLAUDE.md's `ops.silent_zero` exists for.
Degraded rows stay eligible, keep their marker, and are picked FIRST on the next run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.core.supabase_client import get_supabase_client
from app.services.facets import canonicalize_product_attributes

logger = logging.getLogger(__name__)

# Raise this when a canonicalization rule change should propagate to existing products.
# History:
#   1 — #316, the first sweep: replays everything ingested before the sweep existed.
CURRENT_FACET_CANONICALIZATION_VERSION = 1

_SELECT = (
    'id, workspace_id, attributes, attributes_raw, metadata, '
    'facet_canonicalization_version'
)


def _replay_source(product: Dict[str, Any]) -> Dict[str, Any]:
    """What to feed back through the canonicalizer.

    `attributes_raw` is the contract and wins. The `metadata` fallback is for products
    ingested before that column was populated: their raw values only ever existed in the
    source metadata blob, and without the fallback the sweep would skip exactly the oldest
    products — the ones most likely to predate the current rules.
    """
    raw = product.get('attributes_raw') or {}
    if isinstance(raw, dict) and raw:
        return dict(raw)
    meta = product.get('metadata') or {}
    return dict(meta) if isinstance(meta, dict) else {}


async def recanonicalize_products(
    *,
    target_version: int = CURRENT_FACET_CANONICALIZATION_VERSION,
    max_products: int = 200,
    batch_size: int = 25,
    workspace_id: Optional[str] = None,
    degraded_only: bool = False,
) -> Dict[str, Any]:
    """Replay `attributes_raw` through the canonicalizer for stale products.

    Bounded and safe to call repeatedly. Returns per-outcome counts plus `remaining`, so the
    caller knows whether another pass is needed rather than having to guess.
    """
    supabase = get_supabase_client()

    # One embedding service for the whole sweep. Instantiating per product would re-open the
    # client for every row and turn a catalog sweep into a connection storm.
    embedding_service = None
    try:
        from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
        embedding_service = RealEmbeddingsService()
    except Exception as e:  # pragma: no cover - dependency wiring
        logger.warning(f"facet sweep: no embedding service ({e}); canonicalizer will build its own")

    summary: Dict[str, Any] = {
        'target_version': target_version,
        'scanned': 0,
        'updated': 0,
        'degraded': 0,
        'skipped_no_raw': 0,
        'failed': 0,
        'remaining': 0,
        'workspace_id': workspace_id,
        'degraded_only': degraded_only,
    }

    def _base_query():
        q = supabase.client.table('products').select(_SELECT)
        q = q.lt('facet_canonicalization_version', target_version)
        if workspace_id:
            q = q.eq('workspace_id', workspace_id)
        return q

    # Degraded rows first — they are the ones actively missing facets right now, as opposed to
    # merely being canonicalized under older rules. Two explicit passes rather than an ORDER BY
    # over a jsonb path, which PostgREST cannot express and which no index would serve.
    def _fetch(limit: int, only_degraded: bool) -> List[Dict[str, Any]]:
        q = _base_query()
        if only_degraded:
            q = q.not_.is_('metadata->facet_canonicalization->>status', 'null')
        return (q.order('id').limit(limit).execute().data or [])

    processed_ids: set[str] = set()
    passes = [True] if degraded_only else [True, False]

    for only_degraded in passes:
        while summary['scanned'] < max_products:
            want = min(batch_size, max_products - summary['scanned'])
            try:
                rows = _fetch(want + len(processed_ids), only_degraded)
            except Exception as e:
                logger.exception("facet sweep: query failed")
                summary['error'] = f"query failed: {e}"
                return summary

            rows = [r for r in rows if str(r['id']) not in processed_ids][:want]
            if not rows:
                break

            for product in rows:
                pid = str(product['id'])
                processed_ids.add(pid)
                summary['scanned'] += 1

                source = _replay_source(product)
                if not source:
                    # Nothing lossless to replay. Bump anyway: re-reading an empty raw map on
                    # every future sweep would make this product permanently expensive for no
                    # possible gain, and it is not degraded — it simply has no facets.
                    summary['skipped_no_raw'] += 1
                    try:
                        supabase.client.table('products').update(
                            {'facet_canonicalization_version': target_version}
                        ).eq('id', pid).execute()
                    except Exception as e:
                        logger.warning(f"facet sweep: version bump failed for {pid}: {e}")
                    continue

                try:
                    result = await canonicalize_product_attributes(
                        supabase,
                        source,
                        source='bulk_recanonicalize',
                        product_id=pid,
                        embedding_service=embedding_service,
                        workspace_id=product.get('workspace_id'),
                    )
                except Exception as e:
                    logger.exception(f"facet sweep: canonicalize raised for {pid}")
                    summary['failed'] += 1
                    continue

                if getattr(result, 'status', 'ok') != 'ok':
                    # Do NOT bump. See the module docstring: a degraded row that counts as
                    # swept is a silent zero with a success report on top.
                    summary['degraded'] += 1
                    logger.warning(
                        f"facet sweep: {pid} still degraded (status={result.status}) — "
                        f"left at version {product.get('facet_canonicalization_version')} for retry"
                    )
                    continue

                metadata = dict(product.get('metadata') or {})
                # The marker described a failure that has now been repaired. Leaving it would
                # keep the product at the front of every future sweep forever.
                metadata.pop('facet_canonicalization', None)

                try:
                    supabase.client.table('products').update({
                        'attributes': result.attributes,
                        'attributes_raw': result.attributes_raw,
                        'metadata': metadata,
                        'facet_canonicalization_version': target_version,
                    }).eq('id', pid).execute()
                    summary['updated'] += 1
                except Exception as e:
                    logger.exception(f"facet sweep: write failed for {pid}")
                    summary['failed'] += 1

    # What is LEFT. A sweep that quietly stops at max_products and reports only successes reads
    # as "the catalog is canonical now" when it may have covered 200 of 40,000.
    try:
        q = supabase.client.table('products').select('id', count='exact')
        q = q.lt('facet_canonicalization_version', target_version)
        if workspace_id:
            q = q.eq('workspace_id', workspace_id)
        summary['remaining'] = int(q.limit(1).execute().count or 0)
    except Exception as e:
        logger.warning(f"facet sweep: remaining count failed: {e}")
        summary['remaining'] = -1  # explicit "unknown", never a comforting 0

    logger.info(f"📊 Facet re-canonicalization sweep: {summary}")
    return summary
