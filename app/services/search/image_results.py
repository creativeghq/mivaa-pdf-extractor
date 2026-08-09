"""Turn image-similarity hits into rows a caller can actually use (#277).

VECS answers an image search with `{image_id, similarity_score}` — a UUID and a number.
That is the whole result. Two endpoints returned exactly that:

  * `/api/search/by-<aspect>` — what the agent calls for "find a similar texture to this"
  * `rag_service.image_similarity_search` — `strategy=image`

Neither is *wrong*; the ranking is real. But a UUID cannot be shown to a user or described
by an agent, so the answer was unusable, and the failure was invisible because the endpoint
reported success with a populated `results` array. `/api/rag/search?strategy=multi_vector`
never had this problem — it resolves to products and the route enriches them, keying on
`result.get('id')`, which image rows do not carry.

So: one enrichment, used by both, resolving each `image_id` to what it depicts —
the image itself (caption, url, page, document) and the product it is associated with, if
any. Batched: two queries for the whole result set, not two per row.

TENANCY: `workspace_id` is required and filtered on, not decorative. These ids come back
from a VECS collection whose metadata filter is the only thing scoping the search; re-scoping
here means a caller cannot enrich — and thereby read the caption and URL of — an image id
that belongs to someone else, whatever the vector store returned (invariant 1).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


async def enrich_image_rows(
    image_ids: Sequence[str],
    workspace_id: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Map `image_id -> {caption, image_url, page_number, document_id, product}`.

    Missing ids are simply absent from the result. Never raises: enrichment is additive, and
    losing it should degrade a search result rather than fail the search that produced it —
    but it IS logged, because silently returning bare rows again is the bug this fixes.
    """
    ids = [i for i in dict.fromkeys(image_ids) if i]
    if not ids or not workspace_id:
        if ids and not workspace_id:
            logger.error(
                "Refusing to enrich %d image rows without a workspace_id — "
                "unscoped enrichment would leak captions and URLs across tenants",
                len(ids),
            )
        return {}

    try:
        from app.services.core.supabase_client import get_supabase_client
        client = get_supabase_client().client
    except Exception as e:
        logger.warning("Image enrichment unavailable (no Supabase client): %s", e)
        return {}

    out: Dict[str, Dict[str, Any]] = {}

    try:
        rows = client.table("document_images").select(
            "id, document_id, image_url, caption, page_number"
        ).in_("id", ids).eq("workspace_id", workspace_id).execute()
        for r in rows.data or []:
            out[r["id"]] = {
                "document_id": r.get("document_id"),
                "image_url": r.get("image_url"),
                "caption": r.get("caption"),
                "page_number": r.get("page_number"),
                "product": None,
            }
    except Exception as e:
        logger.warning("document_images enrichment failed for %d ids: %s", len(ids), e)
        return {}

    # The product an image depicts, best association first. Only for ids that survived the
    # workspace filter above — an id that is not ours must not gain a product either.
    ours = list(out.keys())
    if ours:
        try:
            assoc = client.table("image_product_associations").select(
                "image_id, product_id, overall_score, products(id, name)"
            ).in_("image_id", ours).order("overall_score", desc=True).execute()
            for a in assoc.data or []:
                entry = out.get(a.get("image_id"))
                if entry is None or entry.get("product"):
                    continue  # keep the highest-scoring association only
                product = a.get("products") or {}
                if product.get("id"):
                    entry["product"] = {"id": product["id"], "name": product.get("name")}
        except Exception as e:
            logger.warning("product association enrichment failed: %s", e)

    logger.info(
        "🖼️ Enriched %d/%d image rows (%d with a product)",
        len(out), len(ids), sum(1 for v in out.values() if v.get("product")),
    )
    return out


def apply_enrichment(
    results: List[Dict[str, Any]],
    enrichment: Dict[str, Dict[str, Any]],
    id_key: str = "image_id",
) -> List[Dict[str, Any]]:
    """Merge enrichment into result dicts in place, and give each row a usable `name`.

    `name` resolves product name → caption → "Image p<N>". Callers render *something*
    rather than a UUID even when the image has no product and no caption.
    """
    for r in results:
        extra = enrichment.get(r.get(id_key))
        if not extra:
            continue
        r.update({k: v for k, v in extra.items() if k != "product"})
        product = extra.get("product")
        if product:
            r["product_id"] = product["id"]
            r["product_name"] = product.get("name")
        page = extra.get("page_number")
        r["name"] = (
            (product or {}).get("name")
            or extra.get("caption")
            or (f"Image p{page}" if page else "Image")
        )
    return results
