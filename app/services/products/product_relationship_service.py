"""
Product Relationship Service

Related products are served from the gold-layer `product_edges` table:
- material_family (same material type)
- pattern_match (same finish + overlapping colors)
- collection (same collection / designer / factory)
- complementary (products that work together, via the category map)
- alternative (similar technical specs — slip, fire rating, dimensions)

These five deterministic edge types are DERIVED IN SQL from silver
(`products.attributes` canonical facets, else `products.metadata`) by the
`rebuild_product_edges(workspace)` RPC and read back with a single indexed
query via `get_related_products(...)`. The old per-query full-catalog Python
scans (five workspace-wide table pulls per call) are gone — the read path is
now O(neighbours), and the relationships are persisted so every consumer
(search enrichment, agent tools, moodboard/quote surfaces) shares one edge set.

`custom` relationships stay LIVE: they are per-prompt and not a stable edge, so
they are evaluated by the LLM at query time and never persisted.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from app.config import get_settings
from app.services.core.ai_client_service import get_ai_client_service

logger = logging.getLogger(__name__)

# Edge types materialised in `product_edges` (served by the read RPC).
_STANDARD_TYPES = [
    'material_family',
    'pattern_match',
    'collection',
    'complementary',
    'alternative',
]


class ProductRelationshipService:
    """Service for finding related products from persisted gold-layer edges."""

    def __init__(self, supabase_client):
        """Initialize the service with Supabase client."""
        self.supabase = supabase_client
        self.settings = get_settings()

    async def find_related_products(
        self,
        product_id: str,
        workspace_id: str,
        relationship_types: Optional[List[str]] = None,
        limit: int = 5,
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find related products for a given product.

        The five standard relationship types are read from the persisted
        `product_edges` table in one indexed round trip. `custom` (if requested
        with a prompt) is evaluated live by the LLM and merged in.

        Args:
            product_id: ID of the product to find relationships for
            workspace_id: Workspace ID for scoped search
            relationship_types: Subset of
                {material_family, pattern_match, collection, complementary,
                 alternative, custom}. None => all standard types.
            limit: Maximum number of related products to return
            custom_prompt: Optional NLP prompt for the `custom` type

        Returns:
            List of related products with relationship metadata, each:
            {id, name, description, relationship_type, relevance_score, reason, metadata}
        """
        try:
            if relationship_types is None:
                requested_standard: Optional[List[str]] = None  # all
                wants_custom = False
            else:
                requested_standard = [t for t in relationship_types if t in _STANDARD_TYPES]
                wants_custom = 'custom' in relationship_types

            all_related: List[Dict[str, Any]] = []

            # Persisted edges — single indexed read (None p_types => all standard types).
            if requested_standard is None or requested_standard:
                # Pull a slightly larger pool when we still have to merge custom
                # results, so the final top-`limit` reflects both sources.
                pool = max(limit, 20) if (wants_custom and custom_prompt) else limit
                edges_response = self.supabase.rpc(
                    'get_related_products',
                    {
                        'p_workspace_id': workspace_id,
                        'p_product_id': product_id,
                        'p_types': requested_standard,  # None => all
                        'p_limit': pool,
                    },
                ).execute()
                all_related.extend(edges_response.data or [])

            # Custom (live LLM, per-prompt, never persisted).
            if wants_custom and custom_prompt:
                source_response = self.supabase.table('products').select(
                    'id, name, description, metadata'
                ).eq('id', product_id).eq('workspace_id', workspace_id).single().execute()
                if source_response.data:
                    custom_related = await self._find_custom_relationships(
                        source_response.data, workspace_id, product_id, custom_prompt
                    )
                    all_related.extend(custom_related)

            # Deduplicate (keep highest score per product), sort, limit.
            deduplicated = self._deduplicate_products(all_related)
            deduplicated.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
            return deduplicated[:limit]

        except Exception as e:
            logger.error(f"Error finding related products: {e}", exc_info=True)
            return []

    async def rebuild_edges(self, workspace_id: str) -> int:
        """
        Rebuild all SQL-derived edges for a workspace from silver.

        Idempotent and cheap to re-run — it rebuilds gold in place (no bronze /
        PDF re-read). Call after a batch of product writes (ingest, XML import,
        re-canonicalization). Preserves any llm/manual edges.

        Returns the number of edges written.
        """
        try:
            response = self.supabase.rpc(
                'rebuild_product_edges', {'p_workspace_id': workspace_id}
            ).execute()
            count = response.data if isinstance(response.data, int) else 0
            logger.info(f"Rebuilt {count} product edges for workspace {workspace_id}")
            return count
        except Exception as e:
            logger.error(
                f"Error rebuilding product edges for workspace {workspace_id}: {e}",
                exc_info=True,
            )
            return 0

    async def _find_custom_relationships(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str,
        custom_prompt: str
    ) -> List[Dict[str, Any]]:
        """Find products using a custom LLM-evaluated prompt against the product catalog."""
        try:
            # Fetch candidate products from workspace
            response = self.supabase.table("products").select(
                "id, name, description, material_type, color, finish, collection"
            ).eq("workspace_id", workspace_id).neq("id", exclude_id).limit(50).execute()

            if not response.data:
                return []

            candidates = response.data

            # Ask Claude to select related products based on the custom prompt
            ai = get_ai_client_service()
            source_summary = {
                "name": source_product.get("name", ""),
                "description": source_product.get("description", ""),
                "material_type": source_product.get("material_type", ""),
                "color": source_product.get("color", ""),
                "finish": source_product.get("finish", ""),
                "collection": source_product.get("collection", ""),
            }
            candidate_list = [
                {"index": i, "id": c["id"], "name": c.get("name", ""),
                 "material_type": c.get("material_type", ""), "color": c.get("color", ""),
                 "finish": c.get("finish", "")}
                for i, c in enumerate(candidates)
            ]

            prompt = (
                f"{custom_prompt}\n\n"
                f"Source product: {json.dumps(source_summary)}\n\n"
                f"Candidate products: {json.dumps(candidate_list)}\n\n"
                "Return a JSON array of objects with 'index' and 'relevance_score' (0.0-1.0) "
                "for candidates that match the custom relationship criteria. "
                "Include only matches with score >= 0.5. "
                "Example: [{\"index\": 2, \"relevance_score\": 0.85}]. "
                "Return ONLY the JSON array."
            )

            from app.services.core.claude_helper import tracked_claude_call_async
            resp = await tracked_claude_call_async(
                task="product_relationship_detection",
                model="claude-haiku-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            matches = json.loads(resp.content[0].text.strip())

            related = []
            for match in matches:
                idx = match.get("index")
                score = float(match.get("relevance_score", 0.5))
                if idx is not None and idx < len(candidates):
                    product = candidates[idx]
                    related.append({
                        **product,
                        "relevance_score": score,
                        "relationship_type": "custom",
                    })

            logger.info(f"Custom relationship detection found {len(related)} matches")
            return related

        except Exception as e:
            logger.error(f"Error in custom relationship detection: {e}")
            return []

    def _deduplicate_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate products, keeping the one with highest relevance score."""
        seen: Dict[str, Dict[str, Any]] = {}
        for product in products:
            product_id = product['id']
            if product_id not in seen or product.get('relevance_score', 0.0) > seen[product_id].get('relevance_score', 0.0):
                seen[product_id] = product

        return list(seen.values())
