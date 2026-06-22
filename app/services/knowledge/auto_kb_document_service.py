"""
Auto KB Document Service — DEPRECATED / DISABLED (2026-06-22).

Historically this created ONE KB doc PER PRODUCT for packaging, compliance,
care and certifications. That produced massive duplication: a 60-product
catalog generated ~150 near-identical docs (every product got its own
"… - Certifications" / "… - Compliance & Safety" copy of catalog-wide facts),
and each re-ingest of the catalog multiplied them again.

Those facts are now owned in exactly one place each:

  * Catalog-wide knowledge (certifications, compliance, care/cleaning,
    regulations, iconography + packing legends, brand) →
    `catalog_knowledge_extractor` + `catalog_legend_extractor_v2`, which create
    ONE doc per (catalog, knowledge-type) and attach it to every product. The
    `upsert_kb_doc` RPC dedupes these on a stable catalog identity, so
    re-ingesting the same catalog updates the doc instead of duplicating it.
  * Per-SKU packaging numbers (pieces/box, m²/box, weight, pallet config) →
    already persisted on the product's own `metadata.packaging` and rendered on
    the product detail page. No separate KB doc needed.
  * Certifications → propagated onto each product's
    `metadata.compliance.certifications` (rendered as chips) by the catalog
    extractors.

So this per-product KB-doc generator is intentionally a no-op. It is kept as a
thin shim only so existing call sites stay safe; do NOT re-enable per-product
KB-doc creation here — extend the catalog-wide extractors instead.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AutoKBDocumentService:
    """Deprecated no-op. Per-product KB-doc generation was removed 2026-06-22.

    See the module docstring for where each kind of knowledge now lives.
    """

    async def create_kb_documents_from_metadata(
        self,
        product_id: str,
        product_name: str,
        workspace_id: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Intentionally creates nothing. Catalog-wide knowledge is generated
        # once per catalog by the catalog extractors; per-SKU packaging lives on
        # the product metadata. Returning the same shape callers already expect.
        logger.debug(
            "AutoKBDocumentService is disabled (no-op) — skipping per-product "
            "KB docs for '%s'", product_name,
        )
        return {"documents_created": 0, "errors": [], "skipped": "deprecated_no_op"}
