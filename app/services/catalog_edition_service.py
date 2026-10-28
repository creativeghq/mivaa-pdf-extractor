"""Catalogue editions: what a newer catalogue adds, removes and changes against the one it supersedes.

Matching is the products.import_identity unique index: a product present in both editions is the
same row (Stage 4 reuses it), so a match is recorded as the OLD product with its field diffs.
Nothing is applied here -- rows land as `proposed` and an operator applies them.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MISSING_POLICIES = ("retire", "needs_replacement", "keep")
_DIFF_METADATA_KEYS = (
    "dimensions", "sizes", "available_sizes", "heat_output", "colour_options", "order_codes",
    "warranty", "material", "finish", "weight", "power", "connection",
)


def _comparable(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def compute_field_diffs(old_row: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{field: {old, new}} for the fields that differ between a stored product and its new extraction."""
    diffs: Dict[str, Dict[str, Any]] = {}
    for key in ("name", "description"):
        old_v, new_v = old_row.get(key), new_data.get(key)
        if new_v and (old_v or "").strip() != str(new_v).strip():
            diffs[key] = {"old": old_v, "new": new_v}
    old_meta = old_row.get("metadata") or {}
    new_meta = new_data.get("metadata") or {}
    for key in _DIFF_METADATA_KEYS:
        if key not in new_meta:
            continue
        old_v, new_v = _comparable(old_meta.get(key)), _comparable(new_meta.get(key))
        if new_v not in (None, "", [], {}) and old_v != new_v:
            diffs[f"metadata.{key}"] = {"old": old_v, "new": new_v}
    return diffs


def _document(client: Any, document_id: str) -> Optional[Dict[str, Any]]:
    rows = client.table("documents").select(
        "id, workspace_id, edition_label, edition_date, supersedes_document_id, missing_policy"
    ).eq("id", document_id).limit(1).execute().data
    return rows[0] if rows else None


def record_match(client: Any, workspace_id: str, new_document_id: str, old_row: Dict[str, Any],
                 new_data: Dict[str, Any]) -> None:
    """Called by Stage 4 when an extracted product already exists; a no-op unless the document supersedes another."""
    doc = _document(client, new_document_id)
    if not doc or not doc.get("supersedes_document_id"):
        return
    diffs = compute_field_diffs(old_row, new_data)
    client.table("catalog_edition_changes").insert({
        "workspace_id": workspace_id,
        "new_document_id": new_document_id,
        "old_document_id": doc["supersedes_document_id"],
        "change_kind": "changed" if diffs else "unchanged",
        "old_product_id": old_row["id"],
        "model_code": old_row.get("sku") or old_row.get("name"),
        "field_diffs": diffs,
        "proposed_action": "update" if diffs else None,
    }).execute()


def finalize_edition(client: Any, document_id: str) -> Dict[str, int]:
    """Write the added / removed rows once the job has finished. Idempotent: reruns replace `proposed` rows of those two kinds."""
    doc = _document(client, document_id)
    if not doc or not doc.get("supersedes_document_id"):
        return {}
    ws, old_doc = doc["workspace_id"], doc["supersedes_document_id"]
    client.table("catalog_edition_changes").delete() \
        .eq("new_document_id", document_id).eq("status", "proposed") \
        .in_("change_kind", ["added", "removed"]).execute()

    matched = {
        r["old_product_id"]
        for r in client.table("catalog_edition_changes").select("old_product_id")
        .eq("new_document_id", document_id).in_("change_kind", ["changed", "unchanged"]).execute().data or []
    }
    added = client.table("products").select("id, name, sku") \
        .eq("source_document_id", document_id).eq("workspace_id", ws).execute().data or []
    old_products = client.table("products").select("id, name, sku, discontinued_by_supplier_on") \
        .eq("source_document_id", old_doc).eq("workspace_id", ws).execute().data or []
    removed = [p for p in old_products if p["id"] not in matched and not p.get("discontinued_by_supplier_on")]

    policy = doc.get("missing_policy") or "keep"
    rows: List[Dict[str, Any]] = [
        {"workspace_id": ws, "new_document_id": document_id, "old_document_id": old_doc,
         "change_kind": "added", "new_product_id": p["id"], "model_code": p.get("sku") or p.get("name"),
         "proposed_action": "create"} for p in added
    ] + [
        {"workspace_id": ws, "new_document_id": document_id, "old_document_id": old_doc,
         "change_kind": "removed", "old_product_id": p["id"], "model_code": p.get("sku") or p.get("name"),
         "proposed_action": policy} for p in removed
    ]
    for i in range(0, len(rows), 200):
        client.table("catalog_edition_changes").insert(rows[i:i + 200]).execute()
    counts = {"added": len(added), "removed": len(removed), "matched": len(matched)}
    logger.info(f"   catalogue edition {document_id} vs {old_doc}: {counts}")
    return counts


def apply_edition(client: Any, workspace_id: str, document_id: str, user_id: Optional[str]) -> Dict[str, Any]:
    """Apply every `proposed` row of this edition as proposed. Returns counts and the products whose text changed."""
    doc = _document(client, document_id)
    if not doc or str(doc["workspace_id"]) != str(workspace_id):
        return {"error": "not_found"}
    rows = client.table("catalog_edition_changes").select("*") \
        .eq("new_document_id", document_id).eq("workspace_id", workspace_id).eq("status", "proposed") \
        .execute().data or []
    on = (doc.get("edition_date") or date.today().isoformat())
    result = {"published": 0, "retired": 0, "needs_replacement": 0, "updated": 0, "kept": 0, "changed_product_ids": []}

    for r in rows:
        kind, action = r["change_kind"], r.get("proposed_action")
        if kind == "added" and r.get("new_product_id"):
            client.table("products").update({"status": "active"}) \
                .eq("id", r["new_product_id"]).eq("workspace_id", workspace_id).eq("status", "draft").execute()
            result["published"] += 1
        elif kind == "removed" and r.get("old_product_id"):
            if action in ("retire", "needs_replacement"):
                patch: Dict[str, Any] = {"discontinued_by_supplier_on": on}
                if action == "needs_replacement":
                    cur = client.table("products").select("metadata").eq("id", r["old_product_id"]).limit(1).execute().data
                    meta = dict((cur[0]["metadata"] if cur else None) or {})
                    meta["needs_new_source"] = True
                    patch["metadata"] = meta
                client.table("products").update(patch) \
                    .eq("id", r["old_product_id"]).eq("workspace_id", workspace_id).execute()
                result["retired" if action == "retire" else "needs_replacement"] += 1
            else:
                result["kept"] += 1
        elif kind == "changed" and r.get("old_product_id"):
            cur = client.table("products").select("name, description, metadata") \
                .eq("id", r["old_product_id"]).eq("workspace_id", workspace_id).limit(1).execute().data
            if cur:
                patch = {}
                meta = dict(cur[0].get("metadata") or {})
                for field, d in (r.get("field_diffs") or {}).items():
                    if field.startswith("metadata."):
                        meta[field.split(".", 1)[1]] = d["new"]
                    else:
                        patch[field] = d["new"]
                patch["metadata"] = meta
                client.table("products").update(patch) \
                    .eq("id", r["old_product_id"]).eq("workspace_id", workspace_id).execute()
                result["updated"] += 1
                result["changed_product_ids"].append(r["old_product_id"])
        client.table("catalog_edition_changes").update(
            {"status": "applied", "decided_by": user_id, "decided_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", r["id"]).execute()
    return result
