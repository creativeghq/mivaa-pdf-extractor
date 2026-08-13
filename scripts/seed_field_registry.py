"""
Seed `material_metadata_fields` from the Python registries (#347 phase 3.1).

WHY A SCRIPT AND NOT A MIGRATION
--------------------------------
The data is ~290 rows derived from two Python sources. Hand-transcribing that into a SQL
migration would create a third copy of the same facts, frozen at one moment — precisely the
problem this phase exists to remove. This reads the sources and upserts, so it is idempotent,
re-runnable, and reviewable as a diff when the sources change.

It is the MIGRATION PATH, not a permanent fixture. Once the DB is authoritative
(`is_canonicalizable()` reading `canonicalize`, MIVAA building prompts from `description`), the
Python registries and this script are deleted together — phases 3.2 to 3.4.

WHAT IT DERIVES
---------------
* field_name / description / applies_to_categories  <- CATEGORY_FIELD_REGISTRY (276 unique fields
  across the 10 categories; the same key in several categories becomes ONE row scoped to all).
* canonicalize                                      <- CANONICALIZABLE_FACETS (24 keys).
* role                                              <- the seed below, plus the plurality rule.
* destination                                       <- COLUMN_BACKED below.

TWO DISAGREEMENTS THIS SURFACED, both silent until the sources were compared:

  1. 16 of the 24 canonicalizable facets are NEVER requested by the extraction prompt —
     `color`, `material`, `room`, `zone_intent`, `wood_type`, `bowl_shape`, `flush_type`,
     `faucet_type`, `socket`, `light_color`, `surface_pattern`, `available_colors`,
     `application`, `upholstery`, `weave`, `fiber`. The facet system is allowed to canonicalise
     them, but nothing ever asks the model for them, so they can only arrive if it volunteers
     them under exactly that name. They are seeded here so the registry knows they exist.
  2. `material_categories` has ELEVEN categories; the Python registry knows ten. Nothing
     references `building_materials`.

ON `role`
---------
The plan's test is "do two units differing only in this field sit in DIFFERENT PILES in the
warehouse?" The plurality rule alone (a field the catalog enumerates for one product is a
variant axis) proved too weak — it found five, two of them wrong (`standards`,
`vision_variants` are not axes). So IDENTITY_SEED below is an explicit first pass, and it is
exactly the ~40 decisions the plan says a human owes. **Treat it as a draft to review, not an
answer.** Everything not seeded defaults to `descriptive`, which is the direction that changes
no behaviour.

Usage:  python scripts/seed_field_registry.py [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# ── The identity seed — the reviewable human pass ───────────────────────────────────────────
# "Two units differing only in this sit in different piles in the warehouse."
IDENTITY_SEED: set[str] = {
    # cross-category
    "available_sizes", "available_colors", "colors", "color", "finish", "material",
    "size", "material_subtype",
    # tiles — 60x60 matt and 30x60 matt are different boxes; PEI and shade variation are not
    "thickness_mm", "rectified", "format_code", "body_type",
    # lighting — 2700K and 4000K are different boxes. lumens_lm is a CONSEQUENCE of the fitting
    # you already chose, which is the pair that makes this test concrete.
    "color_temperature_k", "lamp_type", "wattage_w", "dimmable", "ip_rating",
    "number_of_lights", "body_material", "shade_material", "mounting_type",
    # sanitary
    "bowl_shape", "flush_type", "faucet_type",
    # wood
    "wood_type", "core_type",
    # furniture / textiles
    "fabric_options", "upholstery", "weave", "fiber",
}

#: Plural-looking, but a property of the chosen product rather than an axis you pick along.
NOT_AXES: set[str] = {
    "standards", "vision_variants", "lumens_lm", "certifications", "sku_codes",
    "product_codes", "grout_suppliers", "grout_color_codes", "grout_details",
    "environments", "application_areas", "patterns",
}

#: Fields with a real column already. Extraction must populate the COLUMN, not a jsonb key
#: nothing reads — `product_packaging` holds the irreducible pack facts, and `order_items`
#: snapshots customs per line.
COLUMN_BACKED: set[str] = {
    "pieces_per_box", "boxes_per_pallet", "country_of_origin", "taric_code",
}


def _literal(path: Path, name: str) -> Any:
    """Read a module-level assignment without importing the module (no app deps, no DB client)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign) and node.targets:
            target = getattr(node.targets[0], "id", None)
        if target == name:
            return ast.literal_eval(ast.unparse(node.value))
    raise SystemExit(f"{name} not found in {path}")


def build_rows() -> list[dict[str, Any]]:
    registry = _literal(ROOT / "app/services/metadata/category_field_registry.py",
                        "CATEGORY_FIELD_REGISTRY")
    canon: set[str] = set(_literal(ROOT / "app/services/facets/facet_whitelist.py",
                                   "CANONICALIZABLE_FACETS"))

    merged: dict[str, dict[str, Any]] = {}
    for category, config in registry.items():
        for section, fields in config.get("priority_fields", {}).items():
            for key, description in fields:
                row = merged.setdefault(key, {"cats": set(), "desc": description, "section": section})
                row["cats"].add(category)
                # Longest description wins: the same field is described more fully in some
                # categories than others, and the registry is where that text has to live.
                if len(description) > len(row["desc"]):
                    row["desc"] = description

    def role_for(key: str) -> str:
        if key in NOT_AXES:
            return "descriptive"
        if key in IDENTITY_SEED or key.startswith("available_"):
            return "identity"
        return "descriptive"

    rows = [
        {
            "field_name": key,
            "display_name": key.replace("_", " ").title(),
            "description": data["desc"][:400],
            "applies_to_categories": sorted(data["cats"]),
            "role": role_for(key),
            "canonicalize": key in canon,
            "destination": "column" if key in COLUMN_BACKED else "metadata",
            "extraction_hints": data["section"],
            "status": "active",
        }
        for key, data in sorted(merged.items())
    ]

    # Facets the whitelist canonicalises that extraction never asks for. Seeded with a NULL
    # category scope (they apply anywhere) so the registry at least knows they are real.
    for key in sorted(canon - set(merged)):
        rows.append({
            "field_name": key,
            "display_name": key.replace("_", " ").title(),
            "description": ("Canonicalizable facet. Was in facet_whitelist but never requested by "
                            "any category's extraction prompt — see #347 phase 3."),
            "applies_to_categories": None,
            "role": role_for(key),
            "canonicalize": True,
            "destination": "metadata",
            "extraction_hints": "appearance",
            "status": "active",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print a summary and write no rows")
    args = parser.parse_args()

    rows = build_rows()
    identity = [r["field_name"] for r in rows if r["role"] == "identity"]
    print(f"fields            : {len(rows)}")
    print(f"  role=identity   : {len(identity)}")
    print(f"  canonicalize    : {sum(1 for r in rows if r['canonicalize'])}")
    print(f"  dest=column     : {sum(1 for r in rows if r['destination'] == 'column')}")
    print(f"  identity fields : {', '.join(sorted(identity))}")

    if args.dry_run:
        (ROOT / "registry_seed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("\n--dry-run: wrote registry_seed.json, no rows written")
        return 0

    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("\nSUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are required to write. Use --dry-run to "
              "inspect the derived rows without them.", file=sys.stderr)
        return 2

    from supabase import create_client  # imported late so --dry-run needs no deps

    client = create_client(url, key)
    written = 0
    for start in range(0, len(rows), 100):
        batch = rows[start:start + 100]
        # on_conflict on field_name: re-running must update, never duplicate. The unique index
        # added in the same phase is what makes that safe.
        result = client.table("material_metadata_fields").upsert(
            batch, on_conflict="field_name",
        ).execute()
        written += len(result.data or [])
        print(f"  upserted {written}/{len(rows)}")
    print(f"\ndone: {written} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
