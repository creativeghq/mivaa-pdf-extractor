"""
Day-one seed of common English canonical values per facet.

Why: without seeding, the first ingest in a non-English locale becomes the
canonical row (e.g., Greek "λευκό" wins before English "white" ever lands).
With ~25 English canonicals pre-populated, every L2 cosine search has an
English target to merge into.

Usage:
    cd mivaa-pdf-extractor
    python scripts/seed_facet_canonicals.py [--dry-run] [--facet color,material]

The script is idempotent — re-running skips already-seeded canonicals (ON CONFLICT).
Safe to run anytime.

Cost: 1 Haiku call per facet (~$0.001) + 1 Voyage batch embed per facet (~$0.0005).
For 24 whitelisted facets: total ~$0.04. Pennies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import httpx

# Make the app package importable when run from the script directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.facets.facet_canonicalizer import normalize_string  # noqa: E402
from app.services.facets.facet_whitelist import CANONICALIZABLE_FACETS  # noqa: E402

logger = logging.getLogger("seed_facet_canonicals")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_MODEL = "claude-haiku-4-5-20251001"

_SEED_TOOL = {
    "name": "submit_canonical_seeds",
    "description": (
        "Return the N most common English canonical values for a product-attribute "
        "facet across catalogs spanning lighting, tiles, furniture, bathroom, "
        "kitchen, and hardware."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Lowercase English canonical values. Use simple, established "
                    "vocabulary — 'white' not 'snow white'; 'metal' not 'metallic'; "
                    "'oak' not 'oak wood'. For codes/ratings (sockets, IP, etc.), "
                    "use the literal standard ('E27', 'IP44', 'R10'). Order doesn't "
                    "matter. No duplicates. Aim for breadth of coverage, not exotic "
                    "variants."
                ),
            }
        },
        "required": ["values"],
    },
}


_FACET_GUIDANCE: Dict[str, str] = {
    "color":              "named colors used in product catalogs (e.g. white, black, grey, beige, brown, blue, green, red, yellow, gold, silver, chrome, copper, brass, ivory, cream, charcoal, anthracite, taupe, navy)",
    "available_colors":   "same as 'color' — named colors used in product catalogs",
    "material":           "primary construction materials (e.g. metal, wood, glass, ceramic, porcelain, stone, marble, granite, leather, fabric, plastic, rattan, bamboo, concrete, brass, copper, steel, aluminum)",
    "material_type":      "specific material classifications (e.g. porcelain tile, ceramic tile, natural stone, engineered wood, oak hardwood, brushed brass, tempered glass, polished marble)",
    "finish":             "single-word finish types (matte, glossy, satin, brushed, honed, polished, textured, patinated, lacquered, antiqued, hammered, smooth, rough)",
    "style":              "aesthetic categories (modern, contemporary, traditional, industrial, scandinavian, minimalist, vintage, retro, rustic, bohemian, classic, art deco, mid-century, transitional)",
    "application":        "where a material is typically used (kitchen countertop, bathroom floor, feature wall, outdoor patio, indoor flooring, wet area, ceiling, facade, accent wall)",
    "room":               "room types (bedroom, living room, kitchen, bathroom, dining room, hallway, garden, office, child room, garage, staircase)",
    "zone_intent":        "high-level surface / placement intent (floor, wall, ceiling, decor, fixture, accent, structural)",
    "socket":             "lighting socket / lamp base codes (E27, E14, GU10, G9, G13, G5, G4, GX53, GY6.35, integrated LED)",
    "light_color":        "white-light color temperature labels (warm white, neutral white, cool white, daylight, RGB)",
    "mounting_type":      "how a fixture is mounted (wall mounted, ceiling mounted, recessed, pendant, table top, floor standing, clip-on, surface mounted, track mounted)",
    "surface_pattern":    "visible pattern / layout (herringbone, chevron, stacked bond, running bond, mosaic, grid, hexagon, geometric, floral, striped, solid)",
    "slip_resistance":    "slip-rating codes (R9, R10, R11, R12, R13, A, B, C)",
    "pei_rating":         "PEI abrasion ratings (PEI I, PEI II, PEI III, PEI IV, PEI V)",
    "frost_resistance":   "frost-resistance values (yes, no, frost resistant, not frost resistant)",
    "wood_type":          "wood species (oak, walnut, pine, beech, ash, cherry, maple, mahogany, teak, birch, spruce, alder)",
    "bowl_shape":         "basin / sink bowl shapes (round, oval, rectangular, square, d-shape, asymmetric)",
    "flush_type":         "toilet flush mechanisms (single flush, dual flush, rimless, wash down, siphonic)",
    "faucet_type":        "faucet / mixer categories (basin faucet, shower faucet, bath faucet, kitchen mixer, bidet mixer, wall mounted faucet, freestanding faucet)",
    "weave":              "textile weave patterns (plain weave, twill, satin weave, jacquard, basket weave)",
    "fiber":              "textile / upholstery fibers (cotton, wool, linen, silk, polyester, velvet, leather, jute, nylon)",
    "upholstery":         "upholstery materials / treatments (fabric, leather, velvet, faux leather, suede, microfiber)",
    "ip_rating":          "ingress protection codes (IP20, IP21, IP22, IP23, IP44, IP54, IP55, IP65, IP66, IP67, IP68)",
}


def _build_user_prompt(facet_key: str, target_count: int) -> str:
    guidance = _FACET_GUIDANCE.get(facet_key, f"common English values for the '{facet_key}' product facet")
    return (
        f"Generate the {target_count} most common English canonical values for the "
        f"product-attribute facet '{facet_key}'.\n\n"
        f"Examples / scope:\n  {guidance}\n\n"
        "Rules:\n"
        "  - lowercase English only\n"
        "  - simple, established vocabulary (not exotic variants)\n"
        "  - codes / ratings stay in their canonical uppercase form (E27, IP44, R10)\n"
        "  - no duplicates, no synonyms of the same canonical concept\n"
        "  - aim for breadth of coverage, not exhaustive completeness\n\n"
        "Call submit_canonical_seeds with the list."
    )


async def _haiku_seed(facet_key: str, target_count: int, api_key: str) -> List[str]:
    payload = {
        "model": _MODEL,
        "max_tokens": 1024,
        "tools": [_SEED_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_canonical_seeds"},
        "messages": [{"role": "user", "content": _build_user_prompt(facet_key, target_count)}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        resp = await client.post(_ANTHROPIC_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    for block in (data.get("content") or []):
        if block.get("type") == "tool_use" and block.get("name") == "submit_canonical_seeds":
            return [str(v).strip() for v in ((block.get("input") or {}).get("values") or []) if str(v).strip()]
    return []


async def _voyage_embed(texts: List[str], voyage_key: str) -> List[Optional[List[float]]]:
    payload = {
        "model": "voyage-4",
        "input": texts,
        "input_type": "document",
        "truncation": True,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        resp = await client.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {voyage_key}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return [item.get("embedding") for item in (data.get("data") or [])]


async def _supabase_request(supabase_url: str, service_key: str, path: str, *, method: str = "POST", payload: Optional[dict] = None) -> dict:
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
        if method == "GET":
            resp = await client.get(f"{supabase_url}{path}", headers=headers)
        else:
            resp = await client.request(method, f"{supabase_url}{path}", headers=headers, json=payload)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {}


async def seed_facet(
    facet_key: str,
    target_count: int,
    anthropic_key: str,
    voyage_key: str,
    supabase_url: str,
    supabase_service_key: str,
    dry_run: bool,
) -> dict:
    started = time.time()
    raw_values = await _haiku_seed(facet_key, target_count, anthropic_key)
    if not raw_values:
        return {"facet_key": facet_key, "generated": 0, "inserted": 0, "elapsed_ms": int((time.time()-started)*1000), "skipped_reason": "haiku_returned_empty"}

    # Normalize + dedupe
    normalized = []
    seen: set = set()
    for v in raw_values:
        n = normalize_string(v)
        if not n or not n.isascii():
            continue
        if n in seen:
            continue
        seen.add(n)
        normalized.append(n)

    if dry_run:
        logger.info(f"[dry-run] {facet_key}: would seed {len(normalized)} canonicals: {normalized}")
        return {"facet_key": facet_key, "generated": len(normalized), "inserted": 0, "elapsed_ms": int((time.time()-started)*1000), "skipped_reason": "dry_run"}

    embeddings = await _voyage_embed(normalized, voyage_key)
    rows = []
    for canonical, emb in zip(normalized, embeddings):
        if emb is None or len(emb) != 1024:
            logger.warning(f"  skipping '{canonical}' — bad embedding (got {len(emb) if emb else 'None'} dims)")
            continue
        # halfvec(1024) over PostgREST requires the string form '[0.1,0.2,...]'.
        # Sending a JSON array silently fails the implicit cast on some Supabase
        # versions; the stringified form matches the format Stage 4 uses for
        # text_embedding_1024 (see pdf_processing/stage_4_products.py:646).
        embedding_str = '[' + ','.join(str(x) for x in emb) + ']'
        rows.append({
            "facet_key": facet_key,
            "canonical_value": canonical,
            "embedding": embedding_str,
            "embedding_model": "voyage-4",
            "aliases": [canonical],
            "alias_count": 0,
        })

    if not rows:
        return {"facet_key": facet_key, "generated": len(normalized), "inserted": 0, "elapsed_ms": int((time.time()-started)*1000), "skipped_reason": "no_valid_embeddings"}

    # Use Supabase REST upsert via on_conflict (no_op via Prefer: resolution=ignore-duplicates)
    headers = {
        "apikey": supabase_service_key,
        "Authorization": f"Bearer {supabase_service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=ignore-duplicates,return=representation",
    }
    inserted = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
        resp = await client.post(
            f"{supabase_url}/rest/v1/facet_canonical_values?on_conflict=facet_key,canonical_value",
            headers=headers,
            json=rows,
        )
        if resp.status_code >= 400:
            logger.error(f"  upsert failed for {facet_key}: {resp.status_code} {resp.text[:200]}")
        else:
            try:
                inserted = len(resp.json() or [])
            except Exception:
                inserted = len(rows)

    return {"facet_key": facet_key, "generated": len(normalized), "inserted": inserted, "elapsed_ms": int((time.time()-started)*1000)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print what would be seeded without writing")
    parser.add_argument("--target-count", type=int, default=25, help="Canonical values per facet (default 25)")
    parser.add_argument("--facet", type=str, default=None, help="Comma-separated facet keys (default: all whitelisted)")
    args = parser.parse_args()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or ""
    voyage_key = os.getenv("VOYAGE_API_KEY") or ""
    supabase_url = os.getenv("SUPABASE_URL") or ""
    supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY") or ""

    if not args.dry_run:
        missing = []
        if not anthropic_key:        missing.append("ANTHROPIC_API_KEY")
        if not voyage_key:           missing.append("VOYAGE_API_KEY")
        if not supabase_url:         missing.append("SUPABASE_URL")
        if not supabase_service_key: missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if missing:
            logger.error(f"Missing env vars: {missing}")
            sys.exit(1)
    elif not anthropic_key:
        logger.error("Even --dry-run needs ANTHROPIC_API_KEY to generate seed values")
        sys.exit(1)

    if args.facet:
        facets = [f.strip() for f in args.facet.split(",") if f.strip()]
    else:
        facets = sorted(CANONICALIZABLE_FACETS)

    summary = []
    for facet_key in facets:
        try:
            result = await seed_facet(
                facet_key=facet_key,
                target_count=args.target_count,
                anthropic_key=anthropic_key,
                voyage_key=voyage_key,
                supabase_url=supabase_url,
                supabase_service_key=supabase_service_key,
                dry_run=args.dry_run,
            )
            summary.append(result)
            logger.info(f"{facet_key}: generated={result['generated']} inserted={result['inserted']} ({result['elapsed_ms']}ms)")
        except Exception as e:
            logger.error(f"{facet_key}: exception {e}")
            summary.append({"facet_key": facet_key, "error": str(e)[:200]})

    print(json.dumps({"summary": summary, "total_facets": len(facets), "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
