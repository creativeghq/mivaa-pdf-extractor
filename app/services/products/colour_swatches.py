"""Measure a hex for every product colour from its own catalogue pages, or record why not.

A colour leaves ingestion with `hex` or with `hex_status` naming the reason (no_sample, ambiguous,
pattern_name, patterned, no_pages, shared_page) — never silently without either, which is what
`catalog.colour_without_hex` probes for.
"""

import asyncio
from typing import Any, Dict, Iterable, List, Optional, Set

from app.services.products import swatch_sampler as S

ROLLUP_LIMIT = 200


def product_pages(metadata: Dict[str, Any]) -> List[int]:
    pages: Set[int] = set()
    for key in ("page_range", "catalogue_page_number", "catalogue_detail_page", "_start_page"):
        raw = metadata.get(key)
        for v in raw if isinstance(raw, list) else [raw]:
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n > 0:
                pages.add(n)
    return sorted(pages)


def _needs_measuring(options: Iterable[Any]) -> bool:
    return any(isinstance(o, dict) and not o.get("hex") for o in options or [])


def measure_product(name: str, options: List[Dict[str, Any]], samples: List[Dict[str, Any]],
                    foreign_pages: Set[int], checked_at: str) -> Dict[str, Any]:
    """New colour_options for one product, each colour carrying `hex` or `hex_status`."""
    own = [s for s in samples if s.get("page") not in foreign_pages]
    shared = [s for s in samples if s.get("page") in foreign_pages]
    # A bare "Gris" caption on a page that also prints another series could be either one's.
    usable = own + [s for s in shared if S.fold(name) in S.fold(s.get("label") or "")]
    new, report = S.merge_hex_into_colour_options(options, usable, series=name)
    out = []
    for opt, rep in zip(new, report):
        if opt.get("hex"):
            out.append({k: v for k, v in opt.items() if k not in ("hex_status", "hex_checked_at")})
            continue
        reason = rep.get("reason") or "no_sample"
        if reason == "no_sample" and shared and not own:
            reason = "shared_page"
        out.append({**opt, "hex_status": reason, "hex_checked_at": checked_at})
    applied = sum(1 for r in report if r.get("status") == "applied")
    return {"colour_options": out, "applied": applied, "unmeasured": [o["name"] for o in out if not o.get("hex")]}


async def sample_document_swatches(supabase: Any, document_id: str, pdf_bytes: Optional[bytes], *,
                                   apply: bool = True, job_id: Optional[str] = None) -> Dict[str, Any]:
    """Fill `colour_options[].hex` for every product of a document from the catalogue pages."""
    from datetime import datetime, timezone

    import fitz

    from app.services.discovery.series_facts import _all
    from app.services.pdf.ingestion_corrections import CORRECTED, DEGRADED, Correction, record

    db = getattr(supabase, "client", supabase)
    document = (db.table("documents").select("id, workspace_id").eq("id", document_id).limit(1).execute().data or [None])[0]
    if not document:
        return {"error": "document not found"}
    products = _all(lambda: db.table("products").select("id, name, metadata").eq("source_document_id", document_id).order("id"))
    todo = [p for p in products if _needs_measuring((p.get("metadata") or {}).get("colour_options"))]
    summary: Dict[str, Any] = {"products": len(todo), "colours_measured": 0, "colours_unmeasured": 0, "failed": [],
                               "per_product": []}
    if not todo:
        return summary
    checked_at = datetime.now(timezone.utc).isoformat()
    if not pdf_bytes:
        for p in todo:
            m = p.get("metadata") or {}
            m["colour_options"] = [o if o.get("hex") else {**o, "hex_status": "no_pdf", "hex_checked_at": checked_at}
                                   for o in m["colour_options"] if isinstance(o, dict)]
            if apply:
                db.table("products").update({"metadata": m}).eq("id", p["id"]).execute()
        summary["colours_unmeasured"] = sum(len(p["metadata"]["colour_options"]) for p in todo)
        return summary

    names = [S.fold(p["name"]) for p in products if p.get("name")]

    def measure() -> List[Dict[str, Any]]:
        import tempfile

        from app.utils.pdf_to_images import analyze_pdf_layout

        # Product pages are PHYSICAL (printed) pages; on a spread PDF two of them share one PDF page.
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            physical = analyze_pdf_layout(tmp.name).physical_to_pdf_map
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        by_pdf_page: Dict[int, List[Dict[str, Any]]] = {}
        cache: Dict[int, List[Dict[str, Any]]] = {}
        text: Dict[int, str] = {}

        def physical_page(n: int) -> None:
            idx, side = physical[n]
            if idx not in by_pdf_page:
                by_pdf_page[idx] = S.sample_page(None, idx + 1, doc=doc)
            page = doc[idx]
            mid = page.rect.width / 2.0
            half = {"left": fitz.Rect(0, 0, mid, page.rect.height),
                    "right": fitz.Rect(mid, 0, page.rect.width, page.rect.height)}.get(side, page.rect)
            cache[n] = [{**s, "page": n} for s in by_pdf_page[idx]
                        if half.x0 <= (s["rect"][0] + s["rect"][2]) / 2.0 <= half.x1]
            text[n] = S.fold(page.get_text("text", clip=half))

        results = []
        try:
            for p in todo:
                m = dict(p.get("metadata") or {})
                pages = [n for n in product_pages(m) if n in physical]
                if not pages:
                    opts = [o if o.get("hex") else {**o, "hex_status": "no_pages", "hex_checked_at": checked_at}
                            for o in m["colour_options"] if isinstance(o, dict)]
                    results.append({"product": p, "colour_options": opts, "applied": 0,
                                    "unmeasured": [o["name"] for o in opts if not o.get("hex")]})
                    continue
                samples: List[Dict[str, Any]] = []
                foreign: Set[int] = set()
                own = S.fold(p["name"])
                for n in pages:
                    if n not in cache:
                        physical_page(n)
                    samples.extend(cache[n])
                    if any(o != own and len(o) > 2 and f" {o} " in f" {text[n]} " for o in names):
                        foreign.add(n)
                r = measure_product(p["name"], [o for o in m["colour_options"] if isinstance(o, dict)],
                                    samples, foreign, checked_at)
                results.append({"product": p, **r})
        finally:
            doc.close()
        return results

    try:
        results = await asyncio.to_thread(measure)
    except Exception as e:
        return {**summary, "error": f"{type(e).__name__}: {e}"[:300]}
    measured_names: List[str] = []
    unmeasured_names: List[str] = []
    for r in results:
        p = r["product"]
        summary["colours_measured"] += r["applied"]
        summary["colours_unmeasured"] += len(r["unmeasured"])
        summary["per_product"].append({"product": p["name"], "colours": {
            o.get("name"): o.get("hex") or o.get("hex_status") for o in r["colour_options"]}})
        if r["applied"]:
            measured_names.append(p["name"])
        if r["unmeasured"]:
            unmeasured_names.append(f"{p['name']}: {', '.join(r['unmeasured'])}")
        if apply:
            try:
                m = dict(p.get("metadata") or {})
                m["colour_options"] = r["colour_options"]
                db.table("products").update({"metadata": m}).eq("id", p["id"]).execute()
            except Exception as e:
                summary["failed"].append(f"{p['name']}: {type(e).__name__}: {str(e)[:160]}")
    if apply:
        corrections = []
        if summary["colours_measured"]:
            corrections.append(Correction(
                "colour_hex_measured", CORRECTED,
                f"{summary['colours_measured']} colour(s) got a hex measured from their catalogue swatch",
                summary["colours_measured"], measured_names[:ROLLUP_LIMIT]))
        if summary["colours_unmeasured"]:
            corrections.append(Correction(
                "colour_hex_unmeasured", DEGRADED,
                f"{summary['colours_unmeasured']} colour(s) have no swatch the catalogue labels unambiguously; "
                "each carries hex_status with the reason",
                summary["colours_unmeasured"], unmeasured_names[:ROLLUP_LIMIT]))
        if corrections:
            await record(db, document_id, document.get("workspace_id"), "colour_swatches", corrections, job_id)
    return summary
