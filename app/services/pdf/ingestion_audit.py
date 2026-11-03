from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from app.services.pdf.ocr_completeness import numeric_tokens

PASS = "pass"
WARN = "warn"
FAIL = "fail"
UNVERIFIABLE = "unverifiable"

PAGE_MISSING_WARN = 0.02
PAGE_MISSING_FAIL = 0.05
MIN_NUMBERS_PER_PAGE = 12
MIN_MISSING_FOR_FAIL = 3
CODE_COVERAGE_WARN = 0.99
CODE_COVERAGE_FAIL = 0.95
MIN_CODES_FOR_SHAPE = 8
MAX_EXAMPLES = 25

PRICE_TEXT = re.compile("[€$£]|[0-9] ?(eur|euro)(?![a-z])|(?<![a-z])(price|prezzo|preis|prix|precio|τιμή)(?![a-z])", re.I)
CODE_TOKEN = re.compile("[A-Za-z0-9][A-Za-z0-9._/-]{3,}")


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    count: int = 0
    examples: List[Any] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "count": self.count,
            "examples": self.examples[:MAX_EXAMPLES],
        }


@dataclass
class AuditResult:
    status: str
    metrics: Dict[str, Any]
    findings: List[Finding]

    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "metrics": self.metrics, "findings": [f.as_dict() for f in self.findings]}


def code_signature(token: str) -> str:
    out = []
    for ch in token:
        out.append("9" if ch.isdigit() else ("A" if ch.isalpha() else ch))
    return "".join(out)


def dominant_code_signature(first_column_values: Iterable[Any]) -> Optional[str]:
    sigs = Counter(
        code_signature(str(v).strip())
        for v in first_column_values
        if v is not None and CODE_TOKEN.fullmatch(str(v).strip()) and any(c.isdigit() for c in str(v))
    )
    if not sigs:
        return None
    signature, count = sigs.most_common(1)[0]
    return signature if count >= MIN_CODES_FOR_SHAPE else None


def tokens_with_signature(text: str, signature: str) -> Set[str]:
    return {t for t in CODE_TOKEN.findall(text or "") if code_signature(t) == signature}


def page_number_gaps(pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    gaps = []
    for page in pages:
        reference: Set[str] = page.get("reference_numbers") or set()
        if len(reference) < MIN_NUMBERS_PER_PAGE:
            continue
        have: Set[str] = page.get("platform_numbers") or set()
        missing = reference - have
        share = len(missing) / len(reference)
        if share >= PAGE_MISSING_WARN:
            gaps.append({"page": page["page"], "missing_share": round(share, 3), "missing": len(missing), "of": len(reference), "route": page.get("route"), "examples": sorted(missing)[:8], "all": sorted(missing)[:80]})
    return gaps


def coverage_findings(pages: Sequence[Dict[str, Any]], findings: List[Finding], metrics: Dict[str, Any]) -> None:
    gaps = page_number_gaps(pages)
    checked = sum(1 for p in pages if len(p.get("reference_numbers") or ()) >= MIN_NUMBERS_PER_PAGE)
    metrics["pages_checked_for_numbers"] = checked
    metrics["pages_with_missing_numbers"] = len(gaps)
    if checked:
        total_ref = sum(len(p["reference_numbers"]) for p in pages if len(p.get("reference_numbers") or ()) >= MIN_NUMBERS_PER_PAGE)
        total_have = sum(len(p["reference_numbers"] & (p.get("platform_numbers") or set())) for p in pages if len(p.get("reference_numbers") or ()) >= MIN_NUMBERS_PER_PAGE)
        metrics["number_coverage"] = round(total_have / total_ref, 4) if total_ref else None
    hard = [g for g in gaps if g["missing_share"] >= PAGE_MISSING_FAIL and g["missing"] >= MIN_MISSING_FOR_FAIL]
    soft = [g for g in gaps if g not in hard]
    if hard:
        findings.append(Finding(FAIL, "numbers_missing", f"{len(hard)} page(s) have 5% or more of the numbers printed in the PDF missing from what the platform stored", len(hard), hard))
    if soft:
        findings.append(Finding(WARN, "numbers_missing_few", f"{len(soft)} page(s) are missing 2-5% of their printed numbers", len(soft), soft))


def code_findings(tables: Sequence[Dict[str, Any]], pdf_text: Optional[str], platform_text: str, findings: List[Finding], metrics: Dict[str, Any]) -> None:
    keys = [row[0] for t in tables for row in ((t.get("table_data") or {}).get("rows") or []) if row]
    signature = dominant_code_signature(keys)
    metrics["code_signature"] = signature
    if not signature or not pdf_text:
        return
    inventory = tokens_with_signature(pdf_text, signature)
    have = tokens_with_signature(platform_text, signature)
    missing = sorted(inventory - have)
    metrics["codes_in_pdf"] = len(inventory)
    metrics["codes_on_platform"] = len(inventory & have)
    coverage = (len(inventory) - len(missing)) / len(inventory) if inventory else 1.0
    metrics["code_coverage"] = round(coverage, 4)
    if coverage < CODE_COVERAGE_FAIL:
        findings.append(Finding(FAIL, "codes_missing", f"{len(missing)} item code(s) printed in the PDF are not in the saved tables or chunks", len(missing), missing))
    elif missing:
        findings.append(Finding(WARN, "codes_missing_few", f"{len(missing)} item code(s) printed in the PDF are not saved", len(missing), missing))


UNSUPPORTED_WARN = 0.05


def table_cell_numbers(tables: Sequence[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for t in tables:
        for row in ((t.get("table_data") or {}).get("rows") or []):
            for cell in row or []:
                out |= numeric_tokens(str(cell))
    return out


def unsupported_findings(tables: Sequence[Dict[str, Any]], reference: Set[str], findings: List[Finding], metrics: Dict[str, Any]) -> None:
    if not reference:
        return
    stored = table_cell_numbers(tables)
    if not stored:
        return
    unsupported = sorted(stored - reference)
    share = len(unsupported) / len(stored)
    metrics["table_numbers"] = len(stored)
    metrics["table_numbers_not_in_source"] = len(unsupported)
    if share >= UNSUPPORTED_WARN:
        findings.append(Finding(WARN, "numbers_not_in_source", f"{len(unsupported)} of {len(stored)} numbers in the saved tables do not appear in the source (derived values, or a misread)", len(unsupported), unsupported))


def catalogue_findings(plans: Sequence[Dict[str, Any]], unmatched: Sequence[Dict[str, Any]], tables: Sequence[Dict[str, Any]], findings: List[Finding], metrics: Dict[str, Any]) -> None:
    have = {(t.get("product_id"), t.get("page_number")) for t in tables if (t.get("metadata") or {}).get("scope") == "catalogue_wide"}
    missing = [{"page": p["page"], "title": p["title"]} for p in plans for pid in p["products"] if (pid, p["page"]) not in have]
    metrics["catalogue_tables_unattached"] = len(missing)
    metrics["catalogue_tables_without_product"] = len(unmatched)
    if missing:
        findings.append(Finding(WARN, "catalogue_tables_not_attached", f"{len(missing)} table(s) on pages that belong to no product are not attached to the products their heading names", len(missing), missing))
    if unmatched:
        findings.append(Finding(WARN, "catalogue_tables_without_product", f"{len(unmatched)} table(s) on catalogue-wide pages have a heading that names no product", len(unmatched), unmatched))


GENERIC_WORDS = {"stella", "acoustic", "panels", "panel", "wall", "mdf", "interior", "slats", "slat", "for", "the", "and", "profile", "plate", "mounting"}
WEB_UNSUPPORTED_WARN = 0.05
WEB_UNSUPPORTED_FAIL = 0.25


def product_key(name: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (name or "").lower()) if w not in GENERIC_WORDS]


def match_snapshots(name: str, snapshots: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    key = product_key(name)
    if not key:
        return []
    joined = "_".join(key)
    out = []
    for snap in snapshots:
        title = ((snap.get("text") or "").strip().split("\n") or [""])[0].lower()
        words = re.findall(r"[a-z0-9]+", title)
        if any(w.isalpha() for w in words):
            if all(k in words for k in key):
                out.append(snap)
            continue
        url = (snap.get("url") or "").lower().replace("-", "_")
        if joined in url or all(k in re.findall(r"[a-z0-9]+", url) for k in key):
            out.append(snap)
    return out


def web_product_findings(products: Sequence[Dict[str, Any]], snapshots: Sequence[Dict[str, Any]], findings: List[Finding], metrics: Dict[str, Any]) -> None:
    unmatched: List[str] = []
    bad: List[Dict[str, Any]] = []
    checked = 0
    for p in products:
        matched = match_snapshots(p["name"], snapshots)
        if not matched:
            unmatched.append(p["name"])
            continue
        reference: Set[str] = set()
        for snap in matched:
            reference |= numeric_tokens(snap.get("text") or "")
        stored = p.get("table_numbers") or set()
        if not stored:
            continue
        checked += 1
        missing = sorted(stored - reference)
        share = len(missing) / len(stored)
        if share >= WEB_UNSUPPORTED_WARN:
            bad.append({"product": p["name"], "share": round(share, 3), "not_on_page": len(missing), "of": len(stored), "examples": missing[:8], "pages": [m.get("url") for m in matched][:3]})
    metrics["web_products_checked"] = checked
    metrics["web_products_without_page"] = len(unmatched)
    if unmatched:
        findings.append(Finding(WARN, "web_products_without_page", f"{len(unmatched)} product(s) have no page in the saved snapshots (the page may be gone from the site), so their values cannot be verified", len(unmatched), unmatched))
    hard = [b for b in bad if b["share"] >= WEB_UNSUPPORTED_FAIL]
    if hard:
        findings.append(Finding(FAIL, "web_numbers_not_on_page", f"{len(hard)} product(s) store numbers that their page does not print", len(hard), hard))
    soft = [b for b in bad if b not in hard]
    if soft:
        findings.append(Finding(WARN, "web_numbers_not_on_page_few", f"{len(soft)} product(s) store a few numbers their page does not print (derived values or a misread)", len(soft), soft))


def table_findings(tables: Sequence[Dict[str, Any]], findings: List[Finding], metrics: Dict[str, Any]) -> None:
    flagged = [t for t in tables if (t.get("metadata") or {}).get("needs_review")]
    fragments = [t for t in tables if t.get("extractor") == "paddleocr-vl" and not (t.get("metadata") or {}).get("title")]
    metrics["tables"] = len(tables)
    metrics["tables_needing_review"] = len(flagged)
    metrics["untitled_ocr_fragments"] = len(fragments)
    if flagged:
        findings.append(Finding(WARN, "tables_need_review", f"{len(flagged)} table(s) are flagged as possibly incomplete", len(flagged), [t.get("id") for t in flagged]))
    if fragments and len(fragments) > len(tables) - len(fragments):
        findings.append(Finding(WARN, "mostly_raw_fragments", f"{len(fragments)} of {len(tables)} stored tables are raw OCR fragments with no title", len(fragments), [t.get("id") for t in fragments]))


def chunk_findings(chunks: Sequence[Dict[str, Any]], tables: Sequence[Dict[str, Any]], findings: List[Finding], metrics: Dict[str, Any], phase: str = "post_embed") -> None:
    unembedded = [c for c in chunks if not c.get("has_text_embedding")] if phase == "post_embed" else []
    table_chunks = [c for c in chunks if (c.get("metadata") or {}).get("source") == "table_block"]
    titled = [t for t in tables if (t.get("metadata") or {}).get("title") or t.get("extractor") != "paddleocr-vl"]
    covered = {(c.get("metadata") or {}).get("table_id") for c in table_chunks}
    uncovered = [t.get("id") for t in titled if t.get("id") not in covered]
    leaks = [c.get("id") for c in table_chunks if PRICE_TEXT.search(c.get("content") or "")]
    metrics.update({"chunks": len(chunks), "table_chunks": len(table_chunks), "unembedded_chunks": len(unembedded), "tables_without_chunk": len(uncovered), "price_like_table_chunks": len(leaks)})
    if unembedded:
        findings.append(Finding(FAIL, "chunks_not_embedded", f"{len(unembedded)} chunk(s) have no embedding, so search cannot find them", len(unembedded), [c.get("id") for c in unembedded]))
    if uncovered:
        findings.append(Finding(FAIL, "tables_without_chunks", f"{len(uncovered)} stored table(s) have no searchable chunk", len(uncovered), uncovered))
    if leaks:
        findings.append(Finding(FAIL, "price_in_chunks", f"{len(leaks)} table chunk(s) contain price-like text", len(leaks), leaks))


def product_findings(products: Sequence[Dict[str, Any]], findings: List[Finding], metrics: Dict[str, Any], phase: str = "post_embed") -> None:
    def lacking(key: str) -> List[Any]:
        return [p.get("name") for p in products if not p.get(key)]

    metrics["products"] = len(products)
    for key, code, label in (("n_tables", "products_without_tables", "table"), ("n_chunks", "products_without_chunks", "chunk"), ("has_embedding", "products_without_embedding", "embedding"), ("n_images", "products_without_images", "image")):
        if phase == "pre_embed" and key == "has_embedding":
            continue
        missing = lacking(key)
        metrics[code] = len(missing)
        if missing:
            severity = FAIL if code in {"products_without_embedding"} else WARN
            findings.append(Finding(severity, code, f"{len(missing)} product(s) have no {label}", len(missing), missing))


MODEL_REFUSAL = re.compile(
    r"^(the )?(source|catalogue|catalog)( text)?( for \S+)? (contains|does not|has|provides|offers|consists)"
    r"|no (real |narrative )?description|there is no description|nothing to (write|describe)", re.I)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()


def product_content_findings(products: Sequence[Dict[str, Any]], findings: List[Finding], metrics: Dict[str, Any]) -> None:
    """A description that is the model saying it had nothing to write, and attributes naming another series."""
    refusals = [p.get("name") for p in products if MODEL_REFUSAL.search((p.get("description") or "").strip())]
    names = {_fold(p.get("name") or "") for p in products}
    leaks = []
    for p in products:
        own = _fold(p.get("name") or "")
        a = p.get("attributes") or {}
        attrs = _fold(json.dumps(a, ensure_ascii=False))
        # A series name that is also this product's own colour or decor ("Oxido", "Cotto") is not a leak.
        own_words = {_fold(v) for k in ("color", "finish", "decorative_variant")
                     for v in (a.get(k) if isinstance(a.get(k), list) else [a.get(k)]) if isinstance(v, str)}
        others = sorted(o for o in names if o != own and len(o) > 3 and o not in own_words and re.search(r"\b" + re.escape(o) + r"\b", attrs))
        if others:
            leaks.append({"product": p.get("name"), "names": others})
    metrics.update({"descriptions_model_refusal": len(refusals), "attributes_naming_other_series": len(leaks)})
    if refusals:
        findings.append(Finding(FAIL, "description_is_model_refusal", f"{len(refusals)} description(s) are the model explaining it had nothing to write", len(refusals), refusals))
    if leaks:
        findings.append(Finding(FAIL, "attributes_name_other_series", f"{len(leaks)} product(s) carry attribute values that name another series of this document", len(leaks), leaks))


def series_coverage_findings(coverage: Sequence[Dict[str, Any]], findings: List[Finding], metrics: Dict[str, Any]) -> None:
    """A variant the maker photographed that the product does not offer, and the series' own pages outside its range."""
    missing = [{"product": c["product"], "variants": c["missing_variants"]} for c in coverage if c.get("missing_variants")]
    outside = [{"product": c["product"], "pages": c["pages_outside"]} for c in coverage if c.get("pages_outside")]
    off_site = [c["product"] for c in coverage if c.get("site_status") == "not_found"]
    metrics.update({"products_missing_site_variants": len(missing), "products_with_own_pages_outside_range": len(outside),
                    "series_not_on_maker_site": len(off_site)})
    if off_site:
        findings.append(Finding(WARN, "series_not_on_maker_site", f"{len(off_site)} series were not found on the maker's site: their images and variants come from the PDF alone", len(off_site), off_site))
    if missing:
        findings.append(Finding(FAIL, "variants_missing_vs_sources", f"{len(missing)} product(s) lack a colour, decor or finish the maker's own photos name", len(missing), missing))
    if outside:
        findings.append(Finding(FAIL, "series_pages_outside_range", f"{len(outside)} product(s) have pages printing their name beside a size outside their page range", len(outside), outside))


def audit_document(
    *,
    source_kept: bool,
    pages: Sequence[Dict[str, Any]],
    tables: Sequence[Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]],
    products: Sequence[Dict[str, Any]],
    pdf_text: Optional[str],
    platform_text: str,
    phase: str = "post_embed",
    source_kind: str = "pdf",
    catalogue: Optional[Dict[str, Any]] = None,
    source_state: str = "kept",
    web: Optional[Dict[str, Any]] = None,
    product_images: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    other_series: Optional[Sequence[Dict[str, Any]]] = None,
    unsettled_images: Optional[Sequence[Dict[str, Any]]] = None,
    series_coverage: Optional[Sequence[Dict[str, Any]]] = None,
) -> AuditResult:
    findings: List[Finding] = []
    metrics: Dict[str, Any] = {"source_kept": source_kept, "phase": phase, "source_kind": source_kind}
    if not source_kept:
        if source_state == "missing_object":
            findings.append(Finding(FAIL, "source_missing_in_storage", "The source file is recorded for this document but is not in storage any more, so the content cannot be verified"))
        else:
            findings.append(Finding(FAIL, "source_not_kept", "The source file or page snapshot was not kept, so the content cannot be verified"))
    elif source_kind == "web":
        web_product_findings((web or {}).get("products") or [], (web or {}).get("snapshots") or [], findings, metrics)
    else:
        coverage_findings(pages, findings, metrics)
        code_findings(tables, pdf_text, platform_text, findings, metrics)
        unsupported_findings(tables, set().union(*(p.get("reference_numbers") or set() for p in pages)) if pages else set(), findings, metrics)
    table_findings(tables, findings, metrics)
    if catalogue:
        catalogue_findings(catalogue["plans"], catalogue["unmatched"], tables, findings, metrics)
    chunk_findings(chunks, tables, findings, metrics, phase)
    product_findings(products, findings, metrics, phase)
    product_content_findings(products, findings, metrics)
    series_coverage_findings(series_coverage or [], findings, metrics)
    if phase != "pre_embed":
        image_findings(product_images or {}, findings, metrics)
        image_integrity_findings(product_images or {}, other_series or [], findings, metrics)
        unsettled_image_findings(unsettled_images or [], findings, metrics)
    unrouted = sum(1 for p in pages if not p.get("route")) if source_kind == "pdf" else 0
    metrics["pages"] = len(pages)
    metrics["pages_never_checked_against_text_layer"] = unrouted
    if source_kept and source_kind == "pdf" and pages and unrouted == len(pages):
        findings.append(Finding(WARN, "layout_not_routed", "This document's layout predates the text-layer check; upgrade it to verify the pages", unrouted))
    severities = {f.severity for f in findings}
    if not source_kept:
        status = UNVERIFIABLE
    elif FAIL in severities:
        status = FAIL
    elif WARN in severities:
        status = WARN
    else:
        status = PASS
    return AuditResult(status, metrics, findings)


def _weak_image(img: Dict[str, Any]) -> bool:
    m = img.get("metadata") or {}
    return (img.get("extraction_layer") == "restored" and m.get("restore_method") == "classical")         or (m.get("image_quality") or {}).get("verdict") == "low_res"


def image_findings(product_images: Dict[str, List[Dict[str, Any]]], findings: List[Finding], metrics: Dict[str, Any]) -> None:
    """Images the products ended up with, keyed by product name: nothing embedded from a weak
    source, nothing left waiting, no product left with only weak images."""
    low_res_embedded, awaiting, weak_only = [], [], []
    for name, imgs in product_images.items():
        live = [i for i in imgs if "excluded_from_products" not in (i.get("metadata") or {})]
        for i in live:
            q = (i.get("metadata") or {}).get("image_quality") or {}
            # Any sourcing state means it was looked at (kept_source: no better version, repair not faithful).
            if (i.get("extraction_layer") != "restored" and q.get("verdict") == "low_res"
                    and i.get("has_slig_embedding") and not q.get("sourcing")):
                low_res_embedded.append(i["id"])
            if q.get("sourcing") == "pending":
                awaiting.append(i["id"])
        if live and not any(i.get("extraction_layer") == "manufacturer_original" for i in live)                 and all(_weak_image(i) for i in live):
            weak_only.append(name)
    metrics.update({"images_low_res_embedded": len(low_res_embedded), "images_awaiting_sourcing": len(awaiting),
                    "products_only_weak_images": len(weak_only)})
    if low_res_embedded:
        findings.append(Finding(FAIL, "images_low_res_embedded", f"{len(low_res_embedded)} low-resolution image(s) were embedded without a better version being looked for", len(low_res_embedded), low_res_embedded))
    if awaiting:
        findings.append(Finding(WARN, "images_awaiting_sourcing", f"{len(awaiting)} image(s) are still waiting for the manufacturer-site search or repair", len(awaiting), awaiting))
    if weak_only:
        findings.append(Finding(WARN, "products_only_weak_images", f"{len(weak_only)} product(s) have only low-resolution or smoothed images and no manufacturer image", len(weak_only), weak_only))


UPSCALE_LIMIT = 2.0


def _longest(px: Any) -> Optional[float]:
    try:
        return float(max(int(px[0]), int(px[1])))
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _upscale(img: Dict[str, Any]) -> Optional[float]:
    """Stored size over the real pixels it came from: a repair over its source, a crop over the raster."""
    m = img.get("metadata") or {}
    stored = _longest([m.get("width"), m.get("height")])
    if img.get("extraction_layer") == "restored":
        real = _longest(m.get("source_native_px") or img.get("source_px"))
    else:
        real = _longest((m.get("image_quality") or {}).get("native_px"))
    return stored / real if stored and real else None


def _position(img: Dict[str, Any]) -> Optional[tuple]:
    bbox = img.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return (img.get("document_id"), img.get("page_number"), img.get("extraction_layer"), *(round(float(v), 3) for v in bbox))


HASH_NEAR_BITS = 4


def _hash_twins(images: Sequence[Dict[str, Any]]) -> List[List[str]]:
    """One product's images from different PDFs whose perceptual hashes differ by HASH_NEAR_BITS or fewer bits."""
    hashed = [(i["id"], i.get("document_id"), int(h, 16)) for i in images
              for h in [((i.get("metadata") or {}).get("layer_info") or {}).get("perceptual_hash")] if h]
    pairs = []
    for n, (a, da, ha) in enumerate(hashed):
        for b, db_, hb in hashed[n + 1:]:
            if da != db_ and bin(ha ^ hb).count("1") <= HASH_NEAR_BITS:
                pairs.append([a, b])
    return pairs


def image_integrity_findings(product_images: Dict[str, List[Dict[str, Any]]], other_series: Sequence[Dict[str, Any]],
                             findings: List[Finding], metrics: Dict[str, Any]) -> None:
    """A repair enlarged past its source, one crop stored twice, an image on another series' page."""
    live = {i["id"]: i for imgs in product_images.values() for i in imgs
            if "excluded_from_products" not in (i.get("metadata") or {})}
    upscaled_repairs, upscaled_crops = [], []
    for i in live.values():
        ratio = _upscale(i)
        if ratio and ratio > UPSCALE_LIMIT:
            (upscaled_repairs if i.get("extraction_layer") == "restored" else upscaled_crops).append(i["id"])
    by_position: Dict[tuple, List[str]] = defaultdict(list)
    for i in live.values():
        key = _position(i)
        if key:
            by_position[key].append(i["id"])
    duplicates = [ids for ids in by_position.values() if len(ids) > 1]
    twins = [pair for imgs in product_images.values() for pair in _hash_twins([i for i in imgs if i["id"] in live])]
    metrics.update({"images_hash_twins_across_documents": len(twins)})
    if twins:
        findings.append(Finding(WARN, "images_hash_twins_across_documents", f"{len(twins)} pair(s) of images from different PDFs look identical (a hash cannot tell plain colours apart)", len(twins), twins))
    metrics.update({"images_upscaled_repairs": len(upscaled_repairs), "images_upscaled_crops": len(upscaled_crops),
                    "images_duplicated_position": sum(len(d) - 1 for d in duplicates),
                    "images_on_other_series_page": len(other_series)})
    if upscaled_repairs:
        findings.append(Finding(FAIL, "images_upscaled", f"{len(upscaled_repairs)} repaired image(s) are over {UPSCALE_LIMIT:g}x the size of the pixels they came from: enlarged, not repaired", len(upscaled_repairs), upscaled_repairs))
    if upscaled_crops:
        findings.append(Finding(WARN, "crops_rendered_above_source", f"{len(upscaled_crops)} crop(s) were rendered over {UPSCALE_LIMIT:g}x their source raster: a bigger file, no more detail", len(upscaled_crops), upscaled_crops))
    if duplicates:
        findings.append(Finding(FAIL, "duplicate_images_same_position", f"{sum(len(d) - 1 for d in duplicates)} extra image(s) were cut from the same spot on the same page", sum(len(d) - 1 for d in duplicates), duplicates))
    if other_series:
        findings.append(Finding(FAIL, "images_on_other_series_page", f"{len(other_series)} image(s) belong to a product but sit on a page that prints only other series' names", len(other_series), list(other_series)))


def owes_a_product(row: Dict[str, Any]) -> bool:
    """A live material image: not retired, not a spec icon, not decorative, not classified as non-material."""
    meta = row.get("metadata") or {}
    if "excluded_from_products" in meta or meta.get("bundle_skipped_reason") or row.get("image_type") == "DECORATIVE":
        return False
    return (meta.get("ai_classification") or {}).get("is_material") is not False


def unsettled_images(images: Sequence[Dict[str, Any]], linked_ids: Set[str]) -> List[Dict[str, Any]]:
    """Live material images holding no link."""
    return [{"image_id": i["id"], "page": i.get("page_number")} for i in images
            if i["id"] not in linked_ids and owes_a_product(i)]


def unsettled_image_findings(unsettled: Sequence[Dict[str, Any]], findings: List[Finding], metrics: Dict[str, Any]) -> None:
    """A live material image with no product and no exclusion: nothing decided what it is."""
    metrics["images_unsettled"] = len(unsettled)
    if unsettled:
        findings.append(Finding(FAIL, "images_unsettled", f"{len(unsettled)} material image(s) are linked to no product and were never excluded", len(unsettled), list(unsettled)))


def platform_numbers_for_page(layout_text: str, chunk_texts: Iterable[str], table_texts: Iterable[str]) -> Set[str]:
    out = set(numeric_tokens(layout_text or ""))
    for text in chunk_texts:
        out |= numeric_tokens(text or "")
    for text in table_texts:
        out |= numeric_tokens(text or "")
    return out


DB_STATUS = {PASS: "passed", WARN: "needs_review", FAIL: "failed", UNVERIFIABLE: "unverifiable"}


def db_status(status: str) -> str:
    return DB_STATUS.get(status, "audit_error")
