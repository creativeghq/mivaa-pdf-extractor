from __future__ import annotations

import re
from collections import Counter
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
            gaps.append({"page": page["page"], "missing_share": round(share, 3), "missing": len(missing), "of": len(reference), "route": page.get("route")})
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
    hard = [g for g in gaps if g["missing_share"] >= PAGE_MISSING_FAIL]
    soft = [g for g in gaps if g["missing_share"] < PAGE_MISSING_FAIL]
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
) -> AuditResult:
    findings: List[Finding] = []
    metrics: Dict[str, Any] = {"source_kept": source_kept, "phase": phase}
    if not source_kept:
        findings.append(Finding(FAIL, "source_not_kept", "The source file or page snapshot was not kept, so the content cannot be verified"))
    else:
        coverage_findings(pages, findings, metrics)
        code_findings(tables, pdf_text, platform_text, findings, metrics)
    table_findings(tables, findings, metrics)
    chunk_findings(chunks, tables, findings, metrics, phase)
    product_findings(products, findings, metrics, phase)
    unrouted = sum(1 for p in pages if not p.get("route"))
    metrics["pages"] = len(pages)
    metrics["pages_never_checked_against_text_layer"] = unrouted
    if source_kept and pages and unrouted == len(pages):
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
