"""
Type aliases for page-counting in the PDF processing pipeline.

Why this file exists
--------------------
On 2026-05-01 we shipped a fix for a bug that silently dropped 50% of
products in spread-layout catalogs. Root cause was a parameter named
`total_pages` whose meaning differed between callers:

  - Some callers passed `len(fitz_doc)` — the PDF SHEET count
    (e.g. 71 for a spread-layout catalog).
  - Other callers passed `catalog.total_pages` — the PHYSICAL page
    count (e.g. 140 for the same catalog).

Both are `int` so the type system couldn't catch the mismatch. The
function that received it used the value as a physical-page upper bound,
which silently dropped any page > 71.

These aliases give the two distinct meanings distinct names that mypy
(or any structural type-checker that respects NewType) can flag when
someone passes the wrong one. At runtime they're plain ints — there's
no overhead.

When to use:
  * `PhysicalPage` — a physical page number (1-based) as printed in
    the catalog. For spread-layout PDFs, two physical pages per PDF
    sheet. The full document spans 1..catalog.total_pages.
  * `PdfSheetIndex` — a 0-based PDF sheet index (the value PyMuPDF
    uses for `doc[i]`). The full document spans 0..len(fitz_doc)-1.
  * `PhysicalPageBound` — a physical-page upper bound used for
    validation. Equivalent to PhysicalPage in type, but the name
    documents intent at function signatures.

The helpers `as_physical_page()` and `as_pdf_sheet_index()` exist for
the rare case where you genuinely need to convert a plain int into the
typed form (e.g. when constructing a page number from arithmetic on
intermediate values).
"""
from __future__ import annotations

from typing import NewType


# 1-based physical page number — what the user sees printed in the catalog.
PhysicalPage = NewType("PhysicalPage", int)

# 0-based PDF sheet index — what PyMuPDF and other PDF libs use to index
# into the document. NEVER pass this where a `PhysicalPage` is expected
# (especially for spread-layout PDFs, where the relationship is 1:2).
PdfSheetIndex = NewType("PdfSheetIndex", int)

# Inclusive upper bound for physical-page validation. Always equal to
# `catalog.total_pages` (the largest physical page number that exists).
# Distinct from a count-of-pages because the name documents the role.
PhysicalPageBound = NewType("PhysicalPageBound", int)


def as_physical_page(value: int) -> PhysicalPage:
    """Wrap a plain int as a PhysicalPage. Use sparingly — prefer
    propagating typed values through your call chain."""
    if value < 1:
        raise ValueError(
            f"PhysicalPage values are 1-based; got {value}. "
            f"Did you mean PdfSheetIndex (0-based)?"
        )
    return PhysicalPage(value)


def as_pdf_sheet_index(value: int) -> PdfSheetIndex:
    """Wrap a plain int as a PdfSheetIndex. Use sparingly."""
    if value < 0:
        raise ValueError(f"PdfSheetIndex values are 0-based; got {value}")
    return PdfSheetIndex(value)


def as_physical_page_bound(value: int) -> PhysicalPageBound:
    """Wrap a plain int as a PhysicalPageBound. The bound must be ≥1
    because a PDF with 0 pages can't have any physical pages either."""
    if value < 1:
        raise ValueError(
            f"PhysicalPageBound must be ≥1; got {value}. "
            f"For empty documents, don't call validation at all."
        )
    return PhysicalPageBound(value)


__all__ = [
    "PhysicalPage",
    "PdfSheetIndex",
    "PhysicalPageBound",
    "as_physical_page",
    "as_pdf_sheet_index",
    "as_physical_page_bound",
]
