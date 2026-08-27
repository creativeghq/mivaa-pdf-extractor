"""One bounds check for every place this service renders or buffers a PDF.

WHY IT IS SHARED
----------------
The same gap has now been found three times, in three files, by three separate audits:
#22 M9-6 (`pdf_processor`), #24 M11-6 (`catalog_routes` rasterisation) and #35 M20-2
(the ingestion orchestrator). Each time the fix was a local constant and a local check.
Three copies of a rule is how one of them drifts — the credit-debit helper had SEVEN
before it was unified (#30 M16-1) — so this is the rule, once.

WHAT THE COMPRESSED SIZE DOES NOT TELL YOU
------------------------------------------
Every one of those sites had an upload size cap in front of it and was still unbounded.
A small compressed PDF can declare enormous page dimensions or thousands of pages: the
render size is `page_rect * dpi/72` and the allocation happens inside `get_pixmap`, long
after the bytes have been accepted. **The check has to run before the allocation, not
before the download.**

RAISES ValueError, NOT HTTPException
------------------------------------
The three callers are a route, a service and a background orchestrator, and only one of
them can return a status code. A shared helper that raises `HTTPException` would force
the other two to catch a web exception to do bookkeeping. Each caller translates.
"""

from __future__ import annotations

from typing import Any

#: Past any real page at 400 DPI. An A0 sheet at 400 DPI is ~13k x 18k px.
MAX_RASTER_EDGE = 20_000
MAX_RASTER_PIXELS = 40_000_000

#: A catalogue is large; a decompression bomb is absurd. This separates them.
MAX_PDF_PAGES = 5_000


class PdfBoundsError(ValueError):
    """A PDF (or one of its pages) is too large to process safely."""


def assert_page_count(page_count: int, *, limit: int = MAX_PDF_PAGES) -> None:
    """Reject a document with an implausible number of pages, before any per-page work.

    Cheap and first: every downstream stage is per-page, so this bounds the whole
    pipeline rather than one allocation.
    """
    if page_count is None or page_count <= 0:
        raise PdfBoundsError("PDF reports no pages")
    if page_count > limit:
        raise PdfBoundsError(f"PDF has {page_count} pages; the maximum is {limit}")


def assert_renderable(page: Any, zoom: float) -> None:
    """Reject before `get_pixmap` allocates. The ORDER is the whole fix.

    `zoom` is `dpi / 72`. `page.rect` is in points, so `rect * zoom` is the pixel size
    the renderer is about to allocate — which is the number the compressed byte count
    cannot predict.
    """
    rect = getattr(page, "rect", None)
    if rect is None:
        raise PdfBoundsError("page has no geometry")

    width = int(rect.width * zoom)
    height = int(rect.height * zoom)
    if width <= 0 or height <= 0:
        raise PdfBoundsError("page has no renderable area")
    if width > MAX_RASTER_EDGE or height > MAX_RASTER_EDGE:
        raise PdfBoundsError(
            f"render would be {width}x{height}px; the maximum edge is {MAX_RASTER_EDGE}px"
        )
    if width * height > MAX_RASTER_PIXELS:
        raise PdfBoundsError(
            f"render would be {width * height} pixels; the maximum is {MAX_RASTER_PIXELS}"
        )
