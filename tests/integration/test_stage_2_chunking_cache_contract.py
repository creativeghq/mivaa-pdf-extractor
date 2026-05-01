"""
Stage 2 chunker layout-cache contract test.

Why this file exists
--------------------
Stage 2 chunking has its own reader for `document_layout_analysis` —
`get_layout_from_document_cache_with_status` in
`stage_1_layout_precompute.py`. Unlike `pdf_processor._load_cached_layout_for_pages`
(which gates on `processing_version`), this reader gates on
`analysis_metadata.cache_status`:

  - `cache_status == 'success'`       → use layout-aware chunking
  - `cache_status == 'ocr_failed'`    → fall back to text-based chunking
                                         (Stage 1.5 will retry)
  - `cache_status == 'empty_page'`    → no regions, treat as empty
  - missing field OR no row           → cache miss, fall back

If a future refactor renames `cache_status`, moves it out of
`analysis_metadata`, or changes one of the magic-string values, the
chunker will silently fall back to text-only chunking on every page.
The visible symptom is "chunks are smaller / less coherent than usual"
— hard to detect at the catalog level, harder to diagnose. This test
pins the contract so the silent-degradation class can't recur.

The test deliberately exercises the actual reader (no mocking inside it)
against an in-memory fake supabase, mirroring the Stage 1.5 contract test.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# -- Contract constants ----------------------------------------------------

# Cache-status values the chunker's reader recognizes. If you add or rename
# any, update both the writer (stage_1_layout_precompute.py) and this test.
CACHE_STATUS_SUCCESS = "success"
CACHE_STATUS_OCR_FAILED = "ocr_failed"
CACHE_STATUS_EMPTY_PAGE = "empty_page"
CACHE_STATUS_PAGE_FAILED = "page_failed"
CACHE_STATUS_YOLO_ONLY = "yolo_only"


# -- Helpers ---------------------------------------------------------------


def _build_chunker_payload(
    document_id: str,
    page_number: int,
    layout_elements: List[Dict[str, Any]],
    cache_status: str = CACHE_STATUS_SUCCESS,
) -> Dict[str, Any]:
    """Mirror what Stage 1.5 writes — the bits the chunker reads.

    The chunker only cares about `layout_elements` and
    `analysis_metadata.cache_status`. It does NOT check `processing_version`
    (that's the image-extraction reader's gate).
    """
    return {
        "document_id": document_id,
        "page_number": page_number,
        "layout_elements": layout_elements,
        "reading_order": [],
        "structure_confidence": 0.85,
        "processing_version": "yolo+chandra-v2",
        "analysis_metadata": {
            "stage_1_5": True,
            "cache_status": cache_status,
            "extraction_path": "yolo+text",
            "region_count": len(layout_elements),
        },
    }


def _sample_region(region_type: str = "TEXT", text: str = "Hello world") -> Dict[str, Any]:
    """Layout-element dict shape matching MergedRegion.to_dict()."""
    return {
        "region_type": region_type,
        "bbox": {"x": 10.0, "y": 20.0, "width": 100.0, "height": 80.0},
        "confidence": 0.92,
        "reading_order": 0,
        "text_content": text,
        "fragments": [],
        "metadata": {},
    }


# -- In-memory fake supabase (same shape as the Stage 1.5 test's) ---------


class _FakeQuery:
    def __init__(self, store: "_FakeStore", op: str = "select"):
        self._store = store
        self._op = op
        self._payload: Dict[str, Any] | None = None
        self._filters: Dict[str, Any] = {}
        self._select_columns: str | None = None

    def upsert(self, payload, on_conflict: str | None = None):
        self._op = "upsert"
        self._payload = payload
        return self

    def select(self, columns: str):
        self._op = "select"
        self._select_columns = columns
        return self

    def eq(self, col: str, val: Any):
        self._filters[col] = ("eq", val)
        return self

    def in_(self, col: str, vals: List[Any]):
        self._filters[col] = ("in", vals)
        return self

    def execute(self):
        if self._op == "upsert":
            assert self._payload is not None
            key = (self._payload["document_id"], self._payload["page_number"])
            self._store.rows[key] = dict(self._payload)
            return MagicMock(data=[self._payload])

        out = []
        for (doc_id, pn), row in self._store.rows.items():
            ok = True
            for col, (op, val) in self._filters.items():
                row_val = doc_id if col == "document_id" else (pn if col == "page_number" else row.get(col))
                if op == "eq" and row_val != val:
                    ok = False
                    break
                if op == "in" and row_val not in val:
                    ok = False
                    break
            if ok:
                out.append(dict(row))
        return MagicMock(data=out)


class _FakeStore:
    def __init__(self):
        self.rows: Dict[tuple, Dict[str, Any]] = {}


class _FakeSupabaseInner:
    def __init__(self, store: _FakeStore):
        self._store = store

    def table(self, name: str):
        assert name == "document_layout_analysis"
        return _FakeQuery(self._store)


class _FakeSupabaseClient:
    def __init__(self):
        self._store = _FakeStore()
        self.client = _FakeSupabaseInner(self._store)


@pytest.fixture
def fake_supabase():
    return _FakeSupabaseClient()


# -- Tests ----------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunker_cache_returns_regions_when_status_success(fake_supabase):
    """`cache_status == 'success'` row should round-trip into the chunker."""
    from app.api.pdf_processing.stage_1_layout_precompute import (
        get_layout_from_document_cache,
    )

    document_id = "doc-stage2-1"
    physical_page = 1

    payload = _build_chunker_payload(
        document_id,
        physical_page,
        [_sample_region("TEXT", "title text"), _sample_region("IMAGE", "")],
        cache_status=CACHE_STATUS_SUCCESS,
    )
    fake_supabase.client.table("document_layout_analysis").upsert(payload).execute()

    cached = await get_layout_from_document_cache(
        document_id=document_id,
        physical_pages=[physical_page],
        supabase=fake_supabase,
        logger=logging.getLogger("test"),
    )

    assert physical_page in cached, (
        "Chunker cache reader returned empty for a 'success' row. "
        "If the gate logic changed, update the constant in this test."
    )
    assert len(cached[physical_page]) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunker_cache_skips_empty_pages(fake_supabase):
    """`cache_status='empty_page'` rows must NOT surface to the chunker.

    Empty-page rows record "we ran Stage 1.5 here and the page is genuinely
    empty"; the chunker should treat that as a cache miss so it falls back
    to text-based chunking instead of producing 0 chunks.
    """
    from app.api.pdf_processing.stage_1_layout_precompute import (
        get_layout_from_document_cache,
    )

    document_id = "doc-stage2-empty"
    physical_page = 1

    payload = _build_chunker_payload(
        document_id,
        physical_page,
        [],  # no regions
        cache_status=CACHE_STATUS_EMPTY_PAGE,
    )
    fake_supabase.client.table("document_layout_analysis").upsert(payload).execute()

    cached = await get_layout_from_document_cache(
        document_id=document_id,
        physical_pages=[physical_page],
        supabase=fake_supabase,
        logger=logging.getLogger("test"),
    )

    # The 'thin wrapper' get_layout_from_document_cache filters out pages
    # with empty regions list, so an empty_page row shouldn't appear.
    assert physical_page not in cached, (
        f"Chunker received an empty regions list for page {physical_page}. "
        f"The chunker should not see empty-page rows — it should fall back "
        f"to text-based chunking instead."
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunker_cache_status_with_status_endpoint(fake_supabase):
    """The `_with_status` variant must return ALL rows including failed ones.

    Stage 1.5 needs to know which pages had `ocr_failed` so it can retry
    them on the next run. If this contract drifts (e.g. status field gets
    renamed), retries silently stop happening.
    """
    from app.api.pdf_processing.stage_1_layout_precompute import (
        get_layout_from_document_cache_with_status,
    )

    document_id = "doc-stage2-statuses"

    fake_supabase.client.table("document_layout_analysis").upsert(
        _build_chunker_payload(document_id, 1, [_sample_region()], CACHE_STATUS_SUCCESS)
    ).execute()
    fake_supabase.client.table("document_layout_analysis").upsert(
        _build_chunker_payload(document_id, 2, [], CACHE_STATUS_OCR_FAILED)
    ).execute()
    fake_supabase.client.table("document_layout_analysis").upsert(
        _build_chunker_payload(document_id, 3, [], CACHE_STATUS_PAGE_FAILED)
    ).execute()

    cached = await get_layout_from_document_cache_with_status(
        document_id=document_id,
        physical_pages=[1, 2, 3],
        supabase=fake_supabase,
        logger=logging.getLogger("test"),
    )

    # All three rows should be present in the with_status result, with
    # their respective cache_status values intact.
    assert set(cached.keys()) == {1, 2, 3}, (
        "_with_status reader must return ALL rows (success + failed). "
        f"Got pages: {sorted(cached.keys())}"
    )

    statuses = {pn: payload.get("cache_status") for pn, payload in cached.items()}
    assert statuses[1] == CACHE_STATUS_SUCCESS, (
        f"Page 1 should report cache_status='{CACHE_STATUS_SUCCESS}'; "
        f"got {statuses[1]!r}"
    )
    assert statuses[2] == CACHE_STATUS_OCR_FAILED, (
        f"Page 2 should report cache_status='{CACHE_STATUS_OCR_FAILED}'; "
        f"got {statuses[2]!r}. If this fails, ocr-failed rows won't get "
        f"retried by Stage 1.5 — silent degradation forever."
    )
    assert statuses[3] == CACHE_STATUS_PAGE_FAILED, (
        f"Page 3 should report cache_status='{CACHE_STATUS_PAGE_FAILED}'; "
        f"got {statuses[3]!r}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunker_cache_handles_legacy_rows_without_cache_status(fake_supabase):
    """Pre-2026-05-01 rows lack `cache_status`. They must NOT crash the reader."""
    from app.api.pdf_processing.stage_1_layout_precompute import (
        get_layout_from_document_cache_with_status,
    )

    document_id = "doc-stage2-legacy"
    payload = _build_chunker_payload(document_id, 1, [_sample_region()])
    # Strip cache_status to simulate a row written before the migration
    del payload["analysis_metadata"]["cache_status"]
    fake_supabase.client.table("document_layout_analysis").upsert(payload).execute()

    cached = await get_layout_from_document_cache_with_status(
        document_id=document_id,
        physical_pages=[1],
        supabase=fake_supabase,
        logger=logging.getLogger("test"),
    )

    # Legacy row must still be readable; the contract is that
    # cache_status=None on a legacy row, but the regions still load.
    assert 1 in cached, "Legacy rows without cache_status must still be readable"
    payload_data = cached[1]
    assert payload_data["regions"], (
        "Legacy row regions must surface to the chunker; otherwise "
        "every pre-migration document silently falls back to text chunking."
    )
