"""
product_layout_regions writer <-> reader contract test.

Why this file exists
--------------------
The 2026-05-01 audit flagged `product_layout_regions` as a MEDIUM-severity
contract drift candidate: it has a single writer (Stage 1's
`_store_layout_regions`) and a single reader (Stage 2's
`get_layout_regions`), neither of which uses a version marker. If a future
refactor changes the bbox shape, the field set, or the row keys, the
reader will silently return mis-shaped data and layout-aware chunking
will degrade without any error.

This test pins the contract: the field set the writer emits must match
what the reader expects, and a round-trip must preserve every field the
chunker downstream relies on (bbox, region_type, reading_order,
text_content, page_number, product_id).

If you change either side of the contract, this test will fail and you
must update both at once.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# -- Contract: fields the writer must emit AND the reader must surface ----

REQUIRED_WRITER_FIELDS = {
    "product_id",
    "page_number",
    "region_type",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "confidence",
    "reading_order",
    "text_content",
    "metadata",
}

# Fields the chunker relies on after read. Subset of writer fields plus
# an `id` synthesized by the DB on insert.
REQUIRED_READER_FIELDS = REQUIRED_WRITER_FIELDS | {"id"}


# -- Helpers ---------------------------------------------------------------


def _build_writer_row(
    product_id: str,
    page_number: int,
    region_type: str = "TEXT",
    text_content: str = "Hello",
    reading_order: int = 0,
) -> Dict[str, Any]:
    """Mirror what `_store_layout_regions` (stage_1_focused_extraction.py)
    inserts. If you change the writer payload, also change this helper —
    that's the whole reason this test exists."""
    return {
        "product_id": product_id,
        "page_number": page_number,
        "region_type": region_type,
        "bbox_x": 0.10,
        "bbox_y": 0.20,
        "bbox_width": 0.50,
        "bbox_height": 0.30,
        "confidence": 0.92,
        "reading_order": reading_order,
        "text_content": text_content,
        "metadata": {
            "yolo_model": "yolo-docparser",
            "extraction_method": "yolo_guided",
        },
    }


# -- Fake supabase --------------------------------------------------------


class _FakeQuery:
    def __init__(self, store: "_FakeStore"):
        self._store = store
        self._op: str = "select"
        self._payload: Any = None
        self._filters: Dict[str, Any] = {}
        self._orders: List[str] = []

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def select(self, columns: str = "*"):
        self._op = "select"
        return self

    def eq(self, col: str, val: Any):
        self._filters[col] = val
        return self

    def order(self, col: str):
        self._orders.append(col)
        return self

    def execute(self):
        if self._op == "insert":
            rows = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted = []
            for r in rows:
                row = dict(r)
                row["id"] = f"region-{len(self._store.rows)+1}"
                self._store.rows.append(row)
                inserted.append(row)
            return MagicMock(data=inserted)

        # select
        out = list(self._store.rows)
        for col, val in self._filters.items():
            out = [r for r in out if r.get(col) == val]
        for col in self._orders:
            out.sort(key=lambda r: (r.get(col) is None, r.get(col)))
        return MagicMock(data=out)


class _FakeStore:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []


class _FakeSupabaseInner:
    def __init__(self, store: _FakeStore):
        self._store = store

    def table(self, name: str):
        assert name == "product_layout_regions", (
            f"Test fake only supports 'product_layout_regions', got {name!r}"
        )
        return _FakeQuery(self._store)


class _FakeSupabaseClient:
    def __init__(self):
        self._store = _FakeStore()
        self.client = _FakeSupabaseInner(self._store)


@pytest.fixture
def fake_supabase():
    return _FakeSupabaseClient()


# -- Tests ----------------------------------------------------------------


@pytest.mark.unit
def test_writer_payload_has_all_fields_reader_expects():
    """Cheapest contract check: writer emits every field the reader
    needs (excluding the DB-synthesized `id`)."""
    payload = _build_writer_row(product_id="prod-1", page_number=1)
    missing = REQUIRED_WRITER_FIELDS - set(payload.keys())
    assert not missing, (
        f"_store_layout_regions payload (mirrored by _build_writer_row) "
        f"is missing required fields: {sorted(missing)}. The reader at "
        f"stage_2_chunking.py:get_layout_regions returns whatever rows "
        f"the writer inserted; missing fields surface to the chunker as "
        f"None and silently degrade layout-aware chunking."
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_round_trip_writer_to_reader_preserves_fields(fake_supabase):
    """A row written by the writer must be readable by `get_layout_regions`
    and surface every field intact, sorted by reading_order per page."""
    from app.api.pdf_processing.stage_2_chunking import get_layout_regions

    product_id = "prod-roundtrip"
    rows = [
        _build_writer_row(product_id, page_number=1, region_type="TEXT", reading_order=0, text_content="alpha"),
        _build_writer_row(product_id, page_number=1, region_type="IMAGE", reading_order=1, text_content=""),
        _build_writer_row(product_id, page_number=2, region_type="TITLE", reading_order=0, text_content="page 2"),
    ]
    fake_supabase.client.table("product_layout_regions").insert(rows).execute()

    result = await get_layout_regions(
        product_id=product_id,
        supabase=fake_supabase,
        logger=logging.getLogger("test"),
    )

    # Result shape: {page_number: [region, region]}
    assert set(result.keys()) == {1, 2}, (
        f"Reader should return rows grouped by page; got pages "
        f"{sorted(result.keys())}"
    )

    # Page 1 should have 2 regions, in reading_order
    page_1 = result[1]
    assert len(page_1) == 2
    assert page_1[0]["region_type"] == "TEXT"
    assert page_1[0]["reading_order"] == 0
    assert page_1[1]["region_type"] == "IMAGE"
    assert page_1[1]["reading_order"] == 1

    # Every required field must survive the round-trip on every row.
    for page_num, page_rows in result.items():
        for row in page_rows:
            missing = REQUIRED_READER_FIELDS - set(row.keys())
            assert not missing, (
                f"Page {page_num} region missing fields after read: "
                f"{sorted(missing)}. Available: {sorted(row.keys())}. "
                f"The chunker reads these to position regions on rendered "
                f"pages; missing fields silently degrade chunking."
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reader_returns_empty_dict_for_unknown_product(fake_supabase):
    """Unknown product_id should produce {} — NOT an exception, NOT None.

    The chunker treats {} as 'no cached regions, fall back to text-based
    chunking'. If the reader started raising on missing rows, every job
    with no precomputed regions would fail instead of falling back.
    """
    from app.api.pdf_processing.stage_2_chunking import get_layout_regions

    result = await get_layout_regions(
        product_id="nonexistent-product",
        supabase=fake_supabase,
        logger=logging.getLogger("test"),
    )

    assert result == {}, (
        f"Reader should return {{}} for missing product, not {result!r}. "
        f"The chunker's fallback logic depends on this."
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reader_groups_by_page_number(fake_supabase):
    """The reader's contract is `{page_number: [regions]}`. Verify rows
    from different pages don't bleed into each other."""
    from app.api.pdf_processing.stage_2_chunking import get_layout_regions

    product_id = "prod-multi-page"
    fake_supabase.client.table("product_layout_regions").insert([
        _build_writer_row(product_id, page_number=10, region_type="TEXT"),
        _build_writer_row(product_id, page_number=15, region_type="IMAGE"),
        _build_writer_row(product_id, page_number=10, region_type="TITLE", reading_order=2),
    ]).execute()

    result = await get_layout_regions(
        product_id=product_id,
        supabase=fake_supabase,
        logger=logging.getLogger("test"),
    )

    assert sorted(result.keys()) == [10, 15]
    assert len(result[10]) == 2, "Page 10 should have 2 regions"
    assert len(result[15]) == 1, "Page 15 should have 1 region"

    # No region from page 10 should appear in page 15's list
    page_15_types = {r["region_type"] for r in result[15]}
    assert page_15_types == {"IMAGE"}
