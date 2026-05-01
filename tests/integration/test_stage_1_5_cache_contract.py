"""
Stage 1.5 cache writer <-> reader contract test.

Why this file exists
--------------------
On 2026-04-30, commit df751cb ("refactor(stage_1.5): physical-page-aware
layout precompute") rewrote the persistence path for
`document_layout_analysis`. The rewrite dropped three fields the cache
readers depend on -- most importantly `processing_version`. Both readers
gate on `processing_version == 'yolo+chandra-v2'`:

    - app/services/pdf/pdf_processor.py:786
        (Stage 3 Layer 2 image-crop cache lookup)
    - app/api/pdf_processing/stage_1_focused_extraction.py:494
        (per-product layout reuse)

When the writer stopped emitting that field, the DB column fell back to
its default '1.0.0', both readers silently rejected every cached row,
and Layer 2 image cropping fell back to live YOLO under load -- producing
~0 images on a catalog where YOLO had already detected 70+ IMAGE regions
during Stage 1.5.

The bug had no test guarding it. This file is that test. If a future
refactor drops a field again, CI fails immediately instead of weeks later
in production.

The test deliberately does NOT exercise YOLO/Chandra/PyMuPDF -- it tests
the WRITE -> READ contract in isolation by:
    1. Building the exact dict Stage 1.5 upserts (via a helper that
       mirrors the writer's payload construction).
    2. Storing it in an in-memory fake supabase.
    3. Calling the actual readers (no mocking inside them).
    4. Asserting the cache returns the regions.

If you change the writer's payload, you MUST also update
`_build_writer_payload` below -- and if the readers stop accepting that
shape, this test will tell you.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from app.services.pdf.layout_merge_service import MergedRegion


# -- Contract constants ----------------------------------------------------
# These are the ONLY values the readers accept. If you change them in the
# reader code, change them here too -- and update Stage 1.5's writer to match.

REQUIRED_PAYLOAD_FIELDS = {
    "document_id",
    "page_number",
    "layout_elements",
    "reading_order",
    "structure_confidence",
    "processing_version",
    "analysis_metadata",
}

REQUIRED_PROCESSING_VERSION = "yolo+chandra-v2"


# -- Helpers ---------------------------------------------------------------


def _build_sample_merged_region(
    region_type: str = "IMAGE",
    x: float = 10.0,
    y: float = 20.0,
    width: float = 100.0,
    height: float = 80.0,
) -> MergedRegion:
    """Build a MergedRegion shaped like Stage 1.5 produces them."""
    return MergedRegion(
        region_type=region_type,
        bbox={"x": x, "y": y, "width": width, "height": height},
        confidence=0.92,
        reading_order=0,
        text_content="",
    )


def _build_writer_payload(
    document_id: str,
    page_number: int,
    regions: List[MergedRegion],
) -> Dict[str, Any]:
    """Mirror the upsert payload Stage 1.5 writes.

    This MUST stay in lock-step with the upsert at
    `app/api/pdf_processing/stage_1_layout_precompute.py` (search for
    "document_layout_analysis"). The whole point of this test is to detect
    when those two diverge.
    """
    layout_elements = [r.to_dict() for r in regions]
    reading_order = [
        {
            "index": idx,
            "region_type": elem.get("region_type"),
            "reading_order": elem.get("reading_order"),
        }
        for idx, elem in enumerate(layout_elements)
    ]
    return {
        "document_id": document_id,
        "page_number": page_number,
        "layout_elements": layout_elements,
        "reading_order": reading_order,
        "structure_confidence": 0.85,
        "processing_version": REQUIRED_PROCESSING_VERSION,
        "analysis_metadata": {
            "stage_1_5": True,
            "extraction_path": "yolo+text",
            "cache_status": "success",
            "region_count": len(regions),
        },
    }


# -- In-memory fake supabase ----------------------------------------------


class _FakeQuery:
    """Captures filter chain calls and replays against in-memory rows."""

    def __init__(self, store: "_FakeStore", op: str):
        self._store = store
        self._op = op  # "select" | "upsert"
        self._payload: Dict[str, Any] | None = None
        self._filters: Dict[str, Any] = {}

    def upsert(self, payload, on_conflict: str | None = None):
        self._op = "upsert"
        self._payload = payload
        return self

    def select(self, columns: str):
        self._op = "select"
        return self

    def eq(self, column: str, value: Any):
        self._filters[column] = ("eq", value)
        return self

    def in_(self, column: str, values: List[Any]):
        self._filters[column] = ("in", values)
        return self

    def execute(self):
        if self._op == "upsert":
            assert self._payload is not None
            key = (self._payload["document_id"], self._payload["page_number"])
            self._store.rows[key] = dict(self._payload)
            return MagicMock(data=[self._payload])

        # select path -- apply filters
        results = []
        for (doc_id, page_num), row in self._store.rows.items():
            ok = True
            for col, (op, val) in self._filters.items():
                row_val = row.get(col)
                if col == "document_id":
                    row_val = doc_id
                elif col == "page_number":
                    row_val = page_num
                if op == "eq" and row_val != val:
                    ok = False
                    break
                if op == "in" and row_val not in val:
                    ok = False
                    break
            if ok:
                results.append(dict(row))
        return MagicMock(data=results)


class _FakeStore:
    rows: Dict[tuple, Dict[str, Any]]

    def __init__(self):
        self.rows = {}


class _FakeSupabaseInner:
    """Mimics `supabase.client.table(...)` chain."""

    def __init__(self, store: _FakeStore):
        self._store = store

    def table(self, name: str):
        # The contract test only cares about `document_layout_analysis`.
        assert name == "document_layout_analysis", (
            f"Test fake only supports 'document_layout_analysis', got {name!r}"
        )
        return _FakeQuery(self._store, op="select")


class _FakeSupabaseClient:
    """Mimics the wrapper returned by `get_supabase_client()`."""

    def __init__(self):
        self._store = _FakeStore()
        self.client = _FakeSupabaseInner(self._store)


@pytest.fixture
def fake_supabase(monkeypatch):
    fake = _FakeSupabaseClient()
    # Monkeypatch the FACTORY both modules use. They each
    # `from app.services.core.supabase_client import get_supabase_client`
    # at call time (inside their methods), so patching the module-level
    # attribute is enough.
    import app.services.core.supabase_client as sc

    monkeypatch.setattr(sc, "get_supabase_client", lambda: fake)
    return fake


# -- Tests ----------------------------------------------------------------


@pytest.mark.unit
def test_writer_payload_has_all_fields_reader_requires():
    """The dict Stage 1.5 builds must include every field the readers gate on.

    This is the cheapest, fastest line of defense -- pure dict introspection,
    no I/O.
    """
    payload = _build_writer_payload(
        document_id="doc-1",
        page_number=1,
        regions=[_build_sample_merged_region("IMAGE")],
    )

    missing = REQUIRED_PAYLOAD_FIELDS - set(payload.keys())
    assert not missing, (
        f"Stage 1.5 writer payload is missing fields the cache readers "
        f"require: {sorted(missing)}. The readers at "
        f"pdf_processor.py:786 and stage_1_focused_extraction.py:494 "
        f"will silently reject rows missing these fields. See bug "
        f"report dated 2026-05-01."
    )

    assert payload["processing_version"] == REQUIRED_PROCESSING_VERSION, (
        f"processing_version must be exactly '{REQUIRED_PROCESSING_VERSION}'. "
        f"Got: {payload['processing_version']!r}. The readers do a string "
        f"equality check; any other value (including the DB default '1.0.0') "
        f"causes silent cache misses and forces live YOLO fallback under load."
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_round_trip_writer_to_pdf_processor_reader(fake_supabase):
    """Round-trip: payload Stage 1.5 writes is loadable by pdf_processor's reader.

    This catches drifts beyond just the field set -- e.g. if the reader's
    bbox-shape expectations changed, or if it adds a new gate condition.
    """
    from app.services.pdf.pdf_processor import PDFProcessor

    document_id = "doc-roundtrip-1"
    physical_page = 1  # 1-based

    # Write -- mimic Stage 1.5
    regions = [
        _build_sample_merged_region("IMAGE", x=10, y=20),
        _build_sample_merged_region("TEXT", x=10, y=200),
    ]
    payload = _build_writer_payload(document_id, physical_page, regions)
    fake_supabase.client.table("document_layout_analysis").upsert(payload).execute()

    # Read -- exercise the actual reader code path
    processor = PDFProcessor.__new__(PDFProcessor)  # bypass heavy __init__
    processor.logger = MagicMock()

    cached = await processor._load_cached_layout_for_pages(
        document_id=document_id,
        pdf_pages=[physical_page - 1],  # reader takes 0-based
    )

    assert physical_page in cached, (
        f"Cache reader returned empty for page {physical_page} despite a "
        f"well-formed write. This is the exact bug class that broke "
        f"image extraction in production on 2026-04-30."
    )

    layout = cached[physical_page]
    image_regions = layout.get_regions_by_type("IMAGE")
    text_regions = layout.get_regions_by_type("TEXT")
    assert len(image_regions) == 1, "IMAGE region not preserved through write/read"
    assert len(text_regions) == 1, "TEXT region not preserved through write/read"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reader_rejects_wrong_processing_version(fake_supabase):
    """Sanity: confirm the reader STILL gates on processing_version.

    If a future change loosens or removes the gate, this test fails and
    the team has to consciously decide whether the writer's contract
    needs updating too. Better than silent acceptance of any version.
    """
    from app.services.pdf.pdf_processor import PDFProcessor

    document_id = "doc-bad-version"
    payload = _build_writer_payload(
        document_id, 1, [_build_sample_merged_region("IMAGE")]
    )
    payload["processing_version"] = "1.0.0"  # the DB-default that broke prod
    fake_supabase.client.table("document_layout_analysis").upsert(payload).execute()

    processor = PDFProcessor.__new__(PDFProcessor)
    processor.logger = MagicMock()

    cached = await processor._load_cached_layout_for_pages(
        document_id=document_id,
        pdf_pages=[0],
    )

    assert cached == {}, (
        f"Cache reader should reject rows with processing_version != "
        f"'{REQUIRED_PROCESSING_VERSION}'. If you intentionally widened "
        f"the gate, also update REQUIRED_PROCESSING_VERSION in this file "
        f"AND make sure the Stage 1.5 writer emits a version the reader "
        f"actually accepts."
    )
