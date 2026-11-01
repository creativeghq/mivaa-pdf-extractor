"""A code table has almost no sentence punctuation; the quality gate must not eat it."""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.config"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
sys.modules["app.config.confidence_thresholds"] = _load(
    "app.config.confidence_thresholds_under_test", "app/config/confidence_thresholds.py"
)
svc = _load("unified_chunking_under_test", "app/services/chunking/unified_chunking_service.py")


def _chunk(service, content, **meta):
    return service._create_chunk(
        content=content, document_id="d", chunk_index=0, start_position=0, metadata=meta
    )


def _table_text(rows=70):
    lines = ["Marble - Sizes (page 3)", "Code | Size | Pcs"]
    lines += [f"AB{i:04d} | 60x60 | {i}" for i in range(rows)]
    return "\n".join(lines)


def test_table_chunk_without_punctuation_survives_and_junk_does_not():
    service = svc.UnifiedChunkingService()
    table = _chunk(service, _table_text(), is_table=True)
    table.quality_score = service._calculate_chunk_quality(table)
    assert table.quality_score < service.min_quality_threshold
    junk = _chunk(service, "lorem ipsum dolor sit amet")
    junk.quality_score = service._calculate_chunk_quality(junk)
    kept = service.filter_chunks([table, junk])
    assert kept == [table]
    assert service.quality_metrics.table_chunks_exempt == 1
    assert service.quality_metrics.low_quality_rejected == 1


def test_code_dense_chunk_is_exempt_without_flag_and_not_deduped():
    service = svc.UnifiedChunkingService()
    first = _chunk(service, _table_text(20))
    second = _chunk(service, _table_text(20))
    for c in (first, second):
        c.quality_score = 0.1
    assert len(service.filter_chunks([first, second])) == 2


def test_identical_prose_is_still_deduped():
    service = svc.UnifiedChunkingService()
    text = "This is a proper sentence about the product. It has two. And three. Done."
    a, b = _chunk(service, text), _chunk(service, text)
    a.quality_score = b.quality_score = 0.9
    assert service.filter_chunks([a, b]) == [a]
    assert service.quality_metrics.exact_duplicates_prevented == 1


def test_short_digit_chunk_is_not_code_dense():
    service = svc.UnifiedChunkingService()
    page = _chunk(service, "Page 12")
    page.quality_score = 0.1
    assert service.filter_chunks([page]) == []


def test_oversized_table_is_split_by_row_groups_with_header_repeated():
    service = svc.UnifiedChunkingService()
    text = _table_text(700)
    parts = service._split_table_content(text)
    assert len(parts) > 1
    assert all(len(p) <= service.config.max_chunk_size * 4 for p in parts)
    for part in parts:
        assert part.split("\n")[0] == "Marble - Sizes (page 3)"
    rows = [line for p in parts for line in p.split("\n")[1:]]
    assert rows == text.split("\n")[1:]


def test_small_table_not_split():
    service = svc.UnifiedChunkingService()
    assert service._split_table_content(_table_text(10)) == [_table_text(10)]
