"""Table chunks: pure builder, idempotent writer, embed-only-new-ids, scoped backfill."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tc = _load("table_chunks_under_test", "app/services/chunking/table_chunks.py")


def _table(rows, headers=("Code", "Size", "Price"), **extra):
    base = {
        "id": "t1", "page_number": 4, "headers": list(headers),
        "table_data": {"rows": rows}, "extractor": "vlm",
        "metadata": {"title": "Tile sizes"},
    }
    base.update(extra)
    return base


def test_header_repeats_in_every_part_and_title_line():
    rows = [[f"AB{i:04d}", "60x60", "9.5"] for i in range(60)]
    chunks = tc.build_table_chunks("Marble", [_table(rows)], rows_per_chunk=25)
    assert [c["parts"] for c in chunks] == [3, 3, 3]
    for chunk in chunks:
        lines = chunk["content"].split("\n")
        assert lines[0] == "Marble - Tile sizes (page 4)"
        assert lines[1] == "Code | Size"
    assert chunks[2]["content"].count("\n") == 1 + 10


def test_price_columns_dropped():
    chunks = tc.build_table_chunks(
        "P", [_table([["A1", "x", "12.00"]], headers=("Code", "Size", "Price EUR"))]
    )
    assert "12.00" not in chunks[0]["content"]
    assert "Price" not in chunks[0]["content"]
    chunks = tc.build_table_chunks("P", [_table([["A1", "x", "5"]], headers=("Code", "Size", "RRP €"))])
    assert "RRP" not in chunks[0]["content"]


def test_ocr_fragment_skipped_but_titled_vlm_table_kept():
    fragment = _table([["A1", "x", "1"]], extractor="paddleocr-vl", metadata={})
    titled = _table([["A1", "x", "1"]], extractor="paddleocr-vl")
    assert tc.build_table_chunks("P", [fragment]) == []
    assert len(tc.build_table_chunks("P", [titled])) == 1


def test_none_cells_empty_rows_and_numbers():
    rows = [["A1", None, 3.5], [None, None, None], ["", "  ", ""], ["A2", 60, 1]]
    chunks = tc.build_table_chunks("P", [_table(rows)])
    lines = chunks[0]["content"].split("\n")
    assert lines[2:] == ["A1 | ", "A2 | 60"]


def test_unusable_tables_skipped():
    assert tc.build_table_chunks("P", [_table([["A", "b", "c"]], headers=())]) == []
    assert tc.build_table_chunks("P", [_table([], headers=("Code",))]) == []
    assert tc.build_table_chunks("P", [_table([["9.9"]], headers=("Price",))]) == []


def test_metadata_shape_and_needs_review_default():
    chunk = tc.build_table_chunks("P", [_table([["A1", "x", "1"]])])[0]
    assert chunk["metadata"] == {
        "source": "table_block", "table_id": "t1", "page_number": 4, "part": 1, "parts": 1,
        "extractor": "vlm", "table_title": "Tile sizes", "needs_review": False, "is_table": True,
    }
    flagged = _table([["A1", "x", "1"]], metadata={"title": "T", "needs_review": True})
    assert tc.build_table_chunks("P", [flagged])[0]["metadata"]["needs_review"] is True


def test_chunk_index_unique_even_with_many_parts_and_tables():
    big = [[f"C{i:05d}", "s", "1"] for i in range(400)]
    tables = [_table(big, id="a"), _table(big, id="b"), _table(big, id="c", page_number=5)]
    chunks = tc.build_table_chunks("P", tables, rows_per_chunk=25)
    indexes = tc.assign_chunk_indexes(chunks)
    assert len(indexes) == len(set(indexes)) == 48
    assert min(indexes) >= 100000


def test_content_is_stable():
    rows = [["A1", "x", "1"], ["A2", "y", "2"]]
    first = tc.build_table_chunks("P", [_table(rows)])
    assert first == tc.build_table_chunks("P", [_table(rows)])


def test_item_codes_found():
    codes = tc.item_codes("Marble - T (page 1)\nCode | Size\nAB1234 | 60x60\n123456 | x\nplain words")
    assert {"AB1234", "123456"} <= codes
    assert "plain" not in codes


class FakeResult:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.payload, self.filters = db, table, None, None, []

    def delete(self):
        self.op = "delete"
        return self

    def insert(self, payload):
        self.op, self.payload = "insert", payload
        return self

    def update(self, payload):
        self.op, self.payload = "update", payload
        return self

    def select(self, _cols):
        self.op = "select"
        return self

    def eq(self, col, val):
        self.filters.append(("eq", col, val))
        return self

    def in_(self, col, vals):
        self.filters.append(("in", col, list(vals)))
        return self

    def execute(self):
        self.db.log.append((self.op, self.payload, self.filters))
        if self.op == "insert":
            out = []
            for row in self.payload:
                self.db.counter += 1
                self.db.rows[f"id{self.db.counter}"] = dict(row)
                out.append({"id": f"id{self.db.counter}"})
            return FakeResult(out)
        if self.op == "select":
            ins = [f for f in self.filters if f[0] == "in"]
            if ins:
                return FakeResult([{"id": i, "content": self.db.rows[i]["content"]} for i in ins[0][2] if i in self.db.rows])
            wanted = {f[1]: f[2] for f in self.filters if f[0] == "eq"}
            return FakeResult([
                {"id": i} for i, r in self.db.rows.items()
                if r.get("document_id") == wanted.get("document_id")
                and r.get("product_id") == wanted.get("product_id")
                and (r.get("metadata") or {}).get("source") == wanted.get("metadata->>source")
            ])
        if self.op == "delete":
            for i in next(f[2] for f in self.filters if f[0] == "in"):
                self.db.rows.pop(i, None)
            return FakeResult([])
        if self.op == "update":
            target = next(f[2] for f in self.filters if f[1] == "id")
            self.db.rows[target].update(self.payload)
        return FakeResult([])


class FakeDb:
    def __init__(self):
        self.log, self.rows, self.counter = [], {}, 0

    def table(self, name):
        assert name == "document_chunks"
        return FakeQuery(self, name)


class FakeEmbedder:
    voyage_model = "voyage-fake"

    def __init__(self, fail_on=()):
        self.fail_on, self.seen = fail_on, []

    async def generate_batch_embeddings(self, texts, dimensions, input_type, workspace_id=None):
        self.seen.extend(texts)
        assert dimensions == 1024 and input_type == "document"
        return [None if any(f in t for f in self.fail_on) else [0.1] * 4 for t in texts]


def _chunks(n=3):
    rows = [[f"AB{i:04d}", "60x60", "1"] for i in range(n * 25)]
    return tc.build_table_chunks("P", [_table(rows)])


def test_write_inserts_table_blocks_for_exactly_this_product():
    db = FakeDb()
    ids = asyncio.run(tc.write_table_chunks(db, "ws", "doc", "prod", _chunks()))
    assert len(ids) == 3
    op, _, filters = db.log[0]
    assert op == "select"
    assert ("eq", "metadata->>source", "table_block") in filters
    assert ("eq", "document_id", "doc") in filters and ("eq", "product_id", "prod") in filters
    inserted = db.log[1][1][0]
    assert inserted["chunk_type"] == "technical_specs"
    assert inserted["chunk_type_status"] == "classified"
    assert inserted["has_text_embedding"] is False
    assert inserted["workspace_id"] == "ws" and inserted["product_id"] == "prod"
    assert [r["chunk_index"] for r in db.log[1][1]] == [100401, 100402, 100403]


def test_rewrite_inserts_first_and_removes_only_the_superseded_ids():
    db = FakeDb()
    first = asyncio.run(tc.write_table_chunks(db, "ws", "doc", "prod", _chunks()))
    db.rows["other"] = {"document_id": "doc", "product_id": "prod", "metadata": {"source": "text"}, "content": "keep"}
    second = asyncio.run(tc.write_table_chunks(db, "ws", "doc", "prod", _chunks()))
    ops = [entry[0] for entry in db.log]
    assert ops == ["select", "insert", "select", "insert", "delete"]
    assert not set(first) & set(db.rows) and set(second) <= set(db.rows) and "other" in db.rows


def test_a_failed_insert_leaves_the_previous_chunks_in_place():
    db = FakeDb()
    first = asyncio.run(tc.write_table_chunks(db, "ws", "doc", "prod", _chunks()))

    class Broken(FakeQuery):
        def execute(self):
            if self.op == "insert":
                raise RuntimeError("insert failed")
            return super().execute()

    class BrokenDb(FakeDb):
        def table(self, name):
            return Broken(self, name)

    broken = BrokenDb()
    broken.rows = dict(db.rows)
    try:
        asyncio.run(tc.write_table_chunks(broken, "ws", "doc", "prod", _chunks()))
    except RuntimeError:
        pass
    assert set(first) <= set(broken.rows)


def test_price_columns_are_found_in_other_languages_and_under_a_title_row():
    headers = ["Technical data", "", ""]
    rows = [["Code", "Size", "Τιμή λιανικής"], ["A1234", "60x60", "12,50"]]
    chunks = tc.build_table_chunks("P", [{"id": "t", "headers": headers, "table_data": {"rows": rows}, "extractor": "pymupdf-text", "metadata": {"title": "T"}}])
    assert chunks and "12,50" not in chunks[0]["content"] and "Code | Size" in chunks[0]["content"]


def test_a_column_of_currency_amounts_is_dropped_whatever_its_header():
    rows = [["A1234", "60x60", "12,50 EUR"], ["A1235", "60x60", "13,50 EUR"]]
    chunks = tc.build_table_chunks("P", [{"id": "t", "headers": ["Code", "Size", "List"], "table_data": {"rows": rows}, "extractor": "pymupdf-text", "metadata": {"title": "T"}}])
    assert "EUR" not in chunks[0]["content"] and "A1234 | 60x60" in chunks[0]["content"]


def test_embed_touches_only_given_ids_and_reports_failures():
    db = FakeDb()
    ids = asyncio.run(tc.write_table_chunks(db, "ws", "doc", "prod", _chunks()))
    db.rows["other"] = {"content": "must not be embedded"}
    embedder = FakeEmbedder(fail_on=("AB0030",))
    result = asyncio.run(tc.embed_chunks_once(db, ids, embedder=embedder))
    assert result["embedded"] == 2
    assert [f["id"] for f in result["failed"]] == [ids[1]]
    assert "must not be embedded" not in embedder.seen
    assert db.rows[ids[0]]["has_text_embedding"] is True
    assert db.rows[ids[0]]["embedding_model"] == "voyage-fake"
    assert db.rows[ids[0]]["embedding_dimension"] == 4
    assert db.rows[ids[1]]["has_text_embedding"] is False
    assert "has_text_embedding" not in db.rows["other"]


def test_embed_batch_exception_is_returned_not_swallowed():
    db = FakeDb()
    ids = asyncio.run(tc.write_table_chunks(db, "ws", "doc", "prod", _chunks(1)))

    class Boom(FakeEmbedder):
        async def generate_batch_embeddings(self, *a, **k):
            raise RuntimeError("voyage down")

    result = asyncio.run(tc.embed_chunks_once(db, ids, embedder=Boom()))
    assert result["embedded"] == 0
    assert "voyage down" in result["failed"][0]["reason"]


def test_embed_with_no_ids_is_a_noop():
    assert asyncio.run(tc.embed_chunks_once(FakeDb(), [])) == {"embedded": 0, "failed": []}


def _load_backfill(monkeypatch, query_log):
    class Q:
        @property
        def not_(self):
            return self

        def __getattr__(self, name):
            def call(*args, **kwargs):
                query_log.append((name, args))
                return self
            return call

        def execute(self):
            return FakeResult([{"id": "c1", "content": "x", "workspace_id": "w"}])

    stub_client = types.SimpleNamespace(client=types.SimpleNamespace(table=lambda _n: Q()))
    stubs = {
        "app": types.ModuleType("app"),
        "app.config": types.SimpleNamespace(get_settings=lambda: None),
        "app.services": types.ModuleType("app.services"),
        "app.services.core": types.ModuleType("app.services.core"),
        "app.services.core.supabase_client": types.SimpleNamespace(get_supabase_client=lambda: stub_client),
        "app.services.embeddings": types.ModuleType("app.services.embeddings"),
        "app.services.embeddings.real_embeddings_service": types.SimpleNamespace(RealEmbeddingsService=object),
    }
    for key, value in stubs.items():
        monkeypatch.setitem(sys.modules, key, value)
    return _load("backfill_under_test", "app/services/embeddings/text_embedding_backfill.py")


def test_backfill_scope_defaults_unchanged_and_scoped_when_given(monkeypatch):
    log: list = []
    mod = _load_backfill(monkeypatch, log)
    asyncio.run(mod._fetch_chunks_missing_embedding(10, None))
    assert not any(name in ("in_",) for name, _ in log)
    assert not any(name == "eq" and args[0] == "document_id" for name, args in log)
    log.clear()
    asyncio.run(mod._fetch_chunks_missing_embedding(10, "w", document_id="d1", chunk_ids=["c1", "c2"]))
    assert ("eq", ("document_id", "d1")) in log
    assert ("in_", ("id", ["c1", "c2"])) in log
