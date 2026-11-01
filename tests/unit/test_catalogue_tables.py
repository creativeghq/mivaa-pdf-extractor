import asyncio
import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.services", "app.services.pdf"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_load("app.services.pdf.ocr_completeness", "app/services/pdf/ocr_completeness.py")
ct = _load("catalogue_tables_under_test", "app/services/pdf/catalogue_tables.py")


def table_html(rows):
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


ROWS = [["Type", "Q"], ["K10", "893"], ["K11", "1227"], ["K12", "1500"]]
PRODUCTS = [
    {"id": "v", "name": "KORATHERM VERTIKAL", "metadata": {"page_range": [10]}},
    {"id": "vm", "name": "KORATHERM VERTIKAL - M", "metadata": {"page_range": [11]}},
    {"id": "h", "name": "KORATHERM HORIZONTAL", "metadata": {"page_range": [18]}},
]


def layout_row(page, heading, source="text-layer", completeness=None, needs_review=False):
    meta = {"source": source, "html": table_html(ROWS), "needs_review": needs_review}
    if completeness:
        meta["completeness"] = completeness
    return {"page_number": page, "layout_elements": [
        {"region_type": "TITLE", "text_content": heading, "metadata": {}},
        {"region_type": "TABLE", "text_content": "", "metadata": meta},
    ]}


def test_a_heading_names_every_product_it_covers():
    assert ct.heading_names("KORATHERM VERTIKAL, VERTIKAL - M") == ["VERTIKAL", "VERTIKAL - M"]
    got = ct.products_for_heading("KORATHERM VERTIKAL, VERTIKAL - M", PRODUCTS)
    assert [p["id"] for p in got] == ["v", "vm"]


def test_only_pages_that_no_product_owns_are_planned_and_each_table_goes_to_the_named_products():
    layout = [layout_row(27, "KORATHERM VERTIKAL, VERTIKAL - M"), layout_row(10, "KORATHERM VERTIKAL")]
    plans, unmatched = ct.plan_catalogue_tables(layout, PRODUCTS)
    assert unmatched == [] and len(plans) == 1
    assert plans[0]["page"] == 27 and plans[0]["products"] == ["v", "vm"] and plans[0]["extractor"] == "pymupdf-text"


def test_a_heading_that_names_no_product_is_reported_never_guessed():
    plans, unmatched = ct.plan_catalogue_tables([layout_row(30, "ACCESSORIES")], PRODUCTS)
    assert plans == [] and unmatched == [{"page": 30, "heading": "ACCESSORIES", "rows": 4}]


def test_an_ocr_table_is_used_only_when_it_is_complete_and_a_tiny_table_is_never_used():
    assert ct.plan_catalogue_tables([layout_row(27, "KORATHERM VERTIKAL", source="paddleocr-vl")], PRODUCTS)[0] == []
    plans, _ = ct.plan_catalogue_tables([layout_row(27, "KORATHERM VERTIKAL", source="paddleocr-vl", completeness="complete")], PRODUCTS)
    assert plans[0]["extractor"] == "paddleocr-vl"
    tiny = layout_row(27, "KORATHERM VERTIKAL")
    tiny["layout_elements"][1]["metadata"]["html"] = table_html(ROWS[:2])
    assert ct.plan_catalogue_tables([tiny], PRODUCTS)[0] == []


def test_the_stored_row_is_marked_catalogue_wide_and_keeps_the_review_flag():
    plan = ct.plan_catalogue_tables([layout_row(27, "KORATHERM VERTIKAL", needs_review=True)], PRODUCTS)[0][0]
    row = ct.to_row(plan, "v")
    assert row["metadata"]["scope"] == "catalogue_wide" and row["metadata"]["needs_review"] is True
    assert row["page_number"] == 27 and row["headers"] == ["Type", "Q"] and row["table_data"]["num_rows"] == 3


class Db:
    def __init__(self):
        self.inserted, self.deleted, self.client = [], [], self
        self.existing = [{"id": "old", "product_id": "v", "page_number": 27, "metadata": {"scope": "catalogue_wide"}},
                         {"id": "curated", "product_id": "v", "page_number": 27, "metadata": {"title": "mine"}}]

    def table(self, name):
        db, state = self, {"op": "select", "name": name}

        class Query:
            def select(self, *_):
                return self

            def eq(self, *_):
                return self

            def in_(self, *a):
                state["in"] = a
                return self

            def order(self, *_):
                return self

            def insert(self, row):
                state.update(op="insert", row=row)
                return self

            def delete(self):
                state["op"] = "delete"
                return self

            def execute(self):
                if state["op"] == "insert":
                    db.inserted.append(state["row"])
                    return types.SimpleNamespace(data=[{"id": f"new{len(db.inserted)}"}])
                if state["op"] == "delete":
                    db.deleted.extend(state["in"][1])
                    return types.SimpleNamespace(data=[])
                if state["name"] == "products":
                    return types.SimpleNamespace(data=PRODUCTS)
                if state["name"] == "document_layout_analysis":
                    return types.SimpleNamespace(data=[layout_row(27, "KORATHERM VERTIKAL")])
                return types.SimpleNamespace(data=db.existing)

        return Query()


def test_persisting_replaces_only_its_own_earlier_rows_and_never_a_curated_table():
    db = Db()
    summary = asyncio.run(ct.persist_catalogue_wide_tables("doc", db))
    assert summary["rows_written"] == 1 and len(db.inserted) == 1
    assert db.deleted == ["old"]


def test_a_dry_run_writes_nothing():
    db = Db()
    summary = asyncio.run(ct.persist_catalogue_wide_tables("doc", db, apply=False))
    assert summary["tables"] == 1 and db.inserted == [] and db.deleted == []
