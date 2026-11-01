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
ia = _load("ingestion_audit_under_test", "app/services/pdf/ingestion_audit.py")


def nums(*values):
    return {str(v) for v in values}


def page(n, reference, platform, route="text_layer"):
    return {"page": n, "reference_numbers": set(reference), "platform_numbers": set(platform), "route": route}


CODE_NUMBERS = [f"80{i:05d}" for i in range(10)]
GOOD_PAGES = [page(1, nums(*range(100, 130), *CODE_NUMBERS), nums(*range(100, 130), *CODE_NUMBERS))]
TABLE = {"id": "t1", "extractor": "pymupdf-text", "metadata": {"title": "Codes"}, "table_data": {"rows": [[f"80{i:05d}", "x"] for i in range(10)]}}
CHUNK = {"id": "c1", "has_text_embedding": True, "content": "Codes | 8000000", "metadata": {"source": "table_block", "table_id": "t1"}}
PRODUCT = {"name": "P", "n_tables": 1, "n_chunks": 1, "has_embedding": True, "n_images": 1}


def run(**over):
    args = dict(source_kept=True, pages=GOOD_PAGES, tables=[TABLE], chunks=[CHUNK], products=[PRODUCT], pdf_text=" ".join(f"80{i:05d}" for i in range(10)), platform_text=" ".join(f"80{i:05d}" for i in range(10)))
    args.update(over)
    return ia.audit_document(**args)


def codes(result):
    return {f.code for f in result.findings}


def test_a_complete_ingestion_passes():
    r = run()
    assert r.status == ia.PASS and r.findings == [] and r.metrics["code_coverage"] == 1.0


def test_an_unkept_source_is_unverifiable_not_passed():
    r = run(source_kept=False)
    assert r.status == ia.UNVERIFIABLE and "source_not_kept" in codes(r)


def test_numbers_missing_from_a_page_fail_or_warn_by_share():
    missing_many = [page(1, nums(*range(100, 130), *CODE_NUMBERS), nums(*range(100, 120), *CODE_NUMBERS))]
    r = run(pages=missing_many)
    assert r.status == ia.FAIL and "numbers_missing" in codes(r)
    missing_few = [page(1, nums(*range(100, 150), *CODE_NUMBERS), nums(*range(100, 148), *CODE_NUMBERS))]
    assert "numbers_missing_few" in codes(run(pages=missing_few)) and run(pages=missing_few).status == ia.WARN


def test_a_page_with_few_numbers_is_not_judged():
    empty = dict(TABLE, table_data={"rows": []})
    assert run(pages=[page(1, nums(1000, 2000), set())], tables=[empty]).status == ia.PASS


def test_codes_printed_in_the_pdf_but_not_saved_are_listed():
    pdf = " ".join(f"80{i:05d}" for i in range(12))
    r = run(pdf_text=pdf)
    assert "codes_missing_few" in codes(r) or "codes_missing" in codes(r)
    missing = next(f for f in r.findings if f.code.startswith("codes_missing"))
    assert missing.examples == ["8000010", "8000011"] and r.metrics["code_coverage"] < 1


def test_the_code_shape_is_learned_from_the_saved_tables():
    assert ia.dominant_code_signature([f"80{i:05d}" for i in range(10)]) == "9999999"
    assert ia.dominant_code_signature(["A1", "B2"]) is None
    assert ia.tokens_with_signature("see 8001234 and 12 and AB1234", "9999999") == {"8001234"}


def test_unembedded_chunks_and_tables_without_chunks_fail():
    r = run(chunks=[dict(CHUNK, has_text_embedding=False)])
    assert r.status == ia.FAIL and "chunks_not_embedded" in codes(r)
    r = run(chunks=[])
    assert "tables_without_chunks" in codes(r)


def test_prices_in_a_table_chunk_fail():
    r = run(chunks=[dict(CHUNK, content="Code | List price\n8000001 | 12,50 EUR")])
    assert "price_in_chunks" in codes(r) and r.metrics["price_like_table_chunks"] == 1


def test_flagged_tables_and_raw_fragments_are_reported():
    flagged = dict(TABLE, id="t2", metadata={"title": "x", "needs_review": True})
    fragments = [{"id": f"f{i}", "extractor": "paddleocr-vl", "metadata": {}, "table_data": {"rows": []}} for i in range(3)]
    r = run(tables=[TABLE, flagged] + fragments, chunks=[CHUNK, dict(CHUNK, id="c2", metadata={"source": "table_block", "table_id": "t2"})])
    assert {"tables_need_review", "mostly_raw_fragments"} <= codes(r)


def test_products_lacking_parts_are_named():
    r = run(products=[PRODUCT, dict(PRODUCT, name="Bare", n_tables=0, has_embedding=False)])
    names = {f.code: f.examples for f in r.findings}
    assert names["products_without_tables"] == ["Bare"] and r.status == ia.FAIL


def test_a_layout_that_predates_the_check_is_called_out():
    r = run(pages=[dict(p, route=None) for p in GOOD_PAGES])
    assert "layout_not_routed" in codes(r) and r.status == ia.WARN


def test_platform_numbers_union_layout_chunks_and_tables():
    got = ia.platform_numbers_for_page("a 100 b", ["x 200"], ["<td>300</td>"])
    assert got == {"100", "200", "300"}


def test_the_result_serialises():
    d = run(source_kept=False).as_dict()
    assert d["status"] == "unverifiable" and d["findings"][0]["code"] == "source_not_kept"


def test_numbers_in_saved_tables_that_the_source_never_prints_are_reported():
    bad = dict(TABLE, table_data={"rows": [[f"80{i:05d}", "x", 9999 + i] for i in range(10)]})
    r = run(tables=[bad])
    f = next(f for f in r.findings if f.code == "numbers_not_in_source")
    assert f.severity == ia.WARN and f.count == 10 and r.metrics["table_numbers_not_in_source"] == 10


def test_a_web_source_is_judged_like_a_pdf_but_never_for_layout_routing():
    pages = [page(1, nums(*range(100, 130), *CODE_NUMBERS), nums(*range(100, 130), *CODE_NUMBERS), route=None)]
    r = run(pages=pages, source_kind="web")
    assert r.status == ia.PASS and r.metrics["source_kind"] == "web" and "layout_not_routed" not in codes(r)


def test_one_missing_number_on_a_small_page_is_a_warning_not_a_failure():
    small = [page(1, nums(*range(100, 112), *CODE_NUMBERS), nums(*range(100, 111), *CODE_NUMBERS))]
    r = run(pages=small)
    assert r.status == ia.WARN and "numbers_missing_few" in codes(r) and "numbers_missing" not in codes(r)


def test_a_failed_page_lists_the_numbers_that_are_missing():
    r = run(pages=[page(1, nums(*range(100, 130), *CODE_NUMBERS), nums(*range(100, 120), *CODE_NUMBERS))])
    gap = next(f for f in r.findings if f.code == "numbers_missing").examples[0]
    assert gap["examples"][:3] == ["120", "121", "122"]


def test_a_recorded_source_that_is_gone_from_storage_is_named_as_such():
    r = run(source_kept=False, source_state="missing_object")
    assert "source_missing_in_storage" in codes(r) and r.status == ia.UNVERIFIABLE


SNAPS = [
    {"url": "https://stella-ist.com/catalog/acoustic_panels_1/comfort_9_2780_600_9/", "text": "Comfort 9 2780 600 9 mm pack of 1 weight 12,5 kg"},
    {"url": "https://stella-ist.com/catalog/acoustic_panels_1/mila_9_2780_600_9/", "text": "Mila 9 2780 600 9"},
]


def web_run(products):
    return ia.audit_document(source_kept=True, pages=[page(1, set(), set(), route=None)], tables=[], chunks=[], products=[],
                             pdf_text=None, platform_text="", source_kind="web", web={"snapshots": SNAPS, "products": products})


def test_a_product_is_matched_to_its_own_page_by_the_words_that_name_it():
    assert [s["url"].rsplit("/", 2)[-2] for s in ia.match_snapshots("Acoustic Panels Stella Comfort 9", SNAPS)] == ["comfort_9_2780_600_9"]
    assert ia.match_snapshots("Wall Panels MDF Stella Wave", SNAPS) == []


def test_every_number_a_product_stores_must_be_on_its_page_and_a_gone_page_is_reported():
    ok = web_run([{"name": "Acoustic Panels Stella Comfort 9", "table_numbers": {"2780", "12,5"}}])
    assert ok.status == ia.PASS
    bad = web_run([{"name": "Acoustic Panels Stella Comfort 9", "table_numbers": {"2780", "999", "888"}}])
    f = next(f for f in bad.findings if f.code == "web_numbers_not_on_page")
    assert bad.status == ia.FAIL and f.examples[0]["product"].endswith("Comfort 9") and f.examples[0]["examples"] == ["888", "999"]
    gone = web_run([{"name": "Wall Panels MDF Stella Wave", "table_numbers": {"1", "2"}}])
    assert "web_products_without_page" in codes(gone) and gone.status == ia.WARN


def test_a_few_unprinted_numbers_are_a_warning_only():
    nums_on_page = {str(1000 + i) for i in range(30)}
    snaps = [{"url": "https://x/comfort_9/", "text": " ".join(nums_on_page)}]
    r = ia.audit_document(source_kept=True, pages=[], tables=[], chunks=[], products=[], pdf_text=None, platform_text="", source_kind="web",
                          web={"snapshots": snaps, "products": [{"name": "Comfort 9", "table_numbers": nums_on_page | {"7777", "8888", "9999"}}]})
    assert r.status == ia.WARN and "web_numbers_not_on_page_few" in codes(r)
