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


def test_a_page_the_site_renamed_is_matched_by_its_title_not_its_old_url():
    renamed = [{"url": "https://stella-ist.com/catalog/acoustic_panels_1/trend_20_2780_600_20/", "text": "Trend 18 (2780*600*18)\nTrend 18 2780 600 18"}]
    assert ia.match_snapshots("Acoustic Panels Stella Trend 20", renamed) == []
    assert len(ia.match_snapshots("Acoustic Panels Stella Trend 18", renamed)) == 1


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


def _img(i, layer="region_crop", verdict=None, embedded=False, sourcing=None, method=None, retired=False):
    q = {k: v for k, v in (("verdict", verdict), ("sourcing", sourcing)) if v}
    m = {"image_quality": q}
    if method:
        m["restore_method"] = method
    if retired:
        m["excluded_from_products"] = {"reason": "replaced"}
    return {"id": i, "extraction_layer": layer, "has_slig_embedding": embedded, "metadata": m}


def test_a_low_res_image_embedded_without_sourcing_fails_the_audit():
    findings, metrics = [], {}
    ia.image_findings({"DOCKS": [_img("a", verdict="low_res", embedded=True)]}, findings, metrics)
    codes = {f.code: f.severity for f in findings}
    assert codes["images_low_res_embedded"] == ia.FAIL
    assert metrics["images_low_res_embedded"] == 1


def test_a_product_with_only_smoothed_images_is_reported_and_a_maker_image_clears_it():
    weak = [_img("r", layer="restored", method="classical", verdict="low_res", embedded=True)]
    findings, metrics = [], {}
    ia.image_findings({"DOCKS": weak}, findings, metrics)
    assert [f.examples for f in findings if f.code == "products_only_weak_images"] == [["DOCKS"]]
    findings, metrics = [], {}
    ia.image_findings({"DOCKS": weak + [_img("m", layer="manufacturer_original", verdict="ok", embedded=True)]}, findings, metrics)
    assert not [f for f in findings if f.code == "products_only_weak_images"]


def test_waiting_crops_warn_and_retired_images_are_ignored():
    findings, metrics = [], {}
    ia.image_findings({"X": [_img("p", verdict="low_res", sourcing="pending"),
                             _img("old", verdict="low_res", embedded=True, retired=True)]}, findings, metrics)
    codes = {f.code for f in findings}
    assert "images_awaiting_sourcing" in codes and "images_low_res_embedded" not in codes


def test_a_low_res_image_sourcing_already_looked_at_is_not_reported():
    findings, metrics = [], {}
    ia.image_findings({"SAXA": [_img("k", verdict="low_res", embedded=True, sourcing="kept_source")]}, findings, metrics)
    assert not [f for f in findings if f.code == "images_low_res_embedded"]


def _shot(id_, layer="region_crop", w=160, native=None, src=None, page=172, bbox=None):
    m = {"width": w, "height": w, "image_quality": {"native_px": native} if native else {}}
    if src:
        m["source_native_px"] = src
    return {"id": id_, "extraction_layer": layer, "page_number": page, "bbox": bbox, "metadata": m}


def test_a_repair_enlarged_past_twice_its_source_fails_and_an_oversized_crop_only_warns():
    findings, metrics = [], {}
    imgs = {"Reality": [_shot("r", layer="restored", w=650, src=[94, 94]), _shot("c", w=400, native=[143, 143]),
                        _shot("ok", layer="restored", w=180, src=[143, 72])]}
    ia.image_integrity_findings(imgs, [], findings, metrics)
    by = {f.code: f for f in findings}
    assert by["images_upscaled"].severity == ia.FAIL and by["images_upscaled"].examples == ["r"]
    assert by["crops_rendered_above_source"].severity == ia.WARN and by["crops_rendered_above_source"].examples == ["c"]


def test_two_live_crops_from_one_spot_fail_and_an_image_on_another_series_page_fails():
    spot = [0.144, 0.796, 0.254, 0.128]
    findings, metrics = [], {}
    other = [{"image_id": "x", "page": 172, "product": "calma", "printed": ["alberca", "amazonia"], "label": "amazonia"}]
    ia.image_integrity_findings({"Calma": [_shot("a", bbox=spot), _shot("b", bbox=spot), _shot("c", bbox=[0.6, 0.8, 0.25, 0.13])]},
                                other, findings, metrics)
    assert {f.code for f in findings} == {"duplicate_images_same_position", "images_on_other_series_page"}
    assert metrics["images_duplicated_position"] == 1 and metrics["images_on_other_series_page"] == 1


def test_a_description_that_is_the_model_refusing_fails_and_a_real_one_passes():
    findings, metrics = [], {}
    products = [{"name": "Saboya", "description": "The source text contains only format and finish listings, so there is no description to write."},
                {"name": "Ardesia", "description": "A slate-inspired porcelain in Acero and Antracita."}, {"name": "Onix", "description": ""}]
    ia.product_content_findings(products, findings, metrics)
    assert [(f.code, f.examples) for f in findings] == [("description_is_model_refusal", ["Saboya"])]


def test_an_attribute_naming_another_series_of_the_document_fails_but_its_own_name_does_not():
    findings, metrics = [], {}
    products = [{"name": "Saboya", "attributes": {"slip_resistance": ["antislip m188 (redstone series)"]}},
                {"name": "Redstone", "attributes": {"collection": "Redstone"}}, {"name": "Peldaño", "attributes": {"note": "peldano"}}]
    ia.product_content_findings(products, findings, metrics)
    assert [f.code for f in findings] == ["attributes_name_other_series"]
    assert findings[0].examples == [{"product": "Saboya", "names": ["redstone"]}]


def test_a_series_named_after_a_place_of_use_is_not_a_leak_when_it_is_where_the_product_goes():
    findings, metrics = [], {}
    products = [{"name": "Rebel", "attributes": {"room": ["bathroom", "outdoor"]}}, {"name": "Docks", "attributes": {"application": "Outdoor"}},
                {"name": "Outdoor", "attributes": {}}, {"name": "Area", "attributes": {"notes": "matches the outdoor series"}}]
    ia.product_content_findings(products, findings, metrics)
    assert next(f for f in findings if f.code == "attributes_name_other_series").examples == [{"product": "Area", "names": ["outdoor"]}]


def test_a_variant_the_maker_photographed_and_the_series_own_pages_outside_its_range_fail():
    findings, metrics = [], {}
    cov = [{"product": "Saboya", "missing_variants": ["gris"], "pages_outside": [257]},
           {"product": "Calma", "missing_variants": [], "pages_outside": []}]
    ia.series_coverage_findings(cov, findings, metrics)
    assert {f.code for f in findings} == {"variants_missing_vs_sources", "series_pages_outside_range"}
    assert metrics["products_missing_site_variants"] == 1 and metrics["products_with_own_pages_outside_range"] == 1


def test_a_series_name_that_is_the_products_own_colour_is_not_a_leak():
    findings, metrics = [], {}
    products = [{"name": "Titan", "attributes": {"color": ["greige", "oxido"]}}, {"name": "Oxido", "attributes": {}},
                {"name": "Saboya", "attributes": {"slip_resistance": ["antislip m188 (redstone series)"]}}, {"name": "Redstone", "attributes": {}}]
    ia.product_content_findings(products, findings, metrics)
    leak = next(f for f in findings if f.code == "attributes_name_other_series")
    assert [e["product"] for e in leak.examples] == ["Saboya"]


def test_the_same_spot_in_two_pdfs_is_a_duplicate_only_when_the_pictures_match():
    def img(i, doc, h, bbox=(0.53, 0.179, 0.405, 0.255)):
        return {"id": i, "document_id": doc, "page_number": 64, "extraction_layer": "region_crop", "bbox": list(bbox),
                "metadata": {"width": 800, "height": 400, "layer_info": {"perceptual_hash": h}}}
    findings, metrics = [], {}
    ia.image_integrity_findings({"Ardenza": [img("a", "actual", "afc989ec0ae1e0b6"), img("b", "innova", "a7c989ec0ae0f0be"),
                                             img("c", "actual", "bacd4ff6194a022d", (0.1, 0.1, 0.2, 0.2)), img("d", "innova", "ae720c26192fc4fb", (0.1, 0.1, 0.2, 0.2))]}, [], findings, metrics)
    twins = next(f for f in findings if f.code == "images_hash_twins_across_documents")
    assert twins.severity == ia.WARN and twins.examples == [["a", "b"]]
    assert "duplicate_images_same_position" not in {f.code for f in findings}


def test_a_series_the_makers_site_does_not_show_is_named():
    findings, metrics = [], {}
    ia.series_coverage_findings([{"product": "Padua", "missing_variants": [], "pages_outside": [], "site_status": "not_found"},
                                 {"product": "Saboya", "missing_variants": [], "pages_outside": [], "site_status": "found"}], findings, metrics)
    off = next(f for f in findings if f.code == "series_not_on_maker_site")
    assert off.severity == ia.WARN and off.examples == ["Padua"]


def test_an_image_nothing_linked_or_excluded_fails_the_audit():
    findings, metrics = [], {}
    ia.unsettled_image_findings([{"image_id": "a", "page": 4}], findings, metrics)
    assert metrics["images_unsettled"] == 1
    assert [(f.severity, f.code, f.count) for f in findings] == [(ia.FAIL, "images_unsettled", 1)]
    findings, metrics = [], {}
    ia.unsettled_image_findings([], findings, metrics)
    assert findings == [] and metrics["images_unsettled"] == 0


def test_only_a_live_material_image_is_owed_a_product():
    owes = lambda meta, **row: ia.owes_a_product({"id": "x", "metadata": meta, **row})
    assert owes({}) and owes({"ai_classification": {"is_material": True}})
    assert not owes({"excluded_from_products": {"reason": "duplicate_image"}})
    assert not owes({"bundle_skipped_reason": "icon_candidate_spec_path"})
    assert not owes({"ai_classification": {"is_material": False}})
    assert not owes({}, image_type="DECORATIVE")


def test_unsettled_lists_live_material_images_with_no_link():
    rows = [{"id": "a", "page_number": 1, "metadata": {}}, {"id": "b", "page_number": 2, "metadata": {}},
            {"id": "c", "page_number": 3, "metadata": {"excluded_from_products": {}}},
            {"id": "d", "page_number": 4, "metadata": {"bundle_skipped_reason": "icon_candidate_spec_path"}}]
    assert ia.unsettled_images(rows, {"b"}) == [{"image_id": "a", "page": 1}]


def test_an_image_that_does_not_resolve_to_its_file_fails():
    r = run(unresolved_paths=[{"image_id": "i1", "problem": "url_disagrees_with_path"}])
    assert r.status == ia.FAIL and "image_paths_unresolved" in codes(r) and r.metrics["images_path_unresolved"] == 1


def test_an_open_degraded_or_undecided_correction_keeps_the_audit_in_review():
    open_rows = [{"id": "c1", "code": "images_repaired_without_gemini", "outcome": "degraded", "count": 15, "message": "15 crops"},
                 {"id": "c2", "code": "ai_work_unbilled", "outcome": "needs_decision", "count": 1, "message": "nobody to bill"}]
    r = run(open_corrections=open_rows)
    assert r.status == ia.WARN and {"corrections_degraded", "corrections_need_decision"} <= codes(r)
    assert r.metrics["corrections_degraded_open"] == 15 and r.metrics["corrections_needing_decision"] == 1
    assert run(open_corrections=[]).status == ia.PASS
