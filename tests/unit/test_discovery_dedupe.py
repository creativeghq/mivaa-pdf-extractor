"""A repeat merges, a different model never does on its own; packing follows the size; a site photo follows its file name."""

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.services", "app.services.discovery", "app.services.images"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_load("app.services.discovery.series_merge", "app/services/discovery/series_merge.py")
sys.modules["app.services.discovery"].series_reconcile = _load("app.services.discovery.series_reconcile", "app/services/discovery/series_reconcile.py")
dd = _load("discovery_dedupe_under_test", "app/services/discovery/discovery_dedupe.py")
sf = _load("series_facts_for_packing", "app/services/discovery/series_facts.py")
ll = _load("label_linking_for_photos", "app/services/images/label_linking.py")


@dataclass
class P:
    name: str
    page_range: List[int] = field(default_factory=list)
    description: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def verdict(name, kind, same_as=None, confidence=0.95):
    return {"name": name, "verdict": kind, "same_as": same_as, "confidence": confidence, "reason": "r"}


def test_a_sure_repeat_merges_when_nobody_will_review_and_the_pages_are_unioned():
    products = [P("Amalfi", [1, 2]), P("AMALFI ", [3])]
    out = dd.apply_verdicts(products, [verdict("AMALFI ", dd.SAME_PRODUCT, "Amalfi")], reviewed=False)
    assert [p.name for p in products] == ["Amalfi"] and products[0].page_range == [1, 2, 3]
    assert out["merged"] == [{"name": "AMALFI ", "into": "Amalfi"}]


def test_a_different_model_letter_is_never_merged_automatically():
    assert dd.distinct_models("LINEAR MAX", "LINEAR MAX - M")
    assert dd.distinct_models("LINEAR MAX - E", "LINEAR MAX - ERH (KLTH)")
    assert not dd.distinct_models("Amalfi", "AMALFI (M125)")
    products = [P("LINEAR MAX"), P("LINEAR MAX - M")]
    out = dd.apply_verdicts(products, [verdict("LINEAR MAX - M", dd.SAME_PRODUCT, "LINEAR MAX")], reviewed=False)
    assert len(products) == 2 and out["suggested_merges"] == [{"name": "LINEAR MAX - M", "into": "LINEAR MAX"}]


def test_with_a_review_gate_nothing_merges_and_the_suggestion_is_on_the_row():
    products = [P("Amalfi"), P("AMALFI ")]
    dd.apply_verdicts(products, [verdict("AMALFI ", dd.SAME_PRODUCT, "Amalfi", 0.99)], reviewed=True)
    assert len(products) == 2
    assert products[1].metadata["discovery_suggestion"]["same_as"] == "Amalfi"


def test_a_low_confidence_repeat_and_a_made_up_target_change_nothing():
    products = [P("A one"), P("A two")]
    dd.apply_verdicts(products, [verdict("A two", dd.SAME_PRODUCT, "A one", 0.5), verdict("A one", dd.SAME_PRODUCT, "Nobody")], reviewed=False)
    assert len(products) == 2 and "discovery_suggestion" not in products[0].metadata


def test_an_accessory_is_kept_and_named_as_one():
    products = [P("HM FITTINGS"), P("LINEAR")]
    out = dd.apply_verdicts(products, [verdict("HM FITTINGS", dd.ACCESSORY)], reviewed=False)
    assert len(products) == 2 and out["accessories"] == ["HM FITTINGS"]
    assert products[0].metadata["product_type"] == "accessory"


def test_job_rules_leave_out_what_the_operator_named():
    assert dd.excluded_by_rules("Sample Box 30x60", ["sample box"])
    assert not dd.excluded_by_rules("Samples", ["sample box"])


HEADERS = ["Format", "Variant", "Pieces / box", "m2 / box", "kg / box"]


def packing(*rows):
    return {"table_data": {"headers": HEADERS, "rows": [list(r) for r in rows]}}


def test_a_series_without_packing_takes_the_rows_of_its_own_format_only():
    siblings = [packing(["30X90", "-", "5", "1,35", "23,00"], ["60X120", "-", "2", "1,44", "28,40"]), packing(["30X90", "-", "5", "1,35", "23,00"])]
    by_format = sf.packing_rows_by_format(siblings)
    assert len(by_format["30x90"]) == 1
    tebas = {"name": "Tebas", "metadata": {"available_formats": ["30x90"]}}
    planned = sf.inherited_packing(tebas, [], by_format, HEADERS)
    assert planned["rows"] == [["30X90", "-", "5", "1,35", "23,00"]] and planned["formats"] == ["30x90"]


def test_no_sibling_with_that_format_means_no_invented_table():
    adobe = {"name": "Adobe", "metadata": {"available_formats": ["23x46"]}}
    assert sf.inherited_packing(adobe, [], sf.packing_rows_by_format([packing(["30X90", "-", "5", "1", "2"])]), HEADERS) is None


def test_a_series_that_has_packing_keeps_its_own():
    product = {"name": "Agata", "metadata": {"available_formats": ["30x90"]}}
    own = [packing(["30X90", "-", "9", "9", "9"])]
    assert sf.inherited_packing(product, own, sf.packing_rows_by_format([packing(["30X90", "-", "5", "1", "2"])]), HEADERS) is None


def test_a_photo_belongs_to_the_series_its_file_names():
    series = {n: tuple(ll.tokens_slug(n).replace("_", "-").split("-")) for n in ("Chez Coco", "Arco", "Córdoba", "Valencia", "Peldaño Extrusionado")}
    url = "https://keros.com/wp-content/uploads/CHEZ-COCO-STAR-en-1XvsoesaEScvV4AS.jpg"
    assert ll.series_in_file_name(url, series) == "Chez Coco"
    assert ll.series_in_file_name("https://x/ARCO_2-en-xHfcs81dQ.jpg", series) == "Arco"
    assert ll.series_in_file_name("https://x/CORDOBA-MIX-en-a.jpg", series) == "Córdoba"
    assert ll.series_in_file_name("https://x/SOMETHING-en-a.jpg", series) is None


ct = None


def _catalogue_tables():
    global ct
    if ct is None:
        sys.modules.setdefault("app.services.pdf", types.ModuleType("app.services.pdf"))
        _load("app.services.pdf.ocr_completeness", "app/services/pdf/ocr_completeness.py")
        ct = _load("catalogue_tables_for_routing", "app/services/pdf/catalogue_tables.py")
    return ct


def test_a_table_belongs_to_every_product_its_model_column_names():
    c = _catalogue_tables()
    products = [{"id": "m", "name": "LINEAR MAX (KLM)"}, {"id": "mm", "name": "LINEAR MAX - M (KLMM)"}, {"id": "c", "name": "LINEAR COMFORT", "metadata": {"product_codes": ["KLT"]}}]
    index = c.code_index(products)
    koralux = {"table_data": {"rows": [["KLM\nKLMM", "700.450\n700.450", "690"], [None, None, "70/55"]]}}
    assert c.table_owners(koralux, index) == {"m", "mm"}
    assert c.table_owners({"table_data": {"rows": [["KLT", "1"]]}}, index) == {"c"}


def test_a_code_shared_across_a_range_names_a_format_not_a_model():
    c = _catalogue_tables()
    products = [{"id": str(i), "name": f"S{i}", "metadata": {"product_codes": ["M125"]}} for i in range(5)]
    assert c.code_index(products) == {}
    assert c.table_owners({"table_data": {"rows": [["M125", "1"]]}}, c.code_index(products)) == set()


def test_a_chunk_goes_to_the_one_model_it_names_and_range_text_stays():
    c = _catalogue_tables()
    index = c.code_index([{"id": "m", "name": "LINEAR MAX (KLM)"}, {"id": "t", "name": "LINEAR COMFORT (KLT)"}])
    assert c.chunk_owner("KLT 700.450 heat output", index) == "t"
    assert c.chunk_owner("KLM and KLT share a bracket", index) is None
    assert c.chunk_owner("no code at all", index) is None


def test_a_page_beside_the_series_pages_naming_no_product_is_a_web_only_candidate():
    sm = _load("site_manifest_under_test", "app/services/discovery/site_manifest.py")
    urls = ["https://k.com/koralux-linear-max", "https://k.com/koralux-rondo-booster", "https://k.com/news/koralux-award", "https://k.com/contact"]
    found = sm.series_only_on_site(urls, ["https://k.com/koralux-linear-max"], ["KORALUX LINEAR MAX"])
    assert found == ["https://k.com/koralux-rondo-booster"]
