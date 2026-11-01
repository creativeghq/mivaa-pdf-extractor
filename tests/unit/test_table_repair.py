import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
Image = pytest.importorskip("PIL.Image")


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.services", "app.services.pdf"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
oc = _load("app.services.pdf.ocr_completeness", "app/services/pdf/ocr_completeness.py")
_load("app.services.pdf.text_layer_extractor", "app/services/pdf/text_layer_extractor.py")
tr = _load("table_repair_under_test", "app/services/pdf/table_repair.py")

LINES = 40
WIDTH = 300
HEIGHT = LINES * 12


def page_image():
    image = Image.new("L", (WIDTH, HEIGHT), 255)
    for i in range(LINES):
        for y in range(i * 12, i * 12 + 8):
            for x in range(20, 280):
                image.putpixel((x, y), 0)
    return image


def table(rows, closed=True):
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return "<table>" + body + ("</table>" if closed else "")


def element(html):
    return {
        "region_type": "TABLE",
        "bbox": {"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT, "page": 1},
        "text_content": html,
        "metadata": {"html": html},
    }


class Region:
    region_type = "TABLE"

    def __init__(self, content):
        self.content = content


class Manager:
    def __init__(self, bands=None, boom=False):
        self.calls, self.bands, self.boom = 0, list(bands or []), boom

    def run_structural_pass(self, image, caller, page, job, doc):
        self.calls += 1
        if self.boom:
            raise RuntimeError("endpoint down")
        return {"regions": [Region(self.bands.pop(0))]}


def band_tables(n_bands=3, rows_each=12):
    return [table([[f"b{b}r{r}", str(100 + r), str(200 + r)] for r in range(rows_each)]) for b in range(n_bands)]


def run(html, manager, repair=True, budget=None, reference=None):
    el = element(html)
    report = tr.verify_ocr_tables(manager, page_image(), [el], reference, 1, None, "doc", 250 / 72.0, repair, budget)
    return el, report


TRUNCATED_HTML = table([["h1", "h2", "h3"]] + [[f"r{i}", "1", "2"] for i in range(7)], closed=False)


def test_a_truncated_table_is_rebuilt_from_bands_and_marked_as_recovered():
    manager = Manager(band_tables())
    el, report = run(TRUNCATED_HTML, manager)
    assert manager.calls >= 2 and report["recovered"] == 1 and report["repair_attempts"] == 1
    assert el["metadata"]["recovered_by"] == "band_rerun"
    assert len(oc.table_rows(el["metadata"]["html"])) >= 24
    assert el["metadata"]["needs_review"] is False and el["metadata"]["completeness"] == oc.COMPLETE


def test_a_low_line_ratio_alone_never_triggers_a_paid_repair():
    html = table([["h", "x"]] + [[str(100 + i), "v"] for i in range(9)])
    manager = Manager(band_tables())
    el, report = run(html, manager)
    assert manager.calls == 0 and report["repair_attempts"] == 0
    assert el["metadata"]["needs_review"] is True and el["metadata"]["completeness"] == oc.SUSPECT
    assert el["metadata"]["html"] == html


def test_a_repair_that_gains_no_rows_is_rejected_and_the_original_stays_flagged():
    manager = Manager([table([["a", "1", "2"]])] * 3)
    el, report = run(TRUNCATED_HTML, manager)
    assert report["recovered"] == 0 and el["metadata"]["html"] == TRUNCATED_HTML
    assert el["metadata"]["needs_review"] is True and "recovered_by" not in el["metadata"]


def test_an_endpoint_error_during_repair_is_recorded_and_the_table_stays_flagged():
    el, report = run(TRUNCATED_HTML, Manager(boom=True))
    assert "endpoint down" in el["metadata"]["repair_error"]
    assert el["metadata"]["needs_review"] is True and report["truncated"] == 1


def test_mismatched_band_columns_are_not_stitched():
    bands = [table([["a", "1"]] * 12), table([["a", "1", "2", "3", "4", "5"]] * 12), table([["a", "1"]] * 12)]
    el, report = run(TRUNCATED_HTML, Manager(bands))
    assert report["recovered"] == 0 and el["metadata"]["html"] == TRUNCATED_HTML


def test_the_per_document_budget_stops_repairs():
    manager = Manager(band_tables())
    el, report = run(TRUNCATED_HTML, manager, budget={"left": 0})
    assert manager.calls == 0 and report["repair_attempts"] == 0 and el["metadata"]["needs_review"] is True


def test_repair_can_be_switched_off_while_the_check_still_runs():
    manager = Manager(band_tables())
    el, report = run(TRUNCATED_HTML, manager, repair=False)
    assert manager.calls == 0 and el["metadata"]["completeness"] == oc.TRUNCATED


def test_numbers_the_text_layer_has_and_the_table_lacks_make_it_truncated_and_a_better_repair_is_adopted():
    full = [["h1", "h2", "h3"]] + [[f"r{i}", str(100 + i), str(500 + i)] for i in range(30)]
    reference_words = [(10.0, 10.0 + i, 20.0, 15.0 + i, str(100 + i)) for i in range(30)] + [(10.0, 50.0 + i, 20.0, 55.0 + i, str(500 + i)) for i in range(30)]
    reference = {"words": reference_words, "origin": (0.0, 0.0)}
    partial = table(full[:12])
    bands = [table(full[0:14]), table(full[14:24]), table(full[24:])]
    el, report = run(partial, Manager(bands), reference=reference)
    assert report["recovered"] == 1
    assert el["metadata"]["completeness"] != oc.TRUNCATED and el["metadata"]["missing_reference_share"] == 0.0


def test_a_complete_table_is_left_alone():
    rows = [["h", "a"]] + [[str(100 + i), "v"] for i in range(40)]
    manager = Manager(band_tables())
    el, report = run(table(rows), manager)
    assert manager.calls == 0 and el["metadata"]["needs_review"] is False and report["complete"] == 1
