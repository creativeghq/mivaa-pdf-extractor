import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ocr_completeness_under_test", _ROOT / "app/services/pdf/ocr_completeness.py")
oc = importlib.util.module_from_spec(_spec)
sys.modules["ocr_completeness_under_test"] = oc
_spec.loader.exec_module(oc)


def table(rows):
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


def test_rows_are_read_from_html_without_tags():
    assert oc.table_rows("<table><tr><td>a</td><td><b>1</b></td></tr><tr><td></td><td>2</td></tr></table>") == [["a", "1"], ["", "2"]]


def test_numbers_ignore_page_furniture_and_never_glue_neighbouring_cells():
    assert oc.numeric_tokens("Page 5 | 1 200 | 970 | 1,44 | 12") == {"200", "970", "1,44"}
    assert oc.numeric_tokens("<td>1017</td><td>936</td><td>882</td>") == {"1017", "936", "882"}
    assert oc.numeric_tokens("60 120") == oc.numeric_tokens("60 120")


def test_a_closed_clean_table_is_complete():
    rows = [["H", "a", "b"]] + [[f"r{i}", str(100 + i), str(200 + i)] for i in range(10)]
    a = oc.assess_table(table(rows))
    assert a.status == oc.COMPLETE and a.rows == 11 and a.reasons == []


def test_an_unclosed_table_is_truncated():
    a = oc.assess_table("<table><tr><td>1</td><td>2</td></tr><tr><td>3</td>")
    assert a.status == oc.TRUNCATED and "unclosed_table" in a.reasons


def test_numbers_missing_against_the_text_layer_make_it_truncated():
    rows = [["h1", "h2"]] + [[str(100 + i), str(500 + i)] for i in range(20)]
    reference = {str(100 + i) for i in range(20)} | {str(500 + i) for i in range(20)} | {"970", "1182", "971"}
    a = oc.assess_table(table(rows), reference_numbers=reference)
    assert a.status == oc.TRUNCATED and a.reasons == ["missing_reference_numbers"]
    assert a.missing_share == pytest.approx(3 / 43)


def test_a_small_missing_share_is_only_a_suspicion():
    rows = [["h"]] + [[str(100 + i)] for i in range(60)]
    reference = {str(100 + i) for i in range(60)} | {"999", "998"}
    a = oc.assess_table(table(rows), reference_numbers=reference)
    assert a.status == oc.SUSPECT


def test_the_line_ratio_alone_only_ever_raises_a_suspicion():
    rows = [["h", "x"]] + [[str(100 + i), "v"] for i in range(9)]
    wrapped = oc.assess_table(table(rows), visible_lines=30)
    assert wrapped.status == oc.SUSPECT and wrapped.reasons == ["rows_vs_visible_lines"]
    assert not set(wrapped.reasons) & oc.HARD_REASONS
    assert oc.assess_table(table(rows), visible_lines=12).status == oc.SUSPECT
    assert oc.assess_table(table(rows), visible_lines=10).status == oc.COMPLETE


def test_few_visible_lines_never_judge_a_table():
    assert oc.assess_table(table([["a", "b"], ["1", "2"]]), visible_lines=3).status == oc.COMPLETE


def test_a_repeated_row_loop_is_truncated():
    rows = [["h", "x"], ["1", "2"]] + [["7", "7"]] * 4
    assert oc.assess_table(table(rows)).status == oc.TRUNCATED


def test_an_empty_last_row_in_a_full_table_is_suspect():
    rows = [["h", "a", "b", "c", "d"]] + [[str(i), "1", "2", "3", "4"] for i in range(8)] + [["9", "", "", "", ""]]
    a = oc.assess_table(table(rows))
    assert a.status == oc.SUSPECT and "trailing_empty_row" in a.reasons


def test_assessment_serialises_to_metadata():
    meta = oc.assess_table(table([["a", "b"], ["1", "2"]])).as_metadata()
    assert meta["completeness"] == oc.COMPLETE and meta["rows"] == 2 and meta["line_ratio"] is None


def test_text_lines_are_counted_from_the_ink_profile():
    profile = ([0.0] * 4 + [0.2] * 8) * 5 + [0.0] * 4
    assert oc.count_text_lines(profile) == 5
    assert oc.count_text_lines([0.2] * 3) == 0
    assert oc.count_text_lines([]) == 0


def test_bands_cut_in_the_gaps_between_lines():
    profile = ([0.0] * 4 + [0.2] * 8) * 40
    bands = oc.text_line_bands(profile)
    cuts = oc.plan_bands(bands, len(profile), max_lines_per_band=18)
    assert len(cuts) == 3 and cuts[0][0] == 0 and cuts[-1][1] == len(profile)
    for (_, end), (start, _) in zip(cuts, cuts[1:]):
        assert end == start and profile[end] == 0.0


def test_a_cut_prefers_the_widest_gap_near_the_nominal_position():
    bands, y = [], 0
    for i in range(40):
        bands.append((y, y + 8))
        y += 8 + (30 if i == 15 else 4)
    cuts = oc.plan_bands(bands, y)
    assert cuts[0][1] == (bands[15][1] + bands[16][0]) // 2


def test_a_short_table_is_one_band():
    bands = oc.text_line_bands(([0.0] * 4 + [0.2] * 8) * 5)
    assert oc.plan_bands(bands, 60) == [(0, 60)]


def test_stitching_joins_bands_and_refuses_a_column_mismatch():
    a = table([["h1", "h2"], ["1", "2"]])
    b = table([["3", "4"], ["5", "6"]])
    stitched, reasons = oc.stitch_bands([a, b])
    assert reasons == [] and len(oc.table_rows(stitched)) == 4
    assert oc.stitch_bands([a, table([["3", "4", "5"]])])[1] == []
    wide = table([["3", "4", "5", "6", "7"]])
    assert oc.stitch_bands([a, wide]) == (None, ["band_column_mismatch"])
    assert oc.stitch_bands([a, ""])[0] is None


def test_the_ink_profile_counts_dark_rows_of_a_real_image():
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("L", (200, 100), 255)
    for y0 in (10, 40, 70):
        for y in range(y0, y0 + 10):
            for x in range(20, 120):
                image.putpixel((x, y), 0)
    profile = oc.ink_profile_of(image)
    assert len(profile) == 100 and oc.count_text_lines(profile) == 3


def test_a_decimal_is_the_same_number_whatever_its_trailing_zeros():
    assert oc.numeric_tokens("1080,00 | 1080,0 | 1,440 | 12,50") == {"1080", "1,44", "12,5"}
    assert oc.numeric_tokens("<td>1080,0</td>") == oc.numeric_tokens("1080,00")
