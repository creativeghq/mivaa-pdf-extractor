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


for _pkg in ("app", "app.services", "app.services.pdf", "app.services.utilities"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_load("app.services.pdf.ocr_completeness", "app/services/pdf/ocr_completeness.py")
_load("app.services.pdf.ingestion_audit", "app/services/pdf/ingestion_audit.py")
ts = _load("text_table_structurer_under_test", "app/services/pdf/text_table_structurer.py")


def words():
    out = []
    y = 100.0
    out.append((30.0, y, 60.0, y + 10, "60x120"))
    y += 20
    for i in range(10):
        code = f"80{4000 + i:05d}"
        out.append((238.0, y, 280.0, y + 10, code))
        out.append((286.0, y, 320.0, y + 10, "WHITE" if i % 2 == 0 else "GREY"))
        y += 14
    y += 10
    out.append((637.0, y, 690.0, y + 10, "pz/box"))
    out.append((793.0, y, 800.0, y + 10, "2"))
    return out


LINES = ts.lines_from_words(words())
SIG = ts.page_signature(LINES)
INVENTORY = ts.page_inventory(LINES, SIG)


def index_of(code):
    return next(line.index for line in LINES if code in line.text())


def good_rows():
    rows = []
    for code in sorted(INVENTORY):
        i = index_of(code)
        colour = next(t for _, t in LINES[i].items if t in ("WHITE", "GREY"))
        rows.append({"code": code, "lines": [0, i], "attributes": [{"name": "Colour", "value": colour}, {"name": "Size", "value": "60x120"}]})
    return rows


def test_words_on_one_baseline_make_one_line_in_x_order():
    assert LINES[1].text() == "8004000 WHITE" and LINES[0].text() == "60x120"
    assert SIG == "9999999" and len(INVENTORY) == 10


def test_a_page_with_no_repeated_code_shape_is_not_structured():
    few = ts.lines_from_words([(0.0, 0.0, 10.0, 10.0, "8004000"), (0.0, 20.0, 10.0, 30.0, "80004001")])
    assert ts.page_signature(few) is None


def test_the_model_sees_numbered_positioned_lines():
    text = ts.render_for_model(LINES)
    assert text.splitlines()[1].startswith("[1] y=") and "x238: 8004000" in text


def test_rows_that_cover_every_code_with_values_from_cited_lines_are_accepted():
    verdict = ts.verify(good_rows(), LINES, INVENTORY)
    assert verdict.accepted and verdict.reasons == []


def test_a_missing_foreign_or_duplicated_code_rejects_the_page():
    rows = good_rows()
    missing = ts.verify(rows[1:], LINES, INVENTORY)
    assert not missing.accepted and missing.missing_codes == [sorted(INVENTORY)[0]] and "codes_missing" in missing.reasons
    foreign = ts.verify(rows + [{"code": "8009999", "lines": [1], "attributes": []}], LINES, INVENTORY)
    assert "codes_not_on_page" in foreign.reasons and foreign.foreign_codes == ["8009999"]
    assert "codes_duplicated" in ts.verify(rows + [rows[0]], LINES, INVENTORY).reasons


def test_a_value_that_is_not_on_a_cited_line_is_rejected_never_stored():
    rows = good_rows()
    rows[0]["attributes"].append({"name": "Thickness", "value": "9,5 mm"})
    verdict = ts.verify(rows, LINES, INVENTORY)
    assert not verdict.accepted and verdict.unsupported[0]["value"] == "9,5 mm"


def test_a_code_that_is_not_on_any_line_it_cites_is_rejected():
    rows = good_rows()
    rows[0]["lines"] = [0]
    assert not ts.verify(rows, LINES, INVENTORY).accepted


def test_rows_become_a_table_with_the_union_of_attribute_names_in_first_seen_order():
    headers, body = ts.table_from_rows([
        {"code": "A", "lines": [], "attributes": [{"name": "Colour", "value": "WHITE"}]},
        {"code": "B", "lines": [], "attributes": [{"name": "Size", "value": "60x120"}, {"name": "Colour", "value": "GREY"}]},
    ])
    assert headers == ["Code", "Colour", "Size"] and body == [["A", "WHITE", ""], ["B", "GREY", "60x120"]]


def test_an_end_to_end_call_loads_the_prompt_from_the_database_and_never_stores_a_rejected_page():
    async def load_prompt(prompt_type, category, stage=None, **_):
        assert (prompt_type, category, stage) == ("extraction", "unruled_table_structuring", "table_structuring")
        return "system prompt from the database"

    registry = types.ModuleType("app.services.utilities.prompt_registry")
    registry.load_prompt = load_prompt
    sys.modules["app.services.utilities.prompt_registry"] = registry
    seen = {}

    async def fake_call(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(data={"rows": good_rows()[:-1]})

    out = asyncio.run(ts.structure_page(LINES, heading="ELYSIUM", workspace_id="w", caller=fake_call))
    assert out["status"] == "rejected" and "codes_missing" in out["verdict"]["reasons"]
    assert seen["system"] == "system prompt from the database" and "DATA" in seen["messages"][0]["content"]
    assert seen["output_config"]["format"]["schema"] is ts.OUTPUT_SCHEMA or seen["output_config"]["format"]["schema"] == ts.OUTPUT_SCHEMA
    accepted = asyncio.run(ts.structure_page(LINES, heading="", workspace_id="w", caller=lambda **k: _ok()))
    assert accepted["status"] == "accepted" and len(accepted["rows"]) == 10


async def _ok():
    return types.SimpleNamespace(data={"rows": good_rows()})


def test_pack_table_rows_are_read_deterministically_and_handed_to_the_model():
    lines = ts.lines_from_words([
        (30.0, 10.0, 60.0, 20.0, "pz/box"), (100.0, 10.0, 130.0, 20.0, "m2/box"),
        (30.0, 40.0, 60.0, 50.0, "60x120"), (100.0, 40.0, 120.0, 50.0, "2"), (140.0, 40.0, 160.0, 50.0, "1,44"),
        (180.0, 40.0, 200.0, 50.0, "36"), (220.0, 40.0, 240.0, 50.0, "51,84"),
    ])
    rows = ts.pack_rows(lines)
    assert len(rows) == 1 and rows[0]["size"] == "60x120" and rows[0]["numbers"] == ["2", "1,44", "36", "51,84"]
    assert rows[0]["header_line"] == 0 and "60x120 -> 2 | 1,44 | 36 | 51,84" in ts.pack_hint(lines)
    assert ts.pack_hint(ts.lines_from_words([(0.0, 0.0, 10.0, 10.0, "WHITE")])) == ""
