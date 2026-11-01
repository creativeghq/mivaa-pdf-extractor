import importlib.util
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("page_supplements_under_test", _ROOT / "app/services/pdf/page_supplements.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def line(text):
    return types.SimpleNamespace(text=lambda: text)


def tokens(text):
    return {t for t in text.replace(",", ".").split() if t.replace(".", "").isdigit()} | {t for t in text.split() if t.replace(",", "").isdigit()}


LINES = [line("Heading"), line("Share 33,33 %"), line("Next line"), line("far away"), line("unrelated"), line("Max 100 %")]


def test_only_lines_with_a_missing_number_and_their_neighbours_are_kept():
    assert ps.supplement_text(LINES, ["33,33"], tokens) == ["Heading", "Share 33,33 %", "Next line"]
    assert ps.supplement_text(LINES, ["100"], tokens) == ["unrelated", "Max 100 %"]


def test_a_page_with_nothing_to_carry_makes_no_chunk():
    assert ps.build_supplement("Cat", 4, [], ["1"]) is None
    chunk = ps.build_supplement("Cat", 4, ["a 198"], ["198"])
    assert chunk["metadata"]["source"] == "text_layer_supplement" and chunk["content"].startswith("Cat - page 4")


def test_a_single_owner_takes_the_chunk_and_anything_else_stays_catalogue_level():
    products = [{"id": "a", "metadata": {"page_range": [3, 4]}}, {"id": "b", "metadata": {"page_range": [4]}}, {"id": "c", "metadata": {}}]
    assert ps.owner_for(3, products) == "a" and ps.owner_for(4, products) is None and ps.owner_for(9, products) is None
