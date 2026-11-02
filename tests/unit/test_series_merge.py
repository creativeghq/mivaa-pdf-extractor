"""Discovery keeps ONE product per series: a format chapter or a later batch folds into it."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("series_merge_under_test", _ROOT / "app/services/discovery/series_merge.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)


def _p(name, pages, **meta):
    return SimpleNamespace(name=name, page_range=pages, description=None, metadata=meta or None)


def test_a_size_in_the_name_is_not_part_of_the_series():
    assert sm.series_key("Amalfi 30x60") == sm.series_key("AMALFI 33 x 33 cm") == sm.series_key("Amalfi") == "amalfi"
    assert sm.series_key("Tíber") == "tiber"
    assert sm.series_key("Wood Espiga") != sm.series_key("Wood")
    assert sm.clean_name("60x120") == "60x120"


def test_format_chapters_fold_into_one_product_with_every_page_and_size():
    first, second = _p("Amalfi 30x60", [40, 41], dimensions=["30x60"]), _p("Amalfi 33x33", [88], finishes=["Mate"])
    sm.name_series(first)
    assert (first.name, first.metadata["dimensions"]) == ("Amalfi", ["30x60"])
    sm.name_series(second)
    sm.absorb(first, second)
    assert first.page_range == [40, 41, 88]
    assert first.metadata["dimensions"] == ["30x60", "33x33"]
    assert first.metadata["finishes"] == ["Mate"]
    assert first.metadata["discovered_names"] == ["Amalfi 30x60", "Amalfi 33x33"]


def test_a_series_met_again_in_a_later_batch_keeps_its_new_pages():
    first, again = _p("Ardenza", [10, 11], dimensions=["60x120"]), _p("Ardenza", [210], dimensions=["30x90", "60x120"])
    sm.absorb(first, again)
    assert first.page_range == [10, 11, 210]
    assert first.metadata["dimensions"] == ["60x120", "30x90"]
