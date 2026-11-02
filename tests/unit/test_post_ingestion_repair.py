"""Post-ingestion repair: a page cropped once per product collapses to ONE image row, and only that."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("post_ingestion_repair_under_test", _ROOT / "app/services/pdf/post_ingestion_repair.py")
repair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair)


def _row(id_, page=4, phash="88136ff064f37685", w=900, h=600, embedded=True, created="2026-11-01T10:00", excluded=False):
    m = {"layer_info": {"perceptual_hash": phash}, "width": w, "height": h}
    if excluded:
        m["excluded_from_products"] = {"reason": "x"}
    return {"id": id_, "page_number": page, "metadata": m, "has_slig_embedding": embedded, "created_at": created}


def test_the_same_crop_cut_for_two_products_is_one_group_with_the_embedded_oldest_first():
    rows = [
        _row("b", created="2026-11-01T11:00"),
        _row("a", created="2026-11-01T10:00"),
        _row("c", embedded=False, created="2026-11-01T09:00"),
    ]
    groups = repair.duplicate_image_groups(rows)
    assert [[r["id"] for r in g] for g in groups] == [["a", "b", "c"]]


def test_a_different_page_size_or_hash_is_not_a_duplicate_and_retired_rows_are_ignored():
    rows = [
        _row("a"),
        _row("b", page=5),
        _row("c", w=901),
        _row("d", phash="ffff000011112222"),
        _row("e", excluded=True),
        {"id": "f", "page_number": 4, "metadata": {"width": 900, "height": 600}, "has_slig_embedding": True},
    ]
    assert repair.duplicate_image_groups(rows) == []


def test_crops_with_no_hash_cut_from_the_same_spot_are_one_group_and_a_hash_match_joins_it():
    spot = [0.144, 0.796, 0.254, 0.128]
    rows = [dict(_row(i, phash=None), bbox=spot, extraction_layer="region_crop") for i in ("a", "b", "c")]
    rows.append(dict(_row("d", phash="ff00"), bbox=[0.6, 0.796, 0.252, 0.128], extraction_layer="region_crop"))
    rows.append(dict(_row("e", phash="ff00"), bbox=None, extraction_layer="region_crop"))
    groups = sorted(sorted(r["id"] for r in g) for g in repair.duplicate_image_groups(rows))
    assert groups == [["a", "b", "c"], ["d", "e"]]
