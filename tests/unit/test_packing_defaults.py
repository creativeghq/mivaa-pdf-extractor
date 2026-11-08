import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("_pd", ROOT / "app/services/discovery/packing_defaults.py")
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)


def product(pid, *rows):
    return {"id": pid, "name": pid, "metadata": {"packaging": {"per_variant": list(rows)}}}


IZIDA_60X120 = {"format": "60x120", "m2_per_box": 1.44, "pieces_per_box": 2, "boxes_per_pallet": 36, "m2_per_pallet": 51.84, "weight_per_pallet_kg": 920}


def test_a_size_the_brand_packs_one_way_fills_the_gap_and_is_marked_estimated():
    pool = pd.sibling_pool([product(f"s{i}", dict(IZIDA_60X120)) for i in range(5)])
    out = pd.plan_product(product("aurora", {"format": "60x120", "m2_per_box": 1.44}), [], pool)
    row = out[0]
    assert (row["pieces_per_box"], row["boxes_per_pallet"], row["weight_per_pallet_kg"], row["m2_per_pallet"]) == (2, 36, 920, 51.84)
    assert row["m2_per_box"] == 1.44 and "m2_per_box" not in row["estimated"]["fields"]
    assert set(row["estimated"]["fields"]) == {"pieces_per_box", "boxes_per_pallet", "weight_per_pallet_kg", "m2_per_pallet", "weight_per_box_kg"}
    assert row["weight_per_box_kg"] == 25.6
    assert "not_available" not in row


def test_disagreeing_siblings_fill_nothing():
    pool = pd.sibling_pool([
        product("a", {"format": "30x60", "m2_per_box": 1.44, "boxes_per_pallet": 36}),
        product("b", {"format": "30x60", "m2_per_box": 1.44, "boxes_per_pallet": 24}),
    ])
    row = pd.plan_product(product("c", {"format": "30x60", "m2_per_box": 1.44}), [], pool)[0]
    assert "boxes_per_pallet" not in row
    assert "boxes_per_pallet" in row["not_available"]


def test_the_stated_box_picks_the_matching_siblings_under_a_rounded_name():
    pool = pd.sibling_pool(
        [product(f"g{i}", {"format": "60.6x60.6", "m2_per_box": 1.84, "pieces_per_box": 5, "boxes_per_pallet": 30, "weight_per_pallet_kg": 950}) for i in range(3)]
        + [product(f"h{i}", {"format": "60x60", "m2_per_box": 1.8, "pieces_per_box": 5, "boxes_per_pallet": 30, "weight_per_pallet_kg": 1000}) for i in range(4)])
    row = pd.plan_product(product("cemento", {"format": "60x60", "m2_per_box": 1.84}), [], pool)[0]
    assert row["weight_per_pallet_kg"] == 950


def test_geometry_derives_the_box_and_never_invents_a_piece_count():
    wood = pd.plan_product(product("borgeto", {"format": "15x60", "pieces_per_box": 10}), [], [])[0]
    assert wood["m2_per_box"] == 0.9 and wood["estimated"]["fields"] == ["m2_per_box"]
    assert pd.plan_product(product("odd", {"format": "20x120", "m2_per_box": 1.31}), [], []) is None


def test_an_estimate_is_never_a_source_and_a_stated_value_is_never_replaced():
    guessed = dict(IZIDA_60X120, estimated={"fields": ["boxes_per_pallet"]})
    pool = pd.sibling_pool([product("a", guessed), product("b", dict(guessed))])
    row = pd.plan_product(product("c", {"format": "60x120", "m2_per_box": 1.44, "weight_per_pallet_kg": 900}), [], pool)[0]
    assert "boxes_per_pallet" not in row
    assert row["weight_per_pallet_kg"] == 900


def test_a_format_with_no_row_gets_one_and_a_complete_product_is_left_alone():
    pool = pd.sibling_pool([product(f"s{i}", dict(IZIDA_60X120)) for i in range(3)])
    assert pd.plan_product(product("new"), ["60x120"], pool)[0]["boxes_per_pallet"] == 36
    assert pd.plan_product(product("done", dict(IZIDA_60X120, weight_per_box_kg=25.6)), ["60x120"], pool) is None


def test_flat_packing_on_a_one_size_product_is_evidence_and_is_filled_in_place():
    flat = {"id": "keros", "metadata": {"packaging": {"m2_per_box": "1,44", "pieces_per_box": 3}}}
    assert pd.packing_rows(flat, ["40x120"]) == [{"format": "40x120", "m2_per_box": 1.44, "pieces_per_box": 3}]
    assert pd.packing_rows(flat, ["40x120", "30x90"]) == []
    pool = pd.sibling_pool([dict(flat, id=f"k{i}", metadata={"packaging": {"m2_per_box": 1.44, "boxes_per_pallet": 20}}) for i in range(3)],
                           lambda p: ["40x120"])
    row = pd.plan_product(flat, ["40x120"], pool)[0]
    assert row["boxes_per_pallet"] == 20 and row["pieces_per_box"] == 3


def test_one_series_is_one_vote_and_mixed_pallets_fill_no_pallet_figure():
    white = product("white", {"format": "30x60", "pallet": "108x108", "m2_per_box": 1.44, "boxes_per_pallet": 48},
                    {"format": "30x60", "pallet": "80x120", "m2_per_box": 1.44, "boxes_per_pallet": 40})
    alone = pd.plan_product(product("alma", {"format": "30x60", "pieces_per_box": 8}), [], pd.sibling_pool([white]))[0]
    assert alone["m2_per_box"] == 1.44 and "boxes_per_pallet" not in alone and "series" not in alone["estimated"]["basis"]
    pool = pd.sibling_pool([white, product("b", {"format": "30x60", "pallet": "108x108", "m2_per_box": 1.44, "boxes_per_pallet": 48})])
    row = pd.plan_product(product("alma", {"format": "30x60", "pieces_per_box": 8}), [], pool)[0]
    assert "boxes_per_pallet" not in row and row["m2_per_box"] == 1.44


def test_a_filled_box_weight_is_listed_as_estimated():
    pool = pd.sibling_pool([product(f"t{i}", {"format": "30x90", "m2_per_box": 1.35, "weight_per_box_kg": 23}) for i in range(3)])
    row = pd.plan_product(product("tebas", {"format": "30x90", "m2_per_box": 1.35}), [], pool)[0]
    assert row["weight_per_box_kg"] == 23 and "weight_per_box_kg" in row["estimated"]["fields"]


def branded(pid, brand, *rows):
    return dict(product(pid, *rows), brand_company_id=brand)


def test_a_box_printed_to_one_decimal_still_gives_its_tile_count():
    row = pd.plan_product(product("bellagio", {"format": "20x120", "m2_per_box": 1.4}), [], [])[0]
    assert row["pieces_per_box"] == 6


def test_another_brand_fills_the_box_but_one_brand_is_not_enough_for_a_pallet():
    pool = pd.sibling_pool([branded(f"i{i}", "izida", dict(IZIDA_60X120)) for i in range(3)])
    row = pd.plan_product(branded("keros", "keros", {"format": "60x120"}), [], pool)[0]
    assert (row["m2_per_box"], row["pieces_per_box"]) == (1.44, 2)
    assert "boxes_per_pallet" not in row and "other brands" in row["estimated"]["basis"]
    pool += pd.sibling_pool([branded(f"e{i}", "euro", dict(IZIDA_60X120)) for i in range(2)])
    assert pd.plan_product(branded("keros", "keros", {"format": "60x120"}), [], pool)[0]["boxes_per_pallet"] == 36


def test_pallet_weight_follows_the_brands_kg_per_m2():
    pool = pd.sibling_pool([branded(f"i{i}", "izida", {"format": "60x60", "m2_per_pallet": 54, "weight_per_pallet_kg": 950 + i}) for i in range(3)])
    row = pd.plan_product(branded("x", "izida", {"format": "15x60", "m2_per_box": 0.9, "boxes_per_pallet": 60}), [], pool)[0]
    assert row["m2_per_pallet"] == 54 and row["weight_per_pallet_kg"] == 951
    assert "kg/m²" in row["estimated"]["basis"]
