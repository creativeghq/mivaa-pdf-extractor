from app.services.images.image_variant_tagging import (
    NOT_A_COLOUR,
    SHOT_TYPES,
    answer_schema,
    association_update,
    colour_list,
    parse_answers,
)


def test_colour_list_prefers_colour_options_and_keeps_hex():
    meta = {"colour_options": [{"name": "Nut", "hex": "#7E624D"}, {"name": "oak", "hex": "#d7b99c"}],
            "color": '["birch"]'}
    assert colour_list(meta) == [("nut", "#7E624D"), ("oak", "#d7b99c")]


def test_colour_list_reads_metadata_color_as_json_string_or_list():
    assert colour_list({"color": '["white", "ivory"]'}) == [("white", None), ("ivory", None)]
    assert colour_list({"color": ["gris"]}) == [("gris", None)]
    assert colour_list({}) == []


def test_schema_only_offers_the_products_own_colours():
    enum = answer_schema(["nut", "oak"])["properties"]["images"]["items"]["properties"]["colour"]["enum"]
    assert enum == ["nut", "oak"] + NOT_A_COLOUR


def test_parse_answers_drops_unknown_photos_and_values():
    answer = {"images": [
        {"image": 1, "shot_type": "swatch", "colour": "nut"},
        {"image": 2, "shot_type": "scene", "colour": "walnut"},
        {"image": 9, "shot_type": "swatch", "colour": "oak"},
        {"image": 3, "shot_type": "poster", "colour": "oak"},
        {"image": 4, "shot_type": "scene", "colour": "unclear"},
    ]}
    assert parse_answers(answer, 4, ["nut", "oak"]) == {
        0: {"shot_type": "swatch", "colour": "nut"},
        3: {"shot_type": "scene", "colour": "unclear"},
    }
    assert set(SHOT_TYPES) == {"swatch", "scene", "detail", "technical", "other"}


def test_unclear_or_multiple_never_writes_a_colour():
    for verdict in NOT_A_COLOUR:
        upd = association_update(verdict)
        assert upd["variant_attributes"] is None and upd["variant_status"] == verdict
    tagged = association_update("ivory")
    assert tagged["variant_attributes"] == {"color": "ivory"} and tagged["variant_status"] == "tagged"
    assert tagged["variant_source"] == "vision"
