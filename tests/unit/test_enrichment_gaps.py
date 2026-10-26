from types import SimpleNamespace

from app.services.products.enrichment_gaps import (
    GAPS_NO_SOURCE, GAPS_WITH_SOURCE, NOTHING_MISSING, empty_fields, source_for, summarize, verdict,
)


def _spec(name, role="descriptive", destination="metadata", sensitivity="public"):
    return SimpleNamespace(
        name=name, destination=destination, sensitivity=sensitivity,
        role_in=lambda _cat, _role=role: _role,
    )


def _empty(v):
    return v in (None, "", [], {})


def _gaps(flat, specs):
    return empty_fields(flat, specs, "tiles", _empty, lambda n: n.startswith("cost"))


def test_only_empty_descriptive_public_metadata_fields_are_gaps():
    specs = [
        _spec("water_absorption"),
        _spec("slip_rating"),
        _spec("color", role="identity"),
        _spec("ean", destination="column"),
        _spec("margin_note", sensitivity="internal"),
        _spec("cost_basis"),
    ]
    assert _gaps({"slip_rating": "R10"}, specs) == ["water_absorption"]


def test_a_filled_field_is_never_a_gap():
    assert _gaps({"water_absorption": "0.5%"}, [_spec("water_absorption")]) == []


def test_source_is_a_held_https_url_only():
    assert source_for("https://a.example/p", "https://b.example") == {"kind": "product_url", "url": "https://a.example/p"}
    assert source_for(None, " https://b.example ") == {"kind": "brand_site", "url": "https://b.example"}
    assert source_for("http://a.example/p", "") is None
    assert source_for(None, None) is None


def test_verdicts_separate_no_gaps_from_no_source():
    src = {"kind": "product_url", "url": "https://a.example"}
    assert verdict([], None) == NOTHING_MISSING
    assert verdict(["x"], src) == GAPS_WITH_SOURCE
    assert verdict(["x"], None) == GAPS_NO_SOURCE


def test_summary_counts_every_verdict():
    rows = [{"verdict": GAPS_NO_SOURCE}, {"verdict": NOTHING_MISSING}, {"verdict": NOTHING_MISSING}]
    assert summarize(rows)[NOTHING_MISSING] == 2 and summarize(rows)["products"] == 3
