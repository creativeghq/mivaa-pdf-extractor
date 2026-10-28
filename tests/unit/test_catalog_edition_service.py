from app.services.catalog_edition_service import compute_field_diffs


def test_identical_product_has_no_diffs():
    old = {"name": "A", "description": "d", "metadata": {"warranty": "5 years"}}
    new = {"name": "A", "description": "d", "metadata": {"warranty": "5 years"}}
    assert compute_field_diffs(old, new) == {}


def test_changed_description_and_metadata_value_are_reported_with_both_sides():
    old = {"name": "A", "description": "old", "metadata": {"warranty": {"value": "2 years"}}}
    new = {"name": "A", "description": "new", "metadata": {"warranty": "5 years"}}
    assert compute_field_diffs(old, new) == {
        "description": {"old": "old", "new": "new"},
        "metadata.warranty": {"old": "2 years", "new": "5 years"},
    }


def test_a_key_the_new_extraction_lacks_is_not_a_change():
    old = {"name": "A", "description": "d", "metadata": {"warranty": "5 years"}}
    new = {"name": "A", "description": None, "metadata": {}}
    assert compute_field_diffs(old, new) == {}
