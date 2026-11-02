"""The same picture at two sizes in one PDF keeps the larger copy, whichever comes first."""

import types

import pytest


def test_a_thumbnail_seen_first_gives_way_to_the_full_size_copy(tmp_path):
    pytest.importorskip("imagehash")
    pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    ImageFilter = pytest.importorskip("PIL.ImageFilter")
    processor = pytest.importorskip("app.services.pdf.pdf_processor")
    import numpy as np

    rng = np.random.default_rng(7)
    full = Image.fromarray((rng.random((400, 600, 3)) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(8))
    other = Image.fromarray((rng.random((400, 600, 3)) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(8))
    paths = {}
    for name, img in (("thumb", full.resize((150, 100))), ("full", full), ("other", other)):
        paths[name] = str(tmp_path / f"{name}.png")
        img.save(paths[name])
    images = [{"path": paths[n], "filename": n, "extraction_layer": "embedded"} for n in ("thumb", "full", "other")]

    fake = types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None,
                                                              warning=lambda *a, **k: None, error=lambda *a, **k: None))
    kept = processor.PDFProcessor._deduplicate_images(fake, images, job_id="t")
    assert [i["filename"] for i in kept] == ["full", "other"]
    assert images[0]["is_duplicate"] is True and images[0]["duplicate_of"] == "full"
