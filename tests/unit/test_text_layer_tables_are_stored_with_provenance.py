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


table_extraction = _load("table_extraction_provenance_under_test", "app/services/pdf/table_extraction.py")
for _pkg in ("app", "app.services", "app.services.pdf", "app.services.metadata"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
sys.modules["app.services.pdf.table_extraction"] = table_extraction
metadata_extractor = _load("table_metadata_provenance_under_test", "app/services/metadata/table_metadata_extractor.py")

GRID = [["Type", "Q"], ["K10", "893"], ["K11", "1227"]]


def convert(region_metadata=None):
    return table_extraction.TableExtractor()._convert_table_to_dict(GRID, page_number=26, confidence=0.99, region_metadata=region_metadata)


def test_an_ocr_table_keeps_its_old_shape():
    record = convert()
    assert record["extractor"] == "paddleocr-vl" and record["metadata"] == {}


def test_a_text_layer_table_is_named_titled_and_sourced():
    record = convert({"source": "text-layer", "title": "BASIC TECHNICAL PARAMETERS"})
    assert record["extractor"] == "pymupdf-text"
    assert record["metadata"] == {"source": "PDF text layer", "title": "BASIC TECHNICAL PARAMETERS"}


def test_a_table_flagged_for_review_carries_the_reasons_into_the_database_row():
    record = convert({"source": "text-layer", "needs_review": True, "completeness_reasons": ["missing_reference_numbers"]})
    assert record["metadata"]["needs_review"] is True
    assert record["metadata"]["review_reasons"] == ["missing_reference_numbers"]


def test_a_band_recovered_table_says_so():
    assert convert({"recovered_by": "band_rerun"})["metadata"]["recovered_by"] == "band_rerun"


def test_a_flagged_table_never_enriches_the_product():
    import asyncio

    def tables(flag):
        return [{
            "table_type": "dimensions", "headers": ["Formato", "Spessore"],
            "table_data": {"rows": [["60x120 cm", "9 mm"]]}, "page_number": 3,
            "metadata": {"needs_review": True} if flag else {},
        }]

    class Query:
        def __init__(self, data):
            self.data = data

        def select(self, *_):
            return self

        def eq(self, *_):
            return self

        def execute(self):
            return types.SimpleNamespace(data=self.data)

    def extract(flag):
        client = types.SimpleNamespace(table=lambda *_: Query(tables(flag)))
        extractor = metadata_extractor.TableMetadataExtractor(types.SimpleNamespace(client=client))
        return asyncio.run(extractor.extract_metadata_from_tables("p1"))

    assert extract(False).get("dimensions") or extract(False).get("available_sizes")
    flagged = extract(True)
    assert not flagged.get("dimensions") and not flagged.get("available_sizes")
