"""The disabled per-product KB generator is gone; the catalogue-wide extractors that really create KB docs stay wired."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"


def test_the_deprecated_service_and_every_caller_are_gone():
    assert not (_APP / "services/knowledge/auto_kb_document_service.py").exists()
    offenders = [
        str(p.relative_to(_ROOT)) for p in _APP.rglob("*.py")
        if "auto_kb_document_service" in p.read_text(encoding="utf-8") or "AutoKBDocumentService" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


def test_the_catalogue_wide_knowledge_extractors_are_still_called_by_enrichment():
    src = (_APP / "api/pdf_processing/stage_4_products.py").read_text(encoding="utf-8")
    assert "extract_catalog_knowledge_from_pdf" in src
    assert "catalog_kb_docs_created" in src
    legend = (_APP / "services/knowledge/catalog_legend_extractor_v2.py").read_text(encoding="utf-8")
    assert 'table("kb_docs")' in legend or "kb_docs" in legend
