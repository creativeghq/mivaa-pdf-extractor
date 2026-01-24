"""
Tracking domain services - Progress tracking and job monitoring
"""

from app.services.tracking.checkpoint_recovery_service import ProcessingStage
from app.services.tracking.xml_import_stages import (
    XmlImportStage,
    XML_IMPORT_STAGE_ORDER,
    get_xml_import_progress,
    XML_IMPORT_STAGE_DESCRIPTIONS,
)
from app.services.tracking.web_scraping_stages import (
    WebScrapingStage,
    WEB_SCRAPING_STAGE_ORDER,
    get_web_scraping_progress,
    WEB_SCRAPING_STAGE_DESCRIPTIONS,
)

__all__ = [
    # PDF Processing
    "ProcessingStage",
    # XML Import
    "XmlImportStage",
    "XML_IMPORT_STAGE_ORDER",
    "get_xml_import_progress",
    "XML_IMPORT_STAGE_DESCRIPTIONS",
    # Web Scraping
    "WebScrapingStage",
    "WEB_SCRAPING_STAGE_ORDER",
    "get_web_scraping_progress",
    "WEB_SCRAPING_STAGE_DESCRIPTIONS",
]
