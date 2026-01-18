"""
Centralized Import Helper for PDF Extraction Functions

This module provides a single source of truth for importing PDF extraction functions,
eliminating duplicate import logic scattered across pdf_processor.py and pdf_worker.py.

Usage:
    from app.services.pdf.extractor_imports import (
        extract_pdf_to_markdown,
        extract_pdf_to_markdown_with_doc,
        extract_json_and_images,
        EXTRACTOR_AVAILABLE
    )
"""

import logging

logger = logging.getLogger(__name__)

# Flag to track if extractor functions are available
EXTRACTOR_AVAILABLE = False

# Define placeholder functions for fallback
def _placeholder_extract_pdf_to_markdown(*args, **kwargs):
    raise NotImplementedError("PDF extraction functions not available. Check extractor module installation.")

def _placeholder_extract_pdf_to_markdown_with_doc(*args, **kwargs):
    raise NotImplementedError("PDF extraction functions not available. Check extractor module installation.")

def _placeholder_extract_json_and_images(*args, **kwargs):
    raise NotImplementedError("PDF image extraction functions not available. Check extractor module installation.")


# Initialize with placeholders
extract_pdf_to_markdown = _placeholder_extract_pdf_to_markdown
extract_pdf_to_markdown_with_doc = _placeholder_extract_pdf_to_markdown_with_doc
extract_json_and_images = _placeholder_extract_json_and_images


# Try importing from various locations
def _import_extractor_functions():
    """
    Attempt to import extractor functions from available locations.

    Import order:
    1. app.core.extractor (absolute import - preferred)
    2. ..core.extractor (relative import - fallback)
    """
    global extract_pdf_to_markdown, extract_pdf_to_markdown_with_doc, extract_json_and_images, EXTRACTOR_AVAILABLE

    # Try absolute import first (most common in production)
    try:
        from app.core.extractor import (
            extract_pdf_to_markdown as _extract_md,
            extract_pdf_to_markdown_with_doc as _extract_md_with_doc,
            extract_json_and_images as _extract_json_images
        )
        extract_pdf_to_markdown = _extract_md
        extract_pdf_to_markdown_with_doc = _extract_md_with_doc
        extract_json_and_images = _extract_json_images
        EXTRACTOR_AVAILABLE = True
        logger.debug("Loaded extractor functions from app.core.extractor (absolute import)")
        return True
    except ImportError:
        pass

    # Try relative import (useful during development or testing)
    try:
        from ...core.extractor import (
            extract_pdf_to_markdown as _extract_md,
            extract_pdf_to_markdown_with_doc as _extract_md_with_doc,
            extract_json_and_images as _extract_json_images
        )
        extract_pdf_to_markdown = _extract_md
        extract_pdf_to_markdown_with_doc = _extract_md_with_doc
        extract_json_and_images = _extract_json_images
        EXTRACTOR_AVAILABLE = True
        logger.debug("Loaded extractor functions from ..core.extractor (relative import)")
        return True
    except ImportError as e:
        logger.error(f"Failed to import extractor functions: {e}")
        logger.warning("PDF extraction will not be available. Using placeholder functions.")
        return False


# Run import on module load
_import_extractor_functions()
