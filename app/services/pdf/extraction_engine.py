"""
PDF Extraction Engine Orchestrator

Manages multiple PDF extraction engines (PyMuPDF, Marker) with fallback support.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ExtractionEngine(str, Enum):
    """Available PDF extraction engines."""
    PYMUPDF = "pymupdf"
    MARKER = "marker"


class ExtractionResult:
    """Result from PDF extraction."""
    
    def __init__(
        self,
        markdown_content: str,
        metadata: Dict[str, Any],
        page_chunks: Optional[List[Dict[str, Any]]] = None,
        engine_used: str = "pymupdf",
        fallback_used: bool = False,
        extraction_time: float = 0.0
    ):
        self.markdown_content = markdown_content
        self.metadata = metadata
        self.page_chunks = page_chunks
        self.engine_used = engine_used
        self.fallback_used = fallback_used
        self.extraction_time = extraction_time
        
        # Add engine info to metadata
        self.metadata['extraction_engine'] = engine_used
        self.metadata['fallback_used'] = fallback_used
        self.metadata['extraction_time_seconds'] = extraction_time


def should_trigger_fallback(
    markdown_content: str,
    metadata: Dict[str, Any],
    error: Optional[Exception] = None
) -> bool:
    """
    Determine if fallback engine should be triggered.
    
    Triggers fallback if:
    - Extraction raised an error
    - Content is too short (< 100 chars)
    - Content quality is poor (mostly dashes/newlines)
    """
    if error:
        logger.warning(f"Fallback triggered by error: {error}")
        return True
    
    if not markdown_content or len(markdown_content.strip()) < 100:
        logger.warning(f"Fallback triggered by short content: {len(markdown_content)} chars")
        return True
    
    # Check content quality
    clean_content = markdown_content.replace('-', '').replace('\n', '').replace(' ', '').strip()
    if len(clean_content) < 50:
        logger.warning(f"Fallback triggered by low quality content: {len(clean_content)} meaningful chars")
        return True
    
    return False


async def extract_with_engine(
    pdf_path: str,
    engine: str,
    processing_options: Dict[str, Any]
) -> ExtractionResult:
    """
    Extract PDF content using specified engine.
    
    Args:
        pdf_path: Path to PDF file
        engine: Engine to use ('pymupdf', 'marker')
        processing_options: Processing configuration

    Returns:
        ExtractionResult with content and metadata
    """
    import time
    start_time = time.time()

    try:
        if engine == ExtractionEngine.PYMUPDF:
            result = await _extract_with_pymupdf(pdf_path, processing_options)
        elif engine == ExtractionEngine.MARKER:
            result = await _extract_with_marker(pdf_path, processing_options)
        else:
            raise ValueError(f"Unknown extraction engine: {engine}")
        
        extraction_time = time.time() - start_time
        result.extraction_time = extraction_time
        result.engine_used = engine
        
        logger.info(f"✅ Extraction completed with {engine} in {extraction_time:.2f}s")
        return result
        
    except Exception as e:
        extraction_time = time.time() - start_time
        logger.error(f"❌ Extraction failed with {engine} after {extraction_time:.2f}s: {e}")
        raise


async def _extract_with_pymupdf(
    pdf_path: str,
    processing_options: Dict[str, Any]
) -> ExtractionResult:
    """Extract using PyMuPDF4LLM (current implementation)."""
    from app.services.pdf.pdf_worker import execute_pdf_extraction_job
    
    logger.info("Using PyMuPDF extraction engine")
    markdown_content, metadata, page_chunks = execute_pdf_extraction_job(pdf_path, processing_options)
    
    return ExtractionResult(
        markdown_content=markdown_content,
        metadata=metadata,
        page_chunks=page_chunks,
        engine_used="pymupdf"
    )


async def _extract_with_marker(
    pdf_path: str,
    processing_options: Dict[str, Any]
) -> ExtractionResult:
    """Extract using Marker API (placeholder - to be implemented)."""
    logger.warning("Marker extraction not yet implemented, falling back to PyMuPDF")
    return await _extract_with_pymupdf(pdf_path, processing_options)


async def extract_with_fallback(
    pdf_path: str,
    primary_engine: str,
    fallback_engine: Optional[str],
    processing_options: Dict[str, Any]
) -> ExtractionResult:
    """
    Extract PDF with primary engine and optional fallback.

    Args:
        pdf_path: Path to PDF file
        primary_engine: Primary extraction engine to try first
        fallback_engine: Fallback engine if primary fails (None to disable)
        processing_options: Processing configuration

    Returns:
        ExtractionResult with content, metadata, and engine info
    """
    logger.info(f"🚀 Starting extraction with primary engine: {primary_engine}")

    # Try primary engine
    try:
        result = await extract_with_engine(pdf_path, primary_engine, processing_options)

        # Check if fallback should be triggered
        if fallback_engine and should_trigger_fallback(result.markdown_content, result.metadata):
            logger.warning(f"⚠️ Primary engine ({primary_engine}) produced low-quality output, trying fallback: {fallback_engine}")

            try:
                fallback_result = await extract_with_engine(pdf_path, fallback_engine, processing_options)
                fallback_result.fallback_used = True

                # Compare results and use better one
                if len(fallback_result.markdown_content) > len(result.markdown_content):
                    logger.info(f"✅ Fallback engine ({fallback_engine}) produced better result")
                    return fallback_result
                else:
                    logger.info(f"✅ Primary engine ({primary_engine}) result was better, keeping it")
                    return result

            except Exception as fallback_error:
                logger.error(f"❌ Fallback engine ({fallback_engine}) also failed: {fallback_error}")
                logger.info(f"↩️ Returning primary engine result despite low quality")
                return result

        logger.info(f"✅ Primary engine ({primary_engine}) succeeded without fallback")
        return result

    except Exception as primary_error:
        logger.error(f"❌ Primary engine ({primary_engine}) failed: {primary_error}")

        # Try fallback if configured
        if fallback_engine:
            logger.info(f"🔄 Attempting fallback engine: {fallback_engine}")
            try:
                fallback_result = await extract_with_engine(pdf_path, fallback_engine, processing_options)
                fallback_result.fallback_used = True
                logger.info(f"✅ Fallback engine ({fallback_engine}) succeeded")
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"❌ Fallback engine ({fallback_engine}) also failed: {fallback_error}")
                raise Exception(
                    f"Both primary ({primary_engine}) and fallback ({fallback_engine}) engines failed. "
                    f"Primary error: {primary_error}. Fallback error: {fallback_error}"
                )
        else:
            # No fallback configured, re-raise primary error
            raise

