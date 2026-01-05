"""
PDF Extraction Engine Orchestrator

Manages multiple PDF extraction engines (PyMuPDF, Vision AI, Marker) with fallback support.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ExtractionEngine(str, Enum):
    """Available PDF extraction engines."""
    PYMUPDF = "pymupdf"
    VISION = "vision"
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
        engine: Engine to use ('pymupdf', 'vision', 'marker')
        processing_options: Processing configuration
        
    Returns:
        ExtractionResult with content and metadata
    """
    import time
    start_time = time.time()
    
    try:
        if engine == ExtractionEngine.PYMUPDF:
            result = await _extract_with_pymupdf(pdf_path, processing_options)
        elif engine == ExtractionEngine.VISION:
            result = await _extract_with_vision(pdf_path, processing_options)
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


async def _extract_with_vision(
    pdf_path: str,
    processing_options: Dict[str, Any]
) -> ExtractionResult:
    """
    Extract PDF text using Vision AI (Claude/GPT/Together).

    Renders each page as image and uses vision model to extract text.
    More accurate for scanned documents and complex layouts.
    """
    import fitz  # PyMuPDF for rendering
    from app.services.core.ai_client_service import AIClientService
    from app.config import get_settings

    settings = get_settings()
    ai_client = AIClientService()

    # Get vision provider from settings
    provider = settings.vision_guided_provider  # "anthropic", "openai", or "together"
    model = settings.vision_guided_model

    logger.info(f"Using Vision AI extraction with {provider}/{model}")

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # Get page list or process all pages
        page_list = processing_options.get('page_list')
        if page_list:
            pages_to_process = [p - 1 for p in page_list if 0 < p <= total_pages]  # Convert to 0-indexed
        else:
            pages_to_process = list(range(total_pages))

        logger.info(f"Processing {len(pages_to_process)} pages with Vision AI")

        all_page_texts = []
        page_chunks = []

        for page_idx in pages_to_process:
            try:
                # Render page to image
                page = doc.load_page(page_idx)
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes('jpeg')

                # Convert to base64
                import base64
                img_base64 = base64.b64encode(img_data).decode('utf-8')

                # Extract text using vision model
                page_text = await _call_vision_for_text(
                    img_base64,
                    page_idx + 1,
                    provider,
                    model,
                    ai_client
                )

                all_page_texts.append(page_text)

                # Create page chunk
                page_chunks.append({
                    'page': page_idx,
                    'text': page_text,
                    'metadata': {
                        'extraction_method': 'vision_ai',
                        'provider': provider,
                        'model': model
                    }
                })

                logger.info(f"✅ Page {page_idx + 1}/{total_pages}: {len(page_text)} chars extracted")

            except Exception as page_error:
                logger.error(f"❌ Failed to process page {page_idx + 1}: {page_error}")
                all_page_texts.append(f"[Error extracting page {page_idx + 1}]")

        doc.close()

        # Combine all pages
        markdown_content = "\n\n-----\n\n".join(all_page_texts)

        metadata = {
            'total_pages': len(pages_to_process),
            'extraction_method': 'vision_ai',
            'provider': provider,
            'model': model,
            'content_length': len(markdown_content)
        }

        return ExtractionResult(
            markdown_content=markdown_content,
            metadata=metadata,
            page_chunks=page_chunks,
            engine_used="vision"
        )

    except Exception as e:
        logger.error(f"Vision AI extraction failed: {e}")
        raise


async def _call_vision_for_text(
    img_base64: str,
    page_num: int,
    provider: str,
    model: str,
    ai_client: 'AIClientService'
) -> str:
    """
    Call vision API to extract text from page image.

    Args:
        img_base64: Base64-encoded page image
        page_num: Page number for logging
        provider: "anthropic", "openai", or "together"
        model: Model name
        ai_client: AI client service

    Returns:
        Extracted text in markdown format
    """
    prompt = """Extract ALL text from this PDF page image.

Requirements:
- Preserve the original layout and structure
- Use markdown formatting (headers, lists, tables, etc.)
- Include ALL text, even small print
- Maintain reading order (top to bottom, left to right)
- For tables, use markdown table format
- For multi-column layouts, process left column first, then right

Return ONLY the extracted text in markdown format, no explanations."""

    try:
        if provider == "anthropic":
            client = ai_client.anthropic
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            return response.content[0].text.strip()

        elif provider == "openai":
            client = ai_client.openai
            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }]
            )
            return response.choices[0].message.content.strip()

        else:
            raise ValueError(f"Unsupported vision provider: {provider}")

    except Exception as e:
        logger.error(f"Vision API call failed for page {page_num}: {e}")
        raise


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

