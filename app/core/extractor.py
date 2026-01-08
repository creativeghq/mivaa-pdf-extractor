"""
PDF Extraction Core Module

This module contains the core PDF extraction functionality that was originally
in the root extractor.py file. It provides functions for extracting markdown,
tables, images, and metadata from PDF files.
"""

import pymupdf4llm
import pathlib
from pathlib import Path
import json
import os
import fitz
import csv
import re
from typing import Dict, Any, Optional


def _fix_glyph_names(text: str) -> str:
    """
    Convert PDF glyph names to actual Unicode characters and fix common text issues.

    PyMuPDF4LLM sometimes outputs glyph names instead of actual characters.
    This function post-processes the text to fix these issues.

    Args:
        text: Raw text from pymupdf4llm

    Returns:
        Text with glyph names replaced by actual characters
    """
    replacements = {
        # Numbers
        '/nine.LP': '9', '/eight.LP': '8', '/seven.LP': '7',
        '/six.LP': '6', '/five.LP': '5', '/four.LP': '4',
        '/three.LP': '3', '/two.LP': '2', '/one.LP': '1', '/zero.LP': '0',

        # Punctuation
        '/emdash.cap': '—', '/threequarteremdash': '—',
        '/percent.LP': '%', '/parenleft.cap': '(', '/parenright.cap': ')',
        '/periodcentered.cap': '·', '/minus.cap': '-',
        '/period.LP': '.', '/comma.LP': ',', '/colon.LP': ':',
        '/semicolon.LP': ';', '/slash.LP': '/', '/backslash.LP': '\\',

        # Quotes
        '/quotedbl.LP': '"', '/quotesingle.LP': "'",
        '/quotedblleft': '"', '/quotedblright': '"',
        '/quoteleft': ''', '/quoteright': ''',

        # Math symbols
        '/plus.LP': '+', '/equal.LP': '=', '/less.LP': '<', '/greater.LP': '>',
        '/multiply': '×', '/divide': '÷',

        # Other common glyphs
        '/space.LP': ' ', '/hyphen.LP': '-', '/underscore.LP': '_',
        '/at.LP': '@', '/numbersign.LP': '#', '/dollar.LP': '$',
        '/ampersand.LP': '&', '/asterisk.LP': '*',
        '/question.LP': '?', '/exclam.LP': '!',
        '/bracketleft.LP': '[', '/bracketright.LP': ']',
        '/braceleft.LP': '{', '/braceright.LP': '}',
    }

    # First pass: Replace known glyph names
    for glyph, char in replacements.items():
        text = text.replace(glyph, char)

    # Second pass: Fix ligature patterns using regex
    # Pattern: /letter_letter or /letter_letter_letter
    # Examples: /f_ter -> fter, /t_terns -> tterns, /a/t_tentive -> attentive
    text = re.sub(r'/([a-z])_([a-z]+)', r'\1\2', text)  # /f_ter -> fter
    text = re.sub(r'/([a-z])/([a-z])_([a-z]+)', r'\1\2\3', text)  # /a/t_tentive -> attentive

    # Third pass: Fix any remaining slash-letter patterns that might be ligatures
    # Pattern: /letter (single letter after slash)
    text = re.sub(r'/([a-z])\b', r'\1', text)  # /f -> f (but only at word boundaries)

    # Fix excessive newlines (more than 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix spaces before newlines
    text = re.sub(r' +\n', '\n', text)

    # Fix double spaces
    text = re.sub(r'  +', ' ', text)

    return text


def extract_pdf_to_markdown(file_name, page_number):
    """
    Extract PDF content as Markdown with glyph name fixes.

    Args:
        file_name: Path to the PDF file
        page_number: Specific page number to extract (None for all pages)

    Returns:
        Markdown content as string with glyph names fixed
    """
    page_number_list = None
    if page_number is not None:
        page_number_list = [page_number]

    try:
        # Extract markdown
        # ALWAYS disable header identification to avoid PyMuPDF4LLM hanging
        # PyMuPDF4LLM's header identification can cause indefinite hangs on some PDFs
        hdr_info = False

        # Keep document open to prevent garbage collection
        try:
            doc = fitz.open(file_name)
        except Exception as open_error:
            # If opening fails (e.g., "bad xref"), try with repair mode
            if "xref" in str(open_error).lower() or "damaged" in str(open_error).lower():
                import logging
                logging.getLogger(__name__).warning(f"PDF has structural issues ({open_error}), attempting repair...")
                # PyMuPDF can sometimes repair PDFs automatically on open
                doc = fitz.open(file_name, filetype="pdf")
            else:
                raise

        try:
            # Use default table_strategy for table extraction
            markdown_text = pymupdf4llm.to_markdown(doc, pages=page_number_list, hdr_info=hdr_info)
        finally:
            doc.close()

        # ✅ FIX GLYPH NAMES
        markdown_text = _fix_glyph_names(markdown_text)

        return markdown_text

    except (RuntimeError, ValueError) as e:
        error_msg = str(e).lower()
        if "not a textpage" in error_msg:
            import logging
            logging.getLogger(__name__).warning(f"PyMuPDF 'not a textpage' error - attempting page-by-page extraction")
            # Re-raise to trigger page-by-page extraction in pdf_processor.py
            raise
        elif "xref" in error_msg or "damaged" in error_msg or "corrupt" in error_msg:
            # PDF is corrupted, raise a more informative error
            raise ValueError(f"PDF file is corrupted or damaged: {e}") from e
        else:
            raise
    except ReferenceError as e:
        if "weakly-referenced object no longer exists" in str(e):
            # Retry with a fresh document object
            doc = fitz.open(file_name)
            try:
                markdown_text = pymupdf4llm.to_markdown(doc, pages=page_number_list, hdr_info=hdr_info)
                markdown_text = _fix_glyph_names(markdown_text)
                return markdown_text
            finally:
                doc.close()
        else:
            raise


def extract_pdf_to_markdown_with_doc(doc, page_number):
    """
    Extract PDF content as Markdown using an already-opened fitz.Document.

    This function keeps the document open to avoid garbage collection issues
    when processing pages one-by-one.

    Args:
        doc: fitz.Document object (already opened)
        page_number: Specific page number to extract (None for all pages)

    Returns:
        Markdown content as string with glyph names fixed
    """
    page_number_list = None
    if page_number is not None:
        page_number_list = [page_number]

    try:
        # Extract markdown using the document object
        # Disable header identification to avoid PyMuPDF4LLM bug
        hdr_info = False if page_number is not None else None
        markdown_text = pymupdf4llm.to_markdown(doc, pages=page_number_list, hdr_info=hdr_info)

        # ✅ FIX GLYPH NAMES
        markdown_text = _fix_glyph_names(markdown_text)

        return markdown_text

    except (IndexError, ValueError, RuntimeError) as e:
        error_msg = str(e).lower()
        if "xref" in error_msg or "damaged" in error_msg or "corrupt" in error_msg:
            # PDF is corrupted
            raise ValueError(f"PDF file is corrupted or damaged: {e}") from e
        elif "not in document" in error_msg or "bad page number" in error_msg or "page" in error_msg:
            # Page doesn't exist in document
            total_pages = len(doc) if hasattr(doc, '__len__') else "unknown"
            raise ValueError(f"Page {page_number} not in document (total pages: {total_pages})") from e
        else:
            raise
    except ReferenceError as e:
        if "weakly-referenced object no longer exists" in str(e):
            # Document was garbage collected, raise a more informative error
            raise ValueError(f"Document object was garbage collected while processing page {page_number}. Try using extract_pdf_to_markdown() instead.") from e
        else:
            raise


def extract_pdf_tables(file_name, page_number):
    """
    Extract tables from PDF as CSV using PyMuPDF.

    ⚠️ DEPRECATED: This function uses PyMuPDF's basic table detection which is less
    accurate than Camelot. For new code, use:
    - app.services.pdf.table_extraction.TableExtractor (Camelot-based, YOLO-guided)

    This function is kept for backward compatibility with legacy code only.

    Args:
        file_name: Path to the PDF file
        page_number: Specific page number to extract (None for all pages)

    Returns:
        List of table data
    """
    page_number_list = None
    if page_number is not None:
        page_number_list = [page_number-1]

    doc = fitz.open(file_name)
    tables = []
    
    for page_num in (page_number_list if page_number_list else range(len(doc))):
        page = doc.load_page(page_num)
        page_tables = page.find_tables()
        
        for table in page_tables:
            table_data = table.extract()
            tables.append({
                'page': page_num + 1,
                'data': table_data
            })
    
    doc.close()
    return tables


def extract_json_and_images(file_path, output_dir, page_number, batch_size=5, page_list=None):
    """
    Extract JSON and images from PDF with batch processing to reduce memory usage.

    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save extracted content
        page_number: Specific page number to extract (None for all pages)
        batch_size: Number of pages to process at once (default: 5 for memory efficiency)
        page_list: Optional list of specific page numbers to extract (0-indexed).
                   When provided, only these pages will be processed (for focused extraction).
    """
    import fitz
    import gc

    page_number_list = None
    if page_number is not None:
        page_number_list = [page_number-1]

    image_path = os.path.join(output_dir, 'images')
    os.makedirs(image_path, exist_ok=True)

    # If processing specific page, use original method
    if page_number_list is not None:
        try:
            md_text_images = pymupdf4llm.to_markdown(
                doc=file_path,
                pages=page_number_list,
                page_chunks=True,
                write_images=True,
                image_path=image_path,
                image_format="jpg",
                dpi=200
            )
        except (ValueError, ReferenceError) as e:
            if "not a textpage" in str(e) or "weakly-referenced object" in str(e):
                # Skip this page - no images to extract
                md_text_images = ""
            else:
                raise

        # ✅ FIX GLYPH NAMES
        md_text_images = _fix_glyph_names(str(md_text_images))

        output_file = os.path.join(output_dir, "output.json")
        pathlib.Path(output_file).write_text(json.dumps(md_text_images))
        return

    # ✅ OPTIMIZATION: If page_list is provided (focused extraction), only process those pages
    if page_list is not None:
        # Convert 1-indexed page numbers to 0-indexed for PyMuPDF
        pages_to_process = [p - 1 if p > 0 else 0 for p in page_list]

        # Process in batches for memory efficiency
        all_markdown = []
        for batch_start in range(0, len(pages_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(pages_to_process))
            batch_pages = pages_to_process[batch_start:batch_end]

            # Extract markdown and images for this batch
            try:
                batch_markdown = pymupdf4llm.to_markdown(
                    doc=file_path,
                    pages=batch_pages,
                    page_chunks=True,
                    write_images=True,
                    image_path=image_path,
                    image_format="jpg",
                    dpi=200
                )
                all_markdown.append(batch_markdown)
            except (ValueError, ReferenceError) as e:
                if "not a textpage" in str(e) or "weakly-referenced object" in str(e):
                    # Skip this batch - problematic pages
                    pass
                else:
                    raise

            gc.collect()

        # Combine all batches
        combined_markdown = "\n\n".join(str(m) for m in all_markdown)

        # ✅ FIX GLYPH NAMES
        combined_markdown = _fix_glyph_names(combined_markdown)

        output_file = os.path.join(output_dir, "output.json")
        pathlib.Path(output_file).write_text(json.dumps(combined_markdown))
        return

    # For full PDF extraction, use batch processing to reduce memory usage
    doc = fitz.open(file_path)
    total_pages = len(doc)
    all_markdown = []

    try:
        # Process pages in batches
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_pages = list(range(batch_start, batch_end))

            # Extract markdown and images for this batch
            try:
                batch_markdown = pymupdf4llm.to_markdown(
                    doc=file_path,
                    pages=batch_pages,
                    page_chunks=True,
                    write_images=True,
                    image_path=image_path,
                    image_format="jpg",
                    dpi=200
                )
                all_markdown.append(batch_markdown)
            except (ValueError, ReferenceError) as e:
                if "not a textpage" in str(e) or "weakly-referenced object" in str(e):
                    # Skip this batch - problematic pages
                    pass
                else:
                    raise

            # Force garbage collection after each batch to free memory
            gc.collect()

        # Combine all batches
        combined_markdown = "\n\n".join(str(m) for m in all_markdown)

        # ✅ FIX GLYPH NAMES
        combined_markdown = _fix_glyph_names(combined_markdown)

        output_file = os.path.join(output_dir, "output.json")
        pathlib.Path(output_file).write_text(json.dumps(combined_markdown))

    finally:
        doc.close()
        gc.collect()


def extract_json_and_images_streaming(file_path, output_dir, batch_size=5):
    """
    Extract JSON and images from PDF using streaming/chunked processing.

    This generator function yields markdown content and image info for each batch,
    allowing memory to be freed between batches.

    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save extracted content
        batch_size: Number of pages to process at once (default: 5)

    Yields:
        Tuple of (batch_number, markdown_content, image_count)
    """
    import fitz
    import gc

    image_path = os.path.join(output_dir, 'images')
    os.makedirs(image_path, exist_ok=True)

    doc = fitz.open(file_path)
    total_pages = len(doc)

    try:
        # Process pages in batches
        for batch_num, batch_start in enumerate(range(0, total_pages, batch_size)):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_pages = list(range(batch_start, batch_end))

            # Extract markdown and images for this batch
            try:
                batch_markdown = pymupdf4llm.to_markdown(
                    doc=file_path,
                    pages=batch_pages,
                    page_chunks=True,
                    write_images=True,
                    image_path=image_path,
                    image_format="jpg",
                    dpi=200
                )
            except (ValueError, ReferenceError) as e:
                if "not a textpage" in str(e) or "weakly-referenced object" in str(e):
                    # Skip this batch - problematic pages
                    batch_markdown = ""
                else:
                    raise

            # Count images in this batch
            image_files = os.listdir(image_path) if os.path.exists(image_path) else []
            image_count = len([f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))])

            # ✅ FIX GLYPH NAMES
            batch_markdown_fixed = _fix_glyph_names(str(batch_markdown))

            yield batch_num, batch_markdown_fixed, image_count

            # Force garbage collection after each batch
            gc.collect()

    finally:
        doc.close()
        gc.collect()




