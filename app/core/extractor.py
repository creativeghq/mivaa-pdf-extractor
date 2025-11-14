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


def extract_pdf_to_markdown(file_name, page_number):
    """
    Extract PDF content as Markdown.
    
    Args:
        file_name: Path to the PDF file
        page_number: Specific page number to extract (None for all pages)
        
    Returns:
        Markdown content as string
    """
    page_number_list = None
    if page_number is not None:
        page_number_list = [page_number]   
    
    return pymupdf4llm.to_markdown(file_name, pages=page_number_list)


def extract_pdf_tables(file_name, page_number):
    """
    Extract tables from PDF as CSV.
    
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
        md_text_images = pymupdf4llm.to_markdown(
            doc=file_path,
            pages=page_number_list,
            page_chunks=True,
            write_images=True,
            image_path=image_path,
            image_format="jpg",
            dpi=200
        )

        output_file = os.path.join(output_dir, "output.json")
        pathlib.Path(output_file).write_text(json.dumps(str(md_text_images)))
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
            gc.collect()

        # Combine all batches
        combined_markdown = "\n\n".join(str(m) for m in all_markdown)
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

            # Force garbage collection after each batch to free memory
            gc.collect()

        # Combine all batches
        combined_markdown = "\n\n".join(str(m) for m in all_markdown)

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
            batch_markdown = pymupdf4llm.to_markdown(
                doc=file_path,
                pages=batch_pages,
                page_chunks=True,
                write_images=True,
                image_path=image_path,
                image_format="jpg",
                dpi=200
            )

            # Count images in this batch
            image_files = os.listdir(image_path) if os.path.exists(image_path) else []
            image_count = len([f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))])

            yield batch_num, str(batch_markdown), image_count

            # Force garbage collection after each batch
            gc.collect()

    finally:
        doc.close()
        gc.collect()


def extract_functional_metadata(file_path, page_number=None, extract_mode="comprehensive"):
    """
    Extract comprehensive functional metadata from material tile specification PDFs.
    
    Args:
        file_path: Path to the PDF file
        page_number: Specific page to extract (optional, extracts all pages if None)
        extract_mode: Extraction mode - "comprehensive", "safety", "surface", etc.
    
    Returns:
        Dictionary containing structured functional metadata for the 9 categories
    """
    # This is a placeholder for the functional metadata extraction
    # The full implementation would be quite large, so this provides the interface
    metadata = {
        "surface_properties": {},
        "dimensional_properties": {},
        "mechanical_properties": {},
        "thermal_properties": {},
        "chemical_properties": {},
        "electrical_properties": {},
        "optical_properties": {},
        "environmental_properties": {},
        "safety_properties": {}
    }
    
    # TODO: Implement full metadata extraction logic
    # This would involve complex pattern matching and data extraction
    
    return metadata
