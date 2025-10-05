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


def extract_json_and_images(file_path, output_dir, page_number):
    """
    Extract JSON and images from PDF.
    
    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save extracted content
        page_number: Specific page number to extract (None for all pages)
    """
    page_number_list = None
    if page_number is not None:
        page_number_list = [page_number-1]

    image_path = os.path.join(output_dir, 'images')
    os.makedirs(image_path, exist_ok=True)

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
