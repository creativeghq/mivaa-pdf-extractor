"""
PDF Processing API Routes

This module contains FastAPI routes for PDF processing operations including
markdown conversion, table extraction, and image extraction.

Integrates existing extractor.py functionality with production FastAPI structure.
"""

import logging
import tempfile
import os
import io
import zipfile
import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse

# Import existing extractor functions
try:
    from app.core.extractor import extract_pdf_to_markdown, extract_pdf_tables, extract_json_and_images
except ImportError:
    # Fallback to root level extractor if core module not available
    from extractor import extract_pdf_to_markdown, extract_pdf_tables, extract_json_and_images

from app.utils.exceptions import (
    PDFProcessingError,
    PDFValidationError,
    PDFExtractionError
)
from app.dependencies import get_current_user, get_workspace_context
from app.schemas.auth import User, WorkspaceContext

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter(prefix="/api/v1", tags=["PDF Processing"])


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    """
    Save uploaded file to temporary location.
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        Path to temporary file
    """
    try:
        suffix = Path(upload_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, prefix=upload_file.filename, suffix=suffix) as tmp:
            tmp.write(upload_file.file.read())
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def create_output_dir():
    """Create timestamped output directory."""
    output_dir = os.path.join('output', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_zip_stream(output_dir):
    """Create ZIP stream from output directory."""
    zip_stream = io.BytesIO()  
    with zipfile.ZipFile(zip_stream, "w") as zf:  
        for root, _, files in os.walk(output_dir):  
            for file in files:  
                file_path = os.path.join(root, file)  
                zf.write(file_path, os.path.relpath(file_path, output_dir))  
    zip_stream.seek(0) 
    return zip_stream


@router.post(
    "/extract/markdown",
    summary="Extract Markdown from PDF",
    description="Convert PDF document to Markdown format with optional page selection"
)
async def extract_markdown(
    file: UploadFile = File(..., description="PDF file to process"),
    page_number: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context)
):
    """
    Extract markdown content from a PDF file.
    
    Args:
        file: The uploaded PDF file
        page_number: Optional specific page number to extract (1-based)
        
    Returns:
        Markdown content as string
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        logger.info(f"Processing markdown extraction for file: {file.filename}")
        
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise PDFValidationError("File must be a PDF")
        
        # Save uploaded file to temporary location
        tmp_path = save_upload_file_tmp(file)
        
        try:
            # Use existing extractor function
            result = extract_pdf_to_markdown(tmp_path, page_number)
            return result
            
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                os.unlink(tmp_path)
        
    except PDFValidationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file"
        )
    except Exception as e:
        logger.error(f"Unexpected error in markdown extraction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during PDF processing"
        )


@router.post(
    "/extract/tables",
    response_class=StreamingResponse,
    summary="Extract Tables from PDF",
    description="Extract tables from PDF and return as ZIP file containing CSV files"
)
async def extract_tables(
    file: UploadFile = File(..., description="PDF file to process"),
    page_number: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context)
) -> StreamingResponse:
    """
    Extract tables from a PDF file and return as ZIP archive.
    
    Args:
        file: The uploaded PDF file
        page_number: Optional specific page number to extract (1-based)
        
    Returns:
        StreamingResponse with ZIP file containing CSV tables
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        logger.info(f"Processing table extraction for file: {file.filename}")
        
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise PDFValidationError("File must be a PDF")
        
        # Save uploaded file to temporary location
        tmp_path = save_upload_file_tmp(file)
        
        try:
            # Create output directory
            out_dir = create_output_dir()
            
            # Use existing extractor function
            tables_data = extract_pdf_tables(tmp_path, page_number)

            # Save tables as CSV files in output directory
            if tables_data:
                for i, table in enumerate(tables_data):
                    csv_filename = f"table_{i+1}.csv"
                    csv_path = os.path.join(out_dir, csv_filename)
                    table.to_csv(csv_path, index=False)
            else:
                # Create empty file if no tables found
                empty_csv_path = os.path.join(out_dir, "no_tables_found.txt")
                with open(empty_csv_path, 'w') as f:
                    f.write("No tables found in the specified page(s).")
            
            # Create ZIP stream
            zip_stream = create_zip_stream(out_dir)
            
            # Prepare filename
            file_name, _ = os.path.splitext(file.filename)
            zip_filename = f"{file_name}_csv.zip"
            
            return StreamingResponse(
                io.BytesIO(zip_stream.read()),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
            )
            
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                os.unlink(tmp_path)
        
    except PDFValidationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file"
        )
    except Exception as e:
        logger.error(f"Unexpected error in table extraction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during PDF processing"
        )


@router.post(
    "/extract/images",
    response_class=StreamingResponse,
    summary="Extract Images from PDF",
    description="Extract images and metadata from PDF and return as ZIP file"
)
async def extract_images(
    file: UploadFile = File(..., description="PDF file to process"),
    page_number: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context)
) -> StreamingResponse:
    """
    Extract images from a PDF file and return as ZIP archive.
    
    Args:
        file: The uploaded PDF file
        page_number: Optional specific page number to extract (1-based)
        
    Returns:
        StreamingResponse with ZIP file containing images and metadata
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        logger.info(f"Processing image extraction for file: {file.filename}")
        
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise PDFValidationError("File must be a PDF")
        
        # Save uploaded file to temporary location
        tmp_path = save_upload_file_tmp(file)
        
        try:
            # Create output directory
            out_dir = create_output_dir()
            
            # Use existing extractor function
            extract_json_and_images(tmp_path, out_dir, page_number)
            
            # Create ZIP stream
            zip_stream = create_zip_stream(out_dir)
            
            # Prepare filename
            file_name, _ = os.path.splitext(file.filename)
            zip_filename = f"{file_name}.zip"
            
            return StreamingResponse(
                io.BytesIO(zip_stream.read()),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
            )
            
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                os.unlink(tmp_path)
        
    except PDFValidationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file"
        )
    except Exception as e:
        logger.error(f"Unexpected error in image extraction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during PDF processing"
        )


@router.get(
    "/health",
    summary="PDF Service Health Check",
    description="Check the health and availability of PDF processing services"
)
async def health_check(
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context)
):
    """
    Health check endpoint for PDF processing service.
    
    Returns:
        Service health status and capabilities
    """
    try:
        # Test basic functionality
        import pymupdf4llm
        
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "service": "PDF Processing API",
            "capabilities": {
                "markdown_extraction": True,
                "table_extraction": True,
                "image_extraction": True,
                "pymupdf4llm_available": True
            },
            "dependencies": {
                "pymupdf4llm": True,
                "tempfile_access": os.access(tempfile.gettempdir(), os.W_OK)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "service": "PDF Processing API",
            "error": str(e)
        }