# LEGACY FILE - DEPRECATED
# This file has been replaced by app/main.py
#
# The new comprehensive API is located in app/main.py with 37+ endpoints
# This file is kept for reference only and should not be used
#
# To run the new API:
# uvicorn app.main:app --host 0.0.0.0 --port 8000
#
# For Docker:
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# COMMENTED OUT TO PREVENT ACCIDENTAL USE
"""
from fastapi import FastAPI, UploadFile
from extractor import extract_pdf_to_markdown, extract_pdf_tables, extract_json_and_images
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
import os, datetime, io, zipfile
from fastapi.responses import StreamingResponse

app = FastAPI()

# LEGACY ENDPOINTS - COMMENTED OUT
# These endpoints have been moved to app/api/pdf_routes.py with enhanced functionality
"""
@app.post("/extract/markdown")
def extract_markdown(file: UploadFile, page_number:int=None):

    return extract_pdf_to_markdown(save_upload_file_tmp((file)),page_number)

@app.post("/extract/tables")
def extract_table(file: UploadFile, page_number:int=None):

    out_dir= create_output_dir()
    file_name, file_extension = os.path.splitext(file.filename)
    extract_pdf_tables(save_upload_file_tmp((file)),page_number,out_dir)
    zip_stream = create_zip_stream(out_dir)

    return StreamingResponse(zip_stream, media_type="application/octet-stream", headers={"Content-Disposition": "attachment;  filename="+file_name+"_csv"+".zip"})

@app.post("/extract/images")
def extract_images(file: UploadFile, page_number:int=None, include_functional_metadata:bool=False, extract_mode:str="comprehensive"):
    """
    Extract images and JSON metadata from PDFs, optionally including functional metadata.
    
    Args:
        file: PDF file upload
        page_number: Specific page to extract (optional, extracts all pages if None)
        include_functional_metadata: Whether to include functional metadata extraction
        extract_mode: Functional metadata extraction mode - "comprehensive", "safety", etc.
    """
    
    out_dir= create_output_dir()
    file_name, file_extension = os.path.splitext(file.filename)
    
    # Extract standard images and JSON
    extract_json_and_images(save_upload_file_tmp((file)),out_dir,page_number)
    
    # Optionally add functional metadata
    if include_functional_metadata:
        try:
            from extractor import extract_functional_metadata as extract_func
            file_path = save_upload_file_tmp(file)
            functional_metadata = extract_func(file_path, page_number, extract_mode)

# Add a simple docs redirect
@app.get("/docs")
def docs_redirect():
    """Redirect to comprehensive API docs"""
    return {
        "message": "This is the legacy API. For comprehensive documentation:",
        "comprehensive_docs": "Please use the comprehensive API at app/main.py",
        "endpoints": "37+ endpoints available in the new API",
        "features": [
            "JWT authentication",
            "Performance monitoring",
            "RAG system integration",
            "Vector search capabilities",
            "Multi-modal processing"
        ],
        "migration_guide": "Contact support for migration assistance"
    }

# Simple startup message
if __name__ == "__main__":
    import uvicorn
    print("⚠️  WARNING: This is the legacy API endpoint")
    print("🚀 For the comprehensive API with 37+ endpoints, use: uvicorn app.main:app")
    print("📚 Visit /docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000)