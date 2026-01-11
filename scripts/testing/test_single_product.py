#!/usr/bin/env python3
"""
🧪 Single Product Test Script

This script uploads a PDF and processes ONLY the first product for testing/debugging.
Use this to validate all fixes work correctly before processing all products.

Usage:
    python test_single_product.py <pdf_file_path>

Example:
    python test_single_product.py ~/Downloads/catalog.pdf
"""

import sys
import requests
import time
from pathlib import Path

# Configuration
MIVAA_API = "http://localhost:8000"  # Change to your MIVAA API URL
WORKSPACE_ID = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e"

def upload_pdf_test_mode(pdf_path: str):
    """Upload PDF in test mode (process only first product)."""
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("🧪 SINGLE PRODUCT TEST MODE")
    print("=" * 80)
    print(f"📄 PDF: {pdf_file.name}")
    print(f"🔗 API: {MIVAA_API}")
    print(f"⚠️  Will process ONLY the first product")
    print()
    
    # Prepare upload
    with open(pdf_file, 'rb') as f:
        files = {'file': (pdf_file.name, f, 'application/pdf')}
        data = {
            'workspace_id': WORKSPACE_ID,
            'processing_mode': 'standard',
            'categories': 'products',
            'discovery_model': 'claude-sonnet-4.5',
            'test_single_product': 'true',  # 🧪 TEST MODE ENABLED
        }
        
        print("📤 Uploading PDF...")
        response = requests.post(
            f'{MIVAA_API}/api/rag/documents/upload',
            files=files,
            data=data
        )
    
    if not response.ok:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        sys.exit(1)
    
    result = response.json()
    job_id = result.get('job_id')
    document_id = result.get('document_id')
    
    print(f"✅ Upload successful!")
    print(f"   Job ID: {job_id}")
    print(f"   Document ID: {document_id}")
    print()
    
    # Poll for completion
    print("⏳ Monitoring progress...")
    print()
    
    while True:
        time.sleep(5)
        
        # Get job status
        status_response = requests.get(f'{MIVAA_API}/api/rag/jobs/{job_id}/status')
        if not status_response.ok:
            print(f"⚠️  Failed to get status: {status_response.status_code}")
            continue
        
        status_data = status_response.json()
        job_status = status_data.get('status')
        progress = status_data.get('progress', 0)
        current_stage = status_data.get('current_stage', 'unknown')
        
        print(f"📊 Status: {job_status} | Progress: {progress}% | Stage: {current_stage}")
        
        if job_status in ['completed', 'failed', 'error']:
            break
    
    print()
    print("=" * 80)
    if job_status == 'completed':
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print()
        print("Next steps:")
        print("1. Check logs for any errors")
        print("2. Verify product was created correctly")
        print("3. If all looks good, run with test_single_product=false")
    else:
        print(f"❌ TEST FAILED: {job_status}")
        print()
        print("Check logs for details:")
        print(f"   journalctl -u mivaa-pdf-extractor -f | grep '{job_id}'")
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_single_product.py <pdf_file_path>")
        print()
        print("Example:")
        print("  python test_single_product.py ~/Downloads/catalog.pdf")
        sys.exit(1)
    
    upload_pdf_test_mode(sys.argv[1])

