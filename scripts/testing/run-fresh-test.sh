#!/bin/bash

echo "🧪 Starting fresh NOVA Product Focused End-to-End Test"
echo "======================================================================================================"

# Upload PDF and start processing
curl -X POST "http://127.0.0.1:8000/api/upload-pdf-with-discovery" \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_url": "https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/harmony-signature-book-24-25%20(1).pdf",
    "workspace_id": "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
    "focused_extraction": true,
    "extract_categories": ["products"]
  }' | python3 -m json.tool

