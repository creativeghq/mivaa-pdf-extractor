#!/bin/bash

JOB_ID="ce78e2e0-edc2-4e59-b80a-9f033a553058"
DOC_ID="8d3ad254-683d-417b-bb75-f2e8d29e91f4"

echo "======================================================================================================"
echo "🔍 MONITORING JOB: $JOB_ID"
echo "======================================================================================================"
echo ""

while true; do
  clear
  echo "======================================================================================================"
  echo "🔍 NOVA PDF PROCESSING - LIVE MONITORING"
  echo "======================================================================================================"
  echo "Job ID: $JOB_ID"
  echo "Document ID: $DOC_ID"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "======================================================================================================"
  echo ""
  
  # Get job status
  RESPONSE=$(curl -s "http://127.0.0.1:8000/api/rag/documents/jobs?job_id=$JOB_ID")
  
  STATUS=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['jobs'][0]['status'] if data.get('jobs') else 'unknown')")
  PROGRESS=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['jobs'][0]['progress'] if data.get('jobs') else 0)")
  STAGE=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['jobs'][0]['metadata'].get('current_stage', 'unknown') if data.get('jobs') else 'unknown')")
  
  echo "📊 STATUS: $STATUS"
  echo "📈 PROGRESS: $PROGRESS%"
  echo "🔄 STAGE: $STAGE"
  echo ""
  
  # Show metadata
  echo "📋 METADATA:"
  echo "$RESPONSE" | python3 -m json.tool | grep -A 15 '"metadata"'
  echo ""
  
  # Check if completed
  if [ "$STATUS" = "completed" ]; then
    echo "✅ JOB COMPLETED!"
    echo ""
    echo "Fetching final results..."
    break
  fi
  
  if [ "$STATUS" = "failed" ]; then
    echo "❌ JOB FAILED!"
    echo "$RESPONSE" | python3 -m json.tool
    exit 1
  fi
  
  echo "⏳ Refreshing in 10 seconds... (Press Ctrl+C to stop)"
  sleep 10
done

# Fetch final results
echo ""
echo "======================================================================================================"
echo "📊 FETCHING FINAL RESULTS"
echo "======================================================================================================"
echo ""

echo "1️⃣  PRODUCTS:"
curl -s "http://127.0.0.1:8000/api/rag/products?document_id=$DOC_ID&limit=1000" | python3 -c "
import sys, json
data = json.load(sys.stdin)
products = data.get('products', [])
print(f'   ✅ Total Products: {len(products)}')
with_meta = sum(1 for p in products if p.get('metadata'))
print(f'   ✅ Products with Metadata: {with_meta}')
"

echo ""
echo "2️⃣  IMAGES:"
curl -s "http://127.0.0.1:8000/api/rag/images?document_id=$DOC_ID&limit=10000" | python3 -c "
import sys, json
data = json.load(sys.stdin)
images = data.get('images', [])
print(f'   ✅ Total Images: {len(images)}')
"

# Query embeddings table for image embeddings (document_images columns were dropped)
echo "   ✅ CLIP Embeddings (from embeddings table):"
curl -s "http://127.0.0.1:8000/api/supabase/query" \
  -H "Content-Type: application/json" \
  -d "{\"table\": \"embeddings\", \"filters\": {\"entity_type\": \"image\"}}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
embeddings = data.get('data', [])
visual = sum(1 for e in embeddings if e.get('embedding_type') == 'visual_512')
color = sum(1 for e in embeddings if e.get('embedding_type') == 'color_512')
texture = sum(1 for e in embeddings if e.get('embedding_type') == 'texture_512')
style = sum(1 for e in embeddings if e.get('embedding_type') == 'style_512')
material = sum(1 for e in embeddings if e.get('embedding_type') == 'material_512')
total_clip = visual + color + texture + style + material
print(f'      • Visual: {visual}')
print(f'      • Color: {color}')
print(f'      • Texture: {texture}')
print(f'      • Style: {style}')
print(f'      • Material: {material}')
print(f'      • TOTAL: {total_clip}')
"

echo ""
echo "3️⃣  CHUNKS:"
curl -s "http://127.0.0.1:8000/api/rag/chunks?document_id=$DOC_ID&limit=10000" | python3 -c "
import sys, json
data = json.load(sys.stdin)
chunks = data.get('chunks', [])
print(f'   ✅ Total Chunks: {len(chunks)}')
with_emb = sum(1 for c in chunks if c.get('embedding'))
print(f'   ✅ Chunks with Embeddings: {with_emb}')
with_meta = sum(1 for c in chunks if c.get('metadata'))
print(f'   ✅ Chunks with Metadata: {with_meta}')
"

echo ""
echo "4️⃣  RELEVANCIES:"
curl -s "http://127.0.0.1:8000/api/rag/relevancies?document_id=$DOC_ID&limit=10000" | python3 -c "
import sys, json
data = json.load(sys.stdin)
rels = data.get('relevancies', [])
print(f'   ✅ Chunk-Image Relevancies: {len(rels)}')
"

curl -s "http://127.0.0.1:8000/api/rag/product-image-relationships?document_id=$DOC_ID&limit=10000" | python3 -c "
import sys, json
data = json.load(sys.stdin)
rels = data.get('relationships', [])
print(f'   ✅ Product-Image Relevancies: {len(rels)}')
"

curl -s "http://127.0.0.1:8000/api/rag/chunk-product-relationships?document_id=$DOC_ID&limit=10000" | python3 -c "
import sys, json
data = json.load(sys.stdin)
rels = data.get('relationships', [])
print(f'   ✅ Chunk-Product Relevancies: {len(rels)}')
"

echo ""
echo "======================================================================================================"
echo "✅ MONITORING COMPLETE"
echo "======================================================================================================"
