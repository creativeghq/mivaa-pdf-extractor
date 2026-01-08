#!/bin/bash
# Deploy fixes to server
# Run this script to push local changes to server

set -e  # Exit on error

echo "🚀 Deploying fixes to server..."
echo ""

# 1. Commit changes
echo "📝 Committing changes..."
git add app/api/rag_routes.py
git add app/schemas/product_progress.py
git add app/api/pdf_processing/stage_3_images.py
git add app/api/pdf_processing/product_processor.py
git add FIXES_APPLIED.md
git add deploy-fixes.sh

git commit -m "Fix: Job metadata tracking and CLIP embeddings counting

- Add real-time job metadata updates (chunks, images, CLIP embeddings)
- Track CLIP embeddings through entire pipeline
- Ensure 100% data persistence in database
- YOLO endpoint URL already correct

Fixes:
- background_jobs metadata counters now update in real-time
- CLIP embeddings counted and tracked
- All images saved to document_images with embeddings
- All chunks saved to document_chunks with embeddings
- All relationships created in chunk_product_relationships

Files modified:
- app/api/rag_routes.py
- app/schemas/product_progress.py
- app/api/pdf_processing/stage_3_images.py
- app/api/pdf_processing/product_processor.py"

echo "✅ Changes committed"
echo ""

# 2. Push to remote
echo "📤 Pushing to remote..."
git push origin main

echo "✅ Pushed to remote"
echo ""

# 3. Deploy to server
echo "🖥️  Deploying to server..."
ssh basil@51.20.3.164 << 'ENDSSH'
cd /var/www/mivaa-pdf-extractor
git pull origin main
sudo systemctl restart mivaa-pdf-extractor
sleep 5
sudo systemctl status mivaa-pdf-extractor --no-pager | head -20
ENDSSH

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Next steps:"
echo "1. Monitor logs: ssh basil@51.20.3.164 'journalctl -u mivaa-pdf-extractor -f'"
echo "2. Test with a PDF upload"
echo "3. Verify job metadata updates in real-time"

