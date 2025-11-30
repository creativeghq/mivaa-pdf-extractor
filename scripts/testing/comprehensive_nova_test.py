#!/usr/bin/env python3
"""
COMPREHENSIVE PDF PROCESSING END-TO-END TEST

Tests the complete PDF processing pipeline and reports:
1. Products discovered
2. CLIP embeddings generated (total amount)
3. Total Images added to DB
4. Product relevancies to Images (e.g., 11 products, 123 image relevancies)
5. Total Embeddings (text + CLIP)
6. Meta Generated and Embeddings related to Meta
7. All relationship counts:
   - How many embeddings belong to how many products
   - How many chunks are related to how many products and images
"""

import os
import sys
import time
import requests
from datetime import datetime
from typing import Dict, Any, List

# Configuration
MIVAA_API = 'http://104.248.68.3:8000'
HARMONY_PDF_URL = 'https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/harmony-signature-book-24-25.pdf'
WORKSPACE_ID = 'ffafc28b-1b8b-4b0d-b226-9f9a6154004e'

# Test configuration
DISCOVERY_MODEL = 'claude-vision'  # Use Claude Vision by default
PROCESSING_MODE = 'deep'
EXTRACT_CATEGORIES = 'products'


def log(category: str, message: str, level: str = 'info'):
    """Log a message with timestamp and emoji"""
    emoji_map = {
        'step': '📋',
        'info': '📝',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'data': '📊'
    }
    emoji = emoji_map.get(level, '📝')
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"{emoji} [{timestamp}] [{category}] {message}")


def log_section(title: str):
    """Log a section header"""
    print('\n' + '=' * 100)
    print(f'🎯 {title}')
    print('=' * 100)


def check_active_jobs() -> List[Dict[str, Any]]:
    """Check for any active or pending jobs"""
    log('CHECK', 'Checking for active jobs...', 'step')
    
    try:
        response = requests.get(f'{MIVAA_API}/api/rag/documents/jobs?limit=50&sort=created_at:desc')
        if not response.ok:
            log('CHECK', f'Failed to fetch jobs: {response.status_code}', 'error')
            return []
        
        data = response.json()
        jobs = data.get('jobs', [])
        
        # Filter for active jobs (processing, pending, initialized)
        active_jobs = [j for j in jobs if j.get('status') in ['processing', 'pending', 'initialized', 'running']]
        
        if active_jobs:
            log('CHECK', f'Found {len(active_jobs)} active jobs:', 'warning')
            for job in active_jobs:
                log('CHECK', f"  - Job {job.get('id')}: {job.get('status')} ({job.get('progress', 0)}%)", 'warning')
        else:
            log('CHECK', 'No active jobs found', 'success')
        
        return active_jobs
    except Exception as e:
        log('CHECK', f'Error checking jobs: {e}', 'error')
        return []


def cleanup_old_harmony_jobs():
    """Delete all old Harmony PDF jobs"""
    log('CLEANUP', 'Deleting all old Harmony PDF jobs...', 'step')
    
    try:
        response = requests.get(f'{MIVAA_API}/api/rag/documents/jobs?limit=100&sort=created_at:desc')
        if not response.ok:
            log('CLEANUP', 'Failed to fetch jobs list', 'warning')
            return
        
        data = response.json()
        jobs = data.get('jobs', [])
        
        # Filter for Harmony PDF jobs
        harmony_jobs = [j for j in jobs if 'harmony-signature-book-24-25' in j.get('metadata', {}).get('filename', '').lower()]
        
        if not harmony_jobs:
            log('CLEANUP', 'No old Harmony PDF jobs found', 'info')
            return
        
        log('CLEANUP', f'Found {len(harmony_jobs)} old Harmony PDF jobs to delete', 'info')
        
        for job in harmony_jobs:
            job_id = job.get('id')
            log('CLEANUP', f'Deleting job {job_id}...', 'info')
            
            try:
                delete_response = requests.delete(f'{MIVAA_API}/api/rag/documents/jobs/{job_id}')
                if delete_response.ok:
                    log('CLEANUP', f'✅ Deleted job {job_id}', 'success')
                else:
                    log('CLEANUP', f'⚠️ Failed to delete job {job_id}: {delete_response.status_code}', 'warning')
            except Exception as e:
                log('CLEANUP', f'❌ Error deleting job {job_id}: {e}', 'error')
        
        log('CLEANUP', '✅ Cleanup complete!', 'success')
        time.sleep(2)  # Wait for cleanup to complete
        
    except Exception as e:
        log('CLEANUP', f'Error during cleanup: {e}', 'error')


def upload_pdf() -> Dict[str, str]:
    """Upload PDF for processing"""
    log('UPLOAD', f'Starting PDF upload with discovery_model={DISCOVERY_MODEL}', 'step')
    log('UPLOAD', f'PDF URL: {HARMONY_PDF_URL}', 'info')
    
    # Prepare form data
    form_data = {
        'file_url': HARMONY_PDF_URL,
        'title': f'Comprehensive Test - {DISCOVERY_MODEL}',
        'description': 'Extract all products from Harmony catalog',
        'tags': 'harmony,test,comprehensive',
        'categories': EXTRACT_CATEGORIES,
        'processing_mode': PROCESSING_MODE,
        'discovery_model': DISCOVERY_MODEL,
        'chunk_size': '1024',
        'chunk_overlap': '128',
        'enable_prompt_enhancement': 'true',
        'workspace_id': WORKSPACE_ID
    }
    
    log('UPLOAD', f'Triggering upload: {MIVAA_API}/api/rag/documents/upload', 'info')
    log('UPLOAD', f'Mode: {PROCESSING_MODE} | Categories: {EXTRACT_CATEGORIES} | Discovery: {DISCOVERY_MODEL}', 'info')
    
    response = requests.post(f'{MIVAA_API}/api/rag/documents/upload', data=form_data)
    
    if not response.ok:
        error_text = response.text
        raise Exception(f'Upload failed: {response.status_code} - {error_text}')
    
    result = response.json()
    
    log('UPLOAD', f"✅ Job ID: {result.get('job_id')}", 'success')
    log('UPLOAD', f"✅ Document ID: {result.get('document_id')}", 'success')
    log('UPLOAD', f"✅ Status: {result.get('status')}", 'success')
    
    return {
        'job_id': result.get('job_id'),
        'document_id': result.get('document_id')
    }


def monitor_job(job_id: str, document_id: str) -> Dict[str, Any]:
    """Monitor job progress until completion"""
    log('MONITOR', f'Starting job monitoring for: {job_id}', 'step')

    max_attempts = 480  # 2 hours
    poll_interval = 15  # 15 seconds

    for attempt in range(1, max_attempts + 1):
        time.sleep(poll_interval)

        try:
            response = requests.get(f'{MIVAA_API}/api/rag/documents/job/{job_id}')
            if not response.ok:
                log('MONITOR', f'API returned {response.status_code}', 'warning')
                continue

            job_data = response.json()
            status = job_data.get('status')
            progress = job_data.get('progress', 0)
            metadata = job_data.get('metadata', {})
            current_step = metadata.get('current_step') or metadata.get('stage') or 'Processing'

            # Build progress message
            progress_msg = f'[{attempt}/{max_attempts}] {status.upper()} ({progress}%) - {current_step}'

            if metadata.get('chunks_created'):
                progress_msg += f" | Chunks: {metadata['chunks_created']}"
            if metadata.get('images_extracted'):
                progress_msg += f" | Images: {metadata['images_extracted']}"
            if metadata.get('products_created'):
                progress_msg += f" | Products: {metadata['products_created']}"
            if metadata.get('current_page') and metadata.get('total_pages'):
                progress_msg += f" | Page: {metadata['current_page']}/{metadata['total_pages']}"

            log('MONITOR', progress_msg, 'info')

            if status == 'completed':
                log('MONITOR', '✅ Job completed successfully!', 'success')

                # Display final statistics
                if metadata.get('chunks_created') or metadata.get('images_extracted') or metadata.get('products_created'):
                    log('MONITOR', '📊 Final Statistics:', 'success')
                    log('MONITOR', f"   📄 Chunks: {metadata.get('chunks_created', 0)}", 'info')
                    log('MONITOR', f"   🖼️  Images: {metadata.get('images_extracted', 0)}", 'info')
                    log('MONITOR', f"   📦 Products: {metadata.get('products_created', 0)}", 'info')

                return job_data

            if status == 'failed':
                error = job_data.get('error') or metadata.get('error') or 'Unknown error'
                log('MONITOR', f'❌ Job failed: {error}', 'error')
                raise Exception(f'Job failed: {error}')

            if status == 'interrupted':
                log('MONITOR', '⚠️ Job interrupted!', 'warning')
                raise Exception('Job was interrupted')

        except requests.RequestException as e:
            log('MONITOR', f'Request error: {e}', 'warning')
            continue

    raise Exception('Job monitoring timed out after 2 hours')


def collect_comprehensive_data(document_id: str) -> Dict[str, Any]:
    """Collect all comprehensive data from the database"""
    log('COLLECT', f'Collecting comprehensive data for document: {document_id}', 'step')

    data = {
        'chunks': [],
        'images': [],
        'products': [],
        'chunk_image_relevancies': [],
        'product_image_relevancies': [],
        'chunk_product_relevancies': []
    }

    # Fetch chunks
    try:
        response = requests.get(f'{MIVAA_API}/api/rag/chunks?document_id={document_id}&limit=1000')
        if response.ok:
            chunks_data = response.json()
            data['chunks'] = chunks_data.get('chunks', [])
            log('COLLECT', f"✅ Found {len(data['chunks'])} chunks", 'success')
        else:
            log('COLLECT', f"⚠️ Failed to fetch chunks: {response.status_code}", 'warning')
    except Exception as e:
        log('COLLECT', f'Error fetching chunks: {e}', 'error')

    # Fetch images
    try:
        response = requests.get(f'{MIVAA_API}/api/rag/images?document_id={document_id}&limit=1000')
        if response.ok:
            images_data = response.json()
            data['images'] = images_data.get('images', [])
            log('COLLECT', f"✅ Found {len(data['images'])} images", 'success')
        else:
            log('COLLECT', f"⚠️ Failed to fetch images: {response.status_code}", 'warning')
    except Exception as e:
        log('COLLECT', f'Error fetching images: {e}', 'error')

    # Fetch products
    try:
        response = requests.get(f'{MIVAA_API}/api/rag/products?document_id={document_id}&limit=1000')
        if response.ok:
            products_data = response.json()
            data['products'] = products_data.get('products', [])
            log('COLLECT', f"✅ Found {len(data['products'])} products", 'success')
        else:
            log('COLLECT', f"⚠️ Failed to fetch products: {response.status_code}", 'warning')
    except Exception as e:
        log('COLLECT', f'Error fetching products: {e}', 'error')

    # Fetch chunk-image relevancies
    try:
        response = requests.get(f'{MIVAA_API}/api/rag/relevancies?document_id={document_id}&limit=1000')
        if response.ok:
            rel_data = response.json()
            data['chunk_image_relevancies'] = rel_data.get('relevancies', [])
            log('COLLECT', f"✅ Found {len(data['chunk_image_relevancies'])} chunk-image relevancies", 'success')
        else:
            log('COLLECT', f"⚠️ Failed to fetch chunk-image relevancies: {response.status_code}", 'warning')
    except Exception as e:
        log('COLLECT', f'Error fetching chunk-image relevancies: {e}', 'error')

    # Fetch product-image relevancies
    try:
        response = requests.get(f'{MIVAA_API}/api/rag/product-image-relationships?document_id={document_id}&limit=1000')
        if response.ok:
            rel_data = response.json()
            data['product_image_relevancies'] = rel_data.get('relationships', [])
            log('COLLECT', f"✅ Found {len(data['product_image_relevancies'])} product-image relevancies", 'success')
        else:
            log('COLLECT', f"⚠️ Failed to fetch product-image relevancies: {response.status_code}", 'warning')
    except Exception as e:
        log('COLLECT', f'Error fetching product-image relevancies: {e}', 'error')

    # Fetch chunk-product relevancies
    try:
        response = requests.get(f'{MIVAA_API}/api/rag/chunk-product-relationships?document_id={document_id}&limit=1000')
        if response.ok:
            rel_data = response.json()
            data['chunk_product_relevancies'] = rel_data.get('relationships', [])
            log('COLLECT', f"✅ Found {len(data['chunk_product_relevancies'])} chunk-product relevancies", 'success')
        else:
            log('COLLECT', f"⚠️ Failed to fetch chunk-product relevancies: {response.status_code}", 'warning')
    except Exception as e:
        log('COLLECT', f'Error fetching chunk-product relevancies: {e}', 'error')

    return data


def generate_comprehensive_report(data: Dict[str, Any], job_data: Dict[str, Any]):
    """Generate comprehensive report with all 7 required metrics"""
    log_section('📊 COMPREHENSIVE TEST RESULTS')

    # Calculate metrics
    chunks = data['chunks']
    images = data['images']
    products = data['products']
    chunk_image_rels = data['chunk_image_relevancies']
    product_image_rels = data['product_image_relevancies']
    chunk_product_rels = data['chunk_product_relevancies']

    # Count embeddings
    chunks_with_text_embeddings = sum(1 for c in chunks if c.get('embedding'))
    chunks_with_metadata = sum(1 for c in chunks if c.get('metadata') and len(c.get('metadata', {})) > 0)
    products_with_metadata = sum(1 for p in products if p.get('metadata') and len(p.get('metadata', {})) > 0)

    # Count CLIP embeddings (5 types per image)
    visual_embeddings = sum(1 for img in images if img.get('visual_clip_embedding_512'))
    color_embeddings = sum(1 for img in images if img.get('color_clip_embedding_512'))
    texture_embeddings = sum(1 for img in images if img.get('texture_clip_embedding_512'))
    application_embeddings = sum(1 for img in images if img.get('application_clip_embedding_512'))
    material_embeddings = sum(1 for img in images if img.get('material_clip_embedding_512'))
    total_clip_embeddings = visual_embeddings + color_embeddings + texture_embeddings + application_embeddings + material_embeddings

    total_embeddings = chunks_with_text_embeddings + total_clip_embeddings
    total_metadata = chunks_with_metadata + products_with_metadata
    total_relevancies = len(chunk_image_rels) + len(product_image_rels) + len(chunk_product_rels)

    # Print all 7 required metrics
    print('\n' + '=' * 100)
    print('1️⃣  PRODUCTS')
    print('=' * 100)
    print(f'   ✅ Total Products: {len(products)}')
    print(f'   ✅ Products with Metadata: {products_with_metadata}')

    print('\n' + '=' * 100)
    print('2️⃣  CLIP EMBEDDINGS GENERATED (TOTAL AMOUNT)')
    print('=' * 100)
    print(f'   ✅ Visual Embeddings: {visual_embeddings}')
    print(f'   ✅ Color Embeddings: {color_embeddings}')
    print(f'   ✅ Texture Embeddings: {texture_embeddings}')
    print(f'   ✅ Application Embeddings: {application_embeddings}')
    print(f'   ✅ Material Embeddings: {material_embeddings}')
    print(f'   ✅ TOTAL CLIP Embeddings: {total_clip_embeddings}')

    print('\n' + '=' * 100)
    print('3️⃣  TOTAL IMAGES ADDED TO DB')
    print('=' * 100)
    print(f'   ✅ Total Images: {len(images)}')

    print('\n' + '=' * 100)
    print('4️⃣  PRODUCT RELEVANCIES TO IMAGES')
    print('=' * 100)
    print(f'   ✅ Total Products: {len(products)}')
    print(f'   ✅ Product-Image Relevancies: {len(product_image_rels)}')
    print(f'   📊 Example: {len(products)} products → {len(product_image_rels)} image relationships')

    print('\n' + '=' * 100)
    print('5️⃣  TOTAL EMBEDDINGS (TEXT + CLIP)')
    print('=' * 100)
    print(f'   ✅ Text Embeddings (from chunks): {chunks_with_text_embeddings}')
    print(f'   ✅ CLIP Embeddings (from images): {total_clip_embeddings}')
    print(f'   ✅ TOTAL EMBEDDINGS: {total_embeddings}')

    print('\n' + '=' * 100)
    print('6️⃣  META GENERATED AND EMBEDDINGS RELATED TO META')
    print('=' * 100)
    print(f'   ✅ Chunks with Metadata: {chunks_with_metadata}')
    print(f'   ✅ Products with Metadata: {products_with_metadata}')
    print(f'   ✅ Total Metadata Generated: {total_metadata}')
    print(f'   ✅ Metadata Embeddings (text embeddings include metadata): {chunks_with_text_embeddings}')

    print('\n' + '=' * 100)
    print('7️⃣  ALL RELATIONSHIP COUNTS')
    print('=' * 100)
    print(f'   📊 EMBEDDINGS TO PRODUCTS:')
    print(f'      • Total Text Embeddings (chunks): {chunks_with_text_embeddings}')
    print(f'      • Total CLIP Embeddings (images): {total_clip_embeddings}')
    print(f'      • Products: {len(products)}')
    print(f'      • Chunk-Product Relationships: {len(chunk_product_rels)}')
    print(f'      • Product-Image Relationships: {len(product_image_rels)}')
    print(f'')
    print(f'   📊 CHUNKS TO PRODUCTS:')
    print(f'      • Total Chunks: {len(chunks)}')
    print(f'      • Total Products: {len(products)}')
    print(f'      • Chunk-Product Relationships: {len(chunk_product_rels)}')
    print(f'')
    print(f'   📊 CHUNKS TO IMAGES:')
    print(f'      • Total Chunks: {len(chunks)}')
    print(f'      • Total Images: {len(images)}')
    print(f'      • Chunk-Image Relationships: {len(chunk_image_rels)}')

    print('\n' + '=' * 100)
    print('📊 ALL RELEVANCIES SUMMARY')
    print('=' * 100)
    print(f'   ✅ Chunk-Image Relevancies: {len(chunk_image_rels)}')
    print(f'   ✅ Product-Image Relevancies: {len(product_image_rels)}')
    print(f'   ✅ Chunk-Product Relevancies: {len(chunk_product_rels)}')
    print(f'   ✅ TOTAL RELEVANCIES: {total_relevancies}')

    # Print sample data
    print('\n' + '=' * 100)
    print('📝 SAMPLE CHUNKS (First 3)')
    print('=' * 100)
    for idx, chunk in enumerate(chunks[:3], 1):
        print(f'\nChunk {idx}:')
        print(f"  ID: {chunk.get('id')}")
        print(f"  Content: {chunk.get('content', '')[:150]}...")
        print(f"  Page: {chunk.get('page_number', 'N/A')}")
        if chunk.get('metadata'):
            print(f"  Has Metadata: Yes ({len(chunk.get('metadata', {}))} fields)")

    print('\n' + '=' * 100)
    print('🖼️  SAMPLE IMAGES (First 3)')
    print('=' * 100)
    for idx, img in enumerate(images[:3], 1):
        print(f'\nImage {idx}:')
        print(f"  ID: {img.get('id')}")
        print(f"  URL: {img.get('url') or img.get('storage_path')}")
        print(f"  Page: {img.get('page_number', 'N/A')}")
        print(f"  Has Visual CLIP: {'Yes' if img.get('visual_clip_embedding_512') else 'No'}")
        print(f"  Has Color CLIP: {'Yes' if img.get('color_clip_embedding_512') else 'No'}")

    print('\n' + '=' * 100)
    print('🏷️  ALL PRODUCTS')
    print('=' * 100)
    for idx, product in enumerate(products, 1):
        print(f'\nProduct {idx}:')
        print(f"  ID: {product.get('id')}")
        print(f"  Name: {product.get('name')}")
        print(f"  Designer: {product.get('designer', 'N/A')}")
        print(f"  Description: {product.get('description', '')[:200]}...")
        if product.get('metadata'):
            print(f"  Has Metadata: Yes ({len(product.get('metadata', {}))} fields)")

    print('\n' + '=' * 100)
    print('✅ TEST COMPLETED SUCCESSFULLY')
    print('=' * 100)


def main():
    """Main test execution"""
    log_section('COMPREHENSIVE PDF PROCESSING END-TO-END TEST')

    print(f'PDF: {HARMONY_PDF_URL}')
    print(f'Workspace: {WORKSPACE_ID}')
    print(f'MIVAA API: {MIVAA_API}')
    print(f'Discovery Model: {DISCOVERY_MODEL}')
    print(f'Processing Mode: {PROCESSING_MODE}')
    print(f'Extract Categories: {EXTRACT_CATEGORIES}\n')

    try:
        # Step 1: Check for active jobs
        active_jobs = check_active_jobs()
        if active_jobs:
            log('MAIN', f'⚠️ WARNING: {len(active_jobs)} active jobs found!', 'warning')
            log('MAIN', 'Please wait for them to complete or cancel them before running this test', 'warning')
            sys.exit(1)

        # Step 2: Cleanup old Harmony jobs
        cleanup_old_harmony_jobs()

        # Step 3: Upload PDF
        upload_result = upload_pdf()
        job_id = upload_result['job_id']
        document_id = upload_result['document_id']

        # Step 4: Monitor job
        job_data = monitor_job(job_id, document_id)

        # Step 5: Collect comprehensive data
        data = collect_comprehensive_data(document_id)

        # Step 6: Generate comprehensive report
        generate_comprehensive_report(data, job_data)

        log('MAIN', '✅ All tests completed successfully!', 'success')

    except Exception as e:
        log('MAIN', f'❌ Test failed: {e}', 'error')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


