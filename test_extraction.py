#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append('/var/www/mivaa-pdf-extractor')

from app.services.pdf_processor import PDFProcessor

async def test_wifi_momo_extraction():
    print('🔍 TESTING WIFI MOMO PDF EXTRACTION')
    print('=' * 60)
    
    processor = PDFProcessor()
    pdf_url = 'https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/49f683ad-ebf2-4296-a410-0d8c011ce0be/WIFI%20MOMO%20lookbook%2001s.pdf'
    
    try:
        print('📤 Processing WIFI MOMO PDF directly...')
        result = await processor.process_pdf_from_url(
            pdf_url, 
            'test_wifi_momo', 
            {'extract_images': True, 'extract_text': True}
        )
        
        print('✅ Processing completed!')
        print('\n📊 EXTRACTION RESULTS:')
        print(f'   Document ID: {result.document_id}')
        print(f'   Pages: {result.page_count}')
        print(f'   Word Count: {result.word_count}')
        print(f'   Character Count: {result.character_count}')
        print(f'   Processing Time: {result.processing_time:.2f}s')
        print(f'   Images Extracted: {len(result.extracted_images)}')
        
        # Show text content
        if result.markdown_content:
            chunks = result.markdown_content.split('\n\n')
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            print(f'   Text Chunks: {len(chunks)}')
            
            print('\n📝 FIRST 5 TEXT CHUNKS:')
            for i, chunk in enumerate(chunks[:5]):
                preview = chunk[:100] + '...' if len(chunk) > 100 else chunk
                print(f'   Chunk {i+1}: "{preview}"')
        else:
            print('   ❌ No markdown content extracted')
        
        # Show images
        print(f'\n🖼️ EXTRACTED IMAGES ({len(result.extracted_images)} total):')
        for i, img in enumerate(result.extracted_images[:5]):
            width = img.get('width', '?')
            height = img.get('height', '?')
            path = img.get('path', 'N/A')
            print(f'   Image {i+1}: {path} ({width}x{height})')
        
        # Check OCR results
        if result.ocr_text:
            print(f'\n🔍 OCR EXTRACTION:')
            print(f'   OCR Text Length: {len(result.ocr_text)} characters')
            ocr_preview = result.ocr_text[:200] + '...' if len(result.ocr_text) > 200 else result.ocr_text
            print(f'   OCR Sample: "{ocr_preview}"')
            
            # Check for MOMO mentions
            momo_count = result.ocr_text.lower().count('momo')
            print(f'   MOMO mentions: {momo_count}')
        else:
            print('   ❌ No OCR text extracted')
        
        # Show OCR results details
        if result.ocr_results:
            print(f'\n🔍 OCR RESULTS DETAILS:')
            print(f'   OCR Results Count: {len(result.ocr_results)}')
            for i, ocr_result in enumerate(result.ocr_results[:3]):
                print(f'   OCR Result {i+1}: {ocr_result}')
        
        return result
        
    except Exception as e:
        print(f'❌ Processing failed: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    result = asyncio.run(test_wifi_momo_extraction())
    if result:
        print('\n🎉 EXTRACTION TEST COMPLETED SUCCESSFULLY!')
    else:
        print('\n❌ EXTRACTION TEST FAILED!')
