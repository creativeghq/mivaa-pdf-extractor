#!/usr/bin/env python3
"""
Direct PDF Processing Script for WIFI MOMO Lookbook
Processes PDFs directly to extract text, chunks, and images
"""

import requests
import tempfile
import os
import sys
import json
from typing import List, Dict, Any
import time

def download_pdf(url: str, filename: str) -> str:
    """Download PDF from URL to temporary file"""
    print(f"📥 Downloading PDF: {filename}")
    print(f"🔗 URL: {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    
    with open(temp_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    file_size = os.path.getsize(temp_path)
    print(f"📏 Downloaded: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    return temp_path

def smart_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200, min_chunk_size: int = 10) -> List[str]:
    """Smart text chunking that tries to preserve semantic boundaries."""
    if not text or chunk_size <= 0:
        return []
    
    # First, try to split by paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ''
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            # Save current chunk if it's not empty
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            
            # If paragraph itself is too large, split it
            if len(paragraph) > chunk_size:
                # Simple text splitting
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    if end >= len(paragraph):
                        chunks.append(paragraph[start:])
                        break
                    # Find good breaking point
                    space_pos = paragraph.rfind(' ', start, end)
                    if space_pos > start:
                        end = space_pos
                    chunks.append(paragraph[start:end])
                    start = max(start + 1, end - overlap)
                current_chunk = ''
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
    
    # Add final chunk
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_pdf_with_pymupdf(pdf_path: str) -> Dict[str, Any]:
    """Process PDF using pymupdf4llm (same as MIVAA)"""
    try:
        import pymupdf4llm
        print("📄 Processing PDF with pymupdf4llm...")
        
        start_time = time.time()
        
        # Extract markdown content
        markdown_content = pymupdf4llm.to_markdown(pdf_path)
        
        processing_time = time.time() - start_time
        
        print(f"✅ PDF processed successfully in {processing_time:.2f} seconds")
        print(f"📊 Markdown content length: {len(markdown_content)} characters")
        
        return {
            'markdown_content': markdown_content,
            'processing_time': processing_time,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error processing PDF with pymupdf4llm: {e}")
        return {'success': False, 'error': str(e)}

def extract_images_info(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract image information from PDF"""
    try:
        import fitz  # PyMuPDF
        print("🖼️ Extracting image information...")
        
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                images.append({
                    'page': page_num + 1,
                    'index': img_index,
                    'xref': img[0],
                    'format': 'unknown',
                    'description': f'Image {img_index + 1} on page {page_num + 1}'
                })
        
        doc.close()
        print(f"✅ Found {len(images)} images")
        return images
        
    except Exception as e:
        print(f"❌ Error extracting images: {e}")
        return []

def process_pdf_complete(url: str, filename: str) -> Dict[str, Any]:
    """Complete PDF processing pipeline"""
    print(f"\n🚀 PROCESSING PDF: {filename}")
    print("=" * 50)
    
    try:
        # Step 1: Download PDF
        pdf_path = download_pdf(url, filename)
        
        # Step 2: Extract text content
        pdf_result = process_pdf_with_pymupdf(pdf_path)
        
        if not pdf_result.get('success'):
            return pdf_result
        
        markdown_content = pdf_result['markdown_content']
        
        # Step 3: Create text chunks
        print("\n📝 Creating text chunks...")
        chunks = smart_chunk_text(markdown_content, chunk_size=1000, overlap=200, min_chunk_size=10)
        print(f"✅ Created {len(chunks)} text chunks")
        
        # Step 4: Extract image information
        images = extract_images_info(pdf_path)
        
        # Step 5: Calculate metrics
        page_count = markdown_content.count('-----') if '-----' in markdown_content else 1
        word_count = len(markdown_content.split())
        
        # Clean up temporary file
        try:
            os.remove(pdf_path)
        except:
            pass
        
        return {
            'success': True,
            'filename': filename,
            'processing_time': pdf_result['processing_time'],
            'page_count': page_count,
            'word_count': word_count,
            'markdown_content': markdown_content,
            'chunks': chunks,
            'images': images,
            'metrics': {
                'chunks_count': len(chunks),
                'images_count': len(images),
                'pages_count': page_count,
                'words_count': word_count,
                'characters_count': len(markdown_content)
            }
        }
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main function to process WIFI MOMO PDF"""
    wifi_momo_url = 'https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/49f683ad-ebf2-4296-a410-0d8c011ce0be/WIFI%20MOMO%20lookbook%2001s.pdf'
    
    result = process_pdf_complete(wifi_momo_url, 'WIFI-MOMO-lookbook.pdf')
    
    if result.get('success'):
        print(f"\n🎉 WIFI MOMO LOOKBOOK PROCESSING RESULTS")
        print("=" * 50)
        print(f"📝 Text chunks created: {result['metrics']['chunks_count']}")
        print(f"🖼️ Images extracted: {result['metrics']['images_count']}")
        print(f"📄 Pages processed: {result['metrics']['pages_count']}")
        print(f"📊 Words extracted: {result['metrics']['words_count']:,}")
        print(f"📊 Characters extracted: {result['metrics']['characters_count']:,}")
        print(f"⏰ Processing time: {result['processing_time']:.2f} seconds")
        
        # Show first few chunks
        if result['chunks']:
            print(f"\n📝 CHUNKS ({len(result['chunks'])} total):")
            for i, chunk in enumerate(result['chunks'][:5], 1):
                preview = chunk[:100] + '...' if len(chunk) > 100 else chunk
                print(f"{i}. {preview}")
            if len(result['chunks']) > 5:
                print(f"... and {len(result['chunks']) - 5} more chunks")
        
        # Show image info
        if result['images']:
            print(f"\n🖼️ IMAGES ({len(result['images'])} total):")
            for i, image in enumerate(result['images'][:5], 1):
                print(f"{i}. Page {image['page']}, Index {image['index']} - {image['description']}")
            if len(result['images']) > 5:
                print(f"... and {len(result['images']) - 5} more images")
        
    else:
        print(f"\n❌ PROCESSING FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
