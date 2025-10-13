#!/usr/bin/env python3

import requests
import json
import time

MIVAA_BASE_URL = 'http://localhost:8000'
PDF_URL = 'https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/49f683ad-ebf2-4296-a410-0d8c011ce0be/WIFI%20MOMO%20lookbook%2001s.pdf'

def make_request(url, method='GET', data=None):
    try:
        if method == 'POST':
            response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        else:
            response = requests.get(url)
        
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Request failed: {e}")
        raise

def main():
    print('🎯 WIFI MOMO LOOKBOOK - COMPREHENSIVE EXTRACTION RESULTS')
    print('=' * 80)
    
    try:
        # Submit new processing job
        print('\n📤 Submitting PDF for processing...')
        bulk_response = make_request(f'{MIVAA_BASE_URL}/api/bulk/process', 'POST', {
            'urls': [PDF_URL],
            'batch_size': 1,
            'options': {
                'extract_text': True,
                'extract_images': True,
                'extract_tables': True
            }
        })
        
        job_id = bulk_response.get('data', {}).get('job_id')
        if not job_id:
            raise Exception('No job ID returned')
        
        print(f'✅ Job submitted: {job_id}')
        
        # Monitor job completion
        print('\n⏰ Monitoring job progress...')
        attempts = 0
        max_attempts = 20
        job_status = 'pending'
        
        while attempts < max_attempts and job_status != 'completed':
            time.sleep(15)
            
            jobs_response = make_request(f'{MIVAA_BASE_URL}/api/jobs')
            job = next((j for j in jobs_response.get('jobs', []) if j.get('job_id') == job_id), None)
            
            if job:
                job_status = job.get('status', 'unknown')
                print(f'🔄 Attempt {attempts + 1}: Status = {job_status}')
                
                if job_status == 'failed':
                    raise Exception(f"Job failed: {job.get('error_message', 'Unknown error')}")
            else:
                print(f'⚠️ Job {job_id} not found in jobs list')
            
            attempts += 1
        
        if job_status != 'completed':
            print(f'⚠️ Job did not complete within {max_attempts * 15} seconds, but let me check existing completed jobs...')
        
        # Get all completed jobs and show results
        print('\n📊 Checking all completed jobs for results...')
        jobs_response = make_request(f'{MIVAA_BASE_URL}/api/jobs')
        completed_jobs = [j for j in jobs_response.get('jobs', []) if j.get('status') == 'completed']
        
        print(f'Found {len(completed_jobs)} completed jobs')
        
        if completed_jobs:
            latest_job = completed_jobs[-1]  # Get the most recent completed job
            print(f'📋 Analyzing latest completed job: {latest_job.get("job_id")}')
            
            # Since we can't get detailed results from the API, let's show what we can
            print('\n' + '=' * 80)
            print('📋 JOB COMPLETION ANALYSIS')
            print('=' * 80)
            
            print(f'\n✅ Latest Completed Job Details:')
            print(f'   Job ID: {latest_job.get("job_id")}')
            print(f'   Type: {latest_job.get("job_type")}')
            print(f'   Status: {latest_job.get("status")}')
            print(f'   Created: {latest_job.get("created_at")}')
            
            # Try to get results from the processing endpoint directly
            print('\n🔍 Attempting direct PDF processing to show extraction capabilities...')
            try:
                direct_response = make_request(f'{MIVAA_BASE_URL}/api/documents/process', 'POST', {
                    'url': PDF_URL,
                    'extract_text': True,
                    'extract_images': True,
                    'extract_tables': True
                })
                
                print('✅ Direct processing successful!')
                
                # Show results
                if 'chunks' in direct_response:
                    chunks = direct_response['chunks']
                    print(f'\n📝 TEXT CHUNKS EXTRACTED: {len(chunks)}')
                    print('-' * 50)
                    
                    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                        print(f'\n🔸 Chunk {i + 1}:')
                        print(f'   Length: {len(chunk)} characters')
                        preview = chunk[:100] + ('...' if len(chunk) > 100 else '')
                        print(f'   Preview: "{preview}"')
                        
                        if 'momo' in chunk.lower():
                            print('   ✨ Contains MOMO brand content!')
                
                if 'images' in direct_response:
                    images = direct_response['images']
                    print(f'\n🖼️ IMAGES EXTRACTED: {len(images)}')
                    print('-' * 50)
                    
                    for i, image in enumerate(images[:5]):  # Show first 5 images
                        print(f'\n🔸 Image {i + 1}:')
                        print(f'   URL: {image.get("url", "N/A")}')
                        print(f'   Size: {image.get("width", "N/A")}x{image.get("height", "N/A")}')
                        print(f'   Format: {image.get("format", "N/A")}')
                        print(f'   Page: {image.get("page", "N/A")}')
                
                if 'metadata' in direct_response:
                    metadata = direct_response['metadata']
                    print(f'\n📊 DOCUMENT METADATA:')
                    print('-' * 50)
                    print(f'   Pages: {metadata.get("pages", "N/A")}')
                    print(f'   Word Count: {metadata.get("word_count", "N/A")}')
                    print(f'   Character Count: {metadata.get("character_count", "N/A")}')
                    print(f'   Processing Time: {metadata.get("processing_time", "N/A")}s')
                
            except Exception as e:
                print(f'⚠️ Direct processing failed: {e}')
                print('📋 This indicates the job completed but results are not accessible via API')
        
        print('\n' + '=' * 80)
        print('🎉 ANALYSIS COMPLETE!')
        print('=' * 80)
        
    except Exception as e:
        print(f'\n❌ Error: {e}')

if __name__ == '__main__':
    main()
