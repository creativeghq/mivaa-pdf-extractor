#!/usr/bin/env python3
"""
Monitor remote PDF processing test execution
Polls the API to show real-time progress
"""

import time
import requests
import sys
from datetime import datetime

# Configuration
MIVAA_API = 'http://104.248.68.3:8000'
HARMONY_PDF_NAME = 'harmony-signature-book-24-25'

def log(message: str, level: str = 'info'):
    """Log with timestamp"""
    emoji_map = {
        'info': '📝',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'step': '📋'
    }
    emoji = emoji_map.get(level, '📝')
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"{emoji} [{timestamp}] {message}")
    sys.stdout.flush()

def check_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f'{MIVAA_API}/health', timeout=10)
        if response.ok:
            data = response.json()
            log(f"API Health: {data.get('status')} - Version {data.get('version')}", 'success')
            return True
        else:
            log(f"API returned {response.status_code}", 'error')
            return False
    except Exception as e:
        log(f"Health check failed: {e}", 'error')
        return False

def get_latest_harmony_job():
    """Get the latest Harmony PDF job"""
    try:
        response = requests.get(f'{MIVAA_API}/api/rag/documents/jobs?limit=20&sort=created_at:desc', timeout=10)
        if not response.ok:
            return None
        
        data = response.json()
        jobs = data.get('jobs', [])
        
        # Find latest Harmony job
        for job in jobs:
            filename = job.get('metadata', {}).get('filename', '').lower()
            if HARMONY_PDF_NAME in filename:
                return job
        
        return None
    except Exception as e:
        log(f"Error fetching jobs: {e}", 'error')
        return None

def monitor_job(job_id: str):
    """Monitor a specific job"""
    log(f"Monitoring job: {job_id}", 'step')
    
    last_progress = -1
    last_stage = None
    
    while True:
        try:
            response = requests.get(f'{MIVAA_API}/api/rag/documents/job/{job_id}', timeout=10)
            if not response.ok:
                log(f"API returned {response.status_code}", 'warning')
                time.sleep(5)
                continue
            
            job = response.json()
            status = job.get('status')
            progress = job.get('progress', 0)
            metadata = job.get('metadata', {})
            
            current_stage = metadata.get('current_step') or metadata.get('stage') or 'Processing'
            
            # Show update if progress or stage changed
            if progress != last_progress or current_stage != last_stage:
                msg = f"[{status.upper()}] {progress}% - {current_stage}"
                
                # Add metrics
                if metadata.get('chunks_created'):
                    msg += f" | Chunks: {metadata['chunks_created']}"
                if metadata.get('images_extracted'):
                    msg += f" | Images: {metadata['images_extracted']}"
                if metadata.get('products_created'):
                    msg += f" | Products: {metadata['products_created']}"
                if metadata.get('current_page') and metadata.get('total_pages'):
                    msg += f" | Page: {metadata['current_page']}/{metadata['total_pages']}"
                
                log(msg, 'info')
                last_progress = progress
                last_stage = current_stage
            
            # Check if completed
            if status == 'completed':
                log('Job completed successfully!', 'success')
                log(f"Final stats: Chunks={metadata.get('chunks_created', 0)}, Images={metadata.get('images_extracted', 0)}, Products={metadata.get('products_created', 0)}", 'success')
                return job
            
            if status == 'failed':
                error = job.get('error') or metadata.get('error') or 'Unknown error'
                log(f'Job failed: {error}', 'error')
                return job
            
            if status == 'interrupted':
                log('Job was interrupted', 'warning')
                return job
            
            time.sleep(10)  # Poll every 10 seconds
            
        except KeyboardInterrupt:
            log('Monitoring stopped by user', 'warning')
            return None
        except Exception as e:
            log(f"Error monitoring job: {e}", 'error')
            time.sleep(5)

def main():
    """Main monitoring loop"""
    print("\n" + "="*100)
    print("🔍 REMOTE PDF PROCESSING TEST MONITOR")
    print("="*100 + "\n")
    
    # Check health
    if not check_health():
        log("API is not healthy. Exiting.", 'error')
        sys.exit(1)
    
    log("Waiting for new Harmony PDF job to start...", 'step')
    log("(Start the test on the server now)", 'info')
    
    # Wait for job to appear
    job = None
    for attempt in range(60):  # Wait up to 5 minutes
        job = get_latest_harmony_job()
        if job:
            created_at = job.get('created_at', '')
            # Check if job is recent (created in last 5 minutes)
            log(f"Found job: {job.get('id')} (created: {created_at})", 'success')
            break
        
        time.sleep(5)
        if attempt % 6 == 0:  # Every 30 seconds
            log(f"Still waiting... ({attempt*5}s elapsed)", 'info')
    
    if not job:
        log("No Harmony PDF job found. Please start the test.", 'error')
        sys.exit(1)
    
    # Monitor the job
    final_job = monitor_job(job.get('id'))
    
    if final_job and final_job.get('status') == 'completed':
        log("\n✅ TEST COMPLETED SUCCESSFULLY!", 'success')
    else:
        log("\n❌ TEST DID NOT COMPLETE SUCCESSFULLY", 'error')

if __name__ == '__main__':
    main()

