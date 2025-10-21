#!/usr/bin/env python3
"""
Fix for MIVAA job progress tracking issues
This script fixes the progress callback to properly handle bulk job progress
"""

import re

def fix_progress_callback():
    """Fix the progress callback function in admin.py"""
    
    # Read the current admin.py file
    with open('app/api/admin.py', 'r') as f:
        content = f.read()
    
    # Find and replace the problematic progress_callback function
    old_callback = '''        def progress_callback(progress_percentage: float, current_step: str, details: dict = None):
            """Callback to update job progress during PDF processing"""
            try:
                if job_id:
                    # Update job tracking directly in active_jobs (sync operation)
                    if job_id in active_jobs:
                        job_info = active_jobs[job_id]
                        job_info["progress_percentage"] = progress_percentage
                        job_info["current_step"] = current_step
                        job_info["updated_at"] = datetime.utcnow().isoformat()

                        # Update details
                        if details:
                            if "details" not in job_info:
                                job_info["details"] = {}
                            job_info["details"].update(details)

                        logger.info(f"📊 Job {job_id} progress updated: {progress_percentage}% - {current_step}")
                    else:
                        logger.warning(f"Job {job_id} not found in active_jobs for progress update")
            except Exception as e:
                logger.warning(f"Failed to update job progress: {e}")'''

    new_callback = '''        def progress_callback(progress_percentage: float, current_step: str, details: dict = None):
            """Callback to update individual document progress within bulk job"""
            try:
                if job_id and job_id in active_jobs:
                    job_info = active_jobs[job_id]
                    
                    # Update current step for this document
                    document_step = f"Document {document_id}: {current_step}"
                    job_info["current_step"] = document_step
                    job_info["updated_at"] = datetime.utcnow().isoformat()

                    # Update document-specific details without overwriting main job progress
                    if "details" not in job_info:
                        job_info["details"] = {}
                    
                    # Update details for this specific document
                    if details:
                        job_info["details"].update(details)
                    
                    # Update pages processed if available
                    if details and "pages_processed" in details:
                        job_info["details"]["pages_processed"] = details["pages_processed"]
                    if details and "total_pages" in details:
                        job_info["details"]["total_pages"] = details["total_pages"]

                    logger.info(f"📊 Document {document_id} progress: {progress_percentage}% - {current_step}")
                else:
                    logger.warning(f"Job {job_id} not found in active_jobs for progress update")
            except Exception as e:
                logger.warning(f"Failed to update document progress: {e}")'''

    # Replace the callback function
    if old_callback in content:
        content = content.replace(old_callback, new_callback)
        print("✅ Fixed progress_callback function")
    else:
        print("⚠️ Could not find exact progress_callback function to replace")
        return False
    
    # Write the fixed content back
    with open('app/api/admin.py', 'w') as f:
        f.write(content)
    
    return True

if __name__ == "__main__":
    if fix_progress_callback():
        print("🎉 Progress tracking fix applied successfully!")
    else:
        print("❌ Failed to apply progress tracking fix")
