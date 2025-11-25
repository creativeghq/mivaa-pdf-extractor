#!/usr/bin/env python3
"""
Clear all running jobs before deployment.
This ensures old jobs don't continue running with outdated code.
"""
import os
import sys
from supabase import create_client, Client
from datetime import datetime

def main():
    # Initialize Supabase client
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print('❌ Missing Supabase credentials')
        sys.exit(1)

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    try:
        # Get all running/processing jobs
        response = supabase.table('background_jobs').select('*').in_('status', ['processing', 'pending']).execute()
        
        running_jobs = response.data if response.data else []
        
        if not running_jobs:
            print('✅ No running jobs found - clean state')
        else:
            print(f'🔄 Found {len(running_jobs)} running jobs - marking as interrupted...')
            
            for job in running_jobs:
                job_id = job.get('id')
                filename = job.get('filename', 'Unknown')
                
                # Mark job as interrupted
                supabase.table('background_jobs').update({
                    'status': 'interrupted',
                    'interrupted_at': datetime.utcnow().isoformat(),
                    'error': 'Job interrupted by deployment - service restarted with new code',
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('id', job_id).execute()
                
                print(f'  ✅ Interrupted job {job_id}: {filename}')
            
            print(f'✅ Successfully interrupted {len(running_jobs)} jobs')
        
        # Clean up any stale job_progress entries
        supabase.table('job_progress').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print('✅ Cleared job_progress table')
        
        print('✅ Job cleanup complete - ready for deployment')
        
    except Exception as e:
        print(f'❌ Error clearing jobs: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

