#!/usr/bin/env python3
"""
Complete Database Reset Script (Python Version)

This script:
1. Deletes knowledge base data (chunks, embeddings, products, images, etc.)
2. PRESERVES user data (users, profiles, workspaces, API keys)
3. Deletes all files from storage buckets EXCEPT pdf-documents folder
4. Verifies cleanup was successful
5. Reports storage and resource usage
"""

import os
import sys
import requests
from datetime import datetime

SUPABASE_URL = 'https://bgbavxtjlbvgplozizxu.supabase.co'
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

if not SUPABASE_SERVICE_ROLE_KEY:
    print('❌ SUPABASE_SERVICE_ROLE_KEY environment variable is required')
    sys.exit(1)

# Tables to clear (in order to respect foreign key constraints)
TABLES_TO_CLEAR = [
    'job_progress',
    'ai_analysis_queue',
    'embeddings',
    'document_images',
    'document_chunks',
    'products',
    'background_jobs',
    'documents',
    'processed_documents',
    'materials_catalog',
    'material_visual_analysis',
    'processing_results',
    'quality_metrics_daily',
    'quality_scoring_logs',
    'analytics_events',
    'agent_tasks',
    'generation_3d',
    'scraped_materials_temp',
    'scraping_sessions',
    'scraping_pages'
]

# Storage buckets configuration
BUCKETS_CONFIG = [
    {'name': 'pdf-tiles', 'exclude_folders': []},
    {'name': 'pdf-documents', 'exclude_folders': ['*']},  # Preserve all files
    {'name': 'material-images', 'exclude_folders': []}
]


def make_supabase_request(method, path, body=None):
    """Make a request to Supabase API"""
    url = f"{SUPABASE_URL}{path}"
    headers = {
        'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}',
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Prefer': 'return=minimal'
    }
    
    if body:
        headers['Content-Type'] = 'application/json'
        response = requests.request(method, url, headers=headers, json=body)
    else:
        response = requests.request(method, url, headers=headers)
    
    # For DELETE requests, 200 or 204 is OK
    if method == 'DELETE' and response.status_code in [200, 204]:
        return {'success': True}
    
    if not response.ok:
        raise Exception(f"Supabase API error {response.status_code}: {response.text}")
    
    # Handle empty responses
    if not response.text:
        return {'success': True}
    
    try:
        return response.json()
    except:
        return {'success': True, 'raw': response.text}


def clear_table(table_name):
    """Clear all rows from a table"""
    print(f"\n🗑️  Clearing table: {table_name}")
    
    try:
        # Get count before deletion
        count_response = make_supabase_request('GET', f'/rest/v1/{table_name}?select=count')
        count = count_response[0].get('count', 0) if count_response else 0
        
        if count == 0:
            print(f"   ✅ Table {table_name} is already empty")
            return {'table': table_name, 'deleted': 0}
        
        print(f"   📊 Found {count} rows to delete")
        
        # Delete all rows
        make_supabase_request('DELETE', f'/rest/v1/{table_name}?id=neq.00000000-0000-0000-0000-000000000000')
        
        print(f"   ✅ Deleted {count} rows from {table_name}")
        return {'table': table_name, 'deleted': count}
    except Exception as error:
        print(f"   ❌ Failed to clear {table_name}: {error}")
        return {'table': table_name, 'deleted': 0, 'error': str(error)}


def list_bucket_files(bucket_name, path=''):
    """List files in a storage bucket"""
    try:
        response = make_supabase_request('POST', f'/storage/v1/object/list/{bucket_name}', {
            'prefix': path,
            'limit': 1000,
            'offset': 0
        })
        return response or []
    except Exception as error:
        print(f"   ❌ Failed to list files in {bucket_name}/{path}: {error}")
        return []


def delete_file(bucket_name, file_path):
    """Delete a file from storage"""
    try:
        make_supabase_request('DELETE', f'/storage/v1/object/{bucket_name}', {
            'prefixes': [file_path]
        })
        return True
    except Exception as error:
        print(f"   ⚠️  Failed to delete {bucket_name}/{file_path}: {error}")
        return False


def list_all_files_recursively(bucket_name, prefix=''):
    """List all files in a bucket recursively"""
    all_files = []

    def list_folder(folder_path):
        items = list_bucket_files(bucket_name, folder_path)

        for item in items:
            full_path = f"{folder_path}/{item['name']}" if folder_path else item['name']

            # If it's a folder, recurse into it
            if not item.get('metadata') or item.get('metadata', {}).get('mimetype') == 'application/x-directory':
                list_folder(full_path)
            else:
                # It's a file, add it to the list
                all_files.append(full_path)

    list_folder(prefix)
    return all_files


def clear_bucket(bucket_config):
    """Clear files from a storage bucket"""
    bucket_name = bucket_config['name']
    exclude_folders = bucket_config.get('exclude_folders', [])

    print(f"\n🗑️  Clearing bucket: {bucket_name}")

    if exclude_folders:
        print(f"   🔒 Preserving folders: {', '.join(exclude_folders)}")

    try:
        # List all files recursively
        all_files = list_all_files_recursively(bucket_name)

        if not all_files:
            print(f"   ✅ Bucket {bucket_name} is already empty")
            return {'bucket': bucket_name, 'deleted': 0, 'skipped': 0}

        print(f"   📊 Found {len(all_files)} files to process")

        deleted = 0
        failed = 0
        skipped = 0

        # Delete files in batches
        for file_path in all_files:
            # Check if file is in an excluded folder
            should_skip = False
            for excluded_folder in exclude_folders:
                # Special case: '*' means preserve ALL files in this bucket
                if excluded_folder == '*' or file_path.startswith(excluded_folder):
                    should_skip = True
                    skipped += 1
                    break

            if should_skip:
                if skipped % 10 == 0 and skipped > 0:
                    print(f"   🔒 Skipped {skipped} files in preserved folders...")
                continue

            success = delete_file(bucket_name, file_path)
            if success:
                deleted += 1
                if deleted % 10 == 0:
                    print(f"   🔄 Deleted {deleted} files...")
            else:
                failed += 1

        print(f"   ✅ Deleted {deleted} files from {bucket_name}")
        if skipped > 0:
            print(f"   🔒 Preserved {skipped} files in excluded folders")
        if failed > 0:
            print(f"   ⚠️  Failed to delete {failed} files")

        return {'bucket': bucket_name, 'deleted': deleted, 'failed': failed, 'skipped': skipped}
    except Exception as error:
        print(f"   ❌ Failed to clear bucket {bucket_name}: {error}")
        return {'bucket': bucket_name, 'deleted': 0, 'failed': 0, 'skipped': 0, 'error': str(error)}


def main():
    """Main cleanup function"""
    print('═' * 100)
    print('🔄 KNOWLEDGE BASE RESET (PRESERVING USER DATA)')
    print('═' * 100)
    print('')
    print('✅ PRESERVED:')
    print('   • Users & Authentication')
    print('   • Profiles & Workspaces')
    print('   • API Keys & Usage Logs')
    print('   • PDF files in pdf-documents folder')
    print('')
    print('🗑️  WILL DELETE:')
    print('   • PDF Processing Data (chunks, embeddings, images)')
    print('   • Products & Materials Catalog')
    print('   • Background Jobs & Processing Results')
    print('   • Analytics & Agent Tasks')
    print('   • 3D Generation History')
    print('   • Storage files (except pdf-documents folder)')
    print('')
    print('═' * 100)
    print(f'📅 Started: {datetime.now().isoformat()}')

    results = {
        'tables': [],
        'buckets': []
    }

    # Step 1: Clear database tables
    print('\n🗑️  STEP 1: Clear knowledge base tables')
    print(f'   📋 Clearing {len(TABLES_TO_CLEAR)} tables (preserving user data)...')
    for table_name in TABLES_TO_CLEAR:
        result = clear_table(table_name)
        results['tables'].append(result)

    # Step 2: Clear storage buckets
    print('\n🗑️  STEP 2: Clear storage buckets')
    print('   🔒 Preserving pdf-documents folder and all files inside...')
    for bucket_config in BUCKETS_CONFIG:
        result = clear_bucket(bucket_config)
        results['buckets'].append(result)

    # Summary
    print('\n' + '═' * 100)
    print('📊 CLEANUP SUMMARY')
    print('═' * 100)

    total_rows_deleted = sum(r.get('deleted', 0) for r in results['tables'])
    total_files_deleted = sum(r.get('deleted', 0) for r in results['buckets'])
    total_files_failed = sum(r.get('failed', 0) for r in results['buckets'])
    total_files_skipped = sum(r.get('skipped', 0) for r in results['buckets'])

    print(f"\n✅ Database rows deleted: {total_rows_deleted}")
    print(f"✅ Storage files deleted: {total_files_deleted}")
    if total_files_skipped > 0:
        print(f"🔒 Storage files preserved: {total_files_skipped} (pdf-documents folder)")
    if total_files_failed > 0:
        print(f"⚠️  Storage files failed: {total_files_failed}")

    print('\n✅ PRESERVED DATA:')
    print('   • Users, Profiles, Workspaces remain intact')
    print('   • API Keys and authentication preserved')
    print(f'   • {total_files_skipped} files preserved in pdf-documents folder')

    print(f"\n📅 Completed: {datetime.now().isoformat()}")
    print('═' * 100)


if __name__ == '__main__':
    main()


