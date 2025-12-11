#!/usr/bin/env python3
"""
Complete Database Reset Script (Python Version)

This script performs a comprehensive cleanup of all user-generated data while preserving
system configuration and authentication data.

DELETES:
1. Knowledge base data (chunks, embeddings, products, images, etc.)
2. Agent chat conversations and messages
3. CRM contacts and relationships
4. All jobs (background_jobs, job_checkpoints, job_progress)
5. Moodboards and moodboard items
6. 3D generation history
7. All analytics data
8. All metadata values and relevancy relationships
9. All quotes and quote items
10. All files from storage buckets EXCEPT pdf-documents folder

PRESERVES:
1. User data (users, profiles, workspaces, API keys)
2. Global upsells (admin-managed upsell items)
3. Global timeline elements (timeline_steps)
4. Material metadata field definitions (schema)
5. PDF files in pdf-documents folder
6. System settings and configuration
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
# Order matters! Delete child tables before parent tables
TABLES_TO_CLEAR = [
    # Agent Chat System (DELETE)
    'agent_chat_messages',           # Chat messages (child of conversations)
    'agent_chat_conversations',      # Chat conversations
    'agent_uploaded_files',          # Files uploaded in chat

    # CRM Contacts (DELETE)
    'crm_contact_relationships',     # Contact relationships (child of contacts)
    'crm_contacts',                  # CRM contacts

    # Quotes System (DELETE - except global upsells and timeline steps)
    'quote_timeline',                # Quote timeline progress (child of quotes)
    'quote_upsells',                 # Quote upsells junction (child of quotes)
    'quote_items',                   # Quote items (child of quotes)
    'quotes',                        # Quotes
    'status_tags',                   # Custom status tags
    # NOTE: 'upsells' and 'timeline_steps' are PRESERVED (global data)

    # Moodboards (DELETE)
    'moodboard_quote_requests',      # Moodboard quote requests (child of moodboards)
    'moodboard_products',            # Moodboard products (child of moodboards)
    'moodboard_items',               # Moodboard items (child of moodboards)
    'moodboards',                    # Moodboards

    # 3D Generation (DELETE)
    'generation_3d',                 # 3D generation history

    # Analytics (DELETE)
    'analytics_events',              # Analytics events
    'quality_metrics_daily',         # Daily quality metrics
    'quality_scoring_logs',          # Quality scoring logs
    'recommendation_analytics',      # Recommendation analytics (if exists)

    # Document Entities & Relationships (DELETE)
    'product_document_relationships', # Product-document entity relationships
    'document_entities',             # Document entities (certificates, logos, specs)

    # Relevancy Relationships (DELETE)
    'product_chunk_relationships',   # Product-chunk relevancies
    'chunk_image_relationships',     # Chunk-image relevancies
    'product_image_relationships',   # Product-image relevancies

    # Metadata (DELETE)
    'metafield_values',              # Metafield values (child of metadata fields)
    # NOTE: 'material_metadata_fields' is PRESERVED (schema definition)

    # PDF Processing & Knowledge Base (DELETE)
    'job_checkpoints',               # Job checkpoints (child of background_jobs)
    'job_progress',                  # Job progress tracking (child of background_jobs)
    'ai_analysis_queue',             # AI analysis queue
    'image_processing_queue',        # Image processing queue
    'embeddings',                    # Text and image embeddings
    'document_images',               # Extracted images from PDFs
    'document_chunks',               # Semantic text chunks
    'products',                      # Extracted products
    'background_jobs',               # Processing jobs
    'documents',                     # PDF documents metadata
    'processed_documents',           # Processed document records

    # Materials & Catalog (DELETE)
    'materials_catalog',             # Materials catalog entries
    'material_visual_analysis',      # Visual analysis results

    # Processing & Quality (DELETE)
    'processing_results',            # Processing results

    # Agent Tasks (DELETE)
    'agent_tasks',                   # Agent task records

    # Web Scraping (DELETE)
    'scraped_materials_temp',        # Temporary scraped materials
    'scraping_sessions',             # Scraping sessions
    'scraping_pages',                # Scraping pages

    # Data Import (DELETE)
    'data_import_jobs',              # Data import jobs (if exists)
    'data_import_history',           # Data import history (if exists)
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
    print('🔄 COMPLETE DATABASE RESET (PRESERVING USER DATA & SYSTEM CONFIG)')
    print('═' * 100)
    print('')
    print('✅ PRESERVED:')
    print('   • Users & Authentication')
    print('   • Profiles & Workspaces')
    print('   • API Keys & Usage Logs')
    print('   • Global Upsells (admin-managed)')
    print('   • Global Timeline Elements (timeline_steps)')
    print('   • Material Metadata Field Definitions (schema)')
    print('   • PDF files in pdf-documents folder')
    print('   • System Settings & Configuration')
    print('')
    print('🗑️  WILL DELETE:')
    print('   • Agent Chat Conversations & Messages')
    print('   • CRM Contacts & Relationships')
    print('   • All Jobs (background_jobs, checkpoints, progress)')
    print('   • Moodboards & Moodboard Items')
    print('   • 3D Generation History')
    print('   • All Analytics Data')
    print('   • All Metadata Values & Relevancy Relationships')
    print('   • All Quotes & Quote Items')
    print('   • PDF Processing Data (chunks, embeddings, images)')
    print('   • Products & Materials Catalog')
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
    print('   • Global Upsells (admin-managed upsell items)')
    print('   • Global Timeline Elements (timeline_steps)')
    print('   • Material Metadata Field Definitions (schema)')
    print(f'   • {total_files_skipped} files preserved in pdf-documents folder')
    print('   • System Settings & Configuration')

    print(f"\n📅 Completed: {datetime.now().isoformat()}")
    print('═' * 100)


if __name__ == '__main__':
    main()


