#!/usr/bin/env python3
"""
Run database migration to add product_id column to document_chunks table.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.core.supabase_client import get_supabase_client
from app.core.logging_config import logger


async def run_migration():
    """Run the migration to add product_id column."""
    try:
        logger.info("🔧 Starting database migration...")
        logger.info("   Adding product_id column to document_chunks table")
        
        # Get Supabase client
        supabase_client = get_supabase_client()
        client = supabase_client.client
        
        # Migration SQL
        migration_sql = """
        ALTER TABLE document_chunks 
        ADD COLUMN IF NOT EXISTS product_id UUID REFERENCES products(id) ON DELETE CASCADE;
        """
        
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_document_chunks_product_id 
        ON document_chunks(product_id);
        """
        
        # Note: Supabase Python client doesn't support raw SQL execution directly
        # We need to use the PostgREST API or create a stored procedure
        
        logger.info("✅ Migration SQL prepared:")
        logger.info(f"   {migration_sql.strip()}")
        logger.info(f"   {index_sql.strip()}")
        logger.info("")
        logger.info("⚠️  Please run this SQL manually in Supabase SQL Editor:")
        logger.info("   1. Go to https://supabase.com/dashboard/project/bgbavxtjlbvgplozizxu/sql")
        logger.info("   2. Paste the SQL above")
        logger.info("   3. Click 'Run'")
        logger.info("")
        logger.info("   OR use the Supabase CLI:")
        logger.info("   supabase db execute --file migrations/add_product_id_to_chunks.sql")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_migration())
    sys.exit(0 if success else 1)

