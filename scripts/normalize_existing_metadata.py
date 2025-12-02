"""
Migration Script: Normalize Existing Product Metadata

Normalizes metadata for all existing products in the database to use standardized field names.

Usage:
    python scripts/normalize_existing_metadata.py --workspace-id <workspace_id> [--dry-run]

Options:
    --workspace-id: Workspace ID to process (required)
    --dry-run: Show what would be changed without applying changes
    --verbose: Show detailed normalization reports
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime

# Load environment variables from systemd service if running on server
# This ensures the script has access to SUPABASE_URL and other required env vars
if os.path.exists('/etc/systemd/system/mivaa-pdf-extractor.service'):
    import subprocess
    try:
        # Extract environment variables from systemd service file
        result = subprocess.run(
            ['systemctl', 'show', 'mivaa-pdf-extractor.service', '--property=Environment'],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse Environment=KEY=VALUE format
        env_line = result.stdout.strip()
        if env_line.startswith('Environment='):
            env_vars = env_line[12:].split()  # Remove 'Environment=' prefix
            for var in env_vars:
                if '=' in var:
                    key, value = var.split('=', 1)
                    os.environ[key] = value
    except Exception as e:
        print(f"Warning: Could not load environment from systemd service: {e}")

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import get_settings
from app.services.supabase_client import get_supabase_client, initialize_supabase
from app.services.metadata_normalizer import normalize_metadata, get_normalization_report


async def normalize_products(workspace_id: str, dry_run: bool = False, verbose: bool = False):
    """
    Normalize metadata for all products in a workspace.

    Args:
        workspace_id: Workspace ID to process
        dry_run: If True, show changes without applying them
        verbose: If True, show detailed normalization reports
    """
    # Initialize Supabase client
    settings = get_settings()
    initialize_supabase(settings)
    supabase_client = get_supabase_client()
    supabase = supabase_client.client
    
    print(f"\n{'='*80}")
    print(f"METADATA NORMALIZATION MIGRATION")
    print(f"{'='*80}")
    print(f"Workspace ID: {workspace_id}")
    print(f"Mode: {'DRY RUN (no changes will be applied)' if dry_run else 'LIVE (changes will be applied)'}")
    print(f"{'='*80}\n")
    
    # Fetch all products
    print("📥 Fetching products from database...")
    response = supabase.table("products").select("id, name, metadata").eq("workspace_id", workspace_id).execute()
    
    if not response.data:
        print("❌ No products found for this workspace")
        return
    
    products = response.data
    print(f"✅ Found {len(products)} products\n")
    
    # Process each product
    total_normalized = 0
    total_fields_normalized = 0
    
    for i, product in enumerate(products, 1):
        product_id = product["id"]
        product_name = product["name"]
        original_metadata = product["metadata"]
        
        print(f"\n[{i}/{len(products)}] Processing: {product_name}")
        print(f"Product ID: {product_id}")
        
        if not original_metadata:
            print("  ⚠️  No metadata found - skipping")
            continue
        
        # Normalize metadata
        normalized_metadata = normalize_metadata(original_metadata)
        
        # Get normalization report
        report = get_normalization_report(original_metadata, normalized_metadata)
        
        if report["fields_normalized"] == 0:
            print("  ✅ Already normalized - no changes needed")
            continue
        
        # Show changes
        print(f"  🔄 Normalizing {report['fields_normalized']} fields:")
        
        if verbose:
            for change in report["changes"]:
                print(f"     • {change['category']}.{change['from']} → {change['to']}")
        else:
            # Show summary
            categories_affected = set(c["category"] for c in report["changes"])
            print(f"     Categories affected: {', '.join(categories_affected)}")
        
        # Apply changes (unless dry run)
        if not dry_run:
            try:
                update_response = supabase.table("products").update({
                    "metadata": normalized_metadata,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", product_id).execute()
                
                if update_response.data:
                    print(f"  ✅ Successfully normalized")
                    total_normalized += 1
                    total_fields_normalized += report["fields_normalized"]
                else:
                    print(f"  ❌ Failed to update")
            except Exception as e:
                print(f"  ❌ Error updating: {e}")
        else:
            print(f"  ℹ️  Would normalize (dry run)")
            total_normalized += 1
            total_fields_normalized += report["fields_normalized"]
    
    # Summary
    print(f"\n{'='*80}")
    print(f"MIGRATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total products processed: {len(products)}")
    print(f"Products normalized: {total_normalized}")
    print(f"Total fields normalized: {total_fields_normalized}")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN MODE - No changes were applied")
        print(f"Run without --dry-run to apply changes")
    else:
        print(f"\n✅ Migration completed successfully!")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Normalize existing product metadata")
    parser.add_argument("--workspace-id", required=True, help="Workspace ID to process")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    parser.add_argument("--verbose", action="store_true", help="Show detailed normalization reports")
    
    args = parser.parse_args()
    
    # Run migration
    asyncio.run(normalize_products(
        workspace_id=args.workspace_id,
        dry_run=args.dry_run,
        verbose=args.verbose
    ))


if __name__ == "__main__":
    main()

