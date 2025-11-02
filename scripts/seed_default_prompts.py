"""
Seed Default Extraction Prompts to Database

This script creates default prompts in the extraction_prompts table.
These prompts serve as:
1. Fallback when no custom prompts are defined
2. Starting templates for admins to customize
3. Best-practice examples for extraction

Run this script once to populate the database with default prompts.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.supabase_client import get_supabase_client

# Default workspace ID
DEFAULT_WORKSPACE_ID = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e"

# Default prompts for each stage and category
DEFAULT_PROMPTS = {
    # DISCOVERY STAGE
    ("discovery", "products"): {
        "name": "Product Discovery - Default",
        "template": """Analyze this PDF catalog and identify ALL products with complete metadata.

**REQUIRED INFORMATION:**
- Product name and all variants (colors, finishes, patterns)
- Page ranges where product appears (be precise)
- Designer/brand/studio information
- Material composition and finish details
- ALL dimensions and sizes mentioned (e.g., "15×38", "20×40", "8×45")
- Technical specifications (metafields):
  * Slip resistance ratings (R9, R10, R11, R12, R13)
  * Fire rating classifications (A1, A2, B, C, D, E, F)
  * Water absorption class (Class 1, 2, 3, etc.)
  * Thickness measurements (in mm)
  * Weight per unit
  * Any other technical properties

**QUALITY REQUIREMENTS:**
- Minimum confidence score: 0.7
- Include page numbers for ALL references
- Identify product boundaries clearly (where one product ends and another begins)
- Extract EVERY product, not just the first few

**SPECIAL INSTRUCTIONS:**
- If a product has multiple variants, list them all
- If dimensions are given in ranges, extract all values
- Pay attention to product families and collections
- Note any cross-references between products

**OUTPUT:** Return comprehensive JSON with all products, their metadata, and confidence scores.""",
        "description": "Comprehensive product discovery with full metadata extraction",
        "quality_threshold": 0.7,
        "is_active": True
    },
    
    ("discovery", "certificates"): {
        "name": "Certificate Discovery - Default",
        "template": """Identify ALL certificates, certifications, and compliance documents in this PDF.

**REQUIRED INFORMATION:**
- Certificate name and full title
- Certificate type (ISO, CE, quality management, environmental, safety, etc.)
- Issuing authority/certification body
- Issue date and expiry date (if mentioned)
- Standards referenced (e.g., "ISO 9001:2015", "EN 14411", "CE marking")
- Scope of certification
- Page numbers where certificate appears

**QUALITY REQUIREMENTS:**
- Minimum confidence score: 0.7
- Verify certificate authenticity markers (logos, seals, signatures)
- Distinguish between actual certificates and certificate mentions

**SPECIAL INSTRUCTIONS:**
- Look for certification logos and marks
- Check for validity periods
- Note any certificate numbers or reference codes
- Identify certification scope (what is certified)

**OUTPUT:** Return JSON array with all certificates and their complete details.""",
        "description": "Certificate and compliance document discovery",
        "quality_threshold": 0.7,
        "is_active": True
    },
    
    ("discovery", "logos"): {
        "name": "Logo Discovery - Default",
        "template": """Identify ALL logos, brand marks, and certification marks in this document.

**REQUIRED INFORMATION:**
- Logo name/description
- Logo type: company brand, certification mark, quality seal, partner logo, etc.
- Associated brand/organization
- Page numbers and approximate positions
- Logo quality and clarity assessment

**QUALITY REQUIREMENTS:**
- Minimum confidence score: 0.7
- Distinguish between decorative elements and official logos
- Identify primary vs secondary logos

**SPECIAL INSTRUCTIONS:**
- Company logos (main brand identity)
- Certification logos (ISO, CE, quality marks)
- Partner/supplier logos
- Award or recognition logos
- Do NOT include decorative patterns or design elements

**OUTPUT:** Return JSON array with all logos categorized by type.""",
        "description": "Logo and brand mark identification",
        "quality_threshold": 0.7,
        "is_active": True
    },
    
    ("discovery", "specifications"): {
        "name": "Specification Discovery - Default",
        "template": """Extract ALL technical specifications, data sheets, and instructional content.

**REQUIRED INFORMATION:**
- Specification name/title
- Specification type: technical data, installation guide, maintenance instructions, safety information, etc.
- Technical parameters and values
- Performance data and test results
- Compliance information
- Page numbers

**QUALITY REQUIREMENTS:**
- Minimum confidence score: 0.7
- Preserve numerical accuracy
- Maintain units of measurement
- Keep technical terminology exact

**SPECIAL INSTRUCTIONS:**
- Technical data sheets
- Installation procedures
- Maintenance guidelines
- Safety instructions
- Performance specifications
- Test results and certifications

**OUTPUT:** Return JSON array with all specifications and their complete technical details.""",
        "description": "Technical specification and data sheet extraction",
        "quality_threshold": 0.7,
        "is_active": True
    }
}


async def seed_default_prompts():
    """Seed database with default extraction prompts"""
    
    print("=" * 80)
    print("SEEDING DEFAULT EXTRACTION PROMPTS")
    print("=" * 80)
    
    supabase = get_supabase_client()
    
    created_count = 0
    updated_count = 0
    skipped_count = 0
    
    for (stage, category), prompt_data in DEFAULT_PROMPTS.items():
        try:
            print(f"\n📝 Processing: {stage}/{category}")
            
            # Check if prompt already exists
            existing = supabase.table("extraction_prompts").select("*").eq(
                "workspace_id", DEFAULT_WORKSPACE_ID
            ).eq(
                "stage", stage
            ).eq(
                "category", category
            ).execute()
            
            if existing.data and len(existing.data) > 0:
                print(f"   ⚠️  Prompt already exists (ID: {existing.data[0]['id']})")
                print(f"   Skipping... (use update script to modify existing prompts)")
                skipped_count += 1
                continue
            
            # Create new prompt (matching actual database schema)
            prompt_record = {
                "workspace_id": DEFAULT_WORKSPACE_ID,
                "stage": stage,
                "category": category,
                "prompt_template": prompt_data["template"],  # Column is 'prompt_template' not 'template'
                "system_prompt": prompt_data.get("description", ""),  # Use description as system_prompt
                "is_custom": False,  # These are default prompts
                "version": 1,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = supabase.table("extraction_prompts").insert(prompt_record).execute()
            
            if result.data:
                print(f"   ✅ Created prompt (ID: {result.data[0]['id']})")
                print(f"      Name: {prompt_data['name']}")
                print(f"      Template length: {len(prompt_data['template'])} chars")
                created_count += 1
            else:
                print(f"   ❌ Failed to create prompt")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Created: {created_count}")
    print(f"⚠️  Skipped: {skipped_count}")
    print(f"❌ Failed: {len(DEFAULT_PROMPTS) - created_count - skipped_count}")
    print(f"📊 Total: {len(DEFAULT_PROMPTS)}")
    print("=" * 80)
    
    if created_count > 0:
        print("\n✨ Default prompts have been seeded successfully!")
        print("   Admins can now customize these prompts through the admin panel.")
    elif skipped_count == len(DEFAULT_PROMPTS):
        print("\n✨ All default prompts already exist in the database.")
    else:
        print("\n⚠️  Some prompts were not created. Check errors above.")


if __name__ == "__main__":
    asyncio.run(seed_default_prompts())

