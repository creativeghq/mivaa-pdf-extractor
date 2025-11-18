#!/usr/bin/env python3
"""
Populate material_properties table with comprehensive property definitions.

This script defines 50+ material properties across 9 categories that will be used
for AI extraction and prototype validation.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.supabase_client import get_supabase_client
from datetime import datetime


MATERIAL_PROPERTIES = [
    # Material Properties Category
    {
        "property_key": "material_type",
        "name": "Material Type",
        "display_name": "Material Type",
        "description": "Primary material classification (e.g., ceramic, porcelain, wood, metal)",
        "data_type": "enum",
        "is_required": True,
        "is_searchable": True,
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 1
    },
    {
        "property_key": "composition",
        "name": "Composition",
        "display_name": "Material Composition",
        "description": "Chemical or physical composition of the material",
        "data_type": "text",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 2
    },
    {
        "property_key": "finish",
        "name": "Finish",
        "display_name": "Surface Finish",
        "description": "Surface finish type (e.g., glossy, matte, satin, textured)",
        "data_type": "enum",
        "is_searchable": True,
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 3
    },
    {
        "property_key": "texture",
        "name": "Texture",
        "display_name": "Surface Texture",
        "description": "Physical texture characteristics",
        "data_type": "text",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 4
    },
    {
        "property_key": "pattern",
        "name": "Pattern",
        "display_name": "Pattern Type",
        "description": "Visual pattern or design",
        "data_type": "text",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 5
    },
    
    # Dimensions Category
    {
        "property_key": "length",
        "name": "Length",
        "display_name": "Length",
        "description": "Length dimension in mm or cm",
        "data_type": "number",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 10
    },
    {
        "property_key": "width",
        "name": "Width",
        "display_name": "Width",
        "description": "Width dimension in mm or cm",
        "data_type": "number",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 11
    },
    {
        "property_key": "thickness",
        "name": "Thickness",
        "display_name": "Thickness",
        "description": "Thickness dimension in mm",
        "data_type": "number",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 12
    },
    {
        "property_key": "weight",
        "name": "Weight",
        "display_name": "Weight",
        "description": "Weight per unit (kg/m² or kg/piece)",
        "data_type": "number",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 13
    },
    
    # Appearance Category
    {
        "property_key": "color",
        "name": "Color",
        "display_name": "Color",
        "description": "Primary color or color palette",
        "data_type": "text",
        "is_searchable": True,
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 20
    },
    {
        "property_key": "color_code",
        "name": "Color Code",
        "display_name": "Color Code",
        "description": "Manufacturer color code or reference",
        "data_type": "text",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 21
    },
    {
        "property_key": "gloss_level",
        "name": "Gloss Level",
        "display_name": "Gloss Level",
        "description": "Gloss measurement or classification",
        "data_type": "text",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 22
    },
    {
        "property_key": "transparency",
        "name": "Transparency",
        "display_name": "Transparency",
        "description": "Transparency level (opaque, translucent, transparent)",
        "data_type": "enum",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 23
    },

    # Performance Category
    {
        "property_key": "water_resistance",
        "name": "Water Resistance",
        "display_name": "Water Resistance",
        "description": "Water resistance rating or classification",
        "data_type": "enum",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 30
    },
    {
        "property_key": "slip_resistance",
        "name": "Slip Resistance",
        "display_name": "Slip Resistance",
        "description": "Slip resistance rating (R9-R13, A-C, etc.)",
        "data_type": "enum",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 31
    },
    {
        "property_key": "fire_rating",
        "name": "Fire Rating",
        "display_name": "Fire Rating",
        "description": "Fire resistance classification",
        "data_type": "enum",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 32
    },
    {
        "property_key": "durability_rating",
        "name": "Durability Rating",
        "display_name": "Durability Rating",
        "description": "Durability or wear resistance rating (PEI, etc.)",
        "data_type": "enum",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 33
    },
    {
        "property_key": "frost_resistance",
        "name": "Frost Resistance",
        "display_name": "Frost Resistance",
        "description": "Frost resistance capability",
        "data_type": "boolean",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 34
    },
    {
        "property_key": "chemical_resistance",
        "name": "Chemical Resistance",
        "display_name": "Chemical Resistance",
        "description": "Resistance to chemicals and cleaning agents",
        "data_type": "enum",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 35
    },

    # Application Category
    {
        "property_key": "recommended_use",
        "name": "Recommended Use",
        "display_name": "Recommended Use",
        "description": "Recommended application areas (indoor, outdoor, wall, floor)",
        "data_type": "text",
        "is_searchable": True,
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 40
    },
    {
        "property_key": "installation_method",
        "name": "Installation Method",
        "display_name": "Installation Method",
        "description": "Installation technique or method",
        "data_type": "text",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 41
    },
    {
        "property_key": "suitable_for_underfloor_heating",
        "name": "Suitable for Underfloor Heating",
        "display_name": "Underfloor Heating Compatible",
        "description": "Compatible with underfloor heating systems",
        "data_type": "boolean",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 42
    },

    # Compliance Category
    {
        "property_key": "certifications",
        "name": "Certifications",
        "display_name": "Certifications",
        "description": "Quality certifications and standards",
        "data_type": "array",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 50
    },
    {
        "property_key": "standards",
        "name": "Standards",
        "display_name": "Standards",
        "description": "Compliance standards (ISO, EN, ASTM, etc.)",
        "data_type": "array",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 51
    },
    {
        "property_key": "eco_friendly",
        "name": "Eco Friendly",
        "display_name": "Eco-Friendly",
        "description": "Environmental sustainability features",
        "data_type": "boolean",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 52
    },
    {
        "property_key": "recyclable",
        "name": "Recyclable",
        "display_name": "Recyclable",
        "description": "Material recyclability",
        "data_type": "boolean",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 53
    },

    # Design Category
    {
        "property_key": "designer",
        "name": "Designer",
        "display_name": "Designer",
        "description": "Product designer name",
        "data_type": "text",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 60
    },
    {
        "property_key": "studio",
        "name": "Studio",
        "display_name": "Design Studio",
        "description": "Design studio or firm",
        "data_type": "text",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 61
    },
    {
        "property_key": "collection",
        "name": "Collection",
        "display_name": "Collection",
        "description": "Product collection or series name",
        "data_type": "text",
        "is_searchable": True,
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 62
    },
    {
        "property_key": "style",
        "name": "Style",
        "display_name": "Design Style",
        "description": "Design style (modern, classic, rustic, etc.)",
        "data_type": "enum",
        "is_searchable": True,
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 63
    },

    # Manufacturing Category
    {
        "property_key": "manufacturing_process",
        "name": "Manufacturing Process",
        "display_name": "Manufacturing Process",
        "description": "Production method or technique",
        "data_type": "text",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 70
    },
    {
        "property_key": "country_of_origin",
        "name": "Country of Origin",
        "display_name": "Country of Origin",
        "description": "Manufacturing country",
        "data_type": "text",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 71
    },
    {
        "property_key": "manufacturer",
        "name": "Manufacturer",
        "display_name": "Manufacturer",
        "description": "Manufacturer or brand name",
        "data_type": "text",
        "is_searchable": True,
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 72
    },

    # Commercial Category
    {
        "property_key": "sku",
        "name": "SKU",
        "display_name": "SKU",
        "description": "Stock keeping unit / product code",
        "data_type": "text",
        "is_searchable": True,
        "is_ai_extractable": True,
        "display_order": 80
    },
    {
        "property_key": "availability_status",
        "name": "Availability Status",
        "display_name": "Availability",
        "description": "Product availability status",
        "data_type": "enum",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 81
    },
    {
        "property_key": "price_range",
        "name": "Price Range",
        "display_name": "Price Range",
        "description": "Price category or range",
        "data_type": "enum",
        "is_filterable": True,
        "is_ai_extractable": True,
        "display_order": 82
    },
]


async def populate_properties():
    """Populate material_properties table with property definitions."""
    supabase = get_supabase_client()

    print(f"🚀 Populating material_properties table with {len(MATERIAL_PROPERTIES)} properties...")

    # Check if properties already exist
    existing = supabase.client.table('material_properties').select('property_key').execute()
    existing_keys = {row['property_key'] for row in existing.data}

    inserted = 0
    updated = 0
    skipped = 0

    for prop in MATERIAL_PROPERTIES:
        if prop['property_key'] in existing_keys:
            # Update existing property
            result = supabase.client.table('material_properties').update({
                **prop,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('property_key', prop['property_key']).execute()

            if result.data:
                updated += 1
                print(f"  ✅ Updated: {prop['property_key']}")
            else:
                skipped += 1
                print(f"  ⏭️  Skipped: {prop['property_key']}")
        else:
            # Insert new property
            result = supabase.client.table('material_properties').insert({
                **prop,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }).execute()

            if result.data:
                inserted += 1
                print(f"  ✨ Inserted: {prop['property_key']}")
            else:
                skipped += 1
                print(f"  ❌ Failed: {prop['property_key']}")

    print(f"\n✅ Complete!")
    print(f"   Inserted: {inserted}")
    print(f"   Updated: {updated}")
    print(f"   Skipped: {skipped}")
    print(f"   Total: {len(MATERIAL_PROPERTIES)}")


if __name__ == "__main__":
    asyncio.run(populate_properties())


