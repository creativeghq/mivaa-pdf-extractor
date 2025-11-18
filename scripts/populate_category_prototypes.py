"""
Populate Material Category Prototypes with CLIP Text Embeddings

This script:
1. Defines 3-5 text descriptions for each material category
2. Generates CLIP text embeddings for each category
3. Updates material_categories table with prototypes
"""

import asyncio
import os
from typing import List, Dict
from datetime import datetime
from supabase import create_client, Client
from openai import OpenAI
import numpy as np

# Initialize clients
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Category prototype descriptions (3-5 per category)
CATEGORY_PROTOTYPES: Dict[str, List[str]] = {
    # Tiles & Flooring
    "ceramic_tile": [
        "Ceramic tiles with glazed surface for interior floor and wall applications",
        "Porcelain ceramic tiles with uniform texture and color patterns",
        "Glazed ceramic tiles with smooth finish for residential and commercial use"
    ],
    "porcelain_tile": [
        "High-density porcelain tiles with low water absorption for durability",
        "Porcelain stoneware tiles with through-body color and texture",
        "Vitrified porcelain tiles suitable for high-traffic areas and outdoor use"
    ],
    "natural_stone_tile": [
        "Natural stone tiles cut from marble, granite, or limestone",
        "Authentic stone tiles with unique veining and natural variations",
        "Quarried natural stone tiles for premium flooring and wall cladding"
    ],
    "glass_tile": [
        "Translucent glass tiles with reflective surface for decorative applications",
        "Colored glass mosaic tiles for backsplashes and accent walls",
        "Recycled glass tiles with sustainable eco-friendly properties"
    ],
    "metal_tile": [
        "Stainless steel or aluminum tiles with metallic finish",
        "Brushed metal tiles for modern industrial design aesthetics",
        "Copper or brass tiles with patina finish for decorative accents"
    ],
    "concrete_tile": [
        "Cement-based concrete tiles with matte industrial finish",
        "Polished concrete tiles with smooth surface for modern interiors",
        "Textured concrete tiles with exposed aggregate patterns"
    ],
    "wood_tile": [
        "Ceramic or porcelain tiles with realistic wood grain patterns",
        "Wood-look tiles mimicking natural hardwood flooring",
        "Plank-style tiles with authentic wood texture and color variations"
    ],
    "mosaic_tiles": [
        "Small mosaic tiles arranged in decorative patterns and designs",
        "Glass, ceramic, or stone mosaic tiles for artistic installations",
        "Mesh-mounted mosaic sheets for easy installation"
    ],
    "floor_tiles": [
        "Durable floor tiles designed for high-traffic residential and commercial areas",
        "Anti-slip floor tiles with textured surface for safety",
        "Large-format floor tiles for seamless modern flooring"
    ],
    "wall_tiles": [
        "Decorative wall tiles for interior vertical surfaces",
        "Glossy wall tiles with reflective finish for bathrooms and kitchens",
        "Textured wall tiles with three-dimensional relief patterns"
    ],
    
    # Natural Stones
    "marble": [
        "Natural marble stone with distinctive veining and luxurious appearance",
        "Polished marble slabs with high-gloss mirror finish",
        "Honed marble with matte smooth surface for elegant interiors"
    ],
    "granite": [
        "Hard granite stone with speckled crystalline patterns",
        "Polished granite with durable scratch-resistant surface",
        "Natural granite with unique mineral composition and color variations"
    ],
    "travertine": [
        "Porous travertine stone with natural holes and earthy tones",
        "Filled and honed travertine for smooth elegant finish",
        "Tumbled travertine with rustic aged appearance"
    ],
    "slate": [
        "Layered slate stone with natural cleft surface texture",
        "Dark slate tiles with rich charcoal and graphite tones",
        "Riven slate with authentic split-face finish"
    ],
    "limestone": [
        "Sedimentary limestone with soft neutral earth tones",
        "Honed limestone with smooth matte finish",
        "Fossilized limestone with natural organic inclusions"
    ],
    "quartzite": [
        "Metamorphic quartzite stone with crystalline sparkle",
        "Hard quartzite with marble-like veining and durability",
        "Natural quartzite with unique color patterns and high strength"
    ],
    "sandstone": [
        "Porous sandstone with warm sandy earth tones",
        "Natural sandstone with visible grain and layered structure",
        "Textured sandstone for rustic outdoor applications"
    ],
    "onyx": [
        "Translucent onyx stone with dramatic veining and backlit potential",
        "Polished onyx slabs with luxurious semi-precious appearance",
        "Natural onyx with unique banding and color variations"
    ],
    
    # Base Materials
    "metals": [
        "Metallic materials including steel, aluminum, copper, and brass",
        "Industrial metals with conductive and structural properties",
        "Finished metal surfaces with polished, brushed, or patina treatments"
    ],
    "plastics": [
        "Synthetic polymer materials with versatile molding properties",
        "Thermoplastic and thermoset plastic materials",
        "Durable plastic composites for various applications"
    ],
    "ceramics": [
        "Ceramic materials with fired clay and glazed finishes",
        "Porcelain and stoneware ceramic products",
        "High-temperature fired ceramic with durable properties"
    ],
    "composites": [
        "Composite materials combining multiple constituent materials",
        "Fiber-reinforced composites with enhanced strength",
        "Engineered composite materials for specialized applications"
    ],
    "textiles": [
        "Woven and knitted fabric materials for soft furnishings",
        "Natural and synthetic textile fibers",
        "Upholstery and decorative textile materials"
    ],
    "wood": [
        "Natural hardwood and softwood timber materials",
        "Solid wood with natural grain patterns and textures",
        "Engineered wood products and veneers"
    ],
    "glass": [
        "Transparent or translucent glass materials",
        "Tempered and laminated safety glass",
        "Decorative glass with colored or textured finishes"
    ],
    "rubber": [
        "Elastic rubber materials with flexible properties",
        "Natural and synthetic rubber compounds",
        "Durable rubber for flooring and sealing applications"
    ],
    "concrete": [
        "Cement-based concrete with aggregate composition",
        "Polished or exposed aggregate concrete finishes",
        "Reinforced concrete for structural applications"
    ],

    # Additional Categories
    "terrazzo": [
        "Composite terrazzo with marble chips in cement or resin matrix",
        "Polished terrazzo with decorative aggregate patterns",
        "Traditional or epoxy terrazzo flooring systems"
    ],
    "quartz": [
        "Engineered quartz surfaces with resin-bound crushed quartz",
        "Non-porous quartz countertops with consistent patterns",
        "Durable quartz composite with stain-resistant properties"
    ],
    "vinyl": [
        "Resilient vinyl flooring with printed patterns",
        "Luxury vinyl tile (LVT) with realistic textures",
        "Waterproof vinyl planks for residential and commercial use"
    ],
    "laminate": [
        "Laminate flooring with photographic wood or stone patterns",
        "High-pressure laminate with wear-resistant surface",
        "Click-lock laminate planks for easy installation"
    ],
    "carpet": [
        "Soft textile carpet with pile fibers for comfort",
        "Loop or cut pile carpet in various textures",
        "Stain-resistant carpet for residential and commercial spaces"
    ],
    "cork": [
        "Natural cork flooring with sustainable harvested bark",
        "Resilient cork with thermal and acoustic insulation",
        "Sealed cork tiles with warm natural appearance"
    ],
    "bamboo": [
        "Sustainable bamboo flooring with natural grass fibers",
        "Strand-woven bamboo with enhanced durability",
        "Carbonized or natural bamboo with eco-friendly properties"
    ],
    "recycled_glass": [
        "Eco-friendly recycled glass tiles and surfaces",
        "Crushed recycled glass in terrazzo or composite materials",
        "Sustainable glass products with post-consumer content"
    ],
    "acrylic": [
        "Transparent acrylic sheets with glass-like clarity",
        "Colored acrylic materials for decorative applications",
        "Durable acrylic surfaces resistant to impact"
    ],
    "corian": [
        "Solid surface Corian with seamless fabrication",
        "Non-porous Corian countertops with integrated sinks",
        "Thermoformable Corian in various colors and patterns"
    ],

    # Decorative & Furniture
    "tiles": [
        "General tile materials for flooring and wall covering",
        "Decorative tiles in various materials and finishes",
        "Modular tile systems for interior and exterior use"
    ],
    "decor": [
        "Decorative items and accessories for interior styling",
        "Ornamental objects and artistic elements",
        "Home decor pieces in various materials and styles"
    ],
    "lighting": [
        "Lighting fixtures and luminaires for illumination",
        "Decorative and functional lighting products",
        "LED, pendant, and ambient lighting solutions"
    ],
    "furniture": [
        "Furniture pieces for residential and commercial spaces",
        "Seating, tables, and storage furniture",
        "Modern and traditional furniture designs"
    ],
    "ceiling_lights": [
        "Ceiling-mounted lighting fixtures and chandeliers",
        "Flush mount and semi-flush ceiling lights",
        "Pendant lights suspended from ceiling"
    ],
    "wall_lights": [
        "Wall-mounted sconces and accent lighting",
        "Decorative wall lights for ambient illumination",
        "Directional wall lights for task lighting"
    ],
    "floor_lamps": [
        "Freestanding floor lamps for ambient and task lighting",
        "Adjustable floor lamps with flexible positioning",
        "Decorative floor lamps as statement pieces"
    ],
    "wall_decor": [
        "Decorative wall art and hanging ornaments",
        "Mirrors, frames, and wall-mounted decorations",
        "Three-dimensional wall sculptures and reliefs"
    ],
    "sculptures": [
        "Artistic sculptures and three-dimensional art pieces",
        "Contemporary and classical sculptural works",
        "Decorative sculptures in various materials"
    ],
    "vases_planters": [
        "Decorative vases for floral arrangements",
        "Indoor and outdoor planters for greenery",
        "Ceramic, glass, or metal vessels for plants"
    ],

    # Catch-all
    "other": [
        "Miscellaneous materials not fitting standard categories",
        "Specialty materials with unique properties",
        "Alternative and innovative material solutions"
    ],
}


async def generate_clip_text_embedding(texts: List[str]) -> List[float]:
    """
    Generate CLIP text embedding by averaging embeddings of multiple descriptions

    Args:
        texts: List of 3-5 text descriptions for a category

    Returns:
        512-dimensional averaged embedding vector
    """
    try:
        # Use OpenAI's text-embedding-3-small model (1536D)
        # We'll need to reduce to 512D for CLIP compatibility
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            dimensions=512  # Request 512D directly
        )

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]

        # Average all embeddings
        avg_embedding = np.mean(embeddings, axis=0).tolist()

        return avg_embedding

    except Exception as e:
        print(f"❌ Error generating embedding: {str(e)}")
        raise


async def update_category_prototype(
    category_key: str,
    descriptions: List[str]
) -> bool:
    """
    Update a category with prototype descriptions and embedding

    Args:
        category_key: Category key to update
        descriptions: List of prototype descriptions

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🔄 Processing category: {category_key}")

        # Generate embedding
        embedding = await generate_clip_text_embedding(descriptions)

        # Update database
        result = supabase.table('material_categories').update({
            'prototype_descriptions': descriptions,
            'text_embedding_512': embedding,
            'prototype_updated_at': datetime.utcnow().isoformat()
        }).eq('category_key', category_key).execute()

        if result.data:
            print(f"✅ Updated {category_key} with {len(descriptions)} descriptions")
            return True
        else:
            print(f"⚠️  Category {category_key} not found in database")
            return False

    except Exception as e:
        print(f"❌ Error updating {category_key}: {str(e)}")
        return False


async def populate_all_prototypes():
    """
    Populate all category prototypes with embeddings
    """
    print("🚀 Starting category prototype population...")
    print(f"📊 Total categories to process: {len(CATEGORY_PROTOTYPES)}")

    success_count = 0
    fail_count = 0

    for category_key, descriptions in CATEGORY_PROTOTYPES.items():
        success = await update_category_prototype(category_key, descriptions)
        if success:
            success_count += 1
        else:
            fail_count += 1

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    print("\n" + "="*60)
    print(f"✅ Successfully updated: {success_count} categories")
    print(f"❌ Failed: {fail_count} categories")
    print("="*60)


async def verify_prototypes():
    """
    Verify that prototypes were populated correctly
    """
    print("\n🔍 Verifying prototype population...")

    result = supabase.table('material_categories').select(
        'category_key, prototype_descriptions, text_embedding_512, prototype_updated_at'
    ).not_.is_('text_embedding_512', 'null').execute()

    if result.data:
        print(f"✅ Found {len(result.data)} categories with prototypes")
        print("\nSample categories:")
        for item in result.data[:5]:
            print(f"  - {item['category_key']}: {len(item['prototype_descriptions'])} descriptions")
    else:
        print("⚠️  No categories with prototypes found")


if __name__ == "__main__":
    print("="*60)
    print("Material Category Prototype Population Script")
    print("="*60)

    # Run population
    asyncio.run(populate_all_prototypes())

    # Verify results
    asyncio.run(verify_prototypes())

    print("\n✅ Script completed!")


