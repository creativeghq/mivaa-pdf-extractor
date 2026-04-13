"""
Category Field Registry
=======================
Single source of truth for category-specific extraction fields.

Each category defines:
  - priority_fields: Fields the AI should actively hunt for in the PDF.
                     Grouped by section for prompt clarity.
  - extraction_hints: Category-specific tips telling the AI *where* to look
                      (e.g. "check icon strips at page bottom for PEI ratings").
  - skip_fields:     Field keys from other categories that should NOT appear
                     in the prompt (reduces hallucination / wasted tokens).
  - controlled_vocab: Fine-grained material_category values that map to this
                      upload category (used for validation).

The prompt builder (dynamic_metadata_extractor) combines:
  1. Universal critical fields (factory, material_category, zone_intent)
  2. Category priority fields (from this registry)
  3. Open discovery instruction ("extract anything else you find")

This keeps extraction focused yet flexible — the AI knows what to look for
but is never prevented from capturing unexpected attributes.
"""

from typing import Dict, List, Any


# ─── Registry ───────────────────────────────────────────────────────────────

CATEGORY_FIELD_REGISTRY: Dict[str, Dict[str, Any]] = {

    # ═══════════════════════════════════════════════════════════════════════
    # TILES
    # ═══════════════════════════════════════════════════════════════════════
    "tiles": {
        "display_name": "Tiles",
        "controlled_vocab": [
            "floor_tile", "wall_tile", "bathroom_tile", "shower_tile",
            "porcelain_tile", "ceramic_tile",
        ],
        "priority_fields": {
            "material_properties": [
                ("body_type", "Body type (e.g. white body, full body, red body)"),
                ("finish", "Surface finish (matte, gloss, polished, lappato, structured)"),
                ("material_subtype", "Subtype (glazed, unglazed, through-body, double-fired)"),
                ("thickness_mm", "Tile thickness in mm"),
                ("thickness_inch", "Tile thickness in inches"),
                ("rectified", "Whether tile is rectified (yes/no)"),
            ],
            "dimensions": [
                ("available_sizes", "All available tile formats as array, e.g. ['60x60 cm', '30x60 cm']"),
                ("format_code", "Factory format code (e.g. Q59, R10)"),
            ],
            "appearance": [
                ("colors", "Array of available color names"),
                ("primary_color_hex", "Primary hex color code"),
                ("patterns", "Array of patterns (solid, veined, geometric, 3D relief, etc.)"),
                ("texture", "Surface texture description"),
                ("shade_variation", "Shade variation V-rating (V1, V2, V3, V4)"),
                ("visual_effect", "Visual effect description (e.g. marble-look, concrete-look)"),
            ],
            "performance": [
                ("pei_rating", "PEI abrasion rating (I to V)"),
                ("slip_resistance", "Slip resistance R-value (R9, R10, R11, R12, R13) per DIN 51130"),
                ("water_absorption", "Water absorption class (BIa, BIb, BIIa, BIIb, BIII) per ISO 13006"),
                ("water_absorption_pct", "Water absorption percentage"),
                ("frost_resistance", "Frost resistant yes/no per ISO 10545-12"),
                ("breaking_strength", "Breaking strength in N per ISO 10545-4"),
                ("abrasion_resistance", "Deep abrasion resistance in mm³"),
                ("chemical_resistance", "Chemical resistance class (A, B, C)"),
                ("thermal_shock_resistance", "Thermal shock resistance pass/fail"),
                ("mohs_hardness", "Mohs hardness scale rating"),
            ],
            "packaging": [
                ("pieces_per_box", "Number of pieces per box"),
                ("patterns_count", "Number of distinct patterns/faces in box"),
                ("m2_per_box", "Coverage per box in m²"),
                ("sqft_per_box", "Coverage per box in sqft"),
                ("weight_per_box_kg", "Box weight in kg"),
                ("weight_per_box_lb", "Box weight in lb"),
                ("boxes_per_pallet", "Number of boxes per pallet"),
                ("m2_per_pallet", "Coverage per pallet in m²"),
                ("sqft_per_pallet", "Coverage per pallet in sqft"),
                ("weight_per_pallet_kg", "Pallet weight in kg"),
                ("weight_per_pallet_lb", "Pallet weight in lb"),
                ("pallet_dimensions_cm", "Pallet dimensions LxWxH in cm"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
                ("grout_mapei", "Mapei grout product recommendation"),
                ("grout_kerakoll", "Kerakoll grout product recommendation"),
                ("grout_isomat", "Isomat grout product recommendation"),
                ("grout_technica", "Technica grout product recommendation"),
                ("grout_color_codes", "Object mapping variant/color to grout dose codes"),
                ("grout_suppliers", "Array of grout supplier names"),
                ("vision_variants", "Array of variant objects [{sku, name, color, format, pattern}]"),
                ("grout_details", "Array of [{supplier, product, code, for_variant}]"),
            ],
            "application": [
                ("recommended_use", "Wall, Floor, or both"),
                ("installation_method", "Installation method (thin-set, mortar bed, adhesive)"),
                ("joint_width_mm", "Recommended joint/grout width in mm"),
                ("suitable_rooms", "Suitable room types (bathroom, kitchen, outdoor, etc.)"),
                ("underfloor_heating", "Compatible with underfloor heating yes/no"),
            ],
            "compliance": [
                ("certifications", "Array of certifications (ISO 9001, CE, EN 14411, LEED, etc.)"),
                ("standards", "Array of compliance standards"),
                ("eco_friendly", "Eco-friendly indicators"),
                ("fire_rating", "Fire classification (A1, A2, B, etc.)"),
            ],
            "care": [
                ("care_instructions", "Cleaning and care instructions"),
                ("maintenance", "Maintenance requirements"),
            ],
        },
        "extraction_hints": [
            "Packaging info is often in 'Iconography' or 'Packing' sections with small icons.",
            "Compliance/performance ratings may be on shared legend pages (often near the end of the catalog, pages 60+), not on each product page.",
            "Look for icon strips at the bottom of product pages — they encode PEI, slip resistance, frost resistance as small pictograms.",
            "Grout recommendations are often in separate tables matching color variants to dose numbers.",
            "DIN 51130 R-values (R9-R13) may appear in regulation/legend pages rather than product pages.",
            "Shade variation V-ratings (V1-V4) per ANSI A137.1 may be in the icon strip.",
        ],
        "skip_fields": [
            "wattage", "lumens", "color_temperature_k", "cri", "beam_angle",
            "btu", "kw_output", "energy_class", "thermostat_type", "fuel_type",
            "flow_rate", "flush_volume", "trap_type",
            "species", "janka_hardness", "grain_direction", "wear_layer",
            "coverage_per_litre", "dry_time", "voc_level", "sheen",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # WOOD
    # ═══════════════════════════════════════════════════════════════════════
    "wood": {
        "display_name": "Wood",
        "controlled_vocab": [
            "wood_flooring", "laminate", "vinyl_flooring",
            "hardwood", "engineered_wood", "parquet",
        ],
        "priority_fields": {
            "material_properties": [
                ("species", "Wood species (oak, walnut, maple, teak, etc.)"),
                ("grade", "Wood grade (Select, #1 Common, Rustic, AB, CD, etc.)"),
                ("construction", "Construction type (solid, engineered, 3-layer, multi-layer)"),
                ("core_material", "Core material for engineered (HDF, plywood, spruce)"),
                ("wear_layer_mm", "Wear layer / top layer thickness in mm"),
                ("total_thickness_mm", "Total plank thickness in mm"),
                ("finish_type", "Finish type (lacquered, oiled, brushed, smoked, unfinished)"),
                ("finish_coats", "Number of finish coats"),
                ("bevel_edge", "Edge profile (square, micro-bevel, 4-sided bevel)"),
                ("surface_treatment", "Surface treatment (brushed, hand-scraped, wire-brushed, saw-cut)"),
            ],
            "dimensions": [
                ("plank_length_mm", "Plank length in mm"),
                ("plank_width_mm", "Plank width in mm"),
                ("available_sizes", "Array of available plank dimensions"),
            ],
            "appearance": [
                ("colors", "Array of color/stain names"),
                ("primary_color_hex", "Primary hex color code"),
                ("grain_pattern", "Grain pattern (straight, cathedral, quarter-sawn, rift-sawn)"),
                ("knot_character", "Knot character (clear, few knots, rustic, character)"),
                ("shade_variation", "Shade variation across planks (low, medium, high)"),
            ],
            "performance": [
                ("janka_hardness", "Janka hardness rating (lbf or N)"),
                ("ac_rating", "Abrasion class rating AC1-AC6 (laminate/vinyl)"),
                ("wear_rating", "Wear class rating (21-34)"),
                ("moisture_content_pct", "Moisture content percentage"),
                ("dimensional_stability", "Dimensional stability rating"),
                ("impact_resistance", "Impact resistance class (IC1-IC3)"),
                ("underfloor_heating", "Compatible with underfloor heating yes/no"),
                ("max_surface_temp_c", "Max surface temperature for underfloor heating in celsius"),
                ("fire_rating", "Fire classification (Cfl-s1, Bfl-s1, etc.)"),
                ("slip_resistance", "Slip resistance if applicable"),
                ("acoustic_rating_db", "Impact sound reduction in dB"),
                ("formaldehyde_class", "Formaldehyde emission class (E0, E1, CARB2)"),
            ],
            "installation": [
                ("click_system", "Click/locking system type (Uniclic, 5G, T-lock, tongue & groove)"),
                ("installation_method", "Installation method (floating, glue-down, nail-down)"),
                ("subfloor_requirements", "Subfloor requirements"),
                ("expansion_gap_mm", "Required expansion gap in mm"),
                ("acclimation_days", "Acclimation period in days"),
                ("underlay_required", "Underlay/underlayment required yes/no"),
            ],
            "packaging": [
                ("planks_per_box", "Number of planks per pack"),
                ("m2_per_box", "Coverage per pack in m²"),
                ("sqft_per_box", "Coverage per pack in sqft"),
                ("weight_per_box_kg", "Pack weight in kg"),
                ("boxes_per_pallet", "Packs per pallet"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
                ("vision_variants", "Array of variant objects [{sku, name, color, format}]"),
            ],
            "compliance": [
                ("certifications", "Array of certs (FSC, PEFC, CE, Der Blaue Engel, etc.)"),
                ("fsc_certified", "FSC certification status and chain-of-custody number"),
                ("pefc_certified", "PEFC certification status"),
                ("origin_country", "Country of origin / wood source"),
                ("sustainability_rating", "Sustainability rating"),
                ("eco_friendly", "Eco-friendly indicators"),
            ],
            "care": [
                ("care_instructions", "Cleaning and care instructions"),
                ("maintenance", "Re-oiling/re-finishing schedule"),
                ("compatible_cleaners", "Recommended cleaning products"),
            ],
        },
        "extraction_hints": [
            "Wood flooring specs often include a cross-section diagram showing layers — extract wear layer and total thickness from it.",
            "Look for certification logos (FSC tree, PEFC, Blue Angel) in margins or footer areas.",
            "Click system names are sometimes trademarked (Uniclic, Valinge 5G, etc.) — extract the exact name.",
            "Janka hardness may be listed in a comparison table across species.",
            "AC ratings are specific to laminate/vinyl — real wood uses wear class (21-34) instead.",
        ],
        "skip_fields": [
            "pei_rating", "grout_mapei", "grout_kerakoll", "grout_isomat", "grout_technica",
            "grout_color_codes", "grout_suppliers", "grout_details",
            "joint_width_mm", "rectified", "body_type",
            "wattage", "lumens", "color_temperature_k", "cri", "beam_angle",
            "btu", "kw_output", "thermostat_type", "fuel_type",
            "flow_rate", "flush_volume", "trap_type",
            "coverage_per_litre", "dry_time", "voc_level", "sheen",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # DECOR
    # ═══════════════════════════════════════════════════════════════════════
    "decor": {
        "display_name": "Decor",
        "controlled_vocab": [
            "rug", "curtain", "cushion", "vase", "mirror",
            "wall_art", "sculpture", "candle_holder", "planter",
        ],
        "priority_fields": {
            "material_properties": [
                ("primary_material", "Primary material (ceramic, resin, metal, glass, rattan, cotton, etc.)"),
                ("secondary_material", "Secondary/accent material if applicable"),
                ("finish", "Surface finish (matte, glazed, lacquered, natural, antiqued)"),
                ("handmade", "Handmade/hand-crafted yes/no"),
            ],
            "dimensions": [
                ("width_cm", "Width in cm"),
                ("height_cm", "Height in cm"),
                ("depth_cm", "Depth in cm"),
                ("diameter_cm", "Diameter in cm (for round items)"),
                ("weight_kg", "Weight in kg"),
                ("available_sizes", "Array of available size options"),
            ],
            "appearance": [
                ("colors", "Array of available colors"),
                ("primary_color_hex", "Primary hex color code"),
                ("style", "Design style (modern, bohemian, minimalist, industrial, classic, etc.)"),
                ("pattern", "Pattern description if any"),
                ("texture", "Texture description"),
            ],
            "application": [
                ("indoor_outdoor", "Indoor, Outdoor, or Both"),
                ("wall_mountable", "Wall mountable yes/no"),
                ("freestanding", "Freestanding yes/no"),
                ("suitable_rooms", "Suitable rooms (living room, bedroom, bathroom, etc.)"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
            ],
            "care": [
                ("care_instructions", "Cleaning and care instructions"),
                ("fragile", "Fragile handling required yes/no"),
            ],
        },
        "extraction_hints": [
            "Decor items often list dimensions as W x H x D — extract all three.",
            "Look for material composition details in product descriptions.",
            "Style/collection names are important for decor — they drive search.",
            "Handmade items may note country of origin or artisan details.",
        ],
        "skip_fields": [
            "pei_rating", "slip_resistance", "water_absorption", "frost_resistance",
            "grout_mapei", "grout_kerakoll", "grout_isomat", "grout_technica",
            "grout_color_codes", "grout_suppliers", "grout_details",
            "pieces_per_box", "boxes_per_pallet", "m2_per_box", "sqft_per_box",
            "wattage", "lumens", "color_temperature_k", "cri", "beam_angle",
            "btu", "kw_output", "thermostat_type", "fuel_type",
            "flow_rate", "flush_volume", "trap_type",
            "janka_hardness", "ac_rating", "click_system",
            "coverage_per_litre", "dry_time", "voc_level",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # FURNITURE
    # ═══════════════════════════════════════════════════════════════════════
    "furniture": {
        "display_name": "Furniture",
        "controlled_vocab": [
            "sofa", "armchair", "dining_chair", "accent_chair",
            "dining_table", "coffee_table", "side_table",
            "cabinet", "shelving", "sideboard", "bed", "desk",
            "outdoor_furniture",
        ],
        "priority_fields": {
            "material_properties": [
                ("frame_material", "Frame material (solid wood, metal, plywood, MDF)"),
                ("upholstery_material", "Upholstery material (fabric, leather, velvet, linen)"),
                ("fill_material", "Fill/padding material (foam, down, polyester fibre)"),
                ("leg_material", "Leg material (wood, metal, chrome)"),
                ("top_material", "Table/surface top material (marble, glass, veneer, solid wood)"),
                ("finish", "Finish (lacquered, oiled, powder-coated, chrome, brushed)"),
                ("foam_density_kg_m3", "Foam density in kg/m³"),
            ],
            "dimensions": [
                ("width_cm", "Overall width in cm"),
                ("depth_cm", "Overall depth in cm"),
                ("height_cm", "Overall height in cm"),
                ("seat_height_cm", "Seat height in cm"),
                ("seat_depth_cm", "Seat depth in cm"),
                ("arm_height_cm", "Arm height in cm"),
                ("table_height_cm", "Table height in cm"),
                ("weight_kg", "Product weight in kg"),
                ("weight_capacity_kg", "Max weight capacity in kg"),
                ("available_sizes", "Array of available size options"),
            ],
            "appearance": [
                ("colors", "Array of available colors/upholstery options"),
                ("primary_color_hex", "Primary hex color code"),
                ("fabric_options", "Array of available fabric/leather choices"),
                ("style", "Design style (modern, mid-century, Scandinavian, industrial, etc.)"),
            ],
            "features": [
                ("assembly_required", "Assembly required yes/no"),
                ("stackable", "Stackable yes/no"),
                ("foldable", "Foldable yes/no"),
                ("modular", "Modular / configurable yes/no"),
                ("reclining", "Reclining mechanism yes/no"),
                ("storage", "Built-in storage yes/no"),
                ("adjustable_height", "Height adjustable yes/no"),
                ("indoor_outdoor", "Indoor, Outdoor, or Both"),
                ("number_of_seats", "Number of seats (for sofas/benches)"),
            ],
            "performance": [
                ("martindale_cycles", "Fabric abrasion resistance Martindale cycles"),
                ("pilling_grade", "Pilling grade (1-5)"),
                ("light_fastness", "Light fastness rating (1-8)"),
                ("fire_retardancy", "Fire retardancy class (BS 7176, CRIB 5, CAL 117, etc.)"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
                ("lead_time_weeks", "Production/delivery lead time in weeks"),
            ],
            "compliance": [
                ("certifications", "Array of certifications (FSC, BIFMA, GREENGUARD, etc.)"),
                ("fire_rating", "Fire safety classification"),
            ],
            "care": [
                ("care_instructions", "Cleaning and care instructions"),
                ("removable_covers", "Removable/washable covers yes/no"),
            ],
        },
        "extraction_hints": [
            "Furniture specs often list dimensions as W x D x H — extract each separately.",
            "Seat height is critical for dining chairs and bar stools — look for it specifically.",
            "Upholstery options may be listed in a separate fabric/leather catalog or swatch card.",
            "Martindale abrasion and fire retardancy ratings are key for contract/commercial furniture.",
            "Look for COM (Customer's Own Material) options in commercial furniture catalogs.",
        ],
        "skip_fields": [
            "pei_rating", "slip_resistance", "water_absorption", "frost_resistance",
            "grout_mapei", "grout_kerakoll", "grout_isomat", "grout_technica",
            "grout_color_codes", "grout_suppliers", "grout_details",
            "pieces_per_box", "boxes_per_pallet", "m2_per_box", "sqft_per_box",
            "m2_per_pallet", "sqft_per_pallet", "pallet_dimensions_cm",
            "wattage", "lumens", "color_temperature_k", "cri", "beam_angle",
            "btu", "kw_output", "thermostat_type", "fuel_type",
            "flow_rate", "flush_volume", "trap_type",
            "janka_hardness", "ac_rating", "click_system",
            "coverage_per_litre", "dry_time", "voc_level",
            "body_type", "rectified", "joint_width_mm",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # GENERAL MATERIALS
    # ═══════════════════════════════════════════════════════════════════════
    "general_materials": {
        "display_name": "General Materials",
        "controlled_vocab": [
            "stone_slab", "metal_panel", "glass_panel",
            "countertop", "kitchen_worktop", "cladding",
            "concrete", "terrazzo", "quartz", "composite",
        ],
        "priority_fields": {
            "material_properties": [
                ("material_type", "Primary material type (natural stone, quartz, Dekton, Corian, concrete, etc.)"),
                ("composition", "Material composition / formulation"),
                ("finish", "Surface finish (polished, honed, leathered, flamed, bush-hammered)"),
                ("thickness_mm", "Slab/panel thickness in mm"),
                ("edge_profiles", "Available edge profiles (bullnose, ogee, mitre, etc.)"),
                ("density_kg_m3", "Material density in kg/m³"),
            ],
            "dimensions": [
                ("slab_size", "Slab/panel nominal size (e.g. 320x160 cm)"),
                ("available_sizes", "Array of available sizes/formats"),
                ("weight_per_m2_kg", "Weight per m² in kg"),
            ],
            "appearance": [
                ("colors", "Array of available colors/finishes"),
                ("primary_color_hex", "Primary hex color code"),
                ("pattern", "Pattern (veined, speckled, solid, bookmatched)"),
                ("texture", "Surface texture description"),
                ("translucency", "Translucent/backlit capable yes/no"),
            ],
            "performance": [
                ("compressive_strength", "Compressive strength in MPa"),
                ("flexural_strength", "Flexural/bending strength in MPa"),
                ("water_absorption_pct", "Water absorption percentage"),
                ("scratch_resistance", "Scratch resistance (Mohs or specific test)"),
                ("heat_resistance_c", "Max heat resistance in celsius"),
                ("stain_resistance", "Stain resistance rating"),
                ("uv_resistance", "UV stability / fade resistance"),
                ("fire_rating", "Fire classification"),
                ("frost_resistance", "Frost resistant yes/no"),
                ("acoustic_rating_db", "Acoustic insulation in dB"),
                ("thermal_conductivity", "Thermal conductivity W/mK"),
            ],
            "application": [
                ("recommended_use", "Recommended applications (countertop, wall cladding, flooring, facade)"),
                ("indoor_outdoor", "Indoor, Outdoor, or Both"),
                ("installation_method", "Installation method"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
            ],
            "compliance": [
                ("certifications", "Array of certifications (CE, NSF, GREENGUARD, etc.)"),
                ("eco_friendly", "Eco-friendly indicators"),
                ("recycled_content_pct", "Recycled content percentage"),
            ],
            "care": [
                ("care_instructions", "Cleaning and care instructions"),
                ("sealing_required", "Sealing required yes/no and frequency"),
            ],
        },
        "extraction_hints": [
            "Natural stone and engineered surfaces often have slab sizes and thickness as key specs.",
            "Performance data (flexural strength, water absorption) is critical for architects specifying materials.",
            "Look for technical data sheets linked or embedded in catalog pages.",
            "Translucency/backlit capability is a premium feature worth capturing.",
        ],
        "skip_fields": [
            "grout_mapei", "grout_kerakoll", "grout_isomat", "grout_technica",
            "grout_color_codes", "grout_suppliers", "grout_details",
            "wattage", "lumens", "color_temperature_k", "cri", "beam_angle",
            "btu", "kw_output", "thermostat_type", "fuel_type",
            "flow_rate", "flush_volume", "trap_type",
            "janka_hardness", "ac_rating", "click_system",
            "coverage_per_litre", "dry_time", "voc_level", "sheen",
            "species", "grain_direction",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PAINT / WALL DECOR
    # ═══════════════════════════════════════════════════════════════════════
    "paint_wall_decor": {
        "display_name": "Paint / Wall Decors",
        "controlled_vocab": [
            "wall_paint", "wallpaper", "wall_coating",
            "decorative_plaster", "wall_panel",
        ],
        "priority_fields": {
            "material_properties": [
                ("product_type", "Type (interior paint, exterior paint, wallpaper, decorative plaster, wall panel)"),
                ("base_type", "Base type (water-based, solvent-based, lime-based, acrylic)"),
                ("finish_sheen", "Finish/sheen level (flat/matte, eggshell, satin, semi-gloss, gloss)"),
                ("composition", "Composition / ingredients"),
                ("texture", "Texture (smooth, textured, sand finish, stucco)"),
                ("substrate", "Wallpaper substrate (non-woven, vinyl, paper, fabric-backed)"),
                ("pattern_repeat_cm", "Wallpaper pattern repeat in cm"),
            ],
            "dimensions": [
                ("roll_width_cm", "Wallpaper roll width in cm (typically 53 or 70)"),
                ("roll_length_m", "Wallpaper roll length in m (typically 10 or 10.05)"),
                ("coverage_per_litre_m2", "Paint coverage per litre in m²"),
                ("coverage_per_roll_m2", "Wallpaper coverage per roll in m²"),
                ("can_sizes", "Available can/container sizes (e.g. 1L, 2.5L, 5L, 10L)"),
            ],
            "appearance": [
                ("colors", "Array of available colors"),
                ("color_code_ral", "RAL color code"),
                ("color_code_ncs", "NCS color code"),
                ("color_code_pantone", "Pantone reference"),
                ("primary_color_hex", "Primary hex color code"),
                ("tintable", "Tintable / color-matchable yes/no"),
                ("pattern", "Pattern design name/description"),
                ("design_style", "Design style (contemporary, classic, botanical, geometric)"),
            ],
            "performance": [
                ("washability_class", "Washability class (1-5 per EN 13300)"),
                ("wet_scrub_resistance", "Wet scrub resistance cycles"),
                ("voc_level_g_l", "VOC content in g/L"),
                ("voc_class", "VOC emission class (A+, A, B, C)"),
                ("dry_time_touch_hours", "Touch-dry time in hours"),
                ("dry_time_recoat_hours", "Recoat time in hours"),
                ("coats_recommended", "Number of coats recommended"),
                ("opacity_class", "Hiding power / opacity class (1-4 per EN 13300)"),
                ("fire_rating", "Fire classification (A1, A2, B1, B2, etc.)"),
                ("mould_resistance", "Mould/mildew resistance rating"),
                ("moisture_resistance", "Moisture resistance / wet room suitable"),
            ],
            "application": [
                ("recommended_use", "Interior, Exterior, or Both"),
                ("suitable_surfaces", "Suitable surfaces (plaster, drywall, concrete, wood, etc.)"),
                ("application_method", "Application method (brush, roller, spray, paste)"),
                ("primer_required", "Primer required yes/no"),
                ("suitable_rooms", "Suitable rooms (bathroom, kitchen, bedroom, etc.)"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
            ],
            "compliance": [
                ("certifications", "Array of certifications (EU Ecolabel, Blue Angel, Cradle to Cradle, etc.)"),
                ("eco_friendly", "Eco-friendly indicators"),
                ("low_odour", "Low odour yes/no"),
            ],
            "care": [
                ("care_instructions", "Cleaning instructions for painted surfaces"),
                ("touch_up", "Touch-up guidance"),
            ],
        },
        "extraction_hints": [
            "Paint specs often list coverage rate (m²/L), dry time, VOC level — these are the most queried specs.",
            "Color codes (RAL, NCS) are critical for professional specification — look for them in technical data sheets.",
            "Wallpaper specs must include roll dimensions and pattern repeat for quantity calculation.",
            "Washability class (EN 13300) and VOC class are key for specifiers — often in a technical table.",
            "Look for EN 13300 classification table (opacity class, wet scrub resistance).",
        ],
        "skip_fields": [
            "pei_rating", "slip_resistance", "water_absorption", "frost_resistance",
            "body_type", "rectified", "joint_width_mm",
            "grout_mapei", "grout_kerakoll", "grout_isomat", "grout_technica",
            "grout_color_codes", "grout_suppliers", "grout_details",
            "pieces_per_box", "boxes_per_pallet",
            "wattage", "lumens", "color_temperature_k", "cri", "beam_angle",
            "btu", "kw_output", "thermostat_type", "fuel_type",
            "flow_rate", "flush_volume", "trap_type",
            "janka_hardness", "ac_rating", "click_system", "species",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # HEATING
    # ═══════════════════════════════════════════════════════════════════════
    "heating": {
        "display_name": "Heating",
        "controlled_vocab": [
            "radiator", "towel_rail", "underfloor_heating",
            "heat_pump", "boiler", "fireplace", "convector",
        ],
        "priority_fields": {
            "material_properties": [
                ("product_type", "Type (radiator, towel rail, underfloor heating mat, convector, fireplace)"),
                ("body_material", "Material (steel, cast iron, aluminium, stainless steel)"),
                ("finish", "Finish (powder-coated, chrome, brushed steel, RAL color)"),
                ("panel_type", "Panel type for radiators (Type 10, 11, 21, 22, 33)"),
            ],
            "dimensions": [
                ("width_mm", "Width in mm"),
                ("height_mm", "Height in mm"),
                ("depth_mm", "Depth in mm"),
                ("weight_kg", "Weight in kg"),
                ("available_sizes", "Array of available size configurations"),
                ("sections", "Number of sections (for sectional radiators)"),
            ],
            "thermal_performance": [
                ("heat_output_watts", "Heat output in Watts at Delta T 50"),
                ("heat_output_btu", "Heat output in BTU/h"),
                ("delta_t", "Delta T rating (usually 50K or 30K)"),
                ("energy_class", "Energy efficiency class (A-G)"),
                ("max_operating_pressure_bar", "Max operating pressure in bar"),
                ("max_operating_temp_c", "Max operating temperature in celsius"),
                ("water_content_litres", "Water content in litres"),
                ("flow_connection_size", "Flow/return connection size (e.g. 1/2 inch, 15mm)"),
                ("kw_output", "Output in kW (for boilers/heat pumps)"),
            ],
            "features": [
                ("thermostat_type", "Thermostat type (TRV, digital, smart, none)"),
                ("valve_type", "Valve connection type (bottom, side, integrated)"),
                ("reversible", "Reversible left-right yes/no"),
                ("dual_fuel", "Dual fuel (central heating + electric) yes/no"),
                ("electric_element_watts", "Electric element wattage if dual fuel"),
                ("timer", "Built-in timer yes/no"),
                ("smart_compatible", "Smart home compatible yes/no"),
                ("ip_rating", "IP rating for bathroom use"),
            ],
            "installation": [
                ("mounting_type", "Mounting type (wall-mounted, floor-standing, freestanding)"),
                ("connection_type", "Connection type (bottom centre, bottom corner, side)"),
                ("brackets_included", "Mounting brackets included yes/no"),
                ("installation_method", "Installation requirements"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
                ("warranty_years", "Warranty period in years"),
            ],
            "compliance": [
                ("certifications", "Array of certifications (CE, EN 442, TUV, UKCA, etc.)"),
                ("en_442_compliant", "EN 442 test standard compliance"),
                ("pressure_tested_bar", "Factory pressure test rating in bar"),
                ("ip_rating", "IP rating (for bathroom radiators)"),
            ],
        },
        "extraction_hints": [
            "Heat output is the most critical spec — look for Watts at Delta T 50 (standard test condition).",
            "Radiator catalogs often present output tables: rows = heights, columns = widths/sections.",
            "Panel type (11, 21, 22, 33) describes the number of panels and convector fins.",
            "Towel rails may list both central heating and electric output separately.",
            "Look for EN 442 certification and factory test pressure.",
            "Connection sizes and valve positions are critical for installers.",
        ],
        "skip_fields": [
            "pei_rating", "slip_resistance", "water_absorption", "frost_resistance",
            "body_type", "rectified", "joint_width_mm",
            "grout_mapei", "grout_kerakoll", "grout_isomat", "grout_technica",
            "grout_color_codes", "grout_suppliers", "grout_details",
            "pieces_per_box", "boxes_per_pallet", "m2_per_box", "sqft_per_box",
            "lumens", "color_temperature_k", "cri", "beam_angle",
            "flow_rate", "flush_volume", "trap_type",
            "janka_hardness", "ac_rating", "click_system", "species",
            "coverage_per_litre", "dry_time", "voc_level", "sheen",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SANITARY
    # ═══════════════════════════════════════════════════════════════════════
    "sanitary": {
        "display_name": "Sanitary",
        "controlled_vocab": [
            "toilet", "basin", "bathtub", "shower_tray",
            "bidet", "urinal", "vanity_unit", "shower_enclosure",
            "tap", "faucet", "mixer", "shower_head",
        ],
        "priority_fields": {
            "material_properties": [
                ("product_type", "Type (WC, basin, bathtub, shower tray, tap/faucet, vanity unit)"),
                ("body_material", "Material (vitreous china, fine fireclay, acrylic, solid surface, cast mineral, stainless steel, brass)"),
                ("finish", "Finish (white, matt white, glossy, chrome, brushed nickel, black matt)"),
                ("glaze_type", "Glaze technology (CeramicPlus, AntiBac, HygieneGlaze, AquaBlade, etc.)"),
            ],
            "dimensions": [
                ("width_mm", "Width in mm"),
                ("depth_mm", "Depth/projection in mm"),
                ("height_mm", "Height in mm"),
                ("weight_kg", "Weight in kg"),
                ("bowl_depth_mm", "Bowl/basin depth in mm"),
                ("available_sizes", "Array of available sizes"),
            ],
            "water_performance": [
                ("flow_rate_l_min", "Flow rate in litres per minute"),
                ("flush_volume_l", "Flush volume in litres (full/half flush e.g. '4.5/3')"),
                ("water_efficiency_label", "Water efficiency label (WELL A-F, WaterSense)"),
                ("water_saving_pct", "Water saving percentage vs standard"),
                ("noise_class_db", "Noise class in dB (for flush systems)"),
            ],
            "installation": [
                ("mounting_type", "Mounting (wall-hung, back-to-wall, floor-standing, countertop, undermount)"),
                ("trap_type", "Trap type (P-trap, S-trap, hidden/concealed)"),
                ("connection_size", "Water connection size (e.g. 1/2 inch, 3/8 inch)"),
                ("waste_size_mm", "Waste outlet size in mm"),
                ("concealed_cistern", "Requires concealed cistern / frame yes/no"),
                ("frame_compatibility", "Compatible installation frames (Geberit Duofix, Grohe Rapid SL, etc.)"),
                ("tap_hole_config", "Tap hole configuration (1-hole, 3-hole, no tap hole)"),
                ("cartridge_type", "Cartridge type and size for taps (ceramic 35mm, 40mm)"),
            ],
            "features": [
                ("rimless", "Rimless design yes/no"),
                ("soft_close_seat", "Soft close seat included yes/no"),
                ("quick_release_seat", "Quick release seat yes/no"),
                ("overflow", "Overflow included yes/no"),
                ("thermostatic", "Thermostatic control yes/no (for taps/showers)"),
                ("anti_scald", "Anti-scald safety yes/no"),
                ("anti_fingerprint", "Anti-fingerprint finish yes/no"),
                ("led_indicator", "LED temperature indicator yes/no"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
                ("warranty_years", "Warranty period in years"),
            ],
            "compliance": [
                ("certifications", "Array of certifications (CE, WRAS, WaterMark, ACS, NF, etc.)"),
                ("ip_rating", "IP rating for electrical components"),
                ("acoustic_class", "Acoustic class for in-wall systems"),
                ("accessibility", "DDA/accessibility compliant yes/no"),
            ],
            "care": [
                ("care_instructions", "Cleaning and care instructions"),
                ("compatible_cleaners", "Recommended/incompatible cleaning products"),
            ],
        },
        "extraction_hints": [
            "Sanitary catalogs often list mounting dimensions with technical drawings — extract key dimensions.",
            "Flow rate and flush volume are critical for water efficiency specifications.",
            "Look for WRAS (UK), ACS (France), NF (France) approvals for taps/mixers.",
            "Concealed cistern compatibility (Geberit, Grohe) is important for wall-hung WCs.",
            "Glaze technology names (AquaBlade, TurboFlush) are brand-specific features worth capturing.",
            "Tap/mixer specs need cartridge size, flow rate, connection size, and thermostatic info.",
        ],
        "skip_fields": [
            "pei_rating", "slip_resistance", "water_absorption", "frost_resistance",
            "body_type", "rectified", "joint_width_mm",
            "grout_mapei", "grout_kerakoll", "grout_isomat", "grout_technica",
            "grout_color_codes", "grout_suppliers", "grout_details",
            "pieces_per_box", "boxes_per_pallet", "m2_per_box", "sqft_per_box",
            "wattage", "lumens", "color_temperature_k", "cri", "beam_angle",
            "btu", "kw_output",
            "janka_hardness", "ac_rating", "click_system", "species",
            "coverage_per_litre", "dry_time", "voc_level", "sheen",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # KITCHEN
    # ═══════════════════════════════════════════════════════════════════════
    "kitchen": {
        "display_name": "Kitchen",
        "controlled_vocab": [
            "kitchen_cabinet", "kitchen_worktop", "kitchen_sink",
            "kitchen_tap", "kitchen_hood", "kitchen_appliance",
            "kitchen_handle", "kitchen_organiser",
        ],
        "priority_fields": {
            "material_properties": [
                ("product_type", "Type (cabinet, worktop, sink, tap, hood, handle, organiser)"),
                ("body_material", "Primary material (melamine, solid wood, MDF, quartz, granite, stainless steel, composite)"),
                ("door_material", "Door/front material and finish"),
                ("finish", "Finish (matt, gloss, wood-grain, lacquered, Fenix, laminate)"),
                ("edge_treatment", "Edge treatment (ABS, laser, PUR, solid wood lipping)"),
                ("worktop_thickness_mm", "Worktop thickness in mm"),
            ],
            "dimensions": [
                ("width_mm", "Width in mm"),
                ("height_mm", "Height in mm"),
                ("depth_mm", "Depth in mm"),
                ("weight_kg", "Weight in kg"),
                ("available_sizes", "Array of available sizes/configurations"),
                ("bowl_dimensions", "Sink bowl dimensions if applicable"),
            ],
            "performance": [
                ("heat_resistance_c", "Heat resistance max temperature in celsius"),
                ("scratch_resistance", "Scratch resistance rating"),
                ("stain_resistance", "Stain resistance rating"),
                ("impact_resistance", "Impact resistance"),
                ("noise_level_db", "Noise level in dB (for hoods/appliances)"),
                ("extraction_rate_m3_h", "Extraction rate in m³/h (for hoods)"),
                ("energy_class", "Energy efficiency class"),
                ("water_absorption_pct", "Water absorption (for worktops)"),
            ],
            "features": [
                ("soft_close", "Soft close hinges/drawers yes/no"),
                ("drawer_system", "Drawer runner system (Blum, Hettich, etc.)"),
                ("hinge_type", "Hinge type (Blum Clip-Top, Hettich, concealed)"),
                ("push_to_open", "Push-to-open (handleless) yes/no"),
                ("integrated_lighting", "Integrated LED lighting yes/no"),
                ("adjustable_shelves", "Adjustable shelves yes/no"),
                ("modular", "Modular system yes/no"),
                ("assembly_type", "Assembly type (flatpack, rigid, semi-assembled)"),
            ],
            "installation": [
                ("installation_type", "Installation type (base unit, wall unit, tall unit, island)"),
                ("mounting_method", "Mounting method (wall-mounted, floor-standing)"),
                ("connection_type", "Connection requirements for sinks/appliances"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
                ("warranty_years", "Warranty period in years"),
            ],
            "compliance": [
                ("certifications", "Array of certifications (CE, FIRA Gold, FSC, PEFC, etc.)"),
                ("formaldehyde_class", "Formaldehyde emission class (E0, E1, CARB2)"),
                ("eco_friendly", "Eco-friendly indicators"),
            ],
            "care": [
                ("care_instructions", "Cleaning and care instructions"),
            ],
        },
        "extraction_hints": [
            "Kitchen catalogs often present cabinet ranges with modular unit codes — extract the coding system.",
            "Worktop specs need thickness, heat resistance, and stain resistance prominently.",
            "Hinge and drawer runner brands (Blum, Hettich, Grass) are key quality indicators.",
            "Sink dimensions should include bowl dimensions separately from overall dimensions.",
            "Hood extraction rates and noise levels are compared by consumers — always capture both.",
        ],
        "skip_fields": [
            "pei_rating", "slip_resistance", "water_absorption", "frost_resistance",
            "body_type", "rectified", "joint_width_mm",
            "grout_mapei", "grout_kerakoll", "grout_isomat", "grout_technica",
            "grout_color_codes", "grout_suppliers", "grout_details",
            "pieces_per_box", "boxes_per_pallet", "m2_per_box", "sqft_per_box",
            "btu", "kw_output", "thermostat_type", "fuel_type",
            "lumens", "color_temperature_k", "cri", "beam_angle",
            "janka_hardness", "ac_rating", "click_system", "species",
            "coverage_per_litre", "dry_time", "voc_level",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # LIGHTING
    # ═══════════════════════════════════════════════════════════════════════
    "lighting": {
        "display_name": "Lighting",
        "controlled_vocab": [
            "lighting", "pendant_light", "ceiling_light", "wall_light",
            "floor_lamp", "table_lamp", "spotlight", "track_light",
            "recessed_light", "outdoor_light", "chandelier",
        ],
        "priority_fields": {
            "material_properties": [
                ("product_type", "Type (pendant, ceiling, wall, floor lamp, table lamp, spotlight, track, recessed, chandelier)"),
                ("body_material", "Material (metal, glass, fabric, wood, concrete, rattan)"),
                ("shade_material", "Shade material if separate (fabric, glass, metal, paper)"),
                ("finish", "Finish (chrome, brushed brass, matt black, white, copper, antique bronze)"),
            ],
            "dimensions": [
                ("diameter_mm", "Diameter in mm"),
                ("height_mm", "Height/length in mm"),
                ("width_mm", "Width in mm"),
                ("depth_mm", "Projection/depth in mm (for wall lights)"),
                ("cable_length_mm", "Cable/chain length in mm"),
                ("adjustable_drop", "Adjustable drop range min-max in mm"),
                ("weight_kg", "Weight in kg"),
                ("cutout_diameter_mm", "Ceiling cutout diameter in mm (recessed lights)"),
            ],
            "electrical_specs": [
                ("wattage_w", "Wattage in W (LED equivalent or max for replaceable)"),
                ("lumens_lm", "Luminous flux in lumens"),
                ("color_temperature_k", "Color temperature in Kelvin (e.g. 2700K, 3000K, 4000K)"),
                ("color_temperature_range_k", "Tunable color temp range (e.g. 2700-6500K)"),
                ("cri_ra", "Color Rendering Index Ra value (80+, 90+)"),
                ("beam_angle_deg", "Beam angle in degrees"),
                ("dimmable", "Dimmable yes/no"),
                ("dimming_type", "Dimming type (trailing edge, leading edge, DALI, 1-10V, Casambi)"),
                ("led_integrated", "LED integrated (non-replaceable) yes/no"),
                ("lamp_type", "Lamp type/socket (E27, E14, GU10, G9, integrated LED)"),
                ("lamp_included", "Lamp/bulb included yes/no"),
                ("max_wattage_w", "Max wattage for replaceable lamp socket"),
                ("voltage_v", "Operating voltage (220-240V, 12V, 24V)"),
                ("driver_included", "LED driver included yes/no"),
                ("driver_type", "Driver type (constant current, constant voltage)"),
                ("energy_class", "Energy efficiency class (A-G, new EU label)"),
                ("lifespan_hours", "LED lifespan in hours"),
                ("efficacy_lm_w", "Luminous efficacy lm/W"),
            ],
            "features": [
                ("ip_rating", "IP rating (IP20, IP44, IP54, IP65, IP67)"),
                ("ik_rating", "IK impact rating"),
                ("smart_compatible", "Smart home compatible (Philips Hue, Casambi, etc.)"),
                ("sensor", "Built-in sensor type (PIR, daylight, motion)"),
                ("emergency", "Emergency lighting function yes/no"),
                ("adjustable_tilt", "Adjustable tilt/rotation yes/no"),
                ("number_of_lights", "Number of light points"),
                ("indoor_outdoor", "Indoor, Outdoor, or Both"),
            ],
            "installation": [
                ("mounting_type", "Mounting type (pendant, surface, recessed, track, clip, wall bracket)"),
                ("ceiling_type", "Suitable ceiling type (plasterboard, concrete, suspended)"),
                ("junction_box", "Requires junction box / back box yes/no"),
                ("installation_zone", "Bathroom zone rating (Zone 0, 1, 2, outside zones)"),
            ],
            "commercial": [
                ("sku_codes", "Object mapping variant names to SKU codes"),
                ("product_codes", "Array of product/article codes"),
                ("warranty_years", "Warranty period in years"),
            ],
            "compliance": [
                ("certifications", "Array of certifications (CE, UKCA, ENEC, UL, ETL, etc.)"),
                ("fire_rating", "Fire rating for recessed lights"),
                ("class_protection", "Electrical protection class (I, II, III)"),
            ],
        },
        "extraction_hints": [
            "Lumens, color temperature, and CRI are the three most queried lighting specs — prioritize these.",
            "IP rating is critical: IP20=indoor only, IP44=bathroom zone 2, IP65=outdoor/wet.",
            "Look for photometric data tables or IES file references in technical specs.",
            "Dimmable information must include dimming protocol (trailing edge, DALI, etc.) not just yes/no.",
            "Recessed lights need cutout diameter and ceiling void depth — look in installation diagrams.",
            "Energy label class changed in 2021 (new A-G scale) — extract the correct generation.",
            "Lamp socket type (E27, GU10) determines replaceability — integrated LED means non-replaceable.",
        ],
        "skip_fields": [
            "pei_rating", "slip_resistance", "water_absorption", "frost_resistance",
            "body_type", "rectified", "joint_width_mm",
            "grout_mapei", "grout_kerakoll", "grout_isomat", "grout_technica",
            "grout_color_codes", "grout_suppliers", "grout_details",
            "pieces_per_box", "boxes_per_pallet", "m2_per_box", "sqft_per_box",
            "btu", "kw_output", "thermostat_type", "fuel_type",
            "flow_rate", "flush_volume", "trap_type",
            "janka_hardness", "ac_rating", "click_system", "species",
            "coverage_per_litre", "dry_time", "voc_level", "sheen",
        ],
    },
}


# ─── Helper functions ────────────────────────────────────────────────────────

def get_category_config(category_key: str) -> Dict[str, Any]:
    """Get the field registry for a category. Falls back to general_materials."""
    return CATEGORY_FIELD_REGISTRY.get(
        category_key,
        CATEGORY_FIELD_REGISTRY["general_materials"],
    )


def get_all_category_keys() -> List[str]:
    """Return all registered category keys."""
    return list(CATEGORY_FIELD_REGISTRY.keys())


def get_priority_fields_for_prompt(category_key: str) -> str:
    """
    Build the priority fields section of the extraction prompt for a given category.
    Returns formatted text ready to be injected into the AI prompt.
    """
    config = get_category_config(category_key)
    lines: List[str] = []
    lines.append(f"PRIORITY FIELDS for {config['display_name'].upper()} products:")
    lines.append("(Extract these if present — they are the most important for this category)")
    lines.append("")

    for section_name, fields in config["priority_fields"].items():
        section_label = section_name.replace("_", " ").title()
        lines.append(f"**{section_label}:**")
        for field_key, field_desc in fields:
            lines.append(f"- {field_key}: {field_desc}")
        lines.append("")

    return "\n".join(lines)


def get_extraction_hints_for_prompt(category_key: str) -> str:
    """
    Build the extraction hints section for the AI prompt.
    Returns formatted text with category-specific tips.
    """
    config = get_category_config(category_key)
    hints = config.get("extraction_hints", [])
    if not hints:
        return ""

    lines = [f"CATEGORY-SPECIFIC EXTRACTION TIPS for {config['display_name']}:"]
    for hint in hints:
        lines.append(f"- {hint}")
    return "\n".join(lines)


def get_skip_fields(category_key: str) -> List[str]:
    """Return the list of field keys to exclude from the prompt for this category."""
    config = get_category_config(category_key)
    return config.get("skip_fields", [])


def get_controlled_vocab(category_key: str) -> List[str]:
    """Return the fine-grained material_category values for this upload category."""
    config = get_category_config(category_key)
    return config.get("controlled_vocab", [])
