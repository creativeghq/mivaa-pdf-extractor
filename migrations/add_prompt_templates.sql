-- Prompt Templates Table
-- Stores customizable AI prompts for different extraction stages and industries

CREATE TABLE IF NOT EXISTS prompt_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL,
    
    -- Template identification
    name VARCHAR(255) NOT NULL,
    description TEXT,
    industry VARCHAR(100),  -- e.g., 'construction', 'interior_design', 'manufacturing', 'general'
    
    -- Prompt configuration
    stage VARCHAR(50) NOT NULL,  -- 'discovery', 'metadata_extraction', 'classification', 'chunking'
    category VARCHAR(100),  -- 'products', 'certificates', 'logos', 'specifications', etc.
    
    -- Prompt content
    prompt_template TEXT NOT NULL,
    system_prompt TEXT,
    
    -- AI model configuration
    model_preference VARCHAR(50),  -- 'claude', 'gpt', 'auto'
    temperature DECIMAL(3,2) DEFAULT 0.1,
    max_tokens INTEGER DEFAULT 4096,
    
    -- Template metadata
    is_default BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    version INTEGER DEFAULT 1,
    
    -- Audit fields
    created_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_template_per_stage_category UNIQUE (workspace_id, stage, category, industry, name)
);

-- Prompt Template History
CREATE TABLE IF NOT EXISTS prompt_template_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id UUID NOT NULL REFERENCES prompt_templates(id) ON DELETE CASCADE,
    
    -- Change tracking
    old_prompt TEXT,
    new_prompt TEXT,
    old_system_prompt TEXT,
    new_system_prompt TEXT,
    
    -- Change metadata
    changed_by UUID,
    change_reason TEXT,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_prompt_templates_workspace ON prompt_templates(workspace_id);
CREATE INDEX idx_prompt_templates_stage ON prompt_templates(stage);
CREATE INDEX idx_prompt_templates_category ON prompt_templates(category);
CREATE INDEX idx_prompt_templates_industry ON prompt_templates(industry);
CREATE INDEX idx_prompt_templates_active ON prompt_templates(is_active);
CREATE INDEX idx_prompt_template_history_template ON prompt_template_history(template_id);

-- Insert default templates
INSERT INTO prompt_templates (workspace_id, name, description, industry, stage, category, prompt_template, system_prompt, is_default, is_active)
VALUES 
-- General Material Metadata Extraction
('00000000-0000-0000-0000-000000000000', 'General Material Metadata', 'Default metadata extraction for all material types', 'general', 'metadata_extraction', 'products', 
'You are analyzing a product specification PDF to extract ALL metadata attributes.

CRITICAL FIELDS (MUST extract these):
1. material_category - Primary material type (tile, porcelain, ceramic, stone, marble, granite, wood, metal, glass, etc.)
2. factory_name - Manufacturer or factory name
3. factory_group_name - Parent company or group name (if mentioned)

DYNAMIC FIELDS (extract ANY you find):
Extract all other attributes you discover, organized by category:

- Material Properties: composition, type, blend, texture, finish, pattern, weight, density, durability
- Dimensions: length, width, height, thickness, diameter, size, area, volume
- Appearance: color, color_code, gloss_level, sheen, transparency, grain, visual_effects
- Performance: water_resistance, fire_rating, slip_resistance, wear_rating, strength ratings
- Application: recommended_use, installation_method, room_type, traffic_level, care_instructions
- Compliance: certifications, standards, eco_friendly, sustainability, safety_ratings
- Commercial: pricing, availability, supplier, sku, warranty
- Design: designer, studio, collection, series, aesthetic_style
- Technical: specifications, construction, manufacturing_process, grade, class, rating

IMPORTANT:
- If you find attributes NOT in the above list, STILL EXTRACT THEM!
- Group similar attributes together
- Use snake_case for attribute names (e.g., "slip_resistance" not "Slip Resistance")
- Include confidence scores (0.0-1.0) for each extraction
- If a value is uncertain, mark confidence < 0.7

PDF Content:
{pdf_text}

Extract all metadata now:', 
'You are an expert at extracting structured metadata from product specifications. Always respond with valid JSON.', 
TRUE, TRUE),

-- Construction Materials (Tiles, Flooring)
('00000000-0000-0000-0000-000000000000', 'Construction Materials', 'Optimized for tiles, flooring, and construction materials', 'construction', 'metadata_extraction', 'products',
'You are analyzing a CONSTRUCTION MATERIAL specification (tiles, flooring, wall coverings).

CRITICAL FIELDS:
1. material_category - Type (porcelain tile, ceramic tile, natural stone, engineered wood, vinyl, etc.)
2. factory_name - Manufacturer
3. factory_group_name - Parent company

CONSTRUCTION-SPECIFIC FIELDS (prioritize these):
- Dimensions: tile_size, thickness, coverage_area, pieces_per_box, weight_per_box
- Performance: slip_resistance (R9-R13), wear_rating (PEI 1-5), water_absorption, frost_resistance
- Installation: installation_method, grout_joint_size, substrate_requirements, adhesive_type
- Compliance: CE_marking, ISO_standards, ANSI_standards, environmental_certifications
- Technical: rectified_edges, shade_variation, surface_finish, edge_type
- Application: indoor_outdoor, wall_floor, traffic_level, recommended_rooms

PDF Content:
{pdf_text}

Extract all metadata with focus on technical specifications:', 
'You are a construction materials expert. Extract technical specifications with precision.', 
FALSE, TRUE),

-- Interior Design (Furniture, Decor)
('00000000-0000-0000-0000-000000000000', 'Interior Design Products', 'Optimized for furniture, decor, and design products', 'interior_design', 'metadata_extraction', 'products',
'You are analyzing an INTERIOR DESIGN product specification.

CRITICAL FIELDS:
1. material_category - Type (furniture, lighting, textile, accessory, etc.)
2. factory_name - Brand/Manufacturer
3. factory_group_name - Parent company

DESIGN-SPECIFIC FIELDS (prioritize these):
- Design: designer_name, studio, collection_name, series, design_year, aesthetic_style
- Dimensions: width, depth, height, seat_height, arm_height, clearance
- Materials: primary_material, secondary_material, upholstery_fabric, frame_material, finish
- Appearance: color_options, pattern, texture, gloss_level
- Functionality: adjustable, stackable, foldable, modular, assembly_required
- Care: cleaning_instructions, maintenance, warranty_period
- Sustainability: eco_friendly, recycled_content, certifications

PDF Content:
{pdf_text}

Extract all metadata with focus on design and aesthetics:', 
'You are an interior design expert. Extract design details and specifications.', 
FALSE, TRUE);

COMMENT ON TABLE prompt_templates IS 'Customizable AI prompts for different extraction stages and industries';
COMMENT ON TABLE prompt_template_history IS 'Audit trail for prompt template changes';

