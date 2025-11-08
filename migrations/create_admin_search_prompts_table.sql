-- Create admin_search_prompts table for admin-configurable search prompts
-- This allows admins to customize search behavior without code changes

CREATE TABLE IF NOT EXISTS admin_search_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL,
    
    -- Prompt configuration
    prompt_type VARCHAR(50) NOT NULL CHECK (prompt_type IN ('enhancement', 'formatting', 'filtering', 'enrichment')),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    prompt_text TEXT NOT NULL,
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    
    -- Metadata
    created_by UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Ensure only one default prompt per type per workspace
    UNIQUE(workspace_id, prompt_type, is_default) WHERE is_default = true
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_admin_search_prompts_workspace ON admin_search_prompts(workspace_id);
CREATE INDEX IF NOT EXISTS idx_admin_search_prompts_type ON admin_search_prompts(prompt_type);
CREATE INDEX IF NOT EXISTS idx_admin_search_prompts_active ON admin_search_prompts(is_active) WHERE is_active = true;

-- Add RLS policies (if using Supabase)
ALTER TABLE admin_search_prompts ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view prompts in their workspace
CREATE POLICY "Users can view workspace prompts"
    ON admin_search_prompts
    FOR SELECT
    USING (true);  -- Adjust based on your auth setup

-- Policy: Only admins can insert prompts
CREATE POLICY "Admins can insert prompts"
    ON admin_search_prompts
    FOR INSERT
    WITH CHECK (true);  -- Adjust based on your auth setup

-- Policy: Only admins can update prompts
CREATE POLICY "Admins can update prompts"
    ON admin_search_prompts
    FOR UPDATE
    USING (true);  -- Adjust based on your auth setup

-- Policy: Only admins can delete prompts
CREATE POLICY "Admins can delete prompts"
    ON admin_search_prompts
    FOR DELETE
    USING (true);  -- Adjust based on your auth setup

-- Insert default prompts for demonstration
INSERT INTO admin_search_prompts (workspace_id, prompt_type, name, description, prompt_text, is_active, is_default)
VALUES
    -- Enhancement prompt example
    (
        '00000000-0000-0000-0000-000000000000',  -- Replace with actual workspace ID
        'enhancement',
        'Material Style Enhancement',
        'Enhances search queries with style descriptors',
        'When users search for these terms, interpret them as:
- "modern" → contemporary design, minimalist aesthetic, neutral colors, clean lines
- "luxury" → premium materials, high-end finishes, exclusive collections, designer brands
- "eco-friendly" → sustainable materials, low environmental impact, recyclable, natural
- "industrial" → raw materials, exposed finishes, urban aesthetic, metal and concrete
- "rustic" → natural textures, warm tones, traditional craftsmanship, wood and stone',
        true,
        true
    ),
    
    -- Formatting prompt example
    (
        '00000000-0000-0000-0000-000000000000',  -- Replace with actual workspace ID
        'formatting',
        'Availability-First Ranking',
        'Prioritizes in-stock products in search results',
        'Rank search results by:
1. Availability (in stock first) - 40%
2. Relevance score - 30%
3. Popularity/sales volume - 20%
4. Price competitiveness - 10%

For each result, ensure:
- In-stock products appear first
- Related products are included
- Customer reviews are visible
- Availability status is clear',
        true,
        true
    ),
    
    -- Filtering prompt example
    (
        '00000000-0000-0000-0000-000000000000',  -- Replace with actual workspace ID
        'filtering',
        'Quality Filter',
        'Filters out low-quality and unavailable products',
        'Filter out products that:
- Are out of stock or discontinued
- Have low customer ratings (< 4.0 stars)
- Are marked as clearance or end-of-line
- Have incomplete product information
- Are not available for immediate delivery',
        true,
        true
    ),
    
    -- Enrichment prompt example
    (
        '00000000-0000-0000-0000-000000000000',  -- Replace with actual workspace ID
        'enrichment',
        'Tile Product Enrichment',
        'Adds contextual information for tile products',
        'For each tile product, add:
- Recommended grout colors (based on tile color)
- Installation methods (wall, floor, wet areas)
- Maintenance tips (cleaning, sealing)
- Compatible products (trim, borders, adhesives)
- Design inspiration (room types, styles)
- Technical specifications (slip resistance, water absorption)',
        true,
        true
    );

-- Add trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_admin_search_prompts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER admin_search_prompts_updated_at
    BEFORE UPDATE ON admin_search_prompts
    FOR EACH ROW
    EXECUTE FUNCTION update_admin_search_prompts_updated_at();

-- Comments for documentation
COMMENT ON TABLE admin_search_prompts IS 'Admin-configurable prompts for search result enhancement, formatting, filtering, and enrichment';
COMMENT ON COLUMN admin_search_prompts.prompt_type IS 'Type of prompt: enhancement (query), formatting (ranking), filtering (removal), enrichment (addition)';
COMMENT ON COLUMN admin_search_prompts.is_default IS 'Whether this is the default prompt for this type in this workspace';
COMMENT ON COLUMN admin_search_prompts.prompt_text IS 'The actual prompt text that will be used to process search queries/results';

