"""
Default Prompt Templates for Extraction

These templates are used when no custom prompts are defined.
Admins can override these through the admin panel.
"""

# Stage: Discovery
DISCOVERY_PRODUCTS = """Analyze this PDF catalog and identify all products with the following information:

**Required Information:**
- Product name and variants
- Page ranges where product appears
- Designer/brand/studio information
- Material and finish details
- Dimensions and specifications
- Related products or collections

**Quality Requirements:**
- Minimum confidence score: {quality_threshold}
- Include page numbers for all references
- Identify product boundaries clearly

**Output Format:**
Return JSON array with:
{{
  "products": [
    {{
      "name": "Product Name",
      "variants": ["Variant 1", "Variant 2"],
      "page_range": [start_page, end_page],
      "designer": "Designer Name",
      "materials": ["Material 1", "Material 2"],
      "dimensions": "Dimensions",
      "confidence_score": 0.95
    }}
  ]
}}

**Context:** {document_context}
**Workspace Settings:** {workspace_settings}
"""

DISCOVERY_CERTIFICATES = """Analyze this document and identify all certificates and certifications:

**Required Information:**
- Certificate name and type
- Certification standard (ISO, CE, etc.)
- Validity period
- Issuing authority
- Page numbers

**Quality Requirements:**
- Minimum confidence score: {quality_threshold}
- Verify certificate authenticity markers

**Output Format:**
Return JSON array with certificate details and confidence scores.

**Context:** {document_context}
"""

DISCOVERY_LOGOS = """Identify all logos, brand marks, and certification marks in this document:

**Required Information:**
- Logo type (brand, certification, quality mark)
- Associated brand/organization
- Page numbers and positions
- Logo quality and clarity

**Quality Requirements:**
- Minimum confidence score: {quality_threshold}
- Distinguish between decorative and official logos

**Output Format:**
Return JSON array with logo details.

**Context:** {document_context}
"""

DISCOVERY_SPECIFICATIONS = """Extract all technical specifications and data sheets:

**Required Information:**
- Specification type
- Technical parameters
- Performance data
- Compliance information
- Page numbers

**Quality Requirements:**
- Minimum confidence score: {quality_threshold}
- Preserve numerical accuracy

**Output Format:**
Return JSON array with specification details.

**Context:** {document_context}
"""

# Stage: Chunking
CHUNKING_PRODUCTS = """Create semantic chunks for product content with the following guidelines:

**Chunking Rules:**
- Chunk size: {workspace_settings}
- Keep product information together
- Preserve context and relationships
- Include page numbers and metadata

**Quality Requirements:**
- Maintain readability
- Preserve product boundaries
- Include relevant context

**Category Guidelines:** {category_guidelines}

**Output Format:**
Return chunks with metadata including category, page_number, and confidence_score.
"""

CHUNKING_DEFAULT = """Create semantic chunks with the following guidelines:

**Chunking Rules:**
- Chunk size: {workspace_settings}
- Preserve context and meaning
- Include metadata

**Quality Requirements:**
- Minimum confidence score: {quality_threshold}
- Maintain coherence

**Output Format:**
Return chunks with metadata.
"""

# Stage: Image Analysis
IMAGE_ANALYSIS_PRODUCTS = """Analyze product images and extract the following:

**Required Information:**
- Product identification
- Visual properties (color, texture, finish)
- Quality assessment
- Relationship to text content

**Quality Requirements:**
- Minimum quality score: {quality_threshold}
- Identify material properties
- Assess image clarity

**Category Guidelines:** {category_guidelines}

**Output Format:**
Return JSON with image analysis results and confidence scores.
"""

IMAGE_ANALYSIS_DEFAULT = """Analyze images and extract relevant information:

**Required Information:**
- Image type and content
- Quality assessment
- Metadata extraction

**Quality Requirements:**
- Minimum quality score: {quality_threshold}

**Output Format:**
Return JSON with analysis results.
"""

# Stage: Entity Creation
ENTITY_CREATION_PRODUCTS = """Create product entities from extracted chunks and images:

**Required Information:**
- Product name (from chunks)
- Product variants
- Associated images
- Metadata (dimensions, materials, designer)
- Page ranges

**Quality Requirements:**
- Minimum confidence score: {quality_threshold}
- Validate all relationships
- Ensure data consistency

**Category Guidelines:** {category_guidelines}

**Output Format:**
Return JSON with complete product entities.
"""

ENTITY_CREATION_DEFAULT = """Create entities from extracted content:

**Required Information:**
- Entity type
- Associated content
- Metadata
- Relationships

**Quality Requirements:**
- Minimum confidence score: {quality_threshold}

**Output Format:**
Return JSON with entity details.
"""

# Default prompts dictionary
DEFAULT_PROMPTS = {
    # Discovery stage
    'discovery_products': DISCOVERY_PRODUCTS,
    'discovery_certificates': DISCOVERY_CERTIFICATES,
    'discovery_logos': DISCOVERY_LOGOS,
    'discovery_specifications': DISCOVERY_SPECIFICATIONS,
    'discovery_default': DISCOVERY_PRODUCTS,
    
    # Chunking stage
    'chunking_products': CHUNKING_PRODUCTS,
    'chunking_certificates': CHUNKING_DEFAULT,
    'chunking_logos': CHUNKING_DEFAULT,
    'chunking_specifications': CHUNKING_DEFAULT,
    'chunking_default': CHUNKING_DEFAULT,
    
    # Image analysis stage
    'image_analysis_products': IMAGE_ANALYSIS_PRODUCTS,
    'image_analysis_certificates': IMAGE_ANALYSIS_DEFAULT,
    'image_analysis_logos': IMAGE_ANALYSIS_DEFAULT,
    'image_analysis_specifications': IMAGE_ANALYSIS_DEFAULT,
    'image_analysis_default': IMAGE_ANALYSIS_DEFAULT,
    
    # Entity creation stage
    'entity_creation_products': ENTITY_CREATION_PRODUCTS,
    'entity_creation_certificates': ENTITY_CREATION_DEFAULT,
    'entity_creation_logos': ENTITY_CREATION_DEFAULT,
    'entity_creation_specifications': ENTITY_CREATION_DEFAULT,
    'entity_creation_default': ENTITY_CREATION_DEFAULT,
}


def get_prompt_template(stage: str, category: str) -> str:
    """
    Get prompt template for stage and category
    
    Args:
        stage: Extraction stage
        category: Content category
        
    Returns:
        Prompt template string
    """
    key = f"{stage}_{category}"
    return DEFAULT_PROMPTS.get(key, DEFAULT_PROMPTS.get(f"{stage}_default", "Extract content from this document."))

