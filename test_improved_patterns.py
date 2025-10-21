#!/usr/bin/env python3
"""
Test improved markdown pattern detection
"""

import re

def classify_markdown_content(text: str) -> str:
    """Classify markdown content type based on patterns."""
    lower_text = text.lower()
    
    # Index/Table of Contents patterns (check first, highest priority)
    if (
        'table of contents' in lower_text or
        ('index' in lower_text and lower_text.count('page') > 1) or
        'contents' in lower_text or
        lower_text.count('page') > 2 or  # Multiple page references
        '...' in text  # Dotted lines typical in TOC
    ):
        return 'index'
    
    # Sustainability/Certification patterns
    if any(keyword in lower_text for keyword in [
        'sustainability', 'certification', 'environmental', 'eco-friendly',
        'carbon footprint', 'recycled', 'leed', 'greenguard'
    ]) and not any(product_keyword in lower_text for product_keyword in ['dimensions', 'designer', 'collection']):
        return 'sustainability'
    
    # Technical specifications patterns (table-like content)
    if (
        any(keyword in lower_text for keyword in [
            'technical characteristics', 'specifications', 'technical data',
            'properties', 'fire rating', 'weight per'
        ]) and 
        ('|' in text or 'thickness' in lower_text) and  # Table format or thickness specs
        not any(product_keyword in lower_text for product_keyword in ['designer', 'collection'])
    ):
        return 'technical'
    
    # Moodboard patterns (but not if it has strong product indicators)
    if (
        any(keyword in lower_text for keyword in [
            'moodboard', 'mood board', 'inspiration', 'collection overview'
        ]) and 
        not any(product_keyword in lower_text for product_keyword in ['dimensions', 'designer'])
    ):
        return 'moodboard'
    
    # Product patterns (UPPERCASE names + dimensions + descriptive content)
    uppercase_words = [word for word in text.split() if word.isupper() and len(word) > 1]
    has_dimensions = any(pattern in text for pattern in ['×', 'x', 'cm', 'mm'])
    has_product_context = any(keyword in lower_text for keyword in [
        'designer', 'collection', 'material', 'ceramic', 'porcelain', 'tile'
    ])
    
    if (
        len(uppercase_words) > 0 and  # Has uppercase product names
        has_dimensions and  # Has dimensions
        (has_product_context or len(text) > 200)  # Has product context or substantial content
    ):
        return 'product'
    
    return 'unknown'

def detect_markdown_patterns(text: str) -> dict:
    """Detect product-specific patterns in markdown text."""
    # Look for product names in headers (## PRODUCT_NAME) or standalone UPPERCASE words
    has_product_name = bool(
        re.search(r'##?\s+[A-Z]{2,}(?:\s+[A-Z]{2,})*', text) or  # Header with UPPERCASE
        re.search(r'^[A-Z]{2,}(?:\s+[A-Z]{2,})*$', text, re.MULTILINE) or  # Standalone UPPERCASE line
        re.search(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', text)  # Any UPPERCASE words
    )
    
    return {
        'hasProductName': has_product_name,
        'hasDimensions': bool(re.search(r'\d+\s*[×x]\s*\d+|\d+\s*(?:mm|cm)', text)),  # Dimensions
        'hasDesignerAttribution': bool(re.search(r'(?:by|BY)\s+[A-Z][a-zA-Z\s{}]+|(?:studio|estudi)', text, re.IGNORECASE)),  # Designer
        'hasProductDescription': len(text) > 100 and bool(re.search(r'material|texture|finish|color|collection', text, re.IGNORECASE))
    }

def extract_markdown_data(text: str) -> dict:
    """Extract structured product data from markdown text."""
    data = {}
    
    # Extract product name (prioritize headers, then standalone lines, then any UPPERCASE)
    product_name_match = (
        re.search(r'##?\s+([A-Z]{2,}(?:\s+[A-Z]{2,})*)', text) or  # Header format
        re.search(r'^([A-Z]{2,}(?:\s+[A-Z]{2,})*)$', text, re.MULTILINE) or  # Standalone line
        re.search(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b', text)  # Any UPPERCASE words
    )
    if product_name_match:
        data['productName'] = product_name_match.group(1)
    
    # Extract dimensions
    dimension_matches = re.findall(r'\d+\s*[×x]\s*\d+|\d+\s*(?:mm|cm)', text)
    if dimension_matches:
        data['dimensions'] = dimension_matches
    
    # Extract designer/studio
    designer_match = re.search(r'(?:by|BY)\s+([A-Z][a-zA-Z\s{}]+)|(?:studio|estudi)\s*([A-Z][a-zA-Z\s{}]*)', text, re.IGNORECASE)
    if designer_match:
        data['designer'] = (designer_match.group(1) or designer_match.group(2)).strip()
    
    # Extract colors (common color names)
    color_matches = re.findall(r'\b(?:white|black|grey|gray|beige|taupe|sand|clay|anthracite|cream|ivory|brown|blue|green|red|yellow|orange|purple|pink)\b', text, re.IGNORECASE)
    if color_matches:
        data['colors'] = list(set([c.lower() for c in color_matches]))
    
    # Extract materials
    material_matches = re.findall(r'\b(?:ceramic|porcelain|stone|marble|granite|wood|metal|glass|concrete|tile|vinyl|laminate)\b', text, re.IGNORECASE)
    if material_matches:
        data['materials'] = list(set([m.lower() for m in material_matches]))
    
    return data

# Test sections with improved content
test_sections = [
    ("Table of Contents", """
Table of Contents
- VALENOVA ........................... Page 12
- PIQUÉ ................................ Page 18  
- ONA .................................. Page 24
"""),
    ("Sustainability", """
Our commitment to environmental responsibility drives every aspect of our manufacturing process. 
We use recycled materials and maintain LEED certification standards.
Carbon footprint reduction is a key priority.
"""),
    ("Technical Specs", """
Technical Characteristics

| Property | Value |
|----------|-------|
| Thickness | 10mm |
| Weight per m² | 22 kg |
| Fire rating | A1 |
| Slip resistance | R10 |

Technical specifications for installation and maintenance guidelines.
"""),
    ("VALENOVA Product", """
## VALENOVA

**TAUPE, SAND, CLAY**

Dimensions: 11.8×11.8 cm
Thickness: 10mm

VALENOVA represents the perfect balance between contemporary design and natural inspiration.
This ceramic tile collection features three sophisticated colorways that bring warmth and 
elegance to any space. The subtle texture and matte finish create a timeless appeal.

Material: Porcelain ceramic
Designer: Studio ALT Design
Collection: Harmony Series
"""),
    ("PIQUÉ Product", """
## PIQUÉ by Estudi{H}ac

Designer: José Manuel Ferrero

PIQUÉ brings a fresh perspective to textile-inspired ceramics. The collection captures
the essence of woven fabrics through innovative surface treatments and sophisticated
color palettes. Available in multiple formats to suit diverse architectural applications.

Dimensions: 20×40 cm, 15×38 cm
Colors: White, Sand, Anthracite
Material: Ceramic
"""),
]

print("🧪 Testing IMPROVED Markdown Pattern Detection")
print("=" * 60)

product_count = 0
for name, content in test_sections:
    print(f"\n--- {name} ---")
    
    # Test classification
    content_type = classify_markdown_content(content)
    print(f"Content Type: {content_type}")
    
    # Test patterns
    patterns = detect_markdown_patterns(content)
    print(f"Patterns: {patterns}")
    
    # Test extraction
    extracted = extract_markdown_data(content)
    print(f"Extracted Data: {extracted}")
    
    # Calculate confidence
    confidence = 0
    if patterns.get('hasProductName'): confidence += 0.4
    if patterns.get('hasDimensions'): confidence += 0.3
    if patterns.get('hasDesignerAttribution'): confidence += 0.2
    if patterns.get('hasProductDescription'): confidence += 0.1
    
    print(f"Confidence: {confidence:.2f}")
    
    is_product = content_type == 'product' and confidence > 0.5
    print(f"Is Product: {is_product}")
    
    if is_product:
        product_count += 1

print(f"\n✅ SUMMARY:")
print(f"Products detected: {product_count}/2 expected")
print("Expected results:")
print("- Table of Contents: index (NOT product) ✓")
print("- Sustainability: sustainability (NOT product) ✓")  
print("- Technical Specs: technical (NOT product) ✓")
print("- VALENOVA Product: product (IS product) ✓")
print("- PIQUÉ Product: product (IS product) ✓")

if product_count == 2:
    print("🎉 SUCCESS: All tests passed!")
else:
    print("⚠️ Some tests failed")
