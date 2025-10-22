#!/usr/bin/env python3
"""
Simple test for markdown pattern detection without full service imports
"""

import re

def classify_markdown_content(text: str) -> str:
    """Classify markdown content type based on patterns."""
    lower_text = text.lower()
    
    # Index/Table of Contents patterns
    if (
        'table of contents' in lower_text or
        'index' in lower_text or
        'contents' in lower_text or
        lower_text.count('page') > 2 or  # Multiple page references
        '...' in text  # Dotted lines typical in TOC
    ):
        return 'index'
    
    # Sustainability/Certification patterns
    if any(keyword in lower_text for keyword in [
        'sustainability', 'certification', 'environmental', 'eco-friendly',
        'carbon footprint', 'recycled', 'leed', 'greenguard'
    ]):
        return 'sustainability'
    
    # Technical specifications patterns
    if any(keyword in lower_text for keyword in [
        'technical characteristics', 'specifications', 'technical data',
        'properties', 'fire rating', 'weight per'
    ]) and 'mm' in lower_text and 'thickness' in lower_text:
        return 'technical'
    
    # Moodboard patterns
    if any(keyword in lower_text for keyword in [
        'moodboard', 'mood board', 'inspiration', 'collection overview'
    ]) or ('contrast' in lower_text and not text.count(text.upper()) > 5):
        return 'moodboard'
    
    # Product patterns (UPPERCASE names, dimensions)
    if (
        len([word for word in text.split() if word.isupper() and len(word) > 1]) > 0 and  # Has uppercase words
        any(pattern in text for pattern in ['×', 'x', 'cm', 'mm'])  # Has dimensions
    ):
        return 'product'
    
    return 'unknown'

def detect_markdown_patterns(text: str) -> dict:
    """Detect product-specific patterns in markdown text."""
    return {
        'hasProductName': bool(re.search(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', text)),  # UPPERCASE words
        'hasDimensions': bool(re.search(r'\d+\s*[×x]\s*\d+|\d+\s*(?:mm|cm)', text)),  # Dimensions
        'hasDesignerAttribution': bool(re.search(r'(?:by|BY)\s+[A-Z][a-zA-Z\s{}]+|(?:studio|estudi)', text, re.IGNORECASE)),  # Designer
        'hasProductDescription': len(text) > 100 and bool(re.search(r'material|texture|finish|color|collection', text, re.IGNORECASE))
    }

def extract_markdown_data(text: str) -> dict:
    """Extract structured product data from markdown text."""
    data = {}
    
    # Extract product name (UPPERCASE words)
    product_name_match = re.search(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', text)
    if product_name_match:
        data['productName'] = product_name_match.group(0)
    
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

# Test sections
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
VALENOVA

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
PIQUÉ by Estudi{H}ac

Designer: José Manuel Ferrero

PIQUÉ brings a fresh perspective to textile-inspired ceramics. The collection captures
the essence of woven fabrics through innovative surface treatments and sophisticated
color palettes. Available in multiple formats to suit diverse architectural applications.

Dimensions: 20×40 cm, 15×38 cm
Colors: White, Sand, Anthracite
Material: Ceramic
"""),
]

print("🧪 Testing Markdown Pattern Detection")
print("=" * 50)

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

print("\n✅ SUMMARY:")
print("Expected results:")
print("- Table of Contents: index (NOT product)")
print("- Sustainability: sustainability (NOT product)")  
print("- Technical Specs: technical (NOT product)")
print("- VALENOVA Product: product (IS product)")
print("- PIQUÉ Product: product (IS product)")
