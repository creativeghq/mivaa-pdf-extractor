#!/usr/bin/env python3
"""
Create a realistic material specification PDF with functional metadata for testing.

This script creates a tile/material specification PDF containing the 9 functional 
metadata categories that MIVAA is designed to extract:
1. Slip/Safety Ratings
2. Surface Gloss & Reflectivity  
3. Mechanical Properties
4. Thermal Properties
5. Water/Moisture Resistance
6. Chemical/Hygiene Resistance
7. Acoustic/Electrical Properties
8. Environmental/Sustainability
9. Dimensional/Aesthetic Properties
"""

import os
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors


def create_material_spec_pdf(output_path: Path):
    """Create a realistic tile/material specification PDF."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=20,
        alignment=1,  # Center alignment
    )
    
    header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.darkblue
    )
    
    # Title Page
    story.append(Paragraph("PREMIUM CERAMIC TILE SERIES", title_style))
    story.append(Paragraph("Model: PRO-TILE-600x600-R11", styles['Heading2']))
    story.append(Spacer(1, 20))
    
    # Product Overview
    story.append(Paragraph("PRODUCT SPECIFICATIONS", header_style))
    overview_text = """
    High-performance ceramic tile designed for commercial and residential applications. 
    Engineered for durability, safety, and aesthetic appeal in high-traffic environments.
    Manufactured using advanced pressing technology and precision glazing processes.
    """
    story.append(Paragraph(overview_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # 1. SLIP/SAFETY RATINGS SECTION
    story.append(Paragraph("1. SLIP RESISTANCE & SAFETY RATINGS", header_style))
    
    slip_table_data = [
        ['Test Standard', 'Rating', 'Classification', 'Application'],
        ['DIN 51130', 'R11', 'High Slip Resistance', 'Commercial kitchens, wet areas'],
        ['BS 7976-2', 'Slip Resistance Value: 45+', 'Low Slip Potential', 'Public walkways'],
        ['ANSI A137.1', 'DCOF ≥ 0.55', 'Suitable for level interior spaces', 'Residential floors'],
        ['AS/NZS 4586', 'P4 (Wet)', 'Suitable for external ramps', 'Outdoor applications'],
    ]
    
    slip_table = Table(slip_table_data, colWidths=[2*inch, 1.5*inch, 2*inch, 2*inch])
    slip_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(slip_table)
    story.append(Spacer(1, 15))
    
    # 2. SURFACE GLOSS & REFLECTIVITY
    story.append(Paragraph("2. SURFACE GLOSS & REFLECTIVITY", header_style))
    
    gloss_text = """
    Gloss Level: Matte Finish (Gloss Value: 5-15 at 60°)
    Surface Reflectance: Low reflectance, anti-glare properties
    Light Reflection Coefficient: 0.15 (measured at 85° angle)
    Finish Type: Structured surface with micro-texture for enhanced grip
    """
    story.append(Paragraph(gloss_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # 3. MECHANICAL PROPERTIES
    story.append(Paragraph("3. MECHANICAL PROPERTIES", header_style))
    
    mechanical_table_data = [
        ['Property', 'Value', 'Test Standard', 'Classification'],
        ['Breaking Strength', '≥ 1300 N', 'ISO 10545-4', 'Exceeds minimum requirements'],
        ['Modulus of Rupture', '≥ 35 N/mm²', 'ISO 10545-4', 'High strength'],
        ['Surface Abrasion Resistance', 'Class 4 (PEI IV)', 'ISO 10545-7', 'Heavy commercial use'],
        ['Deep Abrasion Volume', '≤ 175 mm³', 'ISO 10545-6', 'Excellent wear resistance'],
        ['Impact Resistance', 'Coefficient of Restitution: 0.55', 'ISO 10545-5', 'Standard impact'],
    ]
    
    mechanical_table = Table(mechanical_table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 2*inch])
    mechanical_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(mechanical_table)
    story.append(Spacer(1, 15))
    
    # 4. THERMAL PROPERTIES
    story.append(Paragraph("4. THERMAL PROPERTIES", header_style))
    
    thermal_text = """
    Thermal Conductivity: 1.2 W/m·K (excellent heat transfer)
    Thermal Expansion: Linear thermal expansion ≤ 9.0 × 10⁻⁶ K⁻¹
    Thermal Shock Resistance: Passed (tested per ISO 10545-9)
    Freeze-Thaw Resistance: Class FT (frost resistant) - tested for 100 cycles
    Maximum Operating Temperature: 85°C continuous, 120°C intermittent
    Fire Rating: A1 - Non-combustible (EN 13501-1)
    """
    story.append(Paragraph(thermal_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Page break for continued content
    story.append(Spacer(1, inch))
    
    # 5. WATER & MOISTURE RESISTANCE
    story.append(Paragraph("5. WATER & MOISTURE RESISTANCE", header_style))
    
    water_table_data = [
        ['Property', 'Value', 'Classification'],
        ['Water Absorption', '≤ 0.5% (E ≤ 0.5%)', 'Group BIa - Very low absorption'],
        ['Moisture Expansion', '≤ 0.6 mm/m', 'Excellent dimensional stability'],
        ['Stain Resistance', 'Class 5 (Unglazed)', 'Maximum stain resistance'],
        ['Chemical Resistance', 'Class A (Glazed surfaces)', 'Superior chemical resistance'],
        ['Crazing Resistance', 'No visible crazing', 'Excellent glaze integrity'],
    ]
    
    water_table = Table(water_table_data, colWidths=[2.5*inch, 2*inch, 2.5*inch])
    water_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightcyan),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(water_table)
    story.append(Spacer(1, 15))
    
    # 6. CHEMICAL & HYGIENE RESISTANCE
    story.append(Paragraph("6. CHEMICAL & HYGIENE RESISTANCE", header_style))
    
    chemical_text = """
    Cleaning Compatibility: Compatible with standard cleaning agents (pH 3-11)
    Antibacterial Properties: ISO 22196 compliant - 99.9% bacterial reduction
    Chemical Resistance: Resistant to household chemicals, pool chemicals, dilute acids
    Stain Resistance: Class 5 rating - highest level of stain resistance
    Antimicrobial Treatment: Silver ion technology integrated into glaze
    Food Safety: FDA approved for food contact surfaces
    """
    story.append(Paragraph(chemical_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # 7. ACOUSTIC & ELECTRICAL PROPERTIES
    story.append(Paragraph("7. ACOUSTIC & ELECTRICAL PROPERTIES", header_style))
    
    acoustic_text = """
    Sound Absorption: Negligible (hard surface - 0.01 NRC)
    Impact Sound Reduction: ΔRw = 2 dB (with suitable underlayment)
    Footstep Sound: Standard ceramic acoustic signature
    Electrical Conductivity: Non-conductive surface (>10¹² Ω·cm)
    Static Dissipation: Anti-static properties when properly installed
    EMF Shielding: No electromagnetic interference properties
    """
    story.append(Paragraph(acoustic_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # 8. ENVIRONMENTAL & SUSTAINABILITY
    story.append(Paragraph("8. ENVIRONMENTAL & SUSTAINABILITY", header_style))
    
    env_table_data = [
        ['Environmental Aspect', 'Rating/Certification', 'Details'],
        ['Recycled Content', '25% post-consumer recycled materials', 'Sustainable manufacturing'],
        ['GREENGUARD Certification', 'GREENGUARD Gold Certified', 'Low chemical emissions'],
        ['LEED Points', 'Contributes to 4 LEED v4.1 credits', 'Green building standard'],
        ['Cradle to Cradle', 'C2C Silver Certified', 'Circular design principles'],
        ['Carbon Footprint', '12.5 kg CO₂ eq/m²', 'Measured lifecycle impact'],
        ['End of Life', '100% recyclable', 'Circular economy compatible'],
    ]
    
    env_table = Table(env_table_data, colWidths=[2*inch, 2.5*inch, 2.5*inch])
    env_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkgreen),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(env_table)
    story.append(Spacer(1, 15))
    
    # 9. DIMENSIONAL & AESTHETIC PROPERTIES
    story.append(Paragraph("9. DIMENSIONAL & AESTHETIC PROPERTIES", header_style))
    
    dimensional_text = """
    Nominal Size: 600mm x 600mm (24" x 24")
    Actual Size: 597mm x 597mm ± 0.75mm
    Thickness: 10mm ± 0.5mm
    Edge Type: Rectified edges with micro-bevel
    Surface Texture: Natural stone texture with R11 slip resistance
    Color: Neutral Gray (Color Code: NG-2024-01)
    Shade Variation: V2 - Slight variation (2-3 shades)
    Planarity: ± 0.5mm (ISO 10545-2)
    Straightness: ± 0.5mm (ISO 10545-2)
    Rectangularity: ± 0.5mm (ISO 10545-2)
    """
    story.append(Paragraph(dimensional_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Installation Guidelines
    story.append(Paragraph("INSTALLATION & APPLICATION GUIDELINES", header_style))
    
    installation_text = """
    Recommended Applications:
    • High-traffic commercial flooring (R11 slip rating)
    • Wet area installations (low water absorption)
    • Food service environments (antibacterial properties)
    • Outdoor terraces and pool areas (frost resistance)
    • Healthcare facilities (easy cleaning, antimicrobial)
    
    Installation Notes:
    • Use appropriate adhesive for substrate type
    • Minimum 3mm grout joint recommended
    • Expansion joints required every 6m
    • Suitable for underfloor heating systems (max 27°C surface temp)
    """
    story.append(Paragraph(installation_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Technical Certifications
    story.append(Paragraph("CERTIFICATIONS & STANDARDS", header_style))
    
    cert_table_data = [
        ['Standard/Certification', 'Result', 'Status'],
        ['ISO 13006 Group BIa', 'Compliant', '✓ Certified'],
        ['ANSI A137.1', 'Compliant', '✓ Certified'],  
        ['CE Marking', 'EN 14411', '✓ Certified'],
        ['GREENGUARD Gold', 'Low Emissions', '✓ Certified'],
        ['LEED v4.1', 'Material Credits', '✓ Eligible'],
        ['ISO 10545 Series', 'Full Test Suite Passed', '✓ Certified'],
    ]
    
    cert_table = Table(cert_table_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    cert_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(cert_table)
    story.append(Spacer(1, 20))
    
    # Footer with manufacturer info
    footer_text = """
    Manufactured by: Advanced Ceramics International
    Quality Assurance: ISO 9001:2015 certified facility
    Technical Support: technical@advancedceramics.com
    Document Version: v2.1 | Last Updated: March 2024
    """
    story.append(Paragraph(footer_text, styles['Normal']))
    
    doc.build(story)


def main():
    """Create the material specification test PDF."""
    print("Creating material specification test PDF...")
    
    # Create output directory
    test_data_dir = Path(__file__).parent
    test_data_dir.mkdir(exist_ok=True)
    
    # Create material spec subdirectory
    material_spec_dir = test_data_dir / 'material_specs'
    material_spec_dir.mkdir(exist_ok=True)
    
    # Create the PDF
    output_path = material_spec_dir / 'ceramic_tile_pro_600x600_r11.pdf'
    print(f"Creating {output_path}")
    create_material_spec_pdf(output_path)
    
    print("Material specification PDF created successfully!")
    print(f"File created: {output_path}")
    
    return output_path


if __name__ == "__main__":
    main()