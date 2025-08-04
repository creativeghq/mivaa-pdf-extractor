#!/usr/bin/env python3
"""
Script to create test PDF files with various characteristics for testing image extraction.

This script creates different types of PDFs:
- Text-heavy PDFs with minimal images
- Image-heavy PDFs with multiple images
- Mixed content PDFs with text and images
- Scanned document PDFs (simulated)
"""

import os
import tempfile
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image as PILImage
import io


def create_test_data_directory():
    """Create test data directory structure."""
    test_data_dir = Path(__file__).parent
    test_data_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different PDF types
    subdirs = ['text_heavy', 'image_heavy', 'mixed_content', 'scanned_docs']
    for subdir in subdirs:
        (test_data_dir / subdir).mkdir(exist_ok=True)
    
    return test_data_dir


def create_sample_images():
    """Create sample images for embedding in PDFs."""
    images = {}
    
    # Create a simple colored rectangle
    img1 = PILImage.new('RGB', (200, 150), color='red')
    img1_bytes = io.BytesIO()
    img1.save(img1_bytes, format='PNG')
    img1_bytes.seek(0)
    images['red_rectangle'] = img1_bytes
    
    # Create a blue circle (simulated)
    img2 = PILImage.new('RGB', (150, 150), color='blue')
    img2_bytes = io.BytesIO()
    img2.save(img2_bytes, format='PNG')
    img2_bytes.seek(0)
    images['blue_square'] = img2_bytes
    
    # Create a green gradient (simulated)
    img3 = PILImage.new('RGB', (300, 100), color='green')
    img3_bytes = io.BytesIO()
    img3.save(img3_bytes, format='PNG')
    img3_bytes.seek(0)
    images['green_banner'] = img3_bytes
    
    # Create a chart-like image
    img4 = PILImage.new('RGB', (250, 200), color='yellow')
    img4_bytes = io.BytesIO()
    img4.save(img4_bytes, format='PNG')
    img4_bytes.seek(0)
    images['chart'] = img4_bytes
    
    return images


def create_text_heavy_pdf(output_path: Path, images: dict):
    """Create a text-heavy PDF with minimal images."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
    )
    story.append(Paragraph("Text-Heavy Document", title_style))
    story.append(Spacer(1, 12))
    
    # Multiple paragraphs of text
    normal_style = styles['Normal']
    for i in range(10):
        text = f"""
        This is paragraph {i+1} of the text-heavy document. This document contains primarily 
        textual content with minimal images. The purpose is to test PDF processing capabilities 
        when dealing with documents that have extensive text content. Lorem ipsum dolor sit amet, 
        consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna 
        aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut 
        aliquip ex ea commodo consequat.
        """
        story.append(Paragraph(text, normal_style))
        story.append(Spacer(1, 12))
    
    # Add one small image
    story.append(Paragraph("Single Image in Text Document:", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    # Add the red rectangle image
    img = Image(images['red_rectangle'], width=100, height=75)
    story.append(img)
    story.append(Spacer(1, 12))
    
    # More text after image
    for i in range(5):
        text = f"""
        Additional paragraph {i+1} after the image. This continues the text-heavy nature 
        of the document while including minimal visual elements for testing purposes.
        """
        story.append(Paragraph(text, normal_style))
        story.append(Spacer(1, 12))
    
    doc.build(story)


def create_image_heavy_pdf(output_path: Path, images: dict):
    """Create an image-heavy PDF with multiple images."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Image-Heavy Document", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    # Brief intro text
    story.append(Paragraph("This document contains multiple images for testing image extraction capabilities.", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Add multiple images with captions
    image_configs = [
        ('red_rectangle', 'Figure 1: Red Rectangle', 150, 112),
        ('blue_square', 'Figure 2: Blue Square', 120, 120),
        ('green_banner', 'Figure 3: Green Banner', 200, 67),
        ('chart', 'Figure 4: Sample Chart', 180, 144),
    ]
    
    for img_key, caption, width, height in image_configs:
        story.append(Paragraph(caption, styles['Heading3']))
        story.append(Spacer(1, 6))
        
        img = Image(images[img_key], width=width, height=height)
        story.append(img)
        story.append(Spacer(1, 20))
        
        # Brief description
        story.append(Paragraph(f"Description for {caption.lower()}. This image demonstrates different visual elements that should be extracted during PDF processing.", styles['Normal']))
        story.append(Spacer(1, 20))
    
    doc.build(story)


def create_mixed_content_pdf(output_path: Path, images: dict):
    """Create a mixed content PDF with balanced text and images."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Mixed Content Document", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    # Introduction
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    intro_text = """
    This document represents a typical business report with mixed content including text, 
    images, and data visualizations. It demonstrates real-world document structures that 
    PDF processing systems need to handle effectively.
    """
    story.append(Paragraph(intro_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Section 1 with image
    story.append(Paragraph("Section 1: Visual Analysis", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    section1_text = """
    The following chart illustrates key performance metrics for the current quarter. 
    This visual representation helps stakeholders understand trends and patterns in the data.
    """
    story.append(Paragraph(section1_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add chart image
    img1 = Image(images['chart'], width=200, height=160)
    story.append(img1)
    story.append(Spacer(1, 12))
    
    # More text
    analysis_text = """
    As shown in the chart above, there has been significant improvement in key metrics. 
    The data indicates positive trends across multiple categories, suggesting effective 
    implementation of strategic initiatives.
    """
    story.append(Paragraph(analysis_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Section 2 with table and image
    story.append(Paragraph("Section 2: Comparative Analysis", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    # Add a simple table
    table_data = [
        ['Metric', 'Q1', 'Q2', 'Q3', 'Q4'],
        ['Revenue', '$100K', '$120K', '$135K', '$150K'],
        ['Growth', '5%', '8%', '12%', '15%'],
        ['Satisfaction', '85%', '87%', '90%', '92%'],
    ]
    table = Table(table_data)
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Add supporting image
    story.append(Paragraph("Supporting Visual Elements", styles['Heading3']))
    story.append(Spacer(1, 6))
    
    img2 = Image(images['green_banner'], width=250, height=83)
    story.append(img2)
    story.append(Spacer(1, 12))
    
    # Conclusion
    story.append(Paragraph("Conclusion", styles['Heading2']))
    conclusion_text = """
    This mixed-content document demonstrates the complexity of real-world PDF processing 
    requirements. Effective extraction systems must handle various content types while 
    maintaining document structure and context.
    """
    story.append(Paragraph(conclusion_text, styles['Normal']))
    
    doc.build(story)


def create_scanned_document_pdf(output_path: Path, images: dict):
    """Create a PDF that simulates a scanned document."""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    
    # Simulate scanned document by creating a page with image-like content
    c.setTitle("Scanned Document Simulation")
    
    # Add a background that simulates paper texture (light gray)
    c.setFillColorRGB(0.98, 0.98, 0.95)
    c.rect(0, 0, width, height, fill=1)
    
    # Add "scanned" text that looks like it was OCR'd
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont("Helvetica", 12)
    
    # Title
    c.drawString(50, height - 100, "SCANNED DOCUMENT - INVOICE #12345")
    
    # Simulate slightly skewed text (as if scanned)
    c.saveState()
    c.translate(50, height - 150)
    c.rotate(0.5)  # Slight rotation to simulate scan imperfection
    
    text_lines = [
        "Date: March 15, 2024",
        "Customer: ABC Corporation",
        "Address: 123 Business St, City, State 12345",
        "",
        "ITEMS:",
        "1. Professional Services    $1,500.00",
        "2. Software License        $2,000.00", 
        "3. Support Package         $500.00",
        "",
        "SUBTOTAL:                  $4,000.00",
        "TAX (8.5%):               $340.00",
        "TOTAL:                    $4,340.00",
        "",
        "Payment Terms: Net 30 days",
        "Thank you for your business!",
    ]
    
    y_offset = 0
    for line in text_lines:
        c.drawString(0, -y_offset, line)
        y_offset += 20
    
    c.restoreState()
    
    # Add a "scanned" image (simulate a signature or stamp)
    c.saveState()
    c.translate(400, 200)
    c.setFillColorRGB(0.2, 0.2, 0.8)
    c.rect(0, 0, 100, 50, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(10, 20, "APPROVED")
    c.restoreState()
    
    # Add some "noise" to simulate scan artifacts
    import random
    c.setFillColorRGB(0.9, 0.9, 0.9)
    for _ in range(50):
        x = random.randint(0, int(width))
        y = random.randint(0, int(height))
        c.circle(x, y, 1, fill=1)
    
    c.save()


def main():
    """Create all test PDF files."""
    print("Creating test PDF files...")
    
    # Create directory structure
    test_data_dir = create_test_data_directory()
    
    # Create sample images
    images = create_sample_images()
    
    # Create different types of PDFs
    pdf_creators = [
        (create_text_heavy_pdf, 'text_heavy/sample_text_heavy.pdf'),
        (create_image_heavy_pdf, 'image_heavy/sample_image_heavy.pdf'),
        (create_mixed_content_pdf, 'mixed_content/sample_mixed_content.pdf'),
        (create_scanned_document_pdf, 'scanned_docs/sample_scanned_doc.pdf'),
    ]
    
    for creator_func, relative_path in pdf_creators:
        output_path = test_data_dir / relative_path
        print(f"Creating {output_path}")
        creator_func(output_path, images)
    
    print("Test PDF files created successfully!")
    print(f"Files created in: {test_data_dir}")


if __name__ == "__main__":
    main()