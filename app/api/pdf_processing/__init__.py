"""
PDF Processing Pipeline Modules

This package contains the modular PDF processing pipeline stages:
- Stage 0: Product Discovery (discovery.py)
- Stage 1: PDF Extraction (extraction.py)
- Stage 2: Text Chunking (chunking.py)
- Stage 3: Image Processing (images.py)
- Stage 4: Product Creation (products.py)
- Stage 5: Quality Enhancement (quality.py)
"""

from .stage_3_images import process_stage_3_images

__all__ = [
    'process_stage_3_images',
]

