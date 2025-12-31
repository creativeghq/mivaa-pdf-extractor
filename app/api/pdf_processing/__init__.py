"""
PDF Processing Pipeline Modules

This package contains the modular PDF processing pipeline stages:
- Stage 0: Product Discovery (stage_0_discovery.py)
- Stage 1: Focused Extraction (stage_1_focused_extraction.py)
- Stage 2: Text Chunking (stage_2_chunking.py)
- Stage 3: Image Processing (stage_3_images.py)
- Stage 4: Product Creation (stage_4_products.py)
- Stage 5: Quality Enhancement (stage_5_quality.py)
"""

from .stage_0_discovery import process_stage_0_discovery
from .stage_1_focused_extraction import process_stage_1_focused_extraction
from .stage_2_chunking import process_stage_2_chunking
from .stage_3_images import process_stage_3_images
from .stage_4_products import process_stage_4_products
from .stage_5_quality import process_stage_5_quality

__all__ = [
    'process_stage_0_discovery',
    'process_stage_1_focused_extraction',
    'process_stage_2_chunking',
    'process_stage_3_images',
    'process_stage_4_products',
    'process_stage_5_quality',
]


