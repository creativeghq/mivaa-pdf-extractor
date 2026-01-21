"""
PDF Processing Pipeline Modules

Product-centric pipeline with single-product processing functions:
- Stage 0: Product Discovery (stage_0_discovery.py)
- Stage 1: extract_product_pages (stage_1_focused_extraction.py)
- Stage 2: process_product_chunking (stage_2_chunking.py)
- Stage 3: process_product_images (stage_3_images.py)
- Stage 4: create_single_product (stage_4_products.py)
- Stage 5: Quality Enhancement (stage_5_quality.py)
- Product Processor: process_single_product (product_processor.py)
- Parallel Processor: process_products_parallel (parallel_product_processor.py)
"""

from .stage_0_discovery import process_stage_0_discovery
from .stage_1_focused_extraction import extract_product_pages
from .stage_2_chunking import process_product_chunking
from .stage_3_images import process_product_images
from .stage_4_products import create_single_product, propagate_common_fields_to_products
from .stage_5_quality import process_stage_5_quality
from .product_processor import process_single_product
from .parallel_product_processor import (
    process_products_parallel,
    ParallelProcessingConfig,
    ParallelProcessingResult
)

__all__ = [
    'process_stage_0_discovery',
    'extract_product_pages',
    'process_product_chunking',
    'process_product_images',
    'create_single_product',
    'propagate_common_fields_to_products',
    'process_stage_5_quality',
    'process_single_product',
    'process_products_parallel',
    'ParallelProcessingConfig',
    'ParallelProcessingResult',
]


