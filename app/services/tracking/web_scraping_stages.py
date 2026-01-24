"""
Web Scraping Processing Stages

Defines the processing stages for web scraping jobs to enable:
- Stage-by-stage progress tracking in Admin dashboard
- Structured logging for debugging
- Checkpoint-based recovery (if needed)

These stages match the Web Scraping pipeline in web_scraping_service.py.
"""

from enum import Enum


class WebScrapingStage(str, Enum):
    """Processing stages for Web Scraping jobs with checkpoint support"""

    # Initial state
    INITIALIZED = "initialized"

    # Scraping phase
    PAGES_SCRAPED = "pages_scraped"               # Firecrawl pages fetched

    # Product discovery phase
    PRODUCTS_DISCOVERED = "products_discovered"   # AI product extraction complete

    # Image processing phase
    IMAGES_EXTRACTED = "images_extracted"         # Image URLs extracted from markdown
    IMAGES_DOWNLOADED = "images_downloaded"       # Images downloaded to Supabase Storage
    IMAGES_CLASSIFIED = "images_classified"       # Material vs non-material classification
    CLIPS_GENERATED = "clips_generated"           # CLIP embeddings generated for all images

    # Text processing phase
    CHUNKS_CREATED = "chunks_created"             # Smart chunking applied
    EMBEDDINGS_QUEUED = "embeddings_queued"       # Text embeddings queued for generation

    # Final states
    COMPLETED = "completed"                       # All processing complete
    FAILED = "failed"                             # Processing failed


# Stage order for progress calculation (0-100%)
WEB_SCRAPING_STAGE_ORDER = [
    WebScrapingStage.INITIALIZED,
    WebScrapingStage.PAGES_SCRAPED,
    WebScrapingStage.PRODUCTS_DISCOVERED,
    WebScrapingStage.IMAGES_EXTRACTED,
    WebScrapingStage.IMAGES_DOWNLOADED,
    WebScrapingStage.IMAGES_CLASSIFIED,
    WebScrapingStage.CLIPS_GENERATED,
    WebScrapingStage.CHUNKS_CREATED,
    WebScrapingStage.EMBEDDINGS_QUEUED,
    WebScrapingStage.COMPLETED,
]


def get_web_scraping_progress(stage: WebScrapingStage) -> int:
    """
    Get the progress percentage for a given Web Scraping stage.

    Args:
        stage: Current processing stage

    Returns:
        Progress percentage (0-100)
    """
    if stage == WebScrapingStage.FAILED:
        return 0

    try:
        index = WEB_SCRAPING_STAGE_ORDER.index(stage)
        return int((index / (len(WEB_SCRAPING_STAGE_ORDER) - 1)) * 100)
    except ValueError:
        return 0


# Stage descriptions for UI display
WEB_SCRAPING_STAGE_DESCRIPTIONS = {
    WebScrapingStage.INITIALIZED: "Job initialized",
    WebScrapingStage.PAGES_SCRAPED: "Fetching pages from Firecrawl",
    WebScrapingStage.PRODUCTS_DISCOVERED: "Discovering products with AI",
    WebScrapingStage.IMAGES_EXTRACTED: "Extracting image URLs from content",
    WebScrapingStage.IMAGES_DOWNLOADED: "Downloading images to storage",
    WebScrapingStage.IMAGES_CLASSIFIED: "Classifying images (material vs non-material)",
    WebScrapingStage.CLIPS_GENERATED: "Generating CLIP embeddings for images",
    WebScrapingStage.CHUNKS_CREATED: "Creating text chunks with quality scoring",
    WebScrapingStage.EMBEDDINGS_QUEUED: "Queueing text embeddings for generation",
    WebScrapingStage.COMPLETED: "Scraping completed successfully",
    WebScrapingStage.FAILED: "Scraping failed",
}
