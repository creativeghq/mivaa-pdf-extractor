"""
XML Import Processing Stages

Defines the processing stages for XML import jobs to enable:
- Stage-by-stage progress tracking in Admin dashboard
- Structured logging for debugging
- Checkpoint-based recovery (if needed)

These stages match the XML Import pipeline in data_import_service.py.
"""

from enum import Enum


class XmlImportStage(str, Enum):
    """Processing stages for XML Import jobs with checkpoint support"""

    # Initial state
    INITIALIZED = "initialized"

    # Parsing phase
    PRODUCTS_PARSED = "products_parsed"           # XML parsed, products extracted

    # Image processing phase
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
XML_IMPORT_STAGE_ORDER = [
    XmlImportStage.INITIALIZED,
    XmlImportStage.PRODUCTS_PARSED,
    XmlImportStage.IMAGES_DOWNLOADED,
    XmlImportStage.IMAGES_CLASSIFIED,
    XmlImportStage.CLIPS_GENERATED,
    XmlImportStage.CHUNKS_CREATED,
    XmlImportStage.EMBEDDINGS_QUEUED,
    XmlImportStage.COMPLETED,
]


def get_xml_import_progress(stage: XmlImportStage) -> int:
    """
    Get the progress percentage for a given XML Import stage.

    Args:
        stage: Current processing stage

    Returns:
        Progress percentage (0-100)
    """
    if stage == XmlImportStage.FAILED:
        return 0

    try:
        index = XML_IMPORT_STAGE_ORDER.index(stage)
        return int((index / (len(XML_IMPORT_STAGE_ORDER) - 1)) * 100)
    except ValueError:
        return 0


# Stage descriptions for UI display
XML_IMPORT_STAGE_DESCRIPTIONS = {
    XmlImportStage.INITIALIZED: "Job initialized",
    XmlImportStage.PRODUCTS_PARSED: "Parsing XML and extracting products",
    XmlImportStage.IMAGES_DOWNLOADED: "Downloading product images",
    XmlImportStage.IMAGES_CLASSIFIED: "Classifying images (material vs non-material)",
    XmlImportStage.CLIPS_GENERATED: "Generating CLIP embeddings for images",
    XmlImportStage.CHUNKS_CREATED: "Creating text chunks with quality scoring",
    XmlImportStage.EMBEDDINGS_QUEUED: "Queueing text embeddings for generation",
    XmlImportStage.COMPLETED: "Import completed successfully",
    XmlImportStage.FAILED: "Import failed",
}
