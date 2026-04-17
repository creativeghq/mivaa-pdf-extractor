"""
Domain-specific API response schemas.

These models document the actual response shapes returned by each API domain,
eliminating `null` response samples in the OpenAPI spec. Routes that already
have a dedicated response_model (e.g. ImageAnalysisResponse, SearchResponse)
are unaffected — these cover the rest.

Usage:
    from app.schemas.api_responses import JobStatusResponse
    @router.get("/jobs/{id}", response_model=JobStatusResponse)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Generic wrappers ─────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    """Minimal success/failure acknowledgement."""
    success: bool = True
    message: str = ""

class DataResponse(BaseModel):
    """Standard wrapper: success + message + data payload."""
    success: bool = True
    message: str = ""
    data: Optional[Any] = None
    timestamp: Optional[str] = None

class ListDataResponse(BaseModel):
    """Standard wrapper for list results."""
    success: bool = True
    message: Optional[str] = None
    data: List[Any] = Field(default_factory=list)
    count: Optional[int] = None
    timestamp: Optional[str] = None


# ── RAG / Documents ──────────────────────────────────────────────────────────

class JobInfoResponse(BaseModel):
    """Job status response used by /documents/job/{id}."""
    job_id: str
    document_id: Optional[str] = None
    status: str
    progress: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None

class CheckpointListResponse(BaseModel):
    """List of checkpoints for a processing job."""
    job_id: str
    checkpoints: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    stages_completed: List[str] = Field(default_factory=list)

class DocumentContentResponse(BaseModel):
    """Full document content with chunks, images, products."""
    id: str
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    products: List[Dict[str, Any]] = Field(default_factory=list)
    statistics: Optional[Dict[str, Any]] = None

class RelevancyListResponse(BaseModel):
    """Chunk-image relevancy listing."""
    document_id: Optional[str] = None
    relevancies: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    limit: int = 100

class StatsResponse(BaseModel):
    """Workspace or general statistics."""
    success: bool = True
    stats: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

class AITrackingResponse(BaseModel):
    """AI usage tracking data for a job."""
    job_id: str
    tracking: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    summary: Optional[Dict[str, Any]] = None

class StuckJobsResponse(BaseModel):
    """Stuck job analysis results."""
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None


# ── Admin ────────────────────────────────────────────────────────────────────

class SystemHealthResponse(BaseModel):
    """System health overview."""
    success: bool = True
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None

class SystemMetricsResponse(BaseModel):
    """System metrics and performance data."""
    success: bool = True
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)

class CleanupResponse(BaseModel):
    """Data cleanup/backup/export result."""
    success: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    deleted: Optional[int] = None
    timestamp: Optional[str] = None

class PackageStatusResponse(BaseModel):
    """Python package status listing."""
    success: bool = True
    message: str = ""
    packages: List[Dict[str, Any]] = Field(default_factory=list)

class ProductTestResponse(BaseModel):
    """Test product creation result."""
    success: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    products_created: int = 0

class OCRProcessResponse(BaseModel):
    """OCR processing result for an image."""
    success: bool = True
    message: str = ""
    image_id: str = ""
    ocr_text: Optional[str] = None
    confidence: Optional[float] = None


# ── AI Services ──────────────────────────────────────────────────────────────

class ClassificationResponse(BaseModel):
    """Document/image classification result."""
    success: bool = True
    classification: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    processing_time_ms: Optional[float] = None

class BatchClassificationResponse(BaseModel):
    """Batch classification results."""
    success: bool = True
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    processing_time_ms: Optional[float] = None

class BoundaryDetectionResponse(BaseModel):
    """Product boundary detection results."""
    success: bool = True
    boundaries: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0

class ProductGroupingResponse(BaseModel):
    """Product grouping results."""
    success: bool = True
    groups: List[Dict[str, Any]] = Field(default_factory=list)
    ungrouped: int = 0

class ValidationResponse(BaseModel):
    """Product/consensus validation result."""
    success: bool = True
    valid: bool = True
    issues: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None

class EscalationStatsResponse(BaseModel):
    """AI escalation statistics."""
    success: bool = True
    stats: Dict[str, Any] = Field(default_factory=dict)

class ServiceHealthResponse(BaseModel):
    """AI service health check."""
    status: str = "healthy"
    service: str = ""
    version: str = ""
    models: Optional[Dict[str, Any]] = None
    endpoints: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None


# ── Price Monitoring ─────────────────────────────────────────────────────────

class MonitoringActionResponse(BaseModel):
    """Start/stop/check-now monitoring result."""
    success: bool = True
    message: str = ""
    monitoring: Optional[Dict[str, Any]] = None

class PriceHistoryResponse(BaseModel):
    """Price history for a product."""
    success: bool = True
    history: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0

class PriceStatisticsResponse(BaseModel):
    """Price statistics for a product."""
    success: bool = True
    statistics: Optional[Dict[str, Any]] = None

class PriceSourceResponse(BaseModel):
    """Price monitoring source CRUD result."""
    success: bool = True
    message: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, Any]]] = None

class PriceAlertResponse(BaseModel):
    """Price alert CRUD result."""
    success: bool = True
    message: Optional[str] = None
    alert: Optional[Dict[str, Any]] = None
    alerts: Optional[List[Dict[str, Any]]] = None

class PriceJobsResponse(BaseModel):
    """Price monitoring jobs listing."""
    success: bool = True
    jobs: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0


# ── Knowledge Base ───────────────────────────────────────────────────────────

class KBHealthResponse(BaseModel):
    """Knowledge Base API health."""
    status: str = "healthy"
    service: str = "knowledge-base-api"
    version: str = "1.0.0"
    features: Optional[Dict[str, Any]] = None
    endpoints: Optional[Dict[str, Any]] = None


# ── Images ───────────────────────────────────────────────────────────────────

class ImageExportResponse(BaseModel):
    """Image export result."""
    success: bool = True
    message: str = ""
    exported: int = 0
    images: List[Dict[str, Any]] = Field(default_factory=list)

class ImageReclassifyResponse(BaseModel):
    """Image reclassification result."""
    success: bool = True
    message: str = ""
    image_id: str = ""
    old_classification: Optional[str] = None
    new_classification: Optional[str] = None

class SegmentResponse(BaseModel):
    """Image segmentation result."""
    zones: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    processing_time_ms: Optional[float] = None


# ── Health ───────────────────────────────────────────────────────────────────

class DetailedHealthResponse(BaseModel):
    """Detailed system health."""
    overall_status: str
    database: Optional[Dict[str, Any]] = None
    job_monitor: Optional[Dict[str, Any]] = None
    query_metrics: Optional[Dict[str, Any]] = None
    circuit_breaker: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


# ── Monitoring ───────────────────────────────────────────────────────────────

class PerformanceMetricsResponse(BaseModel):
    """Performance metrics summary."""
    success: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None

class StorageEstimateResponse(BaseModel):
    """Storage usage estimates."""
    success: bool = True
    storage: Dict[str, Any] = Field(default_factory=dict)

class PDFHealthResponse(BaseModel):
    """PDF processor health."""
    status: str = "healthy"
    processor: Optional[Dict[str, Any]] = None


# ── Data Import ──────────────────────────────────────────────────────────────

class ImportProcessResponse(BaseModel):
    """Import job creation result."""
    success: bool = True
    message: str = ""
    job_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class ImportHistoryListResponse(BaseModel):
    """Import job history."""
    success: bool = True
    jobs: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0


# ── Job Health ───────────────────────────────────────────────────────────────

class JobDashboardResponse(BaseModel):
    """Job health dashboard overview."""
    success: bool = True
    dashboard: Dict[str, Any] = Field(default_factory=dict)

class StuckJobListResponse(BaseModel):
    """List of stuck jobs."""
    success: bool = True
    stuck_jobs: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0

class JobDiagnosticsResponse(BaseModel):
    """Per-job diagnostic info."""
    success: bool = True
    job_id: str = ""
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# ── Duplicate Detection ──────────────────────────────────────────────────────

class DuplicateDetectionResponse(BaseModel):
    """Duplicate detection results."""
    success: bool = True
    duplicates: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0

class MergeResponse(BaseModel):
    """Product merge/undo result."""
    success: bool = True
    message: str = ""
    merged_product_id: Optional[str] = None

class MergeHistoryResponse(BaseModel):
    """Merge operation history."""
    success: bool = True
    history: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0


# ── Logs ─────────────────────────────────────────────────────────────────────

class LogsResponse(BaseModel):
    """System logs listing."""
    success: bool = True
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0

class LogStatsResponse(BaseModel):
    """Log statistics."""
    success: bool = True
    stats: Dict[str, Any] = Field(default_factory=dict)


# ── Background Agents ────────────────────────────────────────────────────────

class AgentCatalogResponse(BaseModel):
    """List of available agents."""
    success: bool = True
    agents: List[Dict[str, Any]] = Field(default_factory=list)


# ── Interior Design ──────────────────────────────────────────────────────────

class InteriorDesignResponse(BaseModel):
    """Interior design generation result."""
    success: bool = True
    message: str = ""
    task_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# ── Embeddings ───────────────────────────────────────────────────────────────

class EmbeddingHealthResponse(BaseModel):
    """Embedding service health."""
    status: str = "healthy"
    service: str = "embeddings"
    models: Optional[Dict[str, Any]] = None


# ── Category Prototypes ──────────────────────────────────────────────────────

class PrototypeVerifyResponse(BaseModel):
    """Category prototype schema verification result."""
    success: bool = True
    valid: bool = True
    errors: List[str] = Field(default_factory=list)


# ── Chunk Quality ────────────────────────────────────────────────────────────

class FlagReviewResponse(BaseModel):
    """Flagged chunk review result."""
    success: bool = True
    message: str = ""
    flag_id: str = ""
    status: str = ""
