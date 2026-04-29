"""
Pipeline observability helpers — structured Sentry spans and per-job log
correlation for the PDF processing pipeline.

The pipeline runs in a FastAPI BackgroundTask, which means the FastAPI
request scope is gone by the time stages execute. We bridge job_id /
document_id / product_id into both Sentry transactions and the stdlib
logging context, so a single grep can follow one job through its full
lifecycle.
"""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

import sentry_sdk

logger = logging.getLogger(__name__)


# Per-task correlation context. Used by `JobContextLogFilter` to inject
# job_id / document_id / product_id into every log record produced inside
# `pipeline_job_scope` / `pipeline_stage_span`.
_current_job_id: ContextVar[Optional[str]] = ContextVar("pdf_job_id", default=None)
_current_document_id: ContextVar[Optional[str]] = ContextVar("pdf_document_id", default=None)
_current_product_id: ContextVar[Optional[str]] = ContextVar("pdf_product_id", default=None)
_current_stage: ContextVar[Optional[str]] = ContextVar("pdf_stage", default=None)


class JobContextLogFilter(logging.Filter):
    """Stamps each log record with current pipeline correlation IDs.

    Install once at startup. Records are mutated in place so existing
    formatters that reference %(job_id)s / %(document_id)s pick them up.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Use getattr so already-set attributes (from thread-local context
        # or sub-logger filters) aren't clobbered with "-".
        if not hasattr(record, 'job_id'):
            record.job_id = _current_job_id.get() or "-"
        if not hasattr(record, 'document_id'):
            record.document_id = _current_document_id.get() or "-"
        if not hasattr(record, 'product_id'):
            record.product_id = _current_product_id.get() or "-"
        if not hasattr(record, 'pdf_stage'):
            record.pdf_stage = _current_stage.get() or "-"
        return True


def install_job_context_filter() -> None:
    """Make sure every log record has job_id/document_id/product_id/pdf_stage
    attrs set BEFORE any formatter runs.

    The previous design only attached this filter to the root logger,
    which works for records that propagate up the standard Python logging
    tree — but fails for records emitted by sub-loggers that have their
    own handlers (uvicorn, sentry, third-party libs running on threads).
    Those records bypass the root filter, hit a formatter with
    %(job_id)s in the pattern, and crash with KeyError.

    Observed failure mode: yolo_endpoint_manager logs from a worker
    thread → no filter runs → KeyError: 'job_id' → format error handler
    re-emits → infinite loop → kernel OOM kill.

    Robust fix: monkey-patch logging.LogRecord.__init__ so the four
    attrs are ALWAYS pre-set on every record, including ones created by
    libraries we don't control. Filter still attached as belt-and-braces.
    """
    root = logging.getLogger()
    if not any(isinstance(f, JobContextLogFilter) for f in root.filters):
        root.addFilter(JobContextLogFilter())

    # Patch LogRecord so the attrs exist BEFORE any formatter sees them.
    # Idempotent — guard so multiple calls don't double-wrap.
    if getattr(logging.LogRecord, "_kai_patched", False):
        return
    _orig_init = logging.LogRecord.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            self.job_id = _current_job_id.get() or "-"
            self.document_id = _current_document_id.get() or "-"
            self.product_id = _current_product_id.get() or "-"
            self.pdf_stage = _current_stage.get() or "-"
        except Exception:
            # ContextVar API failures must NEVER break logging.
            self.job_id = "-"
            self.document_id = "-"
            self.product_id = "-"
            self.pdf_stage = "-"

    logging.LogRecord.__init__ = _patched_init
    logging.LogRecord._kai_patched = True


@contextmanager
def pipeline_job_scope(
    *,
    job_id: str,
    document_id: str,
    workspace_id: Optional[str] = None,
    discovery_model: Optional[str] = None,
    extra_tags: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Open a Sentry transaction for the entire job + bind correlation IDs.

    Yields the Sentry transaction so callers can attach measurements.
    """
    job_token = _current_job_id.set(job_id)
    doc_token = _current_document_id.set(document_id)
    transaction = sentry_sdk.start_transaction(
        op="pdf.pipeline",
        name=f"pdf_pipeline_job",
    )
    transaction.set_tag("job_id", job_id)
    transaction.set_tag("document_id", document_id)
    if workspace_id:
        transaction.set_tag("workspace_id", workspace_id)
    if discovery_model:
        transaction.set_tag("discovery_model", discovery_model)
    for k, v in (extra_tags or {}).items():
        transaction.set_tag(k, str(v))
    try:
        with transaction:
            yield transaction
    finally:
        _current_job_id.reset(job_token)
        _current_document_id.reset(doc_token)


@contextmanager
def pipeline_stage_span(
    stage: str,
    *,
    description: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    product_id: Optional[str] = None,
) -> Iterator[Any]:
    """Open a child span for one pipeline stage (or LLM call).

    Use `stage` like "stage_0.discovery", "stage_3.image_classify",
    "llm.voyage_text_embed", "vecs.upsert_specialized". Spans nest under
    whatever transaction is active — at minimum the job-level transaction
    opened by `pipeline_job_scope`.
    """
    stage_token = _current_stage.set(stage)
    product_token = _current_product_id.set(product_id) if product_id else None
    span = sentry_sdk.start_span(op=stage, description=description or stage)
    if extra_data:
        for k, v in extra_data.items():
            span.set_data(k, v)
    if product_id:
        span.set_tag("product_id", product_id)
    try:
        with span:
            yield span
    finally:
        if product_token is not None:
            _current_product_id.reset(product_token)
        _current_stage.reset(stage_token)


def annotate_llm_call(
    *,
    span: Any,
    model: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    latency_ms: Optional[int] = None,
    success: Optional[bool] = None,
) -> None:
    """Set standard LLM-call fields on a span for cross-trace analytics."""
    if span is None:
        return
    span.set_tag("llm.model", model)
    if input_tokens is not None:
        span.set_data("llm.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_data("llm.output_tokens", output_tokens)
    if cost_usd is not None:
        span.set_data("llm.cost_usd", cost_usd)
    if latency_ms is not None:
        span.set_data("llm.latency_ms", latency_ms)
    if success is not None:
        span.set_tag("llm.success", "true" if success else "false")
