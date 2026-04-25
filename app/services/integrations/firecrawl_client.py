"""
Shared Firecrawl v2 HTTP client.

Used by:
- Competitor price monitoring (`competitor_scraper_service.py`)
- Public price lookup API (`/api/v1/prices/lookup`)
- Future consumers that need structured extraction from a URL

Design:
- Accepts a Pydantic model → uses `model_json_schema()` so the extraction
  schema cannot drift from the code that reads the result.
- `use_javascript_render=True` opts into a slower path with an explicit
  wait action for JS-heavy pages (costs slightly more Firecrawl credits).
- Retry + exponential backoff + credit logging centralized here so every
  caller gets the same behavior.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Generic, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.services.core.ai_call_logger import AICallLogger

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class FirecrawlError(Exception):
    """Firecrawl API call failed after all retries."""
    pass


class FirecrawlResult(Generic[T]):
    """Typed result wrapper. `.data` is None when extraction failed but the
    scrape itself succeeded (e.g. page returned no matching fields)."""

    __slots__ = ("success", "data", "markdown", "credits_used", "latency_ms", "error", "raw_extract")

    def __init__(
        self,
        success: bool,
        data: Optional[T] = None,
        markdown: Optional[str] = None,
        credits_used: int = 0,
        latency_ms: int = 0,
        error: Optional[str] = None,
        raw_extract: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.data = data
        self.markdown = markdown
        self.credits_used = credits_used
        self.latency_ms = latency_ms
        self.error = error
        self.raw_extract = raw_extract


class FirecrawlClient:
    """Thin HTTP client over Firecrawl v2 `/scrape` with structured extraction."""

    BASE_URL = "https://api.firecrawl.dev/v2"
    DEFAULT_TIMEOUT_MS = 30_000
    JS_RENDER_TIMEOUT_MS = 60_000
    JS_RENDER_WAIT_MS = 3_000
    HTTP_TIMEOUT_S = 75.0  # must exceed JS_RENDER_TIMEOUT_MS with headroom

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.firecrawl_api_key
        self.ai_logger = AICallLogger()
        self.max_retries = 3
        self.base_delay = 1.0
        if not self.api_key:
            logger.warning("⚠️ Firecrawl API key not configured — FirecrawlClient will fail all calls")

    async def scrape(
        self,
        url: str,
        extraction_model: Type[T],
        user_id: str,
        workspace_id: Optional[str] = None,
        extraction_prompt: Optional[str] = None,
        use_javascript_render: bool = False,
        only_main_content: bool = True,
        module_slug: Optional[str] = None,
        source_tag: Optional[str] = None,
    ) -> FirecrawlResult[T]:
        """
        Scrape a URL and extract fields defined by `extraction_model`.

        Args:
            url: Target URL.
            extraction_model: Pydantic model describing the fields to extract.
                Field descriptions guide the LLM.
            user_id: For credit billing + audit log.
            workspace_id: For credit billing.
            extraction_prompt: Optional natural-language nudge for the extractor
                (e.g. "Extract the price for 'Oak Flooring'"). Appended to the
                auto-generated per-field prompt.
            use_javascript_render: True for JS-heavy / single-page-app sites.
                Adds a 3s wait action and doubles the timeout. Costs a bit more.
            only_main_content: Strip nav/footer/ads. True for product pages,
                False if you need the full DOM.

        Returns:
            FirecrawlResult[T] with parsed model instance, raw extract dict,
            markdown, credits_used, and latency.
        """
        if not self.api_key:
            return FirecrawlResult(success=False, error="FIRECRAWL_API_KEY not configured")

        start = datetime.utcnow()
        request_body = self._build_request(
            url=url,
            extraction_model=extraction_model,
            extraction_prompt=extraction_prompt,
            use_javascript_render=use_javascript_render,
            only_main_content=only_main_content,
        )

        try:
            response = await self._call_with_retry(request_body)
        except FirecrawlError as e:
            latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            await self._log_call(
                user_id=user_id,
                workspace_id=workspace_id,
                url=url,
                credits_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
                module_slug=module_slug,
                source_tag=source_tag,
            )
            return FirecrawlResult(success=False, error=str(e), latency_ms=latency_ms)

        latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
        credits_used = int(response.get("creditsUsed") or response.get("credits_used") or 1)
        data_envelope = response.get("data") or {}
        markdown = data_envelope.get("markdown")
        raw_extract = data_envelope.get("extract") or data_envelope.get("json") or {}

        parsed: Optional[T] = None
        if raw_extract:
            try:
                parsed = extraction_model.model_validate(raw_extract)
            except ValidationError as ve:
                logger.warning(f"Firecrawl extract failed model validation for {url}: {ve}")

        request_preview: Dict[str, Any] = {
            "model": extraction_model.__name__,
            "js_render": use_javascript_render,
        }
        if source_tag:
            request_preview["source"] = source_tag

        await self._log_call(
            user_id=user_id,
            workspace_id=workspace_id,
            url=url,
            credits_used=credits_used,
            latency_ms=latency_ms,
            success=True,
            request_preview=request_preview,
            response_preview=raw_extract,
            module_slug=module_slug,
            source_tag=source_tag,
        )

        return FirecrawlResult(
            success=True,
            data=parsed,
            markdown=markdown,
            credits_used=credits_used,
            latency_ms=latency_ms,
            raw_extract=raw_extract,
        )

    def _build_request(
        self,
        url: str,
        extraction_model: Type[BaseModel],
        extraction_prompt: Optional[str],
        use_javascript_render: bool,
        only_main_content: bool,
    ) -> Dict[str, Any]:
        """Build the Firecrawl v2 scrape request body."""
        schema = extraction_model.model_json_schema()

        # Field-description summary helps the extractor when the user-facing
        # prompt is short or empty.
        field_hints = ", ".join(
            f"{name}: {props.get('description', '')}"
            for name, props in (schema.get("properties") or {}).items()
            if props.get("description")
        )
        prompt_parts = [
            f"Extract the following fields from the page: {field_hints}."
        ]
        if extraction_prompt:
            prompt_parts.append(extraction_prompt)

        # Firecrawl v2 rejects a top-level `extract` key — structured extraction
        # is expressed as an object inside `formats`: {type: "json", schema, prompt}.
        # The result lands at data_envelope["json"], not data_envelope["extract"].
        body: Dict[str, Any] = {
            "url": url,
            "formats": [
                "markdown",
                {
                    "type": "json",
                    "schema": schema,
                    "prompt": " ".join(prompt_parts),
                },
            ],
            "onlyMainContent": only_main_content,
            "timeout": self.JS_RENDER_TIMEOUT_MS if use_javascript_render else self.DEFAULT_TIMEOUT_MS,
        }

        if use_javascript_render:
            body["actions"] = [{"type": "wait", "milliseconds": self.JS_RENDER_WAIT_MS}]

        return body

    async def _call_with_retry(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST to Firecrawl with exponential backoff. Raises FirecrawlError
        if all attempts fail."""
        last_error: Optional[Exception] = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT_S) as client:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(f"{self.BASE_URL}/scrape", headers=headers, json=body)
                    if resp.status_code == 200:
                        result = resp.json()
                        if result.get("success"):
                            return result
                        last_error = FirecrawlError(f"Firecrawl error: {result.get('error', 'unknown')}")
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        # Retryable
                        last_error = FirecrawlError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    else:
                        # Non-retryable (4xx other than 429) — fail fast
                        raise FirecrawlError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                except httpx.TimeoutException as e:
                    last_error = FirecrawlError(f"timeout: {e}")
                except FirecrawlError:
                    raise
                except Exception as e:
                    last_error = FirecrawlError(str(e))

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.info(f"Firecrawl retry in {delay}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)

        raise last_error or FirecrawlError("all retries failed")

    async def _log_call(
        self,
        *,
        user_id: str,
        workspace_id: Optional[str],
        url: str,
        credits_used: int,
        latency_ms: int,
        success: bool,
        error: Optional[str] = None,
        request_preview: Optional[Dict[str, Any]] = None,
        response_preview: Optional[Dict[str, Any]] = None,
        module_slug: Optional[str] = None,
        source_tag: Optional[str] = None,
    ) -> None:
        try:
            # Ensure source_tag is visible in request_data even if caller didn't
            # supply a request_preview dict.
            if source_tag:
                request_preview = {**(request_preview or {}), "source": source_tag}
            await self.ai_logger.log_firecrawl_call(
                user_id=user_id,
                workspace_id=workspace_id,
                operation_type="scrape",
                credits_used=credits_used,
                latency_ms=latency_ms,
                url=url,
                pages_scraped=1 if success else 0,
                success=success,
                request_data=request_preview,
                response_data=response_preview,
                error_message=error,
                module_slug=module_slug,
            )
        except Exception as e:
            logger.warning(f"Failed to log Firecrawl call: {e}")


_client: Optional[FirecrawlClient] = None


def get_firecrawl_client() -> FirecrawlClient:
    """Singleton accessor."""
    global _client
    if _client is None:
        _client = FirecrawlClient()
    return _client
