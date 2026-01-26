"""
YOLO DocParser Layout Detection Service

Detects layout regions in PDF pages using YOLO DocParser model.
Identifies TEXT, IMAGE, TABLE, TITLE, and CAPTION regions with bounding boxes.

Architecture:
1. Convert PDF page to image (PyMuPDF)
2. Send image to YOLO HuggingFace endpoint
3. Parse YOLO output to LayoutRegion objects
4. Sort regions by reading order (top-to-bottom, left-to-right)
5. Return LayoutDetectionResult with all regions
"""

import io
import base64
import logging
import time
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import fitz  # PyMuPDF

from app.models.layout_models import LayoutRegion, BoundingBox, LayoutDetectionResult
from app.services.pdf.yolo_endpoint_manager import YoloEndpointManager
from app.config import get_settings

logger = logging.getLogger(__name__)


class YoloLayoutDetector:
    """
    YOLO DocParser layout detection service.
    
    Detects layout regions in PDF pages and returns structured LayoutRegion objects.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize YOLO layout detector.
        
        Args:
            config: Optional configuration dict (uses settings if not provided)
        """
        settings = get_settings()
        
        if config is None:
            config = settings.get_yolo_config()
        
        self.config = config
        self.enabled = config.get("enabled", True)
        self.endpoint_url = config.get("endpoint_url")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.inference_timeout = config.get("inference_timeout", 30)
        
        # Initialize endpoint manager
        self.endpoint_manager = YoloEndpointManager(
            endpoint_url=self.endpoint_url,
            hf_token=config.get("hf_token", ""),
            endpoint_name=config.get("endpoint_name"),
            namespace=config.get("namespace"),
            auto_pause_timeout=config.get("auto_pause_timeout", 60),
            inference_timeout=self.inference_timeout,
            warmup_timeout=config.get("warmup_timeout", 60),
            max_resume_retries=config.get("max_resume_retries", 3),
            enabled=self.enabled
        )
        
        logger.info(f"✅ YOLO Layout Detector initialized (enabled: {self.enabled})")

    async def _ensure_endpoint_ready(self) -> bool:
        """
        Ensure YOLO endpoint is running and ready for inference.

        The endpoint may have auto-paused during long operations like product discovery
        (which can take several minutes and doesn't use YOLO). This method checks the
        endpoint status and resumes + warms up if needed.

        Returns:
            True if endpoint is ready for inference, False otherwise
        """
        try:
            # Run blocking resume_if_needed() in thread pool to avoid blocking event loop
            is_running = await asyncio.to_thread(self.endpoint_manager.resume_if_needed)

            if not is_running:
                logger.error("❌ Failed to resume YOLO endpoint")
                return False

            # If we just resumed, we need to warmup (the endpoint manager tracks this internally)
            if not self.endpoint_manager.warmup_completed:
                logger.info("🔥 YOLO endpoint needs warmup after resume...")
                await asyncio.to_thread(self.endpoint_manager.warmup)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to ensure YOLO endpoint ready: {e}")
            return False

    def convert_pdf_page_to_image(
        self,
        pdf_path: str,
        page_num: int,
        dpi: int = 150
    ) -> Image.Image:
        """
        Convert PDF page to PIL Image for YOLO analysis.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-based)
            dpi: DPI for rendering (default: 150)
        
        Returns:
            PIL Image object
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Render page to pixmap
            zoom = dpi / 72  # 72 DPI is default
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            doc.close()
            
            logger.debug(f"Converted page {page_num} to image: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to convert PDF page to image: {e}")
            raise
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    async def detect_layout_regions(
        self,
        pdf_path: str,
        page_num: int,
        dpi: int = 150
    ) -> LayoutDetectionResult:
        """
        Detect layout regions in a PDF page using YOLO DocParser.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-based)
            dpi: DPI for rendering (default: 150)
        
        Returns:
            LayoutDetectionResult with detected regions
        """
        if not self.enabled:
            logger.warning("YOLO layout detection is disabled")
            return LayoutDetectionResult(page_number=page_num, regions=[])

        start_time = time.time()

        try:
            # Step 0: Ensure endpoint is running (may have auto-paused during product discovery)
            # The endpoint has a 60s auto-pause timeout, and product discovery can take several minutes
            endpoint_ready = await self._ensure_endpoint_ready()
            if not endpoint_ready:
                logger.warning(f"⚠️ YOLO endpoint not ready, skipping layout detection for page {page_num}")
                return LayoutDetectionResult(page_number=page_num, regions=[])

            # Step 1: Convert page to image
            logger.info(f"📄 Converting page {page_num} to image...")
            image = self.convert_pdf_page_to_image(pdf_path, page_num, dpi)

            # Step 2: Call YOLO endpoint
            logger.info(f"🎯 Detecting layout regions on page {page_num}...")
            regions = await self._call_yolo_endpoint(image, page_num)

            # Step 3: Sort by reading order
            regions = self._sort_by_reading_order(regions)

            # Step 4: Mark endpoint as used
            self.endpoint_manager.mark_used()

            # Calculate detection time
            detection_time_ms = int((time.time() - start_time) * 1000)

            # Create result
            result = LayoutDetectionResult(
                page_number=page_num,
                regions=regions,
                detection_time_ms=detection_time_ms,
                model_version="yolo-docparser"
            )

            logger.info(
                f"✅ Detected {result.total_regions} regions on page {page_num} "
                f"(TEXT: {result.text_regions}, IMAGE: {result.image_regions}, "
                f"TABLE: {result.table_regions}, TITLE: {result.title_regions}, "
                f"CAPTION: {result.caption_regions}) in {detection_time_ms}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Layout detection failed for page {page_num}: {e}")
            return LayoutDetectionResult(page_number=page_num, regions=[])

    async def _call_yolo_endpoint(
        self,
        image: Image.Image,
        page_num: int
    ) -> List[LayoutRegion]:
        """
        Call YOLO HuggingFace endpoint for layout detection.

        Args:
            image: PIL Image to analyze
            page_num: Page number (for bbox metadata)

        Returns:
            List of LayoutRegion objects
        """
        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image)

            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.config.get('hf_token')}",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": image_b64,
                "parameters": {
                    "threshold": self.confidence_threshold
                }
            }

            # Call endpoint
            async with httpx.AsyncClient(timeout=self.inference_timeout) as client:
                response = await client.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

            # Parse YOLO output to LayoutRegion objects
            regions = self._parse_yolo_output(result, page_num, image.size)

            return regions

        except Exception as e:
            logger.error(f"YOLO endpoint call failed: {e}")
            return []

    def _parse_yolo_output(
        self,
        yolo_output: Any,
        page_num: int,
        image_size: Tuple[int, int]
    ) -> List[LayoutRegion]:
        """
        Parse YOLO output to LayoutRegion objects.

        YOLO output format (expected):
        [
            {
                "label": "TEXT",
                "score": 0.95,
                "box": {"xmin": 10, "ymin": 20, "xmax": 500, "ymax": 100}
            },
            ...
        ]

        Args:
            yolo_output: Raw YOLO API response
            page_num: Page number (0-based)
            image_size: (width, height) of the image

        Returns:
            List of LayoutRegion objects
        """
        regions = []

        try:
            # Handle different YOLO output formats
            if isinstance(yolo_output, list):
                detections = yolo_output
            elif isinstance(yolo_output, dict) and "predictions" in yolo_output:
                detections = yolo_output["predictions"]
            else:
                logger.warning(f"Unexpected YOLO output format: {type(yolo_output)}")
                return regions

            for detection in detections:
                try:
                    # Extract label and confidence
                    label = detection.get("label", "").upper()
                    confidence = detection.get("score", 0.0)

                    # Skip low-confidence detections
                    if confidence < self.confidence_threshold:
                        continue

                    # Map YOLO labels to our region types
                    region_type = self._map_label_to_region_type(label)
                    if not region_type:
                        continue

                    # Extract bounding box
                    box = detection.get("box", {})

                    # Check for missing or empty bounding box
                    if not box:
                        logger.warning(
                            f"YOLO detection missing 'box' field! "
                            f"Label: {label}, Score: {confidence}, Full detection: {detection}"
                        )
                        continue

                    # Support both YOLO formats:
                    # Format 1 (expected): {"xmin": 10, "ymin": 20, "xmax": 500, "ymax": 100}
                    # Format 2 (actual):   {"x1": 10, "y1": 20, "x2": 500, "y2": 100}
                    x = float(box.get("xmin", box.get("x1", 0)))
                    y = float(box.get("ymin", box.get("y1", 0)))
                    xmax = float(box.get("xmax", box.get("x2", 0)))
                    ymax = float(box.get("ymax", box.get("y2", 0)))
                    width = xmax - x
                    height = ymax - y

                    # Skip degenerate boxes with zero/negative dimensions
                    if width <= 0 or height <= 0:
                        logger.warning(
                            f"YOLO returned degenerate bounding box! "
                            f"Label: {label}, Score: {confidence:.2f}, "
                            f"Box: xmin={x}, ymin={y}, xmax={xmax}, ymax={ymax} "
                            f"-> width={width}, height={height}. "
                            f"This indicates a YOLO model issue or malformed API response."
                        )
                        continue

                    bbox = BoundingBox(
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        page=page_num
                    )

                    # Create LayoutRegion
                    region = LayoutRegion(
                        type=region_type,
                        bbox=bbox,
                        confidence=confidence,
                        metadata={
                            "yolo_label": label,
                            "image_size": image_size
                        }
                    )

                    regions.append(region)

                except Exception as e:
                    logger.warning(f"Failed to parse YOLO detection: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to parse YOLO output: {e}")

        return regions

    def _map_label_to_region_type(self, label: str) -> Optional[str]:
        """
        Map YOLO label to our region type.

        Args:
            label: YOLO label (e.g., "text", "image", "table")

        Returns:
            Region type or None if not recognized
        """
        label_map = {
            "TEXT": "TEXT",
            "PARAGRAPH": "TEXT",
            "IMAGE": "IMAGE",
            "FIGURE": "IMAGE",
            "PICTURE": "IMAGE",
            "TABLE": "TABLE",
            "TITLE": "TITLE",
            "HEADING": "TITLE",
            "CAPTION": "CAPTION",
            "FIGURE_CAPTION": "CAPTION",
            "TABLE_CAPTION": "CAPTION"
        }

        return label_map.get(label.upper())

    def _sort_by_reading_order(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """
        Sort regions by reading order (top-to-bottom, left-to-right).

        Args:
            regions: List of LayoutRegion objects

        Returns:
            Sorted list with reading_order assigned
        """
        # Sort by Y coordinate (top to bottom), then X coordinate (left to right)
        sorted_regions = sorted(
            regions,
            key=lambda r: (r.bbox.y, r.bbox.x)
        )

        # Assign reading order
        for idx, region in enumerate(sorted_regions):
            region.reading_order = idx

        return sorted_regions

    async def detect_layout_batch(
        self,
        pdf_path: str,
        page_numbers: List[int],
        dpi: int = 150
    ) -> List[LayoutDetectionResult]:
        """
        Detect layout regions for multiple pages.

        Args:
            pdf_path: Path to PDF file
            page_numbers: List of page numbers (0-based)
            dpi: DPI for rendering (default: 150)

        Returns:
            List of LayoutDetectionResult objects
        """
        results = []

        for page_num in page_numbers:
            result = await self.detect_layout_regions(pdf_path, page_num, dpi)
            results.append(result)

        return results

    def pause_endpoint(self) -> bool:
        """
        Scale YOLO endpoint to zero replicas to stop billing.
        Call this after batch processing is complete.

        NOTE: Uses scale_to_zero instead of force_pause because:
        - scale_to_zero: Endpoint auto-resumes on next request (recommended)
        - force_pause: Requires manual resume, can cause delays

        Returns:
            True if scaled down successfully
        """
        return self.endpoint_manager.scale_to_zero()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get YOLO endpoint usage statistics.

        Returns:
            Dictionary with usage stats
        """
        return self.endpoint_manager.get_stats()


