"""
Image Download Service

Handles concurrent image downloads from URLs:
- Validates image URLs and content
- Downloads images with retry logic
- Stores images in Supabase Storage
- Returns image references for product linking
- Supports concurrent downloads (max 5 parallel)
"""

import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class ImageDownloadService:
    """Service for downloading and storing images from URLs"""

    def __init__(self):
        supabase_wrapper = get_supabase_client()
        self.supabase = supabase_wrapper.client
        self.max_concurrent = 5  # Maximum concurrent downloads
        self.max_retries = 3  # Maximum retry attempts
        self.timeout = 30  # Request timeout in seconds
        self.max_file_size = 10 * 1024 * 1024  # 10MB max file size

    async def download_images(
        self,
        urls: List[str],
        job_id: str,
        workspace_id: str,
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Download multiple images concurrently.
        
        Args:
            urls: List of image URLs to download
            job_id: Import job ID
            workspace_id: Workspace ID
            max_concurrent: Maximum concurrent downloads (default: 5)
            
        Returns:
            List of image references with storage URLs
        """
        if not urls:
            return []
        
        max_concurrent = max_concurrent or self.max_concurrent
        logger.info(f"[{job_id}] 📥 Downloading {len(urls)} images (max {max_concurrent} concurrent)")

        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)

        # Shared aiohttp session for all downloads in this batch (avoids TCP connection waste)
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._download_single_image(url, job_id, workspace_id, semaphore, index, session)
                for index, url in enumerate(urls)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed downloads
        successful_downloads = []
        failed_count = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[{job_id}] ❌ Image download task raised exception: {result}", exc_info=result)
                failed_count += 1
            elif result and result.get('success'):
                successful_downloads.append(result)
            else:
                failed_count += 1

        logger.info(f"[{job_id}] ✅ Downloaded {len(successful_downloads)}/{len(urls)} images ({failed_count} failed)")

        return successful_downloads

    async def _download_single_image(
        self,
        url: str,
        job_id: str,
        workspace_id: str,
        semaphore: asyncio.Semaphore,
        index: int,
        session: aiohttp.ClientSession = None
    ) -> Dict[str, Any]:
        """
        Download a single image with retry logic.

        Handles HTTP 429 (rate limiting) with longer backoff delays.

        Args:
            url: Image URL
            job_id: Import job ID
            workspace_id: Workspace ID
            semaphore: Semaphore for concurrency control
            index: Image index
            session: Shared aiohttp session for connection reuse (created per batch)

        Returns:
            Image reference with storage URL
        """
        async with semaphore:
            for attempt in range(self.max_retries):
                try:
                    # Validate URL format before making any network request
                    if not self.validate_image_url(url):
                        raise ValueError(f"Invalid image URL: {url}")

                    # Use the shared session (passed from download_images batch call)
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                        # Handle HTTP 429 rate limiting with longer backoff
                        if response.status == 429:
                            retry_after = response.headers.get('Retry-After')
                            try:
                                wait_time = int(retry_after) if retry_after else 30 + (attempt * 30)
                            except ValueError:
                                wait_time = 30 + (attempt * 30)
                            logger.warning(
                                f"[{job_id}] ⏳ Rate limited (429) for {url[:80]}... "
                                f"Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}"
                            )
                            await asyncio.sleep(wait_time)
                            raise ValueError(f"HTTP 429 Rate Limited: {url}")

                        if response.status != 200:
                            raise ValueError(f"HTTP {response.status}: {url}")

                        # Check content type — accept any image/* or octet-stream (some CDNs omit MIME)
                        content_type = response.headers.get('Content-Type', 'image/jpeg')
                        if not content_type.startswith('image/') and 'octet-stream' not in content_type:
                            raise ValueError(f"Invalid content type: {content_type}")

                        # Check declared content-length before reading (fast fail for oversized files)
                        content_length = response.headers.get('Content-Length')
                        if content_length:
                            try:
                                if int(content_length) > self.max_file_size:
                                    raise ValueError(f"File too large: {content_length} bytes")
                            except (ValueError, TypeError):
                                pass  # Malformed header — proceed and check after read

                        # Read image data
                        image_data = await response.read()

                        if len(image_data) == 0:
                            raise ValueError("Empty image data")

                        if len(image_data) > self.max_file_size:
                            raise ValueError(f"File too large: {len(image_data)} bytes")

                    # Generate filename and store
                    filename = self._generate_filename(url, index)

                    storage_result = await self.store_image_in_storage(
                        image_data=image_data,
                        filename=filename,
                        job_id=job_id,
                        workspace_id=workspace_id,
                        content_type=content_type
                    )

                    if not storage_result.get('success'):
                        raise ValueError(f"Storage upload failed: {storage_result.get('error')}")

                    logger.info(f"[{job_id}] ✅ Downloaded image {index + 1}: {url[:100]}")

                    return {
                        'success': True,
                        'original_url': url,
                        'storage_url': storage_result['public_url'],
                        'storage_path': storage_result['storage_path'],
                        'filename': filename,
                        'content_type': content_type,
                        'size_bytes': len(image_data),
                        'index': index
                    }

                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()

                    if attempt < self.max_retries - 1:
                        backoff_time = 30 + (attempt * 30) if is_rate_limit else 2 ** attempt
                        logger.warning(
                            f"[{job_id}] ⚠️ Retry {attempt + 1}/{self.max_retries} for {url[:80]}... "
                            f"Error: {e}. Waiting {backoff_time}s"
                        )
                        await asyncio.sleep(backoff_time)
                    else:
                        logger.error(f"[{job_id}] ❌ Failed to download {url} after {self.max_retries} attempts: {e}")
                        return {
                            'success': False,
                            'original_url': url,
                            'error': str(e),
                            'index': index
                        }

    def validate_image_url(self, url: str) -> bool:
        """
        Validate that a URL is a plausible image URL before making a network request.

        Intentionally permissive: CDN URLs often have no file extension (e.g.
        https://cdn.example.com/media/product/12345). Content-type is checked
        after download, so we only reject clearly non-image patterns here.

        Args:
            url: Image URL to validate

        Returns:
            True if worth attempting to download, False otherwise
        """
        if not url or not isinstance(url, str):
            return False

        # Must be HTTP(S)
        if not url.startswith(('http://', 'https://')):
            return False

        # Reject known non-image paths (documents, stylesheets, scripts, data URIs)
        url_lower = url.lower()
        non_image_extensions = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.css', '.js', '.html', '.htm', '.xml', '.json')
        url_path = url_lower.split('?')[0]
        if any(url_path.endswith(ext) for ext in non_image_extensions):
            return False

        # Must have a non-trivial path (not just a bare domain)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.netloc or len(parsed.path) < 2:
                return False
        except Exception:
            return False

        return True

    async def store_image_in_storage(
        self,
        image_data: bytes,
        filename: str,
        job_id: str,
        workspace_id: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Store image in Supabase Storage.
        
        Args:
            image_data: Image binary data
            filename: Filename for storage
            job_id: Import job ID
            workspace_id: Workspace ID
            content_type: Image content type
            
        Returns:
            Storage result with public URL
        """
        try:
            # Generate storage path
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            storage_path = f"imports/{workspace_id}/{job_id}/{timestamp}_{filename}"
            
            # Upload to pdf-tiles bucket
            supabase_wrapper = get_supabase_client()
            result = await supabase_wrapper.upload_file(
                bucket_name="pdf-tiles",
                file_path=storage_path,
                file_data=image_data,
                content_type=content_type,
                upsert=False
            )
            
            if result.get('success'):
                logger.info(f"✅ Stored image in storage: {storage_path}")
                return {
                    'success': True,
                    'storage_path': storage_path,
                    'public_url': result['public_url']
                }
            else:
                logger.error(f"❌ Failed to store image: {result.get('error')}")
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"❌ Storage upload failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def retry_failed_downloads(
        self,
        failed_urls: List[str],
        job_id: str,
        workspace_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retry downloading failed images.
        
        Args:
            failed_urls: List of URLs that failed to download
            job_id: Import job ID
            workspace_id: Workspace ID
            
        Returns:
            List of successfully downloaded images
        """
        logger.info(f"♻️ Retrying {len(failed_urls)} failed downloads")
        return await self.download_images(failed_urls, job_id, workspace_id)

    def _generate_filename(self, url: str, index: int) -> str:
        """
        Generate a unique filename for the image.
        
        Args:
            url: Original image URL
            index: Image index
            
        Returns:
            Generated filename
        """
        # Extract extension from URL
        url_path = url.split('?')[0]  # Remove query parameters
        extension = Path(url_path).suffix or '.jpg'
        
        # Generate hash from URL for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Create filename
        filename = f"image_{index:03d}_{url_hash}{extension}"
        
        return filename


