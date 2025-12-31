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
from app.services.supabase_client import get_supabase_client

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
        logger.info(f"📥 Downloading {len(urls)} images (max {max_concurrent} concurrent)")
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Download all images concurrently
        tasks = [
            self._download_single_image(url, job_id, workspace_id, semaphore, index)
            for index, url in enumerate(urls)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed downloads
        successful_downloads = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Image download failed: {result}")
                failed_count += 1
            elif result and result.get('success'):
                successful_downloads.append(result)
            else:
                failed_count += 1
        
        logger.info(f"✅ Downloaded {len(successful_downloads)}/{len(urls)} images ({failed_count} failed)")
        
        return successful_downloads

    async def _download_single_image(
        self,
        url: str,
        job_id: str,
        workspace_id: str,
        semaphore: asyncio.Semaphore,
        index: int
    ) -> Dict[str, Any]:
        """
        Download a single image with retry logic.
        
        Args:
            url: Image URL
            job_id: Import job ID
            workspace_id: Workspace ID
            semaphore: Semaphore for concurrency control
            index: Image index
            
        Returns:
            Image reference with storage URL
        """
        async with semaphore:
            for attempt in range(self.max_retries):
                try:
                    # Validate URL
                    if not await self.validate_image_url(url):
                        raise ValueError(f"Invalid image URL: {url}")
                    
                    # Download image
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                            if response.status != 200:
                                raise ValueError(f"HTTP {response.status}: {url}")
                            
                            # Check content type
                            content_type = response.headers.get('Content-Type', '')
                            if not content_type.startswith('image/'):
                                raise ValueError(f"Invalid content type: {content_type}")
                            
                            # Check file size
                            content_length = response.headers.get('Content-Length')
                            if content_length and int(content_length) > self.max_file_size:
                                raise ValueError(f"File too large: {content_length} bytes")
                            
                            # Read image data
                            image_data = await response.read()
                            
                            # Validate image data
                            if len(image_data) == 0:
                                raise ValueError("Empty image data")
                            
                            if len(image_data) > self.max_file_size:
                                raise ValueError(f"File too large: {len(image_data)} bytes")
                    
                    # Generate filename
                    filename = self._generate_filename(url, index)
                    
                    # Store in Supabase Storage
                    storage_result = await self.store_image_in_storage(
                        image_data=image_data,
                        filename=filename,
                        job_id=job_id,
                        workspace_id=workspace_id,
                        content_type=content_type
                    )
                    
                    if not storage_result.get('success'):
                        raise ValueError(f"Storage upload failed: {storage_result.get('error')}")
                    
                    logger.info(f"✅ Downloaded image {index + 1}: {url[:100]}")
                    
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
                    if attempt < self.max_retries - 1:
                        logger.warning(f"⚠️ Retry {attempt + 1}/{self.max_retries} for {url}: {e}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"❌ Failed to download {url} after {self.max_retries} attempts: {e}")
                        return {
                            'success': False,
                            'original_url': url,
                            'error': str(e),
                            'index': index
                        }

    async def validate_image_url(self, url: str) -> bool:
        """
        Validate image URL format.
        
        Args:
            url: Image URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
        
        # Check if URL starts with http:// or https://
        if not url.startswith(('http://', 'https://')):
            return False
        
        # Check for common image extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg')
        url_lower = url.lower()
        
        # URL might have query parameters, so check before '?'
        url_path = url_lower.split('?')[0]
        
        # Either has valid extension or is from known image hosting service
        has_extension = any(url_path.endswith(ext) for ext in valid_extensions)
        is_image_host = any(host in url_lower for host in ['imgur.com', 'cloudinary.com', 'unsplash.com'])
        
        return has_extension or is_image_host

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


