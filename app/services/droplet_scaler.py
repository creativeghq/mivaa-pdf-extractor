"""
Droplet Auto-Scaler Service

Automatically scales DigitalOcean droplet up/down based on PDF processing needs.
This allows using a small (4GB) droplet normally, and scaling to 16GB only when
processing PDFs with CLIP models.

Cost savings: ~$70/month (only pay for large size during processing hours)
"""

import os
import asyncio
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DropletScaler:
    """Manages automatic droplet scaling for PDF processing"""
    
    def __init__(self):
        self.enabled = os.getenv("DROPLET_AUTO_SCALE", "false").lower() == "true"
        self.droplet_id = os.getenv("DROPLET_ID", "")
        self.current_size = "unknown"
        self.scale_lock = asyncio.Lock()
        
        if self.enabled and not self.droplet_id:
            logger.warning("⚠️ DROPLET_AUTO_SCALE enabled but DROPLET_ID not set")
            self.enabled = False
    
    async def scale_up_for_processing(self) -> bool:
        """
        Scale droplet to 16GB before PDF processing.
        
        Returns:
            bool: True if scaled successfully or already at large size
        """
        if not self.enabled:
            logger.debug("Droplet auto-scaling disabled")
            return True
        
        async with self.scale_lock:
            try:
                logger.info("🔄 Checking if droplet needs scaling up...")
                
                # Check current size
                current = await self._get_current_size()
                
                if current == "s-4vcpu-16gb":
                    logger.info("✅ Already at large size (16GB)")
                    return True
                
                logger.info("⬆️  Scaling droplet to 16GB for PDF processing...")
                
                # Execute resize script
                result = await self._execute_resize("up")
                
                if result:
                    logger.info("✅ Droplet scaled to 16GB successfully")
                    self.current_size = "s-4vcpu-16gb"
                    return True
                else:
                    logger.error("❌ Failed to scale droplet")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error scaling droplet: {e}")
                return False
    
    async def scale_down_after_processing(self, force: bool = False) -> bool:
        """
        Scale droplet back to 4GB after PDF processing completes.
        
        Args:
            force: If True, scale down even if there are active jobs
            
        Returns:
            bool: True if scaled successfully or already at small size
        """
        if not self.enabled:
            logger.debug("Droplet auto-scaling disabled")
            return True
        
        async with self.scale_lock:
            try:
                logger.info("🔄 Checking if droplet can scale down...")
                
                # Check current size
                current = await self._get_current_size()
                
                if current == "s-2vcpu-4gb":
                    logger.info("✅ Already at small size (4GB)")
                    return True
                
                # Check for active jobs (unless forced)
                if not force:
                    active_jobs = await self._check_active_jobs()
                    if active_jobs > 0:
                        logger.warning(f"⚠️ Cannot scale down: {active_jobs} active jobs")
                        return False
                
                logger.info("⬇️  Scaling droplet to 4GB to save costs...")
                
                # Execute resize script
                result = await self._execute_resize("down")
                
                if result:
                    logger.info("✅ Droplet scaled to 4GB - saving money! 💰")
                    self.current_size = "s-2vcpu-4gb"
                    return True
                else:
                    logger.error("❌ Failed to scale droplet")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error scaling droplet: {e}")
                return False
    
    async def _get_current_size(self) -> str:
        """Get current droplet size using doctl"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "doctl", "compute", "droplet", "get", self.droplet_id,
                "--format", "Size", "--no-header",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                size = stdout.decode().strip()
                self.current_size = size
                return size
            else:
                logger.error(f"Failed to get droplet size: {stderr.decode()}")
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error getting droplet size: {e}")
            return "unknown"
    
    async def _execute_resize(self, direction: str) -> bool:
        """Execute resize script"""
        try:
            script_path = os.path.join(
                os.path.dirname(__file__),
                "../../scripts/droplet-resize.sh"
            )
            
            proc = await asyncio.create_subprocess_exec(
                script_path, direction,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "DROPLET_ID": self.droplet_id}
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                logger.info(f"Resize output: {stdout.decode()}")
                return True
            else:
                logger.error(f"Resize failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing resize: {e}")
            return False
    
    async def _check_active_jobs(self) -> int:
        """Check for active PDF processing jobs"""
        try:
            from app.services.supabase_service import SupabaseService
            
            supabase = SupabaseService()
            result = supabase.client.table('background_jobs') \
                .select('id') \
                .in_('status', ['pending', 'processing']) \
                .execute()
            
            return len(result.data) if result.data else 0
            
        except Exception as e:
            logger.error(f"Error checking active jobs: {e}")
            return 0


# Global instance
droplet_scaler = DropletScaler()

