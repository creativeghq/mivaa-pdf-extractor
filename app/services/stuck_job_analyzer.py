"""
Stuck Job Analyzer Service

Analyzes why jobs get stuck and provides detailed diagnostics.
Tracks performance bottlenecks and generates optimization recommendations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class StuckJobAnalyzer:
    """
    Analyzes stuck jobs to identify patterns and root causes.
    
    Tracks:
    - Which stage jobs get stuck at
    - How long each stage takes
    - Memory usage patterns
    - API call failures
    - Common error patterns
    """
    
    def __init__(self):
        self.supabase_client = get_supabase_client()
        self.stuck_job_patterns = defaultdict(int)
        self.stage_durations = defaultdict(list)
        self.failure_reasons = defaultdict(int)
    
    async def analyze_stuck_job(self, job_id: str) -> Dict[str, Any]:
        """
        Analyze a stuck job and determine root cause.
        
        Returns:
            Detailed analysis with recommendations
        """
        try:
            # Get job details
            job_result = self.supabase_client.client.table('background_jobs')\
                .select('*')\
                .eq('id', job_id)\
                .single()\
                .execute()
            
            if not job_result.data:
                return {"error": "Job not found"}
            
            job = job_result.data
            
            # Get checkpoint history
            checkpoints_result = self.supabase_client.client.table('job_checkpoints')\
                .select('*')\
                .eq('job_id', job_id)\
                .order('created_at', desc=False)\
                .execute()
            
            checkpoints = checkpoints_result.data or []
            
            # Analyze
            analysis = {
                "job_id": job_id,
                "filename": job.get('filename'),
                "status": job.get('status'),
                "current_stage": job.get('stage'),
                "progress": job.get('progress_percentage'),
                "stuck_duration_minutes": self._calculate_stuck_duration(job),
                "last_update": job.get('updated_at'),
                "checkpoints_completed": len(checkpoints),
                "stage_analysis": self._analyze_stage_progression(checkpoints),
                "bottleneck_stage": self._identify_bottleneck(checkpoints),
                "root_cause": self._determine_root_cause(job, checkpoints),
                "recommendations": self._generate_recommendations(job, checkpoints),
                "recovery_options": self._get_recovery_options(job, checkpoints)
            }
            
            # Track pattern
            stage = job.get('stage', 'unknown')
            self.stuck_job_patterns[stage] += 1
            
            logger.info(f"📊 Stuck job analysis complete: {job_id}")
            logger.info(f"   Root cause: {analysis['root_cause']}")
            logger.info(f"   Bottleneck: {analysis['bottleneck_stage']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze stuck job {job_id}: {e}")
            return {"error": str(e)}
    
    def _calculate_stuck_duration(self, job: Dict[str, Any]) -> float:
        """Calculate how long job has been stuck (in minutes)"""
        try:
            updated_at = datetime.fromisoformat(job['updated_at'].replace('Z', '+00:00'))
            now = datetime.utcnow()
            duration = (now - updated_at).total_seconds() / 60
            return round(duration, 2)
        except:
            return 0.0
    
    def _analyze_stage_progression(self, checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how long each stage took"""
        stage_times = {}
        
        for i in range(len(checkpoints)):
            checkpoint = checkpoints[i]
            stage = checkpoint.get('stage')
            
            if i == 0:
                # First checkpoint - time from job start
                created_at = datetime.fromisoformat(checkpoint['created_at'].replace('Z', '+00:00'))
                stage_times[stage] = {
                    "duration_seconds": 0,
                    "completed_at": checkpoint['created_at']
                }
            else:
                # Calculate time since previous checkpoint
                prev_checkpoint = checkpoints[i-1]
                prev_time = datetime.fromisoformat(prev_checkpoint['created_at'].replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(checkpoint['created_at'].replace('Z', '+00:00'))
                duration = (curr_time - prev_time).total_seconds()
                
                stage_times[stage] = {
                    "duration_seconds": round(duration, 2),
                    "completed_at": checkpoint['created_at']
                }
        
        return stage_times
    
    def _identify_bottleneck(self, checkpoints: List[Dict[str, Any]]) -> str:
        """Identify which stage is taking the longest"""
        stage_analysis = self._analyze_stage_progression(checkpoints)
        
        if not stage_analysis:
            return "unknown"
        
        # Find stage with longest duration
        bottleneck = max(
            stage_analysis.items(),
            key=lambda x: x[1].get('duration_seconds', 0)
        )
        
        return bottleneck[0]
    
    def _determine_root_cause(self, job: Dict[str, Any], checkpoints: List[Dict[str, Any]]) -> str:
        """Determine most likely root cause of stuck job"""
        stage = job.get('stage', 'unknown')
        stuck_duration = self._calculate_stuck_duration(job)
        
        # Pattern matching for common causes
        if stuck_duration > 30:
            return "silent_crash_no_error_handling"
        elif stage == "extracting_images" and stuck_duration > 10:
            return "image_extraction_timeout_or_memory_exhaustion"
        elif stage == "generating_embeddings" and stuck_duration > 15:
            return "clip_api_timeout_or_rate_limit"
        elif stage == "product_discovery" and stuck_duration > 5:
            return "claude_api_timeout_or_failure"
        elif len(checkpoints) == 0:
            return "job_never_started_background_task_failed"
        else:
            return "unknown_timeout_or_unhandled_exception"
    
    def _generate_recommendations(self, job: Dict[str, Any], checkpoints: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations"""
        root_cause = self._determine_root_cause(job, checkpoints)
        recommendations = []
        
        if "silent_crash" in root_cause:
            recommendations.append("Add comprehensive error handling with try-except blocks")
            recommendations.append("Implement heartbeat monitoring to detect crashes within 2 minutes")
            recommendations.append("Add timeout guards to all async operations")
        
        if "memory_exhaustion" in root_cause:
            recommendations.append("Reduce batch size from 10 to 5 images")
            recommendations.append("Add memory pressure monitoring (pause at 80% usage)")
            recommendations.append("Implement streaming processing (page-by-page)")
        
        if "timeout" in root_cause:
            recommendations.append("Reduce stuck job timeout from 30min to 5min")
            recommendations.append("Add circuit breaker for external API calls")
            recommendations.append("Implement progressive timeout strategy per stage")
        
        if "api" in root_cause:
            recommendations.append("Add retry logic with exponential backoff")
            recommendations.append("Implement API call batching for parallel processing")
            recommendations.append("Add fallback models if primary API fails")
        
        return recommendations
    
    def _get_recovery_options(self, job: Dict[str, Any], checkpoints: List[Dict[str, Any]]) -> List[str]:
        """Get available recovery options"""
        options = []
        
        if len(checkpoints) > 0:
            last_checkpoint = checkpoints[-1]
            options.append(f"Resume from last checkpoint: {last_checkpoint.get('stage')}")
        
        options.append("Restart job from beginning")
        options.append("Mark job as failed and notify user")
        
        return options
    
    async def get_stuck_job_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about stuck jobs"""
        try:
            # Get all stuck jobs (processing > 30 min)
            cutoff_time = (datetime.utcnow() - timedelta(minutes=30)).isoformat()
            
            result = self.supabase_client.client.table('background_jobs')\
                .select('*')\
                .eq('status', 'processing')\
                .lt('updated_at', cutoff_time)\
                .execute()
            
            stuck_jobs = result.data or []
            
            # Analyze patterns
            stage_breakdown = defaultdict(int)
            for job in stuck_jobs:
                stage = job.get('stage', 'unknown')
                stage_breakdown[stage] += 1
            
            return {
                "total_stuck_jobs": len(stuck_jobs),
                "stage_breakdown": dict(stage_breakdown),
                "most_common_stuck_stage": max(stage_breakdown.items(), key=lambda x: x[1])[0] if stage_breakdown else "none",
                "patterns_tracked": dict(self.stuck_job_patterns),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stuck job statistics: {e}")
            return {}


# Global instance
stuck_job_analyzer = StuckJobAnalyzer()


