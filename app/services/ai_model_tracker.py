"""
AI Model Tracking Service

Tracks which AI models are used at each stage of PDF processing,
their results, confidence scores, and detailed metrics.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AIModelCall:
    """Represents a single AI model call"""
    model_name: str  # e.g., "Qwen", "Anthropic", "CLIP", "OpenAI"
    stage: str  # e.g., "classification", "boundary_detection", "embedding"
    task: str  # e.g., "product_classification", "image_embedding"
    timestamp: str
    latency_ms: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    confidence_score: Optional[float] = None
    result_summary: Optional[str] = None
    items_processed: int = 0
    success: bool = True
    error: Optional[str] = None


class AIModelTracker:
    """
    Tracks AI model usage throughout PDF processing pipeline.
    
    Provides detailed metrics on:
    - Which models were used at each stage
    - Confidence scores and results
    - Token usage and costs
    - Processing time per model
    - Success/failure rates
    """
    
    def __init__(self, job_id: str):
        """Initialize tracker for a specific job"""
        self.job_id = job_id
        self.calls: List[AIModelCall] = []
        self.stage_summary: Dict[str, Dict[str, Any]] = {}
        self.start_time = datetime.utcnow()
        
    def log_model_call(
        self,
        model_name: str,
        stage: str,
        task: str,
        latency_ms: int,
        confidence_score: Optional[float] = None,
        result_summary: Optional[str] = None,
        items_processed: int = 0,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Log an AI model call.
        
        Args:
            model_name: Name of AI model (Qwen, Anthropic, CLIP, OpenAI)
            stage: Processing stage (classification, boundary_detection, embedding, etc.)
            task: Specific task (product_classification, image_embedding, etc.)
            latency_ms: Response time in milliseconds
            confidence_score: Confidence score (0.0-1.0)
            result_summary: Summary of results
            items_processed: Number of items processed
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            success: Whether call succeeded
            error: Error message if failed
        """
        call = AIModelCall(
            model_name=model_name,
            stage=stage,
            task=task,
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=latency_ms,
            confidence_score=confidence_score,
            result_summary=result_summary,
            items_processed=items_processed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            error=error
        )
        
        self.calls.append(call)
        logger.info(
            f"📊 AI Model Call: {model_name} | Stage: {stage} | Task: {task} | "
            f"Latency: {latency_ms}ms | Confidence: {confidence_score} | "
            f"Items: {items_processed} | Success: {success}"
        )
        
        # Update stage summary
        self._update_stage_summary(call)
    
    def _update_stage_summary(self, call: AIModelCall) -> None:
        """Update summary statistics for a stage"""
        if call.stage not in self.stage_summary:
            self.stage_summary[call.stage] = {
                "models_used": [],
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_latency_ms": 0,
                "total_items_processed": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "avg_confidence": 0.0,
                "confidence_scores": []
            }
        
        summary = self.stage_summary[call.stage]
        
        # Track models used
        if call.model_name not in summary["models_used"]:
            summary["models_used"].append(call.model_name)
        
        # Update counters
        summary["total_calls"] += 1
        if call.success:
            summary["successful_calls"] += 1
        else:
            summary["failed_calls"] += 1
        
        summary["total_latency_ms"] += call.latency_ms
        summary["total_items_processed"] += call.items_processed
        
        if call.input_tokens:
            summary["total_input_tokens"] += call.input_tokens
        if call.output_tokens:
            summary["total_output_tokens"] += call.output_tokens
        
        # Track confidence scores
        if call.confidence_score is not None:
            summary["confidence_scores"].append(call.confidence_score)
            summary["avg_confidence"] = sum(summary["confidence_scores"]) / len(summary["confidence_scores"])
    
    def get_job_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all AI model usage for this job.
        
        Returns:
            Dictionary with detailed AI model tracking information
        """
        total_latency = sum(call.latency_ms for call in self.calls)
        total_items = sum(call.items_processed for call in self.calls)
        total_input_tokens = sum(call.input_tokens or 0 for call in self.calls)
        total_output_tokens = sum(call.output_tokens or 0 for call in self.calls)
        
        # Get unique models used
        models_used = list(set(call.model_name for call in self.calls))
        
        # Calculate success rate
        successful = sum(1 for call in self.calls if call.success)
        total_calls = len(self.calls)
        success_rate = (successful / total_calls * 100) if total_calls > 0 else 0
        
        return {
            "job_id": self.job_id,
            "total_ai_calls": total_calls,
            "successful_calls": successful,
            "failed_calls": total_calls - successful,
            "success_rate_percent": round(success_rate, 2),
            "models_used": models_used,
            "total_latency_ms": total_latency,
            "avg_latency_ms": round(total_latency / total_calls, 2) if total_calls > 0 else 0,
            "total_items_processed": total_items,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "stage_summary": self.stage_summary,
            "detailed_calls": [asdict(call) for call in self.calls]
        }
    
    def get_stage_details(self, stage: str) -> Dict[str, Any]:
        """Get detailed information for a specific stage"""
        if stage not in self.stage_summary:
            return {}
        
        summary = self.stage_summary[stage]
        stage_calls = [call for call in self.calls if call.stage == stage]
        
        return {
            "stage": stage,
            "models_used": summary["models_used"],
            "total_calls": summary["total_calls"],
            "successful_calls": summary["successful_calls"],
            "failed_calls": summary["failed_calls"],
            "success_rate_percent": round(
                (summary["successful_calls"] / summary["total_calls"] * 100) 
                if summary["total_calls"] > 0 else 0, 2
            ),
            "total_latency_ms": summary["total_latency_ms"],
            "avg_latency_ms": round(
                summary["total_latency_ms"] / summary["total_calls"]
                if summary["total_calls"] > 0 else 0, 2
            ),
            "total_items_processed": summary["total_items_processed"],
            "avg_confidence": round(summary["avg_confidence"], 3),
            "total_input_tokens": summary["total_input_tokens"],
            "total_output_tokens": summary["total_output_tokens"],
            "calls": [asdict(call) for call in stage_calls]
        }
    
    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get statistics for a specific AI model"""
        model_calls = [call for call in self.calls if call.model_name == model_name]
        
        if not model_calls:
            return {}
        
        successful = sum(1 for call in model_calls if call.success)
        total_latency = sum(call.latency_ms for call in model_calls)
        total_items = sum(call.items_processed for call in model_calls)
        
        return {
            "model": model_name,
            "total_calls": len(model_calls),
            "successful_calls": successful,
            "failed_calls": len(model_calls) - successful,
            "success_rate_percent": round(
                (successful / len(model_calls) * 100) if model_calls else 0, 2
            ),
            "total_latency_ms": total_latency,
            "avg_latency_ms": round(total_latency / len(model_calls), 2) if model_calls else 0,
            "total_items_processed": total_items,
            "stages_used": list(set(call.stage for call in model_calls)),
            "tasks": list(set(call.task for call in model_calls))
        }
    
    def format_for_metadata(self) -> Dict[str, Any]:
        """Format tracking data for job metadata storage"""
        summary = self.get_job_summary()
        
        return {
            "ai_models_used": summary["models_used"],
            "ai_total_calls": summary["total_ai_calls"],
            "ai_success_rate": summary["success_rate_percent"],
            "ai_total_latency_ms": summary["total_latency_ms"],
            "ai_total_tokens": summary["total_tokens"],
            "ai_stage_summary": {
                stage: {
                    "models": data["models_used"],
                    "calls": data["total_calls"],
                    "success_rate": round(
                        (data["successful_calls"] / data["total_calls"] * 100)
                        if data["total_calls"] > 0 else 0, 2
                    ),
                    "items_processed": data["total_items_processed"],
                    "avg_confidence": round(data["avg_confidence"], 3)
                }
                for stage, data in summary["stage_summary"].items()
            }
        }

