"""
AI Escalation Engine

Intelligent escalation system that routes AI tasks to appropriate models
based on confidence scores, task complexity, and cost constraints.

Features:
- Automatic model escalation for low-confidence results
- Cost-aware routing
- Fallback strategies
- Performance tracking
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging

from app.config.confidence_thresholds import ConfidenceThresholds, EscalationRules
from app.services.ai_call_logger import AICallLogger

logger = logging.getLogger(__name__)


class EscalationEngine:
    """
    Intelligent AI task escalation engine.
    
    Routes tasks to appropriate models based on:
    - Confidence scores
    - Task type and criticality
    - Cost constraints
    - Historical performance
    """
    
    def __init__(self, ai_logger: Optional[AICallLogger] = None):
        """
        Initialize escalation engine.
        
        Args:
            ai_logger: AI call logger instance
        """
        self.ai_logger = ai_logger or AICallLogger()
        self.escalation_stats = {
            "total_escalations": 0,
            "successful_escalations": 0,
            "failed_escalations": 0,
            "cost_saved": 0.0,
            "cost_spent": 0.0,
        }
    
    async def execute_with_escalation(
        self,
        task_type: str,
        task_function: Callable,
        task_data: Dict[str, Any],
        initial_model: str = "llama-4-scout-17b",
        job_id: Optional[str] = None,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute an AI task with automatic escalation on low confidence.
        
        Args:
            task_type: Type of task (e.g., 'material_classification')
            task_function: Async function to execute (should accept model and data)
            task_data: Data to pass to task function
            initial_model: Model to try first
            job_id: Optional job ID for logging
            max_attempts: Maximum escalation attempts
            
        Returns:
            Dict with result, confidence, model_used, escalation_count, etc.
        """
        current_model = initial_model
        escalation_count = 0
        attempts_history = []
        
        logger.info(f"🚀 Starting task '{task_type}' with model '{current_model}'")
        
        for attempt in range(max_attempts):
            try:
                # Execute task with current model
                start_time = datetime.now()
                
                result = await task_function(
                    model=current_model,
                    data=task_data
                )
                
                end_time = datetime.now()
                latency_ms = int((end_time - start_time).total_seconds() * 1000)
                
                # Extract confidence score
                confidence_score = result.get("confidence_score", 0.0)
                
                # Record attempt
                attempts_history.append({
                    "model": current_model,
                    "confidence": confidence_score,
                    "latency_ms": latency_ms,
                    "attempt": attempt + 1,
                })
                
                logger.info(
                    f"✅ Task '{task_type}' completed with model '{current_model}' "
                    f"(confidence: {confidence_score:.2f}, latency: {latency_ms}ms)"
                )
                
                # Check if result is acceptable
                if ConfidenceThresholds.is_acceptable(task_type, confidence_score):
                    # Result is good enough
                    quality_level = ConfidenceThresholds.get_quality_level(task_type, confidence_score)
                    
                    logger.info(
                        f"✅ Result accepted with {quality_level} quality "
                        f"(confidence: {confidence_score:.2f})"
                    )
                    
                    return {
                        "success": True,
                        "result": result,
                        "confidence_score": confidence_score,
                        "quality_level": quality_level,
                        "model_used": current_model,
                        "escalation_count": escalation_count,
                        "attempts_history": attempts_history,
                        "total_latency_ms": sum(a["latency_ms"] for a in attempts_history),
                    }
                
                # Result is not acceptable, check if we should escalate
                if ConfidenceThresholds.should_escalate(task_type, confidence_score):
                    # Try to escalate
                    next_model = EscalationRules.get_next_model(current_model, task_type)
                    
                    if next_model and attempt < max_attempts - 1:
                        escalation_count += 1
                        self.escalation_stats["total_escalations"] += 1
                        
                        logger.warning(
                            f"⚠️ Low confidence ({confidence_score:.2f}), "
                            f"escalating from '{current_model}' to '{next_model}'"
                        )
                        
                        # Log escalation
                        if self.ai_logger and job_id:
                            await self.ai_logger.log_ai_call({
                                "job_id": job_id,
                                "task": task_type,
                                "model": current_model,
                                "action": "escalate",
                                "fallback_reason": f"Low confidence: {confidence_score:.2f}",
                                "confidence_score": confidence_score,
                                "latency_ms": latency_ms,
                            })
                        
                        current_model = next_model
                        continue
                    else:
                        # No more escalation options
                        logger.warning(
                            f"⚠️ No more escalation options available. "
                            f"Accepting result with confidence {confidence_score:.2f}"
                        )
                        
                        return {
                            "success": True,
                            "result": result,
                            "confidence_score": confidence_score,
                            "quality_level": "poor",
                            "model_used": current_model,
                            "escalation_count": escalation_count,
                            "attempts_history": attempts_history,
                            "total_latency_ms": sum(a["latency_ms"] for a in attempts_history),
                            "warning": "Low confidence result - no escalation available",
                        }
                else:
                    # Confidence is acceptable but not great
                    quality_level = ConfidenceThresholds.get_quality_level(task_type, confidence_score)
                    
                    return {
                        "success": True,
                        "result": result,
                        "confidence_score": confidence_score,
                        "quality_level": quality_level,
                        "model_used": current_model,
                        "escalation_count": escalation_count,
                        "attempts_history": attempts_history,
                        "total_latency_ms": sum(a["latency_ms"] for a in attempts_history),
                    }
                    
            except Exception as e:
                logger.error(f"❌ Error executing task with model '{current_model}': {str(e)}")
                
                # Try to escalate on error
                next_model = EscalationRules.get_next_model(current_model, task_type)
                
                if next_model and attempt < max_attempts - 1:
                    escalation_count += 1
                    self.escalation_stats["total_escalations"] += 1
                    self.escalation_stats["failed_escalations"] += 1
                    
                    logger.warning(f"⚠️ Error occurred, escalating to '{next_model}'")
                    
                    current_model = next_model
                    continue
                else:
                    # No more options, return error
                    return {
                        "success": False,
                        "error": str(e),
                        "model_used": current_model,
                        "escalation_count": escalation_count,
                        "attempts_history": attempts_history,
                    }
        
        # Max attempts reached
        logger.error(f"❌ Max attempts ({max_attempts}) reached for task '{task_type}'")
        
        return {
            "success": False,
            "error": f"Max attempts ({max_attempts}) reached",
            "escalation_count": escalation_count,
            "attempts_history": attempts_history,
        }
    
    async def execute_with_fallback(
        self,
        task_type: str,
        primary_function: Callable,
        fallback_function: Callable,
        task_data: Dict[str, Any],
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute task with primary function, fall back to secondary if needed.
        
        Args:
            task_type: Type of task
            primary_function: Primary async function to try
            fallback_function: Fallback async function
            task_data: Data to pass to functions
            job_id: Optional job ID for logging
            
        Returns:
            Result from primary or fallback function
        """
        try:
            # Try primary function
            start_time = datetime.now()
            result = await primary_function(task_data)
            end_time = datetime.now()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            confidence_score = result.get("confidence_score", 0.0)
            
            # Check if result is acceptable
            if ConfidenceThresholds.is_acceptable(task_type, confidence_score):
                logger.info(f"✅ Primary function succeeded for '{task_type}'")
                return {
                    "success": True,
                    "result": result,
                    "source": "primary",
                    "confidence_score": confidence_score,
                    "latency_ms": latency_ms,
                }
            else:
                # Use fallback
                logger.warning(
                    f"⚠️ Primary function low confidence ({confidence_score:.2f}), "
                    f"using fallback for '{task_type}'"
                )
                
                fallback_start = datetime.now()
                fallback_result = await fallback_function(task_data)
                fallback_end = datetime.now()
                fallback_latency = int((fallback_end - fallback_start).total_seconds() * 1000)
                
                return {
                    "success": True,
                    "result": fallback_result,
                    "source": "fallback",
                    "confidence_score": fallback_result.get("confidence_score", 0.0),
                    "latency_ms": latency_ms + fallback_latency,
                    "fallback_reason": f"Low confidence: {confidence_score:.2f}",
                }
                
        except Exception as e:
            # Primary failed, use fallback
            logger.error(f"❌ Primary function failed for '{task_type}': {str(e)}")
            
            try:
                fallback_start = datetime.now()
                fallback_result = await fallback_function(task_data)
                fallback_end = datetime.now()
                fallback_latency = int((fallback_end - fallback_start).total_seconds() * 1000)
                
                return {
                    "success": True,
                    "result": fallback_result,
                    "source": "fallback",
                    "confidence_score": fallback_result.get("confidence_score", 0.0),
                    "latency_ms": fallback_latency,
                    "fallback_reason": f"Primary error: {str(e)}",
                }
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {str(fallback_error)}")
                return {
                    "success": False,
                    "error": f"Both primary and fallback failed: {str(e)}, {str(fallback_error)}",
                }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get escalation statistics."""
        return self.escalation_stats.copy()

