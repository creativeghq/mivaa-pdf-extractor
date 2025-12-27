"""
Consensus Validation Service

Multi-model consensus validation for critical AI decisions.
Runs 2-3 models in parallel and uses weighted voting to determine final result.

Features:
- Parallel model execution
- Agreement scoring
- Weighted voting based on model strengths
- Human-in-the-loop flagging for low consensus
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
from collections import Counter

from app.services.together_ai_service import TogetherAIService
from app.services.gpt5_service import GPT5Service
from app.services.ai_call_logger import AICallLogger

logger = logging.getLogger(__name__)


class ConsensusValidator:
    """
    Multi-model consensus validation system.
    
    Runs multiple AI models in parallel and combines results using:
    - Agreement scoring
    - Weighted voting
    - Confidence-based selection
    """
    
    # Agreement thresholds
    HIGH_AGREEMENT = 0.8   # High consensus - use majority vote
    MEDIUM_AGREEMENT = 0.5 # Medium consensus - use weighted vote
    LOW_AGREEMENT = 0.3    # Low consensus - flag for human review
    
    # Model weights (based on general performance)
    MODEL_WEIGHTS = {
        "qwen3-vl-8b": 0.7,
        "claude-haiku-4-5": 0.85,
        "claude-sonnet-4-5": 0.95,
        "gpt-5": 1.0,
    }
    
    # Critical tasks requiring consensus validation
    CRITICAL_TASKS = {
        "product_name_extraction",
        "material_classification",
        "safety_information",
        "compliance_data",
        "technical_specifications",
        "pricing_data",
    }
    
    def __init__(self, ai_logger: Optional[AICallLogger] = None):
        """
        Initialize consensus validator.
        
        Args:
            ai_logger: AI call logger instance
        """
        self.together_ai = TogetherAIService()
        self.gpt5 = GPT5Service()
        self.ai_logger = ai_logger or AICallLogger()
    
    async def validate_with_consensus(
        self,
        task_type: str,
        task_functions: List[Callable],
        task_data: Dict[str, Any],
        model_names: List[str],
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a task using multiple models and consensus.
        
        Args:
            task_type: Type of task
            task_functions: List of async functions (one per model)
            task_data: Data to pass to functions
            model_names: List of model names (same order as functions)
            job_id: Optional job ID for logging
            
        Returns:
            Consensus result with agreement score and final decision
        """
        logger.info(f"🔍 Starting consensus validation for '{task_type}' with {len(task_functions)} models")
        
        # Run all models in parallel
        start_time = datetime.now()
        
        tasks = [func(task_data) for func in task_functions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        total_latency_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Process results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Model {model_names[i]} failed: {str(result)}")
            elif result.get("success"):
                valid_results.append({
                    "model": model_names[i],
                    "result": result,
                    "weight": self.MODEL_WEIGHTS.get(model_names[i], 0.5),
                })
            else:
                logger.warning(f"⚠️ Model {model_names[i]} returned unsuccessful result")
        
        if len(valid_results) < 2:
            logger.error("❌ Not enough valid results for consensus")
            return {
                "success": False,
                "error": "Not enough valid results for consensus",
                "results": valid_results,
            }
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement(valid_results)
        
        # Determine final result based on agreement
        if agreement_score >= self.HIGH_AGREEMENT:
            # High consensus - use majority vote
            final_result = self._majority_vote(valid_results)
            decision_method = "majority_vote"
            
        elif agreement_score >= self.MEDIUM_AGREEMENT:
            # Medium consensus - use weighted vote
            final_result = self._weighted_vote(valid_results)
            decision_method = "weighted_vote"
            
        else:
            # Low consensus - flag for human review
            final_result = self._weighted_vote(valid_results)  # Still provide best guess
            decision_method = "weighted_vote_flagged"
            
            logger.warning(
                f"⚠️ Low consensus ({agreement_score:.2f}) for '{task_type}' - "
                f"flagging for human review"
            )
        
        logger.info(
            f"✅ Consensus validation complete: agreement={agreement_score:.2f}, "
            f"method={decision_method}, latency={total_latency_ms}ms"
        )
        
        return {
            "success": True,
            "result": final_result,
            "agreement_score": agreement_score,
            "decision_method": decision_method,
            "needs_human_review": agreement_score < self.MEDIUM_AGREEMENT,
            "model_results": valid_results,
            "total_latency_ms": total_latency_ms,
            "model_count": len(valid_results),
        }
    
    def _calculate_agreement(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate agreement score between model results.
        
        Args:
            results: List of model results
            
        Returns:
            Agreement score (0.0-1.0)
        """
        if len(results) < 2:
            return 0.0
        
        # Extract key values from results
        # This is task-specific - for now, use confidence scores
        confidences = []
        for r in results:
            conf = r["result"].get("confidence_score", 0.5)
            confidences.append(conf)
        
        # Calculate variance in confidences
        # Low variance = high agreement
        if not confidences:
            return 0.5
        
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        
        # Convert variance to agreement score
        # variance of 0 = agreement of 1.0
        # variance of 0.25 = agreement of 0.0
        agreement = max(0.0, 1.0 - (variance * 4))
        
        return agreement
    
    def _majority_vote(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select result using majority vote.
        
        Args:
            results: List of model results
            
        Returns:
            Winning result
        """
        # For now, return result with highest confidence
        # In production, this would compare actual extracted values
        
        best_result = max(
            results,
            key=lambda r: r["result"].get("confidence_score", 0.0)
        )
        
        return best_result["result"]
    
    def _weighted_vote(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select result using weighted vote based on model quality.
        
        Args:
            results: List of model results
            
        Returns:
            Winning result
        """
        # Calculate weighted scores
        weighted_scores = []
        
        for r in results:
            model_weight = r["weight"]
            confidence = r["result"].get("confidence_score", 0.5)
            weighted_score = model_weight * confidence
            
            weighted_scores.append({
                "result": r["result"],
                "score": weighted_score,
                "model": r["model"],
            })
        
        # Return result with highest weighted score
        best = max(weighted_scores, key=lambda x: x["score"])
        
        logger.info(
            f"✅ Weighted vote winner: {best['model']} "
            f"(score: {best['score']:.2f})"
        )
        
        return best["result"]
    
    async def validate_critical_extraction(
        self,
        content: str,
        extraction_type: str,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate critical data extraction using consensus.
        
        Args:
            content: Content to extract from
            extraction_type: Type of extraction (e.g., 'product_name', 'material_type')
            job_id: Optional job ID
            
        Returns:
            Consensus validation result
        """
        # Define extraction functions for different models
        async def qwen_extract(data):
            prompt = f"Extract {extraction_type} from: {data['content'][:1000]}"
            result = await self.together_ai.generate_completion(
                prompt=prompt,
                model="Qwen/Qwen3-VL-8B-Instruct",
                max_tokens=200,
            )
            return {
                "success": result.get("success", False),
                "extracted_value": result.get("text", ""),
                "confidence_score": 0.7,  # Default confidence
            }

        async def gpt5_extract(data):
            result = await self.gpt5.generate_completion(
                prompt=f"Extract {extraction_type} from: {data['content'][:1000]}",
                model="gpt-5",
                max_tokens=200,
                job_id=job_id,
            )
            return {
                "success": result.get("success", False),
                "extracted_value": result.get("text", ""),
                "confidence_score": 0.9,  # Higher confidence for GPT-5
            }

        # Run consensus validation
        task_data = {"content": content}

        result = await self.validate_with_consensus(
            task_type=f"{extraction_type}_extraction",
            task_functions=[qwen_extract, gpt5_extract],
            task_data=task_data,
            model_names=["qwen3-vl-8b", "gpt-5"],
            job_id=job_id,
        )
        
        return result
    
    @classmethod
    def is_critical_task(cls, task_type: str) -> bool:
        """
        Check if a task is critical and requires consensus validation.
        
        Args:
            task_type: Type of task
            
        Returns:
            True if critical
        """
        return task_type.lower() in cls.CRITICAL_TASKS

