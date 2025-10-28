"""
AI Call Logger Service

Tracks all AI model calls with costs, latency, confidence scores, and fallback decisions.
Provides comprehensive monitoring and analytics for AI usage across the platform.

Author: Material Kai Vision Platform
Created: 2025-10-27
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from decimal import Decimal
import json

from app.core.supabase_client import SupabaseClient
from app.config.ai_pricing import ai_pricing

logger = logging.getLogger(__name__)


class AICallLogger:
    """
    AI Call Logger Service
    
    Tracks every AI call with:
    - Cost and token usage
    - Latency and performance
    - Confidence scores (4-factor weighted)
    - Fallback decisions
    - Request/response data for debugging
    """
    
    def __init__(self, supabase_client: Optional[SupabaseClient] = None):
        """Initialize AI Call Logger"""
        self.supabase = supabase_client or SupabaseClient()
        self.logger = logging.getLogger(__name__)
    
    async def log_ai_call(
        self,
        task: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: int,
        confidence_score: float,
        confidence_breakdown: Dict[str, float],
        action: str,
        job_id: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Log an AI call to the database.
        
        Args:
            task: Type of task (document_classification, product_extraction, etc.)
            model: AI model used (claude-sonnet-4.5, gpt-5, llama-4-scout, etc.)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            latency_ms: Latency in milliseconds
            confidence_score: Calculated confidence score (0.0-1.0)
            confidence_breakdown: Dict with model_confidence, completeness, consistency, validation
            action: 'use_ai_result' or 'fallback_to_rules'
            job_id: Optional job ID for tracking
            fallback_reason: Reason for fallback (if action is fallback_to_rules)
            request_data: Optional request data for debugging
            response_data: Optional response data for debugging
            error_message: Optional error message if call failed
            
        Returns:
            bool: True if logged successfully, False otherwise
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "job_id": job_id,
                "task": task,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": Decimal(str(cost)),
                "latency_ms": latency_ms,
                "confidence_score": Decimal(str(round(confidence_score, 2))),
                "confidence_breakdown": json.dumps(confidence_breakdown),
                "action": action,
                "fallback_reason": fallback_reason,
                "request_data": json.dumps(request_data) if request_data else None,
                "response_data": json.dumps(response_data) if response_data else None,
                "error_message": error_message
            }
            
            # Insert into database
            result = self.supabase.client.table("ai_call_logs").insert(log_entry).execute()
            
            if result.data:
                self.logger.info(
                    f"✅ AI call logged: {task} | {model} | "
                    f"confidence={confidence_score:.2f} | action={action} | "
                    f"cost=${cost:.4f} | latency={latency_ms}ms"
                )
                return True
            else:
                self.logger.error(f"❌ Failed to log AI call: No data returned")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log AI call: {e}")
            # Don't fail the main operation if logging fails
            return False
    
    async def log_claude_call(
        self,
        task: str,
        model: str,
        response: Any,
        latency_ms: int,
        confidence_score: float,
        confidence_breakdown: Dict[str, float],
        action: str,
        job_id: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a Claude API call.
        
        Args:
            task: Type of task
            model: Claude model (claude-haiku-4-5, claude-sonnet-4-5)
            response: Claude API response object
            latency_ms: Latency in milliseconds
            confidence_score: Calculated confidence score
            confidence_breakdown: Confidence breakdown dict
            action: 'use_ai_result' or 'fallback_to_rules'
            job_id: Optional job ID
            fallback_reason: Optional fallback reason
            request_data: Optional request data
            
        Returns:
            bool: True if logged successfully
        """
        try:
            # Extract token usage from Claude response
            input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else 0
            output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else 0
            
            # Calculate cost based on model
            cost = self._calculate_claude_cost(model, input_tokens, output_tokens)
            
            # Extract response text
            response_text = response.content[0].text if hasattr(response, 'content') else str(response)
            
            return await self.log_ai_call(
                task=task,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                confidence_score=confidence_score,
                confidence_breakdown=confidence_breakdown,
                action=action,
                job_id=job_id,
                fallback_reason=fallback_reason,
                request_data=request_data,
                response_data={"text": response_text[:500]}  # Truncate for storage
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log Claude call: {e}")
            return False
    
    async def log_gpt_call(
        self,
        task: str,
        model: str,
        response: Any,
        latency_ms: int,
        confidence_score: float,
        confidence_breakdown: Dict[str, float],
        action: str,
        job_id: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a GPT API call.
        
        Args:
            task: Type of task
            model: GPT model (gpt-5, gpt-4o)
            response: OpenAI API response object
            latency_ms: Latency in milliseconds
            confidence_score: Calculated confidence score
            confidence_breakdown: Confidence breakdown dict
            action: 'use_ai_result' or 'fallback_to_rules'
            job_id: Optional job ID
            fallback_reason: Optional fallback reason
            request_data: Optional request data
            
        Returns:
            bool: True if logged successfully
        """
        try:
            # Extract token usage from GPT response
            input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
            output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
            
            # Calculate cost based on model
            cost = self._calculate_gpt_cost(model, input_tokens, output_tokens)
            
            # Extract response text
            response_text = response.choices[0].message.content if hasattr(response, 'choices') else str(response)
            
            return await self.log_ai_call(
                task=task,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                confidence_score=confidence_score,
                confidence_breakdown=confidence_breakdown,
                action=action,
                job_id=job_id,
                fallback_reason=fallback_reason,
                request_data=request_data,
                response_data={"text": response_text[:500]}  # Truncate for storage
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log GPT call: {e}")
            return False
    
    async def log_llama_call(
        self,
        task: str,
        model: str,
        response: Dict[str, Any],
        latency_ms: int,
        confidence_score: float,
        confidence_breakdown: Dict[str, float],
        action: str,
        job_id: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a Llama (TogetherAI) API call.
        
        Args:
            task: Type of task
            model: Llama model (llama-4-scout-17b, etc.)
            response: TogetherAI API response dict
            latency_ms: Latency in milliseconds
            confidence_score: Calculated confidence score
            confidence_breakdown: Confidence breakdown dict
            action: 'use_ai_result' or 'fallback_to_rules'
            job_id: Optional job ID
            fallback_reason: Optional fallback reason
            request_data: Optional request data
            
        Returns:
            bool: True if logged successfully
        """
        try:
            # Extract token usage from Llama response
            usage = response.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            
            # Calculate cost based on model
            cost = self._calculate_llama_cost(model, input_tokens, output_tokens)
            
            # Extract response text
            choices = response.get('choices', [])
            response_text = choices[0].get('message', {}).get('content', '') if choices else ''
            
            return await self.log_ai_call(
                task=task,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                confidence_score=confidence_score,
                confidence_breakdown=confidence_breakdown,
                action=action,
                job_id=job_id,
                fallback_reason=fallback_reason,
                request_data=request_data,
                response_data={"text": response_text[:500]}  # Truncate for storage
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log Llama call: {e}")
            return False
    
    def _calculate_claude_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Claude API call using centralized pricing"""
        cost = ai_pricing.calculate_cost(model, input_tokens, output_tokens, provider="anthropic")
        return float(cost)

    def _calculate_gpt_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for GPT API call using centralized pricing"""
        cost = ai_pricing.calculate_cost(model, input_tokens, output_tokens, provider="openai")
        return float(cost)

    def _calculate_llama_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Llama (TogetherAI) API call using centralized pricing"""
        cost = ai_pricing.calculate_cost(model, input_tokens, output_tokens, provider="together")
        return float(cost)

