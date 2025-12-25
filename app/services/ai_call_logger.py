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

from app.services.supabase_client import get_supabase_client
from app.config.ai_pricing import ai_pricing
from app.utils.retry_helper import async_retry_with_backoff
from app.utils.json_encoder import json_dumps

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
    
    def __init__(self):
        """Initialize AI Call Logger"""
        self.supabase = get_supabase_client()
        self.logger = logging.getLogger(__name__)
        # Import credits service here to avoid circular imports
        from app.services.credits_integration_service import get_credits_service
        self.credits_service = get_credits_service()
    @async_retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0, max_delay=10.0)
    
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
                "cost": float(cost),  # Convert to float for JSON serialization
                "latency_ms": latency_ms,
                "confidence_score": round(confidence_score, 2),  # Use float instead of Decimal
                "confidence_breakdown": json_dumps(confidence_breakdown),  # Use custom encoder
                "action": action,
                "fallback_reason": fallback_reason,
                "request_data": json_dumps(request_data) if request_data else None,  # Use custom encoder
                "response_data": json_dumps(response_data) if response_data else None,  # Use custom encoder
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
        request_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None
    ) -> bool:
        """
        Log a Claude API call and debit credits from user account.

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
            user_id: Optional user ID for credit debit
            workspace_id: Optional workspace ID for credit debit

        Returns:
            bool: True if logged successfully
        """
        try:
            # Extract token usage from Claude response
            input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else 0
            output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else 0

            # Calculate cost based on model
            cost = self._calculate_claude_cost(model, input_tokens, output_tokens)

            # Debit credits if user_id provided
            if user_id:
                await self.credits_service.debit_credits_for_ai_operation(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    operation_type=task,
                    model_name=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    metadata={'job_id': job_id} if job_id else None
                )

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
        request_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None
    ) -> bool:
        """
        Log a GPT API call and debit credits from user account.

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
            user_id: Optional user ID for credit debit
            workspace_id: Optional workspace ID for credit debit

        Returns:
            bool: True if logged successfully
        """
        try:
            # Extract token usage from GPT response
            input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
            output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0

            # Calculate cost based on model
            cost = self._calculate_gpt_cost(model, input_tokens, output_tokens)

            # Debit credits if user_id provided
            if user_id:
                await self.credits_service.debit_credits_for_ai_operation(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    operation_type=task,
                    model_name=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    metadata={'job_id': job_id} if job_id else None
                )

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

    async def log_firecrawl_call(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        credits_used: int,
        latency_ms: int,
        url: Optional[str] = None,
        pages_scraped: int = 1,
        success: bool = True,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Log a Firecrawl API call and debit credits.

        Args:
            user_id: User ID who initiated the operation
            workspace_id: Workspace ID (optional)
            operation_type: Type of operation ('scrape', 'crawl', 'extract')
            credits_used: Number of Firecrawl credits consumed
            latency_ms: Latency in milliseconds
            url: URL that was scraped (optional)
            pages_scraped: Number of pages scraped
            success: Whether the operation succeeded
            request_data: Optional request data for debugging
            response_data: Optional response data for debugging
            error_message: Optional error message if call failed

        Returns:
            bool: True if logged successfully
        """
        try:
            # Calculate cost using Firecrawl pricing
            cost_usd = ai_pricing.calculate_firecrawl_cost(credits_used=credits_used)
            platform_credits = int(cost_usd * 100)

            # Prepare operation details
            operation_details = {
                "url": url,
                "pages_scraped": pages_scraped,
                "success": success,
                "latency_ms": latency_ms
            }

            # Debit credits from user account
            if user_id:
                await self.credits_service.debit_credits_for_ai_operation(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    operation_type=f"firecrawl_{operation_type}",
                    model_name="firecrawl-scrape",
                    input_tokens=0,  # Firecrawl doesn't use tokens
                    output_tokens=0,
                    metadata={
                        'credits_used': credits_used,
                        'operation_details': operation_details,
                        'api_provider': 'firecrawl'
                    }
                )

            # Log to ai_usage_logs using the database function
            log_data = {
                'user_id': user_id,
                'workspace_id': workspace_id,
                'operation_type': operation_type,
                'model_name': 'firecrawl-scrape',
                'api_provider': 'firecrawl',
                'input_tokens': 0,
                'output_tokens': 0,
                'credits_used': credits_used,
                'input_cost_usd': 0,
                'output_cost_usd': 0,
                'total_cost_usd': float(cost_usd),
                'credits_debited': platform_credits,
                'operation_details': operation_details,
                'metadata': {
                    'request': request_data,
                    'response': response_data,
                    'error': error_message
                },
                'created_at': datetime.utcnow().isoformat()
            }

            self.supabase.client.table('ai_usage_logs').insert(log_data).execute()

            self.logger.info(
                f"✅ Logged Firecrawl {operation_type}: {credits_used} credits "
                f"(${cost_usd:.4f} = {platform_credits} platform credits) for user {user_id}"
            )

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to log Firecrawl call: {e}")
            return False

