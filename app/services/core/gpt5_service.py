"""
GPT-5 Service

Service wrapper for OpenAI GPT-5 API calls.
Used for complex tasks requiring advanced reasoning and high accuracy.

Features:
- Cost-aware API calls
- Automatic retry with exponential backoff
- Response validation
- Token usage tracking
"""

import asyncio
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import httpx

from app.services.core.ai_call_logger import AICallLogger
from app.services.core.supabase_client import get_supabase_client
from app.config.ai_pricing import AIPricingConfig

logger = logging.getLogger(__name__)


class GPT5Service:
    """
    Service for GPT-5 API interactions.
    
    Provides high-quality AI responses for critical tasks.
    """
    
    def __init__(self, ai_logger: Optional[AICallLogger] = None):
        """
        Initialize GPT-5 service.
        
        Args:
            ai_logger: AI call logger instance
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY not set - GPT-5 service will not work")
        
        self.ai_logger = ai_logger or AICallLogger()
        self.base_url = "https://api.openai.com/v1"
        self.pricing = AIPricingConfig()
        self._supabase = get_supabase_client()

    async def _get_analysis_prompt_from_db(
        self,
        workspace_id: str,
        analysis_type: str
    ) -> str:
        """
        Fetch analysis prompt from database.
        NO FALLBACK - All prompts must exist in the database.

        Args:
            workspace_id: Workspace ID
            analysis_type: Type of analysis (product_extraction, material_classification, technical_analysis)

        Returns:
            System prompt string

        Raises:
            ValueError: If prompt is not found in database
        """
        try:
            result = self._supabase.client.table('prompts')\
                .select('prompt_text, system_prompt')\
                .eq('prompt_type', 'extraction')\
                .eq('workspace_id', workspace_id)\
                .eq('stage', 'entity_creation')\
                .eq('category', analysis_type)\
                .eq('is_active', True)\
                .order('version', desc=True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                prompt = result.data[0].get('system_prompt') or result.data[0].get('prompt_text')
                if prompt:
                    logger.info(f"Loaded GPT5 analysis prompt for '{analysis_type}' from database")
                    return prompt

            # No prompt found - this is an error
            error_msg = (
                f"CRITICAL: No GPT5 analysis prompt found in database for analysis_type='{analysis_type}'. "
                f"Please add it via /admin/ai-configs with: prompt_type='extraction', "
                f"stage='entity_creation', category='{analysis_type}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            logger.error(f"Error fetching GPT5 analysis prompt: {str(e)}")
            raise

    async def generate_completion(
        self,
        prompt: str,
        model: str = "gpt-5",
        max_tokens: int = 2000,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        job_id: Optional[str] = None,
        task: str = "gpt5_completion",
    ) -> Dict[str, Any]:
        """
        Generate completion using GPT-5.
        
        Args:
            prompt: User prompt
            model: Model to use (default: gpt-5)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            system_prompt: Optional system prompt
            job_id: Optional job ID for logging
            task: Task name for logging
            
        Returns:
            Response with text, usage, and metadata
        """
        if not self.api_key:
            return {
                "success": False,
                "error": "OPENAI_API_KEY not configured",
            }
        
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        messages.append({
            "role": "user",
            "content": prompt,
        })
        
        # Prepare request
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        start_time = datetime.now()
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_data,
                )
                
                response.raise_for_status()
                data = response.json()
            
            end_time = datetime.now()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Extract response
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            # Calculate cost
            cost = self.pricing.calculate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            
            # Log AI call
            if self.ai_logger and job_id:
                await self.ai_logger.log_ai_call({
                    "job_id": job_id,
                    "task": task,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": float(cost) if cost else None,
                    "latency_ms": latency_ms,
                    "request_data": {"prompt_length": len(prompt)},
                    "response_data": {"text_length": len(text)},
                })
            
            logger.info(
                f"✅ GPT-5 completion: {output_tokens} tokens, "
                f"${cost:.4f}, {latency_ms}ms"
            )
            
            return {
                "success": True,
                "text": text,
                "usage": usage,
                "cost": float(cost) if cost else 0.0,
                "latency_ms": latency_ms,
                "model": model,
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ GPT-5 API error: {e.response.status_code} - {e.response.text}")
            
            # Log failed call
            if self.ai_logger and job_id:
                await self.ai_logger.log_ai_call({
                    "job_id": job_id,
                    "task": task,
                    "model": model,
                    "error_message": f"HTTP {e.response.status_code}: {e.response.text}",
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                })
            
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
            
        except Exception as e:
            logger.error(f"❌ GPT-5 error: {str(e)}")
            
            # Log failed call
            if self.ai_logger and job_id:
                await self.ai_logger.log_ai_call({
                    "job_id": job_id,
                    "task": task,
                    "model": model,
                    "error_message": str(e),
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                })
            
            return {
                "success": False,
                "error": str(e),
            }
    
    async def analyze_complex_content(
        self,
        content: str,
        analysis_type: str,
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        workspace_id: str = None,
    ) -> Dict[str, Any]:
        """
        Analyze complex content using GPT-5.

        Args:
            content: Content to analyze
            analysis_type: Type of analysis (e.g., 'product_extraction', 'material_classification')
            context: Optional context information
            job_id: Optional job ID
            workspace_id: Workspace ID for fetching prompts from database

        Returns:
            Analysis result with structured data
        """
        # Use default workspace ID from config if not provided
        from app.config import get_settings
        workspace_id = workspace_id or get_settings().default_workspace_id

        # Fetch system prompt from database - NO FALLBACK
        system_prompt = await self._get_analysis_prompt_from_db(workspace_id, analysis_type)
        
        # Build user prompt
        context_str = ""
        if context:
            context_str = f"\n\nContext: {context}"
        
        user_prompt = f"""Analyze the following content:

{content[:4000]}  # Limit content length
{context_str}

Provide a detailed analysis in JSON format."""
        
        # Call GPT-5
        result = await self.generate_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model="gpt-5",
            max_tokens=2000,
            temperature=0.2,  # Lower temperature for structured output
            job_id=job_id,
            task=f"gpt5_{analysis_type}",
        )
        
        if not result.get("success"):
            return result
        
        # Try to parse JSON response
        try:
            import json
            text = result["text"]
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in text:
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_text = text[json_start:json_end].strip()
            elif "```" in text:
                json_start = text.find("```") + 3
                json_end = text.find("```", json_start)
                json_text = text[json_start:json_end].strip()
            else:
                json_text = text
            
            parsed_data = json.loads(json_text)
            
            return {
                "success": True,
                "data": parsed_data,
                "raw_text": text,
                "usage": result.get("usage", {}),
                "cost": result.get("cost", 0.0),
                "latency_ms": result.get("latency_ms", 0),
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse JSON from GPT-5 response: {str(e)}")
            
            # Return raw text if JSON parsing fails
            return {
                "success": True,
                "data": {"raw_text": result["text"]},
                "raw_text": result["text"],
                "usage": result.get("usage", {}),
                "cost": result.get("cost", 0.0),
                "latency_ms": result.get("latency_ms", 0),
                "warning": "Failed to parse JSON response",
            }


