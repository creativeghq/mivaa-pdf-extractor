"""
Search Prompt Service

Manages admin-configurable prompts for search result enhancement, formatting, filtering, and enrichment.
Allows admins to customize search behavior without code changes.
UPDATED: Now uses UnifiedPromptService for all prompt operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from app.services.utilities.unified_prompt_service import UnifiedPromptService

logger = logging.getLogger(__name__)


class SearchPromptService:
    """Service for managing and applying admin-configurable search prompts."""

    # Prompt types
    ENHANCEMENT = "enhancement"
    FORMATTING = "formatting"
    FILTERING = "filtering"
    ENRICHMENT = "enrichment"

    def __init__(self, supabase_client, llm_client=None):
        """
        Initialize the service.

        Args:
            supabase_client: Supabase client for database operations
            llm_client: Optional LLM client for prompt execution (OpenAI, Anthropic, etc.)
        """
        self.supabase = supabase_client
        self.llm_client = llm_client
        self.prompt_service = UnifiedPromptService()
        self._prompt_cache = {}
    
    async def get_active_prompts(
        self,
        workspace_id: str,
        prompt_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active prompts for a workspace.

        Args:
            workspace_id: Workspace ID
            prompt_type: Optional filter by prompt type (enhancement, formatting, filtering, enrichment)

        Returns:
            List of active prompts
        """
        try:
            return await self.prompt_service.get_search_prompts(
                workspace_id=workspace_id,
                prompt_subtype=prompt_type
            )
        except Exception as e:
            logger.error(f"Error getting active prompts: {e}", exc_info=True)
            return []
    
    async def enhance_query(
        self,
        query: str,
        workspace_id: str,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply enhancement prompt to search query.
        
        Args:
            query: Original search query
            workspace_id: Workspace ID
            custom_prompt: Optional custom enhancement prompt
            
        Returns:
            Dict with enhanced_query and metadata
        """
        try:
            # Get enhancement prompts
            prompts = await self.get_active_prompts(workspace_id, self.ENHANCEMENT)
            
            if not prompts and not custom_prompt:
                return {
                    'enhanced_query': query,
                    'original_query': query,
                    'prompts_applied': [],
                    'enhancement_applied': False
                }
            
            # Use custom prompt or first active prompt
            prompt_text = custom_prompt or prompts[0]['prompt_text']
            
            # Apply enhancement using LLM
            if self.llm_client:
                enhanced = await self._apply_llm_enhancement(query, prompt_text)
            else:
                # Fallback: simple keyword expansion
                enhanced = self._simple_enhancement(query, prompt_text)
            
            return {
                'enhanced_query': enhanced,
                'original_query': query,
                'prompts_applied': [prompts[0]['id']] if prompts else [],
                'enhancement_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}", exc_info=True)
            return {
                'enhanced_query': query,
                'original_query': query,
                'prompts_applied': [],
                'enhancement_applied': False,
                'error': str(e)
            }
    
    async def format_results(
        self,
        results: List[Dict[str, Any]],
        workspace_id: str,
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply formatting prompt to search results.
        
        Args:
            results: Search results
            workspace_id: Workspace ID
            custom_prompt: Optional custom formatting prompt
            
        Returns:
            Formatted results
        """
        try:
            # Get formatting prompts
            prompts = await self.get_active_prompts(workspace_id, self.FORMATTING)
            
            if not prompts and not custom_prompt:
                return results
            
            # Use custom prompt or first active prompt
            prompt_text = custom_prompt or prompts[0]['prompt_text']
            
            # Apply formatting using LLM
            if self.llm_client:
                formatted = await self._apply_llm_formatting(results, prompt_text)
            else:
                # Fallback: simple re-ranking
                formatted = self._simple_formatting(results, prompt_text)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}", exc_info=True)
            return results
    
    async def filter_results(
        self,
        results: List[Dict[str, Any]],
        workspace_id: str,
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply filtering prompt to search results.
        
        Args:
            results: Search results
            workspace_id: Workspace ID
            custom_prompt: Optional custom filtering prompt
            
        Returns:
            Filtered results
        """
        try:
            # Get filtering prompts
            prompts = await self.get_active_prompts(workspace_id, self.FILTERING)
            
            if not prompts and not custom_prompt:
                return results
            
            # Use custom prompt or first active prompt
            prompt_text = custom_prompt or prompts[0]['prompt_text']
            
            # Apply filtering using LLM
            if self.llm_client:
                filtered = await self._apply_llm_filtering(results, prompt_text)
            else:
                # Fallback: simple filtering
                filtered = self._simple_filtering(results, prompt_text)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering results: {e}", exc_info=True)
            return results
    
    async def enrich_results(
        self,
        results: List[Dict[str, Any]],
        workspace_id: str,
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply enrichment prompt to search results.
        
        Args:
            results: Search results
            workspace_id: Workspace ID
            custom_prompt: Optional custom enrichment prompt
            
        Returns:
            Enriched results
        """
        try:
            # Get enrichment prompts
            prompts = await self.get_active_prompts(workspace_id, self.ENRICHMENT)
            
            if not prompts and not custom_prompt:
                return results
            
            # Use custom prompt or first active prompt
            prompt_text = custom_prompt or prompts[0]['prompt_text']
            
            # Apply enrichment using LLM
            if self.llm_client:
                enriched = await self._apply_llm_enrichment(results, prompt_text)
            else:
                # Fallback: no enrichment
                enriched = results
            
            return enriched
            
        except Exception as e:
            logger.error(f"Error enriching results: {e}", exc_info=True)
            return results
    
    async def apply_all_prompts(
        self,
        query: str,
        results: List[Dict[str, Any]],
        workspace_id: str,
        use_search_prompts: bool = True
    ) -> Dict[str, Any]:
        """
        Apply all active prompts in sequence.
        
        Args:
            query: Original search query
            results: Search results
            workspace_id: Workspace ID
            use_search_prompts: Whether to use search prompts
            
        Returns:
            Dict with enhanced query, processed results, and metadata
        """
        if not use_search_prompts:
            return {
                'enhanced_query': query,
                'results': results,
                'prompts_applied': [],
                'processing_time_ms': 0
            }
        
        start_time = datetime.now()
        prompts_applied = []
        
        try:
            # 1. Enhance query
            enhancement_result = await self.enhance_query(query, workspace_id)
            enhanced_query = enhancement_result['enhanced_query']
            if enhancement_result.get('enhancement_applied'):
                prompts_applied.extend(enhancement_result.get('prompts_applied', []))
            
            # 2. Format results
            formatted_results = await self.format_results(results, workspace_id)
            
            # 3. Filter results
            filtered_results = await self.filter_results(formatted_results, workspace_id)
            
            # 4. Enrich results
            enriched_results = await self.enrich_results(filtered_results, workspace_id)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'enhanced_query': enhanced_query,
                'original_query': query,
                'results': enriched_results,
                'prompts_applied': prompts_applied,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error applying prompts: {e}", exc_info=True)
            return {
                'enhanced_query': query,
                'results': results,
                'prompts_applied': [],
                'processing_time_ms': 0,
                'error': str(e)
            }

    # Helper methods for LLM-based prompt execution

    async def _apply_llm_enhancement(self, query: str, prompt_text: str) -> str:
        """Apply enhancement using LLM."""
        # Placeholder for LLM integration
        # This would call OpenAI/Anthropic API with the prompt
        logger.info(f"LLM enhancement not yet implemented, using simple enhancement")
        return self._simple_enhancement(query, prompt_text)

    async def _apply_llm_formatting(
        self,
        results: List[Dict[str, Any]],
        prompt_text: str
    ) -> List[Dict[str, Any]]:
        """Apply formatting using LLM."""
        # Placeholder for LLM integration
        logger.info(f"LLM formatting not yet implemented, using simple formatting")
        return self._simple_formatting(results, prompt_text)

    async def _apply_llm_filtering(
        self,
        results: List[Dict[str, Any]],
        prompt_text: str
    ) -> List[Dict[str, Any]]:
        """Apply filtering using LLM."""
        # Placeholder for LLM integration
        logger.info(f"LLM filtering not yet implemented, using simple filtering")
        return self._simple_filtering(results, prompt_text)

    async def _apply_llm_enrichment(
        self,
        results: List[Dict[str, Any]],
        prompt_text: str
    ) -> List[Dict[str, Any]]:
        """Apply enrichment using LLM."""
        # Placeholder for LLM integration
        logger.info(f"LLM enrichment not yet implemented")
        return results

    # Simple fallback methods

    def _simple_enhancement(self, query: str, prompt_text: str) -> str:
        """Simple query enhancement without LLM."""
        # Parse prompt for keyword mappings
        # Example: "modern" → "contemporary design, minimalist aesthetic"
        try:
            # Simple keyword expansion based on prompt
            enhanced = query

            # Look for keyword mappings in prompt
            if "→" in prompt_text or "->" in prompt_text:
                lines = prompt_text.split('\n')
                for line in lines:
                    if "→" in line or "->" in line:
                        parts = line.split("→" if "→" in line else "->")
                        if len(parts) == 2:
                            keyword = parts[0].strip().strip('"').strip("'").strip('-').strip()
                            expansion = parts[1].strip()
                            if keyword.lower() in query.lower():
                                enhanced = f"{query} {expansion}"
                                break

            return enhanced

        except Exception as e:
            logger.error(f"Error in simple enhancement: {e}")
            return query

    def _simple_formatting(
        self,
        results: List[Dict[str, Any]],
        prompt_text: str
    ) -> List[Dict[str, Any]]:
        """Simple result formatting without LLM."""
        # Simple re-ranking based on prompt keywords
        try:
            # Look for ranking criteria in prompt
            if "availability" in prompt_text.lower():
                # Prioritize available products
                results.sort(
                    key=lambda x: (
                        x.get('metadata', {}).get('availability', '') == 'in_stock',
                        x.get('score', 0)
                    ),
                    reverse=True
                )
            elif "price" in prompt_text.lower():
                # Sort by price
                results.sort(
                    key=lambda x: float(x.get('metadata', {}).get('price', 999999))
                )

            return results

        except Exception as e:
            logger.error(f"Error in simple formatting: {e}")
            return results

    def _simple_filtering(
        self,
        results: List[Dict[str, Any]],
        prompt_text: str
    ) -> List[Dict[str, Any]]:
        """Simple result filtering without LLM."""
        # Simple filtering based on prompt keywords
        try:
            filtered = results

            # Filter out of stock if mentioned in prompt
            if "out of stock" in prompt_text.lower() or "out-of-stock" in prompt_text.lower():
                filtered = [
                    r for r in filtered
                    if r.get('metadata', {}).get('availability', '').lower() != 'out_of_stock'
                ]

            # Filter discontinued if mentioned in prompt
            if "discontinued" in prompt_text.lower():
                filtered = [
                    r for r in filtered
                    if not r.get('metadata', {}).get('discontinued', False)
                ]

            return filtered

        except Exception as e:
            logger.error(f"Error in simple filtering: {e}")
            return results

    def clear_cache(self):
        """Clear the prompt cache."""
        self._prompt_cache = {}


