"""
Credits Integration Service

Handles credit debit operations for AI usage in the PDF processing pipeline.
Integrates with Supabase database functions for atomic credit operations.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from decimal import Decimal
from app.services.core.supabase_client import get_supabase_client
from app.config.ai_pricing import AIPricingConfig

logger = logging.getLogger(__name__)


class CreditsIntegrationService:
    """Service for integrating credit debit into AI operations."""
    
    # Model pricing per 1M tokens (in USD)
    MODEL_PRICING = {
        'claude-sonnet-4-5-20250929': {'input': 3.00, 'output': 15.00},
        'claude-sonnet-4-20250514': {'input': 3.00, 'output': 15.00},
        'claude-haiku-4-5-20251001': {'input': 0.80, 'output': 4.00},
        'claude-3-5-haiku-20241022': {'input': 0.80, 'output': 4.00},
        'gpt-5': {'input': 5.00, 'output': 15.00},
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'Qwen/Qwen3-VL-8B-Instruct': {'input': 0.10, 'output': 0.30},
    }
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.logger = logger
    
    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """
        Calculate cost for AI operation based on token usage.
        
        Args:
            model_name: Name of the AI model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dict with input_cost_usd, output_cost_usd, total_cost_usd, credits_debited
        """
        # Get pricing for model (fallback to gpt-4o-mini if unknown)
        pricing = self.MODEL_PRICING.get(
            model_name,
            self.MODEL_PRICING['gpt-4o-mini']
        )
        
        # Calculate costs in USD
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost
        
        # Convert USD to credits (1 credit = $0.01)
        credits = total_cost * 100
        
        return {
            'input_cost_usd': round(input_cost, 6),
            'output_cost_usd': round(output_cost, 6),
            'total_cost_usd': round(total_cost, 6),
            'credits_debited': round(credits, 2)
        }
    
    async def debit_credits_for_ai_operation(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Debit credits for an AI operation and log usage.
        
        Args:
            user_id: User ID who initiated the operation
            workspace_id: Workspace ID (optional)
            operation_type: Type of operation (e.g., 'pdf_vision_discovery', 'agent_chat')
            model_name: AI model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Additional metadata to store
            
        Returns:
            Dict with success status, new balance, and transaction details
        """
        try:
            # Calculate costs
            costs = self.calculate_cost(model_name, input_tokens, output_tokens)
            
            # Debit credits using database function
            response = self.supabase.client.rpc(
                'debit_user_credits',
                {
                    'p_user_id': user_id,
                    'p_amount': costs['credits_debited'],
                    'p_operation_type': operation_type,
                    'p_description': f"{operation_type} using {model_name}",
                    'p_metadata': metadata or {}
                }
            ).execute()
            
            debit_result = response.data[0] if response.data else None
            
            if not debit_result or not debit_result.get('success'):
                error_msg = debit_result.get('error_message', 'Unknown error') if debit_result else 'No response from database'
                self.logger.error(f"❌ Credit debit failed for user {user_id}: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'credits_required': costs['credits_debited']
                }
            
            # Log AI usage
            usage_log = {
                'user_id': user_id,
                'workspace_id': workspace_id,
                'operation_type': operation_type,
                'model_name': model_name,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'input_cost_usd': costs['input_cost_usd'],
                'output_cost_usd': costs['output_cost_usd'],
                'total_cost_usd': costs['total_cost_usd'],
                'credits_debited': costs['credits_debited'],
                'metadata': metadata or {},
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.supabase.client.table('ai_usage_logs').insert(usage_log).execute()
            
            self.logger.info(
                f"✅ Debited {costs['credits_debited']:.2f} credits from user {user_id} "
                f"for {operation_type} ({model_name}). New balance: {debit_result['new_balance']:.2f}"
            )
            
            return {
                'success': True,
                'credits_debited': costs['credits_debited'],
                'new_balance': debit_result['new_balance'],
                'transaction_id': debit_result['transaction_id'],
                'costs': costs
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error debiting credits: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def debit_credits_for_firecrawl(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        credits_used: int,
        url: Optional[str] = None,
        pages_scraped: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Debit credits for a Firecrawl operation and log usage.

        Args:
            user_id: User ID who initiated the operation
            workspace_id: Workspace ID (optional)
            operation_type: Type of operation ('scrape', 'crawl', 'extract')
            credits_used: Number of Firecrawl credits consumed
            url: URL that was scraped (optional)
            pages_scraped: Number of pages scraped
            metadata: Additional metadata to store

        Returns:
            Dict with success status, new balance, and transaction details
        """
        try:
            # Calculate cost using Firecrawl pricing
            cost_usd = AIPricingConfig.calculate_firecrawl_cost(credits_used=credits_used)
            platform_credits = float(cost_usd * 100)

            # Debit credits using database function
            response = self.supabase.client.rpc(
                'debit_user_credits',
                {
                    'p_user_id': user_id,
                    'p_amount': platform_credits,
                    'p_operation_type': f"firecrawl_{operation_type}",
                    'p_description': f"Firecrawl {operation_type}: {url or 'N/A'}",
                    'p_metadata': {
                        **(metadata or {}),
                        'firecrawl_credits': credits_used,
                        'url': url,
                        'pages_scraped': pages_scraped
                    }
                }
            ).execute()

            debit_result = response.data[0] if response.data else None

            if not debit_result or not debit_result.get('success'):
                error_msg = debit_result.get('error_message', 'Unknown error') if debit_result else 'No response from database'
                self.logger.error(f"❌ Firecrawl credit debit failed for user {user_id}: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'credits_required': platform_credits
                }

            # Log Firecrawl usage
            usage_log = {
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
                'operation_details': {
                    'url': url,
                    'pages_scraped': pages_scraped
                },
                'metadata': metadata or {},
                'created_at': datetime.utcnow().isoformat()
            }

            self.supabase.client.table('ai_usage_logs').insert(usage_log).execute()

            self.logger.info(
                f"✅ Debited {platform_credits:.2f} platform credits ({credits_used} Firecrawl credits) "
                f"from user {user_id} for {operation_type}. New balance: {debit_result['new_balance']:.2f}"
            )

            return {
                'success': True,
                'credits_debited': platform_credits,
                'firecrawl_credits': credits_used,
                'new_balance': debit_result['new_balance'],
                'transaction_id': debit_result['transaction_id'],
                'cost_usd': float(cost_usd)
            }

        except Exception as e:
            self.logger.error(f"❌ Error debiting Firecrawl credits: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Singleton instance
_credits_service: Optional[CreditsIntegrationService] = None


def get_credits_service() -> CreditsIntegrationService:
    """Get singleton instance of CreditsIntegrationService."""
    global _credits_service
    if _credits_service is None:
        _credits_service = CreditsIntegrationService()
    return _credits_service


