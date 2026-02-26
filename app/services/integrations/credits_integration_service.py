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
    """Service for integrating credit debit into AI operations with 50% markup."""

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
        Calculate cost for AI operation based on token usage with 50% markup.

        Uses centralized AIPricingConfig for pricing and markup.

        Args:
            model_name: Name of the AI model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dict with input_cost_usd, output_cost_usd, raw_cost_usd, billed_cost_usd,
            markup_multiplier, and credits_debited
        """
        # Use centralized pricing config with markup
        cost_data = AIPricingConfig.calculate_cost(
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            include_markup=True
        )

        return {
            'input_cost_usd': round(float(cost_data['input_cost_usd']), 8),
            'output_cost_usd': round(float(cost_data['output_cost_usd']), 8),
            'raw_cost_usd': round(float(cost_data['raw_cost_usd']), 8),
            'billed_cost_usd': round(float(cost_data['billed_cost_usd']), 8),
            'markup_multiplier': float(cost_data['markup_multiplier']),
            'total_cost_usd': round(float(cost_data['billed_cost_usd']), 6),  # For backward compatibility
            'credits_debited': round(float(cost_data['credits_to_debit']), 2)
        }
    
    async def debit_credits_for_ai_operation(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Debit credits for an AI operation and log usage with 50% markup.

        Args:
            user_id: User ID who initiated the operation
            workspace_id: Workspace ID (optional)
            operation_type: Type of operation (e.g., 'pdf_vision_discovery', 'agent_chat')
            model_name: AI model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            job_id: Background job ID for cost aggregation (optional)
            metadata: Additional metadata to store

        Returns:
            Dict with success status, new balance, and transaction details
        """
        try:
            # Calculate costs with markup
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

            # Log AI usage with raw vs billed costs and job_id
            usage_log = {
                'user_id': user_id,
                'workspace_id': workspace_id,
                'operation_type': operation_type,
                'model_name': model_name,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'input_cost_usd': costs['input_cost_usd'],
                'output_cost_usd': costs['output_cost_usd'],
                'raw_cost_usd': costs['raw_cost_usd'],
                'markup_multiplier': costs['markup_multiplier'],
                'billed_cost_usd': costs['billed_cost_usd'],
                'total_cost_usd': costs['total_cost_usd'],
                'credits_debited': costs['credits_debited'],
                'job_id': job_id,
                'metadata': metadata or {},
                'created_at': datetime.utcnow().isoformat()
            }

            self.supabase.client.table('ai_usage_logs').insert(usage_log).execute()

            # Update job-level cost aggregation if job_id provided
            if job_id:
                try:
                    self.supabase.client.rpc(
                        'increment_job_cost',
                        {
                            'p_job_id': job_id,
                            'p_cost_usd': costs['billed_cost_usd'],
                            'p_credits': costs['credits_debited']
                        }
                    ).execute()
                except Exception as job_err:
                    self.logger.warning(f"⚠️ Failed to update job cost for {job_id}: {job_err}")

            self.logger.info(
                f"✅ Debited {costs['credits_debited']:.2f} credits from user {user_id} "
                f"for {operation_type} ({model_name}). "
                f"Raw: ${costs['raw_cost_usd']:.6f}, Billed: ${costs['billed_cost_usd']:.6f} (50% markup). "
                f"New balance: {debit_result['new_balance']:.2f}"
            )

            return {
                'success': True,
                'credits_debited': costs['credits_debited'],
                'raw_cost_usd': costs['raw_cost_usd'],
                'billed_cost_usd': costs['billed_cost_usd'],
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

    async def debit_credits_for_time_based_ai(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        model_name: str,
        inference_seconds: float,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Debit credits for time-based AI operations (HuggingFace Inference Endpoints).

        HuggingFace endpoints are billed by GPU compute time, not tokens.

        Args:
            user_id: User ID who initiated the operation
            workspace_id: Workspace ID (optional)
            operation_type: Type of operation (e.g., 'vision_analysis')
            model_name: AI model used (e.g., 'qwen3-vl-32b')
            inference_seconds: Time taken for inference in seconds
            job_id: Background job ID for cost aggregation (optional)
            metadata: Additional metadata to store

        Returns:
            Dict with success status, new balance, and transaction details
        """
        try:
            # Check if model is time-based
            if not AIPricingConfig.is_time_based_model(model_name):
                self.logger.warning(f"⚠️ Model {model_name} is not time-based, using token-based fallback")
                # Fall back to token-based with estimates
                return await self.debit_credits_for_ai_operation(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    operation_type=operation_type,
                    model_name=model_name,
                    input_tokens=0,
                    output_tokens=0,
                    job_id=job_id,
                    metadata=metadata
                )

            # Calculate time-based costs with markup
            costs = AIPricingConfig.calculate_time_based_cost(
                model=model_name,
                inference_seconds=inference_seconds,
                include_markup=True
            )

            credits_to_debit = round(float(costs['credits_to_debit']), 2)

            # Debit credits using database function
            response = self.supabase.client.rpc(
                'debit_user_credits',
                {
                    'p_user_id': user_id,
                    'p_amount': credits_to_debit,
                    'p_operation_type': operation_type,
                    'p_description': f"{operation_type} using {model_name} ({inference_seconds:.2f}s)",
                    'p_metadata': metadata or {}
                }
            ).execute()

            debit_result = response.data[0] if response.data else None

            if not debit_result or not debit_result.get('success'):
                error_msg = debit_result.get('error_message', 'Unknown error') if debit_result else 'No response from database'
                self.logger.error(f"❌ Time-based credit debit failed for user {user_id}: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'credits_required': credits_to_debit
                }

            # Log AI usage with time-based metrics
            usage_log = {
                'user_id': user_id,
                'workspace_id': workspace_id,
                'operation_type': operation_type,
                'model_name': model_name,
                'input_tokens': 0,  # N/A for time-based
                'output_tokens': 0,  # N/A for time-based
                'input_cost_usd': 0,
                'output_cost_usd': 0,
                'raw_cost_usd': round(float(costs['raw_cost_usd']), 8),
                'markup_multiplier': float(costs['markup_multiplier']),
                'billed_cost_usd': round(float(costs['billed_cost_usd']), 8),
                'total_cost_usd': round(float(costs['billed_cost_usd']), 6),
                'credits_debited': credits_to_debit,
                'job_id': job_id,
                'metadata': {
                    **(metadata or {}),
                    'billing_type': 'time_based',
                    'inference_seconds': inference_seconds,
                    'hourly_rate_usd': float(costs['hourly_rate_usd'])
                },
                'created_at': datetime.utcnow().isoformat()
            }

            self.supabase.client.table('ai_usage_logs').insert(usage_log).execute()

            # Update job-level cost aggregation if job_id provided
            if job_id:
                try:
                    self.supabase.client.rpc(
                        'increment_job_cost',
                        {
                            'p_job_id': job_id,
                            'p_cost_usd': round(float(costs['billed_cost_usd']), 6),
                            'p_credits': credits_to_debit
                        }
                    ).execute()
                except Exception as job_err:
                    self.logger.warning(f"⚠️ Failed to update job cost for {job_id}: {job_err}")

            self.logger.info(
                f"✅ Debited {credits_to_debit:.2f} credits from user {user_id} "
                f"for {operation_type} ({model_name}, {inference_seconds:.2f}s). "
                f"Raw: ${float(costs['raw_cost_usd']):.6f}, Billed: ${float(costs['billed_cost_usd']):.6f} (50% markup). "
                f"New balance: {debit_result['new_balance']:.2f}"
            )

            return {
                'success': True,
                'credits_debited': credits_to_debit,
                'raw_cost_usd': float(costs['raw_cost_usd']),
                'billed_cost_usd': float(costs['billed_cost_usd']),
                'new_balance': debit_result['new_balance'],
                'transaction_id': debit_result['transaction_id'],
                'inference_seconds': inference_seconds,
                'hourly_rate_usd': float(costs['hourly_rate_usd'])
            }

        except Exception as e:
            self.logger.error(f"❌ Error debiting time-based credits: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def debit_credits_for_external_service(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        service_name: str,
        units: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Debit credits for external (non-AI) service operations.

        Generic method for per-unit services: Twilio, Apollo,
        Hunter.io, ZeroBounce, and Firecrawl (edge function variant).

        Args:
            user_id: User ID who initiated the operation
            workspace_id: Workspace ID (optional)
            operation_type: Type of operation (e.g., 'b2b_manufacturer_search')
            service_name: Service pricing key (e.g., 'twilio-sms', 'firecrawl-scrape')
            units: Number of operations performed (default: 1)
            metadata: Additional metadata to store

        Returns:
            Dict with success status, new balance, and transaction details
        """
        try:
            costs = AIPricingConfig.calculate_external_service_cost(
                service_name=service_name,
                units=units,
                include_markup=True
            )

            credits_to_debit = round(float(costs['credits_to_debit']), 2)

            if credits_to_debit <= 0:
                return {'success': True, 'credits_debited': 0, 'raw_cost_usd': 0, 'billed_cost_usd': 0}

            # Debit credits using database function
            response = self.supabase.client.rpc(
                'debit_user_credits',
                {
                    'p_user_id': user_id,
                    'p_amount': credits_to_debit,
                    'p_operation_type': operation_type,
                    'p_description': f"{service_name} {operation_type} ({units} {costs['unit_type']}{'s' if units != 1 else ''})",
                    'p_metadata': {
                        **(metadata or {}),
                        'service': service_name,
                        'units': units,
                        'unit_type': str(costs['unit_type']),
                    }
                }
            ).execute()

            debit_result = response.data[0] if response.data else None

            if not debit_result or not debit_result.get('success'):
                error_msg = debit_result.get('error_message', 'Unknown error') if debit_result else 'No response from database'
                self.logger.error(f"❌ External service credit debit failed for user {user_id}: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'credits_required': credits_to_debit
                }

            # Log usage to ai_usage_logs
            usage_log = {
                'user_id': user_id,
                'workspace_id': workspace_id,
                'operation_type': operation_type,
                'model_name': service_name,
                'api_provider': service_name.split('-')[0],  # twilio, apollo, hunter, zerobounce, firecrawl
                'input_tokens': 0,
                'output_tokens': 0,
                'input_cost_usd': 0,
                'output_cost_usd': 0,
                'raw_cost_usd': round(float(costs['raw_cost_usd']), 8),
                'markup_multiplier': float(costs['markup_multiplier']),
                'billed_cost_usd': round(float(costs['billed_cost_usd']), 8),
                'total_cost_usd': round(float(costs['billed_cost_usd']), 6),
                'credits_debited': credits_to_debit,
                'metadata': {
                    **(metadata or {}),
                    'billing_type': 'per_unit',
                    'service': service_name,
                    'units': units,
                    'unit_type': str(costs['unit_type']),
                    'cost_per_unit': float(costs['cost_per_unit']),
                },
                'created_at': datetime.utcnow().isoformat()
            }

            self.supabase.client.table('ai_usage_logs').insert(usage_log).execute()

            self.logger.info(
                f"✅ Debited {credits_to_debit:.2f} credits from user {user_id} "
                f"for {service_name} ({units} {costs['unit_type']}{'s' if units != 1 else ''}). "
                f"Raw: ${float(costs['raw_cost_usd']):.6f}, Billed: ${float(costs['billed_cost_usd']):.6f} (50% markup). "
                f"New balance: {debit_result['new_balance']:.2f}"
            )

            return {
                'success': True,
                'credits_debited': credits_to_debit,
                'raw_cost_usd': float(costs['raw_cost_usd']),
                'billed_cost_usd': float(costs['billed_cost_usd']),
                'new_balance': debit_result['new_balance'],
                'transaction_id': debit_result['transaction_id'],
            }

        except Exception as e:
            self.logger.error(f"❌ Error debiting external service credits ({service_name}): {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def debit_credits_for_replicate(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        model_name: str,
        num_generations: int = 1,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Debit credits for Replicate image generation operations.

        Replicate models are billed per generation (image created).

        Args:
            user_id: User ID who initiated the operation
            workspace_id: Workspace ID (optional)
            operation_type: Type of operation (e.g., 'image_generation', '3d_generation')
            model_name: Replicate model used (e.g., 'flux-dev', 'comfyui-interior-remodel')
            num_generations: Number of images generated
            job_id: Background job ID for cost aggregation (optional)
            metadata: Additional metadata to store

        Returns:
            Dict with success status, new balance, and transaction details
        """
        try:
            # Check if model is per-generation
            if not AIPricingConfig.is_per_generation_model(model_name):
                self.logger.warning(f"⚠️ Model {model_name} is not per-generation, using default pricing")
                # Fall back to default cost
                raw_cost = Decimal("0.01") * num_generations
                billed_cost = raw_cost * AIPricingConfig.MARKUP_MULTIPLIER
                credits_to_debit = float(billed_cost * Decimal("100"))
            else:
                # Calculate per-generation costs with markup
                costs = AIPricingConfig.calculate_replicate_cost(
                    model=model_name,
                    num_generations=num_generations,
                    include_markup=True
                )
                credits_to_debit = round(float(costs['credits_to_debit']), 2)
                raw_cost = costs['raw_cost_usd']
                billed_cost = costs['billed_cost_usd']

            # Debit credits using database function
            response = self.supabase.client.rpc(
                'debit_user_credits',
                {
                    'p_user_id': user_id,
                    'p_amount': credits_to_debit,
                    'p_operation_type': operation_type,
                    'p_description': f"{operation_type} using {model_name} ({num_generations} images)",
                    'p_metadata': metadata or {}
                }
            ).execute()

            debit_result = response.data[0] if response.data else None

            if not debit_result or not debit_result.get('success'):
                error_msg = debit_result.get('error_message', 'Unknown error') if debit_result else 'No response from database'
                self.logger.error(f"❌ Replicate credit debit failed for user {user_id}: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'credits_required': credits_to_debit
                }

            # Log AI usage
            usage_log = {
                'user_id': user_id,
                'workspace_id': workspace_id,
                'operation_type': operation_type,
                'model_name': model_name,
                'input_tokens': 0,
                'output_tokens': 0,
                'input_cost_usd': 0,
                'output_cost_usd': 0,
                'raw_cost_usd': round(float(raw_cost), 8),
                'markup_multiplier': float(AIPricingConfig.MARKUP_MULTIPLIER),
                'billed_cost_usd': round(float(billed_cost), 8),
                'total_cost_usd': round(float(billed_cost), 6),
                'credits_debited': credits_to_debit,
                'job_id': job_id,
                'metadata': {
                    **(metadata or {}),
                    'billing_type': 'per_generation',
                    'num_generations': num_generations
                },
                'created_at': datetime.utcnow().isoformat()
            }

            self.supabase.client.table('ai_usage_logs').insert(usage_log).execute()

            # Update job-level cost aggregation if job_id provided
            if job_id:
                try:
                    self.supabase.client.rpc(
                        'increment_job_cost',
                        {
                            'p_job_id': job_id,
                            'p_cost_usd': round(float(billed_cost), 6),
                            'p_credits': credits_to_debit
                        }
                    ).execute()
                except Exception as job_err:
                    self.logger.warning(f"⚠️ Failed to update job cost for {job_id}: {job_err}")

            self.logger.info(
                f"✅ Debited {credits_to_debit:.2f} credits from user {user_id} "
                f"for {operation_type} ({model_name}, {num_generations} images). "
                f"Raw: ${float(raw_cost):.6f}, Billed: ${float(billed_cost):.6f} (50% markup). "
                f"New balance: {debit_result['new_balance']:.2f}"
            )

            return {
                'success': True,
                'credits_debited': credits_to_debit,
                'raw_cost_usd': float(raw_cost),
                'billed_cost_usd': float(billed_cost),
                'new_balance': debit_result['new_balance'],
                'transaction_id': debit_result['transaction_id'],
                'num_generations': num_generations
            }

        except Exception as e:
            self.logger.error(f"❌ Error debiting Replicate credits: {e}")
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


