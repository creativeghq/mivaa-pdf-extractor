"""
Credits Integration Service

Handles credit debit operations for AI usage in the PDF processing pipeline.

EVERY debit goes through ONE RPC — `debit_and_log_ai_usage` (audit mivaa#17 M4-2)
-------------------------------------------------------------------------------
This file used to contain five hand-copied versions of the same two steps:

    rpc('debit_credits', ...)                     # take the money
    table('ai_usage_logs').insert({...})          # write down what it was for

which is two calls that can half-succeed, in the billing layer, five times over. Both
halves failed in production:

  * debit ok / insert raises -> the outer `except` returned {'success': False} while the
    credits were already gone, and nothing anywhere recorded that they went.
  * debit returns success=false -> `logger.error(...)` and nothing else. A failed billing
    event that lives only in process logs cannot be reconciled afterwards, which is what
    made both M4-1 here and the `if user_id:` skip in #16 unmeasurable rather than merely
    wrong.

The insert half was not a hypothetical risk. Three of the five payloads named columns that
do not exist on `ai_usage_logs` (`api_provider`, `credits_used`, `operation_details`), so
PostgREST rejected every one of them with PGRST204. `debit_credits_for_external_service`
has therefore never written a usage row in its life: the eight per-unit rows in that table
all came from an edge function. Money moved; the record did not.

That is why the RPC takes NAMED PARAMETERS. A hand-built dict posted through PostgREST
fails on an unknown key at runtime, silently, in the billing path. A named argument fails
at deploy time, in CI.

A zero debit is no longer a silent success (M4-1). `units <= 0`, an amount that rounds
away, or a call with no payer each produce a row carrying `unbilled_reason`, so "we did
paid work and charged nothing" is something you can SUM rather than grep for.
"""

import logging
from typing import Dict, Any, Optional
from decimal import Decimal
from app.services.core.supabase_client import get_supabase_client
from app.config.ai_pricing import AIPricingConfig

logger = logging.getLogger(__name__)

#: Credits are stored as numeric(10,2) in both wallets, so two decimals is the representable
#: precision — not a rounding policy. (Until 2026-08-16 the personal wallet was BIGINT and
#: rounded every fractional debit, which is the root of M4-1; see the migration
#: `credits_personal_wallet_holds_fractions`.) What changed here is that an amount which
#: rounds away is now RECORDED as unbilled instead of returning success.
CREDIT_QUANTUM = Decimal("0.01")


def _q(value: Any) -> float:
    """Quantize to the wallet's precision. See CREDIT_QUANTUM."""
    return float(round(float(value), 2))


class CreditsIntegrationService:
    """Service for integrating credit debit into AI operations with 50% markup."""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.logger = logger

    # ── The one debit path ───────────────────────────────────────────────────

    async def _debit_and_log(
        self,
        *,
        user_id: Optional[str],
        workspace_id: Optional[str],
        operation_type: str,
        model_name: str,
        credits: float,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        input_cost_usd: float = 0.0,
        output_cost_usd: float = 0.0,
        raw_cost_usd: Optional[float] = None,
        billed_cost_usd: Optional[float] = None,
        markup_multiplier: Optional[float] = None,
        job_id: Optional[str] = None,
        module_slug: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
        unbilled_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Take the credits and record the usage in ONE transaction.

        Returns the RPC's verdict plus `credits_required`, so callers keep the shape they
        had. `success` is False whenever the credits did not move — including the
        zero-amount and no-payer cases, which previously reported success.
        """
        try:
            resp = self.supabase.client.rpc('debit_and_log_ai_usage', {
                'p_user_id': str(user_id) if user_id else None,
                'p_workspace_id': str(workspace_id) if workspace_id else None,
                'p_operation_type': operation_type,
                'p_model_name': model_name,
                'p_credits': credits,
                'p_input_tokens': int(input_tokens or 0),
                'p_output_tokens': int(output_tokens or 0),
                'p_input_cost_usd': float(input_cost_usd or 0),
                'p_output_cost_usd': float(output_cost_usd or 0),
                'p_raw_cost_usd': float(raw_cost_usd) if raw_cost_usd is not None else None,
                'p_billed_cost_usd': float(billed_cost_usd) if billed_cost_usd is not None else None,
                'p_markup_multiplier': (
                    float(markup_multiplier) if markup_multiplier is not None else None
                ),
                'p_description': description,
                'p_metadata': metadata or {},
                'p_job_id': job_id,
                'p_module_slug': module_slug,
                'p_product_id': product_id,
                'p_image_id': image_id,
                'p_unbilled_reason': unbilled_reason,
            }).execute()
            result = resp.data if isinstance(resp.data, dict) else {}
        except Exception as e:  # noqa: BLE001
            # The RPC itself is unreachable. Nothing was debited (it is one transaction), so
            # this is a clean refusal rather than money in limbo.
            self.logger.error(
                "❌ debit_and_log_ai_usage raised for user=%s op=%s model=%s: %s",
                user_id, operation_type, model_name, e,
            )
            return {'success': False, 'error': str(e), 'credits_required': credits}

        if not result.get('success'):
            reason = result.get('unbilled_reason') or 'unknown'
            # Severity is deliberately split. `below_quantum` is the COMMON case: 7,701 of
            # the 8,567 usage rows this platform has ever written charged zero against a
            # real cost, because a per-call AI charge is routinely smaller than the 0.01 a
            # wallet can hold. Sentry is wired at event_level=ERROR (main.py), so logging
            # that at ERROR would raise an event per AI call and bury the cases where money
            # was actually owed and not taken. The aggregatable record is the
            # `unbilled_reason` COLUMN; see ai_call_logger._record_missing_principal, which
            # makes the same argument for the same reason.
            log = self.logger.warning if reason == 'below_quantum' else self.logger.error
            log(
                "⚠️ UNBILLED (%s) user=%s op=%s model=%s credits=%.4f: %s",
                reason, user_id, operation_type, model_name, credits,
                result.get('error') or '',
            )
            result.setdefault('error', f"unbilled: {reason}")
            result.setdefault('credits_required', credits)
            return result

        # Job-level cost aggregation stays a separate, best-effort call: it is a
        # denormalized rollup, not part of the money moving, and a failure here must not
        # roll back a debit that already happened.
        if job_id and billed_cost_usd is not None:
            try:
                self.supabase.client.rpc('increment_job_cost', {
                    'p_job_id': job_id,
                    'p_cost_usd': round(float(billed_cost_usd), 6),
                    'p_credits': credits,
                }).execute()
            except Exception as job_err:  # noqa: BLE001
                self.logger.warning(f"⚠️ Failed to update job cost for {job_id}: {job_err}")

        self.logger.info(
            "✅ Debited %.2f credits from user %s for %s (%s). New balance: %s",
            credits, user_id, operation_type, model_name, result.get('new_balance'),
        )
        return result

    # ── Cost calculation ─────────────────────────────────────────────────────

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
            'credits_debited': _q(cost_data['credits_to_debit']),
        }

    # ── The five public debit surfaces ───────────────────────────────────────

    async def debit_credits_for_ai_operation(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        module_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        """Debit credits for a token-billed AI operation and log usage with 50% markup."""
        costs = self.calculate_cost(model_name, input_tokens, output_tokens)
        credits = costs['credits_debited']

        result = await self._debit_and_log(
            user_id=user_id,
            workspace_id=workspace_id,
            operation_type=operation_type,
            model_name=model_name,
            credits=credits,
            description=f"{operation_type} using {model_name}",
            metadata=metadata or {},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=costs['input_cost_usd'],
            output_cost_usd=costs['output_cost_usd'],
            raw_cost_usd=costs['raw_cost_usd'],
            billed_cost_usd=costs['billed_cost_usd'],
            markup_multiplier=costs['markup_multiplier'],
            job_id=job_id,
            module_slug=module_slug,
        )
        if not result.get('success'):
            return result

        return {
            'success': True,
            'credits_debited': credits,
            'raw_cost_usd': costs['raw_cost_usd'],
            'billed_cost_usd': costs['billed_cost_usd'],
            'new_balance': result.get('new_balance'),
            'transaction_id': result.get('transaction_id'),
            'costs': costs,
        }

    async def debit_credits_for_firecrawl(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        credits_used: int,
        url: Optional[str] = None,
        pages_scraped: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        module_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        """Debit credits for a Firecrawl operation and log usage.

        `credits_used` is FIRECRAWL's own unit; the platform charge is derived from it.
        """
        if credits_used is None or credits_used <= 0:
            return await self._unbillable(
                user_id, workspace_id, operation_type, 'firecrawl-scrape',
                reason='invalid_units',
                detail=f"credits_used={credits_used}",
                metadata={**(metadata or {}), 'firecrawl_credits': credits_used, 'url': url},
            )

        cost_usd = AIPricingConfig.calculate_firecrawl_cost(credits_used=credits_used)
        platform_credits = _q(float(cost_usd) * 100)

        result = await self._debit_and_log(
            user_id=user_id,
            workspace_id=workspace_id,
            operation_type=operation_type,
            model_name='firecrawl-scrape',
            credits=platform_credits,
            description=f"Firecrawl {operation_type}: {url or 'N/A'}",
            metadata={
                **(metadata or {}),
                # `api_provider`, `credits_used` and `operation_details` used to be sent as
                # COLUMNS. None of the three exists on ai_usage_logs, which is why this
                # method's usage row never landed. They live in metadata, where they are
                # queryable and cannot silently reject the whole insert.
                'billing_type': 'per_unit',
                'api_provider': 'firecrawl',
                'firecrawl_credits': credits_used,
                'url': url,
                'pages_scraped': pages_scraped,
            },
            billed_cost_usd=float(cost_usd),
            module_slug=module_slug,
        )
        if not result.get('success'):
            return result

        return {
            'success': True,
            'credits_debited': platform_credits,
            'firecrawl_credits': credits_used,
            'new_balance': result.get('new_balance'),
            'transaction_id': result.get('transaction_id'),
            'cost_usd': float(cost_usd),
        }

    async def debit_credits_for_time_based_ai(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        model_name: str,
        inference_seconds: float,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        module_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        """Debit credits for time-based AI operations (HuggingFace Inference Endpoints).

        HuggingFace endpoints are billed by GPU compute time, not tokens.
        """
        if not AIPricingConfig.is_time_based_model(model_name):
            self.logger.warning(
                f"⚠️ Model {model_name} is not time-based, using token-based fallback"
            )
            return await self.debit_credits_for_ai_operation(
                user_id=user_id,
                workspace_id=workspace_id,
                operation_type=operation_type,
                model_name=model_name,
                input_tokens=0,
                output_tokens=0,
                job_id=job_id,
                metadata=metadata,
                module_slug=module_slug,
            )

        if inference_seconds is None or inference_seconds <= 0:
            return await self._unbillable(
                user_id, workspace_id, operation_type, model_name,
                reason='invalid_units',
                detail=f"inference_seconds={inference_seconds}",
                metadata={**(metadata or {}), 'billing_type': 'time_based'},
                job_id=job_id,
            )

        costs = AIPricingConfig.calculate_time_based_cost(
            model=model_name,
            inference_seconds=inference_seconds,
            include_markup=True
        )
        credits = _q(costs['credits_to_debit'])

        result = await self._debit_and_log(
            user_id=user_id,
            workspace_id=workspace_id,
            operation_type=operation_type,
            model_name=model_name,
            credits=credits,
            description=f"{operation_type} using {model_name} ({inference_seconds:.2f}s)",
            metadata={
                **(metadata or {}),
                'billing_type': 'time_based',
                'inference_seconds': inference_seconds,
                'hourly_rate_usd': float(costs['hourly_rate_usd']),
            },
            raw_cost_usd=round(float(costs['raw_cost_usd']), 8),
            billed_cost_usd=round(float(costs['billed_cost_usd']), 8),
            markup_multiplier=float(costs['markup_multiplier']),
            job_id=job_id,
            module_slug=module_slug,
        )
        if not result.get('success'):
            return result

        return {
            'success': True,
            'credits_debited': credits,
            'raw_cost_usd': float(costs['raw_cost_usd']),
            'billed_cost_usd': float(costs['billed_cost_usd']),
            'new_balance': result.get('new_balance'),
            'transaction_id': result.get('transaction_id'),
            'inference_seconds': inference_seconds,
            'hourly_rate_usd': float(costs['hourly_rate_usd']),
        }

    async def debit_credits_for_external_service(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        service_name: str,
        units: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        module_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Debit credits for external (non-AI) per-unit service operations.

        Twilio, Apollo, Hunter.io, ZeroBounce, Firecrawl (edge variant), and the metered
        MIVAA image endpoints.

        `calculate_external_service_cost` RAISES for a service it does not know — the
        finding's "a missing pricing row returns 0" reading is wrong, and that path already
        surfaced as an error. What did silently succeed with a zero debit was `units <= 0`
        and any price that rounded away; both are now recorded as unbilled.
        """
        if units is None or units <= 0:
            return await self._unbillable(
                user_id, workspace_id, operation_type, service_name,
                reason='invalid_units',
                detail=f"units={units}",
                metadata={**(metadata or {}), 'billing_type': 'per_unit',
                          'service': service_name, 'units': units},
            )

        try:
            costs = AIPricingConfig.calculate_external_service_cost(
                service_name=service_name,
                units=units,
                include_markup=True
            )
        except ValueError as e:
            # An unpriced service is "no verified cost, never guess". Refuse, and say so.
            self.logger.error(f"❌ No pricing for external service {service_name}: {e}")
            return {'success': False, 'error': str(e)}

        credits = _q(costs['credits_to_debit'])

        result = await self._debit_and_log(
            user_id=user_id,
            workspace_id=workspace_id,
            operation_type=operation_type,
            model_name=service_name,
            credits=credits,
            description=(
                f"{service_name} {operation_type} "
                f"({units} {costs['unit_type']}{'s' if units != 1 else ''})"
            ),
            metadata={
                **(metadata or {}),
                'billing_type': 'per_unit',
                # `api_provider` was a phantom COLUMN here too — see debit_credits_for_firecrawl.
                'api_provider': str(service_name).split('-')[0],
                'service': service_name,
                'units': units,
                'unit_type': str(costs['unit_type']),
                'cost_per_unit': float(costs['cost_per_unit']),
            },
            raw_cost_usd=round(float(costs['raw_cost_usd']), 8),
            billed_cost_usd=round(float(costs['billed_cost_usd']), 8),
            markup_multiplier=float(costs['markup_multiplier']),
            module_slug=module_slug,
        )
        if not result.get('success'):
            return result

        return {
            'success': True,
            'credits_debited': credits,
            'raw_cost_usd': float(costs['raw_cost_usd']),
            'billed_cost_usd': float(costs['billed_cost_usd']),
            'new_balance': result.get('new_balance'),
            'transaction_id': result.get('transaction_id'),
        }

    async def debit_credits_for_replicate(
        self,
        user_id: str,
        workspace_id: Optional[str],
        operation_type: str,
        model_name: str,
        num_generations: int = 1,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        module_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        """Debit credits for Replicate image generation (billed per generation)."""
        if num_generations is None or num_generations <= 0:
            return await self._unbillable(
                user_id, workspace_id, operation_type, model_name,
                reason='invalid_units',
                detail=f"num_generations={num_generations}",
                metadata={**(metadata or {}), 'billing_type': 'per_generation'},
                job_id=job_id,
            )

        if not AIPricingConfig.is_per_generation_model(model_name):
            self.logger.warning(
                f"⚠️ Model {model_name} is not per-generation, using default pricing"
            )
            raw_cost = Decimal("0.01") * num_generations
            billed_cost = raw_cost * AIPricingConfig.MARKUP_MULTIPLIER
            credits = _q(billed_cost * Decimal("100"))
        else:
            costs = AIPricingConfig.calculate_replicate_cost(
                model=model_name,
                num_generations=num_generations,
                include_markup=True
            )
            credits = _q(costs['credits_to_debit'])
            raw_cost = costs['raw_cost_usd']
            billed_cost = costs['billed_cost_usd']

        result = await self._debit_and_log(
            user_id=user_id,
            workspace_id=workspace_id,
            operation_type=operation_type,
            model_name=model_name,
            credits=credits,
            description=f"{operation_type} using {model_name} ({num_generations} images)",
            metadata={
                **(metadata or {}),
                'billing_type': 'per_generation',
                'num_generations': num_generations,
            },
            raw_cost_usd=round(float(raw_cost), 8),
            billed_cost_usd=round(float(billed_cost), 8),
            markup_multiplier=float(AIPricingConfig.MARKUP_MULTIPLIER),
            job_id=job_id,
            module_slug=module_slug,
        )
        if not result.get('success'):
            return result

        return {
            'success': True,
            'credits_debited': credits,
            'raw_cost_usd': float(raw_cost),
            'billed_cost_usd': float(billed_cost),
            'new_balance': result.get('new_balance'),
            'transaction_id': result.get('transaction_id'),
            'num_generations': num_generations,
        }

    # ── Unbillable work, recorded rather than waved through ──────────────────

    async def _unbillable(
        self,
        user_id: Optional[str],
        workspace_id: Optional[str],
        operation_type: str,
        model_name: str,
        *,
        reason: str,
        detail: str,
        metadata: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """A caller asked us to bill for nothing. Write it down and refuse.

        `units=0` reaching a paid debit is a caller bug, and the old code answered it with
        {'success': True, 'credits_debited': 0} — which is indistinguishable from a
        successful charge to everything downstream (M4-1).
        """
        self.logger.error(
            "❌ refusing to bill %s/%s: %s (%s)", operation_type, model_name, reason, detail
        )
        await self._debit_and_log(
            user_id=user_id,
            workspace_id=workspace_id,
            operation_type=operation_type,
            model_name=model_name,
            credits=0,
            description=f"{operation_type} — not billed: {reason}",
            metadata={**(metadata or {}), 'unbilled_detail': detail},
            job_id=job_id,
            unbilled_reason=reason,
        )
        return {'success': False, 'error': f"{reason}: {detail}", 'credits_debited': 0}

    # ── Read-only balance ────────────────────────────────────────────────────

    async def get_available_credits(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> Optional[float]:
        """Total credits this user can spend right now: workspace pool + personal balance.

        Read-only; reserves nothing. Exists so a paid pipeline can refuse to START rather than
        discovering it is unfunded one debit at a time. Every debit_credits_* method above runs
        AFTER its upstream call, so a failed debit is money already spent — the log says
        "UNBILLED" and that is all it can do. (invariant 10, audit #286)

        Returns None when the balance cannot be read. Callers MUST treat None as "unknown,
        proceed", never as "empty, block": a transient RPC failure must not stop paying
        customers from processing documents. The failure being removed is unbounded spend by an
        account with zero credits, not one job on a flaky read.
        """
        try:
            resp = self.supabase.client.rpc(
                'get_available_credits',
                {'p_user_id': user_id, 'p_workspace_id': workspace_id},
            ).execute()
            if resp.data is None:
                return None
            return float(resp.data)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                "[credits] could not read available balance user=%s ws=%s: %s",
                user_id, workspace_id, e,
            )
            return None


# Singleton instance
_credits_service: Optional[CreditsIntegrationService] = None


def get_credits_service() -> CreditsIntegrationService:
    """Get singleton instance of CreditsIntegrationService."""
    global _credits_service
    if _credits_service is None:
        _credits_service = CreditsIntegrationService()
    return _credits_service
