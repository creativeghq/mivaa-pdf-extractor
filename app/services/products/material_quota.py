"""
Materials plan-quota helpers (#214).

The hard boundary is the `enforce_material_quota` DB trigger on `products`
(BEFORE INSERT, service rows exempt) — nothing here replaces it. These helpers
exist so bulk pipelines can fail fast / clamp BEFORE burning per-product LLM
spend on rows whose insert would be refused anyway.

The derivation lives in SQL (`material_quota_remaining(workspace)` RPC — the
plan limit minus the workspace's non-service product count; -1 = unlimited),
so Python never re-implements the count. Both helpers FAIL OPEN (-1) on any
error: a broken pre-flight must not block ingestion — the trigger still holds
the line.
"""

import logging

logger = logging.getLogger(__name__)


def material_quota_remaining_sync(client, workspace_id) -> int:
    """Remaining new-material allowance for a workspace. -1 = unlimited. Sync client."""
    if not workspace_id:
        return -1
    try:
        resp = client.rpc(
            'material_quota_remaining', {'p_workspace_id': str(workspace_id)}
        ).execute()
        return int(resp.data) if resp.data is not None else -1
    except Exception as e:
        logger.warning(f"material_quota_remaining failed (fail-open, trigger still enforces): {e}")
        return -1


async def material_quota_remaining_async(db, workspace_id) -> int:
    """Remaining new-material allowance for a workspace. -1 = unlimited. Async facade."""
    if not workspace_id:
        return -1
    try:
        resp = await db.rpc(
            'material_quota_remaining', {'p_workspace_id': str(workspace_id)}
        ).execute()
        return int(resp.data) if resp.data is not None else -1
    except Exception as e:
        logger.warning(f"material_quota_remaining failed (fail-open, trigger still enforces): {e}")
        return -1


def is_quota_error(exc) -> bool:
    """True when an exception came from a quota_exceeded DB trigger refusal."""
    return 'quota_exceeded' in str(exc)
