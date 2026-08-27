"""One implementation of "these ids describe the same thing".

WHY THIS EXISTS
---------------
The audit's most-repeated finding is the two-unchecked-ids class: a function receives
`document_id` and `workspace_id` (and often `product_id`) independently, combines them
into a write, and nothing verifies they are related. It was counted to SIXTEEN separate
instances across the engagement — #20 M7-5, #25 M12-4, #31 M17-4, and thirteen others.

MIVAA has **no RLS backstop by design**: every database call runs as service role, which
bypasses row-level security entirely. So the check either exists in Python or it does not
exist. Security invariant 1 says using the service-role client does not exempt a caller —
it makes the manual check mandatory.

WHY A SHARED MODULE RATHER THAN THE CHECK AT EACH SITE
------------------------------------------------------
The first fix for this class was written inline. By the third it would have been three
copies of a rule whose whole purpose is to be uniform — and this codebase has already
paid for that lesson three times over: seven copies of the credit-debit rule (#30 M16-1),
one of which still held the bug the other six had fixed; three copies of the PDF page
bound (#22, #24, #35); four hand-written forced-tool loops (#32). Consolidating before
the drift rather than after it.

FAILING CLOSED IS THE POINT
---------------------------
Every function here raises on a lookup ERROR as well as on a mismatch. A tenancy check
that treats "I could not find out" as "go ahead" switches itself off exactly when the
database is unhappy, which is when it is least able to protect anything.

They raise `TenancyViolation`, which `main.py` already maps to **404** with the reason
logged rather than returned — a 403 confirms the id is real and belongs to somebody else,
which is the enumeration oracle the check exists to remove.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List

from app.utils.exceptions import TenancyViolation

logger = logging.getLogger(__name__)


def _client(supabase: Any) -> Any:
    """The PostgREST client, whether given the wrapper or the client itself.

    Callers in this codebase pass both shapes — `get_supabase_client()` returns a wrapper
    with `.client`, while some services hold the client directly. Guessing wrong here
    would raise `AttributeError` inside a security check, which is the one place an
    exception must not be mistaken for a refusal.
    """
    return getattr(supabase, "client", supabase)


def assert_document_in_workspace(supabase: Any, document_id: str, workspace_id: str) -> None:
    """Raise unless `document_id` belongs to `workspace_id`.

    The table is `documents`. It is NOT `pdf_documents` — that is the storage BUCKET
    name and there is no such table, so a check written against it would pass on every
    call because it could never find anything to disagree with. That is worse than no
    check, because it reads as one.
    """
    if not document_id or not workspace_id:
        raise TenancyViolation(
            f"both ids are required (document_id={document_id!r}, "
            f"workspace_id={workspace_id!r})"
        )
    try:
        row = (
            _client(supabase)
            .table("documents")
            .select("workspace_id")
            .eq("id", document_id)
            .maybe_single()
            .execute()
        )
    except Exception as e:
        raise TenancyViolation(
            f"could not verify document {document_id} against workspace {workspace_id}: {e}"
        ) from e

    owner = ((row.data if row else None) or {}).get("workspace_id")
    if str(owner) != str(workspace_id):
        raise TenancyViolation(
            f"document {document_id} belongs to workspace {owner!r}, not {workspace_id!r}"
        )


def assert_products_in_document(
    supabase: Any,
    product_ids: Iterable[str],
    document_id: str,
    workspace_id: str,
) -> List[str]:
    """Raise unless every id in `product_ids` belongs to that document AND workspace.

    Returns the verified ids, so the caller uses the checked list rather than the one it
    was handed — a check whose result is discarded is a check that can be removed
    without anything failing.

    Raises rather than filtering. These ids arrive from a pipeline that just created the
    products from this same document, so a mismatch is a bug or an attempt, not a
    routine condition. Silently dropping them would attach catalogue knowledge to some
    products and not others, and report success.
    """
    ids = [p for p in (product_ids or []) if p]
    if not ids:
        return []

    assert_document_in_workspace(supabase, document_id, workspace_id)

    try:
        rows = (
            _client(supabase)
            .table("products")
            .select("id")
            .in_("id", ids)
            .eq("source_document_id", document_id)
            .eq("workspace_id", workspace_id)
            .execute()
        )
    except Exception as e:
        raise TenancyViolation(
            f"could not verify {len(ids)} product(s) against document {document_id}: {e}"
        ) from e

    verified = {r["id"] for r in (rows.data or [])}
    missing = [p for p in ids if p not in verified]
    if missing:
        raise TenancyViolation(
            f"{len(missing)} of {len(ids)} product(s) do not belong to document "
            f"{document_id} in workspace {workspace_id} — first: {missing[0]}"
        )
    return [p for p in ids if p in verified]
