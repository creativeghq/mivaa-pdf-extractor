"""Is a provider refusing our CREDENTIAL, and should we stop paying for calls that cannot work?

The unit of failure is the credential, not the call site: Perplexity answered 401 on 198 calls
a week for six weeks and the two subsystems that share the key each rendered it as zero (#416).
The POLICY here is pure and stdlib-only (MIVAA CI installs pytest alone); the state lives in
`public.provider_credential_health` and is reached through `breaker_store`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

#: Refusals that CANNOT succeed on retry. A timeout, a 500 or a 503 is a bad minute, not a dead
#: key, and must never open the breaker -- opening on those turns an outage into a self-inflicted
#: outage that outlives it.
AUTH_STATUSES = frozenset({401, 403})
QUOTA_STATUSES = frozenset({402})

#: 429 is ambiguous: rate limiting is transient, an exhausted plan is not. Perplexity's own body
#: says which -- "exceeded your current quota ... check your plan and billing" is the dead case.
_QUOTA_PHRASES = (
    "exceeded your current quota",
    "check your plan and billing",
    "insufficient credit",
    "insufficient_quota",
    "billing",
    "payment required",
)


def classify_provider_failure(
    http_status: Optional[int], error_text: Optional[str] = None
) -> Optional[str]:
    """'auth', 'quota', or None when this failure says nothing about the credential."""
    text = (error_text or "").lower()
    if http_status in AUTH_STATUSES:
        return "auth"
    if http_status in QUOTA_STATUSES:
        return "quota"
    if http_status == 429 and any(p in text for p in _QUOTA_PHRASES):
        return "quota"
    if http_status is None and any(p in text for p in _QUOTA_PHRASES):
        # A client that reports no status but quotes the provider's billing message.
        return "quota"
    return None


@dataclass(frozen=True)
class BreakerVerdict:
    """What the caller should do, and what to SAY when it does not call.

    `should_call` False is never rendered as an empty result: `status` is the collector status
    (anti-regression rule 3) and `reason` carries the provider's own last message.
    """

    should_call: bool
    is_probe: bool = False
    status: str = "ok"
    reason: Optional[str] = None
    consecutive_failures: int = 0

    @property
    def refused(self) -> bool:
        return not self.should_call


OPEN_STATUS = "collector_failed"

#: `call_agent` returns this when the breaker is open, so a caller can tell "we did not spend"
#: from "we spent and got nothing".
REFUSED_STATUS = "credential_refused"


class ProviderRefused(RuntimeError):
    """The provider will not serve this CREDENTIAL -- retrying changes nothing.

    Raised rather than returned where a caller's contract is a bare list: a source that returns
    [] is counted as "ran, found nothing", and that is the misreading this whole module exists
    to stop. Every fan-out in this codebase already records a raising source as failed.
    """


def is_credential_refusal(status: Optional[str], http_status: Optional[int],
                          error_text: Optional[str]) -> bool:
    """Did this reply fail because of the credential, rather than for a passing reason?"""
    if status == REFUSED_STATUS:
        return True
    return classify_provider_failure(http_status, error_text) is not None


def verdict_from_state(state: Optional[dict]) -> BreakerVerdict:
    """Turn `provider_breaker_state()` into a decision. An unreadable state CALLS.

    Failing closed here would mean one unreachable table silences every provider at once --
    strictly worse than the bug being fixed.
    """
    if not state:
        return BreakerVerdict(should_call=True)
    if state.get("probe"):
        return BreakerVerdict(
            should_call=True,
            is_probe=True,
            consecutive_failures=int(state.get("consecutive_auth_failures") or 0),
        )
    if not state.get("open"):
        return BreakerVerdict(
            should_call=True,
            consecutive_failures=int(state.get("consecutive_auth_failures") or 0),
        )
    n = int(state.get("consecutive_auth_failures") or 0)
    upstream = state.get("last_error") or "the provider refused our credential"
    return BreakerVerdict(
        should_call=False,
        status=OPEN_STATUS,
        reason=f"{n} consecutive refusals from the provider: {upstream}",
        consecutive_failures=n,
    )
