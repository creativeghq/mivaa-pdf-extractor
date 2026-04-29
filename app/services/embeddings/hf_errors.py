"""
HuggingFace endpoint error classification.

The HF inference-endpoints API returns 403 with body
"Payment method required for namespace: <ns>" when the account can't bill —
expired card, no card on file, payment failed, account suspended. Treating
this as a normal "transient" failure means we burn 3 × 90s retries before
giving up and the upload sits "warming up" for ~5 minutes producing nothing.

`HFBillingError` is the special-cased subclass: when ANY endpoint manager
sees this signature, it raises HFBillingError instead of swallowing it.
The orchestrator catches HFBillingError and fails the job IMMEDIATELY with
a clean message — no retries, no Sentry retry-spam, no silently warming
endpoints that will never come up.
"""


class HFBillingError(RuntimeError):
    """HuggingFace refused to start an endpoint because the account can't bill.

    Distinguished from generic warmup failures because retries CANNOT fix
    this — the user has to add a payment method. Bubbles all the way up
    to the orchestrator which fails the job fast.
    """
    def __init__(self, endpoint_name: str, namespace: str = "basiliskan", original: Exception | None = None):
        self.endpoint_name = endpoint_name
        self.namespace = namespace
        self.original = original
        super().__init__(
            f"HuggingFace billing error for endpoint '{endpoint_name}' "
            f"(namespace='{namespace}'): payment method required. "
            f"Add a card at https://huggingface.co/settings/billing — retries cannot fix this."
        )


_BILLING_ERROR_SIGNATURES = (
    "Payment method required",
    "payment method required",
    "payment_method_required",
    "billing failed",
    "subscription required",
)


def is_hf_billing_error(exc: BaseException) -> bool:
    """Return True if `exc` looks like an HF billing/payment failure.

    Matches on the body text since HF surfaces these as 403s with the
    specific phrase. Both string-cast and `.response.text` are checked
    because the huggingface_hub client wraps the body differently per
    SDK version.
    """
    msg = str(exc) or ""
    if any(sig in msg for sig in _BILLING_ERROR_SIGNATURES):
        return True
    # huggingface_hub.errors.HfHubHTTPError carries the raw response
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = getattr(resp, "text", "") or ""
            if any(sig in body for sig in _BILLING_ERROR_SIGNATURES):
                return True
        except Exception:
            pass
    return False
