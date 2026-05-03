from app.modules._core.types import ModuleManifest

manifest = ModuleManifest(
    slug="mention-monitoring-notifications",
    name="Mention Monitoring Notifications",
    description=(
        "Send users mention-spike, negative-sentiment, new-outlet, and "
        "LLM-visibility-change alerts for tracked subjects. Per-channel "
        "credit-metered: bell free, email costs credits, webhook free. "
        "Opt-in per subject."
    ),
    category="pricing",
    price_tier="pro",
    icon="BellRing",
    version="0.1.0",
)
