from app.modules._core.types import ModuleManifest

manifest = ModuleManifest(
    slug="job-research-notifications",
    name="Job Research Notifications",
    description=(
        "Sends the consolidated daily job-digest email summarising newly-discovered "
        "job listings across all of a user's tracked job searches. Per-channel "
        "credit-metered: bell free, email costs credits, webhook free."
    ),
    category="research",
    price_tier="pro",
    icon="Mail",
    version="0.1.0",
)
