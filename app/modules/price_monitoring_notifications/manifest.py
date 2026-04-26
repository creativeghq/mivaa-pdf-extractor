from app.modules._core.types import ModuleManifest

manifest = ModuleManifest(
    slug="price-monitoring-notifications",
    name="Price Monitoring Notifications",
    description=(
        "Send users price-drop, new-retailer, promo-start, and anomaly alerts "
        "for tracked products. Per-channel credit-metered: bell free, email "
        "costs credits, webhook free. Opt-in per product or per tracked query."
    ),
    category="pricing",
    price_tier="pro",
    icon="BellRing",
    version="0.1.0",
)
