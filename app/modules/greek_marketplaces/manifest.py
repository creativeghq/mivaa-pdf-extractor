from app.modules._core.types import ModuleManifest

manifest = ModuleManifest(
    slug="greek-marketplaces",
    name="Greek Marketplaces",
    description=(
        "Skroutz, Bestprice, and Shopflix as always-on discovery sources "
        "for price monitoring. Returns first-party retailer URLs with "
        "verified prices for Greek-market products."
    ),
    category="pricing",
    price_tier="pro",
    icon="ShoppingCart",
    version="0.1.0",
)
