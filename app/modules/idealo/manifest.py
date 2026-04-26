from app.modules._core.types import ModuleManifest

manifest = ModuleManifest(
    slug="idealo",
    name="Idealo (DACH + IT)",
    description=(
        "Idealo.de / .it / .at / .uk price comparison as a discovery source "
        "for non-Greek markets. Mirrors the greek_marketplaces module shape: "
        "Firecrawl scrape of the public search page sorted by price ascending. "
        "Extracts the cheapest published offer per query."
    ),
    category="pricing",
    price_tier="pro",
    icon="ShoppingCart",
    version="0.1.0",
)
