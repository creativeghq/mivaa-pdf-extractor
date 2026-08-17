"""Per-call and per-unit provider rates — ONE definition, shared by every cost logger.

WHY THIS EXISTS. `job_cost_logger` and `mention_cost_logger` each kept their own copy of the same
numbers: six Sonar constants, two Haiku constants, and the DataForSEO SERP rate under two different
names. Two copies of a price cannot disagree loudly — a wrong rate is a valid number, so nothing
raises, the row inserts, and the dashboard totals look plausible. That is exactly how Sonar Pro came
to be recorded 3x light on input and 15x light on output, and DataForSEO Labs ~12x light: both
constants were wrong in `mention_cost_logger` and had to be fixed in *two* places once found
(main repo #365). The next person to correct a rate would have had the same two-places problem, and
no way to know it.

WHAT BELONGS HERE, AND WHAT DOES NOT. This module holds the rates that have **no
`ai_model_pricing` row**: per-request search fees, per-task API charges, per-credit scrape costs.
They are provider list prices, not platform prices, and there is nowhere else for them to live.

TOKEN rates for models that DO have a row do NOT belong here — they go through
`AIPricingConfig.calculate_cost()`, which reads `ai_model_pricing` first. A literal token rate
beside a DB-backed one is a second USD source, which is the thing CLAUDE.md forbids and the thing
this module is consolidating away from. `claude-haiku-4-5` is the live example: it has a row, so
its rate is resolved, not restated.

Every number carries where it came from and when it was checked. A rate with no source is a guess,
and a guess in this file is indistinguishable from a fact.
"""

# ── Perplexity Sonar ────────────────────────────────────────────────────────
# Billed as a per-request SEARCH FEE plus tokens, and the token rate differs per model AND per
# direction. One shared token constant was correct for Sonar by coincidence and wrong for Sonar Pro
# by a factor of 3 (input) and 15 (output).
# Source: https://docs.perplexity.ai/getting-started/pricing — verified 2026-08-17 against both the
# published rates and our own ledger (solving raw_cost_usd over 385 Sonar and 76 Sonar Pro calls
# reproduced both module totals to the cent).
SONAR_PER_CALL = 0.005            # $5 / 1000 requests
SONAR_PRO_PER_CALL = 0.01         # inside the published $6-14 / 1000 band
SONAR_INPUT_PER_1K = 0.001        # $1 / 1M
SONAR_OUTPUT_PER_1K = 0.001       # $1 / 1M
SONAR_PRO_INPUT_PER_1K = 0.003    # $3 / 1M
SONAR_PRO_OUTPUT_PER_1K = 0.015   # $15 / 1M


def sonar_rates(model: str) -> tuple[float, float, float]:
    """`(per_call, input_per_1k, output_per_1k)` for a Sonar model.

    One place decides what "sonar-pro" means. Both loggers had their own `is_pro` branch, which is
    the same duplication a rung down.
    """
    is_pro = model == "sonar-pro"
    return (
        SONAR_PRO_PER_CALL if is_pro else SONAR_PER_CALL,
        SONAR_PRO_INPUT_PER_1K if is_pro else SONAR_INPUT_PER_1K,
        SONAR_PRO_OUTPUT_PER_1K if is_pro else SONAR_OUTPUT_PER_1K,
    )


# ── DataForSEO ──────────────────────────────────────────────────────────────
# SERP endpoints (Google Organic, News, Jobs) share one standard-queue rate. They were two
# constants — DATAFORSEO_JOBS_PER_CALL and DATAFORSEO_NEWS_PER_CALL — holding the identical number
# in two files, which reads as two independent facts and is one.
# Source: https://dataforseo.com/apis/serp-api/pricing — verified 2026-08-17.
# High priority is $0.0012 and live is $0.002; if a caller switches queue this is wrong.
DATAFORSEO_SERP_PER_CALL = 0.0006

# Labs is a DIFFERENT product and was priced at the SERP rate, ~12x light.
# The per-item component ($0.00012 per returned row) is still not modelled, so a large `limit`
# costs more than this says, and include_clickstream_data doubles the real cost again.
# Source: https://dataforseo.com/pricing/dataforseo-labs/dataforseo-google-api — verified 2026-08-17.
DATAFORSEO_LABS_PER_CALL = 0.012

# ── Firecrawl ───────────────────────────────────────────────────────────────
# Per credit; 1 credit per standard page, 5 on stealth/enhanced. NOTE: the platform is on the FREE
# tier, where the marginal cost is $0 — this constant is what MIVAA has historically applied and is
# kept so historical rows stay explicable. The paid tiers are $0.0032 (Hobby) / $0.00083 (Standard)
# / $0.00066 (Growth) / $0.0006 (Scale); none of them is this number, so it needs revisiting the
# day the account upgrades. See `ai_model_pricing.firecrawl-scrape` for the current decision.
FIRECRAWL_PER_CREDIT = 0.002

# ── Free-quota providers ────────────────────────────────────────────────────
# Explicitly 0 rather than absent: "we checked and it is free" and "nobody priced it" are different
# statements, and only one of them should render as zero.
YOUTUBE_PER_CALL = 0.0

# ── LLM-probe models with no ai_model_pricing row ───────────────────────────
# These are used only by the mention LLM probe. They stay literal because no row exists to resolve
# them against; the moment one does, they should move to AIPricingConfig like Haiku did.
# Source: provider list prices, verified 2026-04-18.
GPT4O_MINI_INPUT_PER_1K = 0.00015
GPT4O_MINI_OUTPUT_PER_1K = 0.0006
GEMINI_FLASH_INPUT_PER_1K = 0.00010
GEMINI_FLASH_OUTPUT_PER_1K = 0.0004


def haiku_token_cost(input_tokens: int, output_tokens: int) -> float:
    """Raw USD for a Haiku 4.5 call, resolved through `ai_model_pricing`.

    NOT a constant. `claude-haiku-4-5` has a row in the platform's single USD source, and both
    loggers previously restated it as `HAIKU_INPUT_PER_1K = 0.001 / HAIKU_OUTPUT_PER_1K = 0.005` —
    a third copy of a number an admin can edit, which would have kept the old value forever after
    any edit, silently and in two files.

    Delegates to `AIPricingConfig`, which reads the DB first and caches for five minutes, so this
    stays cheap in the per-call path.
    """
    from app.config.ai_pricing import AIPricingConfig

    result = AIPricingConfig.calculate_cost(
        model="claude-haiku-4-5",
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
        include_markup=False,   # the loggers apply markup themselves
    )
    return float(result["raw_cost_usd"])
