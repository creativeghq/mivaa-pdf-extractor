"""Provider rates have ONE definition (main repo #365)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "app" / "modules" / "_core" / "provider_pricing.py"

COST_SITES = [
    ROOT / "app" / "services" / "integrations" / "job_cost_logger.py",
    ROOT / "app" / "services" / "integrations" / "mention_cost_logger.py",
    ROOT / "app" / "services" / "integrations" / "perplexity_price_search_service.py",
]

# Module-level assignment of a rate-shaped constant, e.g. `SONAR_PRO_INPUT_PER_1K = 0.003`.
RATE_ASSIGNMENT = re.compile(
    r"^(SONAR_[A-Z0-9_]*|HAIKU_[A-Z0-9_]*|DATAFORSEO_[A-Z0-9_]*PER_CALL|FIRECRAWL_[A-Z0-9_]*|"
    r"GPT4O_MINI_[A-Z0-9_]*|GEMINI_[A-Z0-9_]*|YOUTUBE_PER_CALL)\s*=\s*[0-9]",
    re.MULTILINE,
)


def test_the_shared_module_exists_and_is_not_empty():
    # A guard whose subject vanished would pass by matching nothing, which is the failure mode this
    # whole issue kept finding. Pin the floor.
    assert SHARED.exists(), "app/modules/_core/provider_pricing.py is gone"
    text = SHARED.read_text(encoding="utf-8")
    assert len(RATE_ASSIGNMENT.findall(text)) >= 8, (
        "the shared module no longer defines the provider rates — either it was gutted or the "
        "naming convention changed and this check went blind"
    )


@pytest.mark.parametrize("path", COST_SITES, ids=lambda p: p.name)
def test_no_cost_logger_redefines_a_provider_rate(path: Path):
    assert path.exists(), f"{path} moved — update COST_SITES"
    offenders = RATE_ASSIGNMENT.findall(path.read_text(encoding="utf-8"))
    assert not offenders, (
        f"{path.name} defines its own provider rate(s): {sorted(set(offenders))}.\n"
        "These belong in app/modules/_core/provider_pricing.py. A second copy of a price cannot "
        "disagree loudly — that is how Sonar Pro was recorded 3x/15x light and DataForSEO Labs "
        "~12x light, each needing the same fix in two files."
    )


@pytest.mark.parametrize("path", COST_SITES, ids=lambda p: p.name)
def test_haiku_is_resolved_not_restated(path: Path):
    # `claude-haiku-4-5` has an ai_model_pricing row, so its rate is looked up. A literal beside a
    # DB-backed price keeps the old number forever after an admin edit, silently.
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"HAIKU_(INPUT|OUTPUT)_PER_1K\s*=\s*[0-9]", text), (
        f"{path.name} restates a Haiku token rate. It has an ai_model_pricing row — call "
        "provider_pricing.haiku_token_cost(), which resolves through AIPricingConfig."
    )


def test_the_sonar_model_branch_has_one_home():
    # Every site used to hand-roll `is_pro = model == "sonar-pro"` and pick three rates from it.
    # That is the same duplication one rung down: three places deciding what "pro" means.
    for path in COST_SITES:
        text = path.read_text(encoding="utf-8")
        assert 'SONAR_PRO_PER_CALL if' not in text, (
            f"{path.name} branches on the model to pick Sonar rates — call sonar_rates(model)."
        )


def test_the_shared_rates_carry_a_source():
    # A rate with no provenance is indistinguishable from a guess, and this codebase has shipped
    # guesses that looked like facts. Every block must name where the number came from.
    text = SHARED.read_text(encoding="utf-8")
    assert text.count("Source:") >= 3, "the shared rates have lost their source URLs"
    assert "verified 2026-08-17" in text, "the shared rates have no verification date"
