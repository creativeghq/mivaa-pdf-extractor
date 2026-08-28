"""
Guard: never send `temperature` to a model that rejects it.

WHY THIS EXISTS
---------------
Anthropic REMOVED sampling parameters (`temperature` / `top_p` / `top_k`) across the
current generation. Fable 5, Opus 5, Opus 4.8, Opus 4.7 and Sonnet 5 all answer a
request carrying one with a hard 400.

`claude_helper._build_payload` defaults `temperature=0.0`, so the parameter is sent
unless something actively suppresses it. That suppression used to be a DENYLIST naming
exactly two models:

    _MODELS_WITHOUT_TEMPERATURE = ("claude-opus-4-8", "claude-opus-4-6")

which meant every model NOT in it was sent `temperature` — `claude-opus-5` included.
Every Claude call in MIVAA goes through this helper, so pointing
`anthropic_model_validation` at Opus 5 would have 400'd the vision path, product
discovery, stage 4, the document classifier and chunk-type classification
simultaneously. Nothing in the codebase would have looked wrong; the failure only
exists at request time, and the vision path in particular has never run in production,
so it would have surfaced as "the new model is broken" rather than "we send it a field
it does not accept".

A denylist is the wrong shape for this. It fails OPEN: the danger is a model the list
has never heard of, and a list of things to exclude cannot describe one. The allowlist
fails CLOSED — an unrecognised model runs at the provider's default sampling, which is
a difference nobody will notice.

Source-based: reads the module text, imports no app module and touches no DB, so it
runs in CI (which installs pytest only) in about a second.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_HELPER = _ROOT / "app" / "services" / "core" / "claude_helper.py"

#: Models that reject sampling params with a 400. Sending `temperature` to any of
#: these breaks every call site at once.
_REJECTS_SAMPLING = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def _read_helper() -> str:
    assert _HELPER.exists(), f"missing {_HELPER}"
    return _HELPER.read_text(encoding="utf-8")


def _allowlist_prefixes(src: str) -> list[str]:
    """Pull the string literals out of the `_MODELS_WITH_TEMPERATURE` tuple."""
    m = re.search(
        r"^_MODELS_WITH_TEMPERATURE\s*=\s*\((.*?)\)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, (
        "`_MODELS_WITH_TEMPERATURE` not found in claude_helper.py. If it was renamed, "
        "this guard must be renamed with it — do not delete it."
    )
    return re.findall(r"[\"']([^\"']+)[\"']", m.group(1))


def test_the_gate_is_an_allowlist_not_a_denylist() -> None:
    """A denylist cannot describe a model it has never heard of."""
    src = _read_helper()
    # Match the ASSIGNMENT, not the name: the comment above the allowlist explains
    # what the denylist was and why it was wrong, and that prose must stay legal.
    assert not re.search(r"^_MODELS_WITHOUT_TEMPERATURE\s*=", src, re.MULTILINE), (
        "The denylist is back. It fails open: every model absent from it gets "
        "`temperature`, and the current generation answers that with a 400. Use "
        "`_MODELS_WITH_TEMPERATURE` (allowlist) instead."
    )
    assert re.search(r"^_MODELS_WITH_TEMPERATURE\s*=", src, re.MULTILINE)


def test_sampling_gate_is_applied_when_building_the_payload() -> None:
    """The allowlist is only worth anything if `_build_payload` consults it."""
    src = _read_helper()
    assert re.search(
        r"if\s+temperature\s+is\s+not\s+None\s+and\s+_model_supports_temperature\(",
        src,
    ), (
        "`_build_payload` no longer gates `temperature` on "
        "`_model_supports_temperature(model)`. Without that check the allowlist is "
        "decorative and every current-generation model 400s."
    )


@pytest.mark.parametrize("model", _REJECTS_SAMPLING)
def test_models_that_reject_sampling_are_not_allowlisted(model: str) -> None:
    """The whole point: these must never be handed a `temperature`."""
    prefixes = _allowlist_prefixes(_read_helper())
    matched = [p for p in prefixes if model.startswith(p)]
    assert not matched, (
        f"{model} matches allowlist prefix(es) {matched}, so `_build_payload` will "
        f"send it `temperature` and the API will answer 400. Anthropic removed "
        f"sampling parameters on this model."
    )


def test_allowlist_prefixes_are_specific_enough_to_be_safe() -> None:
    """A prefix like 'claude-' or 'claude-opus' would swallow the whole family.

    Guarding the entries themselves matters more than guarding today's model names:
    a too-short prefix re-creates the denylist's failure mode silently, because it
    would also match every model Anthropic releases next.
    """
    for prefix in _allowlist_prefixes(_read_helper()):
        assert prefix.count("-") >= 1, f"allowlist prefix {prefix!r} is too broad"
        assert prefix not in ("claude-opus", "claude-sonnet", "claude-fable"), (
            f"allowlist prefix {prefix!r} matches a whole model family, including "
            f"future members that may reject sampling params."
        )
