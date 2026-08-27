"""
Guard: a workspace's own probe questions actually reach the probe run.

WHY THIS EXISTS
---------------
`build_probes` carried four hardcoded prompts and a docstring reading "Caller may
add more via source_config". Nothing read `source_config` — not the product route,
not the subject route, not the nightly cron. So the extension point was documented,
discoverable, saveable from the API, and inert: a merchant could store their own
questions and be probed with the stock four forever, with nothing failing.

That is the same shape as a prompt fallback firing silently (CLAUDE.md): the
feature reports success and measures the wrong thing.

Stdlib-only by necessity — this repo's CI installs pytest and nothing else, and
the service module cannot be imported without a Supabase client.
"""

import ast
import importlib.util
import re
from pathlib import Path

_APP = Path(__file__).resolve().parents[2] / "app"
_TEMPLATES = _APP / "services" / "integrations" / "llm_probe_templates.py"
_SERVICE = _APP / "services" / "integrations" / "llm_mention_probe_service.py"
_ROUTES = _APP / "api" / "mention_monitoring_routes.py"


def _load():
    spec = importlib.util.spec_from_file_location("llm_probe_templates", _TEMPLATES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tm = _load()


class _Facets:
    label = "Materials Hub"
    brand = "Materials Hub"
    product_type = "bathroom tiles"
    competitor_brands = ["Flobali", "Tsatsos"]


class TestCustomPromptsReachTheRun:
    def test_a_custom_prompt_is_included_and_rendered(self):
        out = tm.build_probes(
            _Facets(), custom_probes=[{"key": "greek", "prompt": "Who stocks {label} in Greece?"}]
        )
        mine = [p for p in out if p["key"] == "greek"]
        assert mine, "a saved custom prompt must be probed"
        assert mine[0]["prompt"] == "Who stocks Materials Hub in Greece?"

    def test_defaults_can_be_replaced_entirely(self):
        """Asking four irrelevant stock questions alongside four good ones dilutes
        share of voice with noise the merchant never chose to measure."""
        out = tm.build_probes(
            _Facets(), custom_probes=[{"key": "only", "prompt": "Do you know {label}?"}],
            include_defaults=False,
        )
        assert [p["key"] for p in out] == ["only"]

    def test_a_subject_can_never_end_up_with_zero_probes(self):
        """Otherwise it looks tracked and is silently no longer measured."""
        assert len(tm.build_probes(_Facets(), custom_probes=[], include_defaults=False)) == 4

    def test_malformed_rows_are_dropped_not_raised(self):
        """This runs in a nightly sweep over every subject. One bad row must not
        stop the sweep for every other workspace."""
        out = tm.build_probes(
            _Facets(),
            custom_probes=["nonsense", {"no_prompt": 1}, {"prompt": "x"}, None],
        )
        assert [p["key"] for p in out] == [t["key"] for t in tm.DEFAULT_TEMPLATES]

    def test_custom_probes_are_capped(self):
        """Each probe is a paid model call per engine; an unbounded list is an
        unbounded bill."""
        many = [{"key": f"k{i}", "prompt": f"question number {i}?"} for i in range(50)]
        out = tm.build_probes(_Facets(), custom_probes=many, include_defaults=False)
        assert len(out) <= 12

    def test_an_unknown_placeholder_does_not_break_the_prompt(self):
        """A merchant writing 'what does {this} mean' must not get a KeyError —
        which is what str.format would raise."""
        assert tm.render_template("what does {this} mean for {label}?", {"label": "X"}) \
            == "what does {this} mean for X?"


class TestTheWiringIsReal:
    def test_the_service_delegates_to_the_template_module(self):
        src = _SERVICE.read_text(encoding="utf-8")
        assert "from app.services.integrations.llm_probe_templates import build_probes" in src
        assert "def build_probes(" not in src, (
            "build_probes must live in ONE place — a second copy in the service is how "
            "the two sets of questions drift apart"
        )
        assert "custom_probes=custom_probes" in src

    def test_every_probe_call_site_passes_the_overrides(self):
        """The cron is the one that matters: it is where 99% of probes are run, and
        it is the easiest of the three to forget."""
        src = _ROUTES.read_text(encoding="utf-8")
        calls = list(re.finditer(r"\.probe\(", src))
        assert len(calls) >= 3, f"expected the three probe call sites, found {len(calls)}"
        offenders = []
        for m in calls:
            tail = src[m.end():m.end() + 700]
            end = tail.find("\n        )")
            call = tail[: end if end != -1 else len(tail)]
            if "_probe_overrides(" not in call:
                offenders.append(src[: m.start()].count("\n") + 1)
        assert not offenders, (
            f"probe() call sites at lines {offenders} do not pass _probe_overrides(...), "
            "so a workspace's saved questions are ignored on that path"
        )

    def test_the_overrides_helper_reads_both_keys(self):
        src = _ROUTES.read_text(encoding="utf-8")
        helper = src[src.index("def _probe_overrides"):]
        helper = helper[: helper.index("\n@router")]
        assert "custom_probes" in helper and "include_default_probes" in helper
