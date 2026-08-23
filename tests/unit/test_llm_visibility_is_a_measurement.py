"""
Guard: the LLM-visibility probe pipeline measures a TREND, a SHARE, and a CITATION.

WHY THIS EXISTS
---------------
Issue #349. Everything below was live and every one of them returned a well-formed,
plausible number, which is why none of them failed on its own:

  A1  `record_mention` had no `cited_urls`, and Perplexity Sonar's native `citations`
      array was read and discarded. A GHOST CITATION - our page used as the source
      while the brand is never named in the answer - is invisible to a mention count,
      so the pipeline could not see the thing the whole product is about.

  A2  `visibility_snapshot()` read exactly ONE `probe_run_id` while its docstring
      claimed "position trend". Every probe run since the feature shipped was sitting
      in `llm_mention_probes` and nothing read across them.

  A3  Per-probe `sentiment` was persisted and then only ever shown inside four capped
      samples. There was no sentiment number in the snapshot at all, and no per-model
      split - which is the split that matters, because the same brand reads differently
      on different answer engines.

  A4  `/share-of-voice` counted competitors and never the subject, so the brand whose
      page it is had no share of its own voice; and its `days` parameter was declared,
      range-validated, and then never applied to the query.

Pure unit test: the rollups and windowed reads are exercised over hand-built rows, so
it imports no DB client and makes no network call.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parents[2] / "app"
_SERVICE = _APP / "services" / "integrations" / "llm_mention_probe_service.py"
_MATH = _APP / "services" / "integrations" / "llm_visibility_math.py"
_ROUTES = _APP / "api" / "mention_monitoring_routes.py"


def _load_math():
    """Load the derivations WITHOUT importing `app`.

    CI here installs pytest and nothing else - `app.services.__init__` reaches for a
    Supabase client on import, so anything living in the service module is untestable
    by construction. `llm_visibility_math` imports stdlib only, which is precisely
    why the rollups were moved into it.
    """
    spec = importlib.util.spec_from_file_location("llm_visibility_math", _MATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vm = _load_math()
citation_domain = vm.citation_domain
domain_is_ours = vm.domain_is_ours
dedupe_urls = vm.dedupe_urls
_sentiment_rollup = vm.sentiment_rollup
_citation_rollup = vm.citation_rollup


def _row(**kw):
    base = {
        "probe_run_id": "run-1", "run_at": "2026-08-01T00:00:00+00:00",
        "model": "sonar", "mentioned": False, "position": None,
        "sentiment": "neutral", "competitors_mentioned": [],
        "cited_urls": [], "brand_cited": None,
    }
    base.update(kw)
    return base


# ────────────────────────────────────────────────────────────────────────────
# A1 — citations
# ────────────────────────────────────────────────────────────────────────────

class TestCitationsAreCaptured:
    def test_a_cited_url_on_our_domain_counts_as_ours(self):
        assert domain_is_ours("https://flobali.gr/products/x", "flobali.gr")
        assert domain_is_ours("https://www.flobali.gr/", "flobali.gr")
        assert domain_is_ours("https://blog.flobali.gr/post", "flobali.gr")
        # The subject may have stored the domain with the prefix; same answer.
        assert domain_is_ours("https://flobali.gr/x", "www.flobali.gr")

    def test_a_lookalike_domain_is_not_ours(self):
        """A substring test - the obvious implementation - says yes to all three."""
        assert not domain_is_ours("https://notflobali.gr/x", "flobali.gr")
        assert not domain_is_ours("https://flobali.gr.evil.com/x", "flobali.gr")
        assert not domain_is_ours("https://example.com/?ref=flobali.gr", "flobali.gr")

    def test_undecidable_without_a_homepage_domain(self):
        """No domain configured is NOT 'we were never cited'."""
        assert not domain_is_ours("https://flobali.gr/x", None)
        assert not domain_is_ours("https://flobali.gr/x", "")

    def test_citation_domain_strips_scheme_port_and_www(self):
        assert citation_domain("https://www.Example.COM:8443/a/b?c=1") == "example.com"
        assert citation_domain("not a url") == ""
        assert citation_domain("") == ""

    def test_dedupe_keeps_order_and_rejects_non_http(self):
        got = dedupe_urls([
            "https://a.com/x", "https://A.com/x/", "javascript:alert(1)",
            "  ", "https://b.com/y",
        ])
        assert got == ["https://a.com/x", "https://b.com/y"]

    def test_ghost_citation_is_cited_but_not_mentioned(self):
        roll = _citation_rollup([
            # cited AND named - ordinary win, not a ghost
            _row(mentioned=True, brand_cited=True, cited_urls=["https://flobali.gr/a"]),
            # cited and NEVER named - the ghost
            _row(mentioned=False, brand_cited=True, cited_urls=["https://flobali.gr/b"]),
            # somebody else entirely
            _row(mentioned=False, brand_cited=False, cited_urls=["https://rival.com/c"]),
        ])
        assert roll["probes_with_citations"] == 3
        assert roll["brand_cited"] == 2
        assert roll["ghost_citations"] == 1

    def test_no_homepage_domain_is_reported_not_folded_into_zero(self):
        roll = _citation_rollup([
            _row(mentioned=False, brand_cited=None, cited_urls=["https://x.com/a"]),
            _row(mentioned=False, brand_cited=None, cited_urls=["https://y.com/b"]),
        ])
        assert roll["ghost_citations"] == 0
        # ...and the reason it is zero is visible, so the UI can ask for the domain
        # instead of reporting a confident "never cited".
        assert roll["undecidable_no_homepage_domain"] == 2

    def test_top_cited_domains_ranks_by_frequency(self):
        roll = _citation_rollup([
            _row(cited_urls=["https://rival.com/a", "https://rival.com/b"]),
            _row(cited_urls=["https://www.rival.com/c", "https://other.com/d"]),
        ])
        assert roll["top_cited_domains"][0] == ("rival.com", 3)

    def test_the_tool_schema_has_somewhere_to_put_them(self):
        """A1's root cause: nothing in `record_mention` could carry a URL."""
        src = _SERVICE.read_text(encoding="utf-8")
        schema = src[src.index('"name": "record_mention"'):]
        schema = schema[:schema.index("tool_choice")]
        assert '"cited_urls"' in schema

    def test_sonar_citations_are_read_natively_not_re_derived(self):
        src = _SERVICE.read_text(encoding="utf-8")
        pplx = src[src.index("async def _call_perplexity"):]
        pplx = pplx[:pplx.index("async def _extract")]
        # Both shapes: `citations` is the legacy flat list, `search_results` current.
        # Reading only one takes citations to zero on a version bump, with no error.
        assert 'data.get("citations")' in pplx
        assert 'data.get("search_results")' in pplx

    def test_every_model_call_returns_the_citation_channel(self):
        """ModelReply, not a bare tuple - so the next field cannot shift an unpack."""
        src = _SERVICE.read_text(encoding="utf-8")
        assert "class ModelReply(NamedTuple):" in src
        fields = src[src.index("class ModelReply(NamedTuple):"):]
        assert "citations: List[str]" in fields[:fields.index("def ")]
        tree = ast.parse(src)
        callers = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name.startswith("_call_")
        ]
        assert {c.name for c in callers} >= {
            "_call_model", "_call_anthropic", "_call_openai",
            "_call_gemini", "_call_perplexity",
        }
        for fn in callers:
            ann = ast.unparse(fn.returns) if fn.returns else ""
            assert ann == "ModelReply", f"{fn.name} returns {ann!r}, not ModelReply"

    def test_probe_persists_both_citation_columns(self):
        src = _SERVICE.read_text(encoding="utf-8")
        body = src[src.index("    async def probe("):src.index("    # ───── Visibility analytics")]
        assert '"cited_urls": cited_urls' in body
        assert '"brand_cited"' in body


# ────────────────────────────────────────────────────────────────────────────
# A3 — sentiment
# ────────────────────────────────────────────────────────────────────────────

class TestSentimentIsAggregated:
    def test_score_runs_from_minus_one_to_plus_one(self):
        assert _sentiment_rollup([
            _row(mentioned=True, sentiment="positive"),
            _row(mentioned=True, sentiment="positive"),
        ])["score"] == 1.0
        assert _sentiment_rollup([
            _row(mentioned=True, sentiment="negative"),
        ])["score"] == -1.0
        assert _sentiment_rollup([
            _row(mentioned=True, sentiment="positive"),
            _row(mentioned=True, sentiment="negative"),
        ])["score"] == 0.0

    def test_unmentioned_probes_do_not_dilute_the_score(self):
        """The extractor writes `neutral` when the subject was never named.

        Counting those would drag the score toward neutral in proportion to how
        INVISIBLE the brand is, which inverts the signal for the subjects that most
        need it.
        """
        rows = [_row(mentioned=True, sentiment="positive")] + [
            _row(mentioned=False, sentiment="neutral") for _ in range(9)
        ]
        roll = _sentiment_rollup(rows)
        assert roll["basis_probes"] == 1
        assert roll["score"] == 1.0

    def test_never_mentioned_scores_none_not_zero(self):
        """'No opinion recorded' and 'the opinion was neutral' are different facts."""
        roll = _sentiment_rollup([_row(mentioned=False, sentiment="neutral")])
        assert roll["score"] is None
        assert roll["basis_probes"] == 0

    def test_the_snapshot_breaks_sentiment_out_per_model(self):
        src = _SERVICE.read_text(encoding="utf-8")
        snap = src[src.index("    def visibility_snapshot("):src.index("    MAX_WINDOW_ROWS")]
        assert '"sentiment": sentiment_rollup(rows)' in snap
        assert 'd["sentiment"] = sentiment_rollup(model_rows)' in snap
        assert 'd["citations"] = citation_rollup(model_rows)' in snap


# ────────────────────────────────────────────────────────────────────────────
# A2 — trend
# ────────────────────────────────────────────────────────────────────────────

class _FakeService:
    """Stands in for LlmMentionProbeService's two windowed reads.

    The service fetches rows and hands them to `llm_visibility_math`; this exercises
    the second half, which is where every number is actually produced.
    """

    def __init__(self, rows, truncated=False):
        self._rows = list(rows)
        self._truncated = truncated

    def visibility_trend(self, _tracked_mention_id, *, days=90):
        return vm.trend_from_rows(self._rows, days=days, truncated=self._truncated)

    def share_of_voice_series(self, tracked_mention_id, *, subject_label, days=30):
        out = vm.share_of_voice_from_rows(
            self._rows, subject_label=subject_label, days=days, truncated=self._truncated,
        )
        out["tracked_mention_id"] = tracked_mention_id
        return out


class TestTrendReadsAcrossRuns:
    def _two_runs(self):
        return [
            # newest first, the way the query returns them
            _row(probe_run_id="r2", run_at="2026-08-08T00:00:00+00:00",
                 mentioned=True, position=2, sentiment="positive"),
            _row(probe_run_id="r2", run_at="2026-08-08T00:00:01+00:00", mentioned=False),
            _row(probe_run_id="r1", run_at="2026-08-01T00:00:00+00:00",
                 mentioned=True, position=5, sentiment="neutral"),
            _row(probe_run_id="r1", run_at="2026-08-01T00:00:01+00:00", mentioned=False),
        ]

    def test_one_point_per_run_oldest_first(self):
        trend = _FakeService(self._two_runs()).visibility_trend("tm", days=30)
        assert trend["present"] is True
        assert [p["probe_run_id"] for p in trend["points"]] == ["r1", "r2"]
        assert trend["points"][0]["avg_position"] == 5
        assert trend["points"][1]["avg_position"] == 2
        assert trend["points"][0]["share_of_voice"] == 0.5

    def test_change_compares_the_ends_of_the_window(self):
        trend = _FakeService(self._two_runs()).visibility_trend("tm", days=30)
        # rank 5 -> rank 2 is an IMPROVEMENT, so the delta is negative
        assert trend["change"]["avg_position"] == -3
        assert trend["change"]["runs_compared"] == 2

    def test_change_is_none_when_there_is_nothing_to_compare(self):
        rows = [_row(probe_run_id="r1", mentioned=False)]
        trend = _FakeService(rows).visibility_trend("tm", days=30)
        assert trend["change"]["avg_position"] is None

    def test_no_history_is_absent_not_a_flat_line_at_zero(self):
        trend = _FakeService([]).visibility_trend("tm", days=30)
        assert trend["present"] is False
        assert trend["points"] == []

    def test_a_capped_window_says_so(self):
        """A truncated aggregate that looks complete is the silent-zero shape."""
        trend = _FakeService(self._two_runs(), truncated=True).visibility_trend("tm", days=30)
        assert trend["truncated"] is True


# ────────────────────────────────────────────────────────────────────────────
# A4 — share of voice
# ────────────────────────────────────────────────────────────────────────────

class TestShareOfVoiceIncludesTheSubject:
    def _rows(self):
        return [
            _row(probe_run_id="r1", run_at="2026-08-01T00:00:00+00:00",
                 mentioned=True, competitors_mentioned=["Rival A"]),
            _row(probe_run_id="r1", run_at="2026-08-01T00:00:01+00:00",
                 mentioned=False, competitors_mentioned=["Rival A", "Rival B"]),
        ]

    def test_the_subject_has_a_share_of_its_own_voice(self):
        out = _FakeService(self._rows()).share_of_voice_series(
            "tm", subject_label="Flobali", days=30,
        )
        assert out["totals"]["subject_mentions"] == 1
        # 1 of us + 3 competitor namings = 4 named brands
        assert out["totals"]["subject_share_of_named_brands"] == pytest.approx(0.25)
        assert out["totals"]["subject_share_of_probes"] == pytest.approx(0.5)

    def test_it_is_bucketed_over_time_not_one_collapsed_total(self):
        rows = self._rows() + [
            _row(probe_run_id="r2", run_at="2026-08-08T00:00:00+00:00",
                 mentioned=True, competitors_mentioned=["Rival A"]),
        ]
        out = _FakeService(rows).share_of_voice_series("tm", subject_label="Flobali", days=30)
        assert [b["probe_run_id"] for b in out["buckets"]] == ["r1", "r2"]
        assert out["buckets"][1]["subject_mentions"] == 1

    def test_competitors_are_still_counted(self):
        out = _FakeService(self._rows()).share_of_voice_series(
            "tm", subject_label="Flobali", days=30,
        )
        names = {c["name"]: c["count"] for c in out["totals"]["competitor_mentions"]}
        assert names == {"Rival A": 2, "Rival B": 1}

    def test_the_route_applies_the_days_parameter(self):
        """It was declared, `ge=1, le=180`-validated, and then never used."""
        src = _ROUTES.read_text(encoding="utf-8")
        fn = src[src.index("async def share_of_voice("):]
        fn = fn[:fn.index("@router.post")]
        assert "days=days" in fn, "share_of_voice accepts `days` and must pass it down"
        assert "share_of_voice_series" in fn
        # The old body hand-rolled the query right here and pinned it at 500 rows.
        assert ".limit(500)" not in fn

    def test_the_route_hands_the_subject_label_down(self):
        src = _ROUTES.read_text(encoding="utf-8")
        fn = src[src.index("async def share_of_voice("):]
        fn = fn[:fn.index("@router.post")]
        assert "subject_label=" in fn


# ────────────────────────────────────────────────────────────────────────────
# A7 — a tier change is an INSTRUMENT change
# ────────────────────────────────────────────────────────────────────────────

class TestTrendRefusesToCompareAcrossTiers:
    """A frontier run and a cheap run are not two readings of the same thing.

    Different models give different answers, so a subject moved between tiers gets a step
    in its trend line that is an artefact of the instrument, not of its visibility. Both
    numbers are valid; the comparison is not. Nothing about the shape of the data says so,
    which is why it has to be derived and stated.
    """

    def _cheap_then_frontier(self):
        return [
            _row(probe_run_id="r2", run_at="2026-08-08T00:00:00+00:00",
                 model="claude-opus-5", mentioned=True, position=1),
            _row(probe_run_id="r1", run_at="2026-08-01T00:00:00+00:00",
                 model="claude-haiku-4-5-20251001", mentioned=True, position=5),
        ]

    def test_each_point_records_which_models_measured_it(self):
        trend = _FakeService(self._cheap_then_frontier()).visibility_trend("tm", days=30)
        assert trend["points"][0]["models"] == ["claude-haiku-4-5-20251001"]
        assert trend["points"][1]["models"] == ["claude-opus-5"]

    def test_the_first_point_after_a_change_is_flagged(self):
        trend = _FakeService(self._cheap_then_frontier()).visibility_trend("tm", days=30)
        assert trend["points"][0]["comparable_with_previous"] is True   # nothing before it
        assert trend["points"][1]["comparable_with_previous"] is False

    def test_change_is_withheld_when_the_instrument_changed(self):
        """Rank 5 -> rank 1 looks like a triumph. It is a different model answering."""
        trend = _FakeService(self._cheap_then_frontier()).visibility_trend("tm", days=30)
        assert trend["model_set_changed"] is True
        assert trend["change"]["avg_position"] is None
        assert trend["change"]["share_of_voice"] is None
        # A dash with no reason is indistinguishable from missing data.
        assert trend["change"]["not_comparable_reason"]

    def test_a_stable_model_set_still_gets_its_answer(self):
        rows = [
            _row(probe_run_id="r2", run_at="2026-08-08T00:00:00+00:00",
                 model="claude-haiku-4-5-20251001", mentioned=True, position=2),
            _row(probe_run_id="r1", run_at="2026-08-01T00:00:00+00:00",
                 model="claude-haiku-4-5-20251001", mentioned=True, position=5),
        ]
        trend = _FakeService(rows).visibility_trend("tm", days=30)
        assert trend["model_set_changed"] is False
        assert trend["change"]["avg_position"] == -3
        assert trend["change"]["not_comparable_reason"] is None


class TestProbeTierIsWired:
    def test_the_tiers_name_real_models_and_do_not_overlap(self):
        src = _SERVICE.read_text(encoding="utf-8")
        block = src[src.index("TIER_MODELS"):src.index("# ─", src.index("TIER_MODELS"))]
        assert "CHEAP_TIER" in block and "FRONTIER_TIER" in block
        # OpenAI is deliberately absent from the frontier tier: no chat model of theirs is
        # priced, and gpt-4o-mini has failed every probe it has ever run. Slice to the
        # frontier LIST only — reading to end-of-block would pick GPT4O_MINI back up out
        # of the cheap tier and pass for the wrong reason.
        frontier = block[block.index("FRONTIER_TIER:"):]
        frontier = frontier[:frontier.index("]")]
        assert "GPT4O_MINI" not in frontier
        assert "OPUS" in frontier

    def test_every_frontier_model_has_a_cost_entry(self):
        """An unpriced model is costed by a conservative default that is right for a cheap
        model and wrong by an order of magnitude for a frontier one."""
        src = _SERVICE.read_text(encoding="utf-8")
        cost_table = src[src.index("COST_TABLE"):src.index("# ─", src.index("COST_TABLE"))]
        for const in ("OPUS", "GEMINI_PRO", "SONAR_PRO"):
            assert f"{const}: {{" in cost_table, f"{const} missing from COST_TABLE"

    def test_the_probe_reports_what_the_tier_could_not_run(self):
        """A frontier run that silently drops to one model is still billed as frontier."""
        src = _SERVICE.read_text(encoding="utf-8")
        assert '"models_unavailable": unavailable' in src
        assert '"failed_calls": failed' in src

    def test_the_frontier_tier_costs_more_credits_than_the_cheap_one(self):
        logger_src = (_SERVICE.parent.parent / "integrations" / "mention_cost_logger.py").read_text(encoding="utf-8")
        assert '"probe_llm_frontier"' in logger_src
        routes = _ROUTES.read_text(encoding="utf-8")
        # Both metered doors, not just one — the cheaper door is the one that gets used.
        assert routes.count('"probe_llm_frontier" if') == 2

    def test_probe_cost_is_resolved_from_ai_model_pricing_not_a_guess(self):
        logger_src = (_SERVICE.parent.parent / "integrations" / "mention_cost_logger.py").read_text(encoding="utf-8")
        fn = logger_src[logger_src.index("def log_llm_probe_call"):]
        fn = fn[:fn.index("def log_youtube_call")]
        assert "AIPricingConfig.calculate_cost" in fn
        # The old ladder ended in a hardcoded pair for "an unrecognised probe model".
        # Comments are stripped first: the replacement's own comment NAMES that pair to
        # explain what it removed, and a guard that reads prose would fail on the fix.
        code = chr(10).join(
            line for line in fn.splitlines() if not line.strip().startswith("#")
        )
        assert "(0.0005, 0.0015)" not in code


# ────────────────────────────────────────────────────────────────────────────
# Wiring — the probe has to be TOLD which domain is ours
# ────────────────────────────────────────────────────────────────────────────

class TestHomepageDomainReachesEveryProbe:
    def test_every_probe_call_site_passes_it(self):
        """`brand_cited` is NULL without it, so a call site that forgets silently
        turns ghost-citation detection off for that path only."""
        offenders = []
        for path in (
            _ROUTES,
            _ROUTES.parent / "mention_tracking_routes.py",
        ):
            src = path.read_text(encoding="utf-8")
            for m in re.finditer(r"\.probe\(", src):
                # crude but sufficient: the kwargs of one call sit before its `)\n`
                tail = src[m.end():m.end() + 600]
                call = tail[:tail.index("\n        )") if "\n        )" in tail else len(tail)]
                if "homepage_domain=" not in call:
                    offenders.append(f"{path.name}:{src[:m.start()].count(chr(10)) + 1}")
        assert not offenders, (
            "probe() call sites not passing homepage_domain: " + ", ".join(offenders)
        )
