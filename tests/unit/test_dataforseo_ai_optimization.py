"""Guard: the AI Optimization surface is complete, reachable, and sends the right field.

`ai_llm_response` shipped sending `prompt`. The field is `user_prompt`, so every call
returned 40501 and the method had never once worked - invisible because the only caller
was the agent dispatcher and a 40501 costs nothing, so it appears in no spend report.
"""

import ast
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
_CLIENT = _APP / "services" / "integrations" / "dataforseo_unified_client.py"
_PARSING = _APP / "services" / "integrations" / "dataforseo_ai_parsing.py"
_PROBE = _APP / "services" / "integrations" / "llm_mention_probe_service.py"
_ROUTES = _APP / "api" / "seo_agent_routes.py"


def _load(path, name):
    """Import WITHOUT `app` - CI installs pytest and nothing else, and
    `app.services.__init__` reaches for a Supabase client on import."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_p = _load(_PARSING, "dataforseo_ai_parsing")


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _blank_comments(src: str) -> str:
    """A marker satisfied by a comment is not satisfied. Docstrings too - this file
    asserts on CODE."""
    tree = ast.parse(src)
    spans = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                spans.add(ln)
    out = []
    for i, line in enumerate(src.split("\n"), start=1):
        if i in spans:
            out.append("")
        elif line.lstrip().startswith("#"):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


class TestTheFieldNameThatNeverWorked:
    def test_the_prompt_goes_out_as_user_prompt(self):
        src = _blank_comments(_source(_CLIENT))
        assert '"user_prompt": prompt[:PROMPT_MAX_CHARS]' in src

    def test_the_old_field_name_is_gone(self):
        src = _blank_comments(_source(_CLIENT))
        assert '{"prompt": prompt' not in src

    def test_the_500_char_cap_is_applied_not_just_documented(self):
        src = _blank_comments(_source(_CLIENT))
        assert "PROMPT_MAX_CHARS = 500" in src
        assert "prompt[:PROMPT_MAX_CHARS]" in src

    def test_web_search_is_on_by_default_or_there_are_no_citations(self):
        src = _blank_comments(_source(_CLIENT))
        assert "web_search: bool = True" in src
        assert '"web_search": web_search' in src

    def test_the_country_is_stated_rather_than_defaulted_to_the_us(self):
        src = _blank_comments(_source(_CLIENT))
        assert '"web_search_country_iso_code"' in src


class TestEveryPublishedEndpointHasAMethod:
    #: Every path under /v3/ai_optimization that DataForSEO bills for, from the live
    #: account limits payload. A gap here is a capability we silently do not have.
    PUBLISHED = [
        "/ai_optimization/ai_keyword_data/keywords_search_volume/live",
        "/ai_optimization/llm_mentions/search_mentions/live",
        "/ai_optimization/llm_mentions/target_metrics/live",
        "/ai_optimization/llm_mentions/multi_target_metrics/live",
        "/ai_optimization/llm_mentions/top_mentioned_pages/live",
        "/ai_optimization/llm_mentions/top_mentioned_domains/live",
        "/ai_optimization/llm_mentions/top_mentioned_brands/live",
        "/ai_optimization/llm_mentions/top_mentioned_brand_categories/live",
        "/ai_optimization/llm_mentions/target_metrics_lite/live",
        "/ai_optimization/llm_mentions/top_mentioned_pages_lite/live",
        "/ai_optimization/llm_mentions/top_mentioned_domains_lite/live",
        "/ai_optimization/llm_mentions/top_mentioned_brands_lite/live",
        "/ai_optimization/llm_mentions/top_mentioned_brand_categories_lite/live",
        "/ai_optimization/llm_mentions/historical/live",
        "/ai_optimization/llm_mentions/timeseries_delta/live",
        "/ai_optimization/llm_mentions/timeseries_new_lost/live",
        "/ai_optimization/chat_gpt/llm_responses/live",
        "/ai_optimization/claude/llm_responses/live",
        "/ai_optimization/gemini/llm_responses/live",
        "/ai_optimization/perplexity/llm_responses/live",
        "/ai_optimization/chat_gpt/llm_scraper/live",
        "/ai_optimization/gemini/llm_scraper/live",
    ]

    def test_every_one_is_reachable_from_the_client(self):
        src = _source(_CLIENT)
        missing = [p for p in self.PUBLISHED if p not in src]
        assert missing == [], f"AI Optimization endpoints with no client method: {missing}"

    def test_the_lite_tier_falls_back_rather_than_raising(self):
        src = _blank_comments(_source(_CLIENT))
        assert "def llm_mentions_path(" in src
        assert "if lite and key in LLM_MENTIONS_LITE" in src

    def test_the_new_methods_are_on_the_agent_allowlist(self):
        # A client method absent from _ALLOWED_METHODS returns 404 to the agent, so
        # it is unreachable however complete the client looks.
        src = _source(_ROUTES)
        for name in ("ai_llm_mentions_top_brands", "ai_llm_mentions_top_brand_categories",
                     "ai_llm_scraper", "ai_llm_mentions_locations_and_languages",
                     "ai_keyword_locations_and_languages"):
            assert f'"{name}"' in src, f"{name} is not on the agent allowlist"


class TestAnnotationParsing:
    RESULT = {
        "model_name": "gpt-4.1-mini",
        "input_tokens": 12, "output_tokens": 340, "reasoning_tokens": 0,
        "money_spent": 0.0021, "web_search": True,
        "fan_out_queries": ["tile suppliers thessaloniki", "best tiles greece"],
        "items": [{"type": "message", "sections": [
            {"type": "text", "text": "Kerablock is one option.", "annotations": [
                {"url": "https://a.gr/x", "title": "a.gr", "text": "Kerablock",
                 "start_index": 0, "end_index": 9},
                {"url": "https://a.gr/x", "title": "a.gr", "text": "again",
                 "start_index": 10, "end_index": 14},
            ]},
            {"type": "text", "text": "So is Tilehub.", "annotations": [
                {"url": "https://b.gr/y", "title": "b.gr"},
            ]},
        ]}],
    }

    def test_text_is_joined_across_sections(self):
        assert _p.response_text(self.RESULT) == "Kerablock is one option.\nSo is Tilehub."

    def test_annotations_are_read_off_the_section_not_the_item(self):
        # The trap: `items[].annotations` is always absent, so an extractor reading
        # the item finds nothing on every answer and citations stay empty forever.
        anns = _p.response_annotations(self.RESULT)
        assert len(anns) == 3
        assert anns[0]["start_index"] == 0 and anns[0]["end_index"] == 9

    def test_urls_dedupe_while_annotations_keep_every_span(self):
        assert _p.response_urls(self.RESULT) == ["https://a.gr/x", "https://b.gr/y"]
        assert len(_p.response_annotations(self.RESULT)) == 3

    def test_an_answer_with_web_search_off_yields_no_citations(self):
        off = {"items": [{"type": "message", "sections": [
            {"type": "text", "text": "From memory.", "annotations": None}]}]}
        assert _p.response_urls(off) == []
        assert _p.response_text(off) == "From memory."

    def test_usage_keeps_money_spent_separate_from_the_task_price(self):
        u = _p.response_usage(self.RESULT)
        assert u["input_tokens"] == 12 and u["output_tokens"] == 340
        assert u["money_spent"] == 0.0021
        assert u["web_search"] is True

    def test_fan_out_queries_survive_both_shapes(self):
        assert _p.response_fan_out_queries(self.RESULT) == [
            "tile suppliers thessaloniki", "best tiles greece"]
        assert _p.response_fan_out_queries({"fan_out_queries": [{"query": "a"}]}) == ["a"]
        assert _p.response_fan_out_queries({"fan_out_queries": None}) == []

    def test_lite_rows_arrive_as_a_bare_list(self):
        assert _p.mentions_rows([{"domain": "a.gr"}]) == [{"domain": "a.gr"}]
        assert _p.mentions_rows({"items": [{"domain": "b.gr"}]}) == [{"domain": "b.gr"}]
        assert _p.mentions_rows(None) == []

    def test_nothing_raises_on_a_shape_it_does_not_know(self):
        for bad in (None, {}, {"items": None}, {"items": ["str"]}):
            assert _p.response_text(bad) == ""
            assert _p.response_urls(bad) == []


class TestTheRouteIsExplicit:
    def test_dataforseo_engines_are_their_own_model_ids(self):
        src = _blank_comments(_source(_PROBE))
        assert 'DFS_PREFIX = "dfs:"' in src
        assert "DATAFORSEO_TIER: [DFS_CHAT_GPT, DFS_CLAUDE, DFS_GEMINI, DFS_PERPLEXITY]" in src

    def test_there_is_no_automatic_fallback_from_a_dead_vendor_key(self):
        # A silent swap would change WHICH surface answered while the row still said
        # ChatGPT - the same defect shape as a substituted embedding model.
        src = _blank_comments(_source(_PROBE))
        assert "except" not in src.split("async def _call_dataforseo")[1].split("async def")[0]

    def test_an_empty_answer_is_an_error_not_an_answer_naming_nobody(self):
        src = _blank_comments(_source(_PROBE))
        body = src.split("async def _call_dataforseo")[1].split("    # ")[0]
        assert "dataforseo returned no answer text" in body


class TestTheFlattenedItemsTrap:
    #: What DataForSEO actually returns. The client's _call() hoists any result with an
    #: `items` key, so reading result.items[0] hands the parser a MESSAGE item and the
    #: answer comes back empty on a perfectly good response.
    RAW = {
        "tasks": [{"status_code": 20000, "result": [{
            "model_name": "gpt-4.1-mini", "input_tokens": 9, "output_tokens": 120,
            "money_spent": 0.0018, "web_search": True,
            "items": [{"type": "message", "sections": [
                {"type": "text", "text": "Kerablock supplies tiles.", "annotations": [
                    {"url": "https://kerablock.gr/", "title": "kerablock.gr"}]},
            ]}],
        }]}],
    }

    def test_the_result_object_is_read_from_raw(self):
        r = _p.first_ai_result(self.RAW)
        assert r["model_name"] == "gpt-4.1-mini"
        assert _p.response_text(r) == "Kerablock supplies tiles."
        assert _p.response_urls(r) == ["https://kerablock.gr/"]
        assert _p.response_usage(r)["output_tokens"] == 120

    def test_parsing_the_flattened_item_instead_loses_the_answer(self):
        # Pins WHY the code reads raw: this is what result.items[0] hands you.
        flattened = self.RAW["tasks"][0]["result"][0]["items"][0]
        assert _p.response_text(flattened) == ""
        assert _p.response_usage(flattened)["output_tokens"] == 0

    def test_the_probe_reads_raw_not_items(self):
        src = _blank_comments(_source(_PROBE))
        assert "first_ai_result(result.raw)" in src
        assert "(result.items or [{}])[0]" not in src

    def test_an_envelope_with_no_result_does_not_raise(self):
        for bad in (None, {}, {"tasks": []}, {"tasks": [{}]}, {"tasks": [{"result": None}]}):
            assert _p.first_ai_result(bad) == {}


class TestTheCountryReachesTheCall:
    def test_probe_takes_a_country_and_hands_it_down(self):
        src = _blank_comments(_source(_PROBE))
        assert "country_code: Optional[str] = None,\n        tier: str = CHEAP_TIER," in src
        assert "country_code=country_code)" in src

    def test_the_tier_is_storable_not_just_declared(self):
        # Declared in the enum and refused by the CHECK is the vocabulary-wider-than-
        # the-constraint trap: it passes validation and dies as a raw 23514.
        routes = _source(_APP / "api" / "mention_monitoring_routes.py")
        service = _source(_APP / "services" / "integrations" / "tracked_mentions_service.py")
        assert 'pattern="^(cheap|frontier|dataforseo|scraper)$"' in routes
        assert '"cheap", "frontier", "dataforseo", "scraper"' in service


class TestTheScrapedConsumerSurface:
    #: The real shape, from the sandbox on 2026-09-21. Three different spellings of
    #: "the answer" across one vendor: sections (llm_responses), markdown (scraper).
    SCRAPED = {
        "keyword": "best tile suppliers in Thessaloniki",
        "check_url": "https://chatgpt.com/?prompt=x&hints=search",
        "markdown": "Kerablock and Tilehub both supply Thessaloniki.",
        "sources": [
            {"type": "chat_gpt_source", "title": "Kerablock", "domain": "kerablock.gr",
             "url": "https://kerablock.gr/tiles?utm_source=chatgpt.com"},
            {"type": "chat_gpt_source", "title": "Kerablock", "domain": "kerablock.gr",
             "url": "https://kerablock.gr/tiles"},
            {"type": "chat_gpt_source", "title": "Tilehub", "domain": "tilehub.gr",
             "url": "https://tilehub.gr/"},
        ],
        "brand_entities": [
            {"type": "chat_gpt_brand_entity", "title": "Kerablock", "category": "company"},
            {"type": "chat_gpt_brand_entity", "title": "Tilehub", "category": "company"},
        ],
        "items": [{"type": "chat_gpt_text", "markdown": "Kerablock and Tilehub both supply Thessaloniki.",
                   "sources": [], "brand_entities": []}],
    }

    def test_the_answer_is_markdown_not_text(self):
        # scraper_text originally read items[].text. The field is `markdown`, so every
        # scraped answer would have come back empty and recorded as a failed probe.
        assert _p.scraper_text(self.SCRAPED) == "Kerablock and Tilehub both supply Thessaloniki."
        assert _p.scraper_text({"items": [{"markdown": "block one"}]}) == "block one"
        assert _p.scraper_text({}) is None

    def test_a_tracking_parameter_does_not_duplicate_a_source(self):
        # Every ChatGPT source carries ?utm_source=chatgpt.com; a raw dedupe keeps the
        # same page twice the moment one copy arrives without it.
        urls = _p.scraper_source_urls(self.SCRAPED)
        assert len(urls) == 2
        assert urls[0].startswith("https://kerablock.gr/tiles")
        assert "https://tilehub.gr/" in urls

    def test_brands_come_from_the_scraper_not_a_re_read(self):
        assert _p.scraper_brands(self.SCRAPED) == ["Kerablock", "Tilehub"]
        assert _p.scraper_brands({}) == []

    def test_the_check_url_is_kept_so_an_answer_can_be_reproduced(self):
        assert _p.scraper_check_url(self.SCRAPED).startswith("https://chatgpt.com/")
        assert _p.scraper_check_url({}) is None

    def test_nothing_raises_on_a_shape_it_does_not_know(self):
        for bad in (None, {}, {"sources": None}, {"items": ["str"]}, {"brand_entities": [1]}):
            assert _p.scraper_source_urls(bad) == []
            assert _p.scraper_brands(bad) == []


class TestTheScrapeRouteIsItsOwnSurface:
    def test_scraped_engines_are_their_own_model_ids(self):
        src = _blank_comments(_source(_PROBE))
        assert 'SCRAPE_PREFIX = "scrape:"' in src
        assert "SCRAPER_TIER: [SCRAPE_CHAT_GPT, SCRAPE_GEMINI]" in src

    def test_only_the_two_scrapeable_surfaces_are_offered(self):
        # llm_scraper has no claude or perplexity endpoint; listing one would be a
        # tier entry the client refuses.
        src = _blank_comments(_source(_PROBE))
        tier = src.split("SCRAPER_TIER: [")[1].split("]")[0]
        assert "CLAUDE" not in tier and "PERPLEXITY" not in tier

    def test_it_bills_no_tokens_because_nothing_was_generated(self):
        src = _blank_comments(_source(_PROBE))
        body = src.split("async def _call_llm_scraper")[1].split("    # ")[0]
        assert "ModelReply(text, 0, 0, latency, None, scraper_source_urls(row))" in body

    def test_it_reads_raw_like_the_other_dataforseo_route(self):
        src = _blank_comments(_source(_PROBE))
        body = src.split("async def _call_llm_scraper")[1].split("    # ")[0]
        assert "first_ai_result(result.raw)" in body
