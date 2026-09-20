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
