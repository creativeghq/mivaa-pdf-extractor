"""A probe that cannot cite reports zero citations forever.

Every live engine answered from training memory until 2026-09-20, so
`llm_mention_probes.cited_urls` was empty on all 433 successful probes and the panel's
"answers citing you" tile read "Not captured" from the day it shipped. These tests pin
the two halves that made it structurally impossible: asking for sources, and reading
the channel each provider returns them on.
"""

import importlib.util
from pathlib import Path

_MATH = (Path(__file__).resolve().parents[2] / "app" / "services" / "integrations"
         / "llm_visibility_math.py")


def _load_math():
    """Load the derivations WITHOUT importing `app` — CI installs pytest and nothing
    else, and `app.services.__init__` reaches for a Supabase client on import."""
    spec = importlib.util.spec_from_file_location("llm_visibility_math", _MATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vm = _load_math()
anthropic_citation_urls = _vm.anthropic_citation_urls
anthropic_search_tool_type = _vm.anthropic_search_tool_type
gemini_citation_urls = _vm.gemini_citation_urls


class TestAnthropicSearchToolType:
    def test_haiku_gets_the_basic_variant(self):
        assert anthropic_search_tool_type("claude-haiku-4-5-20251001") == "web_search_20250305"

    def test_opus_5_gets_dynamic_filtering(self):
        assert anthropic_search_tool_type("claude-opus-5") == "web_search_20260209"

    def test_an_unknown_model_falls_back_to_the_variant_every_model_accepts(self):
        assert anthropic_search_tool_type("some-future-model") == "web_search_20250305"
        assert anthropic_search_tool_type("") == "web_search_20250305"

    def test_the_cutoff_is_the_generation_not_a_list_of_ids(self):
        # A list of ids goes stale the day a model ships and the failure is a 400.
        assert anthropic_search_tool_type("claude-opus-4-6") == "web_search_20260209"
        assert anthropic_search_tool_type("claude-opus-4-5") == "web_search_20250305"
        assert anthropic_search_tool_type("claude-sonnet-4-6") == "web_search_20260209"
        assert anthropic_search_tool_type("claude-fable-5-1") == "web_search_20260209"

    def test_a_model_newer_than_this_code_still_gets_the_newer_variant(self):
        assert anthropic_search_tool_type("claude-opus-9") == "web_search_20260209"

    def test_the_trailing_date_is_not_read_as_a_version(self):
        # claude-haiku-4-5-20251001 is 4.5, not 4.5.20251001.
        assert anthropic_search_tool_type("claude-haiku-4-5-20251001") == "web_search_20250305"


class TestAnthropicCitations:
    def test_reads_both_searched_and_cited_urls(self):
        blocks = [
            {"type": "web_search_tool_result", "content": [
                {"type": "web_search_result", "url": "https://a.com/x"},
                {"type": "web_search_result", "url": "https://b.com/y"},
            ]},
            {"type": "text", "text": "...", "citations": [
                {"type": "web_search_result_location", "url": "https://c.com/z"},
            ]},
        ]
        assert anthropic_citation_urls(blocks) == [
            "https://a.com/x", "https://b.com/y", "https://c.com/z",
        ]

    def test_a_failed_search_returns_an_error_object_not_a_list(self):
        # The trap: same block type, `content` is a dict. Iterating it walks the
        # error's KEYS, so a failed search used to look like citations.
        blocks = [{"type": "web_search_tool_result",
                   "content": {"type": "web_search_tool_result_error",
                               "error_code": "max_uses_exceeded"}}]
        assert anthropic_citation_urls(blocks) == []

    def test_a_memory_only_answer_cites_nothing(self):
        assert anthropic_citation_urls([{"type": "text", "text": "Some brands are..."}]) == []

    def test_survives_a_shape_it_does_not_know(self):
        assert anthropic_citation_urls(None) == []
        assert anthropic_citation_urls(["not a block"]) == []


class TestGeminiCitations:
    def test_resolves_the_publisher_not_googles_redirect(self):
        candidate = {"groundingMetadata": {"groundingChunks": [
            {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123",
                     "title": "peptidesciences.com"}},
        ]}}
        assert gemini_citation_urls(candidate) == ["https://peptidesciences.com"]

    def test_the_redirect_host_never_becomes_the_cited_domain(self):
        # Storing the uri makes every Gemini citation read as Google, and the
        # subject's own domain can then never match — brand_cited stays false forever.
        candidate = {"groundingMetadata": {"groundingChunks": [
            {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/X",
                     "title": "materialshub.gr"}},
        ]}}
        assert "vertexaisearch" not in gemini_citation_urls(candidate)[0]

    def test_keeps_the_uri_when_the_title_is_not_a_host(self):
        candidate = {"groundingMetadata": {"groundingChunks": [
            {"web": {"uri": "https://example.com/page", "title": "A Long Article Title"}},
        ]}}
        assert gemini_citation_urls(candidate) == ["https://example.com/page"]

    def test_an_ungrounded_answer_cites_nothing(self):
        assert gemini_citation_urls({"content": {"parts": [{"text": "hi"}]}}) == []
        assert gemini_citation_urls(None) == []


anthropic_search_failure = _vm.anthropic_search_failure


class TestAFailedSearchIsNotASourcelessAnswer:
    OK_BLOCK = {"type": "web_search_tool_result", "content": [
        {"type": "web_search_result", "url": "https://a.gr/x"}]}
    ERR_BLOCK = {"type": "web_search_tool_result", "content": {
        "type": "web_search_tool_result_error", "error_code": "unavailable"}}

    def test_a_search_that_never_ran_is_reported(self):
        # The model still answers, from memory, so the call looks successful and the
        # probe would record "answered, no sources" — unknown rendered as zero.
        assert anthropic_search_failure([self.ERR_BLOCK]) == "web_search failed: unavailable"

    def test_a_stop_after_a_good_search_is_not_a_failure(self):
        # max_uses_exceeded with results already read is a stop, not a failure; the
        # citations from those searches are real and must not be thrown away.
        blocks = [self.OK_BLOCK,
                  {"type": "web_search_tool_result",
                   "content": {"error_code": "max_uses_exceeded"}}]
        assert anthropic_search_failure(blocks) is None

    def test_a_clean_search_reports_nothing(self):
        assert anthropic_search_failure([self.OK_BLOCK]) is None

    def test_an_answer_that_never_searched_is_not_a_search_failure(self):
        assert anthropic_search_failure([{"type": "text", "text": "hi"}]) is None
        assert anthropic_search_failure([]) is None
        assert anthropic_search_failure(None) is None

    def test_the_probe_records_it_rather_than_discarding_it(self):
        import re
        from pathlib import Path
        src = (Path(__file__).resolve().parents[2] / "app" / "services" / "integrations"
               / "llm_mention_probe_service.py").read_text(encoding="utf-8")
        body = src.split("async def _call_anthropic")[1].split("async def")[0]
        assert "anthropic_search_failure(blocks)" in body
        assert not re.search(r"\n\s+None,\n\s+dedupe_urls\(anthropic_citation_urls", body)
