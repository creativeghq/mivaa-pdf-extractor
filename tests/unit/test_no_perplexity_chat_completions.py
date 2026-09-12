"""Perplexity is called on `/v1/agent`, never on the retired `/chat/completions`.

`POST https://api.perplexity.ai/chat/completions` stops answering on 2026-09-27 whatever
the account balance is (#400 W1). The four callers moved on 2026-09-12; this fails the
build if the old shape is pasted back, and covers the four ways the move fails silently.
"""
import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
CLIENT = APP / "services" / "integrations" / "perplexity_agent_client.py"

#: The file AND the one function in it that builds the Perplexity call. Three of these
#: files talk to four providers apiece, so a whole-file text match would fail on
#: OpenAI's own `prompt_tokens` and on a Firecrawl `raise_for_status`.
CALLERS = {
    APP / "services" / "integrations" / "perplexity_price_search_service.py": "_perplexity_call",
    APP / "services" / "integrations" / "job_search_service.py": "search_via_perplexity",
    APP / "services" / "integrations" / "mention_search_service.py": "_search_perplexity",
    APP / "services" / "integrations" / "llm_mention_probe_service.py": "_call_perplexity",
}

RETIRED = "api.perplexity.ai/chat/completions"


def _blank_comments(source: str) -> str:
    """Comments and docstrings blanked, so a rule can only be satisfied by code."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from comment_budget import blank_comments
        return blank_comments(source)
    finally:
        sys.path.pop(0)


def _function_source(path: Path, name: str) -> str:
    """Just that function's body, comments blanked."""
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _blank_comments(ast.get_source_segment(source, node) or "")
    raise AssertionError(f"{path.name} no longer defines {name}() — rename the guard with it")


def _load_client():
    spec = importlib.util.spec_from_file_location("perplexity_agent_client", CLIENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───── the retired endpoint ─────

def test_no_file_calls_the_retired_endpoint():
    offenders = []
    for path in APP.rglob("*.py"):
        if RETIRED in _blank_comments(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        f"{RETIRED} stops answering 2026-09-27; these still call it: {offenders}"
    )


def test_every_caller_goes_through_the_one_client():
    for path in CALLERS:
        body = _blank_comments(path.read_text(encoding="utf-8"))
        assert "perplexity_agent_client" in body, (
            f"{path.name} does not import the Agent API client — a second copy of the "
            "envelope walk is a second place for the silent-zero shapes to come back"
        )
        assert "call_agent(" in _function_source(path, CALLERS[path])


def test_the_client_posts_to_the_agent_endpoint():
    assert _load_client().AGENT_API_URL == "https://api.perplexity.ai/v1/agent"


# ───── W1e-1: nothing searches unless the tool is asked for ─────

def test_a_request_that_wants_search_carries_the_web_search_tool():
    m = _load_client()
    body = m.build_agent_body(
        tier=m.TIER_SONAR, input_text="anything", tools=[m.web_search_tool()]
    )
    assert body["tools"][0]["type"] == "web_search", (
        "without the tool the Agent API answers ungrounded, uncited and HTTP 200 — "
        "which reads downstream as zero hits rather than as a failure"
    )


def test_every_grounded_caller_asks_for_web_search():
    for path, fn in CALLERS.items():
        assert "web_search_tool(" in _function_source(path, fn), (
            f"{path.name}:{fn} builds an Agent API request with no web_search tool"
        )


# ───── W1e-2: a failed run arrives as HTTP 200 ─────

def test_a_failed_run_on_http_200_is_not_ok():
    m = _load_client()
    reply = m.parse_agent_reply(
        {"status": "failed", "error": {"message": "model unavailable"}, "output": []},
        http_status=200,
    )
    assert reply.ok is False, "status=failed on HTTP 200 must never read as a success"
    assert "model unavailable" in (reply.error or "")


def test_an_incomplete_run_names_its_reason():
    m = _load_client()
    reply = m.parse_agent_reply(
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        },
        http_status=200,
    )
    assert reply.ok is False
    assert "max_output_tokens" in (reply.error or "")


def test_a_body_with_no_status_is_not_ok():
    m = _load_client()
    assert m.parse_agent_reply({"output": []}, http_status=200).ok is False


def test_no_caller_decides_success_from_the_http_code_alone():
    for path, fn in CALLERS.items():
        body = _function_source(path, fn)
        assert "reply.ok" in body, (
            f"{path.name}:{fn} does not branch on reply.ok — an Agent API failure is an "
            "HTTP 200, so a status-code check books it as a clean run with no results"
        )
        assert "raise_for_status" not in body, (
            f"{path.name}:{fn} still gates the Perplexity call on the HTTP status"
        )
        assert "status_code" not in body, (
            f"{path.name}:{fn} still reads an HTTP status code directly"
        )


# ───── W1e-3: the token keys were renamed ─────

def test_usage_is_read_from_the_new_token_keys():
    m = _load_client()
    reply = m.parse_agent_reply(
        {
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 4718, "output_tokens": 450},
        },
        http_status=200,
    )
    assert (reply.input_tokens, reply.output_tokens) == (4718, 450)


def test_the_old_token_keys_are_not_silently_accepted():
    m = _load_client()
    reply = m.parse_agent_reply(
        {
            "status": "completed",
            "output": [],
            "usage": {"prompt_tokens": 4718, "completion_tokens": 450},
        },
        http_status=200,
    )
    assert (reply.input_tokens, reply.output_tokens) == (0, 0), (
        "reading the retired keys would book a real call at zero tokens, and a real "
        "call at $0 is exactly the silent zero anti-regression rule 2 describes"
    )


def test_no_caller_still_reads_the_old_token_keys():
    for path, fn in CALLERS.items():
        body = _function_source(path, fn)
        for retired in ("prompt_tokens", "completion_tokens", "choices"):
            assert retired not in body, f"{path.name}:{fn} still reads the retired {retired!r}"


# ───── W1e-4: citations moved inside output[] ─────

def test_citations_are_read_from_the_search_results_item():
    m = _load_client()
    reply = m.parse_agent_reply(
        {
            "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "output": [
                {
                    "type": "search_results",
                    "queries": ["materialshub"],
                    "results": [
                        {"id": 1, "title": "A", "url": "https://a.example/x"},
                        {"id": 2, "title": "B", "url": "https://b.example/y"},
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "They are mentioned."}],
                },
            ],
        },
        http_status=200,
    )
    assert reply.ok is True
    assert reply.text == "They are mentioned."
    assert reply.citation_urls == ["https://a.example/x", "https://b.example/y"]
    assert len(reply.search_results) == 2


def test_a_top_level_citations_array_is_no_longer_where_we_look():
    m = _load_client()
    reply = m.parse_agent_reply(
        {
            "status": "completed",
            "citations": ["https://legacy.example/z"],
            "search_results": [{"url": "https://legacy.example/w"}],
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
            ],
        },
        http_status=200,
    )
    assert reply.citation_urls == [], (
        "the legacy top-level arrays are dead; reading them yields an empty list with "
        "no error, which the probe reports as 'not mentioned'"
    )


def test_message_annotations_also_count_as_citations():
    m = _load_client()
    reply = m.parse_agent_reply(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "x",
                            "annotations": [{"url": "https://cited.example/a"}],
                        }
                    ],
                }
            ],
        },
        http_status=200,
    )
    assert reply.citation_urls == ["https://cited.example/a"]


# ───── the request contract ─────

def test_sonar_pro_is_not_sent_as_a_model_slug():
    m = _load_client()
    body = m.build_agent_body(tier=m.TIER_SONAR_PRO, input_text="q")
    assert body.get("model") == m.SONAR_MODEL
    assert body.get("preset") == "low", (
        "perplexity/sonar-pro does not exist on the Agent API; the deep tier is the "
        "`low` preset with the model pinned back to Sonar"
    )


def test_the_cheap_tier_is_plain_sonar():
    m = _load_client()
    body = m.build_agent_body(tier=m.TIER_SONAR, input_text="q")
    assert body == {"model": m.SONAR_MODEL, "input": "q"}


def test_the_retired_sonar_field_names_are_never_emitted():
    m = _load_client()
    body = m.build_agent_body(
        tier=m.TIER_SONAR,
        input_text="q",
        instructions="sys",
        max_output_tokens=800,
        response_schema={"type": "object"},
        tools=[m.web_search_tool(domains=["a.com"], recency="week", country="gr")],
    )
    for retired in ("messages", "max_tokens", "web_search_options", "search_domain_filter",
                    "search_recency_filter", "choices"):
        assert retired not in body, (
            f"strict mode rejects unknown fields with a 400 — {retired!r} must not sit "
            "alongside its replacement"
        )
    assert body["input"] == "q"
    assert body["instructions"] == "sys"
    assert body["max_output_tokens"] == 800


def test_the_search_filters_live_on_the_tool():
    m = _load_client()
    tool = m.web_search_tool(
        domains=["https://www.Example.com/", "-spam.com", "example.com"],
        recency="week",
        country="gr",
        context_size="high",
        max_results=10,
    )
    assert tool["filters"]["search_domain_filter"] == ["example.com", "-spam.com"]
    assert tool["filters"]["search_recency_filter"] == "week"
    assert tool["user_location"] == {"country": "GR"}
    assert tool["max_results"] == 10
    assert tool["search_context_size"] == "high", (
        "verified 2026-09-12: search_context_size is still a string enum on the tool, "
        "not the number the migration notes predicted"
    )


def test_the_domain_cap_is_twenty_not_ten():
    m = _load_client()
    assert m.DOMAIN_FILTER_MAX == 20
    assert len(m.clean_domains([f"d{i}.com" for i in range(40)])) == 20


def test_a_bad_enum_is_refused_here_rather_than_by_a_400():
    m = _load_client()
    for kwargs in ({"context_size": "very-high"}, {"recency": "fortnight"}, {"max_results": 99}):
        try:
            m.web_search_tool(**kwargs)
        except m.PerplexityContractError:
            continue
        raise AssertionError(f"web_search_tool accepted {kwargs}, which the API rejects")


def test_an_unknown_tier_is_refused():
    m = _load_client()
    try:
        m.build_agent_body(tier="sonar-reasoning-pro", input_text="q")
    except m.PerplexityContractError:
        return
    raise AssertionError("an unmapped tier must not silently build a request")


def test_the_schema_name_is_forward_safe():
    m = _load_client()
    body = m.build_agent_body(
        tier=m.TIER_SONAR, input_text="q",
        response_schema={"type": "object"}, schema_name="price results!",
    )
    name = body["response_format"]["json_schema"]["name"]
    assert name == "price_results_" and 1 <= len(name) <= 64


def test_strict_is_opt_in():
    m = _load_client()
    loose = m.build_agent_body(tier=m.TIER_SONAR, input_text="q", response_schema={"type": "object"})
    assert "strict" not in loose["response_format"]["json_schema"]
    strict = m.build_agent_body(
        tier=m.TIER_SONAR, input_text="q", response_schema={"type": "object"}, schema_strict=True,
    )
    assert strict["response_format"]["json_schema"]["strict"] is True


# ───── cost ─────

def test_the_provider_reported_cost_is_carried_through():
    m = _load_client()
    reply = m.parse_agent_reply(
        {
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cost": {"input_cost": 0.00826, "output_cost": 0.0063, "total_cost": 0.01706},
                "tool_calls_details": {"search_web": {"invocation": 3}},
            },
        },
        http_status=200,
    )
    assert reply.reported_cost_usd == 0.01706
    assert reply.reported_input_cost_usd == 0.00826
    assert reply.reported_output_cost_usd == 0.0063
    assert reply.search_invocations == 3


def test_an_absent_cost_block_is_none_not_zero():
    m = _load_client()
    reply = m.parse_agent_reply({"status": "completed", "output": [], "usage": {}}, http_status=200)
    assert reply.reported_cost_usd is None, (
        "None means 'the provider did not say', which falls back to the rate table; "
        "0.0 would mean 'this call was free'"
    )
