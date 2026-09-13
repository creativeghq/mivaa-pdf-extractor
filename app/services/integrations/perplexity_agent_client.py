"""Perplexity Agent API (`POST /v1/agent`) — the ONE client all four callers use.

`/chat/completions` stops answering 2026-09-27 whatever the balance is. The envelope walk,
the `status` branch and the usage keys are identical for every caller, so they live here
once: four copies would be four places for this migration's silent-zero shapes (no search,
HTTP-200 failure, renamed token keys, moved citations) to reappear independently.
Contract verified 2026-09-12 against docs.perplexity.ai/docs/agent-api.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# httpx is imported inside `call_agent`, not here: everything above it is pure contract,
# and MIVAA's CI installs pytest alone. A module-level provider import would put the
# whole file out of reach of the guard tests that load it by path.

logger = logging.getLogger(__name__)


AGENT_API_URL = "https://api.perplexity.ai/v1/agent"

#: The one Perplexity-family slug on the Agent API. `perplexity/sonar-pro` DOES NOT EXIST.
SONAR_MODEL = "perplexity/sonar"

#: Legacy tier names, kept verbatim: `ai_model_pricing`, `job_listings.source`,
#: `tracked_jobs.sources_enabled` and every stored row still say these (issue #400 W1f).
TIER_SONAR = "sonar"
TIER_SONAR_PRO = "sonar-pro"

#: The deep tier has no Perplexity-native model, so it is the `low` preset with the model
#: pinned back to Sonar — the probe must stay Perplexity's own answer, not the preset's
#: default OpenAI one. Documented as valid; it is the first thing the live smoke checks.
_TIER_ROUTING: Dict[str, Dict[str, Any]] = {
    TIER_SONAR: {"model": SONAR_MODEL},
    TIER_SONAR_PRO: {"preset": "low", "model": SONAR_MODEL},
}

#: Was 10 on `/chat/completions`. The Agent API takes 20.
DOMAIN_FILTER_MAX = 20

_CONTEXT_SIZES = {"low", "medium", "high"}
_RECENCY_VALUES = {"hour", "day", "week", "month", "year"}


class PerplexityContractError(ValueError):
    """A request this client would build that the Agent API is known to reject."""


@dataclass
class AgentReply:
    """One Agent API answer, already reduced to what a caller needs.

    `ok` is False for a transport failure, a non-2xx, AND a 200 carrying
    `status != "completed"` — the last is the shape that would otherwise be booked as a
    successful call returning nothing.
    """

    ok: bool
    status: str = ""
    text: str = ""
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    citation_urls: List[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    reported_cost_usd: Optional[float] = None
    reported_input_cost_usd: Optional[float] = None
    reported_output_cost_usd: Optional[float] = None
    search_invocations: int = 0
    latency_ms: int = 0
    http_status: Optional[int] = None
    error: Optional[str] = None


def web_search_tool(
    *,
    domains: Optional[List[str]] = None,
    recency: Optional[str] = None,
    country: Optional[str] = None,
    context_size: Optional[str] = None,
    max_results: Optional[int] = None,
) -> Dict[str, Any]:
    """The `web_search` tool entry.

    Nothing searches unless this is in `tools`: a request without it returns a fluent,
    ungrounded, uncited answer and HTTP 200, which reads downstream as zero hits.
    """
    tool: Dict[str, Any] = {"type": "web_search"}

    if context_size:
        if context_size not in _CONTEXT_SIZES:
            raise PerplexityContractError(
                f"search_context_size must be one of {sorted(_CONTEXT_SIZES)}, got {context_size!r}"
            )
        tool["search_context_size"] = context_size

    if max_results is not None:
        if not 1 <= int(max_results) <= 50:
            raise PerplexityContractError(f"max_results must be 1-50, got {max_results}")
        tool["max_results"] = int(max_results)

    filters: Dict[str, Any] = {}
    cleaned = clean_domains(domains)
    if cleaned:
        filters["search_domain_filter"] = cleaned
    if recency:
        if recency not in _RECENCY_VALUES:
            raise PerplexityContractError(
                f"search_recency_filter must be one of {sorted(_RECENCY_VALUES)}, got {recency!r}"
            )
        filters["search_recency_filter"] = recency
    if filters:
        tool["filters"] = filters

    if country:
        tool["user_location"] = {"country": str(country).upper()}

    return tool


def clean_domains(domains: Optional[List[str]]) -> List[str]:
    """Normalise and cap a domain filter. Scheme-less, deduped, `-domain.com` denies."""
    out: List[str] = []
    for raw in domains or []:
        if not raw or not isinstance(raw, str):
            continue
        d = raw.strip().lower()
        for prefix in ("https://", "http://"):
            d = d.removeprefix(prefix)
        negated = d.startswith("-")
        d = d.lstrip("-").removeprefix("www.").rstrip("/")
        if not d:
            continue
        d = f"-{d}" if negated else d
        if d not in out:
            out.append(d)
    return out[:DOMAIN_FILTER_MAX]


def build_agent_body(
    *,
    tier: str,
    input_text: str,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    response_schema: Optional[Dict[str, Any]] = None,
    schema_name: str = "result",
    schema_strict: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble a `/v1/agent` request.

    Strict mode: an unknown field is a 400, so only documented keys are emitted and the
    Sonar names (`messages`, `max_tokens`, `web_search_options`, `search_domain_filter`
    at top level) must never be added back beside them.
    """
    routing = _TIER_ROUTING.get(tier)
    if routing is None:
        raise PerplexityContractError(
            f"unknown tier {tier!r}; expected one of {sorted(_TIER_ROUTING)}"
        )

    body: Dict[str, Any] = dict(routing)
    body["input"] = input_text
    if instructions:
        body["instructions"] = instructions
    if max_output_tokens:
        body["max_output_tokens"] = int(max_output_tokens)
    if temperature is not None:
        body["temperature"] = float(temperature)
    if tools:
        body["tools"] = tools
    if response_schema is not None:
        # `strict` is per-caller and defaults OFF: it demands every property be listed in
        # `required`, which two of our schemas deliberately are not. Turning it on for
        # them would 400 rather than degrade, so each caller keeps the setting it had.
        json_schema: Dict[str, Any] = {
            "name": _safe_schema_name(schema_name),
            "schema": response_schema,
        }
        if schema_strict:
            json_schema["strict"] = True
        body["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    return body


def _safe_schema_name(name: str) -> str:
    """`json_schema.name` must be 1-64 chars of `[A-Za-z0-9_-]`."""
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in (name or ""))
    return (cleaned[:64] or "result")


def parse_agent_reply(
    data: Dict[str, Any], *, http_status: int = 200, latency_ms: int = 0
) -> AgentReply:
    """Reduce a `/v1/agent` body to an `AgentReply`.

    The four renames this migration has to survive are all read here: `status` (a failed
    run arrives as HTTP 200), `usage.input_tokens`/`output_tokens` (were
    `prompt_tokens`/`completion_tokens`), and `search_results` (was top level, is now an
    item inside `output[]`).
    """
    status = str(data.get("status") or "")
    usage = data.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)

    cost_block = usage.get("cost") or {}

    def _money(key: str) -> Optional[float]:
        value = cost_block.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    reported_cost_usd = _money("total_cost")

    tool_details = usage.get("tool_calls_details") or {}
    search_invocations = 0
    for key in ("search_web", "web_search"):
        entry = tool_details.get(key) or {}
        if isinstance(entry, dict):
            search_invocations += int(entry.get("invocation") or 0)

    text_parts: List[str] = []
    search_results: List[Dict[str, Any]] = []
    citation_urls: List[str] = []

    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message":
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("output_text", "text") and block.get("text"):
                    text_parts.append(str(block["text"]))
                for note in block.get("annotations") or []:
                    if isinstance(note, dict) and note.get("url"):
                        citation_urls.append(str(note["url"]))
        elif kind == "search_results":
            for result in item.get("results") or []:
                if isinstance(result, dict):
                    search_results.append(result)
                    if result.get("url"):
                        citation_urls.append(str(result["url"]))

    deduped: List[str] = []
    for url in citation_urls:
        if url not in deduped:
            deduped.append(url)

    reply = AgentReply(
        ok=False,
        status=status,
        text="\n".join(text_parts).strip(),
        search_results=search_results,
        citation_urls=deduped,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reported_cost_usd=reported_cost_usd,
        reported_input_cost_usd=_money("input_cost"),
        reported_output_cost_usd=_money("output_cost"),
        search_invocations=search_invocations,
        latency_ms=latency_ms,
        http_status=http_status,
    )

    if http_status >= 300:
        reply.error = f"perplexity HTTP {http_status}"
        return reply

    if status == "completed":
        reply.ok = True
        return reply

    if status == "incomplete":
        detail = (data.get("incomplete_details") or {}).get("reason") or "unknown"
        reply.error = f"incomplete: {detail}"
        return reply

    err = data.get("error") or {}
    detail = err.get("message") if isinstance(err, dict) else str(err)
    reply.error = f"agent status={status or 'missing'}: {detail or 'no detail'}"
    return reply


#: The credential this client spends. One name for both subsystems that share the key (#416).
PROVIDER = "perplexity"


async def call_agent(
    *,
    api_key: str,
    body: Dict[str, Any],
    timeout_s: float = 60.0,
    workspace_id: Optional[str] = None,
    use_breaker: bool = True,
) -> AgentReply:
    """POST one request and return the parsed reply. Never raises for a bad answer.

    A credential the provider has refused five times running is not called again until the
    half-open probe (#416): 198 calls a week for six weeks were paid for and refused, and both
    callers rendered the result as zero. `status="credential_refused"` carries the provider's own
    last message so the collector can say WHY it has no data instead of reporting none.
    """
    import httpx

    if not api_key:
        return AgentReply(ok=False, status="no_key", error="PERPLEXITY_API_KEY not configured")

    if use_breaker:
        from app.services.integrations.breaker_store import breaker_verdict

        verdict = breaker_verdict(PROVIDER, workspace_id)
        if verdict.refused:
            return AgentReply(
                ok=False, status="credential_refused", http_status=None, error=verdict.reason
            )

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                AGENT_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
    except Exception as e:
        return AgentReply(
            ok=False,
            status="transport_error",
            latency_ms=int((time.time() - start) * 1000),
            error=f"request failed: {e}",
        )

    latency_ms = int((time.time() - start) * 1000)
    try:
        data = resp.json()
    except Exception as e:
        return AgentReply(
            ok=False,
            status="unparseable",
            latency_ms=latency_ms,
            http_status=resp.status_code,
            error=f"perplexity HTTP {resp.status_code}, unparseable body: {e}",
        )

    reply = parse_agent_reply(data, http_status=resp.status_code, latency_ms=latency_ms)
    if not reply.ok and reply.http_status and reply.http_status >= 300:
        detail = (data.get("error") or {}).get("message") if isinstance(data.get("error"), dict) else None
        reply.error = f"perplexity HTTP {resp.status_code}: {detail or str(data)[:300]}"
    if use_breaker:
        from app.services.integrations.breaker_store import record_outcome

        record_outcome(
            PROVIDER,
            ok=reply.ok,
            http_status=reply.http_status,
            error=reply.error,
            workspace_id=workspace_id,
        )
    return reply
