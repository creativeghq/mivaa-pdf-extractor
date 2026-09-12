#!/usr/bin/env python3
"""Live smoke for the Perplexity Agent API migration. Run the moment the account is funded.

The migration shipped blind — out of credit since 2026-08-01, which Perplexity returns as a
401 — so no request shape was ever seen by the real server. These checks are the ones the
unit tests CANNOT make: they assert on recorded fixtures, and the question here is whether
the live API accepts what we build. Exit 0 = every shape accepted.

    PERPLEXITY_API_KEY=... python scripts/smoke_perplexity_agent.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.integrations.perplexity_agent_client import (  # noqa: E402
    TIER_SONAR,
    TIER_SONAR_PRO,
    build_agent_body,
    call_agent,
    web_search_tool,
)

API_KEY = os.getenv("PERPLEXITY_API_KEY") or ""

CHECKS: list[tuple[str, dict]] = [
    (
        "cheap tier: perplexity/sonar with a plain web_search tool",
        build_agent_body(
            tier=TIER_SONAR,
            input_text="What is materialshub.gr? Answer in one sentence.",
            max_output_tokens=300,
            tools=[web_search_tool()],
        ),
    ),
    (
        "deep tier: preset low + model pinned to perplexity/sonar",
        build_agent_body(
            tier=TIER_SONAR_PRO,
            input_text="Name two Greek tile retailers. One line each.",
            max_output_tokens=300,
            tools=[web_search_tool()],
        ),
    ),
    (
        "search_context_size as a STRING on the tool (not the number the notes predicted)",
        build_agent_body(
            tier=TIER_SONAR,
            input_text="Price of a Grohe kitchen tap in Greece?",
            max_output_tokens=300,
            tools=[web_search_tool(context_size="high", country="gr")],
        ),
    ),
    (
        "20-domain filter passes without a 400 or a silent truncation at 10",
        build_agent_body(
            tier=TIER_SONAR,
            input_text="Any remote Python jobs posted this week?",
            max_output_tokens=300,
            tools=[
                web_search_tool(
                    domains=[f"example{i}.com" for i in range(19)] + ["linkedin.com"],
                    recency="week",
                )
            ],
        ),
    ),
    (
        "strict json_schema still accepted",
        build_agent_body(
            tier=TIER_SONAR,
            input_text="Name one Greek bathroom retailer.",
            max_output_tokens=300,
            schema_strict=True,
            schema_name="smoke_result",
            response_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            tools=[web_search_tool()],
        ),
    ),
    (
        "a deliberately bad model slug is reported as a FAILURE, never a success with 0 hits",
        {**build_agent_body(tier=TIER_SONAR, input_text="hello", max_output_tokens=50),
         "model": "perplexity/does-not-exist"},
    ),
]


async def main() -> int:
    if not API_KEY:
        print("PERPLEXITY_API_KEY is unset — nothing to smoke.")
        return 2

    failures = 0
    for index, (label, body) in enumerate(CHECKS):
        expect_failure = "bad model slug" in label
        reply = await call_agent(api_key=API_KEY, body=body, timeout_s=90.0)

        if expect_failure:
            if reply.ok:
                print(f"FAIL  {label}\n      the API accepted a nonexistent model")
                failures += 1
            else:
                print(f"ok    {label}\n      refused as: {reply.error}")
            continue

        if not reply.ok:
            print(f"FAIL  {label}\n      {reply.error}\n      request: {json.dumps(body)[:400]}")
            failures += 1
            continue

        notes = []
        if reply.input_tokens == 0 and reply.output_tokens == 0:
            notes.append("ZERO TOKENS — usage keys renamed again")
            failures += 1
        if not reply.citation_urls:
            notes.append("NO CITATIONS — nothing searched, or they moved again")
            failures += 1
        if reply.reported_cost_usd is None:
            notes.append("no usage.cost — the rate-table fallback is in play")
        print(
            f"ok    {label}\n"
            f"      {reply.input_tokens} in / {reply.output_tokens} out, "
            f"{len(reply.citation_urls)} citations, "
            f"cost {reply.reported_cost_usd}, {reply.search_invocations} searches"
            + ("".join(f"\n      ! {n}" for n in notes))
        )

    print("\nall shapes accepted" if not failures else f"\n{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
