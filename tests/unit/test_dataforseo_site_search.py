"""Guard (#497): Reddit / Pinterest / Trustpilot keyword search runs a live `site:` SERP.

The social_media endpoints are keyed by URL and the Trustpilot one is task_post, so every
keyword search read as "0 results" with nothing flagging it.
"""

import asyncio

import pytest

pytest.importorskip("httpx")

from tests.unit.test_dataforseo_free_refusal_retry import client_mod  # noqa: E402

SERP_ITEMS = [
    {"type": "people_also_ask", "title": "skip me"},
    {"type": "organic", "title": "Best tiles?", "url": "https://www.reddit.com/r/HomeImprovement/comments/abc/best_tiles/",
     "description": "thread"},
    {"type": "organic", "title": "Keros reviews", "url": "https://www.trustpilot.com/review/keros.es",
     "description": "reviews", "rating": {"rating_value": 4.2, "votes_count": 130}},
    {"type": "organic", "title": "no url"},
]


def _client(captured):
    c = client_mod.DataForSEOUnifiedClient(sandbox=True)

    async def fake_call(path, body=None, **kw):
        captured.append((path, body))
        return client_mod.DataForSEOResult(ok=True, items=[dict(i) for i in SERP_ITEMS])

    c._call = fake_call
    return c


def test_reddit_uses_a_site_restricted_live_serp():
    captured = []
    res = asyncio.run(_client(captured).business_reddit(keyword="floor tiles"))
    path, body = captured[0]
    assert path == "/serp/google/organic/live/advanced"
    assert body[0]["keyword"] == "site:reddit.com floor tiles"
    assert res.ok and len(res.items) == 2, "only organic results that carry a url"
    assert res.items[0]["subreddit"] == "r/HomeImprovement"
    assert "_rating" not in res.items[0]
    assert "people_also_ask_click_depth" not in body[0], "DataForSEO refuses a PAA depth of 0 with 40501"


def test_pinterest_uses_a_site_restricted_live_serp():
    captured = []
    asyncio.run(_client(captured).business_pinterest(keyword="terrazzo"))
    assert captured[0][1][0]["keyword"] == "site:pinterest.com terrazzo"


def test_trustpilot_reads_domain_and_rating_from_the_serp():
    captured = []
    res = asyncio.run(_client(captured).business_trustpilot_search(keyword="tiles"))
    assert captured[0][0] == "/serp/google/organic/live/advanced", "task_post never answered synchronously"
    tp = [i for i in res.items if i["domain"] == "keros.es"][0]
    assert tp["rating"] == 4.2 and tp["reviews_count"] == 130


def test_a_failed_serp_is_returned_as_a_failure_not_zero_results():
    c = client_mod.DataForSEOUnifiedClient(sandbox=True)

    async def failing(path, body=None, **kw):
        return client_mod.DataForSEOResult(ok=False, error="402 Payment Required", status_code=402)

    c._call = failing
    res = asyncio.run(c.business_reddit(keyword="x"))
    assert not res.ok and res.error


def test_organic_task_sends_a_paa_depth_only_in_range():
    from app.services.integrations.dataforseo_serp_targeting import organic_task
    base = dict(keyword="k", country_location=2300, language_code="en", depth=10)
    assert "people_also_ask_click_depth" not in organic_task(**base, paa_depth=0)
    assert organic_task(**base, paa_depth=2)["people_also_ask_click_depth"] == 2
    assert organic_task(**base, paa_depth=9)["people_also_ask_click_depth"] == 4
