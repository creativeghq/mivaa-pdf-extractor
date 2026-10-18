"""#482 F7: a mention refresh with a failed source must not report clean."""

from app.services.integrations.tracked_mentions_service import refresh_error_summary


def test_no_failed_source_is_clean():
    assert refresh_error_summary({}) is None
    assert refresh_error_summary(None) is None
    assert refresh_error_summary({"gdelt": ""}) is None


def test_every_failed_source_is_named():
    summary = refresh_error_summary({
        "perplexity_sonar": "401 Unauthorized",
        "dataforseo_news": "402 Payment Required",
        "gdelt": "",
    })
    assert summary == "dataforseo_news: 402 Payment Required; perplexity_sonar: 401 Unauthorized"


def test_summary_is_bounded():
    assert len(refresh_error_summary({f"s{i}": "x" * 500 for i in range(20)})) <= 1000
