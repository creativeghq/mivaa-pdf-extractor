"""ATS board parsing and discovery (#462).

A board that changed shape must raise, never parse to an empty list: an empty list is
byte-identical to "this company is not hiring". Modules load by path; CI installs pytest only.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


h2m = _load("app.utils.html_to_markdown", "app/utils/html_to_markdown.py")
ats = _load("app.services.integrations.ats_boards", "app/services/integrations/ats_boards.py")
fp = _load("app.services.integrations.tech_fingerprints", "app/services/integrations/tech_fingerprints.py")


def _b(obj) -> bytes:
    return json.dumps(obj).encode()


def test_html_to_markdown_keeps_structure_and_drops_scripts():
    md = h2m.html_to_markdown(
        "<h2>Requirements</h2><ul><li>SAP <b>S/4HANA</b></li><li>Greek</li></ul>"
        "<script>steal()</script><iframe>x</iframe><p>Apply&nbsp;now</p>"
    )
    assert "## Requirements" in md
    assert "- SAP **S/4HANA**" in md
    assert "- Greek" in md
    assert "steal" not in md and "Apply" in md


def test_greenhouse_content_is_entity_encoded_html():
    company, rows = ats.parse_board("greenhouse", _b({"jobs": [{
        "id": 1, "title": "ERP Lead", "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1",
        "company_name": "Acme", "location": {"name": "Remote, Greece"},
        "content": "&lt;p&gt;We are migrating to &lt;strong&gt;SAP&lt;/strong&gt;&lt;/p&gt;",
        "departments": [{"name": "Finance"}], "first_published": "2026-10-02T11:31:50-04:00",
    }]}))
    assert company == "Acme"
    assert rows[0]["description_md"] == "We are migrating to **SAP**"
    assert rows[0]["is_remote"] is True and rows[0]["department"] == "Finance"
    assert rows[0]["posted_at"].startswith("2026-10-02")


def test_lever_body_includes_lists_and_additional():
    _, rows = ats.parse_board("lever", _b([{
        "id": "u1", "text": "Buyer", "hostedUrl": "https://jobs.lever.co/acme/u1",
        "description": "<div>Intro</div>", "additional": "<div>Perks</div>",
        "lists": [{"text": "You will", "content": "<li>Source tiles</li>"}],
        "categories": {"commitment": "Full-time", "department": "Ops", "location": "Athens"},
        "workplaceType": "onsite", "createdAt": 1789748062697,
    }]))
    body = rows[0]["description_md"]
    assert "Intro" in body and "## You will" in body and "- Source tiles" in body and "Perks" in body
    assert rows[0]["is_remote"] is False and rows[0]["posted_at"].startswith("2026")


def test_ashby_skips_unlisted_and_workable_reads_shortcode():
    _, rows = ats.parse_board("ashby", _b({"jobs": [
        {"id": "a", "title": "PM", "jobUrl": "https://jobs.ashbyhq.com/x/a", "descriptionHtml": "<p>Body</p>"},
        {"id": "b", "title": "Hidden", "jobUrl": "https://jobs.ashbyhq.com/x/b", "isListed": False},
    ]}))
    assert [r["external_id"] for r in rows] == ["a"]
    company, rows = ats.parse_board("workable", _b({"name": "Plaisio", "jobs": [{
        "shortcode": "D25", "title": "Sales", "url": "https://apply.workable.com/j/D25",
        "description": "<p>Body</p>", "city": "Athens", "country": "Greece", "telecommuting": False,
    }]}))
    assert company == "Plaisio" and rows[0]["external_id"] == "D25" and rows[0]["location"] == "Athens, Greece"


@pytest.mark.parametrize("provider,payload", [
    ("greenhouse", {"postings": []}),
    ("lever", {"jobs": []}),
    ("ashby", []),
    ("workable", {"name": "x"}),
])
def test_a_changed_shape_raises_instead_of_reading_as_not_hiring(provider, payload):
    with pytest.raises(ats.AtsShapeError):
        ats.parse_board(provider, _b(payload))


def test_a_genuinely_empty_board_is_an_empty_list():
    assert ats.parse_board("greenhouse", _b({"jobs": []})) == (None, [])
    assert ats.parse_board("lever", _b([])) == (None, [])


def test_non_json_body_raises():
    with pytest.raises(ats.AtsShapeError):
        ats.parse_board("ashby", b"<html>Cloudflare</html>")


def test_teamtailor_rss():
    pytest.importorskip("defusedxml")
    rss = b"""<?xml version="1.0"?><rss version="2.0" xmlns:tt="https://teamtailor.com/locations">
      <channel><title>Orac</title><item><title>Product Manager</title>
      <description>&lt;h6&gt;The role&lt;/h6&gt;&lt;p&gt;Launch products&lt;/p&gt;</description>
      <pubDate>Mon, 29 Jun 2026 20:16:54 +0200</pubDate>
      <link>https://jobs.oracdecor.com/jobs/7989030-product-manager</link>
      <remoteStatus>hybrid</remoteStatus><guid>1c18</guid>
      <tt:locations><tt:location><tt:city>Oostende</tt:city><tt:country>Belgium</tt:country></tt:location></tt:locations>
      <tt:department>Marketing</tt:department></item></channel></rss>"""
    company, rows = ats.parse_board("teamtailor", rss)
    assert company == "Orac"
    r = rows[0]
    assert r["external_id"] == "1c18" and r["department"] == "Marketing"
    assert r["location"] == "Oostende, Belgium" and "#### The role" in r["description_md"]


def test_teamtailor_token_must_be_a_host():
    assert ats.board_url("teamtailor", "jobs.oracdecor.com") == "https://jobs.oracdecor.com/jobs.rss"
    with pytest.raises(ValueError):
        ats.board_url("teamtailor", "evil.com/../x")


def test_board_url_quotes_the_token():
    assert ats.board_url("lever", "a/b?c") == "https://api.lever.co/v0/postings/a%2Fb%3Fc?mode=json"


def test_find_ats_accounts_on_real_link_shapes():
    page = """
      <a href="https://apply.workable.com/plaisio-careers/">Careers</a>
      <a href="https://boards.greenhouse.io/acme">Jobs</a>
      <a href="https://jobs.lever.co/matchgroup/123">x</a>
      <a href="https://careers.smartrecruiters.com/JYSK/foo">x</a>
      <a href="https://teamtailor.com/?utm_campaign=poweredby&amp;utm_content=jobs.oracdecor.com">Powered by Teamtailor</a>
      <a href="https://apply.workable.com/j/D25DD806E1">a job</a>
    """
    found = ats.find_ats_accounts(page)
    assert ("workable", "plaisio-careers") in found
    assert ("greenhouse", "acme") in found
    assert ("lever", "matchgroup") in found
    assert ("smartrecruiters", "jysk") in found
    assert ("teamtailor", "jobs.oracdecor.com") in found
    assert ("workable", "j") not in found
    assert ("workable", "apply") not in found


def test_find_careers_links_stays_on_site():
    page = """<a href="/lavora-con-noi?a=1&amp;b=2">Lavora con noi</a>
              <a href="https://jobs.example.com/">Open roles</a>
              <a href="https://linkedin.com/company/x/jobs">LinkedIn</a>
              <a href="/prodotti">Prodotti</a>"""
    links = ats.find_careers_links(page, "https://www.example.com/")
    assert links == ["https://www.example.com/lavora-con-noi?a=1&b=2", "https://jobs.example.com/"]


def test_eu_boards_and_the_greenhouse_js_embed():
    page = """
      <script src="https://boards.greenhouse.io/embed/job_board/js?for=acmeeu"></script>
      <a href="https://job-boards.eu.greenhouse.io/eucorp/jobs/1">x</a>
      <a href="https://jobs.eu.lever.co/leveu/abc">x</a>
    """
    found = ats.find_ats_accounts(page)
    assert ("greenhouse", "acmeeu") in found
    assert ("greenhouse", "eucorp") in found
    assert ("lever", "eu:leveu") in found
    assert ("greenhouse", "embed") not in found
    assert ats.board_url("lever", "eu:leveu") == "https://api.eu.lever.co/v0/postings/leveu?mode=json"


def test_a_percent_encoded_token_is_not_encoded_twice():
    assert ats.board_url("ashby", "Acme%20Labs") == "https://api.ashbyhq.com/posting-api/job-board/Acme%20Labs"


def test_homepage_fingerprints():
    page = """<html><head><meta name="generator" content="WordPress 6.6">
      <script src="https://www.googletagmanager.com/gtag/js?id=G-1"></script>
      <script async src="//static.klaviyo.com/onsite/js/klaviyo.js"></script></head>
      <body><a href="/wp-content/uploads/x.jpg">x</a></body></html>"""
    rules = [
        {"slug": "wordpress", "detect_html": {"meta_generator": ["wordpress"], "html": ["/wp-content/"]}},
        {"slug": "google-analytics", "detect_html": {"script_src": ["googletagmanager.com/gtag/js"]}},
        {"slug": "klaviyo", "detect_html": {"script_src": ["static.klaviyo.com"]}},
        {"slug": "shopify", "detect_html": {"script_src": ["cdn.shopify.com"], "html": ["Shopify.theme"]}},
        {"slug": "empty", "detect_html": {"html": [""]}},
    ]
    assert fp.detect(page, rules) == ["wordpress", "google-analytics", "klaviyo"]


def test_a_script_url_in_page_text_is_not_a_script_tag():
    page = "<p>We moved off static.klaviyo.com last year</p>"
    rules = [{"slug": "klaviyo", "detect_html": {"script_src": ["static.klaviyo.com"]}}]
    assert fp.detect(page, rules) == []
