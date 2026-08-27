"""The schema-drift lint's own logic, checked without a database (mivaa#26 M13-3).

`scripts/check_schema_drift.py` needs the live schema to do its job, and CI has the
service-role key for that. This file guards the half that needs nothing: the AST
extraction, and the rule about what gets skipped.

That split matters. A lint whose extractor silently stops matching is worse than no
lint — it reports "0 stale references" out of 0 pairs examined and looks exactly like a
clean checkout. The cases below feed it synthetic source with known answers, including
every shape it is supposed to REFUSE to guess at.

The module is stdlib-only by design (`ast` + `urllib`), so unlike most of `app/` it can
simply be imported here — CI installs pytest alone.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_schema_drift.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_schema_drift", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through sys.modules.
    sys.modules["check_schema_drift"] = module
    spec.loader.exec_module(module)
    return module


csd = _load()


def _collect(src: str, tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text(src, encoding="utf-8")
    original = csd.ROOT
    csd.ROOT = tmp_path
    try:
        return csd.collect([f])
    finally:
        csd.ROOT = original


# ── extraction ──────────────────────────────────────────────────────────────

def test_it_finds_columns_through_a_builder_chain(tmp_path):
    """The real call sites are `client.table('x').select('a, b').eq(...).execute()`,
    which parses inside-out — the select's `func.value` is the table call."""
    usages, skips = _collect(
        'client.table("products").select("id, name, sku").eq("id", pid).execute()', tmp_path
    )
    assert {(u.table, u.column) for u in usages} == {
        ("products", "id"), ("products", "name"), ("products", "sku")
    }
    assert not skips


def test_it_follows_a_chain_that_starts_deeper(tmp_path):
    usages, _ = _collect(
        'self.supabase.client.table("documents").select("id, workspace_id").execute()', tmp_path
    )
    assert {(u.table, u.column) for u in usages} == {
        ("documents", "id"), ("documents", "workspace_id")
    }


def test_an_alias_resolves_to_the_real_column(tmp_path):
    """`alias:real_column` — PostgREST resolves the right-hand side, so that is what
    has to exist."""
    usages, _ = _collect('client.table("products").select("label:name").execute()', tmp_path)
    assert [(u.table, u.column) for u in usages] == [("products", "name")]


def test_the_count_kwarg_does_not_hide_the_columns(tmp_path):
    """`select('id', count='exact')` is common in this tree and the columns are still
    in the first positional argument."""
    usages, _ = _collect(
        'client.table("document_chunks").select("id", count="exact").execute()', tmp_path
    )
    assert [(u.table, u.column) for u in usages] == [("document_chunks", "id")]


# ── what it refuses to guess at ─────────────────────────────────────────────

@pytest.mark.parametrize(
    ("src", "reason"),
    [
        ('client.table("products").select("*").execute()', "contains *"),
        ('client.table("products").select("id, images(url)").execute()', "embedded resource"),
        ('client.table("products").select(cols).execute()', "not a string literal"),
        ('client.table("products").select(f"id, {extra}").execute()', "not a string literal"),
        ('client.table(name).select("id").execute()', "table not a literal"),
    ],
    ids=["star", "embedded", "variable", "fstring", "dynamic-table"],
)
def test_unresolvable_shapes_are_skipped_never_guessed(tmp_path, src: str, reason: str):
    """Guessing here would produce false failures, and a lint that cries wolf gets
    deleted rather than debugged."""
    usages, skips = _collect(src, tmp_path)
    assert not usages, f"the lint invented a column from {src!r}"
    assert [s.reason for s in skips] == [reason]


def test_every_skip_is_recorded(tmp_path):
    """A guard that silently narrows its own coverage is the failure mode this repo
    keeps finding, so skips are counted and printed rather than dropped."""
    usages, skips = _collect(
        'client.table("products").select("*").execute()\n'
        'client.table("products").select("id").execute()\n',
        tmp_path,
    )
    assert len(usages) == 1 and len(skips) == 1


# ── the verdict ─────────────────────────────────────────────────────────────

def test_a_missing_column_is_reported_and_a_present_one_is_not():
    registry = {"products": {"id", "name", "attributes"}}
    usages = [
        csd.Usage("products", "name", "f.py", 1),
        csd.Usage("products", "material_type", "f.py", 2),
    ]
    stale, checked, unknown = csd.lint(registry, usages)
    assert [(u.table, u.column) for u in stale] == [("products", "material_type")]
    assert checked == 2 and unknown == 0


def test_a_table_outside_the_registry_is_counted_not_failed():
    """A name that is not a public table is a storage bucket, a view, or a typo. This
    lint is about columns; failing on it would make it noisy about something it cannot
    judge."""
    stale, checked, unknown = csd.lint(
        {"products": {"id"}}, [csd.Usage("some_view", "id", "f.py", 1)]
    )
    assert not stale and checked == 0 and unknown == 1


def test_an_incomplete_registry_fails_instead_of_passing(monkeypatch):
    """A truncated response describes a SMALLER database, so every table it lost becomes
    "unknown table" and is skipped — the check would pass having examined almost
    nothing. The platform's own lint carries this floor for the same reason."""
    import json as _json

    class _Resp:
        def read(self):
            return _json.dumps({"tables": {"products": {"columns": ["id"]}}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    monkeypatch.setattr(csd.urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(SystemExit) as e:
        csd.fetch_registry()
    assert "only 1 tables" in str(e.value)


def test_it_refuses_to_run_without_the_key(monkeypatch):
    """There is nothing in this repository to compare a checkout against, so a missing
    key must stop the gate rather than let it report success."""
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        csd.fetch_registry()
    assert "SUPABASE_SERVICE_ROLE_KEY is required" in str(e.value)


# ── the ten it was built for ────────────────────────────────────────────────

#: Every stale (table, column) pair from #26, all confirmed absent from the live schema.
#: This asserts the REPAIR, not the lint: if one comes back, PostgREST rejects that call
#: site again.
AUDIT_26_STALE = [
    ("products", "material_type"), ("products", "tags"), ("products", "image_url"),
    ("products", "color"), ("products", "finish"), ("products", "collection"),
    ("products", "manufacturer"),
    ("background_jobs", "discovery_model"), ("background_jobs", "extract_categories"),
    ("image_product_associations", "document_id"),
]


def test_none_of_the_ten_stale_references_came_back():
    app_files = sorted(
        p for p in (ROOT / "app").rglob("*.py")
        if not any(part in csd.SKIP_DIRS for part in p.parts)
    )
    usages, _ = csd.collect(app_files)
    found = sorted({
        (u.table, u.column, u.path, u.line)
        for u in usages if (u.table, u.column) in set(AUDIT_26_STALE)
    })
    assert not found, "stale column references are back:\n" + "\n".join(
        f"  {p}:{ln}  {t}.{c}" for t, c, p, ln in found
    )


def test_the_sweep_still_examines_a_meaningful_number_of_pairs():
    """The companion to the case above. "0 stale references" is equally true of a
    healthy tree and of an extractor that stopped matching, and only this tells them
    apart."""
    app_files = sorted(
        p for p in (ROOT / "app").rglob("*.py")
        if not any(part in csd.SKIP_DIRS for part in p.parts)
    )
    usages, _ = csd.collect(app_files)
    assert len(usages) > 500, (
        f"the extractor found only {len(usages)} (table, column) pairs in app/. It "
        "matched ~1050 when written, so it has probably stopped recognising the "
        "builder chain rather than the tree having shrunk."
    )
