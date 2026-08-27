"""
The field registry lives in the DATABASE, and nothing may quietly re-copy it (#347 phase 3.5).

There were SIX copies of "what fields exist" and FOUR of "what material_category values are
allowed". None of them agreed, and the disagreements were load-bearing in both directions:
values the prompt offered were rejected by the validator (so a correctly-classified product got
re-classified on every run), and values the validator accepted were offered by no prompt at all
(so nothing could ever produce them). Neither shows up as a failure — both look like a working
pipeline that just never produces certain answers.

`material_metadata_fields` and `material_categories` are now the single source. These tests are
SOURCE-BASED on purpose: they must run in CI in a second with no database and no credentials, so
they check the shape of the code rather than the contents of the tables.

Three properties:
  1. The retired hardcoded copies stay retired.
  2. Every consumer of a synchronous accessor loads the registry first.
  3. Every accessor refuses to answer before the load, instead of returning a plausible default.
"""
import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
_REGISTRY = _APP / "services/metadata/field_registry.py"


def _app_sources():
    for path in sorted(_APP.rglob("*.py")):
        yield path, path.read_text(encoding="utf-8")


# ── 1. The retired copies stay retired ───────────────────────────────────────────────────────

#: Names that used to hold a hardcoded copy of the registry or the vocabulary. Every one of them
#: is now a DB column. A name reappearing means a copy reappeared.
RETIRED = {
    "CATEGORY_FIELD_REGISTRY": "material_metadata_fields (one row per field)",
    "CANONICALIZABLE_FACETS": "material_metadata_fields.canonicalize",
    "MATERIAL_CATEGORY_VOCAB": "material_categories.controlled_vocab",
    "get_priority_fields_for_prompt": "field_registry.priority_fields_prompt()",
    "get_extraction_hints_for_prompt": "field_registry.extraction_tips_prompt()",
    "get_skip_fields": "field_registry.skip_fields()",
    "get_category_config": "field_registry.category()",
    # #30 M16-7. These two were the last facts this module stated rather than read.
    "FALLBACK_CATEGORY": "material_categories.is_fallback (exactly one row)",
    "SECTION_ORDER": "material_field_sections.display_order",
}


def test_no_hardcoded_registry_comes_back():
    offenders = []
    for path, src in _app_sources():
        # Strip comments and docstrings before matching: the modules that REPLACED these names
        # legitimately explain what they replaced, and a prose mention is not a copy.
        try:
            tree = ast.parse(src)
        except SyntaxError:  # pragma: no cover - the syntax gate catches this first
            continue
        code_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        for name, replacement in RETIRED.items():
            if name in code_names:
                offenders.append(f"{path.relative_to(_ROOT)}: `{name}` -> use {replacement}")

    assert offenders == [], (
        "a hardcoded copy of the field registry / vocabulary came back:\n  "
        + "\n  ".join(offenders)
        + "\n\nThese are DB-backed now (#347 phase 3.2). A second copy does not fail loudly — it "
          "drifts, and then the prompt and the validator disagree about which values are legal."
    )


# ── 2. Every consumer loads before it reads ──────────────────────────────────────────────────

#: Synchronous accessors. Calling one before `ensure_loaded()` raises FieldRegistryNotLoaded.
SYNC_ACCESSORS = {
    "is_canonicalizable", "destination_of", "is_identity", "identity_fields", "spec_for",
    "fields_for_category", "category", "skip_fields", "controlled_vocab",
    "all_controlled_vocab", "controlled_vocab_by_category", "priority_fields_prompt",
    "extraction_tips_prompt", "skip_fields_prompt", "controlled_vocab_prompt",
    "category_prompt_block",
}

#: FUNCTIONS that read an accessor without loading it, each for a stated reason.
#: SHRINK-ONLY — an entry is a promise that some caller further up did the load.
#:
#: This is keyed by function, not by file. A file-level allowlist was tried first and it does
#: not work: stage_4_products already awaits ensure_loaded in `_classify_product`, so a NEW
#: function in that file reading the registry without loading it would inherit the pass. The
#: mutation test caught exactly that.
LOAD_EXEMPT = {
    "app/services/facets/facet_whitelist.py::is_canonicalizable":
        "pure delegation seam with no entry point of its own; its callers load",
}


def _functions(tree):
    """Every function in the module, including methods, keyed by name."""
    return {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _accessors_used(node):
    return {
        sub.attr for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute)
        and sub.attr in SYNC_ACCESSORS
        and isinstance(sub.value, ast.Name)
        and sub.value.id == "field_registry"
    }


def _calls_names(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def test_every_registry_consumer_loads_it_first():
    """Checked per FUNCTION, not per file.

    A function that reads the registry must either load it, guard on `is_loaded` (the degraded
    path, which must never raise), or be called by a function in the same module that loads
    first.
    """
    missing = []
    for path, src in _app_sources():
        if path == _REGISTRY or "field_registry" not in src:
            continue
        rel = path.relative_to(_ROOT).as_posix()
        try:
            tree = ast.parse(src)
        except SyntaxError:  # pragma: no cover
            continue

        funcs = _functions(tree)
        loaders = {name for name, node in funcs.items() if "ensure_loaded" in _calls_names(node)}

        for name, node in funcs.items():
            used = _accessors_used(node)
            if not used or f"{rel}::{name}" in LOAD_EXEMPT:
                continue
            if name in loaders:
                continue
            # Branching on `is_loaded` is the documented degraded pattern: answer differently
            # rather than raise, because the caller is often the except block where the load
            # is what failed.
            body = ast.unparse(node)
            if "field_registry.is_loaded" in body:
                continue
            # Or a caller in this module loads before delegating.
            if any(name in _calls_names(funcs[l]) for l in loaders):
                continue
            missing.append(f"{rel}::{name} calls {sorted(used)} without loading the registry")

    assert missing == [], (
        "a function reads the field registry without loading it:\n  " + "\n  ".join(missing)
        + "\n\nThe registry is loaded EXPLICITLY, not lazily, because `is_canonicalizable` is "
          "called from a sync path that cannot await. Add `await field_registry.ensure_loaded()` "
          "at that path's entry point, or guard on `field_registry.is_loaded` if this is a "
          "degraded path that must not raise."
    )


def test_the_load_exemptions_still_apply():
    """An exemption is a claim about a function. If the function goes, the claim goes with it."""
    stale = []
    for entry in LOAD_EXEMPT:
        rel, _, func = entry.partition("::")
        path = _ROOT / rel
        if not path.exists():
            stale.append(f"{entry} (file gone)")
            continue
        if func not in _functions(ast.parse(path.read_text(encoding="utf-8"))):
            stale.append(f"{entry} (function gone)")
    assert stale == [], f"LOAD_EXEMPT is stale: {stale}"

    canonicalizer = (_ROOT / "app/services/facets/facet_canonicalizer.py").read_text(
        encoding="utf-8")
    assert "field_registry.is_loaded" in canonicalizer, (
        "collect_raw_attributes stopped branching on `is_loaded`. It is called from the except "
        "block where the registry load is exactly what failed — if it now raises instead of "
        "over-capturing, a registry outage silently costs every product its attributes_raw, "
        "which is the lossless replay contract."
    )
    assert "await field_registry.ensure_loaded()" in canonicalizer, (
        "canonicalize_product no longer loads the registry; _collect_pending reads it "
        "synchronously and would raise."
    )


# ── 3. No accessor answers before the load ───────────────────────────────────────────────────

def test_every_accessor_refuses_before_load():
    """Each public accessor must reach `self._require()`, directly or via another accessor.

    An accessor that returned a default instead would be the silent-zero shape in its purest
    form: `is_canonicalizable` answering False for everything means ingestion keeps running and
    writes zero facets forever, with nothing raising and nothing logged.
    """
    tree = ast.parse(_REGISTRY.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "FieldRegistry")

    methods = {
        n.name: n for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # `_section_rank` is a staticmethod over a constant, and loading/introspection helpers are
    # what the guard is defined in terms of.
    exempt = {"__init__", "ensure_loaded", "_load_blocking", "_require",
              "is_loaded", "is_stale", "_section_rank", "_resolve_category"}

    def reaches_require(name, seen=frozenset()):
        node = methods.get(name)
        if node is None or name in seen:
            return False
        calls = {
            c.func.attr for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and isinstance(c.func.value, ast.Name) and c.func.value.id == "self"
        }
        if "_require" in calls:
            return True
        return any(reaches_require(c, seen | {name}) for c in calls)

    unguarded = [
        name for name in methods
        if not name.startswith("__") and name not in exempt and not reaches_require(name)
    ]
    assert unguarded == [], (
        f"these accessors can answer before the registry is loaded: {unguarded}. "
        "Route them through `self._require()` so they raise instead of returning a default."
    )


def test_load_refuses_an_empty_result():
    """Zero rows is never a legitimate registry — it must raise, not cache emptiness."""
    src = _REGISTRY.read_text(encoding="utf-8")
    body = src[src.index("def _load_blocking"):src.index("def _require")]

    assert re.search(r"if not field_rows:\s*\n\s*raise", body), (
        "_load_blocking no longer raises on zero fields. Caching an empty registry means every "
        "later extraction asks for no fields and canonicalizes nothing, silently and forever."
    )
    assert body.index("raise") < body.index("self._cache = _Cache"), (
        "the empty-result check must come BEFORE the cache assignment, or a bad read replaces a "
        "good cache."
    )


# ── 6. The last two hardcoded facts (#30 M16-7) ──────────────────────────────────────────────

def test_the_fallback_category_is_read_not_stated():
    """`FALLBACK_CATEGORY = "general_materials"` decided where every unrecognised upload
    category went, from a file an admin editing the registry cannot see. It is now
    `material_categories.is_fallback`, with a partial unique index so there is exactly
    one — a second fallback is not a merge, it is a coin toss."""
    src = _REGISTRY.read_text(encoding="utf-8")
    body = src[src.index("def _resolve_category"):src.index("def category(")]
    assert "cache.fallback_category" in body, (
        "the fallback category is a literal again (#30 M16-7)"
    )
    assert '"general_materials"' not in body


def test_a_registry_with_no_fallback_row_raises():
    """There is deliberately no constant left to fall back TO. A fallback is invisible
    when it fires, which is the whole reason this module exists."""
    src = _REGISTRY.read_text(encoding="utf-8")
    body = src[src.index("def _load_blocking"):src.index("def _require")]
    assert re.search(r"if not fallback:\s+raise", body), (
        "_load_blocking accepts a registry with no is_fallback row — an unknown "
        "category would then route to whatever `.get()` returned"
    )


def test_section_order_comes_from_its_own_table():
    src = _REGISTRY.read_text(encoding="utf-8")
    assert "material_field_sections" in src, (
        "prompt section ordering is back in Python (#30 M16-7)"
    )
    body = src[src.index("def _section_rank"):src.index("def priority_fields_prompt")]
    assert "self._require().section_order" in body
    assert "material_properties" not in body, "the section list is hardcoded again"


def test_an_unplaced_section_still_sorts_last_rather_than_first():
    """The old constant put an unlisted section after every listed one, and `design` is
    live in exactly that state. `.get(section, 0)` would silently promote it to the top
    of every extraction prompt — a plausible reordering that nothing would raise on."""
    src = _REGISTRY.read_text(encoding="utf-8")
    body = src[src.index("def _section_rank"):src.index("def priority_fields_prompt")]
    assert "max(order.values(), default=0) + 1" in body, (
        "an unplaced section no longer sorts last"
    )


def test_the_section_table_is_required_like_the_other_two():
    src = _REGISTRY.read_text(encoding="utf-8")
    body = src[src.index("def _load_blocking"):src.index("def _require")]
    assert re.search(r"if not section_rows:\s+raise", body), (
        "an empty material_field_sections is accepted, which silently reorders every "
        "extraction prompt"
    )
    assert body.index("if not section_rows") < body.index("self._cache = _Cache"), (
        "the check must precede the cache assignment, or a bad read replaces a good cache"
    )
