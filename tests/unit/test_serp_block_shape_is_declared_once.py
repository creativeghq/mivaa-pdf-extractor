"""
The SERP block shape is declared once, and every key a reader indexes is in it.

THE FAILURE THIS PREVENTS
-------------------------
`mention_opportunity_service` builds one dict of SERP feature blocks and then indexes it by
literal key — `blocks["videos"]`, `blocks["paid"]`, `this_round["shopping"]`. A missing key is
therefore a KeyError, not a falsy value.

That shape was written out TWICE: the full set inside `_parse_serp_blocks`, and a five-key copy as
the "nothing found yet" default in `_serp_signals`. v0.4.6 added videos / news_stories /
knowledge_graph / paid / shopping to the parser and not to the default.

The two could never disagree on a good day, which is what made it survive: when DataForSEO
answered, the parser's dict REPLACED the default and every key was present. Only when every seed
errored — or every round returned no signal — did the five-key default survive to be read, and
then the first v0.4.6 reader raised `KeyError: 'videos'`. The caller wraps the call in
`except Exception`, so the entire SERP half of the opportunity report vanished behind a single
WARNING. A failure inside the failure path, observable only as absence.

Seen live 2026-08-26 05:16: `opportunity: SERP signals fetch failed: 'videos'`.

This is the shape DataForSEO's 2027-01-20 deprecation notice warns about, from the other side: they
are removing keys (`is_image`, `is_video`, `is_featured_snippet`, `is_malicious`, `is_web_story`,
`amp_version`) that direct indexing would turn into a hard error. We read none of those six — the
one that bit us was our own.

WHY SOURCE-BASED
----------------
MIVAA's CI installs pytest and nothing else, so this module cannot be imported (httpx, app.config,
supabase). The defect is a literal in the source, and that is what this reads.
"""
import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = _ROOT / "app/services/integrations/mention_opportunity_service.py"

#: Locals that hold a SERP-blocks dict. Indexing any of them requires a declared key.
_BLOCK_HOLDERS = {"blocks", "this_round", "result"}

#: DataForSEO removes these from organic (and the first two from paid) items on 2027-01-20.
#: Direct indexing turns their absence into a hard error, so nothing may index them.
_REMOVED_BY_DATAFORSEO_2027 = {
    "is_image", "is_video", "is_featured_snippet", "is_malicious", "is_web_story", "amp_version",
}


def _tree() -> ast.Module:
    return ast.parse(_SERVICE.read_text(encoding="utf-8"))


def _declared_keys(tree: ast.Module) -> set:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_empty_serp_blocks":
            for d in ast.walk(node):
                if isinstance(d, ast.Dict):
                    return {k.value for k in d.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    raise AssertionError(
        "_empty_serp_blocks() is gone — the block shape has more than one declaration again"
    )


def _indexed_keys(tree: ast.Module, holders: set) -> set:
    out = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id in holders
                and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)):
            out.add(n.slice.value)
    return out


def test_every_indexed_block_key_is_declared():
    tree = _tree()
    declared = _declared_keys(tree)
    indexed = _indexed_keys(tree, _BLOCK_HOLDERS)
    assert indexed, "no block key is indexed any more — this guard no longer guards anything"
    missing = indexed - declared
    assert not missing, (
        f"indexed but not declared in _empty_serp_blocks(): {sorted(missing)}. "
        "These raise KeyError on the path where DataForSEO returned nothing — which is the only "
        "path that reads the default at all."
    )


def test_the_shape_is_not_written_out_a_second_time():
    """A dict literal carrying the block keys anywhere else is the drift coming back."""
    tree = _tree()
    declared = _declared_keys(tree)
    signature = {"pao", "related_searches", "organic"}
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_empty_serp_blocks":
            continue
        for d in ast.walk(node):
            if not isinstance(d, ast.Dict):
                continue
            keys = {k.value for k in d.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if signature <= keys and keys != declared:
                offenders.append((getattr(d, "lineno", "?"), sorted(keys)))
    assert not offenders, (
        f"a second, DIFFERENT declaration of the SERP block shape: {offenders}. "
        "Call _empty_serp_blocks() instead — a copy that disagrees only on the failure path is "
        "exactly how KeyError: 'videos' shipped."
    )


def test_no_reader_touches_a_field_dataforseo_removes_in_2027():
    """The six legacy booleans DataForSEO drops on 2027-01-20.

    They evaluate to False on every result today, so anything reading them is already dead code —
    and on the removal date direct indexing turns it into a hard error instead.
    """
    src = _SERVICE.read_text(encoding="utf-8")
    found = sorted(f for f in _REMOVED_BY_DATAFORSEO_2027 if f in src)
    assert not found, (
        f"reads a DataForSEO field removed on 2027-01-20: {found}. They are False on every result "
        "today; after the removal, indexing them raises."
    )
