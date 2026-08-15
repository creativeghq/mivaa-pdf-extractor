"""
Guard: a keyword argument no callee accepts (#277).

Python binds keyword arguments at CALL time, so `svc.upsert(details=...)` against a
`def upsert(..., metadata)` is not a syntax error, not an import error, and not
something any linter in this repo runs. It raises TypeError the moment that line
executes — and every one of these call sites sits inside a broad `except Exception`
that logs and continues. The endpoint then reports success having done nothing.

That is the `ops.silent_zero` shape in application code: activity happened, the
metric it should have produced is zero, and nothing complains. Found when #277 went
looking for why the four aspect collections were thin — the endpoint that populates
them (`/regenerate-image-embeddings`) had been raising on EVERY image since a
`details` → `metadata` rename, marking its job COMPLETED with progress 100 each time.

Five call sites were broken this way, in three unrelated subsystems (image
embeddings, chunking, Claude validation). None of them was a typo — each was written
by copying a neighbouring call whose callee used the other name.

AMBIGUITY. Same-named definitions in different classes are judged against the UNION
of what they accept, so the sweep flags only keywords that no candidate could bind.
That is weaker than resolving the receiver, and deliberately so: it never guesses, and
a guard that guesses is a guard that gets deleted the first time it is wrong.

Static on purpose: importing these modules pulls the Supabase client, the settings
bootstrap and the whole runtime dependency set. This has to run in CI in a second
with nothing installed but pytest, which is the difference between a guard that runs
on every push and one that quietly never runs.
"""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"

# A decorator that makes a function an HTTP endpoint. Route handlers are invoked by
# FastAPI, never called by name, so a same-named method on some service object is a
# different function entirely — `page.get_images(full=True)` (PyMuPDF) collides with
# the `get_images` route in rag_routes.py. Without this the sweep reports 5 phantoms.
_ROUTE_DECORATORS = ("get", "post", "put", "patch", "delete", "head", "options")

# Attribute names that also belong to a type the sweep cannot see: builtins, and the
# verb methods of every HTTP client. `results.get(...)` is `dict.get`; `client.get(...)`
# is httpx's. A same-named function under app/ proves nothing about those receivers, so
# these names are judged only when exactly one definition exists (the strict rule below).
_UNIVERSAL_ATTRS = (
    set(dir(dict)) | set(dir(list)) | set(dir(str)) | set(dir(set)) | set(dir(bytes))
    | {"get", "post", "put", "patch", "delete", "head", "options", "request",
       "execute", "json", "read", "write", "close", "send", "run"}
)


def _is_route_handler(node) -> bool:
    for dec in node.decorator_list:
        call = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(call, ast.Attribute) and call.attr in _ROUTE_DECORATORS:
            return True
    return False


def _collect():
    """(name -> [(params, accepts_kwargs)]) for every non-route def under app/."""
    sigs: dict[str, list[tuple[set[str], bool]]] = {}
    trees: list[tuple[Path, ast.Module]] = []

    for path in sorted(APP.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        trees.append((path, tree))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_route_handler(node):
                continue
            a = node.args
            params = {
                x.arg
                for x in a.posonlyargs + a.args + a.kwonlyargs
                if x.arg not in ("self", "cls")
            }
            sigs.setdefault(node.name, []).append((params, a.kwarg is not None))
    return sigs, trees


def test_no_call_passes_a_keyword_its_callee_cannot_accept():
    sigs, trees = _collect()
    offenders = []

    for path, tree in trees:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            defs = sigs.get(node.func.attr)
            if not defs:
                continue
            # An ambiguous name is still judged, just more weakly (unless it is one
            # of the _UNIVERSAL_ATTRS above). We cannot tell
            # statically WHICH `search_similar_images` a call resolves to — but if
            # a keyword is accepted by NONE of them, the call raises TypeError
            # whichever one it is, and that verdict needs no guess. Requiring a
            # unique definition was itself the hole: `search_similar_images` has
            # two, so the sweep skipped the call site where
            # material_visual_search_service passed `reference_image_id` /
            # `max_results` to a callee taking `query_image_id` / `limit`. That
            # endpoint had never once returned a result (audit #18, alongside M5-2).
            if any(accepts_kwargs for _params, accepts_kwargs in defs):
                continue
            if len(defs) != 1 and node.func.attr in _UNIVERSAL_ATTRS:
                continue
            passed = {k.arg for k in node.keywords if k.arg}
            accepted_by_someone = set().union(*(params for params, _ in defs))
            unknown = sorted(passed - accepted_by_someone)
            if unknown:
                offenders.append(
                    f"{path.relative_to(APP.parent)}:{node.lineno}: "
                    f"{node.func.attr}({', '.join(unknown)}=...) — no definition of "
                    f"{node.func.attr} accepts that; they accept "
                    f"{sorted(accepted_by_someone)}"
                )

    assert not offenders, (
        "Call site(s) pass a keyword the callee does not accept. Each raises "
        "TypeError at runtime, and in this codebase that is swallowed by an "
        "enclosing `except Exception` — the work silently never happens:\n  "
        + "\n  ".join(offenders)
    )
