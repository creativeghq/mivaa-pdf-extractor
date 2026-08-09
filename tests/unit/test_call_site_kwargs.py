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
            # Only judge unambiguous names. If two classes both define `run`, we
            # cannot tell statically which one this call resolves to, and a guard
            # that guesses is a guard that gets deleted the first time it is wrong.
            if not defs or len(defs) != 1:
                continue
            params, accepts_kwargs = defs[0]
            if accepts_kwargs:
                continue
            passed = {k.arg for k in node.keywords if k.arg}
            unknown = sorted(passed - params)
            if unknown:
                offenders.append(
                    f"{path.relative_to(APP.parent)}:{node.lineno}: "
                    f"{node.func.attr}({', '.join(unknown)}=...) — accepts only "
                    f"{sorted(params)}"
                )

    assert not offenders, (
        "Call site(s) pass a keyword the callee does not accept. Each raises "
        "TypeError at runtime, and in this codebase that is swallowed by an "
        "enclosing `except Exception` — the work silently never happens:\n  "
        + "\n  ".join(offenders)
    )
