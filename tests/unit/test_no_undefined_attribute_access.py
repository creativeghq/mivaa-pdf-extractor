"""Two shapes that raise AttributeError on every call, swept out of `app/` (mivaa#20, #22 M9-4).

Both look like working code, both sit inside a `try`, both get swallowed by a broad
`except`, and both take out precisely the behaviour the surrounding comment promises.
Neither is visible to the tools already gating this repo: ruff does not resolve
attributes, and the tests never enter the branch.

    1. `self.X` read but never assigned.
       `image_processing_service` -> the embedding-failure marker never written, so
       nothing could tell "never run" from "ran and failed".
       `material_visual_search_service` -> the outage fallback dead, so an outage
       presented as a search returning nothing.
       `search_prompt_service` -> all four search-prompt features (#19 M6-1).

    2. The DOUBLE `.client` unwrap: `x = <expr>.client` and then `x.client.table(...)`.
       Invisible to the first sweep, because the name IS assigned — it is the second
       `.client` that is wrong. 16 sites across 3 files: nothing was ever queued for
       image processing, and saved-search deduplication had never found a duplicate
       (it raised, printed to stdout, and returned [] — which is exactly what a
       genuinely unique search looks like).

Both sweeps DERIVE their scope from the tree, so unlike a declared list they cannot
drift as files are added. Neither has an exemption list, deliberately: the moment one
exists, the next finding goes into it.

Static analysis, so it is bounded on purpose. A class with a base class, or one that
calls `setattr(self, ...)`, is skipped — its attributes can come from anywhere. Most of
this tree is pydantic models, so MORE classes are skipped than examined and that is
correct rather than a gap; what is pinned instead is a floor on the number actually
examined, so the sweep cannot quietly narrow itself to nothing.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
SKIP_DIRS = {"__pycache__", ".venv", "venv", "node_modules"}


def _py_files() -> List[Path]:
    return sorted(p for p in APP.rglob("*.py") if not any(d in p.parts for d in SKIP_DIRS))


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


# ─────────────────────────────────────────────────────────────────────────────
# Shape 1 — self.X read, never assigned
# ─────────────────────────────────────────────────────────────────────────────

def _assigned_attrs(cls: ast.ClassDef) -> Tuple[Set[str], bool]:
    """Everything `self.X` could legitimately resolve to, and whether to give up.

    Give up (return dynamic=True) on a subclass or on any `setattr(self, ...)`: both
    can supply attributes this file never mentions, and guessing there produces false
    failures — which is how a guard earns its deletion.
    """
    names: Set[str] = set()
    dynamic = bool(cls.bases)

    for node in ast.walk(cls):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setattr":
            dynamic = True
        # class-level constants and annotations (incl. dataclass fields)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        # self.X = ...  /  self.X: T = ...
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for t in targets:
            for sub in ast.walk(t):
                if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"):
                    names.add(sub.attr)
        # `for self.x in ...`, `with ... as self.x`
        if isinstance(node, ast.withitem) and node.optional_vars is not None:
            for sub in ast.walk(node.optional_vars):
                if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"):
                    names.add(sub.attr)

    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names, dynamic


def _read_attrs(cls: ast.ClassDef) -> List[Tuple[str, int]]:
    out = []
    for node in ast.walk(cls):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self" and isinstance(node.ctx, ast.Load)):
            out.append((node.attr, node.lineno))
    return out


def _sweep_undefined() -> Tuple[List[str], int, int]:
    findings: List[str] = []
    classes = skipped = 0
    for path in _py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            assigned, dynamic = _assigned_attrs(cls)
            if dynamic:
                skipped += 1
                continue
            classes += 1
            for attr, line in _read_attrs(cls):
                if attr not in assigned and not attr.startswith("__"):
                    findings.append(f"{_rel(path)}:{line}  {cls.name}.self.{attr}")
    return sorted(set(findings)), classes, skipped


def test_no_attribute_is_read_that_is_never_assigned():
    """Every hit is a call that raises AttributeError, every time, in production."""
    findings, classes, _ = _sweep_undefined()
    assert not findings, (
        f"{len(findings)} undefined attribute read(s) — each raises on every call:\n  "
        + "\n  ".join(findings)
        + "\n\nAssign it, or delete the branch. And check what repairing it ACTIVATES: "
          "material_visual_search's dead fallback was hiding a query with no tenant "
          "predicate, so fixing the attribute alone would have created a cross-tenant read."
    )


def test_the_undefined_sweep_still_examines_the_tree():
    """"No findings" is equally true of a clean tree and of a sweep that stopped
    matching. Only the count separates them."""
    _, classes, skipped = _sweep_undefined()
    assert classes > 150, (
        f"the sweep only examined {classes} classes; it covered ~300 when written, so it "
        "has probably stopped recognising them rather than the tree having shrunk"
    )
    # `skipped` is LARGER than `classes` and that is expected, not a defect: almost
    # every class in this tree is a pydantic BaseModel, whose fields come from the base
    # machinery. What matters is that the examined count stays healthy, which the floor
    # above pins. Recorded here so the ratio is never "fixed" by loosening the skip rule.
    assert skipped > 0


# ─────────────────────────────────────────────────────────────────────────────
# Shape 2 — the double `.client` unwrap
# ─────────────────────────────────────────────────────────────────────────────

def _self_unwrapped(tree: ast.AST) -> Set[str]:
    """`self.X = <expr>.client` — module-wide, because the assignment is in `__init__`
    and the uses are in methods."""
    out: Set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "client"):
            continue
        for t in node.targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                out.add(t.attr)
    return out


def _local_unwrapped(scope: ast.AST) -> Set[str]:
    """Local names bound to `<expr>.client` WITHIN one function.

    Per-function on purpose. `admin.py` has `sb = _get_sb()` (the wrapper, where
    `sb.client` is correct) in one function and `sb = get_supabase_client().client` in
    another; tracking locals module-wide credits the second assignment to the first
    function's uses and reports two findings that are not real. That is exactly the
    false positive the hand sweep in #22 produced and then withdrew.
    """
    out: Set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not scope:
            continue  # nested function: its own scope
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "client"):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                out.add(t.id)
    return out


def _double_unwraps_in(scope: ast.AST, locals_: Set[str], selfs: Set[str], path: Path) -> List[str]:
    found = []
    for node in ast.walk(scope):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "client"):
            continue
        base = node.value.value
        if isinstance(base, ast.Name) and base.id in locals_:
            key = base.id
        elif (isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
              and base.value.id == "self" and base.attr in selfs):
            key = f"self.{base.attr}"
        else:
            continue
        found.append(f"{_rel(path)}:{node.lineno}  {key}.client.{node.attr}")
    return found


def _sweep_double_unwrap() -> List[str]:
    findings: List[str] = []
    for path in _py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        selfs = _self_unwrapped(tree)
        functions = [n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for fn in functions:
            findings += _double_unwraps_in(fn, _local_unwrapped(fn), selfs, path)
        # module-level code, outside any function
        if selfs:
            module_only = ast.Module(
                body=[n for n in tree.body
                      if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
                type_ignores=[],
            )
            findings += _double_unwraps_in(module_only, _local_unwrapped(module_only), selfs, path)
    return sorted(set(findings))


def test_no_client_is_unwrapped_twice():
    """`x = something.client` makes `x` the PostgREST client already, so `x.client.table(...)`
    raises. The undefined-attribute sweep above cannot see it: the name IS assigned."""
    findings = _sweep_double_unwrap()
    assert not findings, (
        f"{len(findings)} double `.client` unwrap(s) — each raises on every call:\n  "
        + "\n  ".join(findings)
        + "\n\nDrop the second `.client`. Check the handler around it before assuming the "
          "impact is small: one of these returned [] rather than raising, so the feature "
          "reported 'no matches' for its entire life."
    )


def test_the_shapes_the_sweeps_are_meant_to_catch_are_actually_caught():
    """The sweeps are only worth their runtime if they FIRE. Both are run here against
    synthetic source carrying the exact defect, so a refactor that quietly breaks the
    matching fails this instead of reporting a clean tree."""
    undefined_src = ast.parse(
        "class C:\n"
        "    def __init__(self):\n"
        "        self.supabase_client = make()\n"
        "    def go(self):\n"
        "        return self.supabase.client.table('x')\n"
    )
    cls = next(n for n in ast.walk(undefined_src) if isinstance(n, ast.ClassDef))
    assigned, dynamic = _assigned_attrs(cls)
    assert not dynamic
    assert "supabase" not in assigned and "supabase_client" in assigned
    assert any(attr == "supabase" for attr, _ in _read_attrs(cls))

    double_src = ast.parse(
        "class C:\n"
        "    def __init__(self, w):\n"
        "        self.supabase = w.client\n"
        "    def go(self):\n"
        "        return self.supabase.client.table('x')\n"
    )
    assert "supabase" in _self_unwrapped(double_src)

    # And the false positive the original hand sweep produced, which is the reason
    # locals are scoped PER FUNCTION: `f` binds the WRAPPER (so `sb.client` is correct)
    # while `g` binds the client. Tracking locals module-wide credits g's assignment to
    # f's use and reports a finding that is not real — twice, on admin.py.
    ok_src = ast.parse(
        "def f():\n"
        "    sb = get_supabase_client()\n"
        "    return sb.client.table('x')\n"
        "def g():\n"
        "    sb = get_supabase_client().client\n"
        "    return sb.table('x')\n"
    )
    f_fn, g_fn = (n for n in ok_src.body if isinstance(n, ast.FunctionDef))
    assert not _local_unwrapped(f_fn), "the wrapper form must not be flagged"
    assert _local_unwrapped(g_fn) == {"sb"}, "the unwrapped form must be recognised"
    assert not _double_unwraps_in(f_fn, _local_unwrapped(f_fn), set(), Path("x.py")), (
        "per-function scoping is what stops g()'s assignment reaching f()'s use"
    )
