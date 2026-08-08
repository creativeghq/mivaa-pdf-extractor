"""
Guard: every `from <our own module> import <name>` must actually resolve.

This catches a bug class the linter structurally CANNOT. `ruff --select F821`
reports undefined names, but an import statement *binds* the name it imports — so

    from ..services.embeddings.real_embeddings_service import get_embeddings_service

is invisible to F821 even when that module has never exported `get_embeddings_service`.
The failure surfaces only as an ImportError at the moment the line executes. When the
import is lazy (inside a function, as this codebase often does to break cycles) that
moment may be months after the code shipped.

That exact case was live in `app/api/admin.py`: an admin OCR-reprocessing endpoint
imported a factory function that has never existed in the target module, which exports
a class instead. Dead on arrival, and clean to every reader and every tool.

Same family as `JobTracker` (a class deleted while six call sites kept calling it) and
`vecs_service.search_similar` (a method that never existed) — the platform's
characteristic failure: code on a path nobody exercised, reporting nothing.

Resolves relative imports properly, including the level-1-inside-`__init__.py` case
where `.` means the package itself rather than its parent. Getting that wrong produces
false positives on every well-formed package `__init__`. Recognises legitimate
submodule imports (`from pkg import module_file`) as resolvable.

Pure AST over the source tree: no `app` import, no DB, ~1s.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"

# Imports that cannot be resolved statically, each with the reason it is exempt.
# Shrink-only: if you are adding to this, prefer fixing the import.
KNOWN_UNRESOLVABLE: set = set()


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _own_package(path: Path) -> list:
    """The package that this module's relative imports resolve against."""
    parts = list(path.relative_to(_ROOT).with_suffix("").parts)
    # For a/b/__init__.py the package IS a.b, so `.` means a.b — not a.
    # For a/b/mod.py the package is a.b as well.
    return parts[:-1]


def _resolve(node: ast.ImportFrom, importer: Path):
    if not node.level:
        return node.module
    base = _own_package(importer)
    climb = node.level - 1  # one dot = own package; each extra dot climbs
    if climb:
        base = base[:-climb] if climb <= len(base) else []
    return ".".join([*base, node.module]) if node.module else ".".join(base)


def _module_level_names(tree: ast.AST) -> set:
    """Every name the module binds and therefore can export."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_every_intra_app_import_resolves():
    py_files = sorted(_APP.rglob("*.py"))
    assert len(py_files) > 100, "found suspiciously few app modules — did the glob break?"

    all_modules = {_module_name(p) for p in py_files}
    provides = {}
    for path in py_files:
        try:
            provides[_module_name(path)] = _module_level_names(
                ast.parse(path.read_text(encoding="utf-8"))
            )
        except SyntaxError as e:
            pytest.fail(f"{path} does not parse: {e}")

    unresolvable = []
    for path in py_files + sorted((_ROOT / "tests").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _resolve(node, path)
            if target not in provides:
                continue  # third-party or outside app/ — not ours to check
            for alias in node.names:
                if alias.name == "*":
                    continue
                if alias.name in provides[target]:
                    continue
                if f"{target}.{alias.name}" in all_modules:
                    continue  # `from pkg import submodule` is legitimate
                entry = f"{target}.{alias.name}"
                if entry in KNOWN_UNRESOLVABLE:
                    continue
                rel = path.relative_to(_ROOT)
                unresolvable.append(f"{rel}: from {target} import {alias.name}")

    assert not unresolvable, (
        "These imports name something their target module does not provide. Each is an "
        "ImportError waiting for the line to execute — and F821 cannot see it, because "
        "an import binds the name it imports:\n  " + "\n  ".join(sorted(unresolvable))
    )
