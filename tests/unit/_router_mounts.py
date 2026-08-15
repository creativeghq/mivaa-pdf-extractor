"""Resolve which SOURCE FILES serve a given URL prefix, by reading main.py.

Why this exists (issue #15, the headline finding)
-------------------------------------------------
MIVAA has 23 guard tests. They are good tests — clear names, real reasoning in the
docstrings — and all of them pass. That was the problem: several of them scan a
**hardcoded list of files** while reading as though they cover their whole class, and
every defect found in issues #14 and #15 landed in the gap between the two:

    test_paid_route_metering       3 hardcoded doors in 2 files
    test_no_silent_degradation     5 declared subtrees
    test_audit_12_gates_hold       1 declared file (SSRF)
    test_kb_chunk_retrieval        3 declared filenames, one now deleted

The repo already knew: `test_prompts_come_from_the_database` refuses to keep a declared
file list because *"a declared list is exactly the kind that rots"*. And
`test_workspace_binding_coverage`, the one guard using full-tree `APP.rglob("*.py")`,
covers the class with the LEAST evidence of defects. That correlation is the argument.

So: derive, don't declare. A router added to main.py tomorrow is covered the day it is
added, with no one remembering to update a list.

What it does NOT do
-------------------
This is a static reader, not an import of the app. It resolves
`app.include_router(<name>[, prefix=...])` back to the module that `<name>` was
imported from. It cannot see a router mounted through a variable computed at runtime,
and it deliberately raises nothing when it finds none — the CALLER asserts non-empty,
because "found nothing" must fail loudly rather than pass vacuously. That failure mode
is the whole point: a guard that silently scans zero files is worse than no guard,
since it reports green.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional


def _module_to_path(root: Path, module: str) -> Optional[Path]:
    """`app.api.documents.query_routes` -> <root>/app/api/documents/query_routes.py"""
    rel = Path(*module.split("."))
    for candidate in (root / rel.with_suffix(".py"), root / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _router_symbol_sources(root: Path, tree: ast.AST) -> Dict[str, Path]:
    """Map each imported name to the file it came from.

    Handles both shapes main.py uses:
        from app.api.rag_routes import router as rag_router
        from app.api.documents import management_router
    The second resolves to the package `__init__.py`; when that package re-exports a
    router from a submodule, the re-export is followed one hop.
    """
    sources: Dict[str, Path] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        path = _module_to_path(root, node.module)
        if path is None:
            continue
        for alias in node.names:
            local = alias.asname or alias.name
            resolved = path
            # Follow a package re-export (`from .management_routes import router as x`)
            # so the reported file is the one that actually declares the routes.
            if path.name == "__init__.py":
                try:
                    sub = ast.parse(path.read_text(encoding="utf-8"))
                except (SyntaxError, OSError):
                    sub = None
                if sub is not None:
                    for sub_node in ast.walk(sub):
                        if not isinstance(sub_node, ast.ImportFrom) or not sub_node.module:
                            continue
                        for sub_alias in sub_node.names:
                            if (sub_alias.asname or sub_alias.name) != alias.name:
                                continue
                            # relative import inside the package
                            pkg = node.module if sub_node.level else sub_node.module
                            target = _module_to_path(
                                root,
                                f"{pkg}.{sub_node.module}" if sub_node.level else sub_node.module,
                            )
                            if target is not None:
                                resolved = target
            sources[local] = resolved
    return sources


def routers_mounted_at(root: Path, prefix: str, main_py: Optional[Path] = None) -> List[Path]:
    """Source files whose routers end up serving `prefix`.

    A router reaches `prefix` two ways, and both count:
      * `app.include_router(r, prefix="/api/rag")` — mounted there explicitly.
      * `app.include_router(r)` where the router declares `APIRouter(prefix="/api/rag")`
        itself. This is the case that matters most: rag_routes.py carries its own
        prefix, so a guard keyed only on the include_router kwarg would miss the very
        file every duplicate is a duplicate OF.
    """
    main_py = main_py or (root / "app" / "main.py")
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    symbols = _router_symbol_sources(root, tree)

    wanted = prefix.rstrip("/")
    found: List[Path] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "include_router"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue

        path = symbols.get(node.args[0].id)
        if path is None:
            continue

        mount_prefix = None
        for kw in node.keywords:
            if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                mount_prefix = str(kw.value.value).rstrip("/")

        if mount_prefix == wanted:
            if path not in found:
                found.append(path)
            continue

        # No explicit mount prefix — does the router declare its own?
        if mount_prefix is None:
            try:
                src = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if f'APIRouter(prefix="{wanted}"' in src or f"APIRouter(prefix='{wanted}'" in src:
                if path not in found:
                    found.append(path)

    return found
