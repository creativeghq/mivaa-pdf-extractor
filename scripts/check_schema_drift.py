#!/usr/bin/env python3
"""Schema-drift lint for `app/` — does this checkout still agree with the live schema?

WHAT IT CATCHES
---------------
A `.select("id, name, material_type")` naming a column that no longer exists. PostgREST
rejects the whole request, so the call site is a hard runtime error from the moment the
migration lands — and it stays invisible until something runs that path.

Audit mivaa#26 found ten such references across five call sites by hand, every one on a
field the platform had moved into a jsonb column. `agent_routes` was the worst: the
material-tagger handler logged "Material tagging started" and then failed, every time,
for as long as it had existed.

WHY MIVAA NEEDS ITS OWN
-----------------------
The platform repo already understood this class and built `npm run schema:writers` for
it. MIVAA is a separate repository and was not covered by it: the `products` columns were
dropped, the platform's writers were fixed, and MIVAA's five readers were not, because
nothing checked them. This is that lint, ported — same contract, Python side.

WHY IT NEEDS THE DATABASE
-------------------------
Migrations are applied straight to the live project; there are no migration files in
either repository. So neither repo contains a statement of what the schema IS, and a
repo-only guard would be comparing a checkout against nothing. The registry comes from
`schema_column_registry()`, which is SECURITY DEFINER and granted to `service_role` only.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not try to understand every query shape. A select containing `*`, an f-string, a
variable, or an embedded resource (`products(name)`) is SKIPPED rather than guessed at —
and every skip is counted and printed, because a guard that silently narrows its own
coverage is the exact failure mode this repository keeps finding. What it checks, it
checks exactly.

It also covers only the LOUD half of schema drift. The silent half — `select('*')` then
`.get('missing_column')` returning None — is #25 M12-1, where visual similarity became a
constant, and no lint can see it. That half needs the code to stop guessing.

USAGE
-----
    SUPABASE_SERVICE_ROLE_KEY=... python scripts/check_schema_drift.py

Runs in CI as a gate (deploy.yml already carries the key), and should be run by hand
straight after applying any migration that drops or renames a column — that is the
earliest possible catch, because the migration is live at that moment and CI has not run.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

SKIP_DIRS = {"__pycache__", ".venv", "venv", "node_modules", ".git"}

#: A registry that arrives INCOMPLETE must fail rather than pass: it describes a smaller
#: database, so every table it lost becomes "unknown table" and is silently skipped. The
#: live project carries ~517 public tables; this floor only catches a truncated response.
MIN_TABLES = 300

#: Tables whose columns are legitimately dynamic, or that this lint cannot reason about.
#: Shrink-only, and each entry states why.
IGNORED_TABLES: Dict[str, str] = {}


@dataclass(frozen=True)
class Usage:
    table: str
    column: str
    path: str
    line: int


@dataclass
class Skip:
    reason: str
    path: str
    line: int
    detail: str


# ── the live schema ──────────────────────────────────────────────────────────

def fetch_registry() -> Dict[str, Set[str]]:
    url = (os.getenv("SUPABASE_URL") or "https://bgbavxtjlbvgplozizxu.supabase.co").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
    if not key:
        raise SystemExit(
            "SUPABASE_SERVICE_ROLE_KEY is required — the live schema is the reference, and "
            "there is nothing in this repository to compare against without it."
        )

    req = urllib.request.Request(
        f"{url}/rest/v1/rpc/schema_column_registry",
        data=b"{}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"schema_column_registry -> {e.code} {e.read()[:200]!r}. It is granted to "
            "service_role only; an anon key gets 404."
        ) from e

    tables = (payload or {}).get("tables")
    if not isinstance(tables, dict):
        raise SystemExit("schema_column_registry returned no `tables` object")

    registry = {
        name: set(spec.get("columns") or [])
        for name, spec in tables.items()
        if isinstance(spec, dict)
    }
    if len(registry) < MIN_TABLES:
        raise SystemExit(
            f"schema_column_registry returned only {len(registry)} tables (expected "
            f">= {MIN_TABLES}). A truncated registry silently narrows this check to "
            "nothing, so it fails instead of passing."
        )
    return registry


# ── what the checkout asks for ───────────────────────────────────────────────

def _table_of(node: ast.AST) -> Optional[str]:
    """Walk back down a builder chain to the `.table("x")` that started it.

    `client.table("x").select("a").eq(...)` parses inside-out, so the `.select` Call's
    `func.value` is the `.table` Call. Anything that is not a plain string literal —
    `.table(name)` — returns None and the whole chain is skipped.
    """
    cur = node
    while isinstance(cur, ast.Call):
        func = cur.func
        if isinstance(func, ast.Attribute):
            if func.attr == "table":
                if cur.args and isinstance(cur.args[0], ast.Constant) and isinstance(cur.args[0].value, str):
                    return cur.args[0].value
                return None
            cur = func.value
        else:
            return None
    return None


def _columns_of(arg: ast.AST) -> Tuple[Optional[List[str]], str]:
    """The column names in a `.select(...)` argument, or (None, why-it-was-skipped)."""
    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
        return None, "not a string literal"
    raw = arg.value
    if "*" in raw:
        return None, "contains *"
    if "(" in raw or ")" in raw:
        # `products(name, sku)` — an embedded resource, whose columns belong to a
        # different table. Resolving that needs the FK graph, which is out of scope.
        return None, "embedded resource"
    cols: List[str] = []
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        # `alias:real_column` — the real column is what PostgREST resolves.
        if ":" in name:
            name = name.split(":", 1)[1].strip()
        if not name or "!" in name or "." in name:
            return None, "modifier syntax"
        cols.append(name)
    return (cols, "") if cols else (None, "empty")


def collect(paths: List[Path]) -> Tuple[List[Usage], List[Skip]]:
    usages: List[Usage] = []
    skips: List[Skip] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as e:
            raise SystemExit(f"{path.relative_to(ROOT)}: {e}")
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "select" or not node.args:
                continue
            table = _table_of(node.func.value)
            if table is None:
                skips.append(Skip("table not a literal", rel, node.lineno, ""))
                continue
            cols, why = _columns_of(node.args[0])
            if cols is None:
                skips.append(Skip(why, rel, node.lineno, table))
                continue
            for col in cols:
                usages.append(Usage(table, col, rel, node.lineno))
    return usages, skips


# ── the verdict ──────────────────────────────────────────────────────────────

def lint(registry: Dict[str, Set[str]], usages: List[Usage]) -> Tuple[List[Usage], int, int]:
    stale: List[Usage] = []
    checked = 0
    unknown_table = 0
    for u in usages:
        if u.table in IGNORED_TABLES:
            continue
        cols = registry.get(u.table)
        if cols is None:
            # Not a public table — a storage bucket name, a view outside the registry,
            # or a typo'd table. Counted, not failed: this lint is about columns.
            unknown_table += 1
            continue
        checked += 1
        if u.column not in cols:
            stale.append(u)
    return stale, checked, unknown_table


def main() -> int:
    files = sorted(
        p for p in APP.rglob("*.py")
        if not any(part in SKIP_DIRS for part in p.parts)
    )
    usages, skips = collect(files)
    registry = fetch_registry()
    stale, checked, unknown_table = lint(registry, usages)

    by_reason: Dict[str, int] = {}
    for s in skips:
        by_reason[s.reason] = by_reason.get(s.reason, 0) + 1

    print(
        f"Schema drift: {checked} (table, column) pairs checked against {len(registry)} "
        f"live tables across {len(files)} files."
    )
    print(
        "  skipped: "
        + (", ".join(f"{n} {r}" for r, n in sorted(by_reason.items())) or "none")
        + f"; {unknown_table} pair(s) on tables outside the public registry"
    )

    if not stale:
        print("Schema drift OK - every explicitly selected column exists.")
        return 0

    print(f"\n{len(stale)} STALE column reference(s) — PostgREST rejects each of these:\n")
    for u in sorted(stale, key=lambda x: (x.path, x.line)):
        print(f"  {u.path}:{u.line}  {u.table}.{u.column}  does not exist")
    print(
        "\nFix the reader. If the column moved into a jsonb column (this is what happened "
        "to products.material_type / color / finish / collection / tags / manufacturer / "
        "image_url), read it from there — and if the call site was dead because of this, "
        "check what repairing it ACTIVATES before you repair it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
