"""Guard: the text half of material visual search never reads across tenants (M3-1, #16).

`_search_by_text_description` queried `products` and `document_images` on the
service-role client with no tenant predicate, while the VECS half of the very
same function built its filter from `request.workspace_id` and failed closed.
So one search returned tenant-scoped visual hits merged with platform-wide text
hits — names, descriptions and metadata from every workspace on the platform.

Nothing could catch it from the outside. MIVAA has no RLS backstop by design,
every row returned is a valid row, and a cross-tenant leak looks exactly like a
generous search. So the guard has to watch the queries themselves.

There is no "shared catalog" reading that would make this legal. Cross-workspace
publishing on this platform is an explicit granted act into a separate surface —
`workspace_catalog_grants`, `catalog_master_products`, `marketplace_listings` —
never a widened read of `products`, whose `workspace_id` is NOT NULL and whose
RLS select policy is `is_workspace_member(workspace_id)`.

Static, not runtime: CI installs pytest alone (`deploy.yml`) and these unit
tests import nothing from `app`, so this parses the source instead. The upside
is that a NEWLY ADDED query against a tenant table fails this test by default
rather than needing the guard to be taught about it first.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SERVICE = REPO / "app" / "services" / "search" / "material_visual_search_service.py"
HELPER = REPO / "app" / "utils" / "postgrest_filters.py"

TARGET = "_search_by_text_description"
CALLER = "_perform_database_search"

# Tables read in this helper that hold tenant rows and must never be read
# unscoped. image_product_associations is deliberately absent: it carries no
# workspace_id column, so its predicate lands on the embedded document_images.
TENANT_TABLES = {"products", "document_images"}


def _load_escape_like():
    """Import the pure helper by path, bypassing app/__init__ and its deps."""
    spec = importlib.util.spec_from_file_location("_postgrest_filters", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.escape_like


escape_like = _load_escape_like()


@pytest.fixture(scope="module")
def tree():
    return ast.parse(SERVICE.read_text(encoding="utf-8"))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {SERVICE.name} - re-point this guard")


def _const(node):
    return node.value if isinstance(node, ast.Constant) else None


def _chain(call):
    """Unwind `a.table('x').select(...).eq(...)` into {method: [args], ...}."""
    steps = {}
    node = call
    while isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Attribute):
            break
        steps.setdefault(func.attr, []).append(node.args)
        node = func.value
    return steps


def _query_chains(func):
    """Every PostgREST builder chain in `func`, as (table_name, steps)."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "execute"):
            continue
        steps = _chain(node)
        table_args = steps.get("table")
        if not table_args:
            continue
        name = _const(table_args[0][0]) if table_args[0] else None
        if name:
            yield name, steps


def _eq_columns(steps):
    return {
        _const(args[0])
        for args in steps.get("eq", [])
        if args and _const(args[0]) is not None
    }


def test_workspace_id_is_a_required_parameter(tree):
    func = _find_function(tree, TARGET)
    names = [a.arg for a in func.args.args]
    assert "workspace_id" in names, (
        f"{TARGET} takes no workspace_id - it reads tenant tables on the "
        "service-role client and cannot scope itself without one"
    )
    # Required means no default. A default would let a caller forget it and
    # silently get the old platform-wide behaviour back.
    defaulted = names[len(names) - len(func.args.defaults):] if func.args.defaults else []
    assert "workspace_id" not in defaulted, (
        "workspace_id must not have a default - an omitted argument would "
        "silently restore the cross-tenant read"
    )


def test_a_missing_workspace_id_raises_instead_of_searching(tree):
    """Fail closed. An unscoped search is worse than no search."""
    func = _find_function(tree, TARGET)
    raises = [n for n in ast.walk(func) if isinstance(n, ast.Raise)]
    assert raises, f"{TARGET} has no raise - a missing workspace_id must not proceed"

    guarded = False
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        checks_ws = any(
            isinstance(sub, ast.Name) and sub.id == "workspace_id"
            for sub in ast.walk(test)
        )
        if checks_ws and any(isinstance(n, ast.Raise) for n in ast.walk(node)):
            guarded = True
    assert guarded, (
        "no `if not workspace_id: raise` guard - the raise must be reachable "
        "from the missing-workspace case specifically"
    )


def test_every_tenant_table_query_carries_a_workspace_predicate(tree):
    func = _find_function(tree, TARGET)
    chains = list(_query_chains(func))
    assert chains, "no queries found - re-point this guard"

    touched = {table for table, _ in chains}
    assert TENANT_TABLES <= touched, (
        f"{TENANT_TABLES - touched} no longer queried here - if a query moved, "
        "move its workspace predicate with it"
    )

    for table, steps in chains:
        if table not in TENANT_TABLES:
            continue
        columns = _eq_columns(steps)
        assert any(c == "workspace_id" or c.endswith(".workspace_id") for c in columns), (
            f"{table} is queried with no workspace predicate (eq columns: {columns or 'none'}). "
            "MIVAA runs on the service-role client with no RLS backstop, so this "
            "returns matching rows from every tenant on the platform."
        )


def test_the_association_join_cannot_bridge_workspaces(tree):
    """image_product_associations has no workspace_id, so the image carries it."""
    func = _find_function(tree, TARGET)
    chains = [(t, s) for t, s in _query_chains(func) if t == "image_product_associations"]
    assert chains, "product images are no longer read - re-point this guard"

    for _, steps in chains:
        select_args = steps.get("select", [[]])[0]
        selected = _const(select_args[0]) if select_args else ""
        assert "document_images!inner" in selected, (
            "the embedded document_images must be an INNER join - with a plain "
            "embed, a filter on the embedded table nulls the child instead of "
            "dropping the association row"
        )
        assert "workspace_id" in selected, (
            "select the embedded workspace_id so the in-Python recheck has it"
        )
        assert "document_images.workspace_id" in _eq_columns(steps), (
            "no tenant filter on the embedded image - an association row can "
            "otherwise point at another workspace's image"
        )


def test_ilike_patterns_are_escaped(tree):
    """A raw `%` or `*` from a user would widen the match to the whole table."""
    func = _find_function(tree, TARGET)

    # Locals whose value was built through escape_like().
    escaped_locals = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        uses_escape = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "escape_like"
            for sub in ast.walk(node.value)
        )
        if uses_escape:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    escaped_locals.add(target.id)

    patterns = [
        args[1]
        for _, steps in _query_chains(func)
        for args in steps.get("ilike", [])
        if len(args) > 1
    ]
    assert patterns, "no ILIKE filters found - re-point this guard"

    for pattern in patterns:
        if isinstance(pattern, ast.Constant):
            continue
        if isinstance(pattern, ast.Name):
            assert pattern.id in escaped_locals, (
                f"ILIKE pattern {pattern.id!r} was not built with escape_like()"
            )
            continue
        if isinstance(pattern, ast.JoinedStr):
            for part in pattern.values:
                if not isinstance(part, ast.FormattedValue):
                    continue
                inner = part.value
                ok = (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "escape_like"
                ) or (isinstance(inner, ast.Name) and inner.id in escaped_locals)
                assert ok, (
                    "raw user input interpolated into an ILIKE pattern - wrap it "
                    "in escape_like()"
                )
            continue
        raise AssertionError(f"unrecognised ILIKE pattern expression: {ast.dump(pattern)}")


def test_the_caller_passes_its_workspace_through(tree):
    """A scoped helper is no use if the call site omits the argument."""
    caller = _find_function(tree, CALLER)
    calls = [
        node
        for node in ast.walk(caller)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == TARGET
    ]
    assert calls, f"{CALLER} no longer calls {TARGET} - re-point this guard"

    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert "workspace_id" in passed, (
            f"{CALLER} calls {TARGET} without workspace_id"
        )
        value = next(kw.value for kw in call.keywords if kw.arg == "workspace_id")
        assert isinstance(value, ast.Attribute) and value.attr == "workspace_id", (
            "workspace_id must come from the request the VECS filter already "
            "uses, not from a literal or a fresh lookup"
        )


# -- the escaping helper itself ------------------------------------------------


@pytest.mark.parametrize(
    "term,expected",
    [
        ("marble", "marble"),
        ("100% wool", r"100\% wool"),
        ("a_b", r"a\_b"),
        ("*", r"\*"),
        ("back\\slash", r"back\\slash"),
        ("", ""),
    ],
)
def test_escape_like_neutralises_wildcards(term, expected):
    assert escape_like(term) == expected


def test_escape_like_escapes_the_backslash_first():
    """Otherwise the escape character it just inserted gets escaped again."""
    assert escape_like("%") == r"\%"
    assert escape_like("\\%") == r"\\\%"
