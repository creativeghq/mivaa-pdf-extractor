"""Is this generated SQL a single read-only SELECT over the tables we loaded, and nothing else?

Two layers stand between a model's SQL and the engine, and this is the first (the second
is `loader.lock_down`, which makes DuckDB refuse files, extensions and configuration
even if a query got past here). Both are needed because DuckDB can reach the filesystem
and the network from INSIDE a plain SELECT:

    SELECT * FROM read_csv('/etc/passwd')   -- parses as a SELECT
    SELECT * FROM glob('/**')               -- so does this

So beyond refusing writes and DDL, this refuses any table function, any table not
loaded by us, and ANY FUNCTION IT DOES NOT KNOW — an allow-list, not a deny-list, which
is what makes it hold for DuckDB extension functions that do not exist yet.

Pure with respect to the app: needs only `sqlglot`. Loaded by path in the unit test when
sqlglot is importable, and the shape is pinned statically when it is not (CI installs
pytest alone).
"""

from __future__ import annotations

from typing import Iterable, Set

import sqlglot
from sqlglot import exp


class SqlRejected(ValueError):
    """The SQL was refused. `reason` is fed back to the model on the next attempt."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


#: Scalar, aggregate and window functions a question about a spreadsheet can need. Anything
#: not listed is refused by name, which is the point: `read_csv`, `glob`, `read_blob`,
#: `read_text`, `httpfs`-anything and next year's extension function are all "not listed".
ALLOWED_FUNCTIONS: Set[str] = {
    # aggregates
    "count", "sum", "avg", "min", "max", "median", "mode", "stddev", "stddev_pop", "stddev_samp",
    "variance", "var_pop", "var_samp", "string_agg", "group_concat", "listagg", "list", "array_agg",
    "first", "last", "any_value", "bool_and", "bool_or", "quantile", "quantile_cont", "quantile_disc",
    "approx_count_distinct", "countif", "sumif",
    # window
    "row_number", "rank", "dense_rank", "percent_rank", "ntile", "lag", "lead", "first_value", "last_value", "nth_value",
    # numeric
    "round", "floor", "ceil", "ceiling", "abs", "mod", "power", "pow", "sqrt", "ln", "log", "log10", "log2", "exp",
    "sign", "greatest", "least", "trunc", "truncate", "random",
    # null handling / conditionals
    "coalesce", "nullif", "ifnull", "if", "iif", "isnull", "isnotnull",
    # casting
    "cast", "try_cast", "typeof",
    # text
    "lower", "upper", "trim", "ltrim", "rtrim", "length", "len", "char_length", "character_length", "substr", "substring",
    "left", "right", "concat", "concat_ws", "replace", "reverse", "repeat", "lpad", "rpad", "strip_accents",
    "contains", "starts_with", "prefix", "suffix", "ends_with", "position", "strpos", "instr", "split_part",
    "string_split", "str_split", "regexp_matches", "regexp_replace", "regexp_extract", "like", "ilike", "similar_to",
    "initcap", "format", "printf", "levenshtein", "jaro_winkler_similarity", "damerau_levenshtein",
    # dates
    "now", "current_date", "current_timestamp", "today", "date_trunc", "date_part", "datepart", "extract", "strftime",
    "strptime", "year", "month", "day", "hour", "minute", "second", "dayofweek", "dayofyear", "weekofyear", "week",
    "quarter", "date_diff", "datediff", "date_add", "dateadd", "date_sub", "age", "make_date", "last_day", "epoch",
    "epoch_ms", "to_timestamp", "to_date",
    # lists (bounded, no I/O)
    "array_length", "list_contains", "array_contains", "list_sort", "unnest", "generate_series", "range",
    # sqlglot's own names for expressions it folds into functions
    "anonymous",
}

#: Expression types that are never read-only.
FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Merge,
    exp.Command, exp.Transaction, exp.Commit, exp.Rollback, exp.Set, exp.Pragma,
    exp.Copy, exp.AlterTable if hasattr(exp, "AlterTable") else exp.Alter,
)


def _cte_names(tree: exp.Expression) -> Set[str]:
    return {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name}


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name or "").lower()
    # sqlglot names expressions by class; `sql_name()` is the SQL spelling.
    try:
        return node.sql_name().lower()
    except Exception:  # noqa: BLE001 — a class with no sql_name is still a function to us
        return node.key.lower()


def validate_sql(sql: str, allowed_tables: Iterable[str], *, max_rows: int = 100) -> str:
    """Return the SQL to run (a LIMIT applied or clamped), or raise `SqlRejected`.

    - exactly one statement, and it is a SELECT (a WITH ... SELECT, or a UNION of SELECTs)
    - no write, DDL, transaction, PRAGMA, SET, COPY, ATTACH/INSTALL/LOAD (those parse to Command)
    - every table is one we loaded or a CTE the query itself defines — no table functions
    - every function is on the allow-list
    - LIMIT is present and at most `max_rows`
    """
    if not isinstance(sql, str) or not sql.strip():
        raise SqlRejected("empty SQL")
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except Exception as e:  # noqa: BLE001 — sqlglot raises several error types
        raise SqlRejected(f"could not parse the SQL: {e}") from e
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SqlRejected(f"expected exactly one statement, got {len(statements)}")
    tree = statements[0]

    if isinstance(tree, FORBIDDEN_NODES):
        raise SqlRejected(f"only a SELECT is allowed, not {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SqlRejected(f"only a SELECT is allowed; found {node.key.upper()}")

    # The outermost statement must be a query.
    body = tree
    if isinstance(body, exp.With):
        body = body.this
    if not isinstance(body, (exp.Select, exp.Union, exp.Subquery)):
        raise SqlRejected(f"only a SELECT is allowed, not {tree.key.upper()}")

    allowed = {t.lower() for t in allowed_tables}
    ctes = _cte_names(tree)
    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        if table.db or table.catalog:
            raise SqlRejected(f"cross-schema reference is not allowed: {table.sql()}")
        if not name:
            # A function in FROM position (read_csv(...), glob(...)) parses as a Table wrapping a Func.
            raise SqlRejected(f"table functions are not allowed: {table.sql()}")
        if name not in allowed and name not in ctes:
            raise SqlRejected(f"unknown table '{name}'; loaded tables are: {', '.join(sorted(allowed))}")
        if isinstance(table.this, exp.Func):
            raise SqlRejected(f"table functions are not allowed: {table.sql()}")

    for func in tree.find_all(exp.Func):
        fname = _function_name(func)
        if fname not in ALLOWED_FUNCTIONS:
            raise SqlRejected(f"function '{fname}' is not allowed")

    # A LIMIT, and a small one. Set on the outermost query.
    limit_node = body.args.get("limit") if isinstance(body, (exp.Select, exp.Union)) else None
    if limit_node is None:
        body = body.limit(max_rows)
        if isinstance(tree, exp.With):
            tree.set("this", body)
        else:
            tree = body
    else:
        try:
            current = int(limit_node.expression.this)
        except (TypeError, ValueError, AttributeError):
            raise SqlRejected("LIMIT must be a plain integer") from None
        if current > max_rows:
            body.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return tree.sql(dialect="duckdb")
