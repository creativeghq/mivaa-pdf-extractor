"""Files → tables in an in-memory DuckDB, profiled, then the engine is locked.

What a real supplier spreadsheet looks like: a title in row 1, a blank spacer, the header
in row 4, a "Σύνολο" row after every group, prices written `1.234,56`, a notes block at
the bottom. The clean-up here is DETERMINISTIC and cheap — no model call — and every
removal is logged so a row count that looks wrong is answerable from the log.

Order matters and is the whole security story: the files are read while external
access is ON (our loader, our paths), then `lock_down` turns it OFF and locks the
configuration BEFORE the first generated query runs. After that DuckDB itself refuses to
read files, install extensions, attach databases or re-enable any of it.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SPREADSHEET_EXTENSIONS = {"xlsx", "xlsm", "xls"}
DELIMITED_EXTENSIONS = {"csv", "tsv", "txt"}
SUPPORTED_EXTENSIONS = SPREADSHEET_EXTENSIONS | DELIMITED_EXTENSIONS

MAX_ROWS_PER_TABLE = 200_000
MAX_COLUMNS = 200
LOW_CARDINALITY = 12
PROFILE_SAMPLES = 6

_EURO_NUMBER = re.compile(r"^\s*-?\d{1,3}(\.\d{3})+(,\d+)?\s*$|^\s*-?\d+,\d+\s*$")
_PLAIN_NUMBER = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_pct: float
    distinct: int
    samples: List[str] = field(default_factory=list)
    minimum: Optional[str] = None
    maximum: Optional[str] = None


@dataclass
class TableInfo:
    name: str
    source: str
    rows: int
    columns: List[ColumnProfile]
    dropped_rows: int = 0
    notes: List[str] = field(default_factory=list)


def clean_identifier(raw: Any, fallback: str = "col") -> str:
    """The header text as a quoted SQL identifier — Unicode kept, because DuckDB quotes it fine
    and the model must be able to name the column the sheet names it. "Κωδικός" stays "Κωδικός";
    folding it to ASCII produced `col`, `col_2`, ... and a schema that meant nothing."""
    s = unicodedata.normalize("NFKC", str(raw if raw is not None else ""))
    s = s.replace('"', "'").replace("\x00", "")
    s = re.sub(r"\s+", " ", s).strip()
    if not s or s.startswith("Unnamed:"):
        return fallback
    return s[:60]


def unique_names(names: Sequence[Any], fallback: str) -> List[str]:
    out: List[str] = []
    seen: Dict[str, int] = {}
    for i, n in enumerate(names):
        base = clean_identifier(n, f"{fallback}{i + 1}")
        k = seen.get(base, 0)
        seen[base] = k + 1
        out.append(base if k == 0 else f"{base}_{k + 1}")
    return out


def parse_european_number(value: Any) -> Optional[float]:
    """`1.234,56` → 1234.56; `12,5` → 12.5; a plain `1234.5` unchanged; anything else None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    if _EURO_NUMBER.match(s):
        return float(s.replace(".", "").replace(",", "."))
    if _PLAIN_NUMBER.match(s):
        return float(s)
    return None


def _tidy_frame(df, source: str):
    """Header detection + edge trimming + structural subtotal removal, all deterministic."""
    import pandas as pd

    notes: List[str] = []
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return df, 0, notes
    df = df.reset_index(drop=True)

    # Header = first row that is mostly populated AND mostly text.
    header_idx = 0
    for i in range(min(len(df), 15)):
        row = df.iloc[i]
        populated = row.notna().sum()
        if populated < max(2, 0.5 * df.shape[1]):
            continue
        texty = sum(1 for v in row if isinstance(v, str) and not _PLAIN_NUMBER.match(v) and not _EURO_NUMBER.match(v))
        if texty >= 0.5 * populated:
            header_idx = i
            break
    if header_idx > 0:
        notes.append(f"header found on row {header_idx + 1}; {header_idx} title row(s) above it dropped")
    header = [str(v) if pd.notna(v) else None for v in df.iloc[header_idx]]
    df = df.iloc[header_idx + 1:].reset_index(drop=True)
    df.columns = unique_names(header, "col")
    df = df.dropna(how="all")

    # A trailing notes block ("Σημειώσεις:", "prices exclude VAT"): from the bottom, rows with
    # fewer than half their cells filled and no number in them are commentary, not data. Stops at
    # the first row that looks like data, so a sparse LAST data row is not eaten.
    if df.shape[1] >= 3:
        trailing = 0
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            filled = row.notna().sum()
            has_number = any(parse_european_number(v) is not None for v in row if pd.notna(v))
            if filled < 0.5 * df.shape[1] and not has_number:
                trailing += 1
            else:
                break
        if trailing:
            df = df.iloc[: len(df) - trailing]
            notes.append(f"dropped {trailing} trailing note row(s): mostly empty, no numbers")

    # European decimals: convert a column only when EVERY non-null value parses as a number.
    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            non_null = series.dropna()
            if len(non_null) and all(parse_european_number(v) is not None for v in non_null):
                df[col] = series.map(parse_european_number)

    # Subtotal rows, structurally: descriptive (text) columns blank, numeric columns filled.
    text_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols = [c for c in df.columns if c not in text_cols]
    dropped = 0
    if text_cols and num_cols and len(df) >= 6:
        mask = df[text_cols].isna().all(axis=1) & df[num_cols].notna().any(axis=1)
        # Only on sheets that look messy — a tidy sheet never trips this.
        if 0 < mask.sum() <= 0.3 * len(df):
            dropped = int(mask.sum())
            df = df[~mask]
            notes.append(f"dropped {dropped} subtotal/section row(s): descriptive columns blank, numbers present")
    if len(df) > MAX_ROWS_PER_TABLE:
        notes.append(f"truncated to the first {MAX_ROWS_PER_TABLE} rows")
        df = df.head(MAX_ROWS_PER_TABLE)
    if df.shape[1] > MAX_COLUMNS:
        notes.append(f"truncated to the first {MAX_COLUMNS} columns")
        df = df.iloc[:, :MAX_COLUMNS]
    for n in notes:
        logger.info("tabular loader [%s]: %s", source, n)
    return df.reset_index(drop=True), dropped, notes


def _read_delimited(data: bytes):
    import pandas as pd

    for enc in ("utf-8-sig", "utf-8", "cp1253", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None, dtype=object)


def _read_workbook(data: bytes) -> Dict[str, Any]:
    import pandas as pd

    return pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=object)


def load_sources(con, sources: Sequence[Tuple[str, bytes]]) -> List[TableInfo]:
    """Load every (filename, bytes) into `con`. One table per CSV, one per non-empty sheet."""
    tables: List[TableInfo] = []
    taken: set = set()

    def register(table_base: str, df, source: str, dropped: int, notes: List[str]) -> None:
        if df.empty or df.shape[1] == 0:
            logger.info("tabular loader: %s is empty after clean-up — not loaded", source)
            return
        name = clean_identifier(table_base, "sheet")
        k = 2
        base = name
        while name in taken:
            name = f"{base}_{k}"
            k += 1
        taken.add(name)
        con.register(f"_df_{name}", df)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "_df_{name}"')
        con.unregister(f"_df_{name}")
        tables.append(TableInfo(name=name, source=source, rows=len(df), columns=profile(con, name), dropped_rows=dropped, notes=notes))

    for filename, data in sources:
        ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
        stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if ext in DELIMITED_EXTENSIONS:
            df, dropped, notes = _tidy_frame(_read_delimited(data), filename)
            register(stem, df, filename, dropped, notes)
        elif ext in SPREADSHEET_EXTENSIONS:
            for sheet, raw in _read_workbook(data).items():
                df, dropped, notes = _tidy_frame(raw, f"{filename}/{sheet}")
                register(f"{sheet}" if len(sources) == 1 else f"{stem}_{sheet}", df, f"{filename} / {sheet}", dropped, notes)
        else:
            raise ValueError(f"unsupported file type .{ext} — use csv, tsv, txt, xlsx, xlsm or xls")
    if not tables:
        raise ValueError("no table could be read from the file(s)")
    return tables


def profile(con, table: str) -> List[ColumnProfile]:
    cols = con.execute(f'DESCRIBE "{table}"').fetchall()
    total = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] or 0
    out: List[ColumnProfile] = []
    for col_name, col_type, *_ in cols:
        q = f'SELECT count(*) - count("{col_name}"), count(DISTINCT "{col_name}") FROM "{table}"'
        nulls, distinct = con.execute(q).fetchone()
        prof = ColumnProfile(
            name=col_name, dtype=str(col_type),
            null_pct=round(100.0 * nulls / total, 1) if total else 0.0,
            distinct=int(distinct or 0),
        )
        if distinct and distinct <= LOW_CARDINALITY:
            rows = con.execute(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT {PROFILE_SAMPLES}').fetchall()
            prof.samples = [str(r[0])[:40] for r in rows]
        elif str(col_type).upper() in ("BIGINT", "INTEGER", "DOUBLE", "DECIMAL", "FLOAT", "HUGEINT", "SMALLINT") or str(col_type).upper().startswith("DECIMAL"):
            mn, mx = con.execute(f'SELECT min("{col_name}"), max("{col_name}") FROM "{table}"').fetchone()
            prof.minimum, prof.maximum = (None if mn is None else str(mn)), (None if mx is None else str(mx))
        else:
            rows = con.execute(f'SELECT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT 3').fetchall()
            prof.samples = [str(r[0])[:40] for r in rows]
        out.append(prof)
    return out


def lock_down(con, *, memory_limit: str = "512MB") -> None:
    """Close the engine to the outside world. Irreversible for the life of the connection."""
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")


def schema_for_prompt(tables: Sequence[TableInfo]) -> str:
    """What the model sees: the data, not just its shape. Low-cardinality VALUES are listed —
    that is what stops a query filtering on a category that does not exist."""
    lines: List[str] = []
    for t in tables:
        lines.append(f'TABLE "{t.name}"  -- from {t.source}, {t.rows} row(s)' + (f", {t.dropped_rows} subtotal row(s) removed" if t.dropped_rows else ""))
        lines.append("(")
        for c in t.columns:
            bits = [f'  "{c.name}" {c.dtype}']
            detail = []
            if c.null_pct:
                detail.append(f"{c.null_pct}% NULL")
            detail.append(f"{c.distinct} distinct")
            if c.samples and c.distinct <= LOW_CARDINALITY:
                detail.append("values: " + ", ".join(c.samples))
            elif c.minimum is not None or c.maximum is not None:
                detail.append(f"range {c.minimum}..{c.maximum}")
            elif c.samples:
                detail.append("e.g. " + ", ".join(c.samples))
            bits.append("  -- " + "; ".join(detail))
            lines.append("".join(bits))
        lines.append(")")
    return "\n".join(lines)
