from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

COMPLETE = "complete"
SUSPECT = "suspect"
TRUNCATED = "truncated"

MIN_LINES_FOR_RATIO = 6
SUSPECT_RATIO = 0.88
MISSING_NUMBERS_TRUNCATED = 0.05
MISSING_NUMBERS_SUSPECT = 0.02
MAX_ROWS_PER_BAND = 18
CUT_SNAP_LINES = 2
BAND_WIDTH_TOLERANCE = 2
HARD_REASONS = frozenset({"unclosed_table", "missing_reference_numbers", "repetition_loop"})

_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)
_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def table_rows(content: str) -> List[List[str]]:
    rows = []
    for row in _ROW.findall(content or ""):
        rows.append([_TAG.sub("", c).strip() for c in _CELL.findall(row)])
    return rows


def numeric_tokens(text: str) -> set:
    text = _TAG.sub("|", text or "")
    out = set()
    for m in _NUMBER.findall(text):
        m = m.replace(".", ",")
        if "," in m or len(m) >= 3:
            out.add(m)
    return out


def count_text_lines(ink_profile: Sequence[float], min_run: int = 5, min_gap: int = 2, min_ink: float = 0.012) -> int:
    return len(text_line_bands(ink_profile, min_run, min_gap, min_ink))


def text_line_bands(ink_profile: Sequence[float], min_run: int = 5, min_gap: int = 2, min_ink: float = 0.012) -> List[Tuple[int, int]]:
    if not ink_profile:
        return []
    ink = [v >= min_ink for v in ink_profile]
    bands: List[Tuple[int, int]] = []
    start: Optional[int] = None
    gap = 0
    for i, flag in enumerate(ink):
        if flag:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                end = i - gap + 1
                if end - start >= min_run:
                    bands.append((start, end))
                start, gap = None, 0
    if start is not None:
        end = len(ink) - gap
        if end - start >= min_run:
            bands.append((start, end))
    return bands


def ink_profile_of(image: Any, dark_below: int = 200) -> List[float]:
    from PIL import Image

    gray = image.convert("L")
    width, height = gray.size
    if width < 1 or height < 1:
        return []
    dark = gray.point(lambda v: 255 if v < dark_below else 0)
    return [v / 255.0 for v in dark.resize((1, height), Image.BOX).getdata()]


def plan_bands(bands: Sequence[Tuple[int, int]], total_height: int, max_lines_per_band: int = MAX_ROWS_PER_BAND) -> List[Tuple[int, int]]:
    n_lines = len(bands)
    if n_lines <= max_lines_per_band or total_height <= 0:
        return [(0, total_height)]
    parts = math.ceil(n_lines / max_lines_per_band)
    per = math.ceil(n_lines / parts)
    cuts = [0]
    for k in range(1, parts):
        nominal = min(k * per, n_lines - 1)
        lo, hi = max(1, nominal - CUT_SNAP_LINES), min(n_lines - 1, nominal + CUT_SNAP_LINES)
        idx = max(range(lo, hi + 1), key=lambda i: bands[i][0] - bands[i - 1][1])
        cuts.append((bands[idx - 1][1] + bands[idx][0]) // 2)
    cuts.append(total_height)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]


@dataclass
class Assessment:
    status: str = COMPLETE
    reasons: List[str] = field(default_factory=list)
    rows: int = 0
    visible_lines: Optional[int] = None
    line_ratio: Optional[float] = None
    missing_share: Optional[float] = None

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "completeness": self.status,
            "completeness_reasons": list(self.reasons),
            "rows": self.rows,
            "visible_lines": self.visible_lines,
            "line_ratio": None if self.line_ratio is None else round(self.line_ratio, 3),
            "missing_reference_share": None if self.missing_share is None else round(self.missing_share, 3),
        }


def assess_table(
    content: str,
    visible_lines: Optional[int] = None,
    reference_numbers: Optional[set] = None,
    generation_capped: bool = False,
) -> Assessment:
    result = Assessment()
    text = content or ""
    rows = table_rows(text)
    result.rows = len(rows)
    hard: List[str] = []
    weak: List[str] = []

    if re.search(r"<table\b", text, re.I) and len(re.findall(r"<table\b", text, re.I)) != len(re.findall(r"</table>", text, re.I)):
        hard.append("unclosed_table")

    if generation_capped:
        weak.append("generation_cap_near")

    if visible_lines is not None:
        result.visible_lines = visible_lines
        if visible_lines >= MIN_LINES_FOR_RATIO and rows:
            ratio = len(rows) / visible_lines
            result.line_ratio = ratio
            if ratio < SUSPECT_RATIO:
                weak.append("rows_vs_visible_lines")

    if reference_numbers:
        have = numeric_tokens(text)
        missing = len(reference_numbers - have) / len(reference_numbers)
        result.missing_share = missing
        if missing >= MISSING_NUMBERS_TRUNCATED:
            hard.append("missing_reference_numbers")
        elif missing >= MISSING_NUMBERS_SUSPECT:
            weak.append("missing_reference_numbers")

    if len(rows) >= 4:
        run = 1
        for prev, cur in zip(rows, rows[1:]):
            run = run + 1 if prev == cur and any(prev) else 1
            if run >= 3:
                hard.append("repetition_loop")
                break
        widths = sorted(len(r) for r in rows)
        modal = widths[len(widths) // 2]
        last = rows[-1]
        if modal and len(last) == modal and sum(1 for c in last if not c) >= max(3, modal // 2):
            body_empty = sum(1 for r in rows[1:-1] for c in r if not c) / max(1, sum(len(r) for r in rows[1:-1]))
            if body_empty < 0.25:
                weak.append("trailing_empty_row")

    result.reasons = hard + weak
    result.status = TRUNCATED if hard else (SUSPECT if weak else COMPLETE)
    return result


def stitch_bands(band_contents: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    reasons: List[str] = []
    all_rows: List[List[str]] = []
    widths = set()
    for content in band_contents:
        rows = table_rows(content)
        if not rows:
            reasons.append("empty_band")
            continue
        widths.add(max(len(r) for r in rows))
        all_rows.extend(rows)
    if reasons or not all_rows:
        return None, reasons or ["no_rows"]
    if max(widths) - min(widths) > BAND_WIDTH_TOLERANCE:
        return None, ["band_column_mismatch"]
    width = max(widths)
    cells = []
    for row in all_rows:
        padded = row + [""] * (width - len(row))
        cells.append("<tr>" + "".join(f"<td>{c}</td>" for c in padded) + "</tr>")
    return "<table>" + "".join(cells) + "</table>", []
