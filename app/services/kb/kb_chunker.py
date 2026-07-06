"""
KB document chunker (2026-07-06).

Splits a kb_docs markdown article into section-level chunks for retrieval.
Fixes the two problems of one-embedding-per-doc + truncation on large docs
(authored, split_v2 reference sets, catalog extractions — source-agnostic).

Guarantees (all asserted, so the backfill fails loudly rather than losing content):

  1. COVERAGE — chunks cover the whole document with no gap: chunk[0].char_start == 0,
     chunk[-1].char_end == len(content), and chunk[k+1].char_start <= chunk[k].char_end
     for every adjacent pair. => reassembling (dropping overlap) reproduces the source.
  2. BOUNDARY-AWARE — never split mid-word / mid-sentence. Splits land on the coarsest
     available boundary: heading -> paragraph -> sentence -> word -> (hard char, last resort).
  3. ATOMIC STRUCTURE — markdown tables and ``` code fences are never split internally.
  4. OVERLAP — adjacent chunks share ~overlap chars of whole trailing sentences (boundary
     clean), so context straddling a boundary is retrievable from either side.
  5. HEADING CONTEXT — each chunk carries its section heading; the embedding text is
     "{title} > {heading}\n\n{content}" (built by the service), while the raw chunk text
     is stored/returned verbatim.

Pure, deterministic (no randomness / IO) so re-runs and backfills are stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Tunable defaults (chars). ~1300 chars ≈ 325 tokens: precise retrieval, whole for
# small docs. max is a hard ceiling under Voyage's per-input limit.
DEFAULT_TARGET = 1300
DEFAULT_OVERLAP = 150
DEFAULT_MIN = 200
DEFAULT_MAX = 6000

# Sentence boundary: end punctuation (Latin + Greek ; ·) followed by whitespace, OR a
# blank line, OR end-of-span. Non-greedy so each match consumes exactly one sentence.
_SENTENCE_RE = re.compile(r".*?(?:[.!?;·]\s+|\n\s*\n|\Z)", re.S)


@dataclass
class Chunk:
    chunk_index: int
    heading: str
    content: str
    char_start: int
    char_end: int


# ── atom building (a strict partition of [0, N) into indivisible spans) ──────────

def _top_atoms(content: str) -> List[Tuple[int, int, str]]:
    """Partition content into (start, end, kind) atoms at structural boundaries.
    kind ∈ {heading, table, code, text}. Tables/code/headings are whole; 'text' spans
    are further split by _split_text later. Strictly covers [0, len(content))."""
    lines = content.splitlines(keepends=True)
    starts, p = [], 0
    for ln in lines:
        starts.append(p)
        p += len(ln)
    n = len(content)

    atoms: List[Tuple[int, int, str]] = []
    cursor = 0          # everything before cursor is already emitted
    i = 0

    def emit_text_gap(upto: int):
        nonlocal cursor
        if upto > cursor:
            atoms.append((cursor, upto, "text"))
        cursor = upto

    while i < len(lines):
        s = starts[i]
        line = lines[i]
        e = s + len(line)
        stripped = line.strip()

        if stripped.startswith("```"):  # fenced code — consume to the closing fence
            emit_text_gap(s)
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            end = (starts[j] + len(lines[j])) if j < len(lines) else n  # tolerate unclosed
            atoms.append((s, end, "code"))
            cursor = end
            i = (j + 1) if j < len(lines) else len(lines)
            continue

        if stripped.startswith("#"):  # heading line
            emit_text_gap(s)
            atoms.append((s, e, "heading"))
            cursor = e
            i += 1
            continue

        if stripped.startswith("|"):  # markdown table — consume consecutive | rows
            emit_text_gap(s)
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            end = starts[j - 1] + len(lines[j - 1])
            atoms.append((s, end, "table"))
            cursor = end
            i = j
            continue

        i += 1  # ordinary text line — folded into the next text gap

    emit_text_gap(n)
    return atoms


def _split_text(content: str, s: int, e: int) -> List[Tuple[int, int, str]]:
    """Split a 'text' span [s,e) into sentence atoms, preserving offsets + full coverage."""
    text = content[s:e]
    out: List[Tuple[int, int, str]] = []
    pos = 0
    for m in _SENTENCE_RE.finditer(text):
        seg = m.group()
        if not seg:
            break
        out.append((s + pos, s + pos + len(seg), "text"))
        pos += len(seg)
    if pos < len(text):  # tail (defensive; \Z usually catches it)
        out.append((s + pos, e, "text"))
    return out or [(s, e, "text")]


def _hard_split(s: int, e: int, kind: str, max_size: int) -> List[Tuple[int, int, str]]:
    """Last-resort fixed-width split for an atom that still exceeds max_size (a giant
    sentence, an unsplittable table/code block). Coverage preserved."""
    if e - s <= max_size:
        return [(s, e, kind)]
    return [(w, min(w + max_size, e), kind) for w in range(s, e, max_size)]


def _atoms(content: str, max_size: int) -> List[Tuple[int, int, str]]:
    result: List[Tuple[int, int, str]] = []
    for (s, e, kind) in _top_atoms(content):
        pieces = _split_text(content, s, e) if kind == "text" else [(s, e, kind)]
        for (ps, pe, pk) in pieces:
            result.extend(_hard_split(ps, pe, pk, max_size))
    return result


# ── packing atoms into overlapping chunks ───────────────────────────────────────

def chunk_document(
    content: str,
    title: Optional[str] = None,
    target: int = DEFAULT_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    min_size: int = DEFAULT_MIN,
    max_size: int = DEFAULT_MAX,
) -> List[Chunk]:
    content = content or ""
    n = len(content)
    if n == 0:
        return []
    overlap = min(overlap, max(0, target // 3))  # bound overlap so it can't stall

    atoms = _atoms(content, max_size)
    if not atoms:
        return []

    # Running section heading per atom (updates on heading atoms; else inherits).
    headings: List[str] = []
    cur = (title or "").strip()
    base = cur
    for (s, e, kind) in atoms:
        if kind == "heading":
            h = content[s:e].strip().lstrip("#").strip()
            cur = f"{base} › {h}" if base and h else (h or base)
        headings.append(cur)

    chunks: List[Chunk] = []
    m = len(atoms)
    i = 0
    while i < m:
        size = 0
        j = i
        while j < m:
            alen = atoms[j][1] - atoms[j][0]
            if size > 0 and size + alen > target:
                break
            size += alen
            j += 1
            if size >= target:
                break
        last = j - 1
        cs, ce = atoms[i][0], atoms[last][1]
        chunks.append(Chunk(len(chunks), headings[i], content[cs:ce], cs, ce))
        if j >= m:
            break
        # Next chunk starts `overlap` chars back (whole trailing atoms), but always
        # advances past i so we make progress and never gap.
        ov = 0
        k = last
        while k > i + 1 and ov + (atoms[k][1] - atoms[k][0]) <= overlap:
            ov += atoms[k][1] - atoms[k][0]
            k -= 1
        i = max(k, i + 1)

    # Merge a tiny trailing chunk into its predecessor (no orphan fragments).
    if len(chunks) >= 2 and len(chunks[-1].content) < min_size:
        prev, tail = chunks[-2], chunks.pop()
        chunks[-1] = Chunk(prev.chunk_index, prev.heading,
                           content[prev.char_start:tail.char_end], prev.char_start, tail.char_end)

    _assert_coverage(chunks, n)
    return chunks


def _assert_coverage(chunks: List[Chunk], n: int) -> None:
    """Coverage invariant — raises if any content would be lost."""
    if not chunks:
        raise ValueError("chunker produced 0 chunks for non-empty content")
    if chunks[0].char_start != 0:
        raise ValueError(f"coverage gap at start: first chunk begins at {chunks[0].char_start}")
    if chunks[-1].char_end != n:
        raise ValueError(f"coverage gap at end: last chunk ends at {chunks[-1].char_end}, len {n}")
    for a, b in zip(chunks, chunks[1:]):
        if b.char_start > a.char_end:  # gap between adjacent chunks
            raise ValueError(f"coverage gap: chunk {a.chunk_index} ends {a.char_end}, "
                             f"chunk {b.chunk_index} starts {b.char_start}")


def embedding_text(title: Optional[str], chunk: Chunk) -> str:
    """Text handed to Voyage — chunk content prefixed with its location for sharper
    matching. The raw chunk.content is what gets stored/returned to the agent."""
    loc = chunk.heading or (title or "")
    return f"{loc}\n\n{chunk.content}" if loc else chunk.content
