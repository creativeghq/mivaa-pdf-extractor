#!/usr/bin/env python3
"""Lint gate for `app/` — the missing half of this repo's CI.

Why this exists
---------------
Until now MIVAA had a unit-test gate (deploy.yml) and no linter. Tests are the wrong
instrument for the failure mode this codebase actually produces: **code on a path
nobody calls**. A test cannot fail on a branch it never enters, so these all shipped
and survived, some for months:

  * `vecs_service.search_similar(...)` — name never bound, method never existed.
    Raised NameError on every call into a swallowed `except`, so entity search
    returned `[]` forever while logging a tidy warning.
  * `JobTracker(job_id)` — undefined since 2025-11-22, on the first line of six
    pipeline stage endpoints. Every one answered 500 having done nothing.
  * `re`, `datetime`, `asyncio` — used in four modules that never imported them.
    One sat in the handler that marks a job FAILED, so failures went unrecorded.

`ruff --select F` finds every one of them in under a second.

The gate
--------
Two tiers, deliberately:

  ZERO_TOLERANCE  Rules currently at zero. Any new occurrence fails the build.
                  F821 (undefined name) is the entire bug class above; it was 41
                  across app/ and is now 0. It stays 0.

  RATCHET         Everything else, held against .github/ruff-baseline.json. A count
                  may fall, never rise. These are hygiene (unused imports, unused
                  locals, placeholder-free f-strings) — real debt, but gating them
                  at zero today would be unlandable, and an unlandable gate gets
                  disabled within a week.

Fix findings; do not edit the baseline upward. Same contract as
.github/edge-typecheck-baseline.json on the platform side.
"""

from __future__ import annotations

import collections
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / ".github" / "ruff-baseline.json"
TARGET = "app/"

# At zero today. Keep them there.
ZERO_TOLERANCE = {"F821", "invalid-syntax"}


def run_ruff() -> collections.Counter:
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F",
         "--output-format", "json", TARGET],
        capture_output=True, text=True, cwd=ROOT,
        # Explicit: `text=True` alone decodes with the locale codec, which is cp1252
        # on Windows and blows up on ruff's UTF-8 output (source snippets carry the
        # emoji this codebase logs with). CI is Linux/UTF-8, so this would have been
        # a gate that only failed on the machine of whoever ran it by hand.
        encoding="utf-8", errors="replace",
    )
    # ruff exits 1 when it finds something — that is not a failure of this script.
    # An exit >1, or no parseable stdout, means ruff itself broke; treat that as a
    # hard error rather than "no findings", which would silently pass the gate.
    if proc.returncode > 1 or not proc.stdout.strip():
        print("ruff did not run:", proc.stderr.strip() or f"exit {proc.returncode}")
        raise SystemExit(2)
    try:
        findings = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"could not parse ruff output: {e}")
        raise SystemExit(2)
    # Syntax errors carry a null code — they mean ruff could not read the file at
    # all, which is worse than any finding: the file is unchecked and looks clean.
    return collections.Counter(f.get("code") or "invalid-syntax" for f in findings)


def main() -> int:
    counts = run_ruff()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["counts"]
    failures: list[str] = []

    for rule in sorted(ZERO_TOLERANCE):
        found = counts.get(rule, 0)
        if found:
            failures.append(
                f"  {rule}: {found} (must be 0) - an undefined name is code that "
                f"cannot run; fix it, do not baseline it"
            )

    for rule, allowed in sorted(baseline.items()):
        if rule in ZERO_TOLERANCE:
            continue
        found = counts.get(rule, 0)
        if found > allowed:
            failures.append(f"  {rule}: {found} > {allowed} baseline (+{found - allowed})")

    new_rules = set(counts) - set(baseline) - ZERO_TOLERANCE
    for rule in sorted(new_rules):
        failures.append(f"  {rule}: {counts[rule]} (new rule, not in baseline)")

    if failures:
        print("Ruff gate FAILED:\n" + "\n".join(failures))
        print("\nRun: python -m ruff check --select F app/")
        return 1

    improved = {
        r: (allowed, counts.get(r, 0))
        for r, allowed in baseline.items()
        if counts.get(r, 0) < allowed
    }
    total = sum(counts.values())
    print(f"Ruff gate OK - {total} finding(s), none above baseline, "
          f"{', '.join(sorted(ZERO_TOLERANCE))} at zero.")
    if improved:
        print("Improved (ratchet the baseline down in the same PR):")
        for rule, (was, now) in sorted(improved.items()):
            print(f"  {rule}: {was} -> {now}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
