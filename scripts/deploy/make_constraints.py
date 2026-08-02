#!/usr/bin/env python3
"""
Build the pip constraints file used by the deploy's dependency reconcile step.

WHY THIS EXISTS
---------------
The deploy clones the known-good live venv and reconciles it to requirements.txt under a
constraints file, so that loose `>=` ranges do not silently resolve to a newer wheel that
segfaults on import. That intent is right and is preserved here.

What was wrong was the constraints file itself. It was the WHOLE of `pip freeze`, which pins
every installed package to its current exact version - including the packages requirements.txt
is actively trying to move. The result:

    requirements.txt   Pillow>=12.3.0,<13.0.0     (raised for 24 CVEs, audit #300)
    constraints        pillow==10.4.0             (whatever happens to be installed)
    pip                ERROR: ResolutionImpossible

and because the install line ended in `| tee -a`, the pipeline exit status was tee's 0. The step
went green, the blue-green swap promoted a venv that was byte-identical to the old one, and the
deploy reported success. Every security bump to a Python dependency had been discarded this way,
silently, since the constraint mechanism was introduced. Measured on production 2026-08-02:
pillow 10.4.0 (24 advisories) and python-dotenv 1.0.0 (1) both still installed, months after
requirements.txt was raised past them.

THE RULE
--------
Constrain only what requirements.txt does NOT name.

A package listed in requirements.txt is one a human has an opinion about - that opinion is the
whole point of the file, and it must win. Everything else is transitive, nobody chose its
version, and pinning it to the version already proven to work on this box is exactly the
protection the original comment describes.

Usage:  make_constraints.py <freeze-file> <requirements-file> > constraints.txt
"""
import re
import sys


def canonical(name: str) -> str:
    """PEP 503 normalisation - `Pillow`, `pillow` and `pil_low` must compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(path: str) -> set:
    names = set()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            # strip environment markers, then extras, then the version specifier
            line = line.split(";", 1)[0].strip()
            m = re.match(r"^([A-Za-z0-9._-]+)", line)
            if m:
                names.add(canonical(m.group(1)))
    return names


def main() -> int:
    freeze_path, req_path = sys.argv[1], sys.argv[2]
    declared = requirement_names(req_path)

    kept, released = [], []
    with open(freeze_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # `pip freeze` also emits `name @ file:///...` and VCS lines; keep those verbatim,
            # they are not version pins we could conflict with.
            m = re.match(r"^([A-Za-z0-9._-]+)==", line)
            if not m:
                kept.append(line)
                continue
            if canonical(m.group(1)) in declared:
                released.append(line)      # requirements.txt owns this one
            else:
                kept.append(line)

    for line in kept:
        print(line)

    # To stderr so it lands in the deploy log without polluting the constraints file. This is the
    # audit trail: if a package is NOT in this list and did not move, the reconcile is a no-op for
    # a reason, not by accident.
    print(
        f"[constraints] {len(kept)} transitive packages pinned; "
        f"{len(released)} left free because requirements.txt names them:",
        file=sys.stderr,
    )
    for line in released:
        print(f"[constraints]   free: {line}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
