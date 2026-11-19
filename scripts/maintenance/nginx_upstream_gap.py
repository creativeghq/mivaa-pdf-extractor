"""nginx.org advisories that still apply to the installed nginx (#590).

Usage: nginx_upstream_gap.py <installed-version> < security_advisories.html
Prints a JSON list; an empty list means no published advisory affects the version.
Exits 2 when the page cannot be parsed, so a changed page is a failure, not "clean".
"""

import html
import json
import re
import sys


def parse_version(text):
    parts = text.strip().split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def affected(installed, not_vulnerable, vulnerable):
    for fixed in not_vulnerable:
        v = parse_version(fixed.rstrip("+"))
        if v and v[:2] == installed[:2] and installed >= v:
            return False
    for span in vulnerable:
        bounds = [parse_version(b) for b in span.split("-")]
        if any(b is None for b in bounds):
            continue
        lo, hi = bounds[0], bounds[-1]
        if lo <= installed <= hi:
            return True
    return False


def main():
    installed = parse_version(sys.argv[1] if len(sys.argv) > 1 else "")
    if installed is None:
        print(json.dumps({"error": "installed version is not x.y.z"}))
        return 2
    page = sys.stdin.read()
    items = re.findall(r"<li><p>(.*?)</p>", page, flags=re.S)
    if not items:
        print(json.dumps({"error": "no advisories found on the page"}))
        return 2
    gap = []
    for raw in items:
        fields = [html.unescape(re.sub(r"<[^>]+>", "", f)).strip() for f in raw.split("<br>")]
        nv = next((f.split(":", 1)[1] for f in fields if f.startswith("Not vulnerable:")), "")
        vu = next((f.split(":", 1)[1] for f in fields if f.startswith("Vulnerable:")), "")
        not_vulnerable = [x.strip() for x in nv.split(",") if x.strip()]
        vulnerable = [x.strip() for x in vu.split(",") if x.strip()]
        if not affected(installed, not_vulnerable, vulnerable):
            continue
        severity = next((f.split(":", 1)[1].strip() for f in fields if f.startswith("Severity:")), "")
        gap.append({
            "title": fields[0],
            "severity": severity,
            "cves": sorted(set(re.findall(r"CVE-\d{4}-\d+", raw))),
            "fixed_in": not_vulnerable,
        })
    print(json.dumps(gap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
