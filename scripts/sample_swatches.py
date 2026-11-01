"""Print sampled swatch colours for PDF pages as JSON. No database access.

Usage: python scripts/sample_swatches.py --pdf catalog.pdf --pages 15,21,27   (--pages all for every page)
"""

import argparse
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_sampler():
    path = _ROOT / "app" / "services" / "products" / "swatch_sampler.py"
    spec = importlib.util.spec_from_file_location("swatch_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import fitz

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", required=True)
    args = ap.parse_args()
    sampler = _load_sampler()
    doc = fitz.open(args.pdf)
    pages = range(1, len(doc) + 1) if args.pages == "all" else [int(p) for p in args.pages.split(",")]
    result = {str(p): sampler.sample_page(args.pdf, p, doc=doc) for p in pages}
    print(json.dumps({k: v for k, v in result.items() if v}, indent=2))


if __name__ == "__main__":
    main()
