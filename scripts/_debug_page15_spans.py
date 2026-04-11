"""One-shot debug: dump every text span on VALENOVA spec page 15.
Run on server via:
  cd /var/www/mivaa-pdf-extractor && .venv/bin/python3 scripts/_debug_page15_spans.py
"""

import fitz

PDF = "/tmp/pdf_processor_39e6c3af-86e8-41a9-97e6-78aebb1a3ca7/39e6c3af-86e8-41a9-97e6-78aebb1a3ca7.pdf"

doc = fitz.open(PDF)
page = doc[15]
td = page.get_text("dict")

spans = []
for block in td.get("blocks", []):
    if block.get("type", 0) != 0:
        continue
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = (span.get("text") or "").strip()
            if not text:
                continue
            bbox = span.get("bbox") or [0, 0, 0, 0]
            spans.append({
                "text": text,
                "x0": round(bbox[0], 1),
                "y0": round(bbox[1], 1),
                "x1": round(bbox[2], 1),
                "y1": round(bbox[3], 1),
                "cy": round((bbox[1] + bbox[3]) / 2, 1),
            })

spans.sort(key=lambda s: (s["y0"], s["x0"]))
print(f"Total spans on page 15: {len(spans)}")
print()
for s in spans:
    print(f"  y={s['y0']:6.1f} x={s['x0']:6.1f}  cy={s['cy']:6.1f}  {s['text'][:70]!r}")

# Also group by rough y-bucket to see row structure
print()
print("=== Spans grouped by cy (rounded to 3px buckets) ===")
from collections import defaultdict
buckets = defaultdict(list)
for s in spans:
    bucket = int(s["cy"] // 3) * 3
    buckets[bucket].append(s)
for bucket_y in sorted(buckets.keys()):
    row_spans = buckets[bucket_y]
    row_spans.sort(key=lambda s: s["x0"])
    texts = " | ".join(s["text"] for s in row_spans)
    print(f"  cy≈{bucket_y:>6}  ({len(row_spans):>2} spans)  {texts[:150]}")

doc.close()
