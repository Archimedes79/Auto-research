"""One-off: prefix 'skip: ' to papers.csv extract fields that start with neither deep nor skip (catalogue check rule).

Run from the problem folder: wsl python3 temp/loop19/fix_extract_prefix.py
"""
import csv
import os

path = os.path.join("literature", "papers.csv")
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)
fixed = []
for r in rows:
    ext = (r.get("extract") or "").strip()
    if ext and not ext.startswith(("deep", "skip")):
        r["extract"] = "skip: " + ext
        fixed.append(r["id"])
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
print("fixed:", fixed)
