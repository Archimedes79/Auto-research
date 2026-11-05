"""One-off: merge the five F4 scout tables (literature/papers_new_*.csv) into literature/papers.csv.

Rows whose id is already in papers.csv or in an earlier scout table are skipped (first one wins).
Run from the problem folder: wsl python3 temp/loop19/merge_f4_papers.py
"""
import csv
import glob
import os

LIT = "literature"
main_path = os.path.join(LIT, "papers.csv")
with open(main_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)
seen = {r["id"] for r in rows}

added, skipped = [], []
for path in sorted(glob.glob(os.path.join(LIT, "papers_new_*.csv"))):
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = (r.get("id") or "").strip()
            if not rid or rid in seen:
                skipped.append((os.path.basename(path), rid))
                continue
            seen.add(rid)
            rows.append({k: r.get(k, "") for k in fields})
            added.append((os.path.basename(path), rid))

with open(main_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\r\n")  # papers.csv uses CRLF
    w.writeheader()
    w.writerows(rows)

print(f"added {len(added)}, skipped {len(skipped)}")
for src, rid in skipped:
    print("skipped", src, rid)
