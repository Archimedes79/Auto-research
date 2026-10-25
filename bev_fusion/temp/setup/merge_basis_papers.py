"""One-off: merge temp/scout_tables/papers_new_basis.csv into literature/papers.csv (a row with an existing id replaces it).

Run from the bev_fusion folder: wsl python3 temp/setup/merge_basis_papers.py
"""
import csv

main = "literature/papers.csv"
with open(main, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)
with open("temp/scout_tables/papers_new_basis.csv", newline="", encoding="utf-8") as f:
    new = list(csv.DictReader(f))
bad = [r for r in new if set(r) != set(fields) or None in r]
if bad:
    raise SystemExit(f"column mismatch in {len(bad)} rows, first id {bad[0].get('id')}")
new_ids = {r["id"].strip() for r in new}
replaced = [r["id"] for r in rows if r["id"] in new_ids]
rows = [r for r in rows if r["id"] not in new_ids] + [{k: r[k] for k in fields} for r in new]
with open(main, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print(f"added {len(new) - len(replaced)}, replaced {replaced}, total {len(rows)}")
