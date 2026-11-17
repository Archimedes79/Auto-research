import csv, sys, io
rows = list(csv.DictReader(io.open("experiment_execution/LEDGER.csv", encoding="utf-8")))
pre = sys.argv[1:]
for r in rows:
    if any(r["id"] == p or r["id"].startswith(p + ".") for p in pre):
        d = r.get("decision") or ""
        if not d.strip():
            continue
        print("##", r["id"], r["started"][:16], "| RES:", (r.get("result") or "")[:420], "|| DEC:", d[:520])
