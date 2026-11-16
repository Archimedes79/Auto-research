import csv, sys, io
rows = list(csv.DictReader(io.open("experiment_execution/LEDGER.csv", encoding="utf-8")))
for r in rows:
    if any(r["id"].startswith(p) for p in sys.argv[1:]):
        print("##", r["id"], "|", (r.get("hypothesis") or "")[:200], "| RES:", (r.get("result") or "")[:260], "|| DEC:", (r.get("decision") or "")[:380])
