import csv, sys, io
rows = list(csv.DictReader(io.open("experiment_execution/LEDGER.csv", encoding="utf-8")))
pre = sys.argv[1:]
for r in rows:
    if any(r["id"].startswith(p) for p in pre):
        print("##", r["id"], "|", (r.get("result") or "")[:700], "|| DEC:", (r.get("decision") or "")[:500], "|| NOTES:", (r.get("notes") or "")[:300])
