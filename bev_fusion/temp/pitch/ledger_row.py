import csv, sys
rid = sys.argv[1]
for r in csv.DictReader(open("records/LEDGER.csv", newline="", encoding="utf-8")):
    if r.get("id") == rid:
        for k, v in r.items():
            if v.strip():
                print(f"[{k}] {v}")
