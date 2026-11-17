import csv, subprocess, io
head = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/experiment_execution/LEDGER.csv"], capture_output=True, text=True, encoding="utf-8").stdout
a = {r["id"]: r for r in csv.DictReader(io.StringIO(head))}
b = {r["id"]: r for r in csv.DictReader(open("experiment_execution/LEDGER.csv", encoding="utf-8"))}
print("rows head", len(a), "now", len(b))
for k in b:
    if k not in a: print("NEW", k, b[k]["status"])
    elif a[k] != b[k]: print("CHANGED", k, [c for c in b[k] if a[k][c] != b[k][c]])
