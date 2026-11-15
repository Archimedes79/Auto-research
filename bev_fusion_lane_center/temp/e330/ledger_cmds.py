import csv, re
rows = list(csv.reader(open("experiment_execution/LEDGER.csv", encoding="utf-8")))
hdr = rows[0]
print(hdr)
ci = next(i for i, h in enumerate(hdr) if h.lower().startswith("command") or h.lower() == "cmd")
for r in rows[1:]:
    if re.match(r"E35[3457]\.", r[0]) and r[3] in ("ok", "running", "failed", "crashed", "cut", "planned", "started", "") or re.match(r"E35[7]\.", r[0]):
        print(r[0], "|", r[1][:16], r[2][:16], r[3], "|", r[ci][:700])
