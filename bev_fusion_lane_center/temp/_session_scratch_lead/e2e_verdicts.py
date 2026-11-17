import csv, re, sys
want = ["E273","E274","E275","E277","E292","E299","E302","E307","E311","E312","E316","E318","E319","E320","E322","E323","E324","E325","E331","E332","E334","E336","E337","E338","E341","E347","E348","E350","E353","E354","E355","E357"]
rows = list(csv.DictReader(open("experiment_execution/LEDGER.csv", encoding="utf-8")))
print(list(rows[0].keys()))
by = {}
for r in rows:
    m = re.match(r"(E\d+)", r["id"])
    if m:
        by.setdefault(m.group(1), []).append(r)
for e in want:
    rs = by.get(e, [])
    toks = []
    for r in rs:
        d = (r.get("decision") or "").strip()
        if d:
            toks.append(r["id"] + ": " + d[:150].replace("\n", " "))
    print("==", e, "rows", len(rs))
    for t in toks[-3:]:
        print("   ", t)
