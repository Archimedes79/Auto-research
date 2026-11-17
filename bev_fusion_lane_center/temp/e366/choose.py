"""E366: the cell chosen by the rule of the card (section 0), from the two TRAIN-sample sweep tables (scratch; the rule is written in the card before the sweep)"""
import json
import os
import sys

runs = sys.argv[1:] or ["e365_s1_s0", "e365_s1_s1"]
tabs = [json.load(open(os.path.join("runs", r, "token_sweep_train_every3.json"))) for r in runs]
key = lambda row: (row["link_thr"], row["keep_thr"], row["none_gate"])  # noqa: E731
cells = {}
for t in tabs:
    for row in t["rows"]:
        cells.setdefault(key(row), []).append(row)
assert all(len(v) == len(tabs) for v in cells.values()), "a cell is missing in a table"
rec = (0.3, 0.5, True)


def dist(c):                                          # ties: the cell nearest the record (T, then K, then the gate)
    return (abs(c[0] - rec[0]), abs(c[1] - rec[1]), c[2] != rec[2])


rows = []
for c, v in cells.items():
    tf = sum(r["topo_F1"] for r in v) / len(v)
    dup = sum(r["dup_frac"] for r in v) / len(v)
    rows.append((c, tf, dup, [round(r["topo_F1"], 4) for r in v], [round(r["topo_P"], 3) for r in v], [round(r["topo_R"], 3) for r in v], [round(r["lanes_found"], 4) for r in v], [round(r["F1@1.0"], 4) for r in v]))
rows.sort(key=lambda x: (-x[1], dist(x[0])))
print("frames per table", [t["frames"] for t in tabs], "| every", [t.get("every") for t in tabs])
print("cell (link_thr, keep_thr, none_gate) | mean topo F1 | mean dup | topo F1 per seed | P | R | lanes found | F1@1.0")
for r in rows:
    flag = "  <- record" if r[0] == rec else ""
    print(r[0], round(r[1], 4), round(r[2], 4), r[3], r[4], r[5], r[6], r[7], flag)
ok = [r for r in rows if r[2] <= 0.05]
best = ok[0]
top = [r for r in ok if abs(r[1] - best[1]) < 1e-12]
best = sorted(top, key=lambda x: dist(x[0]))[0]
print("CHOSEN", best[0], "mean topo F1", round(best[1], 4), "mean dup", round(best[2], 4))
