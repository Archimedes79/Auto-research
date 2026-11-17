"""E378: the pinned cell rule of the card (E375's rule per head, lanes-found reference = the head's own TRAIN table at the D115 cell) -> prints 'S,L,K,on|off' (scratch).
usage: python3 temp/e378/choose378.py <run>"""
import json
import os
import sys

D115 = (0.3, 0.2, 0.6, False)
key = lambda r: (r["link_thr"], r.get("link_thr_long", r["link_thr"]), r["keep_thr"], r["none_gate"])  # noqa: E731
rows = {key(r): r for r in json.load(open(os.path.join("runs", sys.argv[1], "token_sweep_perclass_train_every3.json")))["rows"]}
assert len(rows) == 36, len(rows)
ok = [c for c, r in rows.items() if r["dup_frac"] <= 0.05 and r["lanes_found"] >= rows[D115]["lanes_found"] - 0.010]
if ok:
    T = max(rows[c]["topo_F1"] for c in ok)
    win = sorted((c for c in ok if rows[c]["topo_F1"] >= T - 0.005), key=lambda c: (-rows[c]["junction_F1"], abs(c[0] - 0.3), abs(c[1] - 0.2), abs(c[2] - 0.6), -c[0], -c[1], -c[2]))
    cell = win[0]
else:
    cell = D115
print(f"{cell[0]:g},{cell[1]:g},{cell[2]:g},{'on' if cell[3] else 'off'}")
