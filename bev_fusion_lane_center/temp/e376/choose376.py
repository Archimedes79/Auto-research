"""E376 (b) cell of ONE head by the E375 rule (temp/e375/choose_pc.py) on its own per-class TRAIN table (scratch). The reference cell of the guards and the tie-break is
printed for both readings of 'the same guards': the E375 code's cell 0.2 / 0.2 / 0.6 (the champion cell when E375 ran) and the D115 cell 0.3 / 0.2 / 0.6 (the champion now).
usage: python3 temp/e376/choose376.py <run>"""
import json
import os
import sys

run = sys.argv[1]
t = json.load(open(os.path.join("runs", run, "token_sweep_perclass_train_every3.json")))
key = lambda r: (r["link_thr"], r.get("link_thr_long", r["link_thr"]), r["keep_thr"], r["none_gate"])  # noqa: E731
rows = {key(r): r for r in t["rows"]}
assert len(rows) == 36, len(rows)
for ref in ((0.2, 0.2, 0.6, False), (0.3, 0.2, 0.6, False)):
    lanes_ref = rows[ref]["lanes_found"]
    ok = [r for r in rows.values() if r["dup_frac"] <= 0.05 and r["lanes_found"] >= lanes_ref - 0.01]
    if not ok:
        print("ref", ref, "no eligible cell -> (b) = the D115 cell")
        continue
    best = max(r["topo_F1"] for r in ok)
    win = [r for r in ok if r["topo_F1"] >= best - 0.005]

    def dist(c):
        return (abs(c[0] - ref[0]), abs(c[1] - ref[1]), abs(c[2] - ref[2]), -c[0], -c[1], -c[2])
    win.sort(key=lambda r: (-r["junction_F1"], dist(key(r))))
    w = win[0]
    print("ref", ref, "lanes_ref %.4f eligible %d best topo %.4f window %d CHOSEN %s" % (lanes_ref, len(ok), best, len(win), key(w)),
          {k: round(w[k], 4) for k in ("topo_F1", "junction_F1", "junction_P", "junction_R", "lanes_found", "dup_frac", "F1@1.0")})
print("D115 cell on TRAIN:", {k: round(rows[(0.3, 0.2, 0.6, False)][k], 4) for k in ("topo_F1", "junction_F1", "lanes_found", "dup_frac")})
