"""E377 (card v2.2): the pinned (b) rule for ONE recipe from the MEAN of its two seeds' per-class TRAIN tables (scratch).
eligible: mean dup_frac <= 0.05 and mean lanes found >= (mean lanes found at the D115 cell) - 0.010; the highest mean topo F1 T; window >= T - 0.005; the highest mean junction F1;
ties by |short - 0.3|, |long - 0.2|, |keep - 0.6|, then the higher thresholds; no eligible cell: the D115 cell.   usage: python3 temp/e377/choose377.py <run_s0> <run_s1>"""
import json
import os
import sys

D115 = (0.3, 0.2, 0.6, False)
key = lambda r: (r["link_thr"], r.get("link_thr_long", r["link_thr"]), r["keep_thr"], r["none_gate"])  # noqa: E731
tabs = [{key(r): r for r in json.load(open(os.path.join("runs", run, "token_sweep_perclass_train_every3.json")))["rows"]} for run in sys.argv[1:3]]
assert all(len(t) == 36 for t in tabs) and tabs[0].keys() == tabs[1].keys()
K = ("topo_F1", "junction_F1", "junction_P", "junction_R", "lanes_found", "dup_frac", "F1@1.0")
mean = {c: {k: sum(t[c][k] for t in tabs) / 2 for k in K} for c in tabs[0]}
lanes_ref = mean[D115]["lanes_found"]
ok = [c for c, m in mean.items() if m["dup_frac"] <= 0.05 and m["lanes_found"] >= lanes_ref - 0.010]
if not ok:
    print("no eligible cell -> (b) = D115", D115)
    sys.exit(0)
T = max(mean[c]["topo_F1"] for c in ok)
win = [c for c in ok if mean[c]["topo_F1"] >= T - 0.005]
win.sort(key=lambda c: (-mean[c]["junction_F1"], abs(c[0] - D115[0]), abs(c[1] - D115[1]), abs(c[2] - D115[2]), -c[0], -c[1], -c[2]))
print("runs", sys.argv[1:3], "| D115 mean lanes %.4f | eligible %d | T %.4f | window %d" % (lanes_ref, len(ok), T, len(win)))
for c in win:
    print("  ", c, {k: round(mean[c][k], 4) for k in K})
print("CHOSEN", win[0], {k: round(mean[win[0]][k], 4) for k in K})
print("D115  ", D115, {k: round(mean[D115][k], 4) for k in K})
