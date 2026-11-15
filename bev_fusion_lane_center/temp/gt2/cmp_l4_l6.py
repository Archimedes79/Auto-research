"""Check of the l6 lane store (temp/gt_l6_store/lane/9ad1d4ce30) against the live l4 store on the VAL frames: the junction GT layer (E285) must come out the same (D83 item 2).
usage: cmp_l4_l6.py [every]"""
import gzip
import json
import os
import pickle
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "tools/viz"):
    sys.path.insert(0, os.path.join(R, sub))
import jgt as J
import qgraph_view as QV

store6 = os.path.join(os.path.dirname(R), "temp", "gt_l6_store", "lane", "9ad1d4ce30")
step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
toks = []
for k, (name, tk) in QV.val_scenes().items():
    toks += tk[::step]
S = dict(frames=0, missing6=0, junction4=0, junction6=0, ent_equal=0, ext_equal=0, pairs_equal=0, order_equal=0, zone_area_close=0, vis_equal=0, driven_equal=0, pieces_equal=0, events_equal=0)
bad = []
for t in toks:
    f6 = os.path.join(store6, t + ".pkl.gz")
    if not os.path.exists(f6):
        S["missing6"] += 1
        continue
    g4 = QV.load_gt(t)
    g6 = pickle.load(gzip.open(f6, "rb"))
    S["frames"] += 1
    S["pieces_equal"] += int(len(g4["pieces"]) == len(g6["pieces"]))
    S["events_equal"] += int(len(g4["events"]) == len(g6["events"]))
    j4, j6 = J.junction_gt(g4), J.junction_gt(g6)
    S["junction4"] += int(j4 is not None); S["junction6"] += int(j6 is not None)
    if j4 is None or j6 is None:
        if (j4 is None) != (j6 is None):
            bad.append((t, "junction presence"))
        continue
    S["ent_equal"] += int(j4["ent"].shape == j6["ent"].shape and np.allclose(j4["ent"], j6["ent"], atol=0.05))
    S["ext_equal"] += int(j4["ext"].shape == j6["ext"].shape and np.allclose(j4["ext"], j6["ext"], atol=0.05))
    pe = j4["pairs"].shape == j6["pairs"].shape and np.array_equal(j4["pairs"][:, :2], j6["pairs"][:, :2])
    S["pairs_equal"] += int(pe)
    S["driven_equal"] += int(pe and np.array_equal(j4["pairs"][:, 9], j6["pairs"][:, 9]))
    S["order_equal"] += int(j4["order"] == j6["order"])
    S["vis_equal"] += int(np.array_equal(j4["ent_vis"], j6["ent_vis"]) and np.array_equal(j4["ext_vis"], j6["ext_vis"]))
    a4, a6 = sum(z["area"] for z in j4["zones"]), sum(z["area"] for z in j6["zones"])
    S["zone_area_close"] += int(abs(a4 - a6) <= 0.02 * max(a4, 1.0))
    if not pe:
        bad.append((t, "pairs differ"))
print(json.dumps(S, indent=1))
print("first differences:", bad[:8])
json.dump(dict(S, differences=bad[:40]), open("cmp_l4_l6.json", "w"), indent=1)
