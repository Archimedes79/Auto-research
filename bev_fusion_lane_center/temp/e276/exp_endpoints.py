"""E276 pre-experiment P6 (video note 4: "do we find endpoints at the right position?"), CPU: for every GT event of the 873 VAL frames (l4 store: type, position, tier, demanded flag) look at the delivered
nodes of B (E275 dump, e >= 0.5, the flag probabilities of the node): is there a delivered node with THAT flag within 3 m (the scorer's event match), within 6 m, is there a delivered node of any kind within
3 m (found but without the flag), and how far is the nearest flagged node (median error of the matched ones, m). Split by event type and by tier of the event (the demanded ones: seen by a camera).
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_endpoints.py <run> <l4 store folder>
"""
import gzip
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qtargets as QT  # noqa: E402

run, store = sys.argv[1], sys.argv[2]
frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e275", f"pred_{run}.pkl"), "rb"))
assert [d["row"] for d in dump] == [f.row for f in frames]
FI = QT.FI
KIND = {"stop": "stop", "exit": "exit", "split": "split", "merge": "merge", "start": "start", "end": "end"}
acc = {}


def add(key, vals):
    a = acc.setdefault(key, [0, 0, 0, 0, 0, []])
    for i, v in enumerate(vals[:5]):
        a[i] += int(v)
    if vals[5] is not None:
        a[5].append(vals[5])


for f, d in zip(frames, dump):
    g = pickle.load(gzip.open(os.path.join(store, f.token + ".pkl.gz"), "rb"))
    keep = d["e"] >= 0.5
    P, F = d["pos"][keep], d["flags"][keep]
    tree = cKDTree(P) if len(P) else None
    for e in g["events"]:
        t = e["type"]
        if t not in KIND or not e.get("vis", True):                                         # the demanded events only (seen by a camera or in the blind zone)
            continue
        pos = np.asarray(e["pos"], float)[None]
        tier = int(e.get("tier", 3))
        flagged = P[F[:, FI[KIND[t]]] >= 0.5] if len(P) else P
        d_flag = cKDTree(flagged).query(pos)[0][0] if len(flagged) else np.inf
        d_any = tree.query(pos)[0][0] if tree is not None else np.inf
        row = (d_flag <= 3.0, d_flag <= 6.0, d_any <= 3.0, 1, d_any <= 6.0, d_flag if d_flag <= 6.0 else None)
        for key in ((t, "all tiers"), (t, f"tier {tier}"), ("all events", f"tier {tier}")):
            add(key, row)
out = {f"{k[0]} / {k[1]}": dict(n=v[3], flag_within_3m=v[0] / v[3], flag_within_6m=v[1] / v[3], any_node_within_3m=v[2] / v[3], any_node_within_6m=v[4] / v[3],
                                  median_error_m_of_flag_within_6m=float(np.median(v[5])) if v[5] else None) for k, v in acc.items() if v[3] > 0}
json.dump(dict(run=run, result=out), open(os.path.join(ROOT, "temp", "e276", f"endpoints_{run}.json"), "w"), indent=1)
print(f"{run}: GT events (demanded, VAL 873 frames) against the delivered nodes of B (e >= 0.5)")
print("| event / tier | n | flag within 3 m | flag within 6 m | any node within 3 m | any node within 6 m | median error of a flag within 6 m (m) |")
print("|---|---|---|---|---|---|---|")
for k, v in out.items():
    if k.endswith("all tiers") or k.startswith("all events"):
        print(f"| {k} | {v['n']} | {v['flag_within_3m']:.3f} | {v['flag_within_6m']:.3f} | {v['any_node_within_3m']:.3f} | {v['any_node_within_6m']:.3f} | "
              f"{'-' if v['median_error_m_of_flag_within_6m'] is None else format(v['median_error_m_of_flag_within_6m'], '.2f')} |")
print("\nby tier of the event (flag within 3 m, any node within 3 m):")
for k, v in out.items():
    if "tier" in k and not k.startswith("all events") and k.split(" / ")[0] in ("start", "end", "stop", "split"):
        print(f"  {k}: n {v['n']}, flag {v['flag_within_3m']:.3f}, any node {v['any_node_within_3m']:.3f}")
