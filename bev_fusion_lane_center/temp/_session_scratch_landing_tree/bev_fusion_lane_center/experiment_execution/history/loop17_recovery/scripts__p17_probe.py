"""Loop 17 probe: which perturbed VAL series exist, how many frames per scene, the spacing in time."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402

D, OUT = L.D, L.OUT
P = dict(np.load(f"{D}/poses_val.npz"))
print("pose keys", list(P.keys()))
es_rows = None
for pn in sorted(os.listdir(f"{D}/pert")):
    f = f"{D}/pert/{pn}/val_rows.npy"
    if not os.path.exists(f):
        print(pn, "no val rows")
        continue
    rows = np.load(f)
    sc = P["scene"][rows]
    ts = P["ts"][rows]
    by = {}
    for r in rows:
        by.setdefault(int(P["scene"][r]), []).append(int(r))
    gaps = []
    for v in by.values():
        v = sorted(v, key=lambda i: P["ts"][i])
        gaps += list(np.diff(P["ts"][v]))
    d57 = os.path.exists(f"{OUT}/e57_{pn}_valset/prob_val.npy")
    print(f"{pn:16s} rows {len(rows):4d} scenes {len(by):3d} median gap {np.median(gaps) if gaps else 0:.2f} s  dense maps e57_valset {d57}")
