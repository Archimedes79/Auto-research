"""scratch (answer to Mirko's questions of 2026-11-08): which classes do the nodes have near junctions? GT nodes and the delivered nodes of baseline B (full dumps, e >= 0.5), VAL 873, by distance to the connector points
that the lane GT marks as inside an intersection (the points that carry no node). Zones: 'in the junction' <= 2 m, 'mouth' 2 - 8 m, 'away' > 8 m (or no intersection in the frame).
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e283/junction_classes.py [--rows N]"""
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

N_ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
ZONES = (("in the junction (<= 2 m)", 0.0, 2.0), ("mouth (2 - 8 m)", 2.0, 8.0), ("away (> 8 m)", 8.0, 1e9))
FL = QT.FLAGS
frames = QE.val_frames()
if N_ROWS:
    frames = frames[:N_ROWS]
res = {}
gt_acc = {z[0]: np.zeros(len(FL) + 2) for z in ZONES}                 # per zone: n nodes, n with any flag, per flag
dl_acc = {s: {z[0]: np.zeros(len(FL) + 4) for z in ZONES} for s in (0, 1)}   # per zone: n, any flag, per flag, n false (no GT node within 2 m), n frames-with-intersection nodes
n_inter_frames = 0
dumps = {s: pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{s}.pkl"), "rb")) for s in (0, 1)}
for i, f in enumerate(frames):
    ref = f.ref()
    nt = ref.nt
    sp = nt["pts"][nt["supp"]]
    tree = cKDTree(sp) if len(sp) else None
    n_inter_frames += int(tree is not None)

    def zone_of(P):
        if tree is None or not len(P):
            return np.full(len(P), 2)
        d = tree.query(P)[0]
        return np.where(d <= 2.0, 0, np.where(d <= 8.0, 1, 2))

    zg = zone_of(nt["pos"])
    for zi, z in enumerate(ZONES):
        m = zg == zi
        a = gt_acc[z[0]]
        a[0] += m.sum()
        a[1] += (nt["flags"][m].any(1)).sum()
        a[2:2 + len(FL)] += nt["flags"][m].sum(0)
    Gt = cKDTree(nt["pos"]) if len(nt["pos"]) else None
    for s in (0, 1):
        d = dumps[s][i]
        assert int(d["row"]) == int(f.row)
        keep = d["e"] >= 0.5
        P, F = d["pos"][keep], d["flags"][keep] >= 0.5
        zp = zone_of(P)
        far = (Gt.query(P)[0] > 2.0) if Gt is not None and len(P) else np.ones(len(P), bool)
        for zi, z in enumerate(ZONES):
            m = zp == zi
            a = dl_acc[s][z[0]]
            a[0] += m.sum()
            a[1] += F[m].any(1).sum()
            a[2:2 + len(FL)] += F[m].sum(0)
            a[2 + len(FL)] += far[m].sum()
    if (i + 1) % 200 == 0:
        print(i + 1, "frames", flush=True)
    f._lane = f._ref = None
out = dict(frames=len(frames), frames_with_intersection=n_inter_frames, gt={k: dict(n=int(v[0]), any_event=int(v[1]), by_flag={FL[j]: int(v[2 + j]) for j in range(len(FL))}) for k, v in gt_acc.items()},
           delivered={str(s): {k: dict(n=int(v[0]), any_event=int(v[1]), by_flag={FL[j]: int(v[2 + j]) for j in range(len(FL))}, false_nodes=int(v[2 + len(FL)])) for k, v in dl_acc[s].items()} for s in (0, 1)})
json.dump(out, open(os.path.join(ROOT, "temp", "e283", "junction_classes.json"), "w"), indent=1)
print("frames", out["frames"], "with an intersection", out["frames_with_intersection"])
for z in ZONES:
    g = out["gt"][z[0]]
    print(f"GT   {z[0]:26s} nodes {g['n']:7d}  with an event flag {100 * g['any_event'] / max(g['n'], 1):5.1f} %  by flag {g['by_flag']}")
    for s in ("0", "1"):
        d = out["delivered"][s][z[0]]
        print(f"B s{s} {z[0]:26s} nodes {d['n']:7d}  flagged {100 * d['any_event'] / max(d['n'], 1):5.1f} %  false nodes {100 * d['false_nodes'] / max(d['n'], 1):5.1f} %  by flag {d['by_flag']}")
