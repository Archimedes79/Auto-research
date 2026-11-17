"""E289 diagnostic (scratch): node jitter of B under a shift of the input features.
usage: shift_probe.py <seed> [max frames]    -> runs/e289/shift_probe_s<seed>.json

The cached 39-channel features (0.5 m raster, rows = x, columns = y) are rolled by (dx, dy) cells; B's pyramid + node head run on both; the delivered nodes (the hard close-node step 1.5 m at e >= 0.5)
of the shifted run are shifted back; a node is stable when a node of the other set lies within 1.0 m. jitter_rate = 1 - the share of stable nodes (both directions). Nodes within 6 m of the window border
are left out (the roll wraps). The shift (0, 0) must give a jitter of exactly 0 (kill check)."""
import json
import os
import sys
import time

import numpy as np
import torch

R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
import lane_gt_data as LD
import qgraph_readout as QR
import qpredict as QP

seed = int(sys.argv[1])
maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = QP.load_model(os.path.join(R, "runs", f"e273_p_s{seed}"), dev)
m.eval()
ds, dl = LD.loader("val_clean", bs=8, workers=2, shuffle=False, map_mode="exact", pixels=False)
SHIFTS = [(0, 0), (1, 0), (0, 1), (2, 2), (4, 0)]                                          # cells of 0.5 m: no shift, 0.5 m x, 0.5 m y, 1 m diagonal, 2 m x (a whole node cell)
CUT = 44.0


def delivered(x):
    with torch.no_grad():
        P = m.pyr(x)
        no = m.node_stage(P)
    e = no["e"].float().sigmoid().flatten(1).cpu().numpy()
    pos = no["pos"].float().flatten(2).transpose(1, 2).cpu().numpy()
    out = []
    for i in range(len(e)):
        keep = QR.suppress_close_nodes(pos[i], e[i], 1.5, 0.5)
        p = pos[i][keep]
        out.append(p[(np.abs(p) < CUT).all(1)])
    return out


acc = {s: dict(n=0, unstable=0, disp=[]) for s in SHIFTS[1:]}
zero = 0
n_frames = 0
t0 = time.time()
for b in dl:
    x = b["x"].to(dev).float()
    base = delivered(x)
    for s in SHIFTS:
        xs = torch.roll(x, shifts=(s[0], s[1]), dims=(2, 3))
        other = delivered(xs)
        sh = np.array([0.5 * s[0], 0.5 * s[1]])
        for i in range(len(base)):
            A, Bn = base[i], other[i] - sh
            Bn = Bn[(np.abs(Bn) < CUT).all(1)]
            if not len(A) or not len(Bn):
                continue
            d = np.linalg.norm(A[:, None, :] - Bn[None, :, :], axis=-1)
            if s == (0, 0):
                zero += int((d.min(1) > 1e-4).sum())
                continue
            a_ok, b_ok = d.min(1) <= 1.0, d.min(0) <= 1.0
            acc[s]["n"] += len(A) + len(Bn)
            acc[s]["unstable"] += int((~a_ok).sum() + (~b_ok).sum())
            acc[s]["disp"] += d.min(1)[a_ok].tolist()
    n_frames += len(base)
    if maxn and n_frames >= maxn:
        break
res = dict(seed=seed, frames=n_frames, seconds=round(time.time() - t0), zero_shift_unmatched_nodes=zero)
for s, a in acc.items():
    res[f"shift_{s[0] * 0.5:g}x_{s[1] * 0.5:g}y"] = dict(jitter_rate=a["unstable"] / max(a["n"], 1), median_displacement_m=float(np.median(a["disp"])) if a["disp"] else None,
                                                         p90_displacement_m=float(np.percentile(a["disp"], 90)) if a["disp"] else None, nodes=a["n"])
res["jitter_rate_mean"] = float(np.mean([v["jitter_rate"] for k, v in res.items() if k.startswith("shift_")]))
print(json.dumps(res, indent=1))
os.makedirs(os.path.join(R, "runs", "e289"), exist_ok=True)
json.dump(res, open(os.path.join(R, "runs", "e289", f"shift_probe_s{seed}.json"), "w"), indent=1)
