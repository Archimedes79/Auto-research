"""scratch (continuation of target_vs_pred.py / lateral_offset.py, Mirko's remark of 2026-11-08): WHERE does the error p - T of the E279 centre map sit? Every cell of the 88 stored frames (temp/e284/target_vs_pred.npz) goes into one region:
  edge      outside |x|, |y| < 48 (the target T is cut there: the training points end at 48 m)
  junction  within 8 m of a connector point that the lane GT marks as inside an intersection
  parallel  elsewhere, and a lane point of ANOTHER piece lies within 3.5 m of the cell centre AND more than 1.5 m from the nearest lane point (two lines side by side: pieces that only touch at their ends do not count)
  single    elsewhere, a lane point within 3.5 m of the cell, no second line
  empty     elsewhere, no lane point within 3.5 m
per region: cells with (T > 0.05 or p > 0.05), the sum of |p - T| (and its share of the total), the cells of the threshold read-out (p >= 0.5 against T >= 0.5): hit, missing, too many.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/error_by_region.py"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402

z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
toks = [str(t) for t in z["tokens"]]
P, T = z["p"].astype(np.float32), z["t"].astype(np.float32)
DJ = z["dj"].astype(np.float32)
by_tok = {f.token: f for f in QE.val_frames()}
c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
cells = np.stack([CX.ravel(), CY.ravel()], 1)
edge = (np.abs(cells) > 48.0).any(1)
NAMES = ("edge", "junction", "parallel", "single", "empty")
acc = {n: dict(cells=0.0, err=0.0, hit=0.0, miss=0.0, false=0.0, tcells=0.0) for n in NAMES}
n_frames = len(toks) - (1 if len(toks) > 88 else 0)
for i in range(n_frames):
    f = by_tok[toks[i]]
    g = f.lane()
    pcs = [np.asarray(p["pts"], float).reshape(-1, 2) for p in g.get("pieces", []) if len(p["pts"]) >= 2]
    allp = np.concatenate(pcs)
    allp = allp[(np.abs(allp) < 48.0).all(1)]
    owner_all = np.concatenate([np.full(len(q), k) for k, q in enumerate(pcs)])
    # owner after the box filter
    keepm = (np.abs(np.concatenate(pcs)) < 48.0).all(1)
    owner = owner_all[keepm]
    tree = cKDTree(allp)
    p, t, dj = P[i], T[i], DJ[i]
    rel = np.where((t > 0.05) | (p > 0.05))[0]
    cls = np.full(len(cells), 4, int)                                   # empty
    d1, a = tree.query(cells[rel])
    near = tree.query_ball_point(cells[rel], 3.5)
    for k, cidx in enumerate(rel):
        if edge[cidx]:
            cls[cidx] = 0
        elif dj[cidx] < 8.0:
            cls[cidx] = 1
        elif not near[k]:
            cls[cidx] = 4
        else:
            ja = np.array(near[k])
            other = owner[ja] != owner[a[k]]
            far = np.linalg.norm(allp[ja] - allp[a[k]], axis=1) > 1.5
            cls[cidx] = 2 if (other & far).any() else 3
    on_p, on_t = p >= 0.5, t >= 0.5
    for ci, name in enumerate(NAMES):
        m = (cls == ci)
        m_rel = m.copy()
        m_rel[~((t > 0.05) | (p > 0.05))] = False
        a_ = acc[name]
        a_["cells"] += float(m_rel.sum())
        a_["err"] += float(np.abs(p - t)[m_rel].sum())
        a_["hit"] += float((on_p & on_t & m).sum())
        a_["miss"] += float((~on_p & on_t & m).sum())
        a_["false"] += float((on_p & ~on_t & m).sum())
        a_["tcells"] += float((on_t & m).sum())
    f._lane = f._ref = None
    if (i + 1) % 20 == 0:
        print(i + 1, "frames", flush=True)
tot_err = sum(a["err"] for a in acc.values())
tot_cells = sum(a["cells"] for a in acc.values())
tot_false = sum(a["false"] for a in acc.values())
tot_miss = sum(a["miss"] for a in acc.values())
res = {n: dict(cells=a["cells"], share_cells=a["cells"] / tot_cells, share_error=a["err"] / tot_err, mean_abs_err=a["err"] / max(a["cells"], 1), hit=a["hit"], miss=a["miss"], false=a["false"], target_cells=a["tcells"],
               share_of_all_false=a["false"] / max(tot_false, 1), share_of_all_miss=a["miss"] / max(tot_miss, 1), false_per_target_cell=a["false"] / max(a["tcells"], 1), miss_per_target_cell=a["miss"] / max(a["tcells"], 1)) for n, a in acc.items()}
json.dump(dict(frames=n_frames, regions=res), open(os.path.join(ROOT, "temp", "e284", "error_by_region.json"), "w"), indent=1)
print("frames", n_frames)
for n, v in res.items():
    print(f"{n:9s} cells {100 * v['share_cells']:5.1f} %  of the error mass {100 * v['share_error']:5.1f} %  mean |p - T| {v['mean_abs_err']:.3f}  | read-out: target cells {v['target_cells']:7.0f}  missing {v['miss']:6.0f} ({100 * v['share_of_all_miss']:4.1f} % of all)  too many {v['false']:6.0f} ({100 * v['share_of_all_false']:4.1f} % of all)  per target cell: miss {v['miss_per_target_cell']:.2f}  false {v['false_per_target_cell']:.2f}")
