"""scratch (continuation of conn_start_and_complexity.py, Mirko's marks of 2026-11-08): the lines are on the GT everywhere (median shift 0.1 - 0.15 m, also at the junction entry and at the start of the connectors), so the 'blur' must be the
EMPTY space between the lines: cells that the target leaves empty (T < 0.1, i.e. farther than 2.15 m from every lane point) but that are within 6 m of a lane point. Mean p, the share with p > 0.3 and p >= 0.5, the share of the p mass
that sits there, inside the junction (< 8 m from the junction points) against outside, and by the number of the neighbouring lines (not computed here: the junction against outside is enough).
Reads temp/e284/target_vs_pred.npz. CPU.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/gap_fill.py"""
import json
import os

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
toks = [str(t) for t in z["tokens"]]
n_stat = len(toks) - (1 if len(toks) > 88 else 0)
P, T = z["p"].astype(np.float32)[:n_stat], z["t"].astype(np.float32)[:n_stat]
DJ, DG = z["dj"].astype(np.float32)[:n_stat], z["dg"].astype(np.float32)[:n_stat]
c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
inbox = ((np.abs(CX) <= 47.0) & (np.abs(CY) <= 47.0)).reshape(-1)
res = {}
for name, jm in (("inside the junction (< 8 m)", lambda dj: dj < 8.0), ("8-24 m from the junction", lambda dj: (dj >= 8.0) & (dj < 24.0)), ("outside (>= 24 m or none)", lambda dj: dj >= 24.0)):
    gap_cells = gap_p = gap_gt3 = gap_ge5 = 0.0
    p_mass_gap = p_mass_all = 0.0
    line_cells = 0.0
    for i in range(n_stat):
        m = jm(DJ[i]) & inbox
        gap = m & (T[i] < 0.1) & (DG[i] < 6.0)
        line = m & (T[i] >= 0.1)
        gap_cells += gap.sum()
        gap_p += P[i][gap].sum()
        gap_gt3 += (P[i][gap] > 0.3).sum()
        gap_ge5 += (P[i][gap] >= 0.5).sum()
        p_mass_gap += P[i][gap].sum()
        p_mass_all += P[i][m & ((T[i] >= 0.1) | gap)].sum()
        line_cells += line.sum()
    res[name] = dict(gap_cells=float(gap_cells), line_cells=float(line_cells), gap_share_of_cells=float(gap_cells / (gap_cells + line_cells)), mean_p_in_gap=float(gap_p / max(gap_cells, 1)), share_p_over_0_3=float(gap_gt3 / max(gap_cells, 1)),
                     share_p_ge_0_5=float(gap_ge5 / max(gap_cells, 1)), share_of_p_mass_in_gap=float(p_mass_gap / max(p_mass_all, 1)))
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "gap_fill.json"), "w"), indent=1)
for k, v in res.items():
    print(f"{k:30s} empty cells (T < 0.1, within 6 m of a line) {v['gap_cells']:7.0f} = {100 * v['gap_share_of_cells']:4.1f} % of the cells near lines | mean p there {v['mean_p_in_gap']:.3f}  p > 0.3 in {100 * v['share_p_over_0_3']:4.1f} %  p >= 0.5 in {100 * v['share_p_ge_0_5']:4.1f} %  | share of the p mass that sits in the empty cells {100 * v['share_of_p_mass_in_gap']:4.1f} %")
