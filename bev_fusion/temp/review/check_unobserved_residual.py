"""Review check: after the losses.py fix, what weight do unobserved boxes keep at their centre cell, and does the head's
heatmap target contain them? Run: wsl ~/miniconda3/envs/gcar/bin/python temp/review/check_unobserved_residual.py"""
import math, os, sys
sys.path.insert(0, "training")
import torch
from gt_bev import NuScenesGT
from lift_pd import Grid
from losses import NUSC_GROUPS, build_layer_targets, head_box, object_head_terms
grid = Grid()
gt = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
samples = [s for s in gt.nusc.sample if s["prev"] and s["next"]][10::30][:12]
ws, n_pos = [], 0
for s in samples:
    g = gt.keyframe_gt(s["token"], grid, 4)
    T = build_layer_targets([g], None)
    w = T["w_obj"][0, 0]
    boxes = g["meta"]["boxes"]
    out = {"hm": torch.zeros(1, 3, grid.nx, grid.ny), "off": torch.zeros(1, 2, grid.nx, grid.ny), "size": torch.zeros(1, 2, grid.nx, grid.ny),
           "rot": torch.zeros(1, 2, grid.nx, grid.ny), "rad": torch.zeros(1, 1, grid.nx, grid.ny), "vel": torch.zeros(1, 2, grid.nx, grid.ny)}
    _, t = object_head_terms(out, [boxes], T["w_obj"], grid, NUSC_GROUPS)
    peaks = {int(i) for i, m in zip(t["ind"][0].tolist(), t["mask"][0].tolist()) if m > 0}
    for bx in boxes:
        hb = head_box(bx, NUSC_GROUPS)
        if hb is None or bx["observed"]:
            continue
        ix, iy = int(math.floor((hb["cx"] - grid.x_min) / grid.res)), int(math.floor((hb["cy"] - grid.y_min) / grid.res))
        if 0 <= ix < grid.nx and 0 <= iy < grid.ny:
            ws.append(float(w[ix, iy]))
            n_pos += (ix * grid.ny + iy) in peaks and any(abs(hb["cx"] - b["center"][0]) < 1e-6 for b in boxes if not b["observed"])
print(f"unobserved boxes with centre in the grid: {len(ws)}; w_obj at their centre cell: min {min(ws):.2f} median {sorted(ws)[len(ws)//2]:.2f} max {max(ws):.2f}; "
      f"heatmap positives among them: {n_pos}")
