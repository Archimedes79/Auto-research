"""One-off: distribution of the Doppler static weight (radar_encoder.static_weight) on the real nuScenes mini detections (cache of 0.5 s sweeps).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/static_weight_stats.py"""
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
import radar_encoder as re  # noqa: E402

fr = torch.load("temp/radar/cache/mini_multisweep.pt")
keys = ["x", "y", "r", "psi", "vr", "rcs", "dt"]
occ_cells, w_cells, det_w, det_vr = [], [], [], []
for i in range(0, len(fr), 8):
    batch = fr[i:i + 8]
    d = {k: torch.cat([f[k][f["dt"] <= 0.5] for f in batch]) for k in keys}
    d["b"] = torch.cat([torch.full(((f["dt"] <= 0.5).sum().item(),), j, dtype=torch.long) for j, f in enumerate(batch)])
    state = re.rasterize_delta_slots(d, len(batch))
    w = re.static_weight(state)[:, 0]
    occ = state[:, 0] > 0
    w_cells.append(w[occ])
    # per detection: the weight of its own cell
    idx, ix, iy, cell, rank, _, _ = re._slots(d, re.Grid(), re.K_SLOTS)
    det_w.append(w[d["b"][idx], ix, iy])
    det_vr.append(d["vr"][idx])
w_cells, det_w, det_vr = torch.cat(w_cells), torch.cat(det_w), torch.cat(det_vr)
print(f"occupied cells {len(w_cells)}, detections {len(det_w)}")
print("cells with weight  >= 0.9: {:.1f} %   0.5-0.9: {:.1f} %   < 0.5: {:.1f} %   < 0.1: {:.1f} %".format(
    *[100 * float(m.float().mean()) for m in (w_cells >= 0.9, (w_cells >= 0.5) & (w_cells < 0.9), w_cells < 0.5, w_cells < 0.1)]))
print("detections with |vr| > 1 m/s: {:.1f} %   > 3 m/s: {:.1f} %".format(*[100 * float((det_vr.abs() > t).float().mean()) for t in (1.0, 3.0)]))
print("mean weight of the cell of a detection with |vr| > 3: {:.3f};  with |vr| < 0.3: {:.3f}".format(
    float(det_w[det_vr.abs() > 3].mean()), float(det_w[det_vr.abs() < 0.3].mean())))
