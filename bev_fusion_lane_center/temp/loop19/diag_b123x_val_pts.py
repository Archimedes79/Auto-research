"""Diagnosis of E251.dist (val part: gt_pts_exact not identical to the stored file). CPU only.

    wsl bash tools/env/_py14.sh temp/loop19/diag_b123x_val_pts.py

Which val frames differ, by how much (point count, max coordinate difference), per scene and location; are the differences confined to few frames, to
frames with few points, or to a rounding of the polyline densification?
"""
import json
import os

import numpy as np

H = os.path.expanduser("~")
old = np.load(f"{H}/exp_crn_b123/feats/val_gt_pts_exact.npy", allow_pickle=True)
new = np.load(f"{H}/exp_gcar_b123x/feats/val_gt_pts_exact.npy", allow_pickle=True)
toks = json.load(open(f"{H}/exp_crn_b123/feats/val_tokens.json"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")
gt_toks = json.load(open(f"{H}/exp_gcar_b123/feats/val_tokens.json"))
print("rows", len(old), len(new), "crn tokens == gcar tokens:", toks == gt_toks)
bad = []
for i, (a, b) in enumerate(zip(old, new)):
    if a.shape != b.shape or not np.array_equal(a, b):
        d = np.abs(a - b).max() if a.shape == b.shape else float("nan")
        bad.append((i, a.shape[0], b.shape[0], d))
print(f"{len(bad)} of {len(old)} frames differ")
for i, na, nb, d in bad[:12]:
    print(f"  row {i} scene {sc['scene'][i]} loc {sc['loc'][i]}: points {na} -> {nb}, max |diff| {d}")
if bad:
    scenes = sorted({int(sc['scene'][i]) for i, *_ in bad})
    print("scenes with differences:", scenes, "| locations:", sorted({int(sc['loc'][i]) for i, *_ in bad}))
    same_shape = [x for x in bad if x[1] == x[2]]
    print(f"same point count: {len(same_shape)}; max |diff| over those: {max([x[3] for x in same_shape], default=0):.6f} m")
    print("point-count change summary:", np.percentile([x[2] - x[1] for x in bad if x[1] != x[2]], [0, 50, 100]) if any(x[1] != x[2] for x in bad) else "none")
