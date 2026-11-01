"""E237 step 0 (CPU): how much of the supervised region does the TF32 damage of the stored distance map touch?

    wsl bash tools/env/_py14.sh temp/loop19/e237_step0.py

Stored map (exp_crn_b123/feats) against the CPU-exact one (exp_gcar_b123x/feats), train and val part, in px (1 px = 0.5 m), fp16 maps. Per frame:
  supervised region = pixels with an exact distance < 8 px (4 m): the seed / direction / builder losses read the map there (near masks use 0.6-0.8 m, 2.5 m)
  damaged = |stored - exact| > 0.5 px (0.25 m) ; strongly damaged = > 2 px (1 m)
Reports the share of frames with any damage, the share of supervised pixels damaged (mean, p90), and the same inside the 1.2 m band around the lines where
F1@1 is decided (exact distance < 2.4 px). Writes runs/e19/e237_step0.json.
"""
import json
import os

import numpy as np

H = os.path.expanduser("~")
OLD, NEW = f"{H}/exp_crn_b123/feats", f"{H}/exp_gcar_b123x/feats"
out = {}
for part in ("train", "val"):
    a = np.load(f"{OLD}/{part}_gt_dist_exact.npy", mmap_mode="r")
    b = np.load(f"{NEW}/{part}_gt_dist_exact.npy", mmap_mode="r")
    n = len(a)
    sup, dmg, dmg2, near, dnear = [], [], [], [], []
    anyd = 0
    for s in range(0, n, 200):
        x, y = np.asarray(a[s:s + 200], np.float32), np.asarray(b[s:s + 200], np.float32)
        d = np.abs(x - y)
        m = y < 8.0
        m2 = y < 2.4
        for i in range(len(x)):
            ms, mn = m[i].sum(), m2[i].sum()
            sup.append(ms)
            dmg.append((d[i] > 0.5)[m[i]].sum() / max(ms, 1))
            dmg2.append((d[i] > 2.0)[m[i]].sum() / max(ms, 1))
            near.append(mn)
            dnear.append((d[i] > 0.5)[m2[i]].sum() / max(mn, 1))
            anyd += bool((d[i] > 0.5).any())
    f = lambda v: dict(mean=float(np.mean(v)), p90=float(np.percentile(v, 90)), max=float(np.max(v)))  # noqa: E731
    out[part] = dict(frames=n, frames_with_damage=anyd / n, supervised_px_mean=float(np.mean(sup)), damaged_share=f(dmg), strongly_damaged_share=f(dmg2),
                     line_band_px_mean=float(np.mean(near)), line_band_damaged_share=f(dnear))
    o = out[part]
    print(f"{part}: {n} frames; frames with any pixel off by > 0.25 m: {100 * o['frames_with_damage']:.1f} %; of the supervised pixels (< 4 m from a line, {o['supervised_px_mean']:.0f} per frame) "
          f"{100 * o['damaged_share']['mean']:.2f} % off by > 0.25 m (p90 {100 * o['damaged_share']['p90']:.2f} %, max {100 * o['damaged_share']['max']:.1f} %), "
          f"{100 * o['strongly_damaged_share']['mean']:.2f} % by > 1 m; inside the 1.2 m band: {100 * o['line_band_damaged_share']['mean']:.2f} % (p90 {100 * o['line_band_damaged_share']['p90']:.2f} %)", flush=True)
os.makedirs(os.path.join(os.environ["BFLC_ROOT"], "runs", "e19"), exist_ok=True)
json.dump(out, open(os.path.join(os.environ["BFLC_ROOT"], "runs", "e19", "e237_step0.json"), "w"), indent=1)
