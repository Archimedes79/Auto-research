import os, numpy as np
from autogt import coverage as C
from autogt.cli import load_keyframe
d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(2)
rows = []
for i in rng.choice(len(files), 400, replace=False):
    g = load_keyframe(d, files[i][:-4]); m = g["meta"]
    for b in m["boxes"]:
        if not b["observed"]: continue
        fp, lo, hi = C.prism_of(b)
        c = fp.mean(0)
        if not (0 <= c[0] <= 60 and abs(c[1]) <= 26): continue
        rows.append((lo, hi, c[0], b["size"][2], b["group"] == "vehicle", b["group"] == "pedestrian"))
r = np.array(rows, float)
print("observed boxes in grid:", len(r))
print("z_lo quantiles 1/5/25/50/75/95/99:", np.quantile(r[:, 0], [.01, .05, .25, .5, .75, .95, .99]).round(2))
print("z_hi quantiles 1/5/25/50/75/95/99:", np.quantile(r[:, 1], [.01, .05, .25, .5, .75, .95, .99]).round(2))
print("share with z_hi < 0.3 (box entirely below the ground plane, hides nothing):", (r[:, 1] < 0.3).mean().round(4))
print("share with z_lo > 0.5:", (r[:, 0] > 0.5).mean().round(4), " > 1.0:", (r[:, 0] > 1.0).mean().round(4))
print("share with |z_lo| > 1.0:", (np.abs(r[:, 0]) > 1.0).mean().round(4))
for lo_d, hi_d in ((0, 20), (20, 40), (40, 60)):
    s = (r[:, 2] >= lo_d) & (r[:, 2] < hi_d)
    print(f"x {lo_d}-{hi_d}: n={s.sum()} z_lo median {np.median(r[s,0]):.2f} p5 {np.quantile(r[s,0],.05):.2f} p95 {np.quantile(r[s,0],.95):.2f}  vehicles z_hi < 1.0: {(r[s & (r[:,4]>0),1] < 1.0).mean():.3f}")
