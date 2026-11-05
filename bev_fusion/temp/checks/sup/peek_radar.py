"""throw-away: what is in the radar cache, how it matches the GT cache files, and how good the real Doppler is against the
GT velocity of the boxes (radial component) at keyframe k."""
import glob
import os
import sys

import numpy as np

sys.path[:0] = ["training"]
fs = sorted(glob.glob("data/cache/radar/*.npz"))
z = np.load(fs[0], allow_pickle=True)
print("radar file", os.path.basename(fs[0]), {k: (z[k].shape, z[k].dtype) for k in z.files})
names = {}
for f in fs:
    names.setdefault(os.path.basename(f).split("_")[0], []).append(os.path.basename(f))
print("samples with radar", len(names), "files per sample", np.bincount([len(v) for v in names.values()]))
print("suffixes", sorted({"_".join(n.split("_")[2:]) for v in list(names.values())[:2000] for n in v}))
gts = sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz"))
sig = {}
for f in gts:
    sig.setdefault(f.rsplit("_", 1)[1], 0)
    sig[f.rsplit("_", 1)[1]] += 1
print("gt signatures", sig)
tok = {os.path.basename(f).split("_")[0] for f in gts}
print("gt samples", len(tok), "with radar", len(tok & set(names)))
# Doppler quality on 200 samples: radar points inside vehicle boxes of k, vr vs u . v_gt (u from the point's psi)
errs, errs_static, n_pts, n_veh = [], [], [], []
xs0, ys0 = -0.0, -26.0
for f in gts[:4000:20]:
    t = os.path.basename(f).split("_")[0]
    if t not in names:
        continue
    g = np.load(f, allow_pickle=True)["gt"].item()["cur"]
    r = np.load(os.path.join("data/cache/radar", names[t][0]))
    ix = np.floor((r["x"] - 0.0) / 0.5).astype(int)
    iy = np.floor((r["y"] + 26.0) / 0.5).astype(int)
    ok = (ix >= 0) & (ix < 120) & (iy >= 0) & (iy < 104)
    n_pts.append(ok.sum())
    ix, iy = ix[ok], iy[ok]
    u = np.stack([np.cos(r["psi"][ok]), np.sin(r["psi"][ok])])
    vr = r["vr"][ok]
    vel = g["velocity"].astype(np.float32)[:, ix, iy]
    inst = g["instance"][ix, iy]
    pred = (u * vel).sum(0)
    veh = (g["vehicle"][ix, iy] > 0.5) & (inst >= 0)
    mov = veh & (np.hypot(*vel) > 1.0)
    n_veh.append(mov.sum())
    errs.append(np.abs(pred - vr)[mov])
    errs_static.append(np.abs(vr)[(inst < 0)])
    if len(n_pts) == 1:
        print("age range", r["dt"].min(), r["dt"].max(), "sensors", np.bincount(r["sensor"]))
e, es = np.concatenate(errs), np.concatenate(errs_static)
print(f"points in grid per sample: mean {np.mean(n_pts):.0f}; on moving vehicles: mean {np.mean(n_veh):.1f}")
print(f"moving-vehicle points |u.v_gt - vr|: median {np.median(e):.2f}, mean {e.mean():.2f}, p90 {np.percentile(e, 90):.2f} m/s (n {e.size})")
print(f"background points |vr|: median {np.median(es):.2f}, mean {es.mean():.2f}, p90 {np.percentile(es, 90):.2f} m/s (n {es.size})")
