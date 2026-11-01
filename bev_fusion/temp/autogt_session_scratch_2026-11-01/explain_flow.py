"""explain the all-red total-flow panel of scene-0061 keyframe 8: numbers per object + an annotated picture"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from autogt import AutoGT
from autogt.view import flow_rgb, SAT, semantic_rgb

gen = AutoGT("~/data/nuscenes", "v1.0-mini")
tok = gen.keyframes("scene-0061")[8]
gt = gen.keyframe(tok)
bx = gt["meta"]["boxes"]
fe = gt["flow_ego"]
print("static world flow at (30 m, 0): ", fe[:, 60, 52], "-> speed", np.hypot(*fe[:, 60, 52]), "m/s =", np.hypot(*fe[:, 60, 52]) * 3.6, "km/h")
rows = []
for i, b in enumerate(bx):
    if b["observed"] and b["vel_known"] and not b["stationary"] and np.hypot(*b["vel"]) > 1.0 and 0 < b["center"][0] < 60 and abs(b["center"][1]) < 26:
        cells = (gt["instance"] == i)
        f = gt["flow"][:, cells].mean(1) if cells.any() else None
        rows.append((b["group"], b["center"][0], b["center"][1], b["vel"], f))
        print(f"{b['group']:11s} at x {b['center'][0]:5.1f} y {b['center'][1]:5.1f}: own velocity (ego axes) {b['vel'].round(1)} m/s -> total flow {None if f is None else f.round(1)} m/s")
n_unknown = int(((gt["flow_weight"] == 0)).sum())
print("cells with weight 0 (unknown):", n_unknown)

fl = np.where((gt["flow_weight"] > 0)[None], gt["flow"], 0)
fig, ax = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
ax[0].imshow(np.flip(np.flip(flow_rgb(fl), 0), 1), extent=[26, -26, 0, 60], interpolation="nearest")
ax[0].set_xlim(26, -26); ax[0].set_ylim(0, 60)
ax[0].set_title("total apparent flow [m/s]: hue = direction, brightness = speed", fontsize=11)
ax[0].set_xlabel("y (m, left +)"); ax[0].set_ylabel("x forward (m)")
shown_ped = False
for g, x, y, v, f in rows:
    if f is None or (g != 'vehicle' and (shown_ped or x < 43)):
        continue
    shown_ped = shown_ped or g != 'vehicle'
    ax[0].annotate(f"{g[:3]}: own {np.hypot(*v) * 3.6:.0f} km/h\nflow ({f[0]:+.1f}, {f[1]:+.1f})", (y, x), xytext=(y + 6, x + 3), color="white", fontsize=7,
                   arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
ax[0].annotate(f"background: static world\nflow ({fe[0, 60, 52]:+.1f}, {fe[1, 60, 52]:+.1f}) m/s\n= the ego car's own speed, backwards", (-12, 8), color="white", fontsize=9,
               bbox=dict(facecolor="black", alpha=0.6))
# right: left a wheel, then the sum written out
n = 61
c = (n - 1) / 2
ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
wf = np.stack([(ii - c) / c * SAT, (jj - c) / c * SAT]).astype(np.float32)
wheel = flow_rgb(wf)
wheel[np.hypot(*wf) > SAT] = 1.0
ax[1].imshow(np.flip(np.flip(wheel, 0), 1), extent=[SAT, -SAT, -SAT, SAT])
ax[1].set_title("colour wheel: the flow vector (forward = up, left = left)\nred = pointing backward, black = no flow", fontsize=11)
ax[1].set_xlabel("flow y [m/s] (left +)"); ax[1].set_ylabel("flow x [m/s] (forward +)")
ax[1].plot(0, fe[0, 60, 52], "wo"); ax[1].annotate("static world", (0, fe[0, 60, 52]), xytext=(3, fe[0, 60, 52] - 1.5), color="white", fontsize=9)
for g, x, y, v, f in rows:
    if f is not None:
        ax[1].plot(-f[1], f[0], "w.", ms=5)
fig.tight_layout()
out = sys.argv[1]
fig.savefig(out, dpi=75)
print("saved", out)
