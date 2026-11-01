"""the red total-flow panel and the same field with the static world moved to the centre (ego motion removed)"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from autogt import AutoGT
from autogt.view import flow_rgb, SAT

gen = AutoGT("~/data/nuscenes", "v1.0-mini")
gt = gen.keyframe(gen.keyframes("scene-0061")[8])
bx = gt["meta"]["boxes"]
w = gt["flow_weight"] > 0
fe = gt["flow_ego"][:, 60, 52]
tot = np.where(w[None], gt["flow"], 0)
rel = np.where(w[None], gt["velocity"], 0)
unk = ~w


def panel(ax, f, title, centre):
    rgb = flow_rgb(f)
    rgb[unk] = (0.5, 0.5, 0.1)
    ax.imshow(np.flip(np.flip(rgb, 0), 1), extent=[26, -26, 0, 60], interpolation="nearest")
    ax.set_xlim(26, -26); ax.set_ylim(0, 60)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("y (m, left +)"); ax.set_ylabel("x forward (m)")


fig, ax = plt.subplots(1, 3, figsize=(21, 7.2), gridspec_kw=dict(width_ratios=[1, 1, 1]))
panel(ax[0], tot, f"1. total apparent flow: everything moves backwards\nat the ego speed ({-fe[0]:.1f} m/s = {-fe[0] * 3.6:.0f} km/h)", None)
panel(ax[1], rel, "2. static world in the centre (ego motion removed):\nthe road is black, only moving things have colour", None)
for g, x, y, v, name in [(b["group"], b["center"][0], b["center"][1], b["vel"], i) for i, b in enumerate(bx)
                         if b["observed"] and b["vel_known"] and not b["stationary"] and np.hypot(*b["vel"]) > 3.0 and 0 < b["center"][0] < 60 and abs(b["center"][1]) < 26]:
    ax[1].annotate(f"{np.hypot(*v) * 3.6:.0f} km/h", (y, x), xytext=(y + 5, x + 4), color="white", fontsize=8, arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
n = 61
c = (n - 1) / 2
ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
wf = np.stack([(ii - c) / c * SAT, (jj - c) / c * SAT]).astype(np.float32)
wheel = flow_rgb(wf)
wheel[np.hypot(*wf) > SAT] = 1.0
ax[2].imshow(np.flip(np.flip(wheel, 0), 1), extent=[SAT, -SAT, -SAT, SAT])
ax[2].set_title("colour wheel = the flow vector (forward up, left left)\nbrightness = speed, centre (black) = no motion", fontsize=10)
ax[2].set_xlabel("flow y [m/s] (left +)"); ax[2].set_ylabel("flow x [m/s] (forward +)")
ax[2].plot(0, 0, "wo", ms=9); ax[2].annotate("panel 2: static world\n(centre)", (0, 0), xytext=(3, 1.5), color="white", fontsize=9)
ax[2].plot(-fe[1], fe[0], "ws", ms=9); ax[2].annotate("panel 1: static world\n(moved by the ego motion)", (-fe[1], fe[0]), xytext=(2.5, fe[0] - 1.2), color="white", fontsize=9)
ax[2].annotate("", (0, 0), (-fe[1], fe[0]), arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=75)
print("saved")
