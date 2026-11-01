"""scratch: scene-1077 keyframe 20 from the stored GT, every box numbered in the camera and in the three bird's-eye panels"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, ".")
import autogt
from autogt import view

OUT = "/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/5472e7ce-e594-4b06-a345-258377292ebc/scratchpad/scene-1077_k20_numbered.png"
tok = autogt.scene_tokens("scene-1077", "v1.0-trainval")[20]
g = autogt.load(tok)
fig = view.stored(tok, None)
ax = fig.axes
K, T = g["meta"]["K"], g["meta"]["T_c2e"]
for i, b in enumerate(g["meta"]["boxes"]):
    cx, cy, cz = b["center"]
    col = "yellow" if b["observed"] else "orange"
    label = f"{i}"
    if b["observed"] and 0 < cx < 62 and abs(cy) < 26:
        for a in ax[2:4]:
            a.text(cy, cx, label, color="black", fontsize=9, ha="center", va="center", weight="bold",
                   bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none", alpha=0.9))
    P = (np.array([cx, cy, cz, 1.0]) @ np.linalg.inv(T).T)[:3]
    if P[2] > 1:
        u, v = K[0, 0] * P[0] / P[2] + K[0, 2], K[1, 1] * P[1] / P[2] + K[1, 2]
        ax[0].text(u, v - 60, f"{label}  {cx:.0f} m", color="black", fontsize=8, ha="center", va="center", weight="bold",
                   bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none", alpha=0.9))
fig.savefig(OUT, dpi=95)
print("saved", OUT)
for i, b in enumerate(g["meta"]["boxes"]):
    print(i, b["group"], "observed" if b["observed"] else "ignored", f"x={b['center'][0]:.0f} y={b['center'][1]:.0f}", f"{np.hypot(*b['vel']) * 3.6:.0f} km/h")
