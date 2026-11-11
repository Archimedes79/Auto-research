"""scratch (Mirko's remark of 2026-11-08: the lane centres in the GT should look different): the GT lane pieces of one VAL frame drawn as lines (lane vs connector, inside the intersection or not), the training target of the
1 m centre map (heat with sigma 1 m, and the cells within 0.75 m) and the E279 prediction, side by side.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/gt_view.py [token]"""
import os
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import qpredict as QP  # noqa: E402
from autogt import lane_gt  # noqa: E402

tok = sys.argv[1] if len(sys.argv) > 1 else "5dd954d0ac0242a990ba35eef950c937"
g = lane_gt.load(tok)
fn = QP.make_pred_fn("e279_w_s0", "exact", "cpu", keep_maps=True)
pred = fn(tok)
p = np.asarray(pred["maps"]["centre1"][0], float)                       # (100, 100) rows = x, columns = y

fig, ax = plt.subplots(1, 3, figsize=(24, 8.2), dpi=60)
for a in ax:
    a.set_xlim(-50, 50)
    a.set_ylim(-50, 50)
    a.set_aspect("equal")
    a.set_xlabel("x right (m)")
    a.set_ylabel("y forward (m)")
# 1. the GT lines
a = ax[0]
for pc in g["pieces"]:
    q = np.asarray(pc["pts"], float)
    for s in pc["segments"]:
        seg = q[s["i0"]: s["i1"] + 1]
        col = "tab:red" if (s["kind"] == "connector" and s["in_intersection"]) else ("tab:orange" if s["kind"] == "connector" else "k")
        a.plot(seg[:, 0], seg[:, 1], "-", color=col, lw=1.6)
    a.plot(q[0, 0], q[0, 1], "o", color="tab:green", ms=3)
for pg in g.get("polygons", {}).get("intersection", []) if isinstance(g.get("polygons"), dict) else []:
    pg = np.asarray(pg["pts"] if isinstance(pg, dict) else pg, float)
    a.fill(pg[:, 0], pg[:, 1], color="0.85", zorder=0)
a.set_title(f"GT lane centre lines of the frame: {len(g['pieces'])} pieces (black lane, orange connector, red connector inside an intersection)")
# 2. the training target: heat (sigma 1 m) of the distance to the nearest GT point
pts = np.concatenate([np.asarray(pc["pts"], float) for pc in g["pieces"]])
c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
from scipy.spatial import cKDTree  # noqa: E402
d = cKDTree(pts).query(np.stack([CX.ravel(), CY.ravel()], 1))[0].reshape(100, 100)
heat = np.exp(-d * d / 2.0)
ax[1].imshow(heat.T, origin="lower", extent=(-50, 50, -50, 50), cmap="gray_r", vmin=0, vmax=1)
ax[1].contour(CX, CY, d, levels=[0.75], colors="tab:blue", linewidths=0.6)
ax[1].set_title("training target of the 1 m centre map: heat exp(-d^2 / 2), sigma 1 m (blue: d = 0.75 m)")
# 3. the prediction at 0.5 over the GT lines
ax[2].imshow((p >= 0.5).T, origin="lower", extent=(-50, 50, -50, 50), cmap="gray_r", vmin=0, vmax=1.4)
for pc in g["pieces"]:
    q = np.asarray(pc["pts"], float)
    ax[2].plot(q[:, 0], q[:, 1], "-", color="tab:red", lw=0.8)
ax[2].set_title("E279 prediction >= 0.5 (grey) with the GT lines (red)")
fig.tight_layout()
out = os.path.join(ROOT, "temp", "e284", "gt_view.png")
fig.savefig(out, dpi=60)
print("written", out, "pieces", len(g["pieces"]), "points", len(pts))
for pc in g["pieces"][:60]:
    print(pc["id"][:6] if isinstance(pc["id"], str) else pc["id"], pc["kind"], "len %.1f" % pc["length"], "pts", len(pc["pts"]), "lane_index", pc.get("lane_index"), "lane_count", pc.get("lane_count"), "road", str(pc.get("road"))[:8])
