"""scratch: full-resolution camera close-ups with the GT map lines and 3D boxes (is the GT on the paint / on the cars?).
  closeups.py SIGNATURE scene:k ...   -> temp/agt_gallery/close/<scene>_k<kk>.png (1600 x 900)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, ".")
from autogt import load, scene_tokens  # noqa: E402
from autogt import view  # noqa: E402

sig = sys.argv[1]
root = os.path.expanduser("~/data/nuscenes_full")
os.makedirs("temp/agt_gallery/close", exist_ok=True)
for pick in sys.argv[2:]:
    name, k = pick.split(":")
    tok = scene_tokens(name, "v1.0-trainval")[int(k)]
    gt = load(tok)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("RGB"))
    for layer, pls in gt["meta"]["vectors"].items():
        for pl in pls:
            d = view.densify(pl, 0.5)
            keep = (d[:, 0] > 1) & (d[:, 0] < 70) & (np.abs(d[:, 1]) < 30)
            for run_ in np.split(np.arange(len(d)), np.where(np.diff(keep.astype(int)) != 0)[0] + 1):
                if keep[run_[0]] and len(run_) >= 2:
                    view.draw_3d(ax, gt, np.c_[d[run_], np.zeros(len(run_))], view.LCOL[layer], lw=1.2)
    for b in gt["meta"]["boxes"]:
        if 0.5 < b["center"][0] < 70 and b["observed"]:
            xyz = view.corners_3d(b)
            for i, j in view.EDGES:
                view.draw_3d(ax, gt, xyz[[i, j]], view.COL[b["group"]], lw=1.4)
    ax.set_xlim(0, 1600)
    ax.set_ylim(900, 0)
    ax.axis("off")
    fig.savefig(f"temp/agt_gallery/close/{name}_k{int(k):02d}.png")
    plt.close(fig)
    print("saved", name, k, flush=True)
