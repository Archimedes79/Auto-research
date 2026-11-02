import math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

g = np.load(sys.argv[1], allow_pickle=True)["gt"].item()["cur"]
cam, rad, occ, und = (np.asarray(g[k], np.float32) > 0 for k in ("camera", "radar", "occluded", "undefined"))
nx, ny = cam.shape
res = 0.5
xs = (np.arange(nx) + 0.5) * res
ys = -26.0 + (np.arange(ny) + 0.5) * res
YY, XX = np.meshgrid(ys, xs)
half = math.atan(800.0 / 1266.4)
wedge = (np.abs(np.arctan2(YY, XX)) <= half) & (XX < 60.0)
EXT = (26, -26, 0, 60)

def show(ax, arr, cmap, title, vmin=0, vmax=1, wedge_lines=True):
    ax.imshow(np.asarray(arr, float)[:, ::-1], origin="lower", extent=EXT, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("y to the left (m)", fontsize=8)
    ax.set_ylabel("x forward (m)", fontsize=8)
    ax.plot([0], [0], marker="^", color="k", ms=8)
    if wedge_lines:
        xe = 26.0 / math.tan(half)
        for sg in (-1, 1):
            ax.plot([0, sg * 26.0], [0, xe], color="k", lw=1.0, ls=":")
    ax.tick_params(labelsize=7)

fig, axs = plt.subplots(2, 3, figsize=(13.5, 9.2))
show(axs[0, 0], cam, "Blues", f"1 Front camera: cells the image sees ({cam.mean()*100:.0f} % of the grid)")
show(axs[0, 1], rad, "Greens", f"2 Radar: union of the five fans ({rad.mean()*100:.1f} % of the grid)")
show(axs[0, 2], occ, "Reds", f"3 Hidden behind boxes, this frame ({occ.mean()*100:.1f} %)")
cls = np.where(cam & rad, 3, np.where(cam, 2, np.where(rad, 1, 0)))
show(axs[1, 0], cls, ListedColormap(["#d9d9d9", "#74c476", "#6baed6", "#08519c"]), "4 Who sees the cell", 0, 3)
ev = np.zeros(cam.shape); ev[~und] = 0.5; ev[cam] = 1.0
show(axs[1, 1], ev, ListedColormap(["#d9d9d9", "#bdbdbd", "#08519c"]), "5 Evaluation: where it scores", 0, 1)
w_map = (~und).astype(float) * np.where(occ, 0.5, 1.0)
w_obj = np.maximum((cam & ~occ).astype(float), rad.astype(float))
show(axs[1, 2], w_obj + 0.0 * w_map, "viridis", f"6 Training weight, objects ({(w_obj>0).mean()*100:.1f} % > 0)")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
axs[1, 0].legend(handles=[Patch(color="#08519c", label="camera + radar"), Patch(color="#74c476", label="radar only"), Patch(color="#d9d9d9", label="none (camera-only cells: 0)")], loc="upper right", fontsize=7)
axs[1, 1].legend(handles=[Patch(color="#08519c", label="maps scored here (camera cells)"), Patch(color="#bdbdbd", label="seen by radar only: not scored"), Line2D([0], [0], color="k", ls=":", label="detections scored inside the wedge")], loc="upper right", fontsize=7)
axs[1, 2].legend(handles=[Patch(color="#fde725", label="weight 1: camera (not hidden) or radar"), Patch(color="#440154", label="weight 0: seen by nothing")], loc="upper right", fontsize=7)
for ax in axs.ravel():
    ax.set_aspect("equal")
    ax.set_xlim(26, -26); ax.set_ylim(0, 60)
    ax.axvline(0, color="k", lw=0.3)
fig.suptitle("Sensor coverage on the 0.5 m grid (x 0 to 60 m, y +-26 m), real sample of the stored ground truth; the rig is fixed, so 1 and 2 hold for every frame", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("coverage.png", dpi=105)
print(f"grid {nx}x{ny}; camera {cam.mean():.3f} radar {rad.mean():.3f} occluded {occ.mean():.3f} undefined {und.mean():.4f}")
print(f"camera cells inside the detection wedge {(cam & wedge).sum()/max(cam.sum(),1):.3f}; wedge cells in the grid {wedge.mean():.3f}; wedge cells the camera does not see {(wedge & ~cam).sum()/wedge.sum():.3f}")
print(f"near blind zone: first camera-visible x at y=0: {xs[np.argmax(cam[:, ny//2])]:.1f} m")
print(f"camera-only {np.mean(cam & ~rad):.3f} radar-only {np.mean(rad & ~cam):.3f} both {np.mean(cam & rad):.3f} none {np.mean(~cam & ~rad):.4f}")
