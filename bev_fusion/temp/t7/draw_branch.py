"""t7 scratch: architecture drawing of the lane branch in the chain -> experiments/t7_lane_branch/images/architecture.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OLD, NEW, FROZEN = "#dfe7f2", "#d8efd9", "#ececec"
fig, ax = plt.subplots(figsize=(15, 6.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6.2)
ax.axis("off")


def box(x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#555555", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x0, y0, x1, y1, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#444444", lw=1))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", fontsize=7.5, color="#333333")


# main camera branch (unchanged)
box(0.2, 4.3, 2.0, 1.0, "front image\n256 x 704\n(+ 2 past frames)", OLD)
box(2.7, 4.3, 2.3, 1.0, "frozen DualBEV heads\n(cached): depth 118 bins\n+ context 80 ch, stride 16", FROZEN)
box(5.5, 4.3, 2.0, 1.0, "lift PD\n8 heights x 80 ch\n-> z-conv 64 ch", OLD)
box(8.0, 4.3, 1.9, 1.0, "temporal fusion\n(window + world\nmemory)", OLD)
arrow(2.2, 4.8, 2.7, 4.8)
arrow(5.0, 4.8, 5.5, 4.8)
arrow(7.5, 4.8, 8.0, 4.8)
# lane branch (new)
box(0.2, 1.6, 2.0, 1.0, "current frame\n256 x 704\n(image cache, JPEG 95)", NEW)
box(2.7, 1.6, 2.3, 1.0, "res4: frozen ResNet-50\nstem + layer 1 (256 ch)\nor rgb: CNN from scratch", NEW)
box(5.5, 1.6, 2.0, 1.0, "1x1 -> 32, 3x3 32\nstride 4\n(64 x 176 cells)", NEW)
box(8.0, 1.6, 1.9, 1.0, "lift PD\n8 heights x 32 ch\n-> 1x1 16 ch", NEW)
box(10.3, 1.6, 1.6, 1.0, "1x1 -> 64\nZERO start", NEW)
arrow(2.2, 2.1, 2.7, 2.1)
arrow(5.0, 2.1, 5.5, 2.1)
arrow(7.5, 2.1, 8.0, 2.1)
arrow(9.9, 2.1, 10.3, 2.1)
arrow(3.85, 4.3, 8.6, 2.6, "depth of the current frame (stride 16)")
# sum and the rest of the chain
ax.add_patch(plt.Circle((11.1, 4.8), 0.25, fc="white", ec="#555555"))
ax.text(11.1, 4.8, "+", ha="center", va="center", fontsize=14)
arrow(9.9, 4.8, 10.85, 4.8)
arrow(11.1, 2.6, 11.1, 4.55, "mirror like cam")
box(11.8, 4.3, 1.4, 1.0, "concat fusion\n+ radar 32 ch", OLD)
box(13.4, 4.3, 1.5, 1.0, "BEV trunk\n+ timed heads\n(sem, objects)", OLD)
arrow(11.35, 4.8, 11.8, 4.8)
arrow(13.2, 4.8, 13.4, 4.8)
ax.text(0.2, 5.85, "t7 lane branch: green = new, blue = reference chain (t2_ref), grey = frozen; graph-free pixel head path, "
        "fully differentiable", fontsize=10)
ax.text(0.2, 0.7, "The lift samples the stride-4 feature map and the stride-16 depth map at the same normalised image position of "
        "each voxel, so the branch reuses the existing depth.\nThe zero-initialised 1x1 makes the chain start exactly as the "
        "reference (gate: outputs equal, max difference 0.0); the branch is trained by the same losses.", fontsize=8.5)
os.makedirs("experiments/t7_lane_branch/images", exist_ok=True)
fig.savefig("experiments/t7_lane_branch/images/architecture.png", dpi=110, bbox_inches="tight")
print("experiments/t7_lane_branch/images/architecture.png")
