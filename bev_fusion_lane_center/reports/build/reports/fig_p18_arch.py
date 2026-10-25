"""Loop 18 architecture drawing: the chain from cached camera + radar features to the planner, with the blocks added in loop 17 (gate, map-conditioned builder) and loop 18 (registration, trust, fork pruning, numba union).
No red blocks. Colours: grey = frozen / cached, blue = learned and differentiable, green = memory, orange = new in loop 18, hatched = discrete step without parameters (assembly).
Writes reports/source/figures/p18_arch.png.  Usage (gcar): fig_p18_arch.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
fig, ax = plt.subplots(figsize=(15, 6.6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6.6)
ax.axis("off")
COL = dict(grey="#d9dce1", blue="#cfe0f5", green="#d3ecd9", orange="#fbdcb0", plain="#ffffff")
EDGE = dict(grey="#6b7480", blue="#1f5fa8", green="#2a9d4a", orange="#c97a12", plain="#444444")


def box(x, y, w, h, text, kind="blue", hatch=None, fs=8.3, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=COL[kind], ec=EDGE[kind], lw=1.6, hatch=hatch)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True, bbox=dict(fc=COL[kind], ec="none", pad=1.5) if hatch else None)
    return (x, y, w, h)


def arrow(a, b, side_a="r", side_b="l", text=None, color="#444444", rad=0.0, dy=0.0, ls="-"):
    def pt(bx, s):
        x, y, w, h = bx
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[s]

    p = FancyArrowPatch(pt(a, side_a), pt(b, side_b), arrowstyle="-|>", mutation_scale=12, lw=1.4, color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(p)
    if text:
        (x0, y0), (x1, y1) = pt(a, side_a), pt(b, side_b)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12 + dy, text, ha="center", fontsize=7, color=color)


# main row
b1 = box(0.1, 3.9, 1.7, 1.2, "camera + radar\nBEV features\n(GaussianCaR,\nfrozen, cached)", "grey")
b2 = box(2.2, 3.9, 1.6, 1.2, "dense lane head\nheat + direction\n(out-of-fold for\ntraining rows)", "blue")
b3 = box(4.2, 3.9, 1.7, 1.2, "memory at the input\nspatial hold +\nlearned gate\n(loop 17)", "green")
b4 = box(6.3, 3.9, 2.3, 1.2, "builder (loop 16 / 17)\nlattice seeds, teardrop tree,\nsurvival read-out,\njunction head, map input", "blue", bold=False)
b5 = box(9.0, 3.9, 2.2, 1.2, "assembly: path union\n(numba kernels), anchoring,\nfork relocation,\nFORK PRUNING (K190)", "orange", hatch="//")
b6 = box(11.6, 3.9, 1.5, 1.2, "sparse lane\ngraph\n(~14 vertices\nper 100 m)", "plain")
b7 = box(13.4, 3.9, 1.5, 1.2, "planner\nconsumer\n(path from the\nroute lanes)", "blue")
for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6), (b6, b7)):
    arrow(a, b)
# route head under the graph
rh = box(10.6, 1.9, 3.0, 1.2, "route head along the edges\n(cross-attention to the SD map,\nTRAINED ON FAILURE FRAMES)", "orange")
arrow(b6, rh, "b", "t", text="points every 4 m")
arrow(rh, b7, "r", "b", text="route per lane", rad=-0.2)
# SD map branch
sd = box(0.1, 0.3, 1.7, 1.0, "SD map\n(roads, route,\nlane count)", "grey")
rg = box(2.4, 0.3, 3.0, 1.4, "rough localisation (K188)\npose-grid correlation, soft-argmax\n(translation x yaw), world-frame\naccumulation over time", "orange")
tr = box(5.9, 0.3, 2.3, 1.4, "map trust (K189)\nevidence share + peak margin\n-> gate 0 / 1 (abstain)", "orange")
arrow(sd, rg)
arrow(rg, tr)
arrow(tr, b4, "t", "b", text="corrected map x trust", dy=-0.1)
arrow(b2, rg, "b", "t", text="heat + direction", rad=0.12, ls="--", color="#1f5fa8", dy=0.3)
# legend
for i, (k, t) in enumerate((("grey", "frozen / cached"), ("blue", "learned, differentiable"), ("green", "memory"), ("orange", "new in loop 18"), ("plain", "output"))):
    ax.add_patch(FancyBboxPatch((0.2 + i * 2.6, 6.0), 0.35, 0.28, boxstyle="round,pad=0.01", fc=COL[k], ec=EDGE[k], lw=1.4))
    ax.text(0.65 + i * 2.6, 6.14, t, fontsize=8, va="center")
ax.add_patch(FancyBboxPatch((13.0, 6.0), 0.35, 0.28, boxstyle="round,pad=0.01", fc="white", ec="#444444", lw=1.4, hatch="//"))
ax.text(13.45, 6.14, "discrete step", fontsize=8, va="center")
ax.text(7.5, 5.55, "Gradient path in training: loss -> graph / route / consumer -> tokens -> builder -> stem; the assembly and the pruning decide which nodes exist and carry no gradient.", ha="center", fontsize=8, style="italic", color="#444444")
fig.savefig(f"{R}/reports/source/figures/p18_arch.png", dpi=140, bbox_inches="tight")
print("saved")
