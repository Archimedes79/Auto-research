"""Loop 17: architecture drawing of the robust, fully differentiable chain: frames -> frozen foundation -> dense head -> learned gated memory (warped previous input) -> lane-graph builder -> assembly -> sparse lane graph -> route head (SD map) -> planner consumer.
No red blocks: learned, parameter-free differentiable, selection, export. Usage: fig_p17_arch.py [out.png]"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/ML-Experiment/report_final/figures/p17_arch.png"
KIND = {"learned": ("#d7e8f7", "#1b6ca8"), "algebra": ("#dff2e3", "#2a8a4a"), "select": ("#fdecc8", "#c98a00"), "export": ("#ececec", "#777777"), "input": ("#f4f4f4", "#444444"), "frozen": ("#e8e8f4", "#5560a0")}
fig, ax = plt.subplots(figsize=(16, 8.2))
ax.set_xlim(0, 160)
ax.set_ylim(0, 84)
ax.axis("off")
NL = "\n"


def block(x, y, w, h, title, sub="", kind="learned", fs=9.6):
    fc, ec = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2", fc=fc, ec=ec, lw=1.6, ls="--" if kind == "export" else "-"))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=fs, fontweight="bold", color="#222")
    if sub:
        ax.text(x + w / 2, y + h - 5.0, sub, ha="center", va="top", fontsize=fs - 1.8, color="#333", linespacing=1.3)


def arrow(x0, y0, x1, y1, color="#555", ls="-", rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=lw, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}"))


ax.text(80, 81, "Loop 17: the robust, fully differentiable lane-graph chain (frame -> sparse lane graph with a navigation attribute -> planner)", ha="center", fontsize=13.5, fontweight="bold", color="#222")
H = 15
# row 1
block(1, 54, 20, H, "Camera + radar", "6 cameras, 7 radar sweeps" + NL + "(the sensors that fail)", "input")
block(26, 54, 24, H, "Foundation (frozen)", "GaussianCaR, 39 BEV channels" + NL + "200 x 200 at 0.5 m, +-50 m", "frozen")
block(55, 54, 24, H, "Dense head", "U-Net: lane heat + direction" + NL + "(3 channels), 42 channels in all", "learned")
block(84, 54, 28, H, "Learned gated memory", "gate g = f(frame, warped memory)" + NL + "y = g * memory + (1 - g) * frame" + NL + "3 convs, 12.5 k parameters", "learned")
block(117, 54, 28, H, "Lane-graph builder D7", "stem, seed lattice, teardrop tracing," + NL + "junction head; 1.2 M parameters" + NL + "(time recursion off: no gain)", "learned")
for x0, x1 in ((21.4, 25.6), (50.4, 54.6), (79.4, 83.6), (112.4, 116.6)):
    arrow(x0, 61.5, x1, 61.5)
# memory loop above the gate
block(84, 74, 28, 5.2, "", "", "algebra")
ax.text(98, 77.6, "previous fused input y(t-1), warped by the ego motion (grid_sample)", ha="center", va="center", fontsize=8.0, color="#222")
arrow(98, 69.6, 98, 73.5, color="#2a8a4a")
arrow(104, 73.6, 104, 69.8, color="#2a8a4a")
ax.text(105, 71.6, "memory", fontsize=7.8, color="#2a8a4a")
ax.text(88.6, 71.6, "y(t)", fontsize=7.8, color="#2a8a4a")
# optional map input to the builder
block(117, 74, 28, 5.2, "", "", "input")
ax.text(131, 77.6, "optional (E138): SD-map raster, 5 channels", ha="center", va="center", fontsize=8.0, color="#222")
arrow(131, 73.6, 131, 69.8, color="#777", ls="--")
# row 2 (right to left reading, drawn left to right under)
arrow(131, 53.6, 131, 45.4)
block(117, 30, 28, H, "Assembly (export)", "beam of hypotheses, path union," + NL + "anchoring at junction peaks;" + NL + "no gradient needed", "export")
block(84, 30, 28, H, "Sparse lane graph", "vertices, directed edges, validity;" + NL + "14 vertices per 100 m" + NL + "node tokens h at every vertex", "algebra")
block(51, 30, 28, H, "Route head along the edges", "tokens every <= 4 m, cross-attention" + NL + "to the SD-map raster (error, wrong," + NL + "no map)", "learned")
block(18, 30, 28, H, "Planner consumer", "edges + route probability + ego" + NL + "motion -> path at 20 m;" + NL + "93 k parameters", "learned")
for x0, x1 in ((116.6, 112.4), (83.6, 79.4), (50.6, 46.4)):
    arrow(x0, 37.5, x1, 37.5)
ax.text(114.6, 39.4, "", fontsize=8)
# SD map into the route head
block(51, 8, 28, 11, "SD map (navigation grade)", "roads, route, lane count;" + NL + "pose error 0 - 3 m, wrong, missing", "input")
arrow(65, 19.6, 65, 29.4)
# gradient arrows
arrow(30, 29.4, 98, 29.4, color="#7a4aa3", ls="--", rad=-0.0, lw=1.2)
ax.text(36, 26.2, "gradient of the route and planner loss: through the node tokens to the builder (E136) and, if wanted, to the gate and the dense head", fontsize=8.2, color="#7a4aa3")
arrow(97, 29.4, 128, 52.6, color="#7a4aa3", ls="--", rad=0.12, lw=1.2)
# legend
lx = 100
for i, (k, t) in enumerate((("learned", "learned module"), ("algebra", "parameter-free, differentiable"), ("select", "selection"), ("export", "export at inference"), ("frozen", "frozen foundation"))):
    fc, ec = KIND[k]
    ax.add_patch(FancyBboxPatch((lx, 16.6 - i * 3.4), 3.0, 2.0, boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=1.4, ls="--" if k == "export" else "-"))
    ax.text(lx + 4.2, 17.6 - i * 3.4, t, fontsize=8.4, va="center", color="#222")
ax.text(1, 3.6, "Every block is learned or differentiable; the only discrete step is the assembly (a selection of hypotheses), through whose kept nodes the gradient flows to the tokens." + NL + "Failure handling (loop 17): the memory gate sees the previous fused input, so a failed frame is bridged by what was seen before.", fontsize=8.8, color="#333", linespacing=1.4)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
