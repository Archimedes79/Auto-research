"""Loop 18 figure: validation frames with ground truth, control (E94), champion and champion + fork pruning (K190), ego frame +-50 m (x right, y forward).
Frames: the three with the largest reduction of the fork-count error by pruning, one typical frame, and the one where pruning hurt most.  Writes reports/source/figures/p18_frames_forkprune.png
Usage (gcar): fig_p18_frames.py"""
import bflc_paths as BP  # project paths from the location of this checkout
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402
import render as RN  # noqa: E402

R = BP.ROOT
raw, gts = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p18/raw_val873.pkl"), "rb"))
net, cfg = TM.load_builder(288, 144)
ctrl = pickle.load(open(f"{R}/runs/e94_a/graphs_val_clean.pkl", "rb"))["preds"]
nop = [PI.assemble(r, None, 0.7, cfg, fast=2, **TM.ASM) for r in raw]
pru = [PI.assemble(r, None, 0.7, cfg, fast=2, fork_prune=0.3, **TM.ASM) for r in raw]


def nf(g):
    s, m = GE.junctions(g)
    return len(s) + len(m)


n_gt = np.array([nf(g) for g in gts])
d = np.array([abs(nf(a) - n) - abs(nf(b) - n) for a, b, n in zip(nop, pru, n_gt)])
order = np.argsort(-d)
mid = [i for i in order[len(order) // 3: len(order) // 3 + 40] if n_gt[i] >= 3][:1]
pick = [int(i) for i in order[:3]] + [int(i) for i in mid] + [int(order[-1])]
labels = ["largest fork-count gain", "largest fork-count gain", "largest fork-count gain", "typical", "pruning hurt most"]
import matplotlib.pyplot as plt  # noqa: E402
import viz_graph as VG  # noqa: E402

fig, axs = plt.subplots(len(pick), 4, figsize=(13, 3.3 * len(pick)))
for row, (i, lab) in enumerate(zip(pick, labels)):
    for col, (nm, g) in enumerate((("ground truth", gts[i]), ("control E94", ctrl[i]), ("champion", nop[i]), ("champion + fork pruning", pru[i]))):
        ax = axs[row, col]
        VG.draw_graph(ax, gts[i], color="#2a9d4a", lw=2.6, arrows=False, marks=False, alpha=0.35, zorder=1)
        if col > 0:
            VG.draw_graph(ax, g, color="#c8553d" if col < 3 else "#1f5fa8", lw=1.5)
        else:
            VG.draw_graph(ax, g, color="#2a9d4a", lw=1.8)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{nm}\nforks {nf(g)}" + (f" (GT {n_gt[i]})" if col else "") + (f"  · {lab}" if col == 0 else ""), fontsize=8)
fig.tight_layout()
out = f"{R}/reports/source/figures/p18_frames_forkprune.png"
fig.savefig(out, dpi=130)
print("saved", out, pick)
