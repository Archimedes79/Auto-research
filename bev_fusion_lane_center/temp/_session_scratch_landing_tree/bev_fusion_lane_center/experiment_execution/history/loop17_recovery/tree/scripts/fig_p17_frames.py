"""Loop 17 report figure: failure-phase frames of one failure type, ground truth / champion single frame / champion with spatial hold / champion with learned gate / control with hold,
selected by percentiles of the per-frame F1@1 difference (gate minus single frame), not by looks.
Usage (gcar): fig_p17_frames.py <pert> [out.png]"""
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p17_compare as CMP  # noqa: E402

pert = sys.argv[1] if len(sys.argv) > 1 else "dark"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/mnt/c/ML-Experiment/report_final/figures/p17_frames_{pert}.png"
GREEN, ORANGE, PURPLE, BLUE, GREY = "#1a7f3c", "#c85f00", "#6a3d9a", "#1f6fb4", "#555555"
ent = {
    "single": CMP.load(f"builder:e130_champion/{pert}_single", pert),
    "hold": CMP.load(f"builder:e131_hold/{pert}_maxdecay_0.95", pert),
    "gate": CMP.load(f"builder:e134_gate/{pert}_gate", pert),
    "ctrl": CMP.load("control:maxdecay_0.95", pert),
}
rows = ent["single"]["rows"]
for k, d in ent.items():
    assert (np.asarray(d["rows"]) == np.asarray(rows)).all(), k
gts = CMP.gts_of(rows)
fail = np.where(ent["single"]["fail"])[0]
f1 = lambda g, t: float(GE.evaluate([g], [t])["F1@1.0"])
fs = {k: np.array([f1(d["graphs"][i], gts[i]) for i in fail]) for k, d in ent.items()}
diff = fs["gate"] - fs["single"]
order = np.argsort(diff)
sel = [int(order[int(q * (len(fail) - 1))]) for q in (0.1, 0.35, 0.65, 0.9)]


def panel(ax, g, t, col, title, gt_only=False):
    for p in t["pts"]:
        p = np.asarray(p)
        if len(p) > 1:
            ax.plot(p[:, 0], p[:, 1], "-", color=GREEN, lw=5 if not gt_only else 1.6, alpha=0.25 if not gt_only else 1.0, solid_capstyle="round")
    if not gt_only:
        for p in g["pts"]:
            p = np.asarray(p)
            if len(p) > 1:
                ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.6)
                ax.plot(p[:, 0], p[:, 1], "o", color=col, ms=2.6)
    ax.plot([0], [0], "s", color="k", ms=4)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7.2, loc="left")


names = [("single", "champion, single frame", ORANGE), ("hold", "champion + spatial hold", BLUE), ("gate", "champion + learned gate", PURPLE), ("ctrl", "control E94 + spatial hold", GREY)]
fig, axs = plt.subplots(5, 4, figsize=(11.2, 13.4), dpi=150)
for j, s in enumerate(sel):
    i = int(fail[s])
    q = ("10th", "35th", "65th", "90th")[j]
    panel(axs[0, j], gts[i], gts[i], GREEN, f"ground truth, frame {int(rows[i])}\n({q} percentile of gate minus single frame)", gt_only=True)
    for r, (k, nm, col) in enumerate(names, start=1):
        panel(axs[r, j], ent[k]["graphs"][i], gts[i], col, f"{nm}: F1@1 {fs[k][s]:.2f}")
fig.suptitle(f"Failure type '{pert}', failure phase. Columns: frames at the 10th / 35th / 65th / 90th percentile of the per-frame F1@1\ndifference (learned gate minus single frame). Green band = ground truth, black square = the car.", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT)
print("saved", OUT, {k: round(float(v.mean()), 3) for k, v in fs.items()})
