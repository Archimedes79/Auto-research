"""t4_soa_front: figure of results_x50.md (numbers parsed from the table, not typed). Usage: python plot_compare.py <results.md> <out.png>"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

tabs, cur = [], None
for ln in open(sys.argv[1]):
    if ln.startswith("|"):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if set(cells[0]) <= set("-"):
            continue
        if cells[0] == "system":
            cur = []
            tabs.append(cur)
        else:
            cur.append(cells)
t1, t2 = tabs
names = {"DualBEV-front": "DualBEV (pre-trained),\nfront camera only", "ours-camera-only": "ours, camera only\n(3000 steps)",
         "ours-ref_s0": "ours, camera + radar\n(3000 steps)", "ours-long-7000x2": "ours, camera + radar\n(7000 x 2 steps)"}
col = {"DualBEV-front": "#7f7f7f", "ours-camera-only": "#9ecae1", "ours-ref_s0": "#3182bd", "ours-long-7000x2": "#08519c"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1.5]})
row = {r[0]: r for r in t1}
for i, k in enumerate(names):
    v = float(row[k][1])
    ax[0].bar(i, v, color=col[k])
    ax[0].text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
ax[0].set_xticks(range(len(names)), [names[k] for k in names], fontsize=7.5)
ax[0].set_ylabel("mAP2 (vehicle + pedestrian, 0.5 / 1 / 2 / 4 m)")
ax[0].set_title("front sector 0-50 m, cold VAL, 1000 frames")
rows2 = {r[0]: r for r in t2}
hdr = ["0-20 m", "20-40 m", "40-50 m"]
w = 0.2
for j, k in enumerate(names):
    vals = [float(x) for x in rows2[k][1:]]
    xs = np.arange(3) + (j - 1.5) * w
    ax[1].bar(xs, vals, w, color=col[k], label=names[k].replace("\n", " "))
    for x, v in zip(xs, vals):
        ax[1].text(x, v + 0.006, f"{v:.2f}", ha="center", fontsize=7)
ax[1].set_xticks(range(3), hdr)
ax[1].set_title("mAP2 by distance of the object")
ax[1].legend(fontsize=7.5, frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("t4: state-of-the-art camera detector vs our system, only the front area visible", fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=140)
print("saved", sys.argv[2])
