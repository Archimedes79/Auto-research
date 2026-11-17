"""scratch (2026-11-15, qualitative report): overview picture of the frame selection: per-frame topo F1 histogram, E324 s against B per frame with the nine picked frames, per-scene pooled topo F1 and own-lane recall.
Numbers from the stored counts pickles of the cards (runs/e324_s_s0, runs/e312_p_s0, seed 0, VAL 873).  usage: wsl bash tools/env/_py14.sh overview_q1115.py <out.png>"""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = os.environ["BFLC_ROOT"]
H = f"{R}/temp/qual_report_1115"
rows = json.load(open(f"{H}/perframe_table.json"))
spec = json.load(open(f"{H}/spec_final2_q1115.json"))
c324 = pickle.load(open(f"{R}/runs/e324_s_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
c312 = pickle.load(open(f"{R}/runs/e312_p_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
sets = {"B": c324["base"], "E312": c312["head"], "E324 s": c324["head"]}
COL = {"B": "#8c8c8c", "E312": "#1f77b4", "E324 s": "#d95f02"}
sc = np.array(c324["scenes"])
scenes = sorted(set(sc.tolist()))


def pooled_topo(cs):
    t = sum(c["topo"]["topo"] for c in cs)
    return 2 * t[0] / (t[1] + t[2])


def pooled_own(cs):
    r = sum(c["tier"]["rec"][0] for c in cs)
    return r[0] / r[1]


fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.6), dpi=100)
fig.subplots_adjust(left=0.065, right=0.99, top=0.93, bottom=0.085, hspace=0.36, wspace=0.2)
a = ax[0, 0]
bins = np.linspace(0, 1, 26)
for key, tk in (("B", "B_topo"), ("E312", "E312_topo"), ("E324 s", "E324s_topo")):
    v = np.array([r[tk] for r in rows])
    a.hist(v, bins=bins, histtype="step", lw=1.6, color=COL[key], label=f"{key}: median {np.median(v):.2f}, mean {v.mean():.2f}")
a.set_xlabel("topo F1 of one frame", fontsize=8)
a.set_ylabel("frames (of 873)", fontsize=8)
a.set_title("(a) per-frame topo F1, VAL seed 0", fontsize=9)
a.legend(fontsize=7, loc="upper left")
a.tick_params(labelsize=7)
b = ax[0, 1]
x = np.array([r["B_topo"] for r in rows])
y = np.array([r["E324s_topo"] for r in rows])
b.scatter(x, y, s=5, color="#bbbbbb", linewidths=0)
b.plot([0, 1], [0, 1], color="k", lw=0.8)
for s in spec:
    r = rows[s["k"]]
    b.scatter([r["B_topo"]], [r["E324s_topo"]], s=26, color="#d95f02", zorder=3)
    b.annotate(s["name"][:2].lstrip("0"), (r["B_topo"], r["E324s_topo"]), xytext=(4, 3), textcoords="offset points", fontsize=8, weight="bold")
b.set_xlabel("B (plain graph): topo F1 of the frame", fontsize=8)
b.set_ylabel("E324 s: topo F1 of the frame", fontsize=8)
b.set_title("(b) E324 s against B, frame by frame (above the line: E324 s better)", fontsize=9)
n_up, n_dn = int((y > x + 0.02).sum()), int((y < x - 0.02).sum())
b.text(0.02, 0.96, f"E324 s better by > 0.02: {n_up} frames\nE324 s worse by > 0.02: {n_dn} frames\nwithin 0.02: {873 - n_up - n_dn} frames\nlabels 1 - 9 = pictures 1 - 9", transform=b.transAxes, fontsize=7, va="top")
b.set_xlim(0, 1)
b.set_ylim(0, 1)
b.tick_params(labelsize=7)
c = ax[1, 0]
order = sorted(scenes, key=lambda s: pooled_topo([c324["head"][i] for i in np.where(sc == s)[0]]))
w = 0.27
for j, (key, cs) in enumerate(sets.items()):
    c.bar(np.arange(len(order)) + (j - 1) * w, [pooled_topo([cs[i] for i in np.where(sc == s)[0]]) for s in order], w, color=COL[key], label=key)
c.set_xticks(np.arange(len(order)))
c.set_xticklabels([str(s) for s in order], fontsize=7)
c.set_xlabel("VAL scene (sorted by E324 s)", fontsize=8)
c.set_ylabel("pooled topo F1 of the scene", fontsize=8)
c.set_title("(c) per scene: pooled topo F1 (weakest: " + ", ".join(str(s) for s in order[:5]) + ")", fontsize=9)
c.legend(fontsize=7, ncol=3, loc="upper left")
c.set_ylim(0, 1.0)
c.tick_params(labelsize=7)
d = ax[1, 1]
order2 = sorted(scenes, key=lambda s: pooled_own([c324["head"][i] for i in np.where(sc == s)[0]]))
for j, (key, cs) in enumerate(sets.items()):
    d.bar(np.arange(len(order2)) + (j - 1) * w, [pooled_own([cs[i] for i in np.where(sc == s)[0]]) for s in order2], w, color=COL[key], label=key)
d.set_xticks(np.arange(len(order2)))
d.set_xticklabels([str(s) for s in order2], fontsize=7)
d.set_xlabel("VAL scene (sorted by E324 s)", fontsize=8)
d.set_ylabel("own-lane recall at 1 m (visible part)", fontsize=8)
d.set_title("(d) per scene: own lane (tier 1), pooled recall", fontsize=9)
d.set_ylim(0, 1.05)
d.tick_params(labelsize=7)
fig.suptitle("Selection overview: where the nine pictures sit (all numbers from the stored per-frame counts of the cards)", fontsize=10, weight="bold")
fig.savefig(sys.argv[1])
