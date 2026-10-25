"""Loop 17 report figures from results_master.json: (1) failure-half composite S by failure type and system, (2) route success under failure, (3) what the map builder uses (synthetic vs real failure frames).
Usage (gcar): fig_p17_bars.py"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E17 = "/mnt/c/ML-Experiment/out/loop2/e17"
FIG = "/mnt/c/ML-Experiment/report_final/figures"
MJ = json.load(open(f"{E17}/results_master.json"))
PERTS = ["dark", "blur", "cam_front3_off", "extr3", "cam_front_off", "fog", "radar_off"]
NAME = dict(dark="night", blur="motion blur", cam_front3_off="3 front cameras off", fog="fog", radar_off="radar off", extr3="extrinsics error", cam_front_off="front camera off")
COL = dict(champ_single="#c85f00", champ_hold="#1f6fb4", gate="#6a3d9a", ctrl_hold="#777777", ctrl_single="#bbbbbb")
LAB = dict(champ_single="champion, single frame", champ_hold="champion + spatial hold", gate="champion + learned gate", ctrl_hold="control E94 + spatial hold", ctrl_single="control E94, single frame")

fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.2), dpi=150)
# (1) S failure half
ax = axs[0, 0]
keys = ["ctrl_single", "champ_single", "ctrl_hold", "champ_hold", "gate"]
w = 0.16
for i, k in enumerate(keys):
    vals = [MJ["perts"][p]["cards"].get(k, {}).get("fail", {}).get("S", np.nan) for p in PERTS]
    ax.bar(np.arange(len(PERTS)) + (i - 2) * w, vals, w, color=COL[k], label=LAB[k])
clean = MJ["perts"]["dark"]["cards"]["champ_single"]["cl"]["S"]
ax.axhline(clean, color="k", ls="--", lw=1)
ax.text(len(PERTS) - 0.55, clean + 0.006, f"clean half of the same scenes: S {clean:.3f}", ha="right", fontsize=8)
ax.set_xticks(range(len(PERTS)))
ax.set_xticklabels([NAME[p] for p in PERTS], fontsize=8, rotation=12)
ax.set_ylim(0.3, 0.8)
ax.set_ylabel("composite S, failure half")
ax.set_title("(1) What the failure costs, and what the memory brings back", fontsize=10, loc="left")
ax.legend(fontsize=7.5, ncol=2, loc="upper left")
# (2) route success
ax = axs[0, 1]
pr = ["dark", "blur", "cam_front3_off"]
sets = [("single", "e135_route/{p}_single", "#c85f00", "champion, single frame (recursion on)"), ("hold", "e135_route/{p}_maxdecay_0.95", "#1f6fb4", "champion + spatial hold (recursion off)"), ("gate", "e135_route/{p}_gate", "#6a3d9a", "champion + learned gate (recursion off)")]
for i, (k, pat, col, lab) in enumerate(sets):
    vals = []
    for p in pr:
        c = MJ["route"].get(pat.format(p=p), {})
        xs = [v["route_success"] for kk, v in c.items() if kk.startswith("sigma0_") and kk.endswith("_fail")]
        vals.append(np.mean(xs) if xs else np.nan)
    ax.bar(np.arange(len(pr)) + (i - 1) * 0.25, vals, 0.25, color=col, label=lab)
cl = [v["route_success"] for kk, v in MJ["route"].get("e135_route/dark_single", {}).items() if kk.startswith("sigma0_") and kk.endswith("_cl")]
ax.axhline(np.mean(cl), color="k", ls="--", lw=1)
ax.text(2.45, np.mean(cl) + 0.01, f"clean half: {np.mean(cl):.2f}", ha="right", fontsize=8)
ax.set_xticks(range(len(pr)))
ax.set_xticklabels([NAME[p] for p in pr], fontsize=8.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel("route success (two route-head seeds, exact map)")
ax.set_title("(2) The navigation attribute is more fragile than the graph", fontsize=10, loc="left")
ax.legend(fontsize=7.5, loc="upper right")
# (3) synthetic degradation: does the map builder use the map?
ax = axs[1, 0]
d = json.load(open(f"{E17}/e138b_mapdiag.json")) if os.path.exists(f"{E17}/e138b_mapdiag.json") else []
if d:
    names = ["champion, degraded frames", "map builder, degraded, no map", "map builder, degraded, wrong map", "map builder, degraded, map error 2 m", "map builder, degraded, exact map"]
    short = ["champion\n(no map input)", "map builder,\nno map", "wrong map", "map error 2 m", "exact map"]
    vals = [next(r["S"] for r in d if r["name"] == n) for n in names]
    found = [next(r["lanes_found"] for r in d if r["name"] == n) for n in names]
    ax.bar(range(5), vals, color=["#c85f00", "#999999", "#c0392b", "#d9a400", "#1a7f3c"])
    for i, (v, f) in enumerate(zip(vals, found)):
        ax.text(i, v + 0.008, f"S {v:.3f}\nfound {f:.2f}", ha="center", fontsize=7.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylim(0.3, 0.8)
ax.set_ylabel("composite S")
ax.set_title("(3) Synthetic loss of perception (as trained): the gain is training, the map adds little", fontsize=10, loc="left")
# (4) real failure frames, map builder (E138) and (E139 if present)
ax = axs[1, 1]
labels, series = [], []
for p in ["dark", "blur", "cam_front3_off"]:
    c = MJ["perts"][p]["cards"]
    labels.append(NAME[p])
    row = [c["norec_hold"]["fail"]["S"]]
    for k in ("map_hold_mnone", "map_hold_mwrong", "map_hold_m2", "map_hold_m0"):
        row.append(c[k]["fail"]["S"] if k in c else np.nan)
    series.append(row)
series = np.array(series)
cols = ["#1f6fb4", "#999999", "#c0392b", "#d9a400", "#1a7f3c"]
labs = ["champion + hold (no map input)", "map builder + hold, no map", "wrong map", "map error 2 m", "exact map"]
for i in range(5):
    ax.bar(np.arange(3) + (i - 2) * 0.16, series[:, i], 0.16, color=cols[i], label=labs[i])
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0.5, 0.7)
ax.set_ylabel("composite S, failure half")
ax.set_title("(4) Real failure frames: the same builder, trained on synthetic loss only", fontsize=10, loc="left")
ax.legend(fontsize=7.3, loc="lower right", ncol=1)
fig.tight_layout()
fig.savefig(f"{FIG}/p17_summary.png")
print("saved p17_summary.png")
