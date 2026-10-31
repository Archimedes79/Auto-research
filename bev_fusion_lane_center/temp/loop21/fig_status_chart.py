"""Loop 21 status report, figure C and the numbers file (reads only metric cards of runs/eval, CPU).

    wsl bash tools/env/_py14.sh temp/loop21/fig_status_chart.py

Writes reports/source/figures/s21_numbers.png (Q against vertices per 100 m with the simplification arrows and the E257 sweeps; the quality metrics of the four systems) and
reports/source/weekly/s21_numbers.json (every number the report quotes, from the cards).
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIG = f"{R}/reports/source/figures"


def card(name, tag=""):
    p = f"{R}/runs/eval/{name}/val{'_' + tag if tag else ''}.json"
    return json.load(open(p, encoding="utf-8-sig"))["metrics"]


names = ["control", "champion", "champion_dp", "recall", "recall_dp", "fast_recall", "fast_recall_dp", "fast_recall_na"]
M = {n: {k: card(n)[k] for k in ("Q", "topo_F1", "lanes_found", "R@1_far", "J", "S", "vert100", "F1@0.5", "F1@1.0")} for n in names}

# E257 sweeps (the files written by the experiment, not typed here)
txt = open(f"{R}/experiments/E257_dp-output-simplification/dp_sweep.txt").read()
blocks = re.split(r"^== ", txt, flags=re.M)[1:]
sweep, vtx = {}, {}
for b in blocks:
    name = b.split(":")[0]
    pts = []
    for ln in b.splitlines():
        m = re.match(r"\s+(none|[0-9.]+)\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s", ln)
        if m:
            pts.append((m.group(1), float(m.group(3)), float(m.group(2))))
    sweep[name] = pts
    m2 = re.search(r"no simplification : (.*)", b)
    vtx[name] = [(float(a), float(c)) for a, c in re.findall(r"vtx_r [0-9.]+:\s+([0-9.]+) / ([0-9.]+)", m2.group(1))] if m2 else []
json.dump(dict(cards=M, sweep={k: v for k, v in sweep.items()}, vtx_sweep=vtx), open(f"{R}/reports/source/weekly/s21_numbers.json", "w"), indent=1)

fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
v = vtx["fast_recall"]
ax.plot([a for a, _ in v], [b for _, b in v], "o-", color="#999999", lw=1.2, ms=3.5, label="fast_recall: fewer vertices by a larger vertex radius vtx_r (1.5 ... 6 m)")
s = sweep["fast_recall"]
ax.plot([b for _, b, _ in s], [c for _, _, c in s], "s-", color="#1d4ea3", lw=1.6, ms=4.5, label="fast_recall: output stage simplify_dp, eps 0 / 0.1 / 0.2 / 0.3 / 0.5 / 0.75 / 1.0 m")
for lab, b, c in s:
    ax.annotate("" if lab == "none" else lab, (b, c), textcoords="offset points", xytext=(0, -12), fontsize=7, ha="center", color="#1d4ea3")
for n, col, mk in (("control", "#c85f00", "D"), ("champion", "#2a9d4a", "o"), ("recall", "#a0449c", "o"), ("fast_recall", "#1d4ea3", "o")):
    ax.plot(M[n]["vert100"], M[n]["Q"], mk, color=col, ms=8, zorder=5)
    ax.annotate(n, (M[n]["vert100"], M[n]["Q"]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=col)
for n, col in (("champion_dp", "#2a9d4a"), ("recall_dp", "#a0449c"), ("fast_recall_dp", "#1d4ea3")):
    b = n.replace("_dp", "")
    ax.annotate("", xy=(M[n]["vert100"], M[n]["Q"]), xytext=(M[b]["vert100"], M[b]["Q"]), arrowprops=dict(arrowstyle="->", color=col, lw=1.5), zorder=4)
    ax.plot(M[n]["vert100"], M[n]["Q"], "*", color=col, ms=13, zorder=6)
    ax.annotate(n, (M[n]["vert100"], M[n]["Q"]), textcoords="offset points", xytext=(0, 10) if n != "fast_recall_dp" else (0, -17), fontsize=8, color=col, ha="center")
ax.axvline(14.3, color="k", ls=":", lw=0.8)
ax.text(14.35, 0.7165, "budget 14.3", fontsize=7)
ax.set_xlim(6.5, 22.5)
ax.set_ylim(0.716, 0.746)
ax.annotate("vtx_r curve continues down: Q 0.616 at 12.0 vertices (off scale)", (11.2, 0.7185), fontsize=7, color="#666666")
ax.set_xlabel("vertices per 100 m (node ids for the plain systems, after simplify_dp for *_dp)")
ax.set_ylabel("Q (VAL 873 frames)")
ax.set_title("Q against the vertex count: the output stage lowers the count at the same Q", fontsize=10)
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.25)

ax = axs[1]
mets = [("Q", "Q"), ("topo_F1", "topo F1"), ("lanes_found", "lanes found"), ("R@1_far", "R@1 far"), ("J", "J (junction)")]
sys_ = [("control", "control E94\n21.8 v/100 m, 628 ms", "#c85f00"), ("champion_dp", "champion_dp\n9.5 v/100 m, 24 ms", "#2a9d4a"), ("recall_dp", "recall_dp\n10.4 v/100 m", "#a0449c"), ("fast_recall_dp", "fast_recall_dp\n9.9 v/100 m, 20-22 ms", "#1d4ea3")]
w = 0.2
x = np.arange(len(mets))
for k, (n, lab, col) in enumerate(sys_):
    vals = [M[n][m] for m, _ in mets]
    b = ax.bar(x + (k - 1.5) * w, vals, w, color=col, label=lab)
    for xx, vv in zip(x + (k - 1.5) * w, vals):
        ax.text(xx, vv + 0.008, f"{vv:.2f}", ha="center", fontsize=6.5, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in mets], fontsize=9)
ax.set_ylim(0, 1.18)
ax.set_title("VAL 873 frames: the four systems", fontsize=10)
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(f"{FIG}/s21_numbers.png", dpi=110)
print("saved s21_numbers.png and s21_numbers.json")
for n in names:
    print(n, {k: round(v, 4) for k, v in M[n].items()})
