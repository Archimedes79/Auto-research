"""Charts for the 'systematic research' deck: effect sizes against noise, with the project's own numbers (digest_week2.md).
Run in gcar: tools/env/_py14.sh make_systematics_charts.py   -> reports/source/weekly/deck/s_*.png"""
import bflc_paths as BP  # project paths from the location of this checkout
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = BP.ROOT + "/reports/source/weekly/deck"
os.makedirs(OUT, exist_ok=True)
NAVY, BLUE, ORANGE, GREEN, GREY, RED = "#14213D", "#1D4EA3", "#C85F00", "#1A7F3C", "#8A939D", "#B3261E"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.spines.top": False, "axes.spines.right": False})

# 1 effects vs noise: composite S (band +-0.010) and route success (seed spread 0.05 - 0.09)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
s_items = [("teardrop pattern", 0.053), ("learned pair head", -0.040), ("ray reach 12 m", 0.020), ("keep gate", -0.015),
           ("stage B refine", -0.004), ("stage C rescore", 0.001), ("3 children + MCL", 0.000)]
names = [n for n, _ in s_items][::-1]
vals = [v for _, v in s_items][::-1]
a1.axvspan(-0.010, 0.010, color="#DCE6FA", zorder=0)
a1.barh(names, vals, color=[BLUE if abs(v) > 0.010 else GREY for v in vals])
for i, v in enumerate(vals):
    a1.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=12)
a1.axvline(0, color="#1b1f24", lw=1)
a1.set_xlim(-0.06, 0.075)
a1.set_title("Loop 16, effect on composite S\n(shaded: noise band ±0.010)", fontsize=14)
a1.set_xlabel("pooled effect on S")

r_items = [("hold vs single, night", 0.28), ("hold vs single, blur", 0.22), ("gate vs hold, night", 0.06), ("joint fine-tune, night hold (E202)", 0.066), ("hold on B, 3 cameras off", -0.05)]
names = [n for n, _ in r_items][::-1]
vals = [v for _, v in r_items][::-1]
a2.axvspan(-0.07, 0.07, color="#FBECEB", zorder=0)
a2.barh(names, vals, color=[BLUE if abs(v) > 0.09 else GREY for v in vals])
for i, v in enumerate(vals):
    a2.text(v + (0.008 if v >= 0 else -0.008), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=12)
a2.axvline(0, color="#1b1f24", lw=1)
a2.set_xlim(-0.2, 0.36)
a2.set_title("Route success effects under failure\n(shaded: seed spread ±0.07)", fontsize=14)
a2.set_xlabel("difference in route success")
fig.tight_layout()
fig.savefig(f"{OUT}/s_noise.png", dpi=170, facecolor="white")
plt.close(fig)

# 2 budget curve: control flat from 2 units, champion tracks it
fig, ax = plt.subplots(figsize=(7.2, 4.4))
u = [1, 2, 4, 8]
ax.plot(u, [0.713, 0.710, 0.707, 0.725], "o-", color=GREY, lw=2.5, label="control arm E94 (seed 0)")
ax.plot(u, [0.692, 0.712, 0.713, 0.726], "s--", color="#6B6F76", lw=2, label="control arm, seed 1")
ax.plot(u, [0.720, 0.718, 0.724, 0.725], "^-", color=BLUE, lw=2.5, label="design D7 (winner)")
ax.set_xscale("log", base=2)
ax.set_xticks(u)
ax.set_xticklabels(["1", "2", "4", "8"])
ax.set_xlabel("training budget (units of 1,200 iterations)")
ax.set_ylabel("composite S (VAL250)")
ax.set_ylim(0.68, 0.74)
ax.legend(frameon=False, fontsize=12, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/s_budget.png", dpi=170, facecolor="white")
plt.close(fig)
print("written", OUT)
