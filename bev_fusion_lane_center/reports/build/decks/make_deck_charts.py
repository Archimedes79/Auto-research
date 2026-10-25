"""Charts for the weekly PowerPoint (matplotlib, large fonts, one palette). Run in gcar: tools/env/_py14.sh make_deck_charts.py
Numbers: reports/source/weekly/digest_week1.md and digest_week2.md (same values as the two reports)."""
import bflc_paths as BP  # project paths from the location of this checkout
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = BP.ROOT + "/reports/source/weekly/deck"
os.makedirs(OUT, exist_ok=True)
NAVY, BLUE, ORANGE, GREEN, GREY, AMBER, RED, LB = "#14213D", "#1D4EA3", "#C85F00", "#1A7F3C", "#8A939D", "#9A6A00", "#B3261E", "#9DB8E8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#5b6570", "axes.labelcolor": "#1b1f24", "xtick.color": "#1b1f24", "ytick.color": "#1b1f24"})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bars(ax, x, series, labels, colors, w=0.38, fmt="{:.3f}", ylim=None, fs=12):
    n = len(series)
    for i, (s, lab, c) in enumerate(zip(series, labels, colors)):
        pos = [k + (i - (n - 1) / 2) * w for k in range(len(x))]
        b = ax.bar(pos, s, w * 0.92, label=lab, color=c)
        for r, v in zip(b, s):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + (0.008 if ylim is None else (ylim[1] - ylim[0]) * 0.012), fmt.format(v), ha="center", va="bottom", fontsize=fs)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    if ylim:
        ax.set_ylim(*ylim)


# 1 foundation
fig, ax = plt.subplots(figsize=(7.4, 4.6))
bars(ax, ["VAL", "TEST"], [[0.624, 0.598], [0.867, 0.835]], ["CRN (detection-pretrained)", "GaussianCaR (map-pretrained)"], [GREY, BLUE], ylim=(0.4, 0.95))
ax.set_ylabel("pixel F1 @ 1 m")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12)
save(fig, "c1_foundation")

# 2 topology
fig, ax = plt.subplots(figsize=(7.6, 4.6))
bars(ax, ["split F1 @6 m", "merge F1 @6 m"], [[0.342, 0.380], [0.552, 0.607], [0.714, 0.637]],
     ["grid graph", "+ learned key points", "+ Bezier connectors"], [GREY, BLUE, ORANGE], w=0.27, ylim=(0, 0.85))
ax.legend(frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
save(fig, "c2_topology")

# 3 route transfer
fig, ax = plt.subplots(figsize=(7.6, 4.6))
x = [0, 1, 2, 3]
ax.plot(x, [0.926, 0.640, 0.305, 0.153], "o-", color=GREY, lw=2.5, label="no registration")
ax.plot(x, [0.926, 0.845, 0.723, 0.608], "s-", color=ORANGE, lw=2.5, label="per-frame registration")
ax.plot(x, [0.926, 0.923, 0.883, 0.782], "^-", color=BLUE, lw=2.5, label="graph-ICP registration")
ax.plot(x, [0.926, 0.920, 0.920, 0.918], "D-", color=GREEN, lw=2.5, label="tracked registration")
ax.set_xlabel("map pose error σ (m)")
ax.set_ylabel("route success (TEST)")
ax.set_xticks(x)
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=12, loc="lower left")
save(fig, "c3_route_transfer")

# 4 failure
fig, ax = plt.subplots(figsize=(8.4, 4.4))
names = ["clean", "fog", "radar off", "front cam off", "extrinsics", "3 cams off", "blur", "night"]
vals = [0.867, 0.815, 0.816, 0.759, 0.695, 0.648, 0.433, 0.259]
cols = [BLUE] + [GREY] * 3 + [ORANGE] * 2 + [RED] * 2
b = ax.bar(names, vals, color=cols)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=12)
ax.set_ylabel("graph F1 @ 1 m (TEST)")
ax.set_ylim(0, 1.0)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
save(fig, "c4_failure")

# 5 hold / gate (loop 17)
fig, ax = plt.subplots(figsize=(8.0, 4.6))
bars(ax, ["night", "motion blur", "3 front cameras off"], [[0.424, 0.516, 0.580], [0.620, 0.639, 0.602], [0.644, 0.652, 0.598]],
     ["single frame", "+ spatial hold", "+ learned gate"], [ORANGE, BLUE, "#6B3FA0"], w=0.27, ylim=(0.3, 0.8), fmt="{:.2f}")
ax.axhline(0.750, color="#1b1f24", ls="--", lw=1.4)
ax.text(2.45, 0.757, "clean half: 0.750", ha="right", fontsize=12)
ax.set_ylabel("composite S, failure half (VAL)")
ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.12))
save(fig, "c5_hold_gate")

# 6 champion vs control
fig, ax = plt.subplots(figsize=(8.4, 4.6))
bars(ax, ["S", "F1 @ 1 m", "J (junctions)", "lanes found"], [[0.724, 0.883, 0.405, 0.800], [0.746, 0.870, 0.265, 0.695]],
     ["control E94 (21.8 vertices / 100 m)", "champion (13.5 vertices / 100 m)"], [GREY, BLUE], w=0.36, ylim=(0, 1.05), fmt="{:.3f}")
ax.legend(frameon=False, fontsize=12, loc="upper right")
ax.set_ylabel("VAL, 873 frames")
save(fig, "c6_champion")

# 7 route gap + speed
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
names = ["vertex\nhead", "head along\nthe edges", "control"]
vals = [0.798, 0.889, 0.885]
b = a1.bar(names, vals, color=[ORANGE, BLUE, GREY])
for r, v in zip(b, vals):
    a1.text(r.get_x() + r.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=13)
a1.set_ylim(0.6, 0.95)
a1.set_ylabel("route success (VAL)")
a1.tick_params(axis="x", labelsize=11)
names = ["champion", "fast setting", "control"]
vals = [31, 25, 628]
b = a2.barh(names, vals, color=[BLUE, "#6B8FD1", GREY])
for r, v in zip(b, vals):
    a2.text(v + 8, r.get_y() + r.get_height() / 2, f"{v} ms", va="center", fontsize=13)
a2.set_xlim(0, 760)
a2.invert_yaxis()
a2.set_xlabel("ms per frame (chain, idle RTX 3060)")
fig.tight_layout()
save(fig, "c7_route_speed")

# 8 map
fig, ax = plt.subplots(figsize=(10.2, 4.6))
cats = ["wrong\nmap", "map error\n2 m", "no\nmap", "map error\n1 m", "exact\nmap"]
night = [0.605, 0.643, 0.651, 0.660, 0.676]
blur = [0.602, 0.648, 0.653, 0.665, 0.677]
cams = [0.584, 0.644, 0.657, 0.666, 0.681]
bars(ax, cats, [night, blur, cams], ["night", "blur", "3 cameras off"], [NAVY, BLUE, LB], w=0.27, ylim=(0.5, 0.72), fmt="{:.2f}", fs=10)
ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=3)
ax.set_ylabel("composite S, failure half (VAL)")
ax.tick_params(axis="x", labelsize=12)
save(fig, "c8_map")

# 9 consumer
fig, ax = plt.subplots(figsize=(8.4, 4.6))
bars(ax, ["champion chain,\nroute head with exact map", "map builder B,\nexact map", "map builder B,\nno map"],
     [[4.11, 0.31, 0.54], [1.81, 0.29, 0.33], [1.78, 0.28, 0.34]], ["single frame", "spatial hold", "learned gate"], [ORANGE, BLUE, "#6B3FA0"], w=0.27, ylim=(0, 4.8), fmt="{:.2f}")
ax.set_ylabel("planner path error at 20 m, night (m)")
ax.legend(frameon=False, fontsize=12, loc="upper right")
ax.tick_params(axis="x", labelsize=12)
save(fig, "c9_consumer")
print("charts written to", OUT)
