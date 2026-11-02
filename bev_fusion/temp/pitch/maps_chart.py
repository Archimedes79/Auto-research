"""Maps of our chains and of GaussianCaR (SOA) on the SAME 400 cold VAL frames, same metric (map_ref.py of t6): a table and a chart from the stored json.
Usage (bev_fusion folder, gcar python): python temp/pitch/maps_chart.py <out.png> <out.md>"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_PNG, OUT_MD = sys.argv[1], sys.argv[2]
SRC = [("t8_ctl_s0", "temp/soa_gap/map_ref_t8_ctl_s0_400.json", "ours, start (3000 steps)"),
       ("t1_r4_soa", "temp/soa_gap/map_ref_r4soa_400.json", "ours, start with 4.7x the training"),
       ("t8_bevfeat_s0", "temp/soa_gap/map_ref_t8_bevfeat_s0_400.json", "ours, map-trained features (3000 steps)"),
       ("t10_dual_s0", "temp/soa_gap/map_ref_t10_dual_s0_400.json", "ours, both feature sets (3000 steps)")]
LAYERS = [("drivable", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("carpark", "carpark_area", "iou@0"),
          ("crossing", "ped_crossing", "iou@0"), ("lane divider\n(0.5 m)", "lane_divider", "iou@0.5"), ("road divider\n(0.5 m)", "road_divider", "iou@0.5")]
data = {}
for key, path, label in SRC:
    if os.path.exists(path):
        data[key] = (label, json.load(open(path)))
if not data:
    raise SystemExit("no map_ref json yet")


def val(d, system, layer, tol):
    return d["systems"][system][layer]["all"]["0.5"][tol]


first = next(iter(data.values()))[1]
series = [("prior map (null)", [val(first, "prior", l, t) for _, l, t in LAYERS], "#cfd4da")]
palette = ["#9ecae1", "#6baed6", "#2171b5", "#08306b"]
for (key, (label, d)), col in zip(data.items(), palette):
    series.append((label, [val(d, "ours", l, t) for _, l, t in LAYERS], col))
series.append(("GaussianCaR (SOA, 6 cameras + radar, map-trained)", [val(first, "gcar", l, t) for _, l, t in LAYERS], "#d9480f"))

fig, ax = plt.subplots(figsize=(15, 6.2), dpi=130)
w = 0.85 / len(series)
for i, (label, vals, col) in enumerate(series):
    xs = [j + (i - (len(series) - 1) / 2) * w for j in range(len(LAYERS))]
    bars = ax.bar(xs, vals, w * 0.92, label=label, color=col, edgecolor="none")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#222")
ax.set_xticks(range(len(LAYERS)))
ax.set_xticklabels([n for n, _, _ in LAYERS], fontsize=11)
ax.set_ylabel("IoU on the camera cells (same 400 frames, tolerance 0 m; dividers 0.5 m)", fontsize=10)
ax.set_title("Maps: our chain vs the state-of-the-art map model on identical frames (cold VAL, front camera field of view)", fontsize=12.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=1)
ax.set_ylim(0, 0.95)
fig.tight_layout()
fig.savefig(OUT_PNG)

lines = ["| layer | " + " | ".join(label for label, _, _ in series) + " | best ours / SOA |", "|" + "---|" * (len(series) + 2)]
for j, (name, _, _) in enumerate(LAYERS):
    ours = [s[1][j] for s in series[1:-1]]
    soa = series[-1][1][j]
    lines.append(f"| {name.replace(chr(10), ' ')} | " + " | ".join(f"{s[1][j]:.3f}" for s in series) + f" | {max(ours) / soa:.2f} |")
lines.append("")
lines.append("frames: " + ", ".join(f"{k} {d['n_frames']} (step {d['step']})" for k, (_, d) in data.items()))
open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
