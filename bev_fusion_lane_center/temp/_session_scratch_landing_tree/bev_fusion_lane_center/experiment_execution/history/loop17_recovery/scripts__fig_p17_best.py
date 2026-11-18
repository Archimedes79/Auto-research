"""Loop 17 report figure of the best chains: failure-half composite S and route success for the three main failure types, from results_master.json.
Left: champion single frame, + spatial hold, + learned gate, the map-conditioned builder trained on real failure frames (B) with hold and with gate (exact map, 2 m error, no map, wrong map), and the no-map control (C).
Right: route success under failure. Usage (gcar): fig_p17_best.py"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E17 = "/mnt/c/ML-Experiment/out/loop2/e17"
FIG = "/mnt/c/ML-Experiment/report_final/figures"
MJ = json.load(open(f"{E17}/results_master.json"))
P3 = ["dark", "blur", "cam_front3_off"]
NAME = dict(dark="night", blur="motion blur", cam_front3_off="3 front cameras off")
fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.4), dpi=150)
ax = axs[0]
sys_ = [("norec_single", "champion, single frame", "#c85f00"), ("norec_hold", "champion + spatial hold", "#1f6fb4"), ("gate", "champion + learned gate", "#6a3d9a"),
        ("m139c_hold_mnone", "C: builder trained on real failures, no map, + hold", "#999999"), ("m139b_hold_m0", "B: + SD map (exact), + hold", "#4caf50"), ("gateB_m0", "B + learned gate, exact map", "#1a7f3c"),
        ("gateB_m2", "B + gate, map error 2 m", "#d9a400"), ("gateB_mwrong", "B + gate, wrong map", "#c0392b")]
n = len(sys_)
w = 0.8 / n
for i, (k, lab, col) in enumerate(sys_):
    vals = [MJ["perts"][p]["cards"].get(k, {}).get("fail", {}).get("S", np.nan) for p in P3]
    ax.bar(np.arange(3) + (i - (n - 1) / 2) * w, vals, w, color=col, label=lab)
    for j, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(j + (i - (n - 1) / 2) * w, v + 0.004, f"{v:.2f}", ha="center", fontsize=6.4, rotation=90)
clean = MJ["perts"]["dark"]["cards"]["norec_single"]["cl"]["S"]
ax.axhline(clean, color="k", ls="--", lw=1)
ax.text(2.45, clean + 0.004, f"clean half: {clean:.2f}", ha="right", fontsize=8)
ax.set_xticks(range(3))
ax.set_xticklabels([NAME[p] for p in P3], fontsize=9)
ax.set_ylim(0.38, 0.8)
ax.set_ylabel("composite S, failure half")
ax.set_title("Graph quality under failure", fontsize=10.5, loc="left")
ax.legend(fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
# route success
ax = axs[1]


def rs(tag, route="s0", sig="0", ph="fail"):
    c = MJ["route"].get(tag)
    if not c:
        return np.nan
    xs = [v["route_success"] for k, v in c.items() if k.startswith(f"sigma{sig}_") and k.endswith(f"_{ph}")]
    return float(np.mean(xs)) if xs else np.nan


rsys = [("e135_route/{p}_single", "champion, single frame", "#c85f00"), ("e135_route/{p}_maxdecay_0.95", "champion + spatial hold", "#1f6fb4"), ("e135_route/{p}_gate", "champion + learned gate", "#6a3d9a"),
        ("e139_route/{p}_single_C", "C: real-failure training, no map, single", "#bbbbbb"), ("e139_route/{p}_maxdecay_0.95_C", "C + hold", "#999999"),
        ("e139_route/{p}_single_B", "B: with SD map, single", "#8bc34a"), ("e139_route/{p}_maxdecay_0.95_B", "B + hold", "#4caf50"), ("e140_route/{p}_gateB_B", "B + learned gate", "#1a7f3c")]
n = len(rsys)
w = 0.84 / n
for i, (pat, lab, col) in enumerate(rsys):
    vals = [rs(pat.format(p=p)) for p in P3]
    ax.bar(np.arange(3) + (i - (n - 1) / 2) * w, vals, w, color=col, label=lab)
    for j, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(j + (i - (n - 1) / 2) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.4, rotation=90)
cl = rs("e135_route/dark_single", ph="cl")
ax.axhline(cl, color="k", ls="--", lw=1)
ax.text(2.45, cl + 0.01, f"clean half: {cl:.2f}", ha="right", fontsize=8)
ax.set_xticks(range(3))
ax.set_xticklabels([NAME[p] for p in P3], fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_ylabel("route success, failure half (exact map, two seeds)")
ax.set_title("Navigation attribute under failure", fontsize=10.5, loc="left")
ax.legend(fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.tight_layout()
fig.savefig(f"{FIG}/p17_best.png")
print("saved p17_best.png")
