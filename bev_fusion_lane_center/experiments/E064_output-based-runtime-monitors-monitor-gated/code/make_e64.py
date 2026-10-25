"""Bericht E64 (Schleife 5): Laufzeit-Monitore und Rueckfall auf die Karten-Route.  Liest out/loop2/e64_monitors/{signals.pkl, metrics.json} (monitor_eval.py); Text loop2/loop5_text_e64.py (TEXT64).  Aufruf (gcar): make_e64.py [figs|report]"""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs5 as AF5  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
D = f"{OUT}/e64_monitors"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e64.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e64.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT64", {})

LAB = {"stored": "clean", "dark": "night", "cam_front_off": "front camera off", "cam_front3_off": "3 forward cameras off", "fog": "fog", "blur": "blur", "radar_off": "radar nearly off", "extr3": "extrinsics 3 deg"}
SIGLAB = {"dis": "ensemble disagreement", "mass": "heat mass (low)", "unassigned": "points without map partner", "regres": "registration residual", "routegap": "map route not covered", "npieces": "few pieces"}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def tables():
    m = jl(f"{D}/metrics.json")
    if not m:
        return [], [], []
    names = list(m["names"].keys())
    h1 = ["condition", "mean graph F1 @1 m", "failure share (F1 < 0.6)"] + [f"AUROC {SIGLAB[k]}" for k in names]
    r1 = []
    for c, v in m["per_cond"].items():
        r1.append([LAB.get(c, c), f3(v["mean_f1"]), f3(v["fail_share"])] + [f3(v["auroc"][k], 2) for k in names])
    h2 = ["signal", "AUROC: failure (pooled over conditions)", "AUROC: perturbed vs clean frame", "weight in the combined monitor"]
    r2 = [[SIGLAB[k], f3(m["pooled"].get(k), 3), f3(m["detect_perturbed"].get(k), 3), f3(m.get("combined_weights", {}).get(k), 2)] for k in names]
    r2.append(["COMBINED (logistic regression, fitted on even samples, evaluated on odd samples)", f3(m.get("combined_auroc_heldout"), 3), "-", "-"])
    r2.append(["(for the train-vs-test diagnosis) the same regression on the even samples it was fitted on", f3(m.get("combined_auroc_fit"), 3), "-", "-"])
    sc = jl(f"{D}/scenecv.json")
    if sc:
        r2.append(["COMBINED, scene-disjoint cross-fit (fitted on one half of the scenes, judged on the other; all conditions)", f3(sc["pooled_scene"], 3), "-", "-"])
    g = m.get("gating", {})
    h3 = ["condition (map pose noise 1 m)", "n", "perception route (SYSTEM)", "map route as given", "registered map route", "switch SYSTEM -> map route when p > 0.2 (share map)", "registered map route unless p > 0.5, then map route as given (share not registered)", "same, p > 0.9", "oracle: better of registered / given map route"]
    r3 = []
    for c, v in g.items():
        r3.append([LAB.get(c, c), v["n"], f3(v["system"], 2), f3(v["maponly"], 2), f3(v["regmap"], 2), f"{f3(v['gated@0.2'], 2)} ({f3(v['gated@0.2_share_map'], 2)})", f"{f3(v['regmap_gated@0.5'], 2)} ({f3(v['regmap_gated@0.5_share_noreg'], 2)})", f"{f3(v['regmap_gated@0.9'], 2)} ({f3(v['regmap_gated@0.9_share_noreg'], 2)})", f3(v["oracle_regmap_or_map"], 2)])
    gp = m.get("gating_pooled")
    if gp:
        r3.append(["mean over the conditions", "-", f3(gp["system"], 2), f3(gp["maponly"], 2), f3(gp["regmap"], 2), f"{f3(gp['gated@0.2'], 2)} ({f3(gp['gated@0.2_share_map'], 2)})", f"{f3(gp['regmap_gated@0.5'], 2)} ({f3(gp['regmap_gated@0.5_share_noreg'], 2)})", f"{f3(gp['regmap_gated@0.9'], 2)} ({f3(gp['regmap_gated@0.9_share_noreg'], 2)})", f3(gp["oracle_regmap_or_map"], 2)])
    return (h1, r1), (h2, r2), (h3, r3)


def loco_table():
    m = jl(f"{D}/loco.json")
    if not m:
        return None
    h = ["held-out failure type (monitor fitted WITHOUT it)", "failure share", "AUROC, frames of this type only", "AUROC, pooled with the clean frames", "standard monitor (type seen, parity fit): frames of this type only", "pooled with the clean frames", "best single signal"]
    rows = []
    for c, v in m.items():
        rows.append([LAB.get(c, c), f3(v["fail_share"], 2), f3(v["auroc_loco"], 3), f3(v["auroc_loco_vs_clean"], 3), f3(v["auroc_parity"], 3), f3(v["auroc_parity_vs_clean"], 3), f3(v["best_single"], 3)])
    return h, rows


def roc_fig():
    m = jl(f"{D}/metrics.json")
    if not m or "roc" not in m:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    cols = {"dis": "#1f77b4", "mass": "#ff7f0e", "unassigned": "#2ca02c", "regres": "#9467bd", "routegap": "#8c564b", "npieces": "#7f7f7f", "combined": "#d62728"}
    for k, r in m["roc"].items():
        axs[0].plot(r["fpr"], r["tpr"], "-", color=cols.get(k, "k"), lw=2.4 if k == "combined" else 1.3, label=SIGLAB.get(k, "COMBINED"))
    axs[0].plot([0, 1], [0, 1], ":", color="k")
    axs[0].set_xlabel("false alarm rate (frames that are fine, flagged)")
    axs[0].set_ylabel("detection rate (failed frames, flagged)")
    axs[0].set_title(f"ROC of the monitors (failure = graph F1 < {m['thr']}; held-out half, n = {m['heldout_n']})", fontsize=9)
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    S = pickle.load(open(f"{D}/signals.pkl", "rb"))
    conds = list(S.keys())
    axs[1].boxplot([S[c]["dis"] for c in conds], showfliers=False)
    axs[1].set_xticklabels([LAB.get(c, c).replace(" ", "\n") for c in conds])
    axs[1].set_ylabel("ensemble disagreement (1 - mean IoU of 4 heads)")
    axs[1].set_title("The ensemble disagrees more as the input degrades", fontsize=9)
    axs[1].tick_params(axis="x", labelsize=6.5)
    axs[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = f"{FIG}/e64_roc.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def gate_fig():
    m = jl(f"{D}/metrics.json")
    g = (m or {}).get("gating")
    if not g:
        return None
    conds = list(g.keys())
    fig, ax = plt.subplots(figsize=(12, 3.9))
    x = np.arange(len(conds))
    w = 0.16
    for k, (key, lab, col) in enumerate((("system", "perception route (SYSTEM)", "#1f77b4"), ("maponly", "map route as given", "#9467bd"), ("regmap", "registered map route", "#ff7f0e"), ("regmap_gated@0.5", "registered map route, monitor-gated (p > 0.5)", "#d62728"), ("oracle_regmap_or_map", "oracle: better of registered / given", "#2ca02c"))):
        ax.bar(x + (k - 2) * w, [g[c][key] for c in conds], w, label=lab, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([LAB.get(c, c) for c in conds], fontsize=8)
    ax.set_ylabel("mean lateral error after 1 / 2 / 3 s (m)")
    ax.set_ylim(0, 1.85)
    ax.set_title("E64: route given to the planner under perception failures - fixed policies against the monitor-gated registration (TEST, held-out half; map with 1 m pose noise)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    p = f"{FIG}/e64_gating.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def main(mode):
    t1, t2, t3 = tables()
    if mode == "figs":
        for t in (t1, t2, t3):
            if t:
                for r in t[1]:
                    print(r)
        return
    images = []
    for fn, cap in ((roc_fig, "Left: ROC curves of the single signals and of the combined monitor for 'the graph of this frame is bad' (held-out half of all frames of all conditions). Right: the disagreement of four independently trained dense heads per condition."), (gate_fig, "Planner-facing route error per condition for fixed policies and for the monitor-gated switch between the perception route and the map route.")):
        try:
            p = fn()
            if p:
                images.append((p, cap))
        except Exception as ex:  # noqa: BLE001
            print(fn.__name__, "Fehler:", ex)
    t = TEXT
    tabs = []
    lt = loco_table()
    if t1:
        tabs.append(("Monitor signals per condition: AUROC for 'this frame fails' (graph F1 @1 m of the frame < 0.6)", t1[0], t1[1]))
    if t2:
        tabs.append(("Pooled over all conditions", t2[0], t2[1]))
    if t3 and t3[1]:
        tabs.append(("Gated route hand-over: mean lateral error (m, anchored) on the held-out half", t3[0], t3[1]))
    if lt:
        tabs.append(("Out-of-distribution check: the combined monitor fitted without one failure type and judged on it (AUROC for 'graph F1 of the frame < 0.6')", lt[0], lt[1]))
    make_report2("E64", t.get("title", "E64"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.monitor_arch, arch_height=3.0)
    print("E64 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
