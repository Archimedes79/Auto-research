"""Bericht E61 (Schleife 5): zeitliche Verfolgung der Kartenausrichtung und der Spur-Zuordnung.  Liest out/loop2/e61_s{1,2,3}/metrics.json (assoc_temporal.py); Text loop2/loop5_text_e61.py (TEXT61).  Aufruf (gcar): make_e61.py [figs|report]"""
import json
import os
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
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e61.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e61.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT61", {})

METHODS = [
    ("M0", "single frame: graph-ICP on every frame"),
    ("M1", "filtered alignment: median of the last 5 per-frame estimates"),
    ("M3", "tracked registration: small search window around the alignment predicted from the last frames"),
    ("M2", "M0 + vote of the warped previous tokens (WASABI-style hysteresis)"),
    ("M12", "M1 + vote"),
    ("M32", "M3 + vote"),
    ("GT", "ground-truth assignment (reference: flicker caused by the geometry itself)"),
]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rows_for(s):
    d = jl(f"{OUT}/e61_s{s}/metrics.json")
    rows = []
    if not d:
        return rows
    for m, lab in METHODS:
        r = d[m]
        rows.append([lab, f3(r["acc"]), f3(r["flicker"]), f3(r["success"]), f3(r.get("successG")) if "successG" in r else "-", f3(r.get("flickerG")) if "flickerG" in r else "-", f"{r['reg_err_median']:.2f} / {r['reg_err_mean']:.2f}" if "reg_err_median" in r else "-"])
    return rows


def curves():
    sig = [1, 2, 3]
    D = {s: jl(f"{OUT}/e61_s{s}/metrics.json") for s in sig}
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    cols = {"M0": "#d62728", "M1": "#ff7f0e", "M3": "#1f77b4", "M2": "#e377c2", "M12": "#bcbd22", "M32": "#17becf"}
    for m, c in cols.items():
        axs[0].plot(sig, [D[s][m]["acc"] if D[s] else np.nan for s in sig], "-o", color=c, label=m)
        axs[1].plot(sig, [D[s][m]["flicker"] if D[s] else np.nan for s in sig], "-o", color=c, label=m)
        if "successG" in (D[1] or {}).get(m, {}):
            axs[2].plot(sig, [D[s][m]["successG"] if D[s] else np.nan for s in sig], "-o", color=c, label=m)
    axs[0].set_title("lane-ID assignment accuracy (TEST scenes)", fontsize=9)
    axs[1].set_title("flicker: share of lane IDs that change between frames", fontsize=9)
    axs[2].set_title("route success, geometric route flag (cov, prec >= 0.8)", fontsize=9)
    for ax in axs:
        ax.set_xlabel("scene-constant pose bias sigma (m)")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = f"{FIG}/e61_curves.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def trace_fig():
    d = jl(f"{OUT}/e61_s3/metrics.json")
    if not d or "trace" not in d:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    for m, c in (("M0", "#d62728"), ("M1", "#ff7f0e"), ("M3", "#1f77b4")):
        ax.plot(d["trace"][m], "-", color=c, label={"M0": "M0 single frame", "M1": "M1 filtered (median of 5)", "M3": "M3 tracked"}[m], lw=1.6)
    ax.set_xlabel("frame of the TEST scene with the largest difference between M0 and M3 (2 Hz)")
    ax.set_ylabel("registration position error (m)")
    ax.set_title("E61: registration error over one TEST scene at 3 m scene-constant pose bias", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = f"{FIG}/e61_trace.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def main(mode):
    head = ["method", "assignment accuracy", "flicker (token)", "route success (token flag)", "route success (geometric flag)", "flicker (geometric route flag)", "registration error median / mean (m)"]
    if mode == "figs":
        for s in (1, 2, 3):
            for r in rows_for(s):
                print(s, r)
        return
    images = []
    p = curves()
    images.append((p, "Assignment accuracy, flicker and route success against the scene-constant pose bias (TEST, 21 scenes): the tracked registration (M3) does not depend on the size of the bias; the single-frame estimate does."))
    p = trace_fig()
    if p:
        images.append((p, "Registration error per frame in the TEST scene with the largest difference between the single-frame and the tracked registration (3 m scene-constant bias): the single-frame ICP finds the wrong local optimum for long stretches, the tracked estimate stays at the level of the map jitter."))
    t = TEXT
    tabs = [(f"Scene-constant pose bias of sigma = {s} m (+ 0.2 m / 0.3 deg per-frame jitter), TEST scenes", head, rows_for(s)) for s in (1, 2, 3)]
    make_report2("E61", t.get("title", "E61"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.temporal_arch5, arch_height=3.0)
    print("E61 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
