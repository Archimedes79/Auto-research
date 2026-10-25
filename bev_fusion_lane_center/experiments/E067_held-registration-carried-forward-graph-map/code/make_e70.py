"""Bericht E70 (Schleife 5): Sprung der Lokalisierung - Nachfuehrung der Registrierung ohne / mit Neustart.  Liest out/loop2/e70_j<J>/metrics.json (jump_eval.py); Text loop2/loop5_text_e70.py (TEXT70).  Aufruf (gcar): make_e70.py [figs|report]"""
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
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e70.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e70.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT70", {})
RUNS = [("e70_j1.5", "jump 1.5 m"), ("e70_j3.0", "jump 3 m"), ("e70_j6.0", "jump 6 m"), ("e70_j10.0", "jump 10 m (beyond the +-8 m restart search)"), ("e70_j3y4", "jump 3 m + 4 deg yaw")]
PL = {"asgiven": "map route as given", "perframe": "per-frame registration (full search +-3 m)", "track": "tracked registration (E61), no restart", "reinit": "tracked + score-triggered restart", "gt": "true registration (floor)"}
PS = {"asgiven": "map as given", "perframe": "per-frame (+-3 m)", "track": "tracked, no restart", "reinit": "tracked + restart", "gt": "true (floor)"}
JS = {"e70_j1.5": "1.5 m", "e70_j3.0": "3 m", "e70_j6.0": "6 m", "e70_j10.0": "10 m", "e70_j3y4": "3 m + 4 deg"}
COL = {"asgiven": "#9467bd", "perframe": "#1f77b4", "track": "#ff7f0e", "reinit": "#d62728", "gt": "#7f7f7f"}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def tables():
    runs = [(k, lab, jl(f"{OUT}/{k}/metrics.json")) for k, lab in RUNS]
    runs = [(k, lab, m) for k, lab, m in runs if m]
    if not runs:
        return []
    labs = runs[0][2]["bin_labels"]
    h1 = ["jump", "method"] + [f"lateral error (m): {l}" for l in labs]
    r1, r2 = [], []
    for k, lab, m in runs:
        for p in m["pols"]:
            r1.append([JS[k], PS[p]] + [f3(x, 2) for x in m["lata"][p]])
            r2.append([JS[k], PS[p]] + [f3(x, 2) for x in m["err"][p]])
    h2 = ["jump", "method"] + [f"registration offset error (m): {l}" for l in labs]
    h3 = ["jump", "tracked: median frames to recover (offset < 1 m) / never", "tracked + restart: median frames / never", "per-frame: median frames / never", "restart events (before the jump = false alarms)"]
    r3 = []
    for k, lab, m in runs:
        rf = m["recover_frames"]
        r3.append([JS[k]] + [f"{rf[p]['median']:.0f} / {rf[p]['never']:.2f}" for p in ("track", "reinit", "perframe")] + [f"{m['reinit_events']} ({m['reinit_before_jump']})"])
    return (h1, r1), (h2, r2), (h3, r3)


def line_fig():
    runs = [(k, lab, jl(f"{OUT}/{k}/metrics.json")) for k, lab in RUNS]
    runs = [(k, lab, m) for k, lab, m in runs if m][:4]
    if not runs:
        return None
    fig, axs = plt.subplots(1, len(runs), figsize=(4.0 * len(runs), 3.7), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (k, lab, m) in zip(axs, runs):
        xl = m["bin_labels"]
        for p in m["pols"]:
            ax.plot(range(len(xl)), m["lata"][p], "o-", color=COL[p], label=PL[p], lw=1.8)
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(xl)))
        ax.set_xticklabels([x.replace(" after", "").replace("before the jump", "before") for x in xl], fontsize=7, rotation=20)
        ax.set_title(lab, fontsize=8.5)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("mean lateral error (m)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    p = f"{FIG}/e70_lines.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def trace_fig():
    m = jl(f"{OUT}/e70_j3.0/metrics.json")
    if not m or not m.get("traces"):
        return None
    tr = list(m["traces"].items())[:3]
    fig, axs = plt.subplots(1, len(tr), figsize=(4.8 * len(tr), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (sc, t) in zip(axs, tr):
        x = np.array(t["since"]) * 0.5
        for p in ("asgiven", "perframe", "track", "reinit"):
            ax.plot(x, t["err"][p], "-", color=COL[p], label=PL[p], lw=1.5)
        ax.axvline(0.0, color="k", ls="--", lw=1)
        ax.set_title(f"scene {sc} (3 m jump at 0 s)", fontsize=9)
        ax.set_xlabel("seconds since the jump")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("registration offset error (m)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    p = f"{FIG}/e70_trace.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def main(mode):
    tabs_ = tables()
    if mode == "figs":
        for t in tabs_:
            for r in t[1]:
                print(r)
        return
    images = []
    for fn, cap in ((line_fig, "Lateral error of the registered map route before and after a jump of the map offset (clean sensors; the jump happens in the middle of every TEST scene)."),
                    (trace_fig, "Registration offset error per frame in three scenes around a 3 m jump: the tracked registration without a restart converges slowly, the score-triggered restart within a frame or two; the per-frame registration is noisy throughout.")):
        try:
            p = fn()
            if p:
                images.append((p, cap))
        except Exception as ex:  # noqa: BLE001
            print(fn.__name__, "Fehler:", ex)
    t = TEXT
    caps = ["Lateral error (m, anchored, mean over 1 / 2 / 3 s) of the registered map route by the time since the jump (0.5 s per frame; 21 TEST scenes, clean sensors); a jump of 10 m is beyond the +-8 m search of the restart",
            "Registration offset error (m) at the vehicle, by the time since the jump",
            "Recovery: frames until the offset error is below 1 m, share of scenes that never recover, and restart events"]
    tabs = [(cap, tb[0], tb[1]) for cap, tb in zip(caps, tabs_)]
    make_report2("E70", t.get("title", "E70"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.jump_arch, arch_height=3.0)
    print("E70 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
