"""Bericht E68 (Schleife 5): Ausfall ueberbruecken - den zuletzt guten Spurgraphen per Eigenbewegung weitergeben.  Liest out/loop2/e68_main (+ e68_odo) (coast_eval.py); Text loop2/loop5_text_e68.py (TEXT68).  Aufruf (gcar): make_e68.py [figs|report]"""
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
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e68.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e68.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT68", {})

LAB = {"dark": "night", "cam_front3_off": "3 forward cameras off", "blur": "blur", "cam_front_off": "front camera off", "fog": "fog", "radar_off": "radar nearly off", "extr3": "extrinsics 3 deg"}
PL = {"perceived": "perceived graph (failing sensors)", "coast_mon": "carried-forward graph (monitor picks the source)", "coast_oracle": "carried-forward graph (oracle picks the source)", "switch_mon": "switch: perceived while p <= 0.5, else carried forward",
      "switch_oracle": "switch with oracle failure label", "merge_mon": "union of perceived and carried-forward when p > 0.5"}
COL = {"perceived": "#1f77b4", "coast_mon": "#ff7f0e", "coast_oracle": "#2ca02c", "switch_mon": "#d62728", "switch_oracle": "#8c564b", "merge_mon": "#9467bd"}
AGE_LAB = ["< 1 s", "1-2 s", "2-4 s", "4-7 s", ">= 7 s"]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def M(name="e68_main"):
    return jl(f"{OUT}/{name}/metrics.json")


def tables():
    m = M()
    if not m:
        return []
    conds = list(m["conds"].keys())
    P1 = ["perceived", "coast_mon", "coast_oracle", "switch_mon", "switch_oracle", "merge_mon"]
    h1 = ["condition after the failure", "frames (share true failures)"] + [PL[p] for p in P1]
    r1, r2 = [], []
    for c in conds:
        f = m["conds"][c]
        r1.append([LAB.get(c, c), f"{f['n_fail']} ({f['true_fail_share']:.2f})"] + [f3(f["fail"][p]["F1"], 3) for p in P1])
        r2.append([LAB.get(c, c), f"{f['n_fail']} ({f['true_fail_share']:.2f})"] + [f3(f["fail"][p]["F1_fwd"], 3) for p in P1])
    r1.append(["mean over the conditions", "-"] + [f3(m["mean"]["fail"][p], 3) for p in P1])
    r2.append(["mean over the conditions", "-"] + [f3(m["mean"]["fail_fwd"][p], 3) for p in P1])
    h3 = ["condition", "perceived: precision / recall", "switch (monitor): precision / recall", "union (monitor): precision / recall", "clean frames before the failure: perceived F1", "switch (monitor) F1 (false alarms of the monitor)"]
    r3 = []
    for c in conds:
        f = m["conds"][c]
        r3.append([LAB.get(c, c)] + [f"{f['fail'][p]['P']:.2f} / {f['fail'][p]['R']:.2f}" for p in ("perceived", "switch_mon", "merge_mon")] + [f3(f["clean"]["perceived"]["F1"], 3), f3(f["clean"]["switch_mon"]["F1"], 3)])
    # nach Alter (alle Bedingungen gepoolt)
    h4 = ["age of the source frame", "frames flagged by the monitor", "perceived graph (same frames): F1 / in front", "carried forward (monitor picks the source): F1 / in front", "carried forward (oracle picks the source): F1 / in front"]
    r4 = []
    age = {}
    for name in ("age_perceived", "age_coast_mon", "age_coast_oracle"):
        rows = []
        for b in range(5):
            cc = np.zeros(8)
            nn = 0
            for c in conds:
                x = m["conds"][c][name][b]
                if x["n"]:
                    cc += np.array(x["counts"])
                    nn += x["n"]
            rows.append((nn, cc))
        age[name] = rows

    def f1c(cc):
        out = []
        for sl in (slice(0, 4), slice(4, 8)):
            tp_p, n_p, tp_g, n_g = cc[sl]
            P_, R_ = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
            out.append(2 * P_ * R_ / max(P_ + R_, 1e-9))
        return out

    for b in range(5):
        row = [AGE_LAB[b], f"{age['age_coast_mon'][b][0]} / {age['age_coast_oracle'][b][0]} (oracle)"]
        for name in ("age_perceived", "age_coast_mon", "age_coast_oracle"):
            nn, cc = age[name][b]
            row.append("-" if nn == 0 else "{:.3f} / {:.3f}".format(*f1c(cc)))
        r4.append(row)
    mo = M("e68_odo")
    ms = M("e68_scv")
    h5 = ["policy (mean over the conditions)", "failure phase: exact ego motion", "failure phase: ego motion with error (1 deg / sqrt(s) yaw, 1 % of the distance)", "failure phase, in front of the vehicle: exact", "with error", "exact ego motion, monitor fitted on other scenes (scene-disjoint)"]
    r5 = [[PL[p], f3(m["mean"]["fail"][p], 3), f3(mo["mean"]["fail"][p], 3) if mo else "-", f3(m["mean"]["fail_fwd"][p], 3), f3(mo["mean"]["fail_fwd"][p], 3) if mo else "-", f3(ms["mean"]["fail"][p], 3) if (ms and "mean" in ms) else "-"] for p in P1]
    return (h1, r1), (h1, r2), (h3, r3), (h4, r4), (h5, r5)


def bar_fig():
    m = M()
    if not m:
        return None
    conds = list(m["conds"].keys())
    ps = ["perceived", "coast_mon", "switch_mon", "switch_oracle", "merge_mon"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
    x = np.arange(len(conds) + 1)
    w = 0.16
    for ax, key, mk, tt in ((axs[0], "F1", "fail", "whole BEV"), (axs[1], "F1_fwd", "fail_fwd", "in front of the vehicle (y > 0)")):
        for k, p in enumerate(ps):
            vals = [m["conds"][c]["fail"][p][key] for c in conds] + [m["mean"][mk][p]]
            ax.bar(x + (k - 2) * w, vals, w, label=PL[p], color=COL[p])
        ax.set_xticks(x)
        ax.set_xticklabels([LAB.get(c, c).replace(" ", "\n") for c in conds] + ["mean"], fontsize=7)
        ax.set_title(f"point F1 @1 m after the failure began - {tt}", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel("point F1 @1 m")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    p = f"{FIG}/e68_bars.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def age_fig():
    m = M()
    if not m:
        return None
    conds = list(m["conds"].keys())
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, sl, tt in ((axs[0], slice(0, 4), "whole BEV"), (axs[1], slice(4, 8), "in front of the vehicle")):
        for name, lab, col in (("age_perceived", "perceived graph (same frames)", COL["perceived"]), ("age_coast_mon", "carried forward (monitor)", COL["coast_mon"]), ("age_coast_oracle", "carried forward (oracle)", COL["coast_oracle"])):
            ys = []
            for b in range(5):
                cc = np.zeros(8)
                for c in conds:
                    x = m["conds"][c][name][b]
                    if x["n"]:
                        cc += np.array(x["counts"])
                tp_p, n_p, tp_g, n_g = cc[sl]
                P_, R_ = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
                ys.append(2 * P_ * R_ / max(P_ + R_, 1e-9) if n_p + n_g > 0 else np.nan)
            ax.plot(range(5), ys, "o-", color=col, label=lab, lw=2)
        ax.set_xticks(range(5))
        ax.set_xticklabels(AGE_LAB)
        ax.set_xlabel("age of the source frame")
        ax.set_title(tt, fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("point F1 @1 m")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    p = f"{FIG}/e68_age.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def bev_fig(cond="dark", offsets=(3, 10, 18)):
    import assoc_eval as AE
    import coast_eval as CE

    m = M()
    if not m or cond not in m["conds"] or not os.path.exists(f"{OUT}/e68_main/frames_{cond}.pkl"):
        return None
    S = pickle.load(open(f"{OUT}/e68_main/frames_{cond}.pkl", "rb"))
    # Szene mit der groessten Verbesserung (gepoolter Treffer-Zaehler: Recall)
    best, bv = None, -1
    Pz = dict(np.load(f"{AE.D}/poses_val.npz"))
    for s in S:
        fr = [r for r in s["recs"] if r["fail_phase"] and r["pf"] > 0.5]
        ii = min(s["f0"] + max(offsets), len(s["recs"]) - 1)
        moved = float(np.linalg.norm(Pz["t"][s["recs"][ii]["row"]] - Pz["t"][s["recs"][s["f0"]]["row"]]))
        if len(fr) >= 15 and moved >= 60.0:  # das Fahrzeug faehrt (Abstand zwischen Beginn des Ausfalls und dem letzten gezeigten Bild)
            v = float(np.mean([(r["counts"]["coast_mon"][2] / max(r["counts"]["coast_mon"][3], 1)) - (r["counts"]["perceived"][2] / max(r["counts"]["perceived"][3], 1)) for r in fr]))
            if v > bv:
                best, bv = s, v
    if best is None:
        return None
    D = AE.D
    P = dict(np.load(f"{D}/poses_val.npz"))
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    g = pickle.load(open(f"{OUT}/e57_{cond}/graphs_test.pkl", "rb"))
    gc = pickle.load(open(f"{OUT}/e57_stored/graphs_test.pkl", "rb"))
    pert = {int(i): x for i, x in zip(g["ids"], g["graphs"])}
    clean = {int(i): x for i, x in zip(gc["ids"], gc["graphs"])}
    recs, f0 = best["recs"], best["f0"]
    fig, axs = plt.subplots(1, len(offsets), figsize=(4.6 * len(offsets), 5.6))
    axs = np.atleast_1d(axs)
    for ax, off in zip(axs, offsets):
        i = min(f0 + off, len(recs) - 1)
        r = recs[i]
        row = r["row"]
        src = r["src"]["mon"]
        r0 = recs[src]["row"]
        gsrc = pert[r0] if recs[src]["fail_phase"] else clean[r0]
        coast = CE.warp_pieces(gsrc["pts"], P["R"][r0], P["t"][r0], P["R"][row], P["t"][row])
        for q in GT[row]["pts"]:
            q = np.asarray(q)
            ax.plot(q[:, 0], q[:, 1], "-", color="#2ca02c", lw=1.4, alpha=0.55, zorder=1, label="ground truth" if q is GT[row]["pts"][0] else None)
        for k, q in enumerate(coast):
            ax.plot(q[:, 0], q[:, 1], "-", color="#ff7f0e", lw=1.8, zorder=3, label="carried forward from the last good frame" if k == 0 else None)
        for k, q in enumerate(pert[row]["pts"]):
            q = np.asarray(q)
            ax.plot(q[:, 0], q[:, 1], "-", color="k", lw=1.8, zorder=4, label="perceived (failing sensors)" if k == 0 else None)
        ax.plot([0], [0], "^", color="k", ms=9, zorder=6)
        ax.set_xlim(-48, 48)
        ax.set_ylim(-48, 48)
        ax.set_aspect("equal")
        def _f(c):
            P_, R_ = c[0] / max(c[1], 1), c[2] / max(c[3], 1)
            return 2 * P_ * R_ / max(P_ + R_, 1e-9)

        ax.set_title(f"{LAB.get(cond, cond)}: source frame {r['age']['mon']:.1f} s old; F1 perceived {_f(r['counts']['perceived']):.2f}, carried forward {_f(r['counts']['coast_mon']):.2f}", fontsize=8)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    p = f"{FIG}/e68_bev.png"
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
    for fn, cap in ((bar_fig, "Point F1 @1 m of the lane graph after the sensor failure began, per condition and policy (left: whole BEV, right: only the part in front of the vehicle)."),
                    (age_fig, "F1 of the carried-forward graph as a function of the age of its source frame (frames flagged by the monitor, all conditions pooled): the perceived graph of the same frames is the horizontal reference."),
                    (bev_fig, "Night, one scene, 1.5 s / 5 s / 9 s after the failure began (x right, y forward, ego at the origin): ground truth (green), the nearly empty perceived graph (black) and the graph carried forward from the last good frame (orange). The carried-forward graph keeps the lanes behind and beside the vehicle; the lanes ahead of it end where the old frame ended.")):
        try:
            p = fn()
            if p:
                images.append((p, cap))
        except Exception as ex:  # noqa: BLE001
            print(fn.__name__, "Fehler:", ex)
    t = TEXT
    caps = ["Failure phase: point F1 @1 m of the lane graph in the whole BEV (±48 m)",
            "Failure phase: point F1 @1 m in front of the vehicle only (y > 0), the part a planner needs",
            "Precision and recall (failure phase, whole BEV) and the cost of the monitor's false alarms before the failure",
            "F1 by the age of the source frame (monitor-flagged frames of all conditions, pooled counts)",
            "Ego-motion error: the same policies when the odometry has an error (yaw 1 deg per sqrt(s) and 1 % of the travelled distance)"]
    tabs = [(cap, tb[0], tb[1]) for cap, tb in zip(caps, tabs_)]
    make_report2("E68", t.get("title", "E68"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.coast_arch, arch_height=3.0)
    print("E68 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
