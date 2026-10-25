"""Bericht E67 (Schleife 5): Ausfall der Wahrnehmung mitten in der Fahrt - Registrierung der Karte halten (zeitliche Verfolgung + Monitor).  Liest out/loop2/e67_s1 (+ e67_drift, e67_s2) (hold_eval.py); Text loop2/loop5_text_e67.py (TEXT67).  Aufruf (gcar): make_e67.py [figs|report]"""
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
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e67.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e67.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT67", {})

LAB = {"dark": "night", "cam_front3_off": "3 forward cameras off", "blur": "blur", "cam_front_off": "front camera off", "fog": "fog", "radar_off": "radar nearly off", "extr3": "extrinsics 3 deg"}
PL = {"asgiven": "map route as given", "perframe": "per-frame registration", "pfdrop@0.5": "per-frame registration, dropped when p > 0.5 (E64 policy)", "track": "tracked, every frame", "gmon@0.35": "tracked + monitor gate p > 0.35", "gmon@0.5": "tracked + monitor gate p > 0.5", "gmon@0.9": "tracked + monitor gate p > 0.9",
      "gscore": "tracked + registration-score gate", "goracle": "tracked + oracle gate", "gt": "true registration (floor)"}
COL = {"asgiven": "#9467bd", "perframe": "#1f77b4", "pfdrop@0.5": "#17becf", "track": "#ff7f0e", "gmon@0.5": "#d62728", "gscore": "#8c564b", "goracle": "#2ca02c", "gt": "#7f7f7f"}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def M(name="e67_s1"):
    return jl(f"{OUT}/{name}/metrics.json")


TB = [(0.0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 1e9)]
TB_LAB = ["0-1.5 s after the failure began", "1.5-3 s", "3-5 s", "5 s or more"]
_TCACHE = {}


def time_bins(name="e67_s1"):
    """mittlerer Querfehler je Zeit seit Beginn des Ausfalls (Sekunden, aus den Zeitstempeln der Bilder): {Bedingung: {n: [...], pol: {pol: [...]}}}"""
    if name in _TCACHE:
        return _TCACHE[name]
    ts = np.load(os.path.expanduser("~") + "/exp_gcar_b123/poses_val.npz")["ts"]
    m = M(name)
    out = {}
    for c in m["conds"]:
        f = f"{OUT}/{name}/frames_{c}.pkl"
        if not os.path.exists(f):
            continue
        S = pickle.load(open(f, "rb"))
        acc = {p: [[] for _ in TB] for p in m["pols"]}
        for sc in S:
            t0 = ts[sc["recs"][sc["f0"]]["row"]]
            for r in sc["recs"]:
                if r["fail_phase"] and r["lata"] is not None:
                    dt = ts[r["row"]] - t0
                    for b, (lo, hi) in enumerate(TB):
                        if lo <= dt < hi:
                            for p in m["pols"]:
                                acc[p][b].append(float(np.mean(r["lata"][p])))
        out[c] = dict(n=[len(acc[m["pols"][0]][b]) for b in range(len(TB))], pol={p: [float(np.mean(v)) if v else float("nan") for v in acc[p]] for p in m["pols"]},
                      sums={p: [float(np.sum(v)) for v in acc[p]] for p in m["pols"]})
    _TCACHE[name] = out
    return out


def tables():
    m = M()
    if not m:
        return []
    conds = list(m["conds"].keys())
    P1 = ["asgiven", "perframe", "pfdrop@0.5", "track", "gmon@0.5", "gscore", "goracle", "gt"]
    h1 = ["condition after the failure", "frames (share true failures)"] + [PL[p] for p in P1]
    r1 = []
    for c in conds:
        f = m["conds"][c]["fail"]
        r1.append([LAB.get(c, c), f"{f['n']} ({f['true_fail_share']:.2f})"] + [f3(f["pols"][p]["lata"], 2) for p in P1])
    r1.append(["mean over the conditions", "-"] + [f3(m["mean"]["fail"][p]["lata"], 2) for p in P1])
    P2 = ["asgiven", "perframe", "pfdrop@0.5", "track", "gmon@0.5", "gscore", "goracle"]
    h2 = ["condition after the failure"] + [f"{PL[p]}: offset m / yaw deg" for p in P2]
    r2 = []
    for c in conds:
        f = m["conds"][c]["fail"]["pols"]
        r2.append([LAB.get(c, c)] + [f"{f[p]['trans']:.2f} / {f[p]['yaw']:.2f}" for p in P2])
    r2.append(["mean over the conditions"] + [f"{m['mean']['fail'][p]['trans']:.2f} / {m['mean']['fail'][p]['yaw']:.2f}" for p in P2])
    h3 = ["condition", "lateral error BEFORE the failure (clean frames): per-frame", "tracked", "tracked + monitor gate", "tracked + score gate", "clean frames rejected: monitor gate", "score gate", "failure-phase frames rejected: monitor gate", "score gate", "true failures in the failure phase"]
    r3 = []
    for c in conds:
        cl, fl = m["conds"][c]["clean"], m["conds"][c]["fail"]
        r3.append([LAB.get(c, c), f3(cl["pols"]["perframe"]["lata"], 2), f3(cl["pols"]["track"]["lata"], 2), f3(cl["pols"]["gmon@0.5"]["lata"], 2), f3(cl["pols"]["gscore"]["lata"], 2), f3(cl["reject_share"]["gmon@0.5"], 2), f3(cl["reject_share"]["gscore"], 2), f3(fl["reject_share"]["gmon@0.5"], 2), f3(fl["reject_share"]["gscore"], 2), f3(fl["true_fail_share"], 2)])
    # Zeit seit dem Ausfall, ueber alle Bedingungen (Bild-gewichtet)
    P4 = ["asgiven", "perframe", "pfdrop@0.5", "track", "gmon@0.5", "gscore", "goracle"]
    h4 = ["method"] + TB_LAB
    tb = time_bins()
    r4 = []
    for p in P4:
        row = [PL[p]]
        for b in range(4):
            num = sum(tb[c]["sums"][p][b] for c in tb)
            den = sum(tb[c]["n"][b] for c in tb)
            row.append(f3(num / max(den, 1), 2))
        r4.append(row)
    r4.append(["frames in the bin (all conditions)"] + [str(sum(tb[c]["n"][b] for c in tb)) for b in range(4)])
    # Varianten
    h5 = ["method (mean over the conditions, failure phase)", "map bias 1 m (main)", "map bias 1 m + drift of 0.1 m and 0.1 deg per frame", "map bias 2 m", "map bias 1 m, monitor fitted WITHOUT the failure type it judges (leave-one-out)", "map bias 1 m, monitor fitted on other scenes (scene-disjoint)"]
    r5 = []
    mm = [M("e67_s1"), M("e67_drift"), M("e67_s2"), M("e67_loco"), M("e67_scv")]
    for p in ("asgiven", "perframe", "pfdrop@0.5", "track", "gmon@0.35", "gmon@0.5", "gmon@0.9", "gscore", "goracle", "gt"):
        r5.append([PL[p]] + [f3(x["mean"]["fail"][p]["lata"], 2) if (x and "mean" in x) else "-" for x in mm])
    # Bootstrap
    keys = [("gmon@0.5-track", "monitor gate - tracked"), ("gmon@0.5-perframe", "monitor gate - per-frame"), ("gmon@0.5-asgiven", "monitor gate - map as given"), ("gmon@0.5-goracle", "monitor gate - oracle gate"), ("gscore-track", "score gate - tracked")]
    h6 = ["condition"] + [f"{lab}: lateral error difference (m) [95 % CI over scenes]" for _, lab in keys]
    r6 = []
    for c in conds:
        b = m["conds"][c]["boot"]
        r6.append([LAB.get(c, c) + f" ({m['conds'][c]['n_scenes']} scenes)"] + [f"{b[k][0]:+.2f} [{b[k][1]:+.2f}, {b[k][2]:+.2f}]" for k, _ in keys])
    return (h1, r1), (h2, r2), (h3, r3), (h4, r4), (h5, r5), (h6, r6)


def timeline(name="e67_trans"):
    """mittlerer Querfehler je Methode ueber der Zeit relativ zum Beginn des Ausfalls (1-s-Faecher, -8 ... +20 s): {Bedingung: {pol: Mittelwerte}}, plus Dauer des Ausfalls"""
    ts = np.load(os.path.expanduser("~") + "/exp_gcar_b123/poses_val.npz")["ts"]
    m = M(name)
    out = {}
    edges = np.arange(-8.0, 21.0, 1.0)
    for c in m["conds"]:
        f = f"{OUT}/{name}/frames_{c}.pkl"
        if not os.path.exists(f):
            continue
        S = pickle.load(open(f, "rb"))
        acc = {p: [[] for _ in edges[:-1]] for p in m["pols"]}
        dur = []
        for sc in S:
            recs = sc["recs"]
            t0 = ts[recs[sc["f0"]]["row"]]
            rr = [r for r in recs if r.get("recovery")]
            if rr:
                dur.append(ts[rr[0]["row"]] - t0)
            for r in recs:
                if r["lata"] is None:
                    continue
                dt = ts[r["row"]] - t0
                b = int(np.floor(dt - edges[0]))
                if 0 <= b < len(edges) - 1:
                    for p in m["pols"]:
                        acc[p][b].append(float(np.mean(r["lata"][p])))
        out[c] = dict(pol={p: [float(np.mean(v)) if len(v) >= 3 else float("nan") for v in acc[p]] for p in m["pols"]}, centers=list((edges[:-1] + 0.5).tolist()), dur=float(np.mean(dur)) if dur else float("nan"))
    return out


def timeline_fig():
    m = M("e67_trans")
    if not m or "recovery" not in m.get("mean", {}):
        return None
    tl = timeline()
    sel = [c for c in ("dark", "blur", "cam_front3_off") if c in tl]
    fig, axs = plt.subplots(1, len(sel), figsize=(4.8 * len(sel), 3.7), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, sel):
        x = tl[c]["centers"]
        for p in ("asgiven", "perframe", "track", "gmon@0.5", "goracle"):
            ax.plot(x, tl[c]["pol"][p], "-", color=COL[p], label=PL[p], lw=1.6)
        ax.axvspan(0.0, tl[c]["dur"], color="#cccccc", alpha=0.45, lw=0)
        ax.text(0.2, ax.get_ylim()[1] * 0.93, "sensors failing", fontsize=7)
        ax.set_title(LAB.get(c, c), fontsize=9)
        ax.set_xlabel("seconds relative to the beginning of the failure")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("mean lateral error (m)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    p = f"{FIG}/e67_timeline.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def trans_tables():
    m = M("e67_trans")
    if not m or "recovery" not in m.get("mean", {}):
        return None
    P_ = ["asgiven", "perframe", "pfdrop@0.5", "track", "gmon@0.5", "gscore", "goracle", "gt"]
    h = ["method", "mean of the 7 conditions: before the failure", "during the failure (about 7 s)", "after it ended (sensors back)", "night: before", "during", "after", "blur: before", "during", "after"]
    rows = []
    for p in P_:
        row = [PL[p]]
        for ph in ("clean", "fail", "recovery"):
            row.append(f3(m["mean"][ph][p]["lata"], 2))
        for c in ("dark", "blur"):
            for ph in ("clean", "fail", "recovery"):
                row.append(f3(m["conds"][c][ph]["pols"][p]["lata"], 2) if c in m["conds"] and ph in m["conds"][c] else "-")
        rows.append(row)
    return h, rows


def seed_table():
    ms = [M("e67_s1")] + [M(f"e67_seed{k}") for k in (1, 2, 3, 4)]
    ms = [m for m in ms if m and "mean" in m]
    if len(ms) < 3:
        return None
    pols = ["asgiven", "perframe", "pfdrop@0.5", "track", "gmon@0.35", "gmon@0.5", "gmon@0.9", "gscore", "goracle", "gt"]
    h = ["method (mean over the seven conditions, failure phase)", "lateral error (m): mean over the seeds", "standard deviation over the seeds", "min - max", "night: mean +- SD"]
    rows = []
    for p in pols:
        v = np.array([m["mean"]["fail"][p]["lata"] for m in ms])
        n_ = np.array([m["conds"]["dark"]["fail"]["pols"][p]["lata"] for m in ms])
        rows.append([PL[p], f3(v.mean(), 3), f3(v.std(ddof=1), 3), f"{v.min():.3f} - {v.max():.3f}", f"{n_.mean():.2f} +- {n_.std(ddof=1):.2f}"])
    return h, rows, len(ms)


def bar_fig():
    m = M()
    if not m:
        return None
    conds = list(m["conds"].keys())
    ps = ["asgiven", "perframe", "pfdrop@0.5", "track", "gmon@0.5", "goracle", "gt"]
    fig, ax = plt.subplots(figsize=(12, 3.9))
    x = np.arange(len(conds) + 1)
    w = 0.115
    for k, p in enumerate(ps):
        vals = [m["conds"][c]["fail"]["pols"][p]["lata"] for c in conds] + [m["mean"]["fail"][p]["lata"]]
        ax.bar(x + (k - 3.0) * w, vals, w, label=PL[p], color=COL[p])
    ax.set_xticks(x)
    ax.set_xticklabels([LAB.get(c, c) for c in conds] + ["mean"], fontsize=8)
    ax.set_ylabel("mean lateral error after 1 / 2 / 3 s (m)")
    ax.set_title("E67: planner-facing route error AFTER the sensor failure (second half of every TEST scene; map pose bias 1 m)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.85)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    p = f"{FIG}/e67_bars.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def time_fig():
    m = M()
    if not m:
        return None
    sel = [c for c in ("dark", "cam_front3_off", "blur") if c in m["conds"]]
    fig, axs = plt.subplots(1, len(sel), figsize=(4.2 * len(sel), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    xl = ["0-1.5", "1.5-3", "3-5", ">= 5"]
    tb = time_bins()
    for ax, c in zip(axs, sel):
        for p in ("asgiven", "perframe", "pfdrop@0.5", "track", "gmon@0.5", "goracle"):
            ax.plot(range(4), tb[c]["pol"][p], "o-", color=COL[p], label=PL[p], lw=1.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(xl)
        ax.set_xlabel("seconds since the failure began")
        ax.set_title(LAB.get(c, c), fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("mean lateral error (m)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    p = f"{FIG}/e67_time.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def trace_fig():
    m = M()
    if not m:
        return None
    sel = [c for c in ("dark", "cam_front3_off", "blur") if c in m["conds"] and "trace" in m["conds"][c]]
    if not sel:
        return None
    fig, axs = plt.subplots(1, len(sel), figsize=(4.6 * len(sel), 3.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, sel):
        t = m["conds"][c]["trace"]
        for p in ("asgiven", "perframe", "track", "gmon@0.5", "goracle"):
            ax.plot(t["f"], t["err"][p], "-", color=COL[p], label=PL[p], lw=1.6)
        ax.axvline(t["f0"] - 0.5, color="k", ls="--", lw=1)
        ax.text(t["f0"], ax.get_ylim()[1] * 0.92, " failure begins", fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(t["f"], t["pf"], ":", color="#555555", lw=1.2)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("monitor failure probability (dotted)", fontsize=7)
        ax.set_title(f"{LAB.get(c, c)} (scene {t['scene']})", fontsize=9)
        ax.set_xlabel("frame in the scene")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("registration offset error (m)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    p = f"{FIG}/e67_trace.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def bev_fig(cond="dark", name="e67_viz", offsets=(1, 12)):
    """BEV-Bild einer Szene kurz nach dem Beginn des Ausfalls: extrahierter (gestoerter) Graph, Karten-Route wie gegeben / Einzelbild / nachgefuehrt / Monitor-Halten / wahre Ausrichtung, gefahrener Weg."""
    import pickle as pk

    import assoc_temporal as AT
    import hold_eval as HE

    m = M(name)
    if not m or cond not in m["conds"] or "trace" not in m["conds"][cond] or not os.path.exists(f"{OUT}/{name}/frames_{cond}.pkl"):
        return None
    S = pk.load(open(f"{OUT}/{name}/frames_{cond}.pkl", "rb"))
    # typische Szene: mittlerer Unterschied (Median) zwischen der Nachfuehrung ohne und mit Monitor-Tor im Versatz der Ausfallphase
    cand = []
    for x in S:
        fr = [r for r in x["recs"] if r["fail_phase"]]
        if len(fr) >= 8:
            cand.append((float(np.mean([r["err"]["track"][0] - r["err"]["gmon@0.5"][0] for r in fr])), x))
    cand.sort(key=lambda t: t[0])
    scene = cand[len(cand) // 2][1]
    sc = scene["scene"]
    f0, recs = scene["f0"], scene["recs"]
    D = os.path.expanduser("~") + "/exp_gcar_b123"
    P = dict(np.load(f"{D}/poses_val.npz"))
    GT = pk.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    RT = pk.load(open(f"{D}/route_gt_val.pkl", "rb"))
    g = pk.load(open(f"{OUT}/e57_{cond}/graphs_test.pkl", "rb"))
    exg = {int(i): x for i, x in zip(g["ids"], g["graphs"])}
    fut = HE.build_future(P)
    fig, axs = plt.subplots(1, len(offsets), figsize=(5.2 * len(offsets), 6.0))
    axs = np.atleast_1d(axs)
    sty = {"asgiven": ("#9467bd", "-", "map route as given"), "perframe": ("#1f77b4", "-", "per-frame registration"), "track": ("#ff7f0e", "-", "tracked, every frame"), "gmon@0.5": ("#d62728", "-", "tracked + monitor gate (hold)"), "gt": ("#2ca02c", "--", "true registration")}
    for ax, off in zip(axs, offsets):
        idx = min(f0 + off, len(recs) - 1)
        r = recs[idx]
        row = r["row"]
        n = r["noise"]
        pieces = AT.noisy_pieces(GT[row], P["t"][row], P["R"][row], np.asarray(n["c0"]), n["psi"], np.asarray(n["b"]), np.asarray(n["jit"]))
        for q in pieces:
            q = np.asarray(q)
            ax.plot(q[:, 0], q[:, 1], "-", color="#dddddd", lw=0.8, zorder=1)
        for q in exg[row]["pts"]:
            q = np.asarray(q)
            ax.plot(q[:, 0], q[:, 1], "-", color="k", lw=1.6, zorder=3, label="extracted graph (failing sensors)" if q is exg[row]["pts"][0] else None)
        rt = RT[row]
        for pol, (col, ls, lab) in sty.items():
            R, t = np.asarray(r["reg"][pol][0]), np.asarray(r["reg"][pol][1])
            first = True
            for pidx in rt["route"]:
                q = np.asarray(pieces[pidx])
                if len(q) < 2:
                    continue
                q = (q - t) @ R
                ax.plot(q[:, 0], q[:, 1], ls, color=col, lw=2.0, zorder=4, label=lab if first else None)
                first = False
        if fut[row] is not None:
            fp = np.vstack([[0.0, 0.0], fut[row]])
            ax.plot(fp[:, 0], fp[:, 1], "o:", color="#555555", ms=3, lw=1.2, zorder=5, label="driven path (3 s)")
        ax.plot([0], [0], "^", color="k", ms=9, zorder=6)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-8, 60)
        ax.set_aspect("equal")
        ax.set_title(f"{LAB.get(cond, cond)}, scene {sc}: {off * 0.5:.1f} s after the failure began\nregistration offset error: given {r['err']['asgiven'][0]:.1f} m, per-frame {r['err']['perframe'][0]:.1f} m, tracked {r['err']['track'][0]:.1f} m, hold {r['err']['gmon@0.5'][0]:.1f} m", fontsize=7.5)
        ax.grid(alpha=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    p = f"{FIG}/e67_bev.png"
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
    for fn, cap in ((bar_fig, "Planner-facing lateral error of the route after the failure began, per condition and method."),
                    (time_fig, "Lateral error as a function of the time since the failure began (night, three forward cameras off, blur): the ungated tracker is contaminated within a few frames, the monitor-gated hold stays close to the oracle gate."),
                    (trace_fig, "Registration offset error per frame in the scene where tracking and the gated hold differ most (solid) and the failure probability of the combined monitor (dotted)."),
                    (bev_fig, "Bird's-eye view of a typical scene (median difference between the tracker with and without gate) at night, 0.5 s and 6 s after the failure began (x right, y forward, ego at the origin): the extracted graph (black) is nearly empty or wrong, the grey lanes are the biased map, the coloured lines are the route of the map placed by each method; the dotted line is the path the vehicle then drove. In the dark the per-frame registration (blue) follows the failing graph, the tracked one is pulled along, the held one stays on the true registration.")):
        try:
            p = fn()
            if p:
                images.append((p, cap))
        except Exception as ex:  # noqa: BLE001
            print(fn.__name__, "Fehler:", ex)
    try:
        p = timeline_fig()
        if p:
            images.append((p, "Transient failure: lateral error over time around a failure of about 7 s (grey; night, blur, three forward cameras off). The tracker without a gate is polluted by the failure and takes time to recover, the monitor-gated hold is nearly unaffected."))
    except Exception as ex:  # noqa: BLE001
        print("timeline_fig Fehler:", ex)
    t = TEXT
    tt = trans_tables()
    caps = ["Failure phase (second half of each scene): anchored lateral error (m) of the map route given to the planner, mean over 1 / 2 / 3 s - the map has a 1 m pose bias",
            "Failure phase: registration error of the map (offset at the vehicle in m / yaw in degrees)",
            "Cost of the gates: lateral error on the clean frames before the failure, share of frames the gate keeps out of the tracker",
            "Lateral error by the time since the failure began (all seven conditions pooled, frame-weighted; the failure lasts up to 10 s)",
            "Robustness of the result to the map pose model (mean over the conditions, failure phase)",
            "Paired scene bootstrap (2,000 draws) of the lateral error difference in the failure phase"]
    tabs = [(cap, tb[0], tb[1]) for cap, tb in zip(caps, tabs_)]
    if tt:
        tabs.append(("Transient failure (the middle third of every scene, about 7 s; clean sensors again afterwards): lateral error (m) before, during and after the failure - does the registration recover?", tt[0], tt[1]))
    st = seed_table()
    if st:
        tabs.append((f"Seed variability: the realisation of the map pose error (offset, yaw, jitter) was drawn {st[2]} times (seed 0 = main run); the 21 scenes and the graphs are the same", st[0], st[1]))
    make_report2("E67", t.get("title", "E67"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.hold_arch, arch_height=3.0)
    print("E67 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
