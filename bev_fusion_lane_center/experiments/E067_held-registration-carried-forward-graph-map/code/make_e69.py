"""Bericht E69 (Schleife 5): Karten-Spuren als Spurgraph, wenn die Wahrnehmung ausfaellt (Ebene 3).  Liest out/loop2/e69_maplanes/metrics.json (maplane_eval.py), e68_main (E68-Vergleich); Text loop2/loop5_text_e69.py (TEXT69).  Aufruf (gcar): make_e69.py [figs|report]"""
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
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e69.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e69.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT69", {})

LAB = {"dark": "night", "cam_front3_off": "3 forward cameras off", "blur": "blur", "cam_front_off": "front camera off", "fog": "fog", "radar_off": "radar nearly off", "extr3": "extrinsics 3 deg"}
MP = {"asgiven": "map lanes as given (no registration)", "perframe": "map lanes, per-frame registration", "track": "map lanes, tracked registration (every frame)", "gmon@0.5": "map lanes, held registration (monitor gate)", "goracle": "map lanes, oracle gate", "gt": "map lanes, TRUE registration (ceiling: 1 by construction)"}
EDGES = np.arange(0.0, 11.0, 1.0)


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f1c(cc, fwd=False):
    cc = np.asarray(cc, float)
    tp_p, n_p, tp_g, n_g = cc[4:8] if fwd else cc[0:4]
    P_, R_ = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
    return 2 * P_ * R_ / max(P_ + R_, 1e-9)


def e68_bins(conds):
    """E68 (Monitor-Umschaltung bzw. wahrgenommen) Treffer-Zaehler je Zeit seit Beginn des Ausfalls, gepoolt ueber 'conds': {key: [8-Zaehler je Fach]}"""
    ts = np.load(os.path.expanduser("~") + "/exp_gcar_b123/poses_val.npz")["ts"]
    acc = {k: [np.zeros(8) for _ in EDGES[:-1]] for k in ("perceived", "switch_mon")}
    for c in conds:
        f = f"{OUT}/e68_main/frames_{c}.pkl"
        if not os.path.exists(f):
            continue
        for sc in pickle.load(open(f, "rb")):
            t0 = ts[sc["recs"][sc["f0"]]["row"]]
            for r in sc["recs"]:
                if not r["fail_phase"]:
                    continue
                b = int(np.floor(ts[r["row"]] - t0))
                if 0 <= b < len(EDGES) - 1:
                    for k in acc:
                        acc[k][b] += np.array(r["counts"][k], float)
    return acc


def tables():
    m = jl(f"{OUT}/e69_maplanes/metrics.json")
    m68 = jl(f"{OUT}/e68_main/metrics.json")
    if not m or not m68:
        return []
    conds = [c for c in LAB if c in m["conds"]]
    pols = ["asgiven", "perframe", "track", "gmon@0.5", "goracle", "gt"]
    h1 = ["condition after the failure", "perceived graph (E68)", "last good graph carried forward (E68, monitor switch)"] + [MP[p] for p in pols]
    r1, r2 = [], []
    sums = {k: np.zeros(8) for k in ["perceived", "switch_mon"] + pols}
    for c in conds:
        row1, row2 = [LAB[c]], [LAB[c]]
        for k in ("perceived", "switch_mon"):
            row1.append(f3(m68["conds"][c]["fail"][k]["F1"], 3))
            row2.append(f3(m68["conds"][c]["fail"][k]["F1_fwd"], 3))
        for p in pols:
            v = m["conds"][c][f"current|{p}|1.0"]
            row1.append(f3(v["F1"], 3))
            row2.append(f3(v["F1_fwd"], 3))
        r1.append(row1)
        r2.append(row2)
    mean1, mean2 = ["mean over the conditions"], ["mean over the conditions"]
    for k in ("perceived", "switch_mon"):
        mean1.append(f3(np.mean([m68["conds"][c]["fail"][k]["F1"] for c in conds]), 3))
        mean2.append(f3(np.mean([m68["conds"][c]["fail"][k]["F1_fwd"] for c in conds]), 3))
    for p in pols:
        mean1.append(f3(np.mean([m["conds"][c][f"current|{p}|1.0"]["F1"] for c in conds]), 3))
        mean2.append(f3(np.mean([m["conds"][c][f"current|{p}|1.0"]["F1_fwd"] for c in conds]), 3))
    r1.append(mean1)
    r2.append(mean2)
    # nach Zeit seit Ausfallbeginn, gepoolt ueber alle Bedingungen
    b68 = e68_bins(conds)
    h3 = ["seconds since the failure began", "perceived graph: whole / in front", "carried forward (E68): whole / in front", "map lanes, held registration: whole / in front", "map lanes, per-frame registration: whole / in front", "map lanes as given: whole / in front"]
    r3 = []
    for i in range(len(EDGES) - 1):
        def pool(key):
            return np.sum([np.array(m["conds"][c][key]["bins"][i], float) for c in conds], axis=0)

        row = [f"{int(EDGES[i])}-{int(EDGES[i + 1])} s"]
        for cc in (b68["perceived"][i], b68["switch_mon"][i], pool("current|gmon@0.5|1.0"), pool("current|perframe|1.0"), pool("current|asgiven|1.0")):
            row.append(f"{f1c(cc):.3f} / {f1c(cc, True):.3f}" if np.sum(cc) > 0 else "-")
        r3.append(row)
    # veraltete Karten und Toleranz
    h4 = ["map used as the lane graph", "held registration, @1 m tolerance: whole / in front", "held registration, @1.5 m: whole / in front", "true registration, @1 m: whole / in front (what staleness alone costs)"]
    r4 = []
    for var, lab in (("current", "current map"), ("removed30", "30 % of the lanes removed (outdated)"), ("shifted30", "30 % of the lanes shifted by 1.5 m (re-painted)")):
        row = [lab]
        for key in (f"{var}|gmon@0.5|1.0", f"{var}|gmon@0.5|1.5", f"{var}|gt|1.0"):
            cc = np.sum([np.array(m["conds"][c][key]["tot"], float) for c in conds if key in m["conds"][c]], axis=0)
            row.append(f"{f1c(cc):.3f} / {f1c(cc, True):.3f}")
        r4.append(row)
    return (h1, r1), (h1, r2), (h3, r3), (h4, r4)


def stale_table():
    m = jl(f"{OUT}/e69_maplanes/stale_signal.json")
    if not m:
        return None
    h = ["outdated map variant (against the current map, same frames)", "signal: share of the carried-forward graph within 1 m of a map lane - median, current map", "median, outdated map", "10 % quantile current / 90 % quantile outdated", "AUROC 'outdated vs current', single frame", "AUROC, mean of 5 consecutive frames (2.5 s)"]
    rows = []
    for var, lab in (("removed30", "30 % of the lanes removed"), ("shifted30", "30 % of the lanes shifted by 1.5 m")):
        v = m[var]
        rows.append([lab, f3(v["median_current"], 3), f3(v["median_stale"], 3), f"{v['q10_current']:.3f} / {v['q90_stale']:.3f}", f3(v["auroc_frame"], 3), f"{v['auroc_5frames']:.3f} ({v['n_windows']} windows)"])
    return h, rows, m["n"]


def policy_tables():
    m = jl(f"{OUT}/e69_maplanes/policy.json")
    if not m:
        return None
    vars_ = [("current", "current map"), ("removed30", "30 % of the lanes removed"), ("shifted30", "30 % of the lanes shifted by 1.5 m"), ("wrong", "WRONG map (the map of another place)")]
    rows_def = [("perceived", "perceived graph only"), ("e68_switch", "last good graph carried forward when the monitor flags (E68)"), ("map_switch", "map lanes (held registration) whenever the monitor flags"),
                ("rule", "rule A: map lanes if the carried-forward graph agrees with them (no signal for sources older than 3 s: trust the map)"), ("rule_valid", "rule B: map lanes if the map-validity signal is high, else the carried-forward graph"), ("oracle", "oracle: best of the three sources per frame")]
    h = ["source of the lane geometry (failure phase, all seven conditions pooled)"] + [f"{lab}: F1 whole / in front" for _, lab in vars_]
    rows = [[lab] + [f"{m[v][k]['F1']:.3f} / {m[v][k]['F1_fwd']:.3f}" for v, _ in vars_] for k, lab in rows_def]
    rows.append(["share of the flagged frames in which rule A / rule B chose the map lanes"] + [f"{m[v]['share_map_chosen']:.2f} / {m[v]['share_map_chosen_valid']:.2f}" for v, _ in vars_])
    h2 = ["map used", "median map validity, current map (same frames)", "median map validity, this map", "AUROC 'not the current map' from the validity", "frames"]
    r2 = [[lab, f3(m[v]["median_validity"][0], 3), f3(m[v]["median_validity"][1], 3), f3(m[v]["auroc_validity"], 3), str(m[v]["n_validity"])] for v, lab in vars_[1:]]
    return (h, rows), (h2, r2, m["tau_valid"], m["n_clean"])


def bars():
    m = jl(f"{OUT}/e69_maplanes/metrics.json")
    m68 = jl(f"{OUT}/e68_main/metrics.json")
    if not m or not m68:
        return None
    conds = [c for c in LAB if c in m["conds"]]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
    x = np.arange(len(conds) + 1)
    w = 0.15
    cols = ["#1f77b4", "#d62728", "#9467bd", "#17becf", "#ff7f0e", "#2ca02c"]
    names = ["perceived graph", "carried forward (E68)", "map lanes as given", "map lanes, per-frame registration", "map lanes, held registration", "map lanes, true registration"]
    for ax, key, mk in ((axs[0], "F1", "whole BEV"), (axs[1], "F1_fwd", "in front of the vehicle")):
        series = [[m68["conds"][c]["fail"]["perceived"][key] for c in conds], [m68["conds"][c]["fail"]["switch_mon"][key] for c in conds],
                  [m["conds"][c]["current|asgiven|1.0"][key] for c in conds], [m["conds"][c]["current|perframe|1.0"][key] for c in conds],
                  [m["conds"][c]["current|gmon@0.5|1.0"][key] for c in conds], [m["conds"][c]["current|gt|1.0"][key] for c in conds]]
        for k, (sv, nm, col) in enumerate(zip(series, names, cols)):
            ax.bar(x + (k - 2.5) * w, sv + [float(np.mean(sv))], w, color=col, label=nm)
        ax.set_xticks(x)
        ax.set_xticklabels([LAB[c].replace(" ", "\n") for c in conds] + ["mean"], fontsize=7)
        ax.set_title(f"point F1 @1 m after the failure began - {mk}", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel("point F1 @1 m")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    p = f"{FIG}/e69_bars.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def time_fig():
    m = jl(f"{OUT}/e69_maplanes/metrics.json")
    if not m:
        return None
    sel = [("dark", "night"), ("blur", "blur"), (None, "all seven conditions")]
    fig, axs = plt.subplots(2, 3, figsize=(14, 6.4), sharex=True, sharey=True)
    x = EDGES[:-1] + 0.5
    for j, (c, nm) in enumerate(sel):
        cs = [c] if c else [k for k in LAB if k in m["conds"]]
        b68 = e68_bins(cs)
        for i, (fwd, tt) in enumerate(((False, "whole BEV"), (True, "in front of the vehicle"))):
            ax = axs[i, j]

            def series(cc_list):
                return [f1c(cc, fwd) if np.sum(cc) > 0 else np.nan for cc in cc_list]

            ax.plot(x, series(b68["perceived"]), "-o", color="#1f77b4", label="perceived graph")
            ax.plot(x, series(b68["switch_mon"]), "-o", color="#d62728", label="last good graph carried forward (E68)")
            for key, lab, col in (("current|gmon@0.5|1.0", "map lanes, held registration", "#ff7f0e"), ("current|perframe|1.0", "map lanes, per-frame registration", "#17becf"), ("current|asgiven|1.0", "map lanes as given", "#9467bd")):
                cc_list = [np.sum([np.array(m["conds"][k][key]["bins"][b], float) for k in cs], axis=0) for b in range(len(x))]
                ax.plot(x, series(cc_list), "-s", color=col, label=lab)
            ax.set_title(f"{nm} - {tt}", fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("seconds since the failure began")
            if j == 0:
                ax.set_ylabel("point F1 @1 m")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    p = f"{FIG}/e69_time.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def bev_fig(cond="dark", offsets=(3, 10, 18)):
    import assoc_eval as AE
    import assoc_temporal as AT
    import coast_eval as CE

    m68 = jl(f"{OUT}/e68_main/metrics.json")
    if not m68 or not os.path.exists(f"{OUT}/e67_s1/frames_{cond}.pkl"):
        return None
    D = AE.D
    P = dict(np.load(f"{D}/poses_val.npz"))
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    S68 = pickle.load(open(f"{OUT}/e68_main/frames_{cond}.pkl", "rb"))
    S67 = {s["scene"]: s for s in pickle.load(open(f"{OUT}/e67_s1/frames_{cond}.pkl", "rb"))}
    g = pickle.load(open(f"{OUT}/e57_{cond}/graphs_test.pkl", "rb"))
    gc = pickle.load(open(f"{OUT}/e57_stored/graphs_test.pkl", "rb"))
    pert = {int(i): x for i, x in zip(g["ids"], g["graphs"])}
    clean = {int(i): x for i, x in zip(gc["ids"], gc["graphs"])}
    # dieselbe (fahrende) Szene wie in E68
    best, bv = None, -1
    for s in S68:
        fr = [r for r in s["recs"] if r["fail_phase"] and r["pf"] > 0.5]
        ii = min(s["f0"] + max(offsets), len(s["recs"]) - 1)
        moved = float(np.linalg.norm(P["t"][s["recs"][ii]["row"]] - P["t"][s["recs"][s["f0"]]["row"]]))
        if len(fr) >= 15 and moved >= 60.0:
            v = float(np.mean([(r["counts"]["coast_mon"][2] / max(r["counts"]["coast_mon"][3], 1)) - (r["counts"]["perceived"][2] / max(r["counts"]["perceived"][3], 1)) for r in fr]))
            if v > bv:
                best, bv = s, v
    if best is None or best["scene"] not in S67:
        return None
    recs, f0 = best["recs"], best["f0"]
    recs67 = S67[best["scene"]]["recs"]
    fig, axs = plt.subplots(1, len(offsets), figsize=(4.6 * len(offsets), 5.6))
    axs = np.atleast_1d(axs)
    for ax, off in zip(axs, offsets):
        i = min(f0 + off, len(recs) - 1)
        r, r67 = recs[i], recs67[i]
        row = r["row"]
        n = r67["noise"]
        src = r["src"]["mon"]
        r0 = recs[src]["row"]
        gsrc = pert[r0] if recs[src]["fail_phase"] else clean[r0]
        coast = CE.warp_pieces(gsrc["pts"], P["R"][r0], P["t"][r0], P["R"][row], P["t"][row])
        pieces = AT.noisy_pieces(GT[row], P["t"][row], P["R"][row], np.asarray(n["c0"]), n["psi"], np.asarray(n["b"]), np.asarray(n["jit"]))
        Rh, th = np.asarray(r67["reg"]["gmon@0.5"][0]), np.asarray(r67["reg"]["gmon@0.5"][1])
        for k, q in enumerate(GT[row]["pts"]):
            q = np.asarray(q)
            ax.plot(q[:, 0], q[:, 1], "-", color="#2ca02c", lw=1.6, alpha=0.55, zorder=1, label="ground truth lane graph" if k == 0 else None)
        for k, q in enumerate(pieces):
            ax.plot(q[:, 0], q[:, 1], ":", color="#999999", lw=1.0, zorder=2, label="map lanes as given (biased)" if k == 0 else None)
        for k, q in enumerate(pieces):
            w = (q - th) @ Rh
            ax.plot(w[:, 0], w[:, 1], "-", color="#1f77b4", lw=1.4, zorder=3, label="map lanes, held registration" if k == 0 else None)
        for k, q in enumerate(coast):
            ax.plot(q[:, 0], q[:, 1], "-", color="#ff7f0e", lw=1.8, zorder=4, label="carried forward (E68)" if k == 0 else None)
        ax.plot([0], [0], "^", color="k", ms=9, zorder=6)
        ax.set_xlim(-48, 48)
        ax.set_ylim(-48, 48)
        ax.set_aspect("equal")
        ax.set_title(f"night, scene {best['scene']}: {off * 0.5:.1f} s after the failure began (source frame {r['age']['mon']:.1f} s old)", fontsize=8)
        ax.grid(alpha=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    p = f"{FIG}/e69_bev.png"
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
    for fn, cap in ((bars, "Point F1 @1 m of the lane graph in the failure phase: perceived graph, carried-forward graph (E68) and the map lanes placed by the registration of E67 (whole BEV left, in front of the vehicle right). The 'true registration' bar is 1 by construction (the ground truth is the map)."),
                    (time_fig, "Point F1 @1 m against the seconds since the failure began: the map lanes under the held registration are better than the carried-forward graph at every moment (0.89 against 0.78 in the first second) and age only slowly (their level is set by the registration offset of about 1 m against the 1 m tolerance), while the carried-forward graph decays, faster in front of the vehicle."),
                    (bev_fig, "Night, a moving vehicle, 1.5 s / 5 s / 9 s after the failure began: ground truth (green), the map lanes as given (grey dotted, biased), the same lanes placed by the held registration (blue) and the graph carried forward from the last good frame (orange).")):
        try:
            p = fn()
            if p:
                images.append((p, cap))
        except Exception as ex:  # noqa: BLE001
            print(fn.__name__, "Fehler:", ex)
    t = TEXT
    caps = ["Failure phase: point F1 @1 m of the lane graph in the whole BEV (+-48 m). Map lanes = the biased map transformed into the ego frame with the registration of E67; the ground truth IS the map, so the true-registration column is the ceiling (1.0)",
            "Failure phase: point F1 @1 m in front of the vehicle only (y > 0)",
            "By the time since the failure began (all seven conditions pooled counts): perceived graph, carried-forward graph (E68) and map lanes under three registrations",
            "Outdated map and tolerance (held registration; failure phase, all conditions pooled): what an outdated map and a looser tolerance do"]
    tabs = [(cap, tb[0], tb[1]) for cap, tb in zip(caps, tabs_)]
    st = stale_table()
    if st:
        tabs.append((f"Detecting an outdated map: the carried-forward graph (source frame at most 3 s old; {st[2]} failure-phase frames of the seven conditions) against the map lanes placed by the held registration", st[0], st[1]))
    pt = policy_tables()
    if pt:
        tabs.append(("Choosing the source of the lane geometry during a failure: fixed sources and two selection rules (point F1 @1 m, failure phase, all seven conditions pooled; 'wrong map' = the map of another place, shown in this frame)", pt[0][0], pt[0][1]))
        tabs.append((f"The map-validity signal: agreement (share of the points within 1 m of a map lane) between the PERCEIVED graph of the last 10 healthy frames (monitor p <= 0.5) and the map lanes placed by the registration; rule B's threshold is the 5 % quantile of this signal on the clean frames of the current map ({pt[1][2]:.3f}; {pt[1][3]} clean frames, no labels)", pt[1][0], pt[1][1]))
    make_report2("E69", t.get("title", "E69"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.maplane_arch, arch_height=3.0)
    print("E69 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
