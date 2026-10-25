"""Bericht E59 (Schleife 5): Zuordnung extrahierter Spurmitten zum Kartengraphen + Uebertragung der Route.  Liest out/loop2/e59_*/metrics.json (assoc_eval.py); Text loop2/loop5_text_e59.py (TEXT59).  Aufruf (gcar): make_e59.py [figs|report]"""
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
import assoc_eval as AE  # noqa: E402
import assoc_lib as AL  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e59.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e59.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT59", {})

CONDS = [
    ("e59_oracle", "ORACLE association (true token of every extracted point; upper bound set by the extraction)"),
    ("e59_nomap", "no map: follow the most straight-ahead successor (baseline)"),
    ("e59_s0", "map, exact pose, nearest map point"),
    ("e59_s0_hmm", "map, exact pose, Viterbi along the polyline"),
    ("e59_s0.5_noreg", "map, pose noise 0.5 m, no registration"),
    ("e59_s0.5_reg", "map, pose noise 0.5 m, graph-ICP registration"),
    ("e59_s1.0_noreg", "map, pose noise 1 m, no registration"),
    ("e59_s1.0_reg", "map, pose noise 1 m, graph-ICP registration"),
    ("e59_s1.0_reg_hmm", "map, pose noise 1 m, graph-ICP + Viterbi"),
    ("e59_s2.0_noreg", "map, pose noise 2 m, no registration"),
    ("e59_s2.0_reg", "map, pose noise 2 m, graph-ICP registration"),
    ("e59_s2.0_reg_hmm", "map, pose noise 2 m, graph-ICP + Viterbi"),
    ("e59_s3.0_noreg", "map, pose noise 3 m, no registration"),
    ("e59_s3.0_reg", "map, pose noise 3 m, graph-ICP registration"),
    ("e59_d0.1_keep", "outdated map: 10 % of the lanes missing (route lanes kept), graph-ICP"),
    ("e59_d0.3_keep", "outdated map: 30 % missing (route lanes kept), graph-ICP"),
    ("e59_d0.5_keep", "outdated map: 50 % missing (route lanes kept), graph-ICP"),
    ("e59_d0.1_any", "outdated map: 10 % missing INCLUDING route lanes, graph-ICP"),
    ("e59_d0.3_any", "outdated map: 30 % missing INCLUDING route lanes, graph-ICP"),
    ("e59_d0.5_any", "outdated map: 50 % missing INCLUDING route lanes, graph-ICP"),
    ("e59_shift30", "re-painted lanes: 30 % shifted sideways by 1.5 m, graph-ICP"),
    ("e59_s1_d02", "pose noise 1 m + 20 % lanes missing (route kept), graph-ICP"),
]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rows_table(sets=("val", "test")):
    rows = []
    for name, lab in CONDS:
        d = jl(f"{OUT}/{name}/metrics.json")
        if not d:
            continue
        for st in sets:
            m = d.get(st)
            if not m:
                continue
            rows.append([lab, st.upper(), f3(m["assoc_acc_all"]), f3(m["assigned_frac"]), f3(m["route_success"]), f"{f3(m.get('routeG_cov'))} / {f3(m.get('routeG_prec'))}", f3(m.get("routeG_success")), f"{f3(m.get('routeG_F1'))} ({f3(m.get('routeG_P'))} / {f3(m.get('routeG_R'))})",
                         (f"{m['reg_pos_median']:.2f} / {m['reg_pos_p90']:.2f} m, {m['reg_yaw_median']:.2f} deg" if "reg_pos_median" in m else "-"), f"{f3(m['novel_P'])} / {f3(m['novel_R'])}"])
    return rows


def curves():
    xs = [0.0, 0.5, 1.0, 2.0, 3.0]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for lab, key, col in (("no registration", "noreg", "#d62728"), ("graph-ICP registration", "reg", "#1f77b4")):
        ys = {m: [] for m in ("assoc_acc_all", "routeG_F1", "routeG_success", "route_success")}
        for s in xs:
            nm = "e59_s0" if s == 0.0 else f"e59_s{s}_{key}"
            d = jl(f"{OUT}/{nm}/metrics.json")
            for m in ys:
                ys[m].append(d["test"][m] if d else np.nan)
        for ax, m, t in zip(axs, ("assoc_acc_all", "routeG_F1", "routeG_success"), ("lane-ID association accuracy (TEST)", "route F1 per point, geometric flag (TEST)", "route success (cov and prec >= 0.8 at 1.5 m)")):
            ax.plot(xs, ys[m], "-o", label=lab, color=col)
            ax.set_title(t, fontsize=9)
            ax.set_xlabel("pose noise sigma (m)")
            ax.grid(alpha=0.3)
        axs[2].plot(xs, ys["route_success"], ":x", color=col, alpha=0.7, label=lab + " - flag by token of nearest lane")
    d = jl(f"{OUT}/e59_nomap/metrics.json")
    if d:
        axs[1].axhline(d["test"]["routeG_F1"], color="k", ls=":", label="no map")
        axs[2].axhline(d["test"]["routeG_success"], color="k", ls=":", label="no map")
    d = jl(f"{OUT}/e59_oracle/metrics.json")
    if d:
        axs[2].axhline(d["test"]["routeG_success"], color="g", ls=":", label="oracle association (extraction limit)")
    axs[0].legend(fontsize=7)
    axs[2].legend(fontsize=6)
    fig.tight_layout()
    p = f"{FIG}/e59_curves.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def examples(n=3, sigma=1.0):
    GT = pickle.load(open(f"{AE.D}/graph_gt_val.pkl", "rb"))
    RT = pickle.load(open(f"{AE.D}/route_gt_val.pkl", "rb"))
    d = pickle.load(open(f"{OUT}/e44_cons_p/graphs_test.pkl", "rb"))
    ids = [int(i) for i in d["ids"]]
    cand = []
    for k, i in enumerate(ids):
        rt = RT[i]
        if rt is None or len(d["graphs"][k]["pts"]) < 6 or len(rt["route"]) < 3:
            continue
        outd = {}
        for a_, b_ in GT[i]["edges"]:
            outd[a_] = outd.get(a_, 0) + 1
        ns = sum(1 for p in GT[i]["pts"] if True) * 0 + sum(1 for p in range(len(GT[i]["pts"])) if outd.get(p, 0) >= 2)
        if ns >= 1:
            cand.append((ns, len(rt["route"]), k))
    cand.sort(reverse=True)
    sel = [c[2] for c in cand[:: max(len(cand) // n, 1)]][:n]
    fig, axs = plt.subplots(n, 3, figsize=(12, 3.9 * n), squeeze=False)
    for r, k in enumerate(sel):
        i = ids[k]
        ex, gt, rt = d["graphs"][k], GT[i], RT[i]
        rng = np.random.default_rng(AE.seed_of(i, 1))
        mp_ = AL.perturb_map(gt, sigma, 0.0, protect=set(rt["route"]), rng=rng)
        route_tok = {gt["tokens"][p] for p in rt["route"]}
        res = AL.associate(ex, mp_, register_graph=True, route_tok=route_tok)
        on = np.asarray(res["point_route"], bool)
        EV, Epid = res["EV"], res["point_piece"]
        for c in range(3):
            ax = axs[r, c]
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.plot(0, 0, "k^", ms=7)
        # (1) noisy map (rot) + extracted graph before registration
        for p in mp_["pieces"]:
            axs[r, 0].plot(p[:, 0], p[:, 1], "-", color="#d62728", lw=1.0, alpha=0.8)
        for p in ex["pts"]:
            p = np.asarray(p)
            axs[r, 0].plot(p[:, 0], p[:, 1], "-", color="#1f77b4", lw=1.2)
        # (2) after registration: extracted graph transformed into the map frame
        Rm, tm = res["R"], res["t"]
        for p in mp_["pieces"]:
            axs[r, 1].plot(p[:, 0], p[:, 1], "-", color="#d62728", lw=1.0, alpha=0.8)
        for p in ex["pts"]:
            q = np.asarray(p) @ Rm.T + tm
            axs[r, 1].plot(q[:, 0], q[:, 1], "-", color="#1f77b4", lw=1.2)
        # (3) extracted graph coloured by transferred route flag, GT route in green underneath
        for p in rt["route"]:
            q = np.asarray(gt["pts"][p])
            axs[r, 2].plot(q[:, 0], q[:, 1], "-", color="#2ca02c", lw=5, alpha=0.35)
        for k2, p in enumerate(ex["pts"]):
            p = AL.resample(np.asarray(p), 1.0)
            sel_pts = np.nonzero(Epid == k2)[0]
            flag = on[sel_pts] if len(sel_pts) == len(p) else on[sel_pts]
            for a_ in range(len(p) - 1):
                col = "#2ca02c" if (len(flag) > a_ and flag[a_]) else "#8c8c8c"
                axs[r, 2].plot(p[a_ : a_ + 2, 0], p[a_ : a_ + 2, 1], "-", color=col, lw=1.6)
        axs[r, 0].set_title(f"sample {i}: map with {sigma:g} m pose noise (red) and extracted graph (blue)" if r == 0 else f"sample {i}", fontsize=8)
        axs[r, 1].set_title("after graph-ICP registration" if r == 0 else "", fontsize=8)
        axs[r, 2].set_title("transferred route flag (green = on route) over GT route (wide green)" if r == 0 else "", fontsize=8)
    fig.tight_layout()
    p = f"{FIG}/e59_examples.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def pert_table():
    """Routen-Uebertragung auf den gestoerten Graphen von E57 (TEST, Pose-Rauschen 1 m, Einzelbild-Registrierung); Bilder mit LEEREM Graphen zaehlen als Fehlschlag"""
    import pickle

    D = os.path.expanduser("~") + "/exp_gcar_b123"
    RT = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    LABELS = [("stored", "clean"), ("dark", "night"), ("cam_front3_off", "3 forward cameras off"), ("blur", "blur"), ("cam_front_off", "front camera off"), ("fog", "fog"), ("radar_off", "radar nearly off"), ("extr3", "extrinsics 3 deg")]
    rows = []
    for c, lab in LABELS:
        m = jl(f"{OUT}/e59p_{c}/metrics.json")
        e57 = jl(f"{OUT}/e57_{c}/metrics.json")
        g = pickle.load(open(f"{OUT}/e57_{c}/graphs_test.pkl", "rb"))
        n_rated = sum(1 for i in g["ids"] if RT[int(i)] is not None)
        if not m or "test" not in m:
            continue
        t = m["test"]
        n_in = t["n_samples"]
        succ_all = t["routeG_success"] * n_in / max(n_rated, 1)
        rows.append([lab, f3((e57 or {}).get("graph", {}).get("F1@1.0"), 3), f"{n_rated - n_in} of {n_rated}", f3(succ_all, 3), f3(t["routeG_success"], 3), f"{t['routeG_cov']:.3f} / {t['routeG_prec']:.3f}", f"{t.get('reg_pos_median', float('nan')):.2f}"])
    head = ["perturbation", "graph F1 @1 m (E57)", "samples with an EMPTY extracted graph", "route success, empty graphs counted as failures", "route success among the samples with a graph", "route coverage / precision @1.5 m (samples with a graph)", "registration error, median (m)"]
    return head, rows


def alt_table():
    """Alternative Navigationsziele an Verzweigungen: TEST-Samples, deren gefahrene Route eine Verzweigung durchquert (Route = anderer Zweig, groesster Richtungsunterschied, bis 40 m dahinter) gegen die gefahrene Route derselben Samples"""
    rows = []
    for s_, lab in (("0", "exact pose"), ("1.0", "pose noise 1 m, graph-ICP registration"), ("2.0", "pose noise 2 m, graph-ICP registration")):
        a, d = jl(f"{OUT}/e59alt_s{s_}/metrics.json"), jl(f"{OUT}/e59drv_s{s_}/metrics.json")
        if not a or not d:
            continue
        ta, td = a["test"], d["test"]
        rows.append([lab, ta["n_samples"], f3(td["routeG_success"], 3), f"{td['routeG_cov']:.3f} / {td['routeG_prec']:.3f}", f3(ta["routeG_success"], 3), f"{ta['routeG_cov']:.3f} / {ta['routeG_prec']:.3f}", f3(td["routeG_F1"], 3), f3(ta["routeG_F1"], 3)])
    head = ["map condition", "TEST samples (driven route passes a split)", "DRIVEN route: route success", "coverage / precision @1.5 m", "ALTERNATIVE route (other branch): route success", "coverage / precision @1.5 m", "route F1 per point: driven", "alternative"]
    return head, rows


def main(mode):
    head = ["condition", "set", "assoc. accuracy (matched points)", "assigned", "route success, flag = token of nearest lane (first reading)", "route coverage / precision @1.5 m, geometric flag", "route success, geometric tie-break flag (final)", "route F1 (P / R) per point, geometric flag", "registration error: median / p90, yaw median", "'not in map' flag P / R"]
    rows = rows_table()
    images = []
    if mode == "figs":
        for r in rows:
            print(r)
        return
    p = curves()
    images.append((p, "Association accuracy, route F1 and route success against the pose noise of the map (TEST): without registration the map-based transfer degrades quickly, graph-ICP registration holds it close to the exact-pose level up to 1 m (0.923 against 0.926) and within 0.04 at 2 m (0.883)."))
    try:
        p = examples()
        images.append((p, "TEST examples at 1 m pose noise: left - noisy map (red) and extracted graph (blue); middle - after registration; right - route flags transferred to the extracted graph over the ground-truth route."))
    except Exception as ex:  # noqa: BLE001
        print("Beispielbild:", ex)
    t = TEXT
    ph, pr = pert_table()
    ah, ar = alt_table()
    make_report2("E59", t.get("title", "E59"), t.get("question", ""), t.get("setup", []), [("Association of the extracted lane graph to the map graph and transfer of the navigation route (final system graphs of E44, TEST / VAL)", head, rows), ("Route transfer on the graphs of the perturbed system (E57; TEST, 1 m map pose noise, per-frame graph-ICP registration, geometric route flag)", ph, pr), ("Alternative navigation goals: the route is changed to the OTHER branch at the first split of the driven route (256 TEST samples whose driven route passes a split; the extracted graph is the same)", ah, ar)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.assoc_arch, arch_height=3.1)
    print("E59 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
