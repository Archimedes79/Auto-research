"""Bericht E62 (Schleife 5): planer-nahe Auswertung der uebertragenen Route (offene Schleife).  Liest out/loop2/e62_*/metrics.json (plan_eval.py); Text loop2/loop5_text_e62.py (TEXT62).  Aufruf (gcar): make_e62.py [figs|report]"""
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
import plan_eval as PE  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e62.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e62.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT62", {})

METH = [
    ("const", "straight at the current speed (no map, no lanes)"),
    ("nomap", "extracted graph, no map: straight-ahead successor at every split"),
    ("system", "SYSTEM: extracted graph + route transferred from the map (exact pose)"),
    ("oracle", "ORACLE route flags on the extracted graph (extraction limit)"),
    ("regmap", "map route polyline registered with the extracted graph (perception only places the map)"),
    ("maponly", "map route polyline as given (noisy pose, no registration, no perception)"),
    ("gt_route", "map route polyline at exact pose (upper bound of route following)"),
]
CONDS = [
    ("e62_s0", "exact pose"),
    ("e62_s1_reg", "pose noise 1 m, graph-ICP registration"),
    ("e62_s1_noreg", "pose noise 1 m, no registration"),
    ("e62_s2_reg", "pose noise 2 m, graph-ICP registration"),
    ("e62_d03", "30 % of the lanes missing (route kept), registration"),
]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def table_methods(cond="e62_s0", st="test"):
    d = jl(f"{OUT}/{cond}/metrics.json")
    rows = []
    if not d:
        return rows
    m = d[st]
    for k, lab in METH:
        rows.append([lab, f3(m[f"all_{k}_lata"], 2), f3(m[f"all_{k}_lata_med"], 2), f3(m.get(f"all_{k}_deva")), f3(m[f"curve_{k}_lata"], 2), f3(m[f"curve_{k}_lata_med"], 2), f3(m.get(f"curve_{k}_deva")), f3(m[f"all_{k}_lat"], 2), f3(m[f"all_{k}_l2"], 2), f3(m[f"all_{k}_conf"]), f3(m.get(f"all_{k}_nopoly", float("nan")))])
    return rows


def table_conditions(st="test"):
    rows = []
    for c, lab in CONDS:
        d = jl(f"{OUT}/{c}/metrics.json")
        if not d:
            continue
        m = d[st]
        for k, kl in (("system", "SYSTEM (perceived lanes + transferred route)"), ("regmap", "registered map route"), ("maponly", "map route as given")):
            if f"all_{k}_lat" not in m:
                continue
            rows.append([lab, kl, f3(m[f"all_{k}_lata"], 2), f3(m[f"all_{k}_lata_med"], 2), f3(m[f"curve_{k}_lata"], 2), f3(m[f"curve_{k}_lata_med"], 2), f3(m[f"all_{k}_conf"]), f3(m[f"curve_{k}_conf"]), f3(m.get(f"all_{k}_nopoly"))])
    d = jl(f"{OUT}/e62_s0/metrics.json")
    if d:
        m = d[st]
        for k, kl in (("nomap", "no map: straight-ahead successor"), ("const", "straight at the current speed")):
            rows.append(["(reference, independent of the map)", kl, f3(m[f"all_{k}_lata"], 2), f3(m[f"all_{k}_lata_med"], 2), f3(m[f"curve_{k}_lata"], 2), f3(m[f"curve_{k}_lata_med"], 2), f3(m[f"all_{k}_conf"]), f3(m[f"curve_{k}_conf"]), f3(m.get(f"all_{k}_nopoly"))])
    return rows


def bars(st="test"):
    d = jl(f"{OUT}/e62_s1_reg/metrics.json")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.8))
    cols = {"const": "#7f7f7f", "nomap": "#d62728", "system": "#1f77b4", "oracle": "#17becf", "regmap": "#ff7f0e", "gt_route": "#2ca02c", "maponly": "#9467bd"}
    names = {"const": "straight", "nomap": "no map", "system": "SYSTEM", "oracle": "oracle flags", "regmap": "registered map route", "gt_route": "map route (exact)", "maponly": "map route as given"}
    for ax, grp, ttl in zip(axs, ("all", "curve"), ("all frames", "curve / junction frames")):
        for i, k in enumerate(names):
            m = d[st][f"{grp}_{k}_lata"]
            md = d[st][f"{grp}_{k}_lata_med"]
            ax.bar(i - 0.2, m, 0.38, color=cols[k], alpha=0.9, label="mean" if i == 0 else None)
            ax.bar(i + 0.2, md, 0.38, color=cols[k], alpha=0.45, hatch="//", label="median" if i == 0 else None)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(list(names.values()), fontsize=8, rotation=15)
        ax.set_ylabel("lateral error (m)")
        ax.set_title(f"{ttl} (TEST, {d[st][grp + '_n']} frames)", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    p = f"{FIG}/e62_bars.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def examples(n=4, sigma=1.0):
    GT = pickle.load(open(f"{AE.D}/graph_gt_val.pkl", "rb"))
    RT = pickle.load(open(f"{AE.D}/route_gt_val.pkl", "rb"))
    P = np.load(f"{AE.D}/poses_val.npz")
    T, R, prev = P["t"], P["R"], P["prev"]
    N = len(T)
    nxt = -np.ones(N, int)
    for i in range(N):
        if prev[i] >= 0:
            nxt[prev[i]] = i
    d = pickle.load(open(f"{OUT}/e44_cons_p/graphs_test.pkl", "rb"))
    ids = [int(i) for i in d["ids"]]
    cand = []
    for k, i in enumerate(ids):
        rt = RT[i]
        if rt is None or not np.isfinite(rt["v0"]) or len(d["graphs"][k]["pts"]) < 6:
            continue
        j, seq = nxt[i], []
        while j >= 0 and len(seq) < 6:
            seq.append(j)
            j = nxt[j]
        if len(seq) < 6:
            continue
        pos = np.stack([(T[s] - T[i]) @ R[i] for s in seq])
        outd = {}
        for a_, b_ in GT[i]["edges"]:
            outd[a_] = outd.get(a_, 0) + 1
        n_split = sum(1 for p in rt["route"] if outd.get(p, 0) >= 2)
        if n_split >= 1 and abs(pos[-1, 0]) > 3.0:
            cand.append((k, pos))
    sel = cand[:: max(len(cand) // n, 1)][:n]
    fig, axs = plt.subplots(1, len(sel), figsize=(4.2 * len(sel), 4.4), squeeze=False)
    for c, (k, pos) in enumerate(sel):
        i = ids[k]
        ex, gt, rt = d["graphs"][k], GT[i], RT[i]
        route_tok = {gt["tokens"][p] for p in rt["route"]}
        rng = np.random.default_rng(AE.seed_of(i, 1))
        mp_ = AL.perturb_map(gt, sigma, 0.0, protect=set(rt["route"]), rng=rng)
        r = AL.associate(ex, mp_, register_graph=True, route_tok=route_tok)
        EV, ET, Epid, _, _ = AL.gt_assign(ex, gt)
        poly_c, _ = PE.route_polyline_from_flags(ex, r["point_route"], EV, Epid)
        sel_s = AE.follow_straight(ex)
        out_e = {}
        for a_, b_ in ex["edges"]:
            out_e.setdefault(a_, []).append(b_)
        poly_d = np.zeros((0, 2))
        if sel_s:
            start = min(sel_s, key=lambda k_: float(np.min(np.linalg.norm(AL.resample(ex["pts"][k_], 1.0), axis=1))) if len(ex["pts"][k_]) >= 2 else 1e9)
            seq, cur = [start], start
            while True:
                nx = [j for j in out_e.get(cur, []) if j in sel_s and j not in seq]
                if not nx:
                    break
                t0 = AL.tangents(np.asarray(ex["pts"][cur], np.float64))[-1]
                cur = max(nx, key=lambda j: float(AL.tangents(np.asarray(ex["pts"][j], np.float64))[0] @ t0))
                seq.append(cur)
            poly_d = PE.chain_polyline([np.asarray(ex["pts"][k_], np.float64) for k_ in seq])
        ax = axs[0, c]
        ax.set_xlim(-30, 30)
        ax.set_ylim(-10, 50)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        for p in ex["pts"]:
            p = np.asarray(p)
            ax.plot(p[:, 0], p[:, 1], "-", color="#b0b0b0", lw=1.0)
        for p in rt["route"]:
            q = np.asarray(gt["pts"][p])
            ax.plot(q[:, 0], q[:, 1], "-", color="#2ca02c", lw=6, alpha=0.3, label="map route (ground truth)" if c == 0 and p == rt["route"][0] else None)
        if len(poly_d) > 1:
            ax.plot(poly_d[:, 0], poly_d[:, 1], "--", color="#d62728", lw=2, label="no map: straight-ahead" if c == 0 else None)
        if len(poly_c) > 1:
            ax.plot(poly_c[:, 0], poly_c[:, 1], "-", color="#1f77b4", lw=2.2, label="SYSTEM route (1 m pose noise)" if c == 0 else None)
        ax.plot(pos[[1, 3, 5], 0], pos[[1, 3, 5], 1], "ko", ms=5, label="driven: 1 / 2 / 3 s" if c == 0 else None)
        ax.plot(0, 0, "k^", ms=8)
        ax.set_title(f"sample {i}", fontsize=8)
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    p = f"{FIG}/e62_examples.png"
    fig.savefig(p, dpi=95)
    plt.close(fig)
    return p


def main(mode):
    head1 = ["route given to the planner", "lateral err. all (mean, m)", "(median)", "share > 2 m at 3 s", "curve / junction (mean)", "(median)", "share > 2 m", "same, not anchored: all (mean)", "L2 1-3 s with v0 (mean, m)", "compliance (3 s end <= 1.5 m of the map route)", "no route polyline"]
    head2 = ["map condition", "route given to the planner", "lateral err. all (mean, m)", "(median)", "curve / junction (mean)", "(median)", "compliance all", "compliance curve / junction", "no route polyline"]
    if mode == "figs":
        for r in table_methods():
            print(r)
        for r in table_conditions():
            print(r)
        return
    images = []
    p = bars()
    images.append((p, "Lateral error of the predicted position after 1 / 2 / 3 s (anchored at the ego pose) when a planner follows the route polyline with the TRUE longitudinal profile (TEST, map with 1 m pose noise, registration where a route needs it). The perceived route is limited by the extraction; the map route placed by the registration is accurate; the map route as given suffers from the pose noise."))
    try:
        p = examples()
        images.append((p, "TEST curve / junction frames at 1 m pose noise: extracted graph (grey), ground-truth route (wide green), SYSTEM route polyline (blue), no-map straight-ahead choice (red dashed), driven positions after 1 / 2 / 3 s (black)."))
    except Exception as ex:  # noqa: BLE001
        print("Beispielbild:", ex)
    t = TEXT
    tabs = [("Route given to the planner, exact pose (TEST)", head1, table_methods("e62_s0", "test")), ("Route given to the planner, exact pose (VAL)", head1, table_methods("e62_s0", "val")),
            ("SYSTEM route under map disturbances (TEST)", head2, table_conditions("test")), ("SYSTEM route under map disturbances (VAL)", head2, table_conditions("val"))]
    make_report2("E62", t.get("title", "E62"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.plan_arch, arch_height=3.0)
    print("E62 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
