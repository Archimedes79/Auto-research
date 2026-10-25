"""Bericht E60 (Schleife 5): routen-bedingter dichter Kopf (Karte + Navigationsroute als Eingabe).  Liest out/loop2/e60_{main,route_only,nomap}/metrics[_pert_<bed>].json (route_head.py), e59_* (geometrische Uebertragung), e57_<bed>/metrics.json (Kopf ohne Karte);
Text loop2/loop5_text_e60.py (TEXT60).  Aufruf (gcar): make_e60.py [figs|report]"""
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
import dense_head as DH  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
HOME = os.path.expanduser("~")
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e60.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e60.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT60", {})

HEADS = [("e60_main", "route + lanes + directions (main)"), ("e60_route_only", "route only (control)"), ("e60_nomap", "no map input (control)")]
CONDS = [("clean", "exact map"), ("s1", "pose noise 1 m"), ("s2", "pose noise 2 m"), ("d30", "30 % of the lanes missing"), ("s1d30", "1 m noise + 30 % missing"), ("wrong", "WRONG map (another sample's)"), ("none", "no map given")]
PERT = [("dark", "night"), ("cam_front_off", "front camera off"), ("cam_front3_off", "3 forward cameras off"), ("blur", "blur"), ("extr3", "extrinsics 3 deg"), ("radar_off", "radar nearly off"), ("fog", "fog")]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def cell(d, key):
    try:
        x = d[key]
        return x["lane"]["0.6"]["F1@1.0"], x["route"]["0.5"]
    except Exception:  # noqa: BLE001
        return float("nan"), None


def table_cond(st):
    rows = []
    D = {h: jl(f"{OUT}/{h}/metrics.json") for h, _ in HEADS}
    for c, lab in CONDS:
        r = [lab]
        for h, _ in HEADS:
            lf, rt = cell(D[h] or {}, f"{st}_{c}")
            r += [f3(lf), f"{f3(rt['success']) if rt else '-'} ({f3(rt['cov']) if rt else '-'} / {f3(rt['prec']) if rt else '-'})"]
        rows.append(r)
    return rows


def table_geo():
    rows = []
    d0 = jl(f"{OUT}/e59_s0/metrics.json")
    m = jl(f"{OUT}/e60_main/metrics.json") or {}
    ro = jl(f"{OUT}/e60_route_only/metrics.json") or {}
    nm = jl(f"{OUT}/e60_nomap/metrics.json") or {}
    for lab, geo, key in (("exact pose", "e59_s0", "clean"), ("pose noise 1 m (+ registration)", "e59_s1.0_reg", "s1"), ("pose noise 2 m (+ registration)", "e59_s2.0_reg", "s2"), ("30 % of the lanes missing (route kept)", "e59_d0.3_keep", "d30"), ("no map", "e59_nomap", "none")):
        g = jl(f"{OUT}/{geo}/metrics.json")
        rows.append([lab, f3(g["test"]["routeG_success"]) if g else "-", f3(cell(m, f"test_{key}")[1]["success"]) if cell(m, f"test_{key}")[1] else "-", f3(cell(ro, f"test_{key}")[1]["success"]) if cell(ro, f"test_{key}")[1] else "-", f3(cell(nm, f"test_{key}")[1]["success"]) if cell(nm, f"test_{key}")[1] else "-"])
    return rows


def table_fit():
    m = jl(f"{OUT}/e60_main/metrics.json") or {}
    rows = []
    for c, lab in (("clean", "exact map"), ("s1", "pose noise 1 m"), ("s2", "pose noise 2 m"), ("wrong", "WRONG map"), ("none", "no map given")):
        r = [lab]
        for st in ("train", "val", "test"):
            lf, rt = cell(m, f"{st}_{c}")
            r += [f3(lf), f3(rt["success"]) if rt else "-"]
        rows.append(r)
    return rows


def table_pert():
    rows = []
    for p, lab in PERT:
        d = jl(f"{OUT}/e60_main/metrics_pert_{p}.json")
        pl = jl(f"{OUT}/e57_{p}/metrics.json")
        if not d:
            continue
        lf_c, rt_c = cell(d, "test_clean")
        lf_n, rt_n = cell(d, "test_none")
        lf_w, rt_w = cell(d, "test_wrong")
        lf_1, rt_1 = cell(d, "test_s1")
        rows.append([lab, f3(pl["pixel"]["0.6"]["F1@1.0"]) if pl else "-", f3(lf_c), f3(lf_1), f3(lf_n), f3(lf_w), f3(rt_c["success"]) if rt_c else "-", f3(rt_1["success"]) if rt_1 else "-", f3(rt_n["success"]) if rt_n else "-"])
    cl = jl(f"{OUT}/e60_main/metrics.json") or {}
    lf_c, rt_c = cell(cl, "test_clean")
    lf_n, rt_n = cell(cl, "test_none")
    lf_w, rt_w = cell(cl, "test_wrong")
    lf_1, rt_1 = cell(cl, "test_s1")
    e37 = jl(f"{OUT}/e57_stored/metrics.json")
    rows.insert(0, ["clean inputs", f3(e37["pixel"]["0.6"]["F1@1.0"]) if e37 else "-", f3(lf_c), f3(lf_1), f3(lf_n), f3(lf_w), f3(rt_c["success"]) if rt_c else "-", f3(rt_1["success"]) if rt_1 else "-", f3(rt_n["success"]) if rt_n else "-"])
    return rows


def curves():
    xs = [0.0, 1.0, 2.0]
    m = jl(f"{OUT}/e60_main/metrics.json") or {}
    ro = jl(f"{OUT}/e60_route_only/metrics.json") or {}
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
    geo = [jl(f"{OUT}/e59_s0/metrics.json"), jl(f"{OUT}/e59_s1.0_reg/metrics.json"), jl(f"{OUT}/e59_s2.0_reg/metrics.json")]
    axs[0].plot(xs, [g["test"]["routeG_success"] if g else np.nan for g in geo], "-o", color="#2ca02c", label="geometric transfer: registration + route flag (E59)")
    axs[0].plot(xs, [cell(m, f"test_{k}")[1]["success"] if cell(m, f"test_{k}")[1] else np.nan for k in ("clean", "s1", "s2")], "-o", color="#1f77b4", label="learned head, route + lanes (E60)")
    axs[0].plot(xs, [cell(ro, f"test_{k}")[1]["success"] if cell(ro, f"test_{k}")[1] else np.nan for k in ("clean", "s1", "s2")], "-o", color="#ff7f0e", label="learned head, route only")
    nm = jl(f"{OUT}/e59_nomap/metrics.json")
    if nm:
        axs[0].axhline(nm["test"]["routeG_success"], color="k", ls=":", label="no map, straight ahead (rule)")
    rt = cell(m, "test_none")[1]
    if rt:
        axs[0].axhline(rt["success"], color="#1f77b4", ls=":", label="learned head, no map given")
    axs[0].set_xlabel("map pose noise sigma (m)")
    axs[0].set_ylabel("route success (TEST)")
    axs[0].set_title("route transfer: geometric against learned", fontsize=9)
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=6.5)
    labs = [c[1] for c in CONDS]
    x = np.arange(len(CONDS))
    for k, (h, lab) in enumerate(HEADS):
        d = jl(f"{OUT}/{h}/metrics.json") or {}
        axs[1].bar(x + (k - 1) * 0.27, [cell(d, f"test_{c}")[0] for c, _ in CONDS], 0.26, label=lab)
    e37 = jl(f"{OUT}/e57_stored/metrics.json")
    if e37:
        axs[1].axhline(e37["pixel"]["0.6"]["F1@1.0"], color="k", ls=":", label="plain head e37_final")
    axs[1].set_ylim(0.8, 0.93)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(["exact\nmap", "noise\n1 m", "noise\n2 m", "30 %\nmissing", "1 m +\n30 %", "WRONG\nmap", "no map\ngiven"], fontsize=7)
    axs[1].set_ylabel("lane pixel F1 @1 m (TEST, tau 0.6)")
    axs[1].set_title("lane quality with the map as prior and under map errors", fontsize=9)
    axs[1].grid(alpha=0.3, axis="y")
    axs[1].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    p = f"{FIG}/e60_curves.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def pert_bars():
    avail = [(p, lab) for p, lab in PERT if os.path.exists(f"{OUT}/e60_main/metrics_pert_{p}.json")]
    if not avail:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    x = np.arange(len(avail) + 1)
    plain, withmap, nomap = [], [], []
    e37 = jl(f"{OUT}/e57_stored/metrics.json")
    d0 = jl(f"{OUT}/e60_main/metrics.json") or {}
    plain.append(e37["pixel"]["0.6"]["F1@1.0"] if e37 else np.nan)
    withmap.append(cell(d0, "test_clean")[0])
    nomap.append(cell(d0, "test_none")[0])
    for p, lab in avail:
        d = jl(f"{OUT}/e60_main/metrics_pert_{p}.json")
        pl = jl(f"{OUT}/e57_{p}/metrics.json")
        plain.append(pl["pixel"]["0.6"]["F1@1.0"] if pl else np.nan)
        withmap.append(cell(d, "test_clean")[0])
        nomap.append(cell(d, "test_none")[0])
    ax.bar(x - 0.27, plain, 0.26, label="plain head e37_final (E57)", color="#7f7f7f")
    ax.bar(x, nomap, 0.26, label="route-conditioned head, no map given", color="#ff7f0e")
    ax.bar(x + 0.27, withmap, 0.26, label="route-conditioned head, exact map", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(["clean"] + [lab.replace(" camera", "\ncamera").replace("3 forward", "3 forward\n") for _, lab in avail], fontsize=8)
    ax.set_ylabel("lane pixel F1 @1 m (TEST, tau 0.6)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    ax.set_title("E60 under sensor failures (E57 inputs): the map adds about 0.06-0.10 F1 but the head does not fall back on it", fontsize=9)
    fig.tight_layout()
    p = f"{FIG}/e60_pert.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def examples(n=3):
    pr = np.load(f"{OUT}/e60_main/prob_test.npy", mmap_mode="r")
    rp = np.load(f"{OUT}/e60_main/rprob_test.npy", mmap_mode="r")
    ids = np.load(f"{OUT}/e60_main/ids_test.npy")
    pts = pickle.load(open(f"{HOME}/exp_gcar_b123/route_pts_val.pkl", "rb"))
    gts = DH.load_gt_points("b123", "val", ids)
    GT = pickle.load(open(f"{HOME}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
    RT = pickle.load(open(f"{HOME}/exp_gcar_b123/route_gt_val.pkl", "rb"))
    cand = []
    for k, i in enumerate(ids):
        i = int(i)
        if pts[i] is None or RT[i] is None:
            continue
        outd = {}
        for a_, b_ in GT[i]["edges"]:
            outd[a_] = outd.get(a_, 0) + 1
        ns = sum(1 for p in RT[i]["route"] if outd.get(p, 0) >= 2)
        if ns >= 1 and len(pts[i]) > 20:
            cand.append(k)
    sel = cand[:: max(len(cand) // n, 1)][:n]
    fig, axs = plt.subplots(n, 3, figsize=(10.5, 3.45 * n), squeeze=False)
    for r, k in enumerate(sel):
        i = int(ids[k])
        g, rr = np.asarray(pr[k], np.float32), np.asarray(rp[k], np.float32)
        for c in range(3):
            ax = axs[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
        axs[r, 0].imshow(g.T, origin="lower", extent=[-50, 50, -50, 50], cmap="magma", vmin=0, vmax=1)
        axs[r, 0].plot(gts[k][:, 0], gts[k][:, 1], ".", color="#00e5ff", ms=0.6, alpha=0.6)
        axs[r, 1].imshow(rr.T, origin="lower", extent=[-50, 50, -50, 50], cmap="viridis", vmin=0, vmax=1)
        P_ = np.asarray(pts[i], np.float64)
        axs[r, 1].plot(P_[:, 0], P_[:, 1], ".", color="#ff1744", ms=0.8, alpha=0.7)
        axs[r, 2].imshow(0.5 * g.T, origin="lower", extent=[-50, 50, -50, 50], cmap="gray", vmin=0, vmax=1)
        axs[r, 2].imshow(np.ma.masked_less(rr.T, 0.25), origin="lower", extent=[-50, 50, -50, 50], cmap="autumn", vmin=0, vmax=1, alpha=0.9)
        if r == 0:
            axs[r, 0].set_title("lane heat map + ground-truth lanes (cyan)", fontsize=8)
            axs[r, 1].set_title("ROUTE heat map + ground-truth route (red)", fontsize=8)
            axs[r, 2].set_title("route (orange) over the lanes (grey)", fontsize=8)
    fig.tight_layout()
    p = f"{FIG}/e60_examples.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def main(mode):
    head = ["map condition"]
    for _, lab in HEADS:
        head += [f"lane F1 [{lab}]", f"route success (cov / prec) [{lab}]"]
    if mode == "figs":
        for r in table_cond("test"):
            print(r)
        for r in table_geo():
            print(r)
        for r in table_fit():
            print(r)
        for r in table_pert():
            print(r)
        return
    images = []
    for fn, cap in ((curves, "Left: route success of the learned heads against the pose noise of the map, with the geometric transfer of E59 and the no-map references; right: lane F1 under each map condition - the lane raster raises the lane F1, and with a wrong map or no map the lane F1 is that of the plain head (no leakage collapse)."), (examples, "TEST examples (exact map): the lane heat map, the ROUTE heat map with the ground-truth route, and the route over the lanes."), (pert_bars, "Lane F1 on the perturbed sensor inputs of E57 for the plain head and the route-conditioned head with the exact map and without a map.")):
        try:
            p = fn()
            if p:
                images.append((p, cap))
        except Exception as ex:  # noqa: BLE001
            print(fn.__name__, "Fehler:", ex)
    t = TEXT
    tabs = [("TEST: lane F1 @1 m (tau 0.6) and route success (coverage / precision @1.5 m) for every map condition", head, table_cond("test")), ("VAL: the same", head, table_cond("val")),
            ("Learned against geometric route transfer (route success, TEST)", ["map condition", "geometric transfer: registration + route flag (E59)", "learned head: route + lanes", "learned head: route only", "learned head: no map input"], table_geo()),
            ("Fit to the training data (main head): lane F1 and route success on TRAIN-eval, VAL and TEST rows", ["map condition", "TRAIN lane F1", "TRAIN route success", "VAL lane F1", "VAL route success", "TEST lane F1", "TEST route success"], table_fit())]
    tp = table_pert()
    if tp:
        tabs.append(("The main head on the perturbed sensor inputs of E57 (TEST): lane F1 and route success with the map", ["perturbation", "plain head lane F1 (E57)", "main head, exact map: lane F1", "1 m noise: lane F1", "no map: lane F1", "wrong map: lane F1", "route success, exact map", "route success, 1 m noise", "route success, no map"], tp))
    make_report2("E60", t.get("title", "E60"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.routehead_arch, arch_height=3.1)
    print("E60 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
