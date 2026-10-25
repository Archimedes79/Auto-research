"""Bericht E65 (Schleife 5): robustes Training - Merkmalsraum-Augmentierung (E65a) und Nachtrainieren auf echt gestoerten Merkmalen (E65b/c).
Liest out/loop2/e57_<bed>/metrics.json (Basis e37_final), out/loop2/e65_<kopf>_<bed>/metrics.json (eval_pert.py --head), out/loop2/<kopf>/metrics.json (Training / Anpassung); Text loop2/loop5_text_e65.py (TEXT65).  Aufruf (gcar): make_e65.py [figs|report]"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e65.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e65.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT65", {})

HEADS = [
    ("e37_final", "baseline: clean data only (e37_final)"),
    ("e65_faug", "E65a: feature-space augmentation (sector drop, gain, noise, blur, map off; 50 % of the samples)"),
    ("e65_ft", "E65b: fine-tuned on REAL perturbed training features (night, front camera off, fog) + clean"),
    ("e65_ftaug", "E65c: E65b + feature-space augmentation on the clean half"),
    ("e65_ftall", "E65f: fine-tuned on REAL perturbed training features of ALL seven types (every condition is seen)"),
]
# (Ordner, Beschriftung, in E65b gesehen?, Referenz fuer die halbierten Mengen)
CONDS = [
    ("stored", "clean", "-", "stored"),
    ("dark", "night", "seen", "stored"),
    ("cam_front_off", "front camera off", "seen", "stored"),
    ("fog", "fog", "seen", "stored"),
    ("cam_front3_off", "3 forward cameras off", "unseen", "stored"),
    ("blur", "blur", "unseen", "stored_s2"),
    ("radar_off", "radar nearly off", "unseen", "stored_s2"),
    ("extr3", "camera extrinsics 3 deg", "unseen", "stored_s2"),
]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def mfile(head, cond):
    if head == "e37_final":
        return f"{OUT}/e57_{cond}/metrics.json"
    return f"{OUT}/e65_{head}_{cond}/metrics.json"


def val(head, cond, key="F1@1.0", what="graph", tau="0.6"):
    d = jl(mfile(head, cond))
    if not d:
        return float("nan")
    if what == "pixel":
        return d["pixel"][tau][key]
    return d.get("graph", {}).get(key, float("nan"))


def table_graph(key="F1@1.0", what="graph", tau="0.6"):
    rows = []
    for h, lab in HEADS:
        if not any(os.path.exists(mfile(h, c)) for c, *_ in CONDS):
            continue
        rows.append([lab] + [f3(val(h, c, key, what, tau)) for c, *_ in CONDS])
    return rows


def table_fit():
    rows = []
    for h, lab in HEADS:
        d = jl(f"{OUT}/{h}/metrics.json")
        if not d:
            continue
        r = [lab]
        if h in ("e65_ft", "e65_ftaug", "e65_ftall"):
            ce = d.get("clean_train_eval", {}).get("0.5", {}).get("F1@1.0")
            r += [f3(ce)] + [f3(d.get(f"train_{p}", {}).get("0.5", {}).get("F1@1.0")) for p in ("dark", "cam_front_off", "fog")]
            r += [f3(val(h, "stored", "F1@1.0", "pixel", "0.5")), f3(val(h, "dark", "F1@1.0", "pixel", "0.5")), f3(val(h, "cam_front_off", "F1@1.0", "pixel", "0.5")), f3(val(h, "fog", "F1@1.0", "pixel", "0.5"))]
        elif h == "e65_faug":
            te = d.get("train_eval", {})
            ce = te.get("0.5", {}).get("F1@1.0") if te else None
            r += [f3(ce), "-", "-", "-"]
            r += [f3(val(h, "stored", "F1@1.0", "pixel", "0.5")), f3(val(h, "dark", "F1@1.0", "pixel", "0.5")), f3(val(h, "cam_front_off", "F1@1.0", "pixel", "0.5")), f3(val(h, "fog", "F1@1.0", "pixel", "0.5"))]
        else:
            te = jl(f"{OUT}/e37_final/metrics.json") or {}
            ce = te.get("train_eval", {}).get("0.5", {}).get("F1@1.0")
            bb = jl(f"{OUT}/e65_base/metrics.json") or {}
            r += [f3(ce)] + [f3(bb.get(f"train_{p}", {}).get("0.5", {}).get("F1@1.0")) for p in ("dark", "cam_front_off", "fog")]
            r += [f3(val(h, "stored", "F1@1.0", "pixel", "0.5")), f3(val(h, "dark", "F1@1.0", "pixel", "0.5")), f3(val(h, "cam_front_off", "F1@1.0", "pixel", "0.5")), f3(val(h, "fog", "F1@1.0", "pixel", "0.5"))]
        rows.append(r)
    return rows


def table_f7():
    """E65f: alle sieben Stoerungen gesehen - Gewinn auf den TRAIN-Zeilen, auf denen nachtrainiert wurde (in-sample), gegen TEST (Pixel-F1 @1 m, tau 0.5)"""
    base = jl(f"{OUT}/e65_base7/metrics.json") or {}
    ft = jl(f"{OUT}/e65_ftall/metrics.json") or {}
    rows = []
    for c, lab in (("dark", "night"), ("cam_front_off", "front camera off"), ("fog", "fog"), ("cam_front3_off", "3 forward cameras off"), ("blur", "blur"), ("radar_off", "radar nearly off"), ("extr3", "camera extrinsics 3 deg")):
        tb = base.get(f"train_{c}", {}).get("0.5", {}).get("F1@1.0", float("nan"))
        tf = ft.get(f"train_{c}", {}).get("0.5", {}).get("F1@1.0", float("nan"))
        eb = val("e37_final", c, "F1@1.0", "pixel", "0.5")
        ef = val("e65_ftall", c, "F1@1.0", "pixel", "0.5")
        rows.append([lab, f3(tb), f3(tf), f3(tf - tb), f3(eb), f3(ef), f3(ef - eb)])
    return rows


PERT_ALL = [("clean", "clean inputs"), ("dark", "night"), ("cam_front_off", "front camera off"), ("fog", "fog"), ("cam_front3_off", "3 forward cameras off"), ("blur", "blur"), ("radar_off", "radar nearly off"), ("extr3", "extrinsics 3 deg")]


def rcell(head, pert, cond):
    """Route-Kopf: (Spur-F1 tau 0.6, Routen-Erfolg tau 0.5) fuer Stoerung und Kartenbedingung."""
    if pert == "clean":
        d = jl(f"{OUT}/{head}/metrics.json")
    else:
        d = jl(f"{OUT}/{head}/metrics_pert_{pert}.json")
    try:
        x = d[f"test_{cond}"]
        return x["lane"]["0.5"]["F1@1.0"], x["route"]["0.5"]["success"]
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def table_route():
    rows = []
    for p, lab in PERT_ALL:
        r = [lab]
        for head in ("e60_main", "e65d_route_ft"):
            for cond in ("clean", "s1", "none", "wrong"):
                lf, rs = rcell(head, p, cond)
                r.append(f"{f3(lf)} / {f3(rs)}")
        rows.append(r)
    return rows


def route_bars():
    avail = [(p, lab) for p, lab in PERT_ALL if not np.isnan(rcell("e65d_route_ft", p, "clean")[0])]
    if not avail:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.9))
    x = np.arange(len(avail))
    for ax, idx, ttl in ((axs[0], 0, "lane pixel F1 @1 m (tau 0.5)"), (axs[1], 1, "route success")):
        for k, (head, cond, lab, col) in enumerate((("e60_main", "clean", "E60 head, exact map", "#7f7f7f"), ("e65d_route_ft", "clean", "E65d head, exact map", "#1f77b4"), ("e65d_route_ft", "wrong", "E65d head, WRONG map", "#d62728"), ("e65d_route_ft", "none", "E65d head, no map", "#ff7f0e"))):
            ax.bar(x + (k - 1.5) * 0.2, [rcell(head, p, cond)[idx] for p, _ in avail], 0.19, label=lab, color=col)
        ax.set_xticks(x)
        short = {"clean inputs": "clean", "night": "night", "front camera off": "front\ncam off", "fog": "fog", "3 forward cameras off": "3 fwd\ncams off", "blur": "blur", "radar nearly off": "radar\noff", "extrinsics 3 deg": "extr.\n3 deg"}
        ax.set_xticklabels([short.get(lab, lab) for _, lab in avail], fontsize=7)
        ax.set_ylim(0, 1.22)
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axs[0].legend(fontsize=6.5, loc="upper center", ncol=2)
    fig.suptitle("E65d: the map-conditioned head fine-tuned on real perturbed features (TEST) - does it fall back on the map, and does it still reject a wrong one?", fontsize=9)
    fig.tight_layout()
    p = f"{FIG}/e65d_route.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def table_adabn():
    d = jl(f"{OUT}/e65e_adabn/metrics.json") or {}
    rows = []
    for c, lab, *_ in CONDS:
        if c not in d:
            continue
        b, a = d[c]["base"], d[c]["adabn"]
        rows.append([lab, b["n"], d[c]["n_adapt"], f3(b["graph_F1"]), f3(a["graph_F1"]), f3(a["graph_F1"] - b["graph_F1"]), f"{f3(b['pixel_F1'])} / {f3(a['pixel_F1'])}", f"{f3(b['route_f'])} / {f3(a['route_f'])}", f"{f3(b['lanes_found'])} / {f3(a['lanes_found'])}"])
    return rows


def adabn_fig():
    d = jl(f"{OUT}/e65e_adabn/metrics.json") or {}
    cs = [(c, lab) for c, lab, *_ in CONDS if c in d]
    if len(cs) < 3:
        return None
    fig, ax = plt.subplots(figsize=(11, 3.7))
    x = np.arange(len(cs))
    ax.bar(x - 0.2, [d[c]["base"]["graph_F1"] for c, _ in cs], 0.38, color="#7f7f7f", label="clean-trained head (odd TEST rows)")
    ax.bar(x + 0.2, [d[c]["adabn"]["graph_F1"] for c, _ in cs], 0.38, color="#2ca02c", label="+ unsupervised BN adaptation on 200 other (unlabelled) TEST rows")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in cs], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("graph point F1 @1 m")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    ax.set_title("E65e: test-time adaptation of the BatchNorm statistics (no labels, no training)", fontsize=9)
    fig.tight_layout()
    p = f"{FIG}/e65e_adabn.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def qualitative(pert="dark", n=3):
    """Gleiche TEST-Bilder unter Nacht: Basis-Kopf, nachtrainierter Kopf (E65b), Karten-Kopf E65d mit genauer / falscher Karte."""
    import dense_head as DH

    arrs = {}
    need = {"base": f"{OUT}/e57_{pert}", "ft": f"{OUT}/e65_e65_ft_{pert}"}
    for k, d in need.items():
        if not os.path.exists(f"{d}/prob_test.npy"):
            return None
        arrs[k] = (np.load(f"{d}/prob_test.npy", mmap_mode="r"), np.load(f"{d}/ids_test.npy"))
    rh = f"{OUT}/e65d_route_ft"
    cols = [("base", "plain head (clean-trained)"), ("ft", "E65b: fine-tuned on perturbed features")]
    rp = {}
    for cond, lab in (("clean", "E65d map-conditioned head, exact map"), ("wrong", "E65d, WRONG map"), ("none", "E65d, no map")):
        f = f"{rh}/prob_pert_{pert}_{cond}.npy"
        if os.path.exists(f):
            rp[cond] = np.load(f, mmap_mode="r")
            cols.append((cond, lab))
    ids_r = np.load(f"{rh}/ids_pert_{pert}.npy") if os.path.exists(f"{rh}/ids_pert_{pert}.npy") else None
    common = sorted(set(arrs["base"][1].tolist()).intersection(arrs["ft"][1].tolist()))
    if ids_r is not None:
        common = sorted(set(common).intersection(ids_r.tolist()))
    if len(common) < n:
        return None
    sel = [common[int(i)] for i in np.linspace(len(common) * 0.15, len(common) * 0.8, n)]
    gts = DH.load_gt_points("b123", "val", sel)
    fig, axs = plt.subplots(n, len(cols), figsize=(2.6 * len(cols), 2.7 * n), squeeze=False)
    for c, (k, lab) in enumerate(cols):
        for r, row in enumerate(sel):
            if k in arrs:
                pr = np.asarray(arrs[k][0][arrs[k][1].tolist().index(row)], np.float32)
            else:
                pr = np.asarray(rp[k][ids_r.tolist().index(row)], np.float32)
            ax = axs[r, c]
            ax.imshow(pr.T, origin="lower", extent=[-50, 50, -50, 50], cmap="magma", vmin=0, vmax=1)
            ax.plot(gts[r][:, 0], gts[r][:, 1], ".", color="#00e5ff", ms=0.6, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=6.5)
    fig.suptitle(f"The same TEST frames under the perturbation '{pert}' (heat map, ground truth in cyan): how each remedy reads the dark input", fontsize=8)
    fig.tight_layout()
    p = f"{FIG}/e65_qualitative_{pert}.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def bars():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    cols = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    n = len([1 for h, _ in HEADS if any(os.path.exists(mfile(h, c)) for c, *_ in CONDS)])
    w = 0.8 / max(n, 1)
    k = 0
    for i, (h, lab) in enumerate(HEADS):
        if not any(os.path.exists(mfile(h, c)) for c, *_ in CONDS):
            continue
        ax.bar(np.arange(len(CONDS)) + (k - (n - 1) / 2) * w, [val(h, c) for c, *_ in CONDS], w, label=lab.split(":")[0] if ":" in lab else "baseline", color=cols[i])
        k += 1
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([f"{lab}\n({s})" if s != "-" else lab for _, lab, s, _ in CONDS], fontsize=8)
    ax.set_ylabel("graph point F1 @1 m (TEST)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("E65: robust training - graph F1 on clean and perturbed TEST inputs (seen / unseen = perturbation type used in the fine-tuning)", fontsize=10)
    fig.tight_layout()
    p = f"{FIG}/e65_bars.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def main(mode):
    head = ["head"] + [lab + (f" ({s})" if s != "-" else "") for _, lab, s, _ in CONDS]
    if mode == "figs":
        for r in table_graph():
            print(r)
        for r in table_fit():
            print(r)
        return
    images = []
    try:
        images.append((bars(), "Graph F1 of the baseline and the robustly trained heads on clean and perturbed TEST inputs."))
    except Exception as ex:  # noqa: BLE001
        print("bars:", ex)
    head_fit = ["head", "pixel F1 on clean TRAIN-eval rows", "on perturbed TRAIN rows: night", "front camera off", "fog", "pixel F1 on TEST: clean", "night", "front camera off", "fog"]
    tabs = [("Graph point F1 @1 m on TEST (strict tracer), per perturbation", head, table_graph("F1@1.0")), ("Graph route F on TEST", head, table_graph("route_f")), ("Pixel F1 @1 m (tau 0.6) on TEST", head, table_graph("F1@1.0", "pixel", "0.6")), ("Pixel F1 @1 m (tau 0.5) on TEST - the fine-tuned heads have a lower output level on perturbed inputs, so tau 0.5 suits them better", head, table_graph("F1@1.0", "pixel", "0.5")), ("Fit to the training data against TEST (pixel F1 @1 m, tau 0.5)", head_fit, table_fit())]
    try:
        p = qualitative("dark")
        if p:
            images.append((p, "Qualitative: the same TEST frames at night - the clean-trained head finds almost nothing, the head fine-tuned on perturbed features finds the strongest lanes, the map-conditioned head fine-tuned on perturbed features reproduces the map (exact map: right; wrong map: wrong lanes; no map: little)."))
    except Exception as ex:  # noqa: BLE001
        print("qualitative:", ex)
    try:
        p = route_bars()
        if p:
            images.append((p, "E65d: lane F1 and route success of the map-conditioned head before (E60) and after fine-tuning on real perturbed features, with the exact map, with a WRONG map and with no map."))
    except Exception as ex:  # noqa: BLE001
        print("route_bars:", ex)
    try:
        p = adabn_fig()
        if p:
            images.append((p, "E65e: unsupervised adaptation of the BatchNorm statistics of the clean-trained head to the perturbed inputs (even TEST rows, features only) and evaluation on the odd rows."))
    except Exception as ex:  # noqa: BLE001
        print("adabn_fig:", ex)
    if os.path.exists(f"{OUT}/e65_base7/metrics.json") and os.path.exists(f"{OUT}/e65_ftall/metrics.json"):
        tabs.append(("E65f - the head has seen all seven perturbation types: pixel F1 @1 m (tau 0.5) on the perturbed TRAIN rows it was fine-tuned on (in-sample) and on TEST", ["perturbation", "baseline: TRAIN rows", "E65f: TRAIN rows", "gain on the TRAIN rows (in-sample)", "baseline: TEST", "E65f: TEST", "gain on TEST"], table_f7()))
    ta = table_adabn()
    if ta:
        tabs.append(("E65e - test-time BatchNorm adaptation (AdaBN): adapt on features of even TEST rows, evaluate on the odd rows; head e37_final", ["condition", "n eval rows", "n adaptation rows", "graph F1 @1 m: clean-trained head", "with AdaBN", "difference", "pixel F1: base / AdaBN", "route F: base / AdaBN", "lanes found: base / AdaBN"], ta))
    tr = table_route()
    if any("nan" not in c for r in tr for c in r[1:]) and os.path.exists(f"{OUT}/e65d_route_ft/metrics.json"):
        tabs.append(("E65d - map-conditioned head on perturbed TEST inputs: lane F1 / route success for the exact map, 1 m pose noise, no map and a wrong map", ["perturbation", "E60 head: exact map", "1 m noise", "no map", "wrong map", "E65d head: exact map", "1 m noise", "no map", "wrong map"], tr))
    t = TEXT
    from arch_figs import blocks_arch  # noqa: E402

    def arch(ax):
        b = [
            (0.1, 1.95, 2.4, 0.85, "clean TRAIN features\n(8,414 samples)", "in"),
            (0.1, 0.35, 2.6, 0.85, "REAL perturbed TRAIN features\n(night, front camera off, fog;\nevery 10th sample)", "warn"),
            (3.2, 1.15, 2.4, 1.25, "mixed batches\n(4 clean + 4 perturbed)\n[+ feature-space augmentation]", "post"),
            (6.2, 1.15, 2.3, 1.25, "dense head, initialised\nfrom e37_final\n(2000 steps, lr 2e-4)", "net"),
            (9.1, 1.95, 2.3, 0.85, "TEST: clean + 7 perturbations\n(4 types never seen)", "post"),
            (9.1, 0.35, 2.3, 0.85, "TRAIN rows: fit and\nforgetting check", "post"),
            (11.9, 1.15, 1.8, 1.25, "graph F1,\nroute F", "ok"),
        ]
        e = [(0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
        blocks_arch(ax, b, e, 13.8, 3.0)
        ax.set_title("E65: robust training of the dense head on real perturbed features (E65b) and with feature-space augmentation (E65a)", fontsize=10, loc="left")

    make_report2("E65", t.get("title", "E65"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=arch, arch_height=3.0)
    print("E65 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
