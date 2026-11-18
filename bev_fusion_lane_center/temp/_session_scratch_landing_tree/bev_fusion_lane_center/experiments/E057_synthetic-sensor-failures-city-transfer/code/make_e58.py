"""Bericht E58 (Schleife 5): Ortswechsel Boston <-> Singapur bei gleicher Zeilenzahl + Abstand zum naechsten Trainingsbild.  Liest out/loop2/e58_{B,S,M}/metrics.json (dense_head.py), out/loop2/e58_distbins.json (eval_distbins.py);
Text loop2/loop5_text_e58.py (TEXT58).  Aufruf (gcar): make_e58.py [figs|report]"""
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
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e58.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e58.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT58", {})

HEADS = [("e58_B", "B: Boston only (2,200 rows)", 0), ("e58_S", "S: Singapore only (2,200 rows)", 1), ("e58_M", "M: both cities (2,200 rows)", None)]
SETS = [("val_b", "VAL Boston"), ("val_s", "VAL Singapore"), ("test_b", "TEST Boston"), ("test_s", "TEST Singapore")]
TAU = "0.6"


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def tau_star(d):
    """tau je Kopf auf VAL gewaehlt (Mittel der Pixel-F1 @1 m von VAL Boston und VAL Singapur)."""
    best, bt = -1.0, "0.6"
    for t in ("0.3", "0.4", "0.5", "0.6", "0.7", "0.8"):
        try:
            v = 0.5 * (d["val_b"][t]["F1@1.0"] + d["val_s"][t]["F1@1.0"])
        except Exception:  # noqa: BLE001
            continue
        if v > best:
            best, bt = v, t
    return bt


def f1(d, st):
    try:
        return d[st][tau_star(d)]["F1@1.0"]
    except Exception:  # noqa: BLE001
        return float("nan")


def ref_city_f1():
    """e37_final (alle Trainingszeilen) auf den Teilmengen je Ort aus den gespeicherten Wahrscheinlichkeitskarten."""
    import torch

    import dense_head as DH

    cache = f"{OUT}/e58_ref_city.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    loc = np.load(f"{DH.HOME}/exp_gcar_b123/poses_val.npz")["loc"]
    res = {}
    for st in ("val", "test"):
        prob = np.load(f"{OUT}/e37_final/prob_{st}.npy", mmap_mode="r")
        ids = np.load(f"{OUT}/e37_final/ids_{st}.npy")
        gts = DH.load_gt_points("b123", "val", ids)
        for nm, m in (("b", loc[ids] == 0), ("s", loc[ids] != 0)):
            sel = np.nonzero(m)[0]
            r = DH.eval_probs(torch.from_numpy(np.asarray(prob[sel])), [gts[i] for i in sel], taus=(0.6,))
            res[f"{st}_{nm}"] = r["0.6"]["F1@1.0"]
    json.dump(res, open(cache, "w"))
    return res


def table_main():
    ref = ref_city_f1()
    rows = []
    for h, lab, _ in HEADS:
        d = jl(f"{OUT}/{h}/metrics.json")
        if not d:
            continue
        rows.append([lab, f3(f1(d, "train_own")), f"{f3(f1(d, 'train_eval_b'))} / {f3(f1(d, 'train_eval_s'))}"] + [f3(f1(d, s)) for s, _ in SETS])
    rows.append(["(reference) all 8,414 training rows, both cities (e37_final)", "-", "-"] + [f3(ref.get(s)) for s, _ in SETS])
    return rows


def table_gap():
    rows = []
    for h, lab, city in HEADS:
        d = jl(f"{OUT}/{h}/metrics.json")
        if not d:
            continue
        for st in ("val", "test"):
            b, s = f1(d, f"{st}_b"), f1(d, f"{st}_s")
            if city == 0:
                rows.append([lab, st.upper(), f3(b), f3(s), f3(s - b)])
            elif city == 1:
                rows.append([lab, st.upper(), f3(s), f3(b), f3(b - s)])
            else:
                rows.append([lab, st.upper(), f"{f3(b)} (B) / {f3(s)} (S)", "-", "-"])
    return rows


FT = [("e58_B", "head B as trained (no Singapore data)", 0), ("e58b_N100", "+ fine-tune on 100 Singapore rows (400 steps)", 100), ("e58b_N400", "+ fine-tune on 400 Singapore rows (800 steps)", 400), ("e58b_N1600", "+ fine-tune on 1,600 Singapore rows (1,500 steps)", 1600), ("e58_S", "(reference) head S: 2,200 Singapore rows from scratch", 2200)]


def table_ft():
    ref = ref_city_f1()
    rows = []
    for h, lab, n in FT:
        d = jl(f"{OUT}/{h}/metrics.json")
        if not d:
            continue
        rows.append([lab, f3(f1(d, "val_s")), f3(f1(d, "test_s")), f3(f1(d, "val_b")), f3(f1(d, "test_b"))])
    rows.append(["(reference) all 8,414 rows, both cities", f3(ref.get("val_s")), f3(ref.get("test_s")), f3(ref.get("val_b")), f3(ref.get("test_b"))])
    return rows


def ftplot():
    xs, ys_v, ys_t, yb = [], [], [], []
    for h, lab, n in FT:
        d = jl(f"{OUT}/{h}/metrics.json")
        if not d:
            continue
        xs.append(max(n, 30))
        ys_v.append(f1(d, "val_s"))
        ys_t.append(f1(d, "test_s"))
        yb.append(f1(d, "test_b"))
    if len(xs) < 3:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(xs[:-1], ys_t[:-1], "-o", color="#d62728", label="TEST Singapore (fine-tuned B)")
    ax.plot(xs[:-1], ys_v[:-1], "-o", color="#1f77b4", label="VAL Singapore (fine-tuned B)")
    ax.plot(xs[:-1], yb[:-1], "--s", color="#7f7f7f", label="TEST Boston (forgetting)")
    ax.plot([xs[-1]], [ys_t[-1]], "r*", ms=12, label="S from scratch, 2,200 rows (TEST)")
    ax.set_xscale("log")
    ax.set_xlabel("Singapore training rows used for the fine-tuning (30 = none)")
    ax.set_ylabel("pixel F1 @1 m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("E58b: closing the city gap with a few target-domain rows", fontsize=9)
    fig.tight_layout()
    p = f"{FIG}/e58b_ft.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def table_bins():
    d = jl(f"{OUT}/e58_distbins.json")
    if not d:
        return [], []
    keys = ["0-5", "5-10", "10-25", "25-100", "100-inf"]
    head = ["head / evaluation set", "frames within 5 m / 5-10 / 10-25 / 25-100 / > 100 m of a training frame"] + [f"F1 @1 m, {k} m" for k in keys] + ["F1 all"]
    rows = []
    for key in ("e37_final/val", "e37_final/test", "e58_B/test_b", "e58_B/test_s", "e58_S/test_b", "e58_S/test_s", "e58_M/test_b", "e58_M/test_s"):
        if key not in d:
            continue
        v = d[key]
        rows.append([key.replace("e37_final", "all-data head").replace("e58_", "head "), " / ".join(str(v[k]["n"]) if k in v else "0" for k in keys)] + [f3(v[k]["F1"]) if k in v else "-" for k in keys] + [f3(v["all"]["F1"])])
    return head, rows


def bars():
    ref = ref_city_f1()
    fig, ax = plt.subplots(figsize=(11, 3.9))
    sets = SETS
    w = 0.2
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"]
    for k, (h, lab, _) in enumerate(HEADS):
        d = jl(f"{OUT}/{h}/metrics.json")
        if not d:
            continue
        ax.bar(np.arange(len(sets)) + (k - 1.5) * w, [f1(d, s) for s, _ in sets], w, label=lab, color=cols[k])
    ax.bar(np.arange(len(sets)) + 1.5 * w, [ref.get(s, np.nan) for s, _ in sets], w, label="all 8,414 rows (e37_final)", color=cols[3])
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([n for _, n in sets])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("pixel F1 @1 m (tau chosen on VAL per head)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("E58: city transfer with equal training size", fontsize=10)
    fig.tight_layout()
    p = f"{FIG}/e58_bars.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def binplot():
    d = jl(f"{OUT}/e58_distbins.json")
    if not d:
        return None
    keys = ["0-5", "5-10", "10-25", "25-100", "100-inf"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for key, c, ls in (("e37_final/val", "#1f77b4", "-"), ("e37_final/test", "#d62728", "-")):
        if key in d:
            axs[0].plot([k for k in keys if k in d[key]], [d[key][k]["F1"] for k in keys if k in d[key]], "-o", color=c, label=key.replace("e37_final", "all-data head"))
    axs[0].set_title("F1 against the distance of the frame to the nearest training frame (same city)", fontsize=9)
    for key, c, ls in (("e58_B/test_b", "#1f77b4", "-"), ("e58_B/test_s", "#1f77b4", "--"), ("e58_S/test_s", "#d62728", "-"), ("e58_S/test_b", "#d62728", "--")):
        if key in d:
            axs[1].plot([k for k in keys if k in d[key]], [d[key][k]["F1"] for k in keys if k in d[key]], ls, marker="o", color=c, label=key.replace("e58_", "head ").replace("/test_b", " on TEST Boston").replace("/test_s", " on TEST Singapore"))
    axs[1].set_title("city heads (solid: own city, dashed: other city)", fontsize=9)
    for ax in axs:
        ax.set_xlabel("distance to the nearest training frame (m)")
        ax.set_ylabel("pixel F1 @1 m")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = f"{FIG}/e58_bins.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def examples(n_s=3, n_b=2):
    """Gleiche TEST-Bilder unter den drei gleich grossen Koepfen: Singapur-Bilder (B = fremde Stadt) und Boston-Bilder (S = fremde Stadt)."""
    import dense_head as DH

    fig, axs = plt.subplots(n_s + n_b, 3, figsize=(8.6, 2.9 * (n_s + n_b)), squeeze=False)
    k_row = 0
    for st, nrow, lab_city in (("test_s", n_s, "Singapore"), ("test_b", n_b, "Boston")):
        ids = np.load(f"{OUT}/e58_B/ids_{st}.npy")
        probs = {h: np.load(f"{OUT}/{h}/prob_{st}.npy", mmap_mode="r") for h in ("e58_B", "e58_S", "e58_M")}
        diff = np.array([float(np.abs(np.asarray(probs["e58_B"][k], np.float32) - np.asarray(probs["e58_S"][k], np.float32)).mean()) for k in range(len(ids))])
        top = np.argsort(-diff)[: nrow * 5]
        sel = [int(x) for x in top[:: 5][:nrow]]  # Bilder, auf denen die beiden Stadt-Koepfe am staerksten abweichen (ueber die Rangliste verteilt)
        gts = DH.load_gt_points("b123", "val", ids[sel])
        for r, k in enumerate(sel):
            for c, (h, lab) in enumerate((("e58_B", "B: trained on Boston"), ("e58_S", "S: trained on Singapore"), ("e58_M", "M: trained on both"))):
                ax = axs[k_row + r, c]
                ax.imshow(np.asarray(probs[h][k], np.float32).T, origin="lower", extent=[-50, 50, -50, 50], cmap="magma", vmin=0, vmax=1)
                ax.plot(gts[r][:, 0], gts[r][:, 1], ".", color="#00e5ff", ms=0.6, alpha=0.6)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0 or True:
                    ax.set_title(f"{lab} - {lab_city} frame", fontsize=7)
        k_row += nrow
    fig.tight_layout()
    p = f"{FIG}/e58_examples.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def main(mode):
    head1 =["head (training rows; tau chosen on VAL)", "F1 on its own training rows", "F1 on TRAIN-eval rows: Boston / Singapore", "VAL Boston", "VAL Singapore", "TEST Boston", "TEST Singapore"]
    head2 = ["head", "set", "F1 in the training city", "F1 in the other city", "gap (other - own)"]
    if mode == "figs":
        for r in table_main():
            print(r)
        for r in table_gap():
            print(r)
        return
    images = []
    for fn, cap in ((bars, "Pixel F1 of the three equal-size heads on the two cities (VAL, TEST): the matched-city score is always the highest; the mixed head is close to the matched ones in both cities."), (binplot, "F1 against the distance of an evaluation frame to its nearest training frame: left - all-data head; right - city heads.")):
        try:
            p = fn()
            if p:
                images.append((p, cap))
        except Exception as ex:  # noqa: BLE001
            print(fn.__name__, "Fehler:", ex)
    bh, br = table_bins()
    try:
        images.append((examples(), "The same TEST frames under the three equal-size heads: the head of the OTHER city misses lanes and invents few; the mixed head is close to the matched head (heat map in colour, ground-truth centerlines in cyan)."))
    except Exception as ex:  # noqa: BLE001
        print("examples Fehler:", ex)
    try:
        p = ftplot()
        if p:
            images.append((p, "Fine-tuning the Boston head on a few Singapore rows: F1 on Singapore against the number of target-domain rows."))
    except Exception as ex:  # noqa: BLE001
        print("ftplot Fehler:", ex)
    tabs = [("Pixel F1 @1 m of the equal-size heads (2,200 training rows each; tau chosen on VAL per head)", head1, table_main()), ("City-transfer gap", head2, table_gap())]
    if table_ft() and len(table_ft()) > 2:
        tabs.append(("E58b - a few target-domain rows: fine-tuning the Boston head on Singapore rows (pixel F1 @1 m)", ["head", "VAL Singapore", "TEST Singapore", "VAL Boston", "TEST Boston"], table_ft()))
    if br:
        tabs.append(("Distance of the evaluation frames to the nearest training frame (Lilja et al. leakage check) and F1 per distance class", bh, br))
    t = TEXT
    make_report2("E58", t.get("title", "E58"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.city_arch, arch_height=3.0)
    print("E58 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
