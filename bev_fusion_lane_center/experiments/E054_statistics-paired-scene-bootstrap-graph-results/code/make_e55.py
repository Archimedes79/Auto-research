"""Bericht E55 (Schleife 4): Seed-Streuung (Dichtekopf-Rezept von E30 mit den Seeds 21 / 22 / 23, Schluesselpunkt-Kopf Seed 0 / 1 + Standard-Schnitt).
Liest out/loop2/{e30_base2, e55_seed22, e55_seed23, e49_dec96, e50_inorm, e52_both}/metrics.json, {e44_nodes, e55_nodes_s1}/metrics.json, {e44_cons_p, e55_cons_s1}/assemble.json; Text loop2/loop4_text_e55.py (TEXT55).  Aufruf (gcar): make_e55.py [figs|report]"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs4 as AF4  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e55.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e55.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT55", {})


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def dense(name):
    d = jl(f"{OUT}/{name}/metrics.json")
    if not d or "val" not in d or "test" not in d:
        return None
    bt = max(d["val"].items(), key=lambda kv: kv[1]["F1@1.0"])[0]
    return dict(tau=bt, tr=d["train_eval"][bt]["F1@1.0"], val=d["val"][bt]["F1@1.0"], test=d["test"][bt]["F1@1.0"], pv=d["val"][bt]["P@1.0"], rv=d["val"][bt]["R@1.0"], seed=d["args"].get("seed"))


def nodes(name):
    d = jl(f"{OUT}/{name}/metrics.json")
    if not d or "val" not in d:
        return None
    bt = max(d["val"].items(), key=lambda kv: kv[1]["6.0"]["split"]["F1"] + kv[1]["6.0"]["merge"]["F1"])[0]
    g = lambda st: (d[st][bt]["6.0"]["split"]["F1"], d[st][bt]["6.0"]["merge"]["F1"])
    return dict(tau=bt, tr=g("train_eval"), val=g("val"), test=g("test"), seed=d["args"].get("seed"))


def main(mode):
    runs = [("e30_base2", "reference (E30)"), ("e55_seed22", "repeat"), ("e55_seed23", "repeat")]
    h1 = ["run", "seed", "TRAIN_EVAL F1@1m", "VAL F1@1m", "TEST F1@1m", "VAL P / R @1m", "tau (VAL)"]
    r1, vals = [], []
    for nm, lab in runs:
        d = dense(nm)
        if d:
            r1.append([f"{lab} ({nm})", str(d["seed"]), f3(d["tr"]), f3(d["val"]), f3(d["test"]), f"{f3(d['pv'])} / {f3(d['rv'])}", d["tau"]])
            vals.append((d["tr"], d["val"], d["test"]))
    if len(vals) >= 2:
        v = np.array(vals)
        r1.append([f"mean of {len(v)} seeds", "-"] + [f3(x) for x in v.mean(0)] + ["-", "-"])
        r1.append([f"standard deviation (n = {len(v)})", "-"] + [f"{x:.4f}" for x in v.std(0, ddof=1)] + ["-", "-"])
        r1.append(["range (max - min)", "-"] + [f"{x:.4f}" for x in (v.max(0) - v.min(0))] + ["-", "-"])
    ens = jl(f"{OUT}/e55_ens3/metrics.json")
    if ens and "val" in ens and "test" in ens:
        bt = max(ens["val"].items(), key=lambda kv: kv[1]["F1@1.0"])[0]
        r1.append(["ensemble: average of the three runs (heat and direction maps)", "21 + 22 + 23", "-", f3(ens["val"][bt]["F1@1.0"]), f3(ens["test"][bt]["F1@1.0"]), f"{f3(ens['val'][bt]['P@1.0'])} / {f3(ens['val'][bt]['R@1.0'])}", bt])
    h2 = ["variant (one seed, 21)", "experiment", "VAL F1@1m", "TEST F1@1m", "difference to the seed mean VAL / TEST", "inside the seed range of the reference?"]
    r2 = []
    if len(vals) >= 2:
        v = np.array(vals)
        lo, hi = v.min(0), v.max(0)
        for nm, ex, lab in (("e49_dec96", "E49", "wider decoder read-out (96 PCA components)"), ("e50_inorm", "E50", "instance normalisation"), ("e52_both", "E52", "GaussianCaR + CRN fusion")):
            d = dense(nm)
            if d:
                inside = (lo[1] <= d["val"] <= hi[1]) and (lo[2] <= d["test"] <= hi[2])
                r2.append([lab, ex, f3(d["val"]), f3(d["test"]), f"{d['val'] - v[:, 1].mean():+.4f} / {d['test'] - v[:, 2].mean():+.4f}", "yes" if inside else "no (VAL " + ("below" if d["val"] < lo[1] else "above" if d["val"] > hi[1] else "inside") + ", TEST " + ("below" if d["test"] < lo[2] else "above" if d["test"] > hi[2] else "inside") + ")"])
    h3 = ["key-point head", "seed", "VAL split / merge F1 @6 m", "TEST split / merge F1 @6 m", "TRAIN_EVAL split / merge", "tau (VAL)"]
    r3 = []
    for nm, lab in (("e44_nodes", "E44"), ("e55_nodes_s1", "repeat")):
        d = nodes(nm)
        if d:
            r3.append([f"{lab} ({nm})", str(d["seed"]), f"{f3(d['val'][0])} / {f3(d['val'][1])}", f"{f3(d['test'][0])} / {f3(d['test'][1])}", f"{f3(d['tr'][0])} / {f3(d['tr'][1])}", d["tau"]])
    h4 = ["graph with default cut + prune", "key-point head seed", "set", "F1@1m", "route F", "split F1 3 / 6 m", "merge F1 3 / 6 m", "split P / R @6 m", "merge P / R @6 m"]
    r4 = []
    for nm, sd in (("e44_cons_p", "0 (E44)"), ("e55_cons_s1", "1 (repeat)")):
        d = jl(f"{OUT}/{nm}/assemble.json")
        if not d:
            continue
        for st in ("val", "test"):
            m = d[st]
            r4.append([f"default node insertion ({nm})", sd, st.upper(), f3(m["F1@1.0"]), f3(m["route_f"]), f"{f3(m['split_F1'])} / {f3(m['split6_F1'])}", f"{f3(m['merge_F1'])} / {f3(m['merge6_F1'])}", f"{f3(m['split6_P'])} / {f3(m['split6_R'])}", f"{f3(m['merge6_P'])} / {f3(m['merge6_R'])}"])
    images = []
    pts = []
    for nm, lab, c in (("e30_base2", "seed 21 (reference)", "#d62728"), ("e55_seed22", "seed 22", "#d62728"), ("e55_seed23", "seed 23", "#d62728"), ("e49_dec96", "E49 wide PCA", "#1f77b4"), ("e50_inorm", "E50 inorm", "#2ca02c"), ("e52_both", "E52 + CRN", "#9467bd")):
        d = dense(nm)
        if d:
            pts.append((lab, d["val"], d["test"], c))
    if pts:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.4))
        for lab, x, y, c in pts:
            ax.scatter(x, y, color=c, s=45)
            ax.annotate(lab, (x, y), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("VAL F1 @1 m")
        ax.set_ylabel("TEST F1 @1 m")
        ax.set_title("same recipe, different seeds (red) against the foundation variants of E49 / E50 / E52", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = f"{FIG}/e55_seeds.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Pixel F1 @1 m of the dense head on VAL and TEST: three seeds of the reference recipe (red) and the single-seed foundation variants."))
    if mode == "figs":
        for r in r1 + r2 + r3 + r4:
            print(r)
        return
    t = TEXT
    tabs = [("Dense head (pixel-based), reference recipe of E30 with three seeds (4,207 training rows, 3,000 steps)", h1, r1)]
    if r2:
        tabs.append(("The single-seed foundation variants against the seed spread of the reference", h2, r2))
    if r3:
        tabs.append(("Key-point head (pixel-based node heat maps), two seeds", h3, r3))
    if r4:
        tabs.append(("Effect of the key-point seed on the graph (vector level; default cut + prune, key-point threshold 0.5 fixed)", h4, r4))
    make_report2("E55", t.get("title", "E55"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF4.seed_arch)
    print("E55 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
