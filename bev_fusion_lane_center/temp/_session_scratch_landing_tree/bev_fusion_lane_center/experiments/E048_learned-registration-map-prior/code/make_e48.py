"""Bericht E48 (Schleife 4): RAVE-artige gelernte Registrierung des Karten-Vorwissens.
Liest out/loop2/e48_align/align_eval.json (Grobstufe), e48_fine/align_eval_chain.json (Grob + Fein), e48_chain2/align_eval_chain.json (Korrelation + Fein) und die Auswertungen des Vorwissens-Kopfes e48_clean_* / e40_clean_*.
Text: loop2/loop4_text_e48.py (TEXT48).  Aufruf (gcar): make_e48.py [figs|report]"""
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
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e48.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e48.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT48", {})


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def best(name):
    d = jl(f"{OUT}/{name}/metrics.json")
    if not d or "val" not in d:
        return None
    bt = max(d["val"].items(), key=lambda kv: kv[1]["F1@1.0"])[0]
    r = dict(val=d["val"][bt]["F1@1.0"], test=d["test"][bt]["F1@1.0"], pv=d["val"][bt]["P@1.0"], rv=d["val"][bt]["R@1.0"])
    if "val_align" in d:
        r["err_val"], r["err_before"] = d["val_align"]["pos_err_median"], d["val_align"]["pos_err_before_median"]
    return r


def main(mode):
    c1 = jl(f"{OUT}/e48_align/align_eval.json")
    c2 = jl(f"{OUT}/e48_fine/align_eval_chain.json")
    c3 = jl(f"{OUT}/e48_chain2/align_eval_chain.json")
    sig = ["0.5", "1.0", "2.0", "3.0"]
    h1 = ["pose noise sigma (m; yaw 2 sigma deg)", "error before (median, m)", "method", "median position error (m)", "within 0.2 m", "within 0.5 m", "within 1 m", "median yaw error (deg)", "ms / sample"]
    r1 = []
    for s in sig:
        for src, key, lab in ((c1, "correlation", "correlation search (E40)"), (c1, "learned", "learned coarse stage (E48)"), (c2, "learned", "learned coarse + fine stage (E48)"), (c3, "learned", "correlation search + learned fine stage (E48)")):
            if src and s in src:
                r = src[s][key]
                r1.append([s, f3(src[s]["before"]["pos_median"]), lab, f"{r['pos_median']:.2f}", f3(r.get("within02")) if "within02" in r else "-", f3(r["within05"]), f3(r["within1"]), f"{r['yaw_median']:.2f}", f"{src[s]['ms_correlation' if key == 'correlation' else 'ms_learned']:.0f}"])
    h2 = ["pose noise sigma (m)", "registration", "dense F1@1 m VAL", "dense F1@1 m TEST", "P / R @1 m (VAL)", "median error after (m, VAL)"]
    r2 = []
    names = [("0.5", "no registration (E40)", "e40_clean_s05_noalign"), ("0.5", "correlation search (E40)", "e40_clean_s05_align"), ("0.5", "learned coarse stage", "e48_clean_s05_lalign"), ("0.5", "correlation + learned fine stage", "e48_clean_s05_corrfine"),
             ("1", "no registration (E40)", "e40_clean_s1_noalign"), ("1", "correlation search (E40)", "e40_clean_s1_align"), ("1", "learned coarse stage", "e48_clean_s1_lalign"), ("1", "correlation + learned fine stage", "e48_clean_s1_corrfine"), ("1", "learned coarse + fine stage", "e48_clean_s1_chain"),
             ("2", "correlation search", "e48_clean_s2_corr"), ("2", "learned coarse stage", "e48_clean_s2_lalign"), ("2", "correlation + learned fine stage", "e48_clean_s2_corrfine"), ("2", "learned coarse + fine stage", "e48_clean_s2_chain"),
             ("3", "correlation search", "e48_clean_s3_corr"), ("3", "learned coarse stage", "e48_clean_s3_lalign"), ("3", "correlation + learned fine stage", "e48_clean_s3_corrfine")]
    for s, lab, nm in names:
        b = best(nm)
        if b:
            r2.append([s, lab, f3(b["val"]), f3(b["test"]), f"{f3(b['pv'])} / {f3(b['rv'])}", f"{b['err_val']:.2f}" if "err_val" in b else "-"])
    for lab, nm in (("exact pose (reference)", "e40_clean"), ("no map (control)", "e40_nomap_ft")):
        b = best(nm)
        if b:
            r2.append(["-", lab, f3(b["val"]), f3(b["test"]), f"{f3(b['pv'])} / {f3(b['rv'])}", "-"])
    h3 = ["pose noise sigma (m)", "registration", "VAL F1@1m", "VAL route F", "TEST F1@1m", "TEST route F", "TEST split / merge F1 6 m"]
    r3 = []
    for s_, lab, nm in (("1", "correlation search (E40)", "sys_e40_clean_s1_align"), ("1", "correlation + learned fine stage", "sys_e48_clean_s1_corrfine"), ("2", "correlation search", "sys_e48_clean_s2_corr"), ("2", "correlation + learned fine stage", "sys_e48_clean_s2_corrfine"), ("0", "exact pose (reference)", "sys_e40_clean"), ("-", "no map (reference)", "sys_e40_nomap_ft")):
        sj = jl(f"{OUT}/{nm}/metrics.json")
        if sj and "val" in sj:
            v, t = sj["val"]["refined"], sj["test"]["refined"]
            r3.append([s_, lab, f3(v["F1@1.0"]), f3(v["route_f"]), f3(t["F1@1.0"]), f3(t["route_f"]), f"{f3(t['split6_F1'])} / {f3(t['merge6_F1'])}"])
    images = []
    if c1:
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
        xs = [float(s) for s in sig if c1 and s in c1]
        series = [("before", [c1[s]["before"]["pos_median"] for s in sig if s in c1], "#7f7f7f"), ("correlation search (E40)", [c1[s]["correlation"]["pos_median"] for s in sig if s in c1], "#1f77b4"), ("learned coarse", [c1[s]["learned"]["pos_median"] for s in sig if s in c1], "#ff7f0e")]
        if c2:
            series.append(("learned coarse + fine", [c2[s]["learned"]["pos_median"] for s in sig if s in c2], "#2ca02c"))
        if c3:
            series.append(("correlation + learned fine", [c3[s]["learned"]["pos_median"] for s in sig if s in c3], "#d62728"))
        for lab, ys, c in series:
            axs[0].plot(xs[: len(ys)], ys, "-o", label=lab, color=c)
        axs[0].set_xlabel("pose noise sigma (m)")
        axs[0].set_ylabel("median position error after registration (m)")
        axs[0].set_title("registration accuracy (VAL, 300 samples)", fontsize=9)
        axs[0].grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        axs[0].set_yscale("log")
        for lab, nmf, c in (("no registration", lambda s: f"e40_clean_s{s}_noalign", "#7f7f7f"), ("correlation search", lambda s: f"e48_clean_s{s}_corr" if s in ("2", "3") else f"e40_clean_s{s}_align", "#1f77b4"), ("learned coarse", lambda s: f"e48_clean_s{s}_lalign", "#ff7f0e"), ("correlation + learned fine", lambda s: f"e48_clean_s{s}_corrfine", "#d62728")):
            ys, xx = [], []
            for s_, sv in (("05", 0.5), ("1", 1.0), ("2", 2.0), ("3", 3.0)):
                b = best(nmf(s_))
                if b:
                    ys.append(b["val"]), xx.append(sv)
            if ys:
                axs[1].plot(xx, ys, "-o", label=lab, color=c)
        b = best("e40_clean")
        if b:
            axs[1].axhline(b["val"], color="k", ls=":", label="exact pose")
        b = best("e40_nomap_ft")
        if b:
            axs[1].axhline(b["val"], color="g", ls=":", label="no map")
        axs[1].set_xlabel("pose noise sigma (m)")
        axs[1].set_ylabel("dense head F1@1 m (VAL)")
        axs[1].set_title("effect on the map-prior dense head (exact-pose trained)", fontsize=9)
        axs[1].grid(alpha=0.3)
        axs[1].legend(fontsize=7)
        fig.tight_layout()
        p = f"{FIG}/e48_align.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Left: median position error of the registration against the pose noise (log scale). Right: F1 of the exact-pose map-prior head after registration (VAL)."))
    if mode == "figs":
        for r in r1 + r2:
            print(r)
        return
    t = TEXT
    make_report2("E48", t.get("title", "E48"), t.get("question", ""), t.get("setup", []), [("Registration accuracy (VAL, 300 samples, the same noise draws for every method)", h1, r1), ("Effect on the map-prior dense head (trained with the exact pose; evaluation under pose noise)", h2, r2), ("System level: proposals + refiner e39_main + rule links on the dense heat maps of the registered prior", h3, r3)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF4.align_arch)
    print("E48 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
