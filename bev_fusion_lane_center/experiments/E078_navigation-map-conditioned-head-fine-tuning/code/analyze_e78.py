"""Loop 7 / E78: DEV table + figure of the SD-head fine-tuning study -> report_final/tables/tab_e78_dev.json, figures/e78_dev_scores.png
Usage (gcar): analyze_e78.py"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
RF = "/mnt/c/ML-Experiment/bev_fusion_lane_center/reports/source"
RUNS = [
    ("sd_a", "reference S0: init HD head v2 + zero-init SD channels, 7 perturbation types, joint trust 1.0, 1500 steps, lr 2e-4, map error sigma 0.3-2 m (30 % exact)"),
    ("sd_s1", "S0, seed 1"),
    ("sd_s2", "S0, seed 2"),
    ("sd_nocnt", "lane-count channel set to zero"),
    ("sd_init_e60", "init from E60 (clean-trained HD route head, no trust head)"),
    ("sd_init_e65d", "init from E65d (HD route head, 3 perturbation types)"),
    ("sd_steps3000", "S0 with 3,000 steps"),
    ("sd_s3k_s1", "S0 with 3,000 steps, seed 1"),
    ("sd_steps6000", "S0 with 6,000 steps"),
    ("sd_lr1e3", "learning rate 1e-3"),
    ("sd_pex05", "50 % exact SD maps in training"),
    ("sd_sig3", "map error up to 3 m in training"),
    ("sd_qpoor", "poor SD quality in training (shape tolerance 1-4 m, 25 % lane-count errors)"),
    ("sd_notrust", "no trust head (1,500 steps)"),
    ("sd_clean", "clean features only (no perturbation data)"),
    ("soup2", "weight average of the two 3,000-step runs"),
    ("soup3", "weight average of the two 3,000-step and the 6,000-step run (with trust head)"),
    ("sd_nt3k", "no trust head, 3,000 steps"),
    ("sd_nt3k_s1", "no trust head, 3,000 steps, seed 1"),
    ("soup_nt", "FINAL SD head = weight average of the two 3,000-step runs without trust head"),
]


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def load(name, base="e78_"):
    for fn in ("result.json", "result_eval.json"):
        p = f"{OUT}/{base}{name}/{fn}"
        if os.path.exists(p):
            r = json.load(open(p))
            ev = next((v for k, v in r.items() if k.startswith("eval")), None)
            return ev["dev"]["score"] if ev and "dev" in ev else None
    return None


def main():
    rows = []
    for n, d in RUNS:
        s = load(n)
        if s:
            rows.append((n, d, s))
    ref = float(np.mean([s["score"] for n, d, s in rows if n in ("sd_a", "sd_s1", "sd_s2")]))
    sd = float(np.std([s["score"] for n, d, s in rows if n in ("sd_a", "sd_s1", "sd_s2")], ddof=1)) if sum(n in ("sd_a", "sd_s1", "sd_s2") for n, _, _ in rows) > 1 else float("nan")
    head = ["run", "what changes", "DEV score", "vs S0 mean", "clean: lane F1 / route", "1 m map error: route", "wrong map: lane F1", "no map: lane F1", "7 perturbations: lane F1 / route", "perturbed + wrong map: lane F1"]
    out = []
    for n, d, s in rows:
        out.append([n, d, f3(s["score"]), f"{s['score'] - ref:+.4f}", f"{f3(s['clean_exact_lane'])} / {f3(s['clean_exact_route'])}", f3(s["clean_s1_route"]), f3(s["clean_wrong_lane"]), f3(s["clean_none_lane"]), f"{f3(s['pert_exact_lane'])} / {f3(s['pert_exact_route'])}", f3(s["pert_wrong_lane"])])
    # HD references
    for n, d in (("soup_all6", "HD-map head v2 (exact HD map instead of the SD map; reference only)"),):
        s = load(n, "e73_")
        if s:
            out.append(["v2 (HD map)", d, f3(s["score"]), "-", f"{f3(s['clean_exact_lane'])} / {f3(s['clean_exact_route'])}", f3(s["clean_s1_route"]), f3(s["clean_wrong_lane"]), f3(s["clean_none_lane"]), f"{f3(s['pert_exact_lane'])} / {f3(s['pert_exact_route'])}", f3(s["pert_wrong_lane"])])
    os.makedirs(f"{RF}/tables", exist_ok=True)
    json.dump(dict(caption=f"E78: DEV results of the SD-head fine-tuning study (VAL scenes; SD map without pose error unless stated; 's1' = map error sigma 1 m per axis; mean of S0 {ref:.4f}, seed SD {sd:.4f})", head=head, rows=out), open(f"{RF}/tables/tab_e78_dev.json", "w"), indent=1)
    print("tab_e78_dev", len(out), "rows; S0 mean", ref, "seed SD", sd)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    names = [n for n, d, s in rows]
    sc = [s["score"] for n, d, s in rows]
    cols = ["#1a73e8" if n in ("sd_a", "sd_s1", "sd_s2") else "#9aa0a6" for n in names]
    ax.barh(range(len(names)), sc, color=cols)
    ax.axvspan(ref - 2 * sd if sd == sd else ref, ref + 2 * sd if sd == sd else ref, color="#fbbc04", alpha=0.35, label="+-2 seed SD around the S0 mean")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(min(sc) - 0.03, max(sc) + 0.03)
    ax.set_xlabel("DEV score (mean of 8 numbers, VAL scenes)")
    ax.legend(fontsize=8)
    ax.set_title("E78: SD-head fine-tuning study, DEV score per run")
    fig.tight_layout()
    fig.savefig(f"{RF}/figures/e78_dev_scores.png", dpi=130)


if __name__ == "__main__":
    main()
