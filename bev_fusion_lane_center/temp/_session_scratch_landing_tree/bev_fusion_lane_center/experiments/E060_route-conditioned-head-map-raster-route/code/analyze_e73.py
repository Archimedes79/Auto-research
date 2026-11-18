"""E73 analysis: collects the DEV (and TEST) results of all fine-tuning runs, prints a table, writes report_final/tables/e73_dev.json|md and the figures.
Usage (gcar): analyze_e73.py [--figs 1]"""
import argparse
import glob
import json
import os

import numpy as np

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
RF = "/mnt/c/ML-Experiment/bev_fusion_lane_center/reports/source"
PERTS = ["dark", "cam_front_off", "fog", "cam_front3_off", "blur", "extr3", "radar_off"]
SEEN3 = ["dark", "cam_front_off", "fog"]

DESC = {
    "a0_e60": ("anchor", "E60: clean-trained route head (start of all fine-tuning)"),
    "a0_e60b": ("anchor", "E60: clean-trained route head (start of all fine-tuning)"),
    "a1_e65d": ("anchor", "E65d: published recipe (original run)"),
    "r0_s0": ("1 data", "R0 = E65d recipe re-run, seed 0 (3 perturbation types)"),
    "r0_s1": ("1 data", "R0, seed 1"),
    "r0_s2": ("1 data", "R0, seed 2"),
    "clean": ("1 data", "control: 1,500 more steps on clean features only"),
    "p7": ("1 data", "all 7 perturbation types"),
    "p7_m75": ("1 data", "7 types, perturbed share of the draws 0.75 (default 0.55)"),
    "p7_m35": ("1 data", "7 types, perturbed share 0.35"),
    "p7_lr4": ("2 optimiser", "7 types, lr 4e-4 (default 2e-4)"),
    "p7_lr1": ("2 optimiser", "7 types, lr 1e-4"),
    "p7_st3000": ("2 optimiser", "7 types, 3,000 steps (default 1,500)"),
    "p7_st750": ("2 optimiser", "7 types, 750 steps"),
    "p7_stem": ("3 layers", "7 types, train first block only (+ output head)"),
    "p7_dec": ("3 layers", "7 types, train decoder only (+ output head)"),
    "p7_bn": ("3 layers", "7 types, train BatchNorm affine parameters only (+ head)"),
    "p7_l2sp": ("4 regularisation", "7 types, L2-SP penalty 0.01"),
    "p7_ema": ("4 regularisation", "7 types, EMA 0.995: final weights"),
    "p7_ema__ema": ("4 regularisation", "7 types, EMA 0.995: EMA weights"),
    "p7_wr2": ("4 loss", "7 types, route loss weight 2"),
    "p7_lam2": ("4 loss", "7 types, false-alarm weight 2"),
    "p7_wrong25": ("4 loss", "7 types, wrong map 25 %, no map 30 % in training"),
    "p7_trust": ("5 trust", "7 types, map-trust head trained jointly"),
    "trustprobe_e65d": ("5 trust", "map-trust head fitted on the frozen E65d (probe)"),
    "p7_trust_s1": ("5 trust", "7 types, joint trust head, seed 1"),
    "p7_trust_w1": ("5 trust", "7 types, joint trust head, weight 1.0 (default 0.3)"),
    "p7_trust_w01": ("5 trust", "7 types, joint trust head, weight 0.1"),
    "p7_trust_w3": ("5 trust", "7 types, joint trust head, weight 3.0"),
    "p7_trust_w1_s1": ("5 trust", "7 types, joint trust head, weight 1.0, seed 1"),
    "p7_trust_w1_s2": ("5 trust", "7 types, joint trust head, weight 1.0, seed 2"),
    "soup_trust3": ("6 weights", "average of the three joint-trust runs (soup)"),
    "soup_all5": ("6 weights", "average of the five runs with trust weight >= 0.3"),
    "final_v2": ("6 weights", "first final candidate: average of the three weight-1.0 seeds (dropped: night guard)"),
    "soup_all6": ("7 final", "FINAL v2: average of all six trust runs (guard satisfied)"),
    "soup_r0": ("6 weights", "average of the three R0 seeds (soup)"),
    "wise50_trust": ("6 weights", "interpolation 0.5 E60 + 0.5 joint-trust run (WiSE-FT)"),
    "wise75_trust": ("6 weights", "interpolation 0.25 E60 + 0.75 joint-trust run"),
}


def load_runs():
    runs = {}
    for d in sorted(glob.glob(f"{OUT}/e73_*")):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)[4:]
        for fn, tag in (("result.json", ""), ("result_eval.json", "")):
            p = f"{d}/{fn}"
            if os.path.exists(p):
                r = json.load(open(p))
                for k, v in r.items():
                    if k.startswith("eval"):
                        nm = name + k[len("eval"):]
                        runs[nm] = dict(res=v, args=r.get("args", {}), train_s=r.get("train_s"))
    return runs


def row(nm, r):
    d = r["res"].get("dev")
    if not d:
        return None
    s = d["score"]
    c = d["clean"]
    seen = np.mean([d[p]["exact"]["lane_F1"] for p in SEEN3 if p in d])
    unseen = np.mean([d[p]["exact"]["lane_F1"] for p in PERTS if p not in SEEN3 and p in d]) if any(p in d for p in PERTS if p not in SEEN3) else float("nan")
    return dict(name=nm, stage=DESC.get(nm, ("?", nm))[0], desc=DESC.get(nm, ("?", nm))[1], score=s["score"], clean_exact_lane=s["clean_exact_lane"], clean_exact_route=s["clean_exact_route"], clean_s1_route=s["clean_s1_route"],
                clean_wrong_lane=s["clean_wrong_lane"], clean_none_lane=s["clean_none_lane"], pert_exact_lane=s["pert_exact_lane"], pert_exact_route=s["pert_exact_route"], pert_wrong_lane=s["pert_wrong_lane"],
                seen_lane=float(seen), unseen_lane=float(unseen), trust_auc=c.get("_trust_auc"), trust_auc_ew=c.get("_trust_auc_exact_vs_wrong"), train_s=r.get("train_s"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", type=int, default=0)
    a = ap.parse_args()
    runs = load_runs()
    rows = [x for x in (row(n, r) for n, r in runs.items()) if x]
    order = list(DESC.keys())
    rows.sort(key=lambda x: order.index(x["name"]) if x["name"] in order else 999)
    os.makedirs(f"{RF}/tables", exist_ok=True)
    json.dump(rows, open(f"{RF}/tables/e73_dev.json", "w"), indent=1)
    hdr = ["run", "stage", "change", "score", "clean lane", "clean route", "s1 route", "wrong lane", "none lane", "pert lane", "pert route", "pert wrong lane", "seen/unseen types lane", "trust AUC"]
    print(" | ".join(hdr))
    for x in rows:
        print(f"{x['name']:16s} {x['stage']:16s} {x['score']:.4f} | {x['clean_exact_lane']:.3f} {x['clean_exact_route']:.3f} {x['clean_s1_route']:.3f} {x['clean_wrong_lane']:.3f} {x['clean_none_lane']:.3f} | {x['pert_exact_lane']:.3f} {x['pert_exact_route']:.3f} {x['pert_wrong_lane']:.3f} | seen {x['seen_lane']:.3f} unseen {x['unseen_lane']:.3f} | trust {x['trust_auc'] if x['trust_auc'] is None else round(x['trust_auc'], 3)} | {x['desc']}")
    seeds = [x["score"] for x in rows if x["name"] in ("r0_s0", "r0_s1", "r0_s2")]
    if len(seeds) >= 2:
        print(f"seed noise of R0 (n={len(seeds)}): mean {np.mean(seeds):.4f} sd {np.std(seeds, ddof=1):.4f}")


if __name__ == "__main__":
    main()
