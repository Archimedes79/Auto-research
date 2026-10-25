"""Bericht E37: dichter Endkopf (alle 8414 Zeilen), Kreuz-Anpassungen (Szenen-Folds), Mittel der Koepfe. Aufruf (bev): make_e37.py [figs|report]
Laeufe: e27_gcar (Basis, 4207 Zeilen), e37_final, e37_fold0..2, e37_ens (make_ensemble.py). Text: loop2/text_e37_e42.py TEXT['e37']."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs as AF  # noqa: E402
import dense_report_lib as DR  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

FIG = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT", {})
LAB = {"e27_gcar": "E27 baseline (4207 rows, 4000 steps)", "e37_final": "E37 final head (8,414 rows, 6,000 steps, stabilised recipe)", "e37_ens": "mean of final + 3 fold heads", "e37_fold0": "fold head 0 (2/3 of scenes, 4,000 steps)",
       "e37_fold1": "fold head 1", "e37_fold2": "fold head 2"}


def vrow(res, st, tau, k="F1@1.0"):
    r = DR.row_at(res, st, tau)
    return r[k] if r else None


def main(mode):
    runs = {n: DR.load(n) for n in ("e27_gcar", "e37_final", "e37_ens", "e37_fold0", "e37_fold1", "e37_fold2")}
    runs = {k: v for k, v in runs.items() if v is not None}
    # Tabelle 1: Endkopf gegen Basis
    head = ["head", "TRAIN_EVAL F1@1m", "VAL F1@0.5m", "VAL F1@1m", "VAL F1@1.5m", "VAL P@1m", "VAL R@1m", "TEST F1@0.5m", "TEST F1@1m", "TEST F1@1.5m", "TEST P@1m", "TEST R@1m", "tau (VAL)"]
    rows = []
    for nm in ("e27_gcar", "e37_final", "e37_ens"):
        if nm not in runs:
            continue
        res = runs[nm]
        bt = DR.best_tau(res)
        te = lambda k: f3(vrow(res, "test", bt, k))
        va = lambda k: f3(vrow(res, "val", bt, k))
        tr = f3(vrow(res, "train_eval", bt)) if "train_eval" in res else "-"
        rows.append([LAB[nm], tr, va("F1@0.5"), va("F1@1.0"), va("F1@1.5"), va("P@1.0"), va("R@1.0"), te("F1@0.5"), te("F1@1.0"), te("F1@1.5"), te("P@1.0"), te("R@1.0"), bt])
    # Tabelle 2: Kreuz-Anpassungen
    h2 = ["fold head", "training scenes", "OOF (held-out TRAIN scenes) F1@1m", "OOF P / R @1m", "VAL F1@1m", "TEST F1@1m", "tau (OOF-independent: VAL)"]
    r2 = []
    for k in range(3):
        nm = f"e37_fold{k}"
        if nm not in runs or "oof" not in runs[nm]:
            continue
        res = runs[nm]
        bt = DR.best_tau(res)
        o = res["oof"][bt]
        r2.append([nm, "2/3 of the 212 TRAIN scenes", f3(o["F1@1.0"]), f"{f3(o['P@1.0'])} / {f3(o['R@1.0'])}", f3(vrow(res, "val", bt)), f3(vrow(res, "test", bt)), bt])
    if r2:
        r2.append(["mean", "", f3(np.mean([float(r[2]) for r in r2])), "", f3(np.mean([float(r[4]) for r in r2])), f3(np.mean([float(r[5]) for r in r2])), ""])
    # Tabelle 3: Arbeitspunkt des Endkopfes
    h3 = ["tau", "VAL P@1m", "VAL R@1m", "VAL F1@1m", "VAL F0.5@1m (precision-weighted)", "TEST P@1m", "TEST R@1m", "TEST F1@1m"]
    r3 = []
    if "e37_final" in runs:
        res = runs["e37_final"]
        for tau in sorted(res["val"].keys(), key=float):
            v, t = res["val"][tau], res["test"][tau]
            r3.append([tau, f3(v["P@1.0"]), f3(v["R@1.0"]), f3(v["F1@1.0"]), f3(v["F0.5@1.0"]), f3(t["P@1.0"]), f3(t["R@1.0"]), f3(t["F1@1.0"])])
    images = []
    use = {n: r for n, r in runs.items() if n in ("e27_gcar", "e37_final", "e37_ens")}
    if use:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
        for ax, st in zip(axs, ("val", "test")):
            DR.plot_pr(ax, use, st, labels=LAB)
        fig.tight_layout()
        p = f"{FIG}/e37_pr.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Precision-recall over thresholds (1 m tolerance); x marks the threshold chosen on VAL."))
        # Lernkurven der Anpassung
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        DR.curves(ax, {n: r for n, r in runs.items() if "curve" in r}, LAB)
        fig.tight_layout()
        p = f"{FIG}/e37_curves.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Training loss of the final head and the fold heads."))
    if "e37_final" in runs and "e27_gcar" in runs:
        pr, ids = DR.load_prob("e37_final", "val")
        sel = list(np.linspace(5, len(ids) - 6, 6).astype(int))
        tau = {n: float(DR.best_tau(runs[n])) for n in ("e27_gcar", "e37_final")}
        p = DR.qualitative(["e27_gcar", "e37_final"], {"e27_gcar": "E27 baseline head", "e37_final": "E37 final head"}, "val", sel, tau, f"{FIG}/e37_qual_val.png")
        images.append((p, "VAL examples (same samples): E27 baseline vs final head; white lines = ground-truth centerlines, red dots = skeleton of the thresholded heat map."))
        pr, ids = DR.load_prob("e37_final", "test")
        sel = list(np.linspace(5, len(ids) - 6, 6).astype(int))
        p = DR.qualitative(["e27_gcar", "e37_final"], {"e27_gcar": "E27 baseline head", "e37_final": "E37 final head"}, "test", sel, tau, f"{FIG}/e37_qual_test.png")
        images.append((p, "TEST examples (never used for any decision)."))
    if mode == "figs":
        for r in rows + r2 + r3:
            print(r)
        return
    t = TEXT.get("e37", {})
    make_report2("E37", t.get("title", "E37"), t.get("question", ""), t.get("setup", []),
                 [("Final dense head versus the E27 baseline (threshold chosen on VAL, applied to TRAIN_EVAL / VAL / TEST)", head, rows),
                  ("Cross-fitted fold heads (used to produce out-of-fold dense predictions for training the sparse stage)", h2, r2),
                  ("Operating points of the final head (threshold sweep)", h3, r3)],
                 images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=lambda ax: AF.dense_arch(ax))
    print("E37 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
