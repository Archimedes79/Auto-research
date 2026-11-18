"""Berichte E49 (breitere Grundlage-Merkmale: dec96) und E50 (Instanz-Normierung) der Schleife 4. Vergleich dichter Koepfe mit gleichem Rezept (4207 Zeilen, 3000 Schritte, Seed 21).
Liest out/loop2/e30_base2, e49_dec96, e50_inorm (metrics.json); Text: loop2/loop4_text_e49.py / loop4_text_e50.py.  Aufruf (gcar): make_e49_e50.py e49|e50 [figs|report]"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs as AF  # noqa: E402
import arch_figs4 as AF4  # noqa: E402
import dense_report_lib as DR  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)


def main(which, mode):
    _ns = {}
    tf = f"/mnt/c/ML-Experiment/bev_fusion_lane_center/scripts/_archive/reports_build/loop4_text_{which}.py"
    if os.path.exists(tf):
        exec(open(tf, encoding="utf-8").read(), _ns)
    TEXT = _ns.get("TEXT" + which[1:], {})
    second = {"e49": ("e49_dec96", "wider decoder read-out: dec96 + logits (103 ch)"), "e50": ("e50_inorm", "instance-normalised inputs (39 ch)"), "e52": ("e52_both", "GaussianCaR (39 ch) + CRN (32 ch) frozen features (71 ch)")}[which]
    runs = [("e30_base2", "reference: dec32 + logits (39 ch), global normalisation"), second]
    head = ["head (4,207 training rows, 3,000 steps, seed 21)", "TRAIN_EVAL F1@1m", "VAL F1@0.5m", "VAL F1@1m", "VAL F1@1.5m", "VAL P / R @1m", "TEST F1@1m", "TEST P / R @1m", "gap TRAIN_EVAL - VAL", "tau (VAL)"]
    rows, res = [], {}
    for nm, lab in runs:
        r = DR.load(nm)
        if not r or "val" not in r:
            continue
        bt = DR.best_tau(r)
        res[nm] = (r, bt)
        v, t, a = r["val"][bt], r["test"][bt], r["train_eval"][bt]
        rows.append([lab, f3(a["F1@1.0"]), f3(v["F1@0.5"]), f3(v["F1@1.0"]), f3(v["F1@1.5"]), f"{f3(v['P@1.0'])} / {f3(v['R@1.0'])}", f3(t["F1@1.0"]), f"{f3(t['P@1.0'])} / {f3(t['R@1.0'])}", f3(a["F1@1.0"] - v["F1@1.0"]), bt])
    images = []
    if len(res) >= 2:
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        DR.plot_pr(axs[0], {k: res[k][0] for k in res}, "val", labels=dict(runs))
        axs[0].set_title("VAL precision-recall at 1 m", fontsize=9)
        DR.plot_pr(axs[1], {k: res[k][0] for k in res}, "test", labels=dict(runs))
        axs[1].set_title("TEST precision-recall at 1 m", fontsize=9)
        fig.tight_layout()
        p = f"{FIG}/{which}_pr.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Precision-recall over the threshold sweep (x = VAL-chosen threshold)."))
    if mode == "figs":
        for r in rows:
            print(r)
        return
    arch = {"e49": AF4.wide_arch, "e52": AF4.fusion_arch}.get(which, lambda ax: AF.dense_arch(ax))
    make_report2(which.upper(), TEXT.get("title", which), TEXT.get("question", ""), TEXT.get("setup", []), [("Dense head comparison (threshold chosen on VAL, applied to TRAIN_EVAL / TEST)", head, rows)], images, TEXT.get("findings", ["(to be written)"]), TEXT.get("limits", []), arch=arch)
    print(which.upper(), "written")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "report")
