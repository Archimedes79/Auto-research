"""Berichte E27-E30 (dichte Laeufe). Aufruf: make_e27_e30.py <e27|e28|e29|e30> [figs|report]
'figs' erzeugt nur Bilder + gibt Zahlen aus (zum Ansehen), 'report' schreibt reports/loop2/E2x.html/.json/.md mit dem Text aus TEXT (unten, von Hand nach Sicht der Zahlen)."""
import json
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
TEXT = json.load(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e27_e30.json")) if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e27_e30.json") else {}

RUNS = {
    "e27": dict(names=["e27_crn", "e27_gcar"], labels={"e27_crn": "CRN (detection backbone)", "e27_gcar": "GaussianCaR (map backbone)"}),
    "e28": dict(names=["e28_dec32", "e28_fused32", "e28_logit7", "e28_dec_fused", "e27_gcar", "e28_all"],
                labels={"e28_dec32": "decoder PCA32", "e28_fused32": "fused PCA32", "e28_logit7": "7 map logits", "e28_dec_fused": "decoder+fused", "e27_gcar": "decoder+logits (E27)", "e28_all": "decoder+fused+logits"}),
    "e29": dict(names=["e27_gcar", "e29_lam3", "e29_lam10", "e29_miss3"],
                labels={"e27_gcar": "symmetric (lam=1, wpos=1)", "e29_lam3": "false alarms x3", "e29_lam10": "false alarms x10", "e29_miss3": "misses x3 (opposite)"}),
    "e30": dict(names=["e30_s32", "e30_s16", "e30_s8", "e30_s4", "e27_gcar", "e30_full"],
                labels={"e30_s32": "263 samples", "e30_s16": "526", "e30_s8": "1052", "e30_s4": "2104", "e27_gcar": "4207 (even rows)", "e30_full": "8414 (all rows)"}),
}


def runs_loaded(g):
    r = {}
    for nm in RUNS[g]["names"]:
        m = DR.load(nm)
        if m is not None and "val" in m:
            r[nm] = m
    return r


def main_table(runs, labels, sets=("train_eval", "val", "test")):
    head = ["run"] + [f"{DR.NICE[s]}: F1@1m" for s in sets] + ["VAL tau", "VAL P@1m", "VAL R@1m", "VAL F1@0.5m", "VAL F1@1.5m", "VAL 1:1 F1@1m", "VAL F0.5@1m", "TEST P@1m", "TEST R@1m"]
    rows = []
    for nm, res in runs.items():
        bt = DR.best_tau(res)
        v = res["val"][bt]
        t = res["test"][bt] if "test" in res else None
        rows.append([labels[nm]] + [f3(res[s][bt]["F1@1.0"]) if s in res else "-" for s in sets] + [bt, f3(v["P@1.0"]), f3(v["R@1.0"]), f3(v["F1@0.5"]), f3(v["F1@1.5"]), f3(v["F1o2o@1.0"]), f3(v["F0.5@1.0"]), f3(t["P@1.0"]) if t else "-", f3(t["R@1.0"]) if t else "-"])
    return head, rows


def interp_pr(res, st, key_in, val, key_out):
    """Entlang der Schwellwert-Kurve (R, P bei 1 m) linear interpolieren: z.B. Praezision bei Ausbeute 0.8 oder Ausbeute bei Praezision 0.9 (nan, falls nicht erreicht)."""
    taus = sorted(res[st].keys(), key=float)
    xs = np.array([res[st][t][key_in] for t in taus])
    ys = np.array([res[st][t][key_out] for t in taus])
    best = float("nan")
    for a_ in range(len(taus) - 1):
        x0, x1, y0, y1 = xs[a_], xs[a_ + 1], ys[a_], ys[a_ + 1]
        if (x0 - val) * (x1 - val) <= 0 and x0 != x1:
            v = y0 + (y1 - y0) * (val - x0) / (x1 - x0)
            best = v if np.isnan(best) else max(best, v)
    return best


def op_table(runs, labels):
    head = ["run", "VAL P@1m (tau 0.5)", "VAL R@1m (tau 0.5)", "VAL F0.5@1m (tau 0.5)", "VAL P at R=0.80", "VAL R at P=0.90", "TEST P at R=0.75", "TEST R at P=0.90", "VAL false-alarm share of skeleton points (1-P)"]
    rows = []
    for nm, res in runs.items():
        v = res["val"].get("0.5")
        rows.append([labels[nm], f3(v["P@1.0"]), f3(v["R@1.0"]), f3(v["F0.5@1.0"]), f3(interp_pr(res, "val", "R@1.0", 0.80, "P@1.0")), f3(interp_pr(res, "val", "P@1.0", 0.90, "R@1.0")),
                     f3(interp_pr(res, "test", "R@1.0", 0.75, "P@1.0")) if "test" in res else "-", f3(interp_pr(res, "test", "P@1.0", 0.90, "R@1.0")) if "test" in res else "-", f3(1 - v["P@1.0"])])
    return head, rows


def figs(g):
    runs = runs_loaded(g)
    names = list(runs)
    lab = RUNS[g]["labels"]
    out = {}
    # PR-Kurven (VAL und TRAIN_EVAL)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    DR.plot_pr(axs[0], runs, "val", lab)
    DR.plot_pr(axs[1], runs, "train_eval", lab)
    fig.tight_layout()
    out["pr"] = f"{FIG}/{g}_pr.png"
    fig.savefig(out["pr"], dpi=100)
    plt.close(fig)
    # Balken: F1@1m je Menge
    fig, ax = plt.subplots(figsize=(11, 4))
    groups = [lab[n] for n in names]
    series = {}
    for s in ("train_eval", "val", "test"):
        vals = []
        for nm in names:
            r = runs[nm]
            bt = DR.best_tau(r)
            vals.append(r[s][bt]["F1@1.0"] if s in r else 0)
        series[DR.NICE[s]] = vals
    DR.bars(ax, groups, series, "F1 @ 1 m (threshold chosen on VAL)", f"{g.upper()}: train vs val vs test")
    fig.tight_layout()
    out["bars"] = f"{FIG}/{g}_bars.png"
    fig.savefig(out["bars"], dpi=100)
    plt.close(fig)
    # Lernkurven
    fig, ax = plt.subplots(figsize=(7, 3.6))
    DR.curves(ax, runs, lab)
    fig.tight_layout()
    out["curves"] = f"{FIG}/{g}_curves.png"
    fig.savefig(out["curves"], dpi=100)
    plt.close(fig)
    # qualitative (VAL, 4 Samples)
    if all(os.path.exists(f"{DR.OUT}/{n}/prob_val.npy") for n in names):
        ids = np.load(f"{DR.OUT}/{names[0]}/ids_val.npy")
        sel = [int(x) for x in np.linspace(40, len(ids) - 40, 4)]
        taus = {n: float(DR.best_tau(runs[n])) for n in names}
        out["qual"] = DR.qualitative(names[:4] if g == "e28" else names, lab, "val", sel, taus, f"{FIG}/{g}_qual.png")
    return runs, out


def report(g):
    runs, im = figs(g)
    lab = RUNS[g]["labels"]
    head, rows = main_table(runs, lab)
    fo = DR.load("foundation_only") if g == "e27" else None
    if fo is not None and "val" in fo:
        bt = DR.best_tau(fo)
        rows.append(["GaussianCaR 'lane' layer skeleton, no trained head (zero-shot)"] + [f3(fo[s_][bt]["F1@1.0"]) if s_ in fo else "-" for s_ in ("train_eval", "val", "test")] + [bt, f3(fo["val"][bt]["P@1.0"]), f3(fo["val"][bt]["R@1.0"]), f3(fo["val"][bt]["F1@0.5"]), f3(fo["val"][bt]["F1@1.5"]), f3(fo["val"][bt]["F1o2o@1.0"]), f3(fo["val"][bt]["F0.5@1.0"]), f3(fo["test"][bt]["P@1.0"]), f3(fo["test"][bt]["R@1.0"])])
    t = TEXT.get(g, {})
    arch = lambda ax: AF.dense_arch(ax)
    images = [(im[k], c) for k, c in (("qual", "VAL samples: heat map (colour), GT centerlines (thin line), skeleton at the VAL-chosen threshold (red dots)."),
                                       ("pr", "Precision-recall at 1 m over the threshold sweep (x = VAL-chosen threshold)."),
                                       ("bars", "F1@1 m on the training subset, VAL and TEST."),
                                       ("curves", "Training loss.")) if k in im]
    oh, orow = op_table(runs, lab)
    make_report2(t.get("id", g.upper()), t.get("title", g.upper()), t.get("question", ""), t.get("setup", []), [("Main results (threshold chosen on VAL, applied to all sets)", head, rows), ("Operating points along the threshold sweep", oh, orow)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=arch)
    print("report", g, "written")


if __name__ == "__main__":
    g, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "figs")
    if mode == "report":
        report(g)
    else:
        runs, im = figs(g)
        head, rows = main_table(runs, RUNS[g]["labels"])
        print("\t".join(head))
        for r in rows:
            print("\t".join(r))
        print(im)
