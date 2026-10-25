"""Berichte E31 (Raster-Ansatz -> Graph) und E33 (Punktabstand des Spurverfolgers). Aufruf: make_e31_e33.py <dichter Lauf> [figs|report]
Liest out/loop2/<lauf>/graph_grid.json (sweep_trace.py), graph_spacing.json (sweep_spacing.py), out/loop2/oracle_grid.json; Text aus loop2/text_e31_e36.json."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs as AF  # noqa: E402
import graph_from_dense as GD  # noqa: E402
import viz_graph as VG  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
TEXT = json.load(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/scripts/_archive/reports_build/text_e31_e36.json")) if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/scripts/_archive/reports_build/text_e31_e36.json") else {}
COLS = [("F1@0.5", "F1@0.5m"), ("F1@1.0", "F1@1m"), ("F1@1.5", "F1@1.5m"), ("P@1.0", "P@1m"), ("R@1.0", "R@1m"), ("F1o2o@1.0", "1:1 F1@1m"), ("lanes_found", "lanes found"), ("false_len_frac", "false length"),
        ("split_F1", "split F1 3m"), ("split6_F1", "split F1 6m"), ("merge_F1", "merge F1 3m"), ("merge6_F1", "merge F1 6m"), ("route_rec", "route rec"), ("route_prec", "route prec"), ("route_f", "route F"), ("dir_acc", "dir acc")]
NICE = {"train_eval": "TRAIN (eval subset)", "val": "VAL", "test": "TEST"}


def rows_for(variants, sets=("val", "test", "train_eval")):
    rows = []
    for st in sets:
        for lab, d in variants:
            m = d.get(st)
            if m is None:
                continue
            rows.append([NICE[st], lab] + [f3(m.get(k)) for k, _ in COLS])
    return ["set", "variant"] + [h for _, h in COLS], rows


def figs(name):
    d = f"{OUT}/{name}"
    g = json.load(open(f"{d}/graph_grid.json"))
    orc = json.load(open(f"{OUT}/oracle_grid.json"))
    out = {}
    orc_main = {k: v for k, v in orc.items() if k != "lead0"}
    pr = np.load(f"{d}/prob_val.npy" if os.path.exists(f"{d}/prob_val.npy") else f"{d}/heat_val.npy")
    ids = np.load(f"{d}/ids_val.npy")
    dirf = np.load(f"{d}/dir_val.npy")
    sel = [int(x) for x in np.linspace(30, len(ids) - 30, 8)]
    gts = GD.load_gt_graphs("val", ids[sel])
    # vorhergesagt (Spurverfolger) + Soll
    _, preds = GD.eval_variant(pr[sel], dirf[sel], gts, "trace", g["trace"]["params"], g["trace"].get("asm"), workers=1, with_route=False)
    out["pred"] = f"{FIG}/e31_pred.png"
    VG.grid(preds, gts, ids[sel], out["pred"], ncol=4, figsize=4.3, titles=[f"VAL sample {int(ids[k])}" for k in sel])
    # Orakel
    po, do = GD.oracle_inputs("b123", "val", ids[sel])
    _, preds_o = GD.eval_variant(po, do, gts, "trace", None, dict(lead_m=5.0), workers=1, with_route=False)
    out["oracle"] = f"{FIG}/e31_oracle.png"
    VG.grid(preds_o, gts, ids[sel], out["oracle"], ncol=4, figsize=4.3, titles=[f"oracle map, sample {int(ids[k])}" for k in sel])
    # Zerlegung der Fehler (Balken, VAL)
    fig, ax = plt.subplots(figsize=(11, 4))
    keys = [("F1@1.0", "F1@1m"), ("route_f", "route F"), ("split6_F1", "split F1 (6 m)"), ("merge6_F1", "merge F1 (6 m)"), ("dir_acc", "direction acc")]
    var = [("oracle map (GT heat + GT dir)", orc_main["val"]), ("predicted heat + GT dir", g["trace_gtdir"]["val"]), ("predicted heat + predicted dir (tracer)", g["trace"]["val"]), ("predicted, skeleton graph", g["skel"]["val"])]
    x = np.arange(len(keys))
    w = 0.2
    for k, (lab, m) in enumerate(var):
        vals = [m.get(kk, 0) for kk, _ in keys]
        ax.bar(x + (k - 1.5) * w, vals, w, label=lab)
        for xi, v in zip(x + (k - 1.5) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in keys], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("VAL: where the grid approach loses accuracy (geometry vs direction vs assembly)", fontsize=9)
    fig.tight_layout()
    out["decomp"] = f"{FIG}/e31_decomp.png"
    fig.savefig(out["decomp"], dpi=100)
    plt.close(fig)
    return g, orc, out


def report_e31(name):
    g, orc, im = figs(name)
    t = TEXT.get("e31", {})
    variants = [("oracle map (GT heat + GT direction) -> tracer, junction lead 5 m", {k: v for k, v in orc.items() if k != "lead0"}), ("oracle map -> tracer, no junction lead", orc.get("lead0", {})),
                ("predicted heat + GT direction -> tracer, lead 5 m", g["trace_gtdir"]), ("predicted heat + predicted direction -> tracer, lead 5 m (grid approach)", g["trace"]),
                ("predicted -> tracer, no junction lead", g.get("trace_lead0", {})), ("predicted -> skeleton graph", g["skel"])]
    head, rows = rows_for(variants)
    images = [(im["pred"], "VAL: predicted graph (red, arrows = travel direction, triangle = split, diamond = merge) over ground truth (green)."),
              (im["oracle"], "Same samples from the PERFECT map (GT heat + GT direction): ceiling of the rule-based assembly."),
              (im["decomp"], "Error decomposition on VAL.")]
    make_report2("E31", t.get("title", "Grid approach: dense field -> directed lane graph"), t.get("question", ""), t.get("setup", []), [("Graph metrics (grid approach)", head, rows)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=lambda ax: AF.grid_graph_arch(ax))
    print("E31 written")


def report_e33(name):
    d = f"{OUT}/{name}"
    sp = json.load(open(f"{d}/graph_spacing.json"))
    t = TEXT.get("e33", {})
    head = ["step (point spacing)", "F1@0.5m", "F1@1m", "P@1m", "R@1m", "1:1 F1@1m", "false length", "ms / sample (1 core)", "support points / sample"]
    rows = []
    for k, v in sp.items():
        m = v["metrics"]
        rows.append([f"{k} m", f3(m["F1@0.5"]), f3(m["F1@1.0"]), f3(m["P@1.0"]), f3(m["R@1.0"]), f3(m["F1o2o@1.0"]), f3(m["false_len_frac"]), f"{v['ms_per_sample']:.0f}", f"{v['n_points']:.0f}"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ks = [float(k) for k in sp]
    ax.plot(ks, [sp[k]["metrics"]["F1@1.0"] for k in sp], "-o", label="F1@1m")
    ax.plot(ks, [sp[k]["metrics"]["F1@0.5"] for k in sp], "-s", label="F1@0.5m")
    ax.set_xlabel("tracer step = point spacing (m)")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("point-wise iterative construction: accuracy vs spacing (VAL)", fontsize=9)
    fig.tight_layout()
    p = f"{FIG}/e33_spacing.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    make_report2("E33", t.get("title", "Point-wise iterative construction with adjustable spacing"), t.get("question", ""), t.get("setup", []), [("Tracer step sweep (VAL)", head, rows)], [(p, "Accuracy vs step size.")], t.get("findings", ["(to be written)"]), t.get("limits", []), arch=lambda ax: AF.grid_graph_arch(ax))
    print("E33 written")


if __name__ == "__main__":
    name = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "figs"
    if mode == "report":
        report_e31(name)
        if os.path.exists(f"{OUT}/{name}/graph_spacing.json"):
            report_e33(name)
    else:
        g, orc, im = figs(name)
        print(im)
