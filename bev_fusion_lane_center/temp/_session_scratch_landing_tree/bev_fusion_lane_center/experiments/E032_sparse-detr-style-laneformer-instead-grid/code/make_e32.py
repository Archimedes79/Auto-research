"""Bericht E32 (duenner Extraktor LaneFormer gegen Raster-Ansatz) und Abbildungen fuer den Vergleich.
Aufruf: make_e32.py [lf_lauf=e32_main] [raster_lauf=e27_gcar] [figs|report]
Liest out/loop2/<lf>/{metrics.json, graph_full.json, graph_grid.json, pred_*.npz}, out/loop2/<raster>/graph_grid.json; Text aus loop2/text_e32_e36.py (TEXT['e32'])."""
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
import lane_former as LF  # noqa: E402
import viz_graph as VG  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e32_e36.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e32_e36.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT", {})
COLS = [("F1@0.5", "F1@0.5m"), ("F1@1.0", "F1@1m"), ("F1@1.5", "F1@1.5m"), ("P@1.0", "P@1m"), ("R@1.0", "R@1m"), ("F1o2o@1.0", "1:1 F1@1m"), ("lanes_found", "lanes found"), ("false_len_frac", "false length"),
        ("split_F1", "split F1 3m"), ("split6_F1", "split F1 6m"), ("merge_F1", "merge F1 3m"), ("merge6_F1", "merge F1 6m"), ("route_rec", "route rec"), ("route_prec", "route prec"), ("route_f", "route F"), ("dir_acc", "dir acc")]
NICE = {"train_eval": "TRAIN (eval subset)", "val": "VAL", "test": "TEST"}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def table(variants, sets=("val", "test", "train_eval")):
    rows = []
    for st in sets:
        for lab, d in variants:
            if d is None or st not in d:
                continue
            rows.append([NICE[st], lab] + [f3(d[st].get(k)) for k, _ in COLS])
    return ["set", "approach"] + [h for _, h in COLS], rows


def load_pred(run, st):
    z = np.load(f"{OUT}/{run}/pred_{st}.npz")
    return {k: (z[k].astype(np.float32) if k in ("cls", "adj") else z[k]) for k in z.files}


def draw_instances(ax, g, cmap="tab20"):
    c = plt.get_cmap(cmap)
    for k, p in enumerate(g["pts"]):
        if len(p) >= 2:
            ax.plot(p[:, 0], p[:, 1], "-", color=c(k % 20), lw=1.8)
    VG.draw_graph(ax, dict(pts=[], edges=[]))


def figs(lf, grid):
    gf = jl(f"{OUT}/{lf}/graph_full.json")
    gg = jl(f"{OUT}/{grid}/graph_grid.json")
    m = jl(f"{OUT}/{lf}/metrics.json")
    out = {}
    R = load_pred(lf, "val")
    sel = [int(x) for x in np.linspace(40, len(R["idx"]) - 40, 6)]
    ids = R["idx"][sel]
    gts = GD.load_gt_graphs("val", ids)
    sparse = [LF.to_graph(R["cls"][i], R["pts"][i], R["adj"][i], gf["tau_c"], gf["tau_a"], gf["dedupe"], gf.get("max_gap", 4.0)) for i in sel]
    pr = np.load(f"{OUT}/{grid}/prob_val.npy")
    dr = np.load(f"{OUT}/{grid}/dir_val.npy")
    ids_g = np.load(f"{OUT}/{grid}/ids_val.npy")
    pos = {int(i): n for n, i in enumerate(ids_g)}
    sg = [pos[int(i)] for i in ids]
    _, gridg = GD.eval_variant(pr[sg], dr[sg], gts, "trace", gg["trace"]["params"], gg["trace"].get("asm"), workers=1, with_route=False)
    fig, axs = plt.subplots(len(sel), 3, figsize=(12.6, 4.2 * len(sel)))
    for r, k in enumerate(sel):
        VG.panel(axs[r, 0], None, gts[r], title=f"ground truth, VAL sample {int(ids[r])}")
        VG.panel(axs[r, 1], gridg[r], gts[r], title="grid approach (dense head + tracer)")
        VG.panel(axs[r, 2], sparse[r], gts[r], title="sparse LaneFormer (chunks + successor matrix)")
    fig.tight_layout()
    out["qual"] = f"{FIG}/e32_qual.png"
    fig.savefig(out["qual"], dpi=80)
    plt.close(fig)
    # Instanzen farbig
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6))
    for a_, i_ in zip(axs.ravel(), range(6)):
        a_.set_aspect("equal")
        draw_instances(a_, sparse[i_])
        VG.setup_ax(a_, f"LaneFormer instances (one colour per chunk), sample {int(ids[i_])}")
        for p in gts[i_]["pts"]:
            a_.plot(p[:, 0], p[:, 1], "-", color="k", lw=0.6, alpha=0.5)
    fig.tight_layout()
    out["inst"] = f"{FIG}/e32_instances.png"
    fig.savefig(out["inst"], dpi=80)
    plt.close(fig)
    # PR-Kurven
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, st in zip(axs, ("val", "train_eval")):
        taus = sorted(m[st].keys(), key=float)
        ax.plot([m[st][t]["R@1.0"] for t in taus], [m[st][t]["P@1.0"] for t in taus], "-o", ms=3, label="LaneFormer (existence threshold sweep, no dedupe)")
        gm = jl(f"{OUT}/{grid}/metrics.json")
        t2 = sorted(gm[st].keys(), key=float)
        ax.plot([gm[st][t]["R@1.0"] for t in t2], [gm[st][t]["P@1.0"] for t in t2], "-s", ms=3, label="dense head + thinning (threshold sweep)")
        ax.plot([gg["trace"][st]["R@1.0"]], [gg["trace"][st]["P@1.0"]], "r*", ms=11, label="grid approach: tracer graph")
        ax.plot([gf[st]["R@1.0"]], [gf[st]["P@1.0"]], "g*", ms=11, label="sparse: chosen operating point")
        ax.set_xlabel("recall @ 1 m")
        ax.set_ylabel("precision @ 1 m")
        ax.set_title(NICE[st], fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out["pr"] = f"{FIG}/e32_pr.png"
    fig.savefig(out["pr"], dpi=90)
    plt.close(fig)
    # Verlauf
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    c = np.array(m["curve"])
    for j, nm in enumerate(["total", "existence", "points", "successors", "dense aux"]):
        axs[0].plot(c[:, 0], c[:, 1 + j], label=nm)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("training loss")
    axs[0].legend(fontsize=7)
    axs[0].grid(alpha=0.3)
    if m.get("mon"):
        st_ = [x[0] for x in m["mon"]]
        for nm in ("train_eval", "val"):
            axs[1].plot(st_, [x[1][nm][0] for x in m["mon"]], "-o", label=f"{NICE[nm]} F1@1m (tau_c 0.5, 120 samples)")
        axs[1].set_xlabel("step")
        axs[1].legend(fontsize=7)
        axs[1].grid(alpha=0.3)
    fig.tight_layout()
    out["curves"] = f"{FIG}/e32_curves.png"
    fig.savefig(out["curves"], dpi=90)
    plt.close(fig)
    return out


def report(lf, grid):
    im = figs(lf, grid)
    gf = jl(f"{OUT}/{lf}/graph_full.json")
    gg = jl(f"{OUT}/{grid}/graph_grid.json")
    gl = jl(f"{OUT}/{lf}/graph_grid.json")
    t = TEXT.get("e32", {})
    variants = [("grid: dense head (E27) + tracer", gg["trace"]), ("grid on the LaneFormer trunk's aux head + tracer", gl["trace"] if gl else None), ("SPARSE LaneFormer (learned successor matrix)", gf), ("SPARSE LaneFormer + ridge snapping (uses the dense aux head)", gf.get("snap"))]
    head, rows = table(variants)
    # Diagnose: Train-vs-Val
    dh = ["approach", "F1@1m TRAIN", "F1@1m VAL", "F1@1m TEST", "gap TRAIN-VAL", "route F TRAIN", "route F VAL", "route F TEST", "split F1@6m VAL", "merge F1@6m VAL"]
    dr_ = []
    for lab, d in variants:
        if d is None:
            continue
        dr_.append([lab, f3(d["train_eval"]["F1@1.0"]), f3(d["val"]["F1@1.0"]), f3(d["test"]["F1@1.0"]), f3(d["train_eval"]["F1@1.0"] - d["val"]["F1@1.0"]), f3(d["train_eval"]["route_f"]), f3(d["val"]["route_f"]), f3(d["test"]["route_f"]), f3(d["val"]["split6_F1"]), f3(d["val"]["merge6_F1"])])
    sw = gf["sweep"]
    sh = ["tau_c", "dedupe", "F1@1m", "P@1m", "R@1m", "1:1 F1@1m"]
    sr = []
    for k, v in sw.items():
        if k.startswith("tc"):
            tc, dd = k[2:].split("_dd")
            sr.append([tc, dd, f3(v["F1"]), f3(v["P"]), f3(v["R"]), f3(v["F1o2o"])])
    gh = ["max gap end(i)->start(j) (m)", "tau_a", "route rec", "route prec", "route F", "split F1@3m", "merge F1@3m"]
    gr = []
    for k, v in sw.items():
        if k.startswith("gap"):
            gp, ta = k[3:].split("_ta")
            gr.append([gp, ta, f3(v["route_rec"]), f3(v["route_prec"]), f3(v["route_f"]), f3(v["split_F1"]), f3(v["merge_F1"])])
    em = jl(f"{OUT}/{lf}/edge_modes.json")
    eh = ["edge construction", "edges / sample (GT piece-level: ~21)", "route rec", "route prec", "route F", "split F1@3m", "split F1@6m", "merge F1@3m", "merge F1@6m"]
    er = []
    if em:
        for k, lab in (("learned", "learned successor matrix (E32)"), ("geometric", "geometric rule only (end -> start within 6 m, heading < 60 deg)"), ("union", "union of both"), ("intersection", "intersection of both")):
            if k in em:
                v = em[k]
                er.append([lab, f"{v['edges_per_sample']:.1f}", f3(v["route_rec"]), f3(v["route_prec"]), f3(v["route_f"]), f3(v["split_F1"]), f3(v["split6_F1"]), f3(v["merge_F1"]), f3(v["merge6_F1"])])
    images = [(im["qual"], "VAL: ground truth | grid approach | sparse LaneFormer (red = prediction with arrows, green = GT, triangles = splits, diamonds = merges)."),
              (im["inst"], "LaneFormer instances: every predicted chunk (<= 20 m, 11 points) in its own colour, GT in black."),
              (im["pr"], "Precision-recall at 1 m: sparse threshold sweep vs dense head sweep; stars = operating points of the graph metrics."),
              (im["curves"], "Training losses and monitor F1 during training.")]
    make_report2("E32", t.get("title", "Sparse extractor vs grid approach"), t.get("question", ""), t.get("setup", []),
                 [("Graph metrics, grid vs sparse", head, rows), ("Train-versus-validation diagnosis", dh, dr_), ("Post-processing sweep on VAL (LaneFormer): existence threshold x duplicate suppression", sh, sr), ("Edge construction sweep on VAL (LaneFormer)", gh, gr)] + ([("Learned successor matrix vs geometric linking of the same predicted chunks (VAL, every 2nd sample)", eh, er)] if er else []), images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=lambda ax: AF.lf_arch(ax))
    print("E32 written")


if __name__ == "__main__":
    lf = sys.argv[1] if len(sys.argv) > 1 else "e32_main"
    grid = sys.argv[2] if len(sys.argv) > 2 else "e27_gcar"
    mode = sys.argv[3] if len(sys.argv) > 3 else "figs"
    if mode == "report":
        report(lf, grid)
    else:
        print(figs(lf, grid))
