"""Bericht E38: Was begrenzt der Raster-Ansatz? Obergrenzen der Gitter-Vorschlaege (Abdeckung, Praezision, Obergrenze bei idealem Einrasten) fuer drei Verfolger-Schwellen, Orakel-Obergrenze des regelbasierten Zusammenbaus
und der Graph des Endkopfes (Spurverfolger + Zusammenbau) auf TRAIN-Auswahl / VAL / TEST. Aufruf (bev): make_e38.py [figs|report]. Text: loop2/text_e37_e42.py TEXT['e38']."""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs as AF  # noqa: E402
import dense_report_lib as DR  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT", {})
NICE = {"train_eval": "TRAIN (eval subset)", "val": "VAL", "test": "TEST", "oof": "TRAIN scenes held out from the head (OOF, fold 0)"}
GCOLS = [("F1@0.5", "F1@0.5m"), ("F1@1.0", "F1@1m"), ("F1@1.5", "F1@1.5m"), ("P@1.0", "P@1m"), ("R@1.0", "R@1m"), ("lanes_found", "lanes found"), ("false_len_frac", "false length"),
         ("split_F1", "split F1 3m"), ("split6_F1", "split F1 6m"), ("merge_F1", "merge F1 3m"), ("merge6_F1", "merge F1 6m"), ("route_rec", "route rec"), ("route_prec", "route prec"), ("route_f", "route F"), ("dir_acc", "dir acc")]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main(mode):
    run = "e37_final"
    # Tabelle 1: Vorschlags-Obergrenzen
    h1 = ["tracer thresholds", "set", "points / sample", "point precision @1m", "GT coverage @1m", "GT coverage @1.5m", "GT coverage @2.5m", "ceiling F1 if snapped (1m)", "ceiling F1 if snapped (1.5m)"]
    r1 = []
    cov = {}
    for tag, lab in (("strict", "strict (t_seed 0.9, t_on 1.1)"), ("relax", "relaxed (1.2 / 2.0)  <- proposals used"), ("wide", "wide (1.4 / 2.4)")):
        for st in ("val", "test", "train_eval"):
            d = jl(f"{OUT}/{run}/props_{tag}_{st}_ceiling.json")
            if not d:
                continue
            cov[(tag, st)] = d
            r1.append([lab, NICE[st], f"{d['points_per_sample']:.0f}", f3(d["tol1.0"]["point_precision"]), f3(d["tol1.0"]["gt_coverage"]), f3(d["tol1.5"]["gt_coverage"]), f3(d["tol2.5"]["gt_coverage"]),
                       f3(d["tol1.0"]["ceiling_F1_if_snapped"]), f3(d["tol1.5"]["ceiling_F1_if_snapped"])])
    # Tabelle 2: Orakel-Obergrenze des Zusammenbaus + Graph des Endkopfes
    orc = jl(f"{OUT}/oracle_grid.json") or {}
    h2 = ["graph source", "set"] + [c[1] for c in GCOLS]
    r2 = []
    for st in ("val", "test"):
        if st in orc:
            r2.append(["ORACLE field (GT heat + GT direction) -> same assembler", NICE[st]] + [f3(orc[st].get(k)) for k, _ in GCOLS])
    for st in ("train_eval", "val", "test"):
        d = jl(f"{OUT}/{run}/graph_{st}_trace.json")
        if d:
            r2.append(["E37 final head -> tracer 'strict' + assembler (grid graph)", NICE[st]] + [f3(d["metrics"].get(k)) for k, _ in GCOLS])
    d = jl(f"{OUT}/e37_fold0/graph_oof_trace.json")
    if d:
        r2.append(["fold head 0 -> same pipeline", NICE["oof"]] + [f3(d["metrics"].get(k)) for k, _ in GCOLS])
    for st in ("val", "test"):
        d = jl(f"{OUT}/e27_gcar/graph_grid.json")
        if d and "trace" in d and st in d["trace"]:
            r2.append(["E27 baseline head (4207 rows) -> same pipeline", NICE[st]] + [f3(d["trace"][st].get(k)) for k, _ in GCOLS])
    images = []
    # Abbildung 1: Abdeckung gegen Toleranz je Schwelle (VAL)
    if cov:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        tols = [1.0, 1.5, 2.5]
        for (tag, st), d in cov.items():
            if st != "val":
                continue
            ax[0].plot(tols, [d[f"tol{t}"]["gt_coverage"] for t in tols], "-o", label=f"{tag} ({d['points_per_sample']:.0f} pts)")
            ax[1].plot(tols, [d[f"tol{t}"]["point_precision"] for t in tols], "-o", label=tag)
        ax[0].set_title("share of GT points within tol of a proposal (VAL)", fontsize=9)
        ax[1].set_title("share of proposal points within tol of GT (VAL)", fontsize=9)
        for a_ in ax:
            a_.set_xlabel("tolerance (m)")
            a_.grid(alpha=0.3)
            a_.legend(fontsize=8)
        fig.tight_layout()
        p = f"{FIG}/e38_coverage.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Coverage and precision of the proposals as a function of tolerance. Relaxing the tracer thresholds trades precision for coverage; the sparse stage must pull proposals onto the centerline and reject false ones."))
    # Abbildung 2: Vorschlaege der drei Schwellen auf VAL-Beispielen
    try:
        ids = np.load(f"{OUT}/{run}/ids_val.npy")
        sel = list(np.linspace(8, len(ids) - 9, 3).astype(int))
        props = {tag: pickle.load(open(f"{OUT}/{run}/props_{tag}_val.pkl", "rb")) for tag in ("strict", "relax", "wide") if os.path.exists(f"{OUT}/{run}/props_{tag}_val.pkl")}
        if props:
            gts = DR.gt_graphs("val", ids[sel])
            fig, axs = plt.subplots(len(sel), len(props), figsize=(4.0 * len(props), 4.0 * len(sel)), squeeze=False)
            for r, k in enumerate(sel):
                for c, (tag, pl) in enumerate(props.items()):
                    ax = axs[r, c]
                    for q in gts[r]["pts"]:
                        ax.plot(q[:, 0], q[:, 1], "-", color="#2ecc71", lw=3.0, alpha=0.55)
                    for q in pl[k]["pts"]:
                        ax.plot(q[:, 0], q[:, 1], "-", color="#d62728", lw=1.0)
                    ax.plot([0], [0], "k^", ms=5)
                    ax.set_xlim(-50, 50)
                    ax.set_ylim(-50, 50)
                    ax.set_aspect("equal")
                    ax.tick_params(labelsize=6)
                    ax.set_title(f"{tag}: sample {int(ids[k])}", fontsize=8)
            fig.suptitle("proposals (red) vs ground-truth centerlines (green, wide) at three tracer settings", fontsize=9)
            fig.tight_layout()
            p = f"{FIG}/e38_proposals.png"
            fig.savefig(p, dpi=90)
            plt.close(fig)
            images.append((p, "Proposals at the three tracer settings on three VAL samples."))
    except Exception as ex:  # noqa: BLE001
        print("Abbildung Vorschlaege:", ex)
    if mode == "figs":
        for r in r1 + r2:
            print(r)
        return
    t = TEXT.get("e38", {})
    make_report2("E38", t.get("title", "E38"), t.get("question", ""), t.get("setup", []),
                 [("Ceiling of the grid proposals (dense head E37 -> line tracer -> polylines at 1 m spacing)", h1, r1),
                  ("What the grid pipeline itself can reach: oracle field and the trained head", h2, r2)],
                 images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=lambda ax: AF.grid_graph_arch(ax))
    print("E38 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
