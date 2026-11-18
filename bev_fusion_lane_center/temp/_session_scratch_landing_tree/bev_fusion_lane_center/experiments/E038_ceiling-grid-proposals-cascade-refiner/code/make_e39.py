"""Bericht E39: Kaskaden-Verfeinerer auf den Gitter-Vorschlaegen (duenner Extraktor auf dem Raster). Aufruf (gcar): make_e39.py [figs|report]
Liest out/loop2/e39_*/{metrics.json, refined_*.pkl, model.pt, tau_sweep_*.json}, e37_final/graph_*_trace.json, e32_main/graph_full.json, oracle_grid.json. Text: loop2/text_e37_e42.py TEXT['e39']."""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs2 as AF2  # noqa: E402
import dense_report_lib as DR  # noqa: E402
import graph_from_dense as GD  # noqa: E402
import refiner as R  # noqa: E402
import viz_graph as VG  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT", {})
NICE = {"train_eval": "TRAIN (eval subset)", "val": "VAL", "test": "TEST", "oofA": "TRAIN scenes, held out from the dense head (OOF fold 0; seen by the refiner)"}
GCOLS = [("F1@0.5", "F1@0.5m"), ("F1@1.0", "F1@1m"), ("F1@1.5", "F1@1.5m"), ("P@1.0", "P@1m"), ("R@1.0", "R@1m"), ("lanes_found", "lanes found"), ("false_len_frac", "false length"),
         ("split_F1", "split F1 3m"), ("split6_F1", "split F1 6m"), ("merge_F1", "merge F1 3m"), ("merge6_F1", "merge F1 6m"), ("route_rec", "route rec"), ("route_prec", "route prec"), ("route_f", "route F"), ("dir_acc", "dir acc")]
ABL = [("e39_main", "main: 3 blocks, lambda_FP 3, jitter 0.5 m, 1 m spacing"), ("e39_b1", "1 block"), ("e39_b2", "2 blocks"), ("e39_b4", "4 blocks"), ("e39_fp1", "lambda_FP 1 (false alarms not up-weighted)"), ("e39_reg", "regularised: input channel dropout 0.2 + feature noise 0.3"),
       ("e39_fp6", "lambda_FP 6"), ("e39_nojit", "no proposal jitter in training"), ("e39_s2", "2 m point spacing")]
TAU = 0.3
PARAMS, ASM = {"t_seed": 0.9, "t_on": 1.1}, {"lead_m": 5.0}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load_refiner(name):
    ck = torch.load(f"{OUT}/{name}/model.pt", map_location="cuda", weights_only=False)
    a = ck["args"]
    m = R.Refiner(39, 32, a["D"], a["blocks"]).cuda()
    m.load_state_dict(ck["state"])
    m.eval()
    return m, a


@torch.no_grad()
def cascade_fig(name, run, st, part, sel, path, tag="relax"):
    model, _ = load_refiner(name)
    mean, std = R.channel_stats()
    mean, std = mean.cuda(), std.cuda()
    ds = R.RData([dict(run=run, st=st, part=part)], tag, train=False)
    ids = ds.S[0]["ids"]
    gts = DR.gt_graphs(part, ids[sel])
    nb = len(model.blocks)
    fig, axs = plt.subplots(len(sel), nb + 1, figsize=(3.7 * (nb + 1), 3.7 * len(sel)), squeeze=False)
    for r, k in enumerate(sel):
        b = ds[k]
        x = ((b["x"][None].cuda() - mean[None, :, None, None]) / std[None, :, None, None])
        heat, dirf, pts, mask = b["heat"][None].cuda(), b["dirf"][None].cuda(), b["pts"][None].cuda(), b["mask"][None].cuda()
        outs, ori = model(x, heat, dirf, pts, mask)
        panels = [("proposals", pts, None)] + [(f"after block {i + 1}", p, torch.sigmoid(ex)) for i, (p, ex) in enumerate(outs)]
        for c, (ttl, p, ex) in enumerate(panels):
            ax = axs[r, c]
            VG.setup_ax(ax, f"{ttl} - sample {int(ids[k])}" if r == 0 else f"sample {int(ids[k])}")
            for q in gts[r]["pts"]:
                ax.plot(q[:, 0], q[:, 1], "-", color="#1a9850", lw=3.0, alpha=0.5, zorder=1)
            m = mask[0].cpu().numpy()
            P_ = p[0].cpu().numpy()
            for a in range(P_.shape[0]):
                if not m[a].any():
                    continue
                q = P_[a][m[a]]
                if ex is None:
                    ax.plot(q[:, 0], q[:, 1], ".", color="#555", ms=2.0, zorder=3)
                else:
                    e = ex[0, a].cpu().numpy()[m[a]]
                    ax.scatter(q[:, 0], q[:, 1], c=e, s=3.0, cmap="RdYlGn", vmin=0, vmax=1, zorder=3)
    fig.suptitle("cascade: proposal points move block by block onto the centerline; colour = predicted existence (red 0 -> green 1); green line = ground truth", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def system_fig(name, run, st, part, sel, path, tau=TAU, tag="relax"):
    ids = np.load(f"{OUT}/{run}/ids_{st}.npy")
    gts = DR.gt_graphs(part, ids[sel])
    prob = np.load(f"{OUT}/{run}/prob_{st}.npy", mmap_mode="r")
    dirf = np.load(f"{OUT}/{run}/dir_{st}.npy", mmap_mode="r")
    ref = pickle.load(open(f"{OUT}/{name}/refined_{'val' if st == 'val' else st}.pkl", "rb"))
    fig, axs = plt.subplots(len(sel), 3, figsize=(11.5, 3.8 * len(sel)), squeeze=False)
    for r, k in enumerate(sel):
        g0 = GD.build(np.asarray(prob[k], np.float32), np.asarray(dirf[k], np.float32), "trace", None, PARAMS, ASM)
        g1 = R.graph_from_refined(ref[k], tau_e=tau)
        VG.panel(axs[r, 0], g0, gts[r], title=f"grid graph (tracer) - sample {int(ids[k])}" if r == 0 else f"sample {int(ids[k])}", background=prob[k])
        VG.panel(axs[r, 1], g1, gts[r], title="grid + cascade refiner" if r == 0 else "")
        VG.panel(axs[r, 2], None, gts[r], title="ground truth" if r == 0 else "")
    fig.suptitle("red = prediction (arrows = travel direction, triangles = splits, diamonds = merges), green = ground truth", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main(mode):
    run = "e37_final"
    mets = {n: jl(f"{OUT}/{n}/metrics.json") for n, _ in ABL}
    mets = {k: v for k, v in mets.items() if v and "val" in v}
    main_m = mets.get("e39_main")
    # Tabelle 1: Vergleich der Systeme
    h1 = ["system", "set"] + [c[1] for c in GCOLS]
    r1 = []
    orc = jl(f"{OUT}/oracle_grid.json") or {}
    sparse = jl(f"{OUT}/e32_main/graph_full.json") or {}
    for st in ("train_eval", "val", "test"):
        if st in sparse:
            r1.append(["sparse only (E32 LaneFormer, no grid)", NICE[st]] + [f3(sparse[st].get(k)) for k, _ in GCOLS])
    for st in ("train_eval", "val", "test"):
        d = jl(f"{OUT}/{run}/graph_{st}_trace.json")
        if d:
            r1.append(["grid graph (E37 head + strict tracer + assembler)", NICE[st]] + [f3(d["metrics"].get(k)) for k, _ in GCOLS])
    if main_m:
        for st in ("val", "test", "oofA"):
            if st in main_m:
                r1.append([f"grid proposals unrefined (relaxed tracer, same linking, tau_e {TAU})", NICE[st]] + [f3(main_m[st]["proposals"].get(k)) for k, _ in GCOLS])
        for st in ("val", "test", "oofA"):
            if st in main_m:
                r1.append([f"grid + cascade refiner (E39 main, tau_e {TAU})", NICE[st]] + [f3(main_m[st]["refined"].get(k)) for k, _ in GCOLS])
    for st in ("val", "test"):
        if st in orc:
            r1.append(["oracle field -> assembler (ceiling of the rule-based builder)", NICE[st]] + [f3(orc[st].get(k)) for k, _ in GCOLS])
    # Tabelle 2: Ablationen
    h2 = ["variant", "VAL F1@1m", "VAL P@1m", "VAL R@1m", "VAL route F", "VAL split F1 6m", "VAL merge F1 6m", "TEST F1@1m", "TEST route F", "OOF-A F1@1m (seen scenes)", "train loss (last 100 steps)"]
    r2 = []
    for n, lab in ABL:
        if n not in mets:
            continue
        m = mets[n]
        v, t, o = m.get("val", {}).get("refined", {}), m.get("test", {}).get("refined", {}), m.get("oofA", {}).get("refined", {})
        cu = m.get("curve") or []
        r2.append([f"{lab}", f3(v.get("F1@1.0")), f3(v.get("P@1.0")), f3(v.get("R@1.0")), f3(v.get("route_f")), f3(v.get("split6_F1")), f3(v.get("merge6_F1")), f3(t.get("F1@1.0")), f3(t.get("route_f")), f3(o.get("F1@1.0")),
                   f3(cu[-1][1]) if cu else "-"])
    # Tabelle 3: Schwelle der Existenz
    sw = jl(f"{OUT}/e39_main/tau_sweep_val.json") or {}
    swt = jl(f"{OUT}/e39_main/tau_sweep_test.json") or {}
    h3 = ["tau_e", "VAL P@1m", "VAL R@1m", "VAL F1@1m", "VAL route F", "VAL split F1 6m", "VAL merge F1 6m", "VAL false length", "TEST F1@1m", "TEST route F"]
    r3 = []
    for k, m in sw.items():
        if k.startswith("dedupe") or k == "reduce_edges":
            continue
        t = swt.get(k, {})
        r3.append([k, f3(m["P@1.0"]), f3(m["R@1.0"]), f3(m["F1@1.0"]), f3(m["route_f"]), f3(m["split6_F1"]), f3(m["merge6_F1"]), f3(m["false_len_frac"]), f3(t.get("F1@1.0")), f3(t.get("route_f"))])
    for k, m in sw.items():
        if k.startswith("dedupe") or k == "reduce_edges":
            r3.append([f"tau_e 0.3, {k.replace('_', ' ')}" + (" (transitive reduction of the edge set)" if k == "reduce_edges" else ""), f3(m["P@1.0"]), f3(m["R@1.0"]), f3(m["F1@1.0"]), f3(m["route_f"]), f3(m["split6_F1"]), f3(m["merge6_F1"]), f3(m["false_len_frac"]), "-", "-"])
    images = []
    if r2:
        names = [n for n, _ in ABL if n in mets]
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.8))
        for ax, key, ttl in ((axs[0], "F1@1.0", "point F1 @ 1 m"), (axs[1], "route_f", "route F (APLS-like)")):
            x = np.arange(len(names))
            for j, (st, col) in enumerate((("val", "#1f77b4"), ("test", "#ff7f0e"), ("oofA", "#2ca02c"))):
                vals = [mets[n].get(st, {}).get("refined", {}).get(key, np.nan) for n in names]
                ax.bar(x + (j - 1) * 0.27, vals, 0.27, label=NICE[st][:28], color=col)
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace("e39_", "") for n in names], fontsize=8)
            lo = np.nanmin([mets[n].get(st, {}).get("refined", {}).get(key, np.nan) for n in names for st in ("val", "test", "oofA")])
            ax.set_ylim(max(lo - 0.05, 0), None)
            ax.set_title(ttl, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = f"{FIG}/e39_ablation.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Ablations of the refiner (same proposals, same training data)."))
    if sw:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ks = [k for k in sw if not k.startswith("dedupe") and k != "reduce_edges"]
        xs = [float(k) for k in ks]
        ax.plot(xs, [sw[k]["P@1.0"] for k in ks], "-o", label="precision @1m")
        ax.plot(xs, [sw[k]["R@1.0"] for k in ks], "-s", label="recall @1m")
        ax.plot(xs, [sw[k]["F1@1.0"] for k in ks], "-^", label="F1 @1m")
        ax.plot(xs, [sw[k]["route_f"] for k in ks], "-d", label="route F")
        ax.set_xlabel("existence threshold tau_e (VAL)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = f"{FIG}/e39_tau.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Operating point of the refiner: the existence threshold plays the role of the dense head's threshold."))
    if os.path.exists(f"{OUT}/e39_main/model.pt"):
        ids = np.load(f"{OUT}/{run}/ids_val.npy")
        sel = list(np.linspace(10, len(ids) - 11, 3).astype(int))
        images.append((cascade_fig("e39_main", run, "val", "val", sel, f"{FIG}/e39_cascade.png"), "Block-by-block movement of the proposal points on three VAL samples (the same input to all blocks is the proposal set; later blocks re-sample the maps at the new positions)."))
        sel = list(np.linspace(6, len(ids) - 7, 4).astype(int))
        images.append((system_fig("e39_main", run, "val", "val", sel, f"{FIG}/e39_system_val.png"), "VAL: grid graph vs grid + refiner vs ground truth."))
        ids = np.load(f"{OUT}/{run}/ids_test.npy")
        sel = list(np.linspace(6, len(ids) - 7, 4).astype(int))
        images.append((system_fig("e39_main", run, "test", "val", sel, f"{FIG}/e39_system_test.png"), "TEST (untouched by all decisions): grid graph vs grid + refiner vs ground truth."))
    if mode == "figs":
        for r in r1 + r2 + r3:
            print(r)
        return
    t = TEXT.get("e39", {})
    make_report2("E39", t.get("title", "E39"), t.get("question", ""), t.get("setup", []),
                 [("System comparison (point and graph metrics; thresholds fixed on VAL)", h1, r1), ("Ablations of the cascade refiner", h2, r2), ("Existence threshold and de-duplication (VAL, with TEST for the threshold sweep)", h3, r3)],
                 images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF2.refiner_arch)
    print("E39 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
