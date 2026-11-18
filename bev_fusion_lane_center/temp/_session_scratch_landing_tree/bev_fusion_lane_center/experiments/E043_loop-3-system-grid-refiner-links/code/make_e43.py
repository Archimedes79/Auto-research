"""Bericht E43: Gesamtsystem - Leiter der Bausteine (Gitter -> Verfeinerer -> gelernte Verbindungen -> Karte -> Zeit), Train-vs-Val-Diagnose, Laufzeit, Beispiele und Fehlerfaelle. Aufruf (gcar): make_e43.py [figs|report]
Liest out/loop2/{e37_final, e37_fold*, e39_main, e38_link, sys_*, e43/*, e41_B}/... Text: loop2/text_e37_e42.py TEXT['e43'] (inkl. 'final' = gewaehlte Konfiguration)."""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs2 as AF2  # noqa: E402
import dense_report_lib as DR  # noqa: E402
import graph_eval as GE  # noqa: E402
import viz_graph as VG  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT", {})
CFG = TEXT.get("e43_cfg", {})   # dict: dense (Lauf der Endkonfiguration), tau_l, sets: Dateien
GCOLS = [("F1@0.5", "F1@0.5m"), ("F1@1.0", "F1@1m"), ("F1@1.5", "F1@1.5m"), ("P@1.0", "P@1m"), ("R@1.0", "R@1m"), ("lanes_found", "lanes found"), ("false_len_frac", "false length"),
         ("split_F1", "split F1 3m"), ("split6_F1", "split F1 6m"), ("merge_F1", "merge F1 3m"), ("merge6_F1", "merge F1 6m"), ("route_rec", "route rec"), ("route_prec", "route prec"), ("route_f", "route F"), ("dir_acc", "dir acc")]
NICE = {"train_eval": "TRAIN (eval subset)", "val": "VAL", "test": "TEST", "oofA": "TRAIN scenes held out from the dense head (OOF)"}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def mrow(lab, st, m):
    return [lab, st] + [f3(m.get(k)) for k, _ in GCOLS]


SHORT = {"S1": "S1 grid graph", "S2": "S2 + cascade refiner", "S3": "S3 + learned links", "S4a": "S4a + map prior, exact pose", "S4b": "S4b + map, 1 m noise, no alignment", "S4c": "S4c + map, 2 m noise, aligned", "S5a": "S5a + temporal A (feature fusion)", "S5b": "S5b + temporal B, gated", "S5c": "S5c + temporal B, gated, strict tau", "S5d": "S5d + temporal B, trained", "S4d": "S4d + map, 1 m noise + alignment", "FINAL without map": "FINAL without map (A + refiner)", "FINAL with map": "FINAL with map (exact pose)"}


def short_label(lab):
    for k in sorted(SHORT, key=len, reverse=True):
        if lab.startswith(k):
            return SHORT[k]
    return lab[:40]


def main(mode):
    head = ["system", "set"] + [c[1] for c in GCOLS]
    rows = []
    sparse = jl(f"{OUT}/e32_main/graph_full.json") or {}
    orc = jl(f"{OUT}/oracle_grid.json") or {}
    for st in ("val", "test"):
        if st in sparse:
            rows.append(mrow("S0  sparse only (E32 LaneFormer, no grid)", st.upper(), sparse[st]))
    for st in ("val", "test"):
        d = jl(f"{OUT}/e37_final/graph_{st}_trace.json")
        if d:
            rows.append(mrow("S1  grid graph: E37 dense head + tracer + assembler", st.upper(), d["metrics"]))
    e39 = jl(f"{OUT}/e39_main/metrics.json")
    if e39:
        for st in ("val", "test"):
            if st in e39:
                rows.append(mrow("S2  + cascade refiner (E39), rule links", st.upper(), e39[st]["refined"]))
    lk = jl(f"{OUT}/e38_link/link_eval.json")
    tl = str(CFG.get("tau_l", "0.5"))
    if lk:
        for st, tag in (("val", "refined_val"), ("test", "refined_test")):
            if tag in lk and tl in lk[tag]:
                rows.append(mrow(f"S3  + learned link head (E42), tau_l {tl}", st.upper(), lk[tag][tl]))
    for key, lab in (("e40_clean", "S4a + map prior, exact pose (E40)"), ("e40_noise", "S4b + map prior, 1 m / 2 deg pose noise in training and test, no alignment (E40)"), ("e40_noise_s2_align", "S4c + map prior, test noise 2 m / 4 deg with alignment (E40)")):
        s = jl(f"{OUT}/sys_{key}/metrics.json")
        if s:
            for st in ("val", "test"):
                if st in s:
                    rows.append(mrow(lab + " [refiner plug-in]", st.upper(), s[st]["refined"]))
    for key, lab in (("e41_A", "S5a + temporal A: feature fusion (E41) [refiner plug-in]"), ("e41_Btr", "S5d + temporal B trained: refiner re-trained with propagated proposals (E41)")):
        s = jl(f"{OUT}/sys_{key}/metrics.json")
        if s:
            for st in ("val", "test"):
                if st in s:
                    rows.append(mrow(lab, st.upper(), s[st]["refined"]))
    tb = {st: jl(f"{OUT}/e41_B/tb_e37_final_{st}.json") for st in ("val", "test")}
    for k, lab in (("gate", "S5b + temporal B: query propagation, gated (E41)"), ("gate_tau", "S5c + temporal B: gated + stricter threshold (E41)")):
        for st in ("val", "test"):
            if tb.get(st) and k in tb[st]:
                rows.append(mrow(lab, st.upper(), tb[st][k]["metrics"]))
    for ent in CFG.get("systems", []):
        fn, lab = ent[0], ent[1]
        mode = ent[2] if len(ent) > 2 else "learned"
        d = jl(f"{OUT}/{fn}")
        if not d:
            continue
        for st in ("val", "test"):
            if st in d:
                k = mode if mode in d[st] else ("learned" if "learned" in d[st] else "rule")
                rows.append(mrow(lab, st.upper(), d[st][k]["metrics"]) + ([f3(d[st][k]["stability"]["pred"]["F1"])] if False else []))
    for st in ("val", "test"):
        if st in orc:
            rows.append(mrow("ceiling: oracle field -> rule-based assembler", st.upper(), orc[st]))
    # Diagnose Train vs Val
    h2 = ["component", "TRAIN_EVAL (seen by head and foundation)", "TRAIN scenes unseen by the head (OOF; foundation saw them)", "VAL (unseen)", "TEST (unseen)", "gap TRAIN_EVAL - VAL", "gap OOF - VAL"]
    r2 = []
    d37 = DR.load("e37_final")
    if d37:
        bt = DR.best_tau(d37)
        folds = [DR.load(f"e37_fold{k}") for k in range(3)]
        oof = [f["oof"][DR.best_tau(f)]["F1@1.0"] for f in folds if f and "oof" in f]
        tr, va, te = d37["train_eval"][bt]["F1@1.0"], d37["val"][bt]["F1@1.0"], d37["test"][bt]["F1@1.0"]
        r2.append(["dense head, point F1@1m (final head; OOF = mean of the 3 fold heads on their held-out scenes)", f3(tr), f3(np.mean(oof)) if oof else "-", f3(va), f3(te), f3(tr - va), f3(np.mean(oof) - va) if oof else "-"])
        fsn, bon = jl(f"{OUT}/e30_fold_seen.json"), jl(f"{OUT}/e30_boot.json")
        if fsn and bon:
            r2.append(["dense head, like-for-like with the SAME heads: the three cross-fitted heads at tau 0.6 (first column = their own training scenes, E30)", f3(fsn["mean_F1"]), f3(bon["oof_all"]), f3(bon["val"]["heads_mean"]), f3(bon["test"]["heads_mean"]), f3(fsn["mean_F1"] - bon["val"]["heads_mean"]), f3(bon["oof_all"] - bon["val"]["heads_mean"])])
        gm = [jl(f"{OUT}/e37_final/graph_{st}_trace.json") for st in ("train_eval", "val", "test")]
        go = [jl(f"{OUT}/e37_fold{k}/graph_oof_trace.json") for k in range(3)]
        go = [g["metrics"] for g in go if g]
        if all(gm):
            for key, lab in (("F1@1.0", "grid graph, point F1@1m"), ("route_f", "grid graph, route F"), ("split6_F1", "grid graph, split F1 6m"), ("merge6_F1", "grid graph, merge F1 6m")):
                o = np.mean([g[key] for g in go]) if go else None
                r2.append([lab, f3(gm[0]["metrics"][key]), f3(o) if o is not None else "-", f3(gm[1]["metrics"][key]), f3(gm[2]["metrics"][key]), f3(gm[0]["metrics"][key] - gm[1]["metrics"][key]), f3(o - gm[1]["metrics"][key]) if o is not None else "-"])
    if e39 and "oofA" in e39 and "val" in e39:
        for key, lab in (("F1@1.0", "refined graph, point F1@1m"), ("route_f", "refined graph, route F"), ("split6_F1", "refined graph, split F1 6m"), ("merge6_F1", "refined graph, merge F1 6m")):
            r2.append([lab + " (OOF column = fold 0, IN the refiner's training data)", "-", f3(e39["oofA"]["refined"][key]), f3(e39["val"]["refined"][key]), f3(e39["test"]["refined"][key]), "-", f3(e39["oofA"]["refined"][key] - e39["val"]["refined"][key])])
        cu = e39.get("curve") or []
        if cu:
            r2.append(["refiner training loss (last 100 steps) / parameters", "", f"{cu[-1][1]:.3f}", "", "", "", f"{e39.get('params', 0) / 1e6:.2f} M"])
    # Zeitliche Stabilitaet
    h4 = ["system", "VAL stability", "VAL flicker (1 - stability)", "TEST stability", "TEST flicker"]
    r4 = []
    for key, lab in (("sys_e37_final", "single frame (final head + refiner)"), ("sys_e41_A", "+ temporal A (feature fusion)"), ("sys_e41_Btr", "+ temporal B, refiner trained with propagated proposals"), ("sys_e40_clean", "+ map prior, exact pose"), ("sys_e40_noise", "+ map prior, noise-trained"), ("sys_e43_joint", "+ map prior (exact pose) + temporal A")):
        sj = jl(f"{OUT}/{key}/metrics.json")
        if sj and "val" in sj and "stability" in sj["val"]:
            v, t = sj["val"]["stability"]["pred"]["F1"], sj.get("test", {}).get("stability", {}).get("pred", {}).get("F1")
            r4.append([lab, f3(v), f3(1 - v), f3(t) if t is not None else "-", f3(1 - t) if t is not None else "-"])
    for k, lab in (("plain", "+ temporal B zero-shot, all propagated pieces"), ("gate", "+ temporal B zero-shot, gated")):
        vs = [jl(f"{OUT}/e41_B/tb_e37_final_{st}.json") for st in ("val", "test")]
        if vs[0] and k in vs[0]:
            v = vs[0][k]["stability"]["pred"]["F1"]
            t = vs[1][k]["stability"]["pred"]["F1"] if vs[1] and k in vs[1] else None
            r4.append([lab, f3(v), f3(1 - v), f3(t) if t is not None else "-", f3(1 - t) if t is not None else "-"])
    gv = jl(f"{OUT}/sys_e37_final/metrics.json")
    if gv and "val" in gv:
        r4.append(["ground-truth graphs (reference)", f3(gv["val"]["stability"]["gt_reference"]["F1"]), f3(1 - gv["val"]["stability"]["gt_reference"]["F1"]), f3(gv.get("test", {}).get("stability", {}).get("gt_reference", {}).get("F1")), f3(1 - gv.get("test", {}).get("stability", {}).get("gt_reference", {}).get("F1", 1.0))])
    # Laufzeit
    bj = jl(f"{OUT}/e43/bench.json") or {}
    h3 = ["component", "device", "ms / sample", "comment"]
    r3 = [[k, v.get("device", ""), f"{v['ms']:.1f}", v.get("note", "")] for k, v in bj.items()]
    images = []
    # Balken: Leiter
    ladder = [r for r in rows if r[1] == "VAL" and not r[0].startswith(("ceiling", "S0"))]
    if ladder:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, col, ttl in ((axs[0], 3, "point F1 @ 1 m (VAL)"), (axs[1], 15, "route F (VAL)"), (axs[2], 11, "merge F1 6 m (VAL)")):
            vals = [float(r[col]) if r[col] != "-" else np.nan for r in ladder]
            ax.barh(range(len(ladder)), vals, color="#1f77b4")
            ax.set_yticks(range(len(ladder)))
            ax.set_yticklabels([short_label(r[0]) for r in ladder] if ax is axs[0] else [""] * len(ladder), fontsize=8)
            ax.invert_yaxis()
            lo = np.nanmin(vals)
            ax.set_xlim(max(lo - 0.05, 0), min(np.nanmax(vals) + 0.03, 1.0))
            ax.set_title(ttl, fontsize=9)
            ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        p = f"{FIG}/e43_ladder.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Building blocks, one at a time (VAL; plug-in rows reuse the refiner trained without map / time)."))
    # Galerie + Fehlerfaelle (TEST, Endkonfiguration)
    gf = CFG.get("graphs_test", "").replace("out/loop2/", "")
    if gf and os.path.exists(f"{OUT}/{gf}"):
        d = pickle.load(open(f"{OUT}/{gf}", "rb"))
        ids, gr = d["ids"], d["graphs"]
        gts = DR.gt_graphs("val", ids)
        sc = []
        for n in range(len(ids)):
            r_ = GE.eval_sample(gr[n], gts[n], n, with_route=True, directed=True)
            P, Rc = r_["cnt"][1, 0] / max(r_["cnt"][1, 1], 1), r_["cnt"][1, 2] / max(r_["cnt"][1, 3], 1)
            sc.append(2 * P * Rc / max(P + Rc, 1e-9))
        sc = np.array(sc)
        order = np.argsort(sc)
        good = [int(order[int(q)]) for q in np.linspace(len(order) * 0.5, len(order) - 1, 6)]
        bad = [int(o) for o in order[:6]]
        for nm, sel, cap in (("gallery", good, "TEST gallery (spread over the upper half of the per-sample F1): final system vs ground truth."), ("failures", bad, "The six WORST TEST samples by point F1@1m (not cherry-picked: this is the failure analysis).")):
            fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.8))
            for ax, n in zip(axs.ravel(), sel):
                VG.panel(ax, gr[n], gts[n], title=f"sample {int(ids[n])}  F1@1m {sc[n]:.2f}")
            fig.suptitle("red = prediction (arrows = travel direction, triangles = splits, diamonds = merges), green = ground truth", fontsize=9)
            fig.tight_layout()
            p = f"{FIG}/e43_{nm}.png"
            fig.savefig(p, dpi=90)
            plt.close(fig)
            images.append((p, cap))
    if mode == "figs":
        for r in rows + r2 + r4 + r3:
            print(r)
        return
    t = TEXT.get("e43", {})
    make_report2("E43", t.get("title", "E43"), t.get("question", ""), t.get("setup", []),
                 [("Final system: building blocks added one at a time (thresholds fixed on VAL; TEST never used for a decision)", head, rows), ("Train-versus-validation diagnosis", h2, r2), ("Temporal stability of the output graphs", h4, r4), ("Run time", h3, r3)],
                 images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF2.system_arch)
    print("E43 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
