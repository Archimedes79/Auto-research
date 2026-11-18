"""Bericht E30: Obergrenze auf der Trainingsmenge, Luecke Train -> Val/Test, Ursachen und Anpassungen. Aufruf: make_e30.py [figs|report]
Laeufe: e30_s16 (526 Samples), e30_s4 (2104), e27_gcar (4207), e30_reg (Gewichtsabfall + Dropout), e30_flip (Spiegel-Augmentierung), e30_valA / e30_trainsub (Kontrolle gesehen/ungesehen).
Text: loop2/text_e27_e30.py TEXT['e30']."""
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
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e27_e30.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e27_e30.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT", {})
CURVE = [("e30_s16", 526), ("e30_s4", 2104), ("e27_gcar", 4207), ("e37_final", 8414)]


OUTD = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"


def jl(p):
    import json

    return json.load(open(p)) if os.path.exists(p) else None


def f1pr(c):
    c = np.asarray(c)
    P, R = c[:, 0].sum() / max(c[:, 1].sum(), 1), c[:, 2].sum() / max(c[:, 3].sum(), 1)
    return 2 * P * R / max(P + R, 1e-9), P, R


def r_at(res, st, tau):
    return res[st][tau] if st in res else None


def main(mode):
    runs = {n: DR.load(n) for n in ["e30_s16", "e30_s4", "e30_s4_diverged", "e27_gcar", "e30_base2", "e37_final", "e30_reg", "e30_flip", "e30_valA", "e30_trainsub", "e30_scene11", "e28_dec32"]}
    runs = {k: v for k, v in runs.items() if v is not None and ("val" in v or "valB" in v)}
    # Lernkurve
    head = ["training samples (rows)", "run", "TRAIN_EVAL F1@1m", "VAL F1@1m", "TEST F1@1m", "gap TRAIN-VAL", "gap TRAIN-TEST", "VAL P@1m", "VAL R@1m", "VAL tau"]
    rows = []
    xs, tr, va, te = [], [], [], []
    for nm, n in CURVE:
        if nm not in runs:
            continue
        res = runs[nm]
        bt = DR.best_tau(res)
        a, b, c = res["train_eval"][bt]["F1@1.0"], res["val"][bt]["F1@1.0"], res["test"][bt]["F1@1.0"]
        rows.append([str(n), nm, f3(a), f3(b), f3(c), f3(a - b), f3(a - c), f3(res["val"][bt]["P@1.0"]), f3(res["val"][bt]["R@1.0"]), bt])
        xs.append(n), tr.append(a), va.append(b), te.append(c)
    for nm, lab in (("e30_s4_diverged", "2104, FIRST attempt (lr 2e-3, loss jumped at step ~1500; discarded)"), ("e30_base2", "4207 rows, 3,000 steps, stabilised recipe (lr 6e-4) = baseline of the adjustments"), ("e30_reg", "4207 rows + weight decay 1e-2 + dropout 0.1"), ("e30_flip", "4207 rows + mirror augmentation (x -> -x, p 0.5)")):
        if nm in runs:
            res = runs[nm]
            bt = DR.best_tau(res)
            a, b, c = res["train_eval"][bt]["F1@1.0"], res["val"][bt]["F1@1.0"], res["test"][bt]["F1@1.0"]
            rows.append([lab, nm, f3(a), f3(b), f3(c), f3(a - b), f3(a - c), f3(res["val"][bt]["P@1.0"]), f3(res["val"][bt]["R@1.0"]), bt])
    # Kontrolle gesehen/ungesehen
    h2 = ["training scenes (about 440 samples in each run)", "foundation saw these scenes' map labels?", "VAL-B F1@1m", "VAL-B P / R @1m", "TEST F1@1m", "TEST P / R @1m", "tau (chosen on VAL-B)"]
    r2 = []
    for nm, lab, seen in (("e30_trainsub", "TRAIN, every 19th sample (about 440 samples from all 212 scenes)", "yes"), ("e30_scene11", "11 TRAIN scenes (scene number divisible by 19; all their samples)", "yes"), ("e30_valA", "VAL-A (11 official-val scenes, all their samples)", "no")):
        if nm in runs and "valB" in runs[nm]:
            res = runs[nm]
            bt = max(res["valB"].items(), key=lambda kv: kv[1]["F1@1.0"])[0]
            v, t = res["valB"][bt], res["test"][bt]
            r2.append([lab, seen, f3(v["F1@1.0"]), f"{f3(v['P@1.0'])} / {f3(v['R@1.0'])}", f3(t["F1@1.0"]), f"{f3(t['P@1.0'])} / {f3(t['R@1.0'])}", bt])
    # Zerlegung der Luecke mit identischen Koepfen + Szenen-Bootstrap + je Stadt
    cnt, fs, bo, bl = jl(f"{OUTD}/e30_boot_counts.json"), jl(f"{OUTD}/e30_fold_seen.json"), jl(f"{OUTD}/e30_boot.json"), jl(f"{OUTD}/e30_boot_loc.json")
    h3 = ["scenes on which the three cross-fitted heads (E37 folds, tau 0.6) are evaluated", "seen by the head?", "seen by the foundation?", "F1@1m (mean of the heads; OOF pooled)", "P / R @1m", "change from the row above"]
    r3, r4, h4 = [], [], ["quantity", "value"]
    zer = []
    if cnt and fs and bo:
        own = (fs["mean_F1"], np.mean([fs[f"fold{k}"]["P"] for k in range(3)]), np.mean([fs[f"fold{k}"]["R"] for k in range(3)]))
        oof = f1pr(np.concatenate([np.array(cnt[f"e37_fold{k}_oof"]["counts"]) for k in range(3)]))
        va_h = [f1pr(cnt[f"e37_fold{k}_val"]["counts"]) for k in range(3)]
        te_h = [f1pr(cnt[f"e37_fold{k}_test"]["counts"]) for k in range(3)]
        vam = tuple(np.mean([x[i] for x in va_h]) for i in range(3))
        tem = tuple(np.mean([x[i] for x in te_h]) for i in range(3))
        zer = [("own training scenes", own[0]), ("held-out TRAIN scenes (OOF)", oof[0]), ("VAL", vam[0]), ("TEST", tem[0])]
        prev = None
        for lab, a_, b_, v in (("own training scenes (every 4th row of the 2/3 of the scenes each head was trained on)", "yes", "yes", own), ("held-out TRAIN scenes (OOF; 212 scenes in total)", "no", "yes (map labels)", oof), ("VAL (22 official-val scenes)", "no", "no", vam), ("TEST (21 official-val scenes)", "no", "no", tem)):
            r3.append([lab, a_, b_, f3(v[0]), f"{f3(v[1])} / {f3(v[2])}", "-" if prev is None else f"{v[0] - prev:+.3f}"])
            prev = v[0]
        b = bo
        r4 = [["scene-level F1@1m of the 212 held-out TRAIN scenes: mean / SD / min / max", f"{b['scene_f1_mean']:.3f} / {b['scene_f1_sd']:.3f} / {b['scene_f1_min']:.3f} / {b['scene_f1_max']:.3f}"],
              ["2,000 random draws of 22 of these scenes: mean F1 (SD), 5-95 % range", f"{b['val']['draw_mean']:.3f} ({b['val']['draw_sd']:.3f}), {b['val']['draw_p5']:.3f} - {b['val']['draw_p95']:.3f}"],
              ["VAL (22 scenes), mean of the three heads: F1 and its percentile among the draws", f"{b['val']['heads_mean']:.3f}; {100 * b['val']['rank']:.1f} %"],
              ["TEST (21 scenes), mean of the three heads: F1 and its percentile among 21-scene draws", (f"{b['test']['heads_mean']:.3f}; {100 * b['test']['rank']:.1f} % (below all 2,000 draws)" if b['test']['rank'] == 0 else f"{b['test']['heads_mean']:.3f}; {100 * b['test']['rank']:.1f} %")],
              ["scene-level F1@1m of the VAL / TEST scenes: mean / SD / min / max", f"VAL {b['val_scene_f1']['mean']:.3f} / {b['val_scene_f1']['sd']:.3f} / {b['val_scene_f1']['min']:.3f} / {b['val_scene_f1']['max']:.3f};  TEST {b['test_scene_f1']['mean']:.3f} / {b['test_scene_f1']['sd']:.3f} / {b['test_scene_f1']['min']:.3f} / {b['test_scene_f1']['max']:.3f}"]]
        if bl:
            r4.append(["per city, held-out TRAIN scenes -> VAL -> TEST: Boston-Seaport", f"{bl['oof_loc0'][1]:.3f} ({bl['oof_loc0'][0]} rows) -> {bl['val_loc0'][1]:.3f} ({bl['val_loc0'][0]}) -> {bl['test_loc0'][1]:.3f} ({bl['test_loc0'][0]})"])
            r4.append(["per city: Singapore-One-North", f"{bl['oof_loc1'][1]:.3f} ({bl['oof_loc1'][0]} rows) -> {bl['val_loc1'][1]:.3f} ({bl['val_loc1'][0]}) -> {bl['test_loc1'][1]:.3f} ({bl['test_loc1'][0]})"])
            r4.append(["per city: other Singapore map in TRAIN only (no VAL / TEST scenes there)", f"{bl['oof_loc3'][1]:.3f} ({bl['oof_loc3'][0]} rows)"])
    images = []
    if len(xs) >= 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(xs, tr, "-o", label="TRAIN_EVAL (seen)")
        axs[0].plot(xs, va, "-s", label="VAL")
        axs[0].plot(xs, te, "-^", label="TEST")
        axs[0].set_xscale("log", base=2)
        axs[0].set_xlabel("number of training samples")
        axs[0].set_ylabel("F1 @ 1 m")
        axs[0].set_title("learning curve (dense head, threshold chosen on VAL)", fontsize=9)
        axs[0].grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        if r2:
            w = 0.38
            axs[1].bar(np.arange(len(r2)) - w / 2, [float(r[2]) for r in r2], w, label="VAL-B", color="#1f77b4")
            axs[1].bar(np.arange(len(r2)) + w / 2, [float(r[4]) for r in r2], w, label="TEST", color="#ff7f0e")
            axs[1].set_xticks(np.arange(len(r2)))
            axs[1].set_xticklabels(["212 scenes\n(seen)", "11 scenes\n(seen)", "11 scenes\n(unseen)"][: len(r2)], fontsize=8)
            axs[1].set_ylim(0.7, 0.9)
            axs[1].set_ylabel("F1 @ 1 m")
            axs[1].set_title("control: about 440 training samples, scenes seen / unseen by the foundation", fontsize=9)
            axs[1].legend(fontsize=8)
            axs[1].grid(axis="y", alpha=0.3)
        else:
            axs[1].axis("off")
        fig.tight_layout()
        p = f"{FIG}/e30_curve.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Left: learning curve (dense head). Right: control at about 440 training samples - 212 seen scenes vs 11 seen scenes vs 11 unseen scenes (VAL-B and TEST)."))
    adj = [n for n in ("e30_base2", "e30_reg", "e30_flip") if n in runs]
    if adj:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3.8))
        lab_ = {"e30_base2": "baseline (same recipe)", "e30_reg": "weight decay 1e-2 + dropout 0.1", "e30_flip": "mirror augmentation"}
        series = {s_: [runs[n][st][DR.best_tau(runs[n])]["F1@1.0"] for n in adj] for s_, st in (("TRAIN_EVAL", "train_eval"), ("VAL", "val"), ("TEST", "test"))}
        DR.bars(ax, [lab_[n] for n in adj], series, "F1 @ 1 m", "adjustments against the train-val gap (4,207 rows, 3,000 steps)")
        ax.set_ylim(0.78, 0.96)
        fig.tight_layout()
        p3 = f"{FIG}/e30_adjust.png"
        fig.savefig(p3, dpi=100)
        plt.close(fig)
        images.append((p3, "Adjustments against the train-validation gap (dense head; identical recipe, 4,207 rows, 3,000 steps, one seed)."))
    if zer and bo and cnt:
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
        axs[0].bar(range(len(zer)), [z[1] for z in zer], color=["#2ca02c", "#98df8a", "#1f77b4", "#ff7f0e"])
        for i, z in enumerate(zer):
            axs[0].text(i, z[1] + 0.003, f"{z[1]:.3f}", ha="center", fontsize=8)
        axs[0].set_xticks(range(len(zer)))
        axs[0].set_xticklabels(["own training\nscenes", "held-out TRAIN\nscenes (OOF)", "VAL", "TEST"], fontsize=8)
        axs[0].set_ylim(0.75, 0.97)
        axs[0].set_ylabel("F1 @ 1 m (tau 0.6)")
        axs[0].set_title("same three heads: from seen to unseen scenes", fontsize=9)
        axs[0].grid(axis="y", alpha=0.3)
        rng = np.random.default_rng(1)
        oc = np.concatenate([np.array(cnt[f"e37_fold{k}_oof"]["counts"]) for k in range(3)])
        osc = np.concatenate([np.array(cnt[f"e37_fold{k}_oof"]["scene"]) for k in range(3)])
        us = np.unique(osc)
        per = {u: oc[osc == u].sum(0) for u in us}
        dr = [f1pr(np.sum([per[u] for u in rng.choice(us, 22, replace=False)], 0)[None])[0] for _ in range(2000)]
        axs[1].hist(dr, bins=40, color="#98df8a", label="22 random held-out TRAIN scenes (2,000 draws)")
        axs[1].axvline(bo["val"]["heads_mean"], color="#1f77b4", lw=2, label=f"VAL {bo['val']['heads_mean']:.3f}")
        axs[1].axvline(bo["test"]["heads_mean"], color="#ff7f0e", lw=2, label=f"TEST {bo['test']['heads_mean']:.3f}")
        axs[1].set_xlabel("pooled F1 @ 1 m of a 22-scene subset")
        axs[1].set_title("is the drop a sampling effect of 22 scenes?", fontsize=9)
        axs[1].legend(fontsize=7)
        fig.tight_layout()
        p2 = f"{FIG}/e30_decomposition.png"
        fig.savefig(p2, dpi=100)
        plt.close(fig)
        images.append((p2, "Left: the three cross-fitted heads on their own training scenes, on held-out TRAIN scenes (OOF), on VAL and on TEST. Right: scene bootstrap - the VAL / TEST value against 2,000 random 22-scene subsets of the held-out TRAIN scenes."))
    if mode == "figs":
        for r in rows:
            print(r)
        for r in r2:
            print(r)
        return
    t = TEXT.get("e30", {})
    make_report2("E30", t.get("title", "E30"), t.get("question", ""), t.get("setup", []), [("Learning curve and adjustments (dense head, 4000 steps unless noted)", head, rows), ("Control: number of training scenes vs foundation-seen / unseen scenes, at equal number of samples (evaluated on VAL-B = the other 11 val scenes, and TEST)", h2, r2)] + ([("Decomposition of the gap with identical heads: seen scenes -> held-out TRAIN scenes -> VAL -> TEST", h3, r3), ("Scene bootstrap and per-city check of the step 'held-out TRAIN scenes -> VAL / TEST'", h4, r4)] if r3 else []),
                 images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=lambda ax: AF.dense_arch(ax))
    print("E30 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
