"""Loop 18 result figures from the stored result files (no GPU): E209, registration + trust (E213), fork pruning (E215), route head with failure frames / joint fine-tune (E214).
Writes reports/source/figures/p18_*.png.  Usage (gcar): fig_p18_results.py"""
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
E17 = f"{R}/runs/e17"
FIG = f"{R}/reports/source/figures"
C = dict(base="#8a8f98", unreg="#d9a441", reg="#1f5fa8", exact="#2a9d4a", none="#6b7a8f", wrong="#a85f9e", a="#1f5fa8", c="#2a9d4a")


def auroc(pos, neg):
    allv = np.concatenate([pos, neg])
    rk = allv.argsort().argsort() + 1.0
    return float((rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def card(path, phase="fail", key="S"):
    return pickle.load(open(f"{E17}/{path}.pkl", "rb"))["cards"][phase][key]


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def fig_regis():
    perts = ["dark", "blur", "cam_front3_off"]
    names = ["night", "blur", "3 cameras off"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.1))
    # (a) pose error
    s = {}
    RE = f"{R}/runs/e18/regis_eval_a"
    for p in ("dark", "blur"):
        tg = f"{p}_maxdecay_0.95"
        for g in ("s1", "s2", "s3"):
            z = np.load(f"{RE}/{tg}_{g}.npz")
            m = z["phase"] == 1
            seen = m & (z["c_acc"] > 1e-6)
            s[f"{tg}_{g}"] = {"fail": dict(none_t=float(np.linalg.norm(z["truth"][m][:, 1:3], axis=1).mean()), single_t=float(np.linalg.norm(z["t"][m] - z["truth"][m][:, 1:3], axis=1).mean()), acc_t=float(np.linalg.norm(z["t_acc"][seen] - z["truth"][seen][:, 1:3], axis=1).mean()))}
        zr, zw = np.load(f"{RE}/{tg}_s1.npz"), np.load(f"{RE}/{tg}_wrong.npz")
        for nm, ph in (("cl", 0), ("fail", 1)):
            mm = zr["phase"] == ph
            s[f"{tg}_auroc_{nm}"] = dict(frame=auroc(zr["trust"][mm], zw["trust"][mm]), acc=auroc(zr["trust_acc"][mm], zw["trust_acc"][mm]))
    sig = ["s1", "s2", "s3"]
    x = np.arange(3)
    w = 0.26
    ax = axs[0]
    for k, (lab, key, col) in enumerate((("map as drawn", "none_t", C["base"]), ("single frame", "single_t", C["unreg"]), ("accumulated (K188)", "acc_t", C["reg"]))):
        ax.bar(x + (k - 1) * w, [np.mean([s[f"{p}_maxdecay_0.95_{g}"]["fail"][key] for p in ("dark", "blur")]) for g in sig], w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["1 m", "2 m", "3 m"])
    ax.set_xlabel("map pose error sigma (translation; yaw 2 sigma deg)", fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "Translation error of the map, failure half\n(night and blur, mean)", "metres")
    # (b) S per map condition
    ax = axs[1]
    cols = [("no map", "mnone", C["none"]), ("2 m error", "m2", C["unreg"]), ("registered, 2 m", "R75_s2", C["reg"]), ("exact map", "m0", C["exact"]), ("wrong map", "mwrong", C["wrong"]), ("wrong, gated", "R75_wrong", "#c9a0c2")]
    w = 0.13
    for k, (lab, key, col) in enumerate(cols):
        vals = []
        for p in perts:
            if key in ("R75_s2", "R75_wrong"):
                path = f"e18_r21b/{p}_gateB_{key}"
            else:
                path = f"e140_gateB/{p}_gateB_{key}"
            vals.append(card(path))
        ax.bar(np.arange(3) + (k - 2.5) * w, vals, w, color=col, label=lab)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names)
    ax.set_ylim(0.55, 0.70)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    style(ax, "Builder + gate, failure half (S, VAL)", "S")
    # (c) AUROC of trust
    ax = axs[2]
    au = [s[f"{p}_maxdecay_0.95_auroc_{ph}"] for p in perts[:2] for ph in ("cl", "fail")]
    lab = ["night clean", "night failure", "blur clean", "blur failure"]
    ax.bar(np.arange(4) - 0.2, [a["frame"] for a in au], 0.38, color=C["unreg"], label="single frame")
    ax.bar(np.arange(4) + 0.2, [a["acc"] for a in au], 0.38, color=C["reg"], label="accumulated")
    ax.axhline(0.5, color="k", lw=0.7)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "Map trust: does it tell the right map from a wrong one?", "AUROC")
    fig.tight_layout()
    fig.savefig(f"{FIG}/p18_registration.png", dpi=140)


def fig_fork():
    res = json.load(open(f"{R}/runs/e18/fork_prune_val/results.json"))["cards"]
    thr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    names = ["champion (no pruning)"] + [f"prune {t}" for t in ("0.1", "0.2", "0.3", "0.4", "0.5")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.1))
    ax = axs[0]
    for key, lab, col in (("J", "J (split / merge F1 at 6 m)", C["reg"]), ("Q", "Q (quality at equal sparsity)", C["exact"]), ("S", "S (composite)", C["unreg"])):
        ax.plot(thr, [res[n][key] for n in names], "o-", color=col, label=lab)
    ax.axhline(0.408, color="k", lw=0.8, ls="--")
    ax.text(0.01, 0.412, "control J 0.408", fontsize=8)
    ax.set_xlabel("junction-head heat below which a fork is un-linked", fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "Fork pruning on all 873 validation frames", "metric")
    ax = axs[1]
    ax.plot(thr, [res[n]["split6_P"] for n in names], "o-", color=C["reg"], label="split precision")
    ax.plot(thr, [res[n]["split6_R"] for n in names], "s--", color=C["reg"], label="split recall")
    ax.plot(thr, [res[n]["merge6_P"] for n in names], "o-", color=C["exact"], label="merge precision")
    ax.plot(thr, [res[n]["merge6_R"] for n in names], "s--", color=C["exact"], label="merge recall")
    ax.set_xlabel("pruning threshold", fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "The loss is precision, not recall", "at 6 m")
    ax = axs[2]
    fp = f"{R}/runs/e18/fork_prune_val/fork_features.npz"
    if os.path.exists(fp):
        z = np.load(fp)
        for kind, nm, col in ((0, "split", C["reg"]), (1, "merge", C["exact"])):
            A = z[f"k{kind}"]
            tp, fpv = A[A[:, 5] == 1, 0], A[A[:, 5] == 0, 0]
            bins = np.linspace(0, 1, 21)
            ax.hist(fpv, bins, density=True, histtype="step", color=col, lw=1.6, ls="--", label=f"{nm}: false forks")
            ax.hist(tp, bins, density=True, histtype="step", color=col, lw=1.6, label=f"{nm}: true forks")
        ax.set_xlabel("junction-head heat at the fork vertex", fontsize=8)
        ax.legend(fontsize=8, frameon=False)
    style(ax, "True and false forks differ in head support", "density")
    fig.tight_layout()
    fig.savefig(f"{FIG}/p18_forkprune.png", dpi=140)


def fig_route():
    E = f"{E17}/e18_j2_route"
    tags = ["dark", "blur", "cam_front3_off"]
    names = ["night", "blur", "3 cameras off"]
    arms = [("original head", "base", None, C["base"]), ("failure-trained head behind the ORIGINAL builder", "origbuilder", None, "#7fb3d5"), ("failure-trained pair, tokens detached (c)", "c", None, C["c"]), ("joint, route loss into builder (a)", "a", None, C["a"])]
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.2))
    w = 0.2
    for k, (lab, sfx, _, col) in enumerate(arms):
        vals, err = [], []
        for t in tags:
            v = []
            for s in (0, 1):
                if sfx == "origbuilder":
                    suf, route = "origbuilder", f"e18_joint_c{s}"
                else:
                    suf = "base" if sfx == "base" else f"{sfx}{s}"
                    route = f"p16_f2hr_route_s{s}" if sfx == "base" else f"e18_joint_{sfx}{s}"
                j = json.load(open(f"{E}/route_{t}_maxdecay_0.95_{suf}.json"))
                v.append(j[f"sigma0_{route}_fail"]["route_success"])
            vals.append(np.mean(v))
            err.append((max(v) - min(v)) / 2)
        ax.bar(np.arange(3) + (k - 1.5) * w, vals, w, yerr=err, color=col, label=lab, capsize=3)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax, "Route success, failure half (spatial hold, no map error; bars = mean of two route-head seeds, whisker = half range)", "route success")
    ax.title.set_fontsize(8.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p18_route_failure_training.png", dpi=140)


def fig_e209():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.1))
    for ax, v, ttl in ((axs[0], "single", "single frame"), (axs[1], "maxdecay_0.95", "spatial hold")):
        ent = [("champion", lambda p: f"e132_norec/{p}_{v}", C["base"]), ("no-map control C2 (unseen)", lambda p: f"e142/{p}_{v}_C2none", C["none"]), ("map builder B2, no map", lambda p: f"e142/{p}_{v}_B2none", C["unreg"]),
               ("map builder B2, exact map", lambda p: f"e142/{p}_{v}_B2m0", C["reg"]), ("same builder, saw night / blur", lambda p: f"e139_b/{p}_{v}_m0", C["exact"])]
        w = 0.16
        for k, (lab, fn, col) in enumerate(ent):
            ax.bar(np.arange(2) + (k - 2) * w, [card(fn(p)) for p in ("dark", "blur")], w, color=col, label=lab)
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(["night", "blur"])
        ax.set_ylim(0.4, 0.7 if v != "single" else 0.65)
        style(ax, f"Unseen failure types, {ttl}: failure half S", "S")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/p18_e209_unseen.png", dpi=140)


if __name__ == "__main__":
    import sys

    which = sys.argv[1:] or ["regis", "fork", "route", "e209"]
    for w in which:
        {"regis": fig_regis, "fork": fig_fork, "route": fig_route, "e209": fig_e209}[w]()
        print("done", w)
