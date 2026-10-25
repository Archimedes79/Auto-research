"""Hilfen fuer die Berichte E27-E30 (dichte Laeufe aus dense_head.py): Kennzahlen je Lauf/Menge, Auswahl des Schwellwerts auf VAL, Bilder (Heatmap-Ueberlagerung, PR-Kurven, Balken, Lernkurven)."""
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
HOME = os.path.expanduser("~")
SETS = ("train_eval", "val", "test")
NICE = {"train_eval": "TRAIN (eval subset)", "val": "VAL", "test": "TEST"}


def load(name):
    p = f"{OUT}/{name}/metrics.json"
    return json.load(open(p)) if os.path.exists(p) else None


def best_tau(res, key="F1@1.0", on="val"):
    return max(res[on].items(), key=lambda kv: kv[1][key])[0]


def f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def row_at(res, st, tau):
    if st not in res:
        return None
    return res[st].get(tau)


def metrics_row(res, tau, sets=SETS, keys=("F1@1.0",)):
    out = []
    for st in sets:
        r = row_at(res, st, tau)
        for k in keys:
            out.append(f(r[k]) if r else "-")
    return out


def pr_curve(res, st, t="1.0"):
    taus = sorted(res[st].keys(), key=float)
    return [float(x) for x in taus], [res[st][x][f"P@{t}"] for x in taus], [res[st][x][f"R@{t}"] for x in taus]


def plot_pr(ax, runs, st, labels=None, t="1.0"):
    """Praezision gegen Ausbeute (alle Schwellwerte) fuer mehrere Laeufe; Kreuz = auf VAL gewaehlter Schwellwert."""
    for k, (nm, res) in enumerate(runs.items()):
        taus, P, R = pr_curve(res, st, t)
        ax.plot(R, P, "-o", ms=3, label=(labels or {}).get(nm, nm))
        bt = best_tau(res)
        ax.plot([res[st][bt][f"R@{t}"]], [res[st][bt][f"P@{t}"]], "kx", ms=8)
    ax.set_xlabel(f"recall @ {t} m")
    ax.set_ylabel(f"precision @ {t} m")
    ax.set_title(f"{NICE.get(st, st)}: precision-recall over thresholds (x = threshold chosen on VAL)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def load_prob(name, st):
    p = f"{OUT}/{name}/prob_{st}.npy"
    return (np.load(p, mmap_mode="r"), np.load(f"{OUT}/{name}/ids_{st}.npy")) if os.path.exists(p) else (None, None)


def gt_graphs(part, ids):
    G = pickle.load(open(f"{HOME}/exp_gcar_b123/graph_gt_{part}.pkl", "rb"))
    return [G[int(i)] for i in ids]


def overlay(ax, prob, gt, tau, title=""):
    """Heatmap (viridis) + Soll-Mittellinien (weiss) + Skelett der Schwelle (rot) im Fahrzeug-System."""
    import torch

    import dense_head as DH

    ax.imshow(np.asarray(prob, np.float32).T, origin="lower", extent=(-50, 50, -50, 50), cmap="viridis", vmin=0, vmax=1, zorder=0)
    for p in gt["pts"]:
        ax.plot(p[:, 0], p[:, 1], "-", color="white", lw=1.0, alpha=0.95, zorder=2)
    sk = DH.skeleton_points(torch.from_numpy(np.asarray(prob, np.float32))[None], tau)[0]
    if len(sk):
        ax.plot(sk[:, 0], sk[:, 1], ".", color="#ff3b30", ms=1.5, zorder=3)
    ax.plot([0], [0], "w^", ms=5, zorder=4)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def qualitative(names, labels, st, sel, tau, path, ncol_extra=0):
    """Zeilen = Samples, Spalten = Laeufe. sel: Positionen in der Menge."""
    probs = {}
    ids = None
    for nm in names:
        pr, ids_ = load_prob(nm, st)
        probs[nm] = pr
        ids = ids_
    part = "train" if st.startswith("train") else "val"
    gts = gt_graphs(part, ids[sel])
    fig, axs = plt.subplots(len(sel), len(names), figsize=(3.6 * len(names), 3.6 * len(sel)), squeeze=False)
    for r, k in enumerate(sel):
        for c, nm in enumerate(names):
            overlay(axs[r, c], probs[nm][k], gts[r], tau[nm] if isinstance(tau, dict) else tau, f"{labels[nm]} - sample {int(ids[k])}" if r == 0 else f"sample {int(ids[k])}")
    fig.suptitle("heat map (colour), ground-truth centerlines (white lines), skeleton of the thresholded map (red dots)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def bars(ax, groups, series, ylabel, title):
    """groups: Namen auf der x-Achse; series: dict Label -> Werte."""
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for k, (lab, vals) in enumerate(series.items()):
        b = ax.bar(x + (k - (n - 1) / 2) * w, vals, w, label=lab)
        for xi, v in zip(x + (k - (n - 1) / 2) * w, vals):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)


def curves(ax, runs, labels=None):
    for nm, res in runs.items():
        c = np.array(res["curve"])
        ax.plot(c[:, 0], c[:, 1], label=(labels or {}).get(nm, nm))
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
