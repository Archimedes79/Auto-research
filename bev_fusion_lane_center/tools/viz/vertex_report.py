"""Result page of the vertex-level cards (E271): quantitative (frontier, per-scene scatter, table) and qualitative (stored frames of `grid_view.py`) in one place.

    wsl bash tools/env/_py14.sh vertex_report.py [--runs e271_a_s0 e271_a_s1] [--control r] [--out runs/e271/report] [--scenes 7 23]

Reads runs/e271/<name>/card.json (written by evaluation/vertex_eval.py) and runs/grid_view/<run>/scene<N>_f*.png (written by tools/viz/grid_view.py; run that first).
Writes <out>/results.png (frontier cov5 / prec1 against vertices per 100 m, per-scene cov5 head against control, per-scene prec1) and <out>/qualitative.png (one stored frame per
scene, the one where the head is closest to its scene mean), <out>/results.md (the table of the figure). Pictures are not in git (runs/); the numbers are in the card's section 3.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def card(name):
    return json.load(open(os.path.join(ROOT, "runs", "e271", name, "card.json")))


def per_scene(c):
    """card -> {scene: (cov5, prec1)}"""
    out = {}
    for s, n in zip(c["scenes"], c["counts"]):
        a = out.setdefault(s, np.zeros(4))
        a += [n["cov"][0], n["cov"][1], n["prec"][0], n["prec"][1]]
    return {s: (a[0] / max(a[1], 1e-9), a[2] / max(a[3], 1e-9)) for s, a in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["e271_a_s0", "e271_a_s1"])
    ap.add_argument("--control", default="r")
    ap.add_argument("--out", default="runs/e271/report")
    ap.add_argument("--scenes", type=int, nargs="*", default=[7, 23])
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    ctl, heads = card(a.control), [card(r) for r in a.runs]
    poly, null = card("r_poly10"), card("null")
    col = ["tab:blue", "tab:cyan", "tab:purple"]

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for k, (m, t) in enumerate((("cov5", "coverage cov5 (higher better)"), ("prec1", "precision prec1 (higher better)"))):
        x = ax[k]
        fr = poly["frontier_by_step_m"]
        x.plot([v["vgt100"] for v in fr.values()], [v[m] for v in fr.values()], "o--", c="tab:orange", label="prototype resampled every 6 ... 20 m")
        for h, c, r in zip(heads, col, a.runs):
            f = h["frontier"]
            x.plot([v["vgt100"] for v in f.values() if v["reached"]], [v[m] for v in f.values() if v["reached"]], "o-", c=c, label=f"head {r[-2:]}")
        x.scatter(ctl["metrics"]["vgt100"], ctl["metrics"][m], marker="s", s=70, c="k", label="r: prototype node ids (control)", zorder=5)
        x.scatter(null["metrics"]["vgt100"], null["metrics"][m], marker="x", s=60, c="tab:red", label="null (random)", zorder=5)
        x.set_xlabel("vertices per 100 m of GT lane")
        x.set_title(t, fontsize=10)
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=7, loc="lower right")
    pc = per_scene(ctl)
    sc = sorted(pc)
    for k, (m, i) in enumerate((("cov5", 0), ("prec1", 1))):
        x = ax[2 + k]
        x.plot([0, 1], [0, 1], c="0.6", lw=1)
        for h, c, r in zip(heads, col, a.runs):
            ph = per_scene(h)
            x.scatter([pc[s][i] for s in sc], [ph[s][i] for s in sc], s=22, c=c, label=f"head {r[-2:]} (22 VAL scenes)")
        x.set_xlabel(f"r {m} per scene")
        x.set_ylabel(f"head {m} per scene")
        x.set_title(f"per scene {m}: above the diagonal = head better", fontsize=10)
        x.set_xlim(0, 1.02), x.set_ylim(0, 1.02), x.grid(alpha=0.3)
    ax[2].legend(fontsize=7, loc="lower right")
    fig.suptitle("E271 grid chain, VAL: head (1 m centre-lane + differentiable sparsification) against the prototype, vertex-level", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "results.png"), dpi=90)
    plt.close(fig)

    # qualitative: one stored frame per scene of the first run (the picture shows head, prototype, GT, assignment, camera)
    imgs = []
    for sn in a.scenes:
        d = os.path.join(ROOT, "runs", "grid_view", a.runs[0])
        fs = sorted(f for f in os.listdir(d) if f.startswith(f"scene{sn}_f") and f.endswith(".png")) if os.path.isdir(d) else []
        if fs:
            imgs.append((sn, plt.imread(os.path.join(d, fs[len(fs) // 2]))))
    if imgs:
        fig, ax = plt.subplots(len(imgs), 1, figsize=(14, 8 * len(imgs)))
        ax = np.atleast_1d(ax)
        for x, (sn, im) in zip(ax, imgs):
            x.imshow(im), x.axis("off"), x.set_title(f"{a.runs[0]}, VAL scene {sn}, middle frame of the stored sequence", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "qualitative.png"), dpi=70)
        plt.close(fig)

    rows = [("null", null), ("r (prototype node ids)", ctl), ("r resampled 10 m", poly)] + [(r, h) for r, h in zip(a.runs, heads)]
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write("| VAL | cov5 | prec1 | vertices / 100 m |\n|---|---|---|---|\n")
        for n, c in rows:
            m = c["metrics"]
            f.write(f"| {n} | {m['cov5']:.3f} | {m['prec1']:.3f} | {m['vgt100']:.2f} |\n")
    print("wrote", os.path.relpath(out, ROOT), os.listdir(out))


if __name__ == "__main__":
    main()
