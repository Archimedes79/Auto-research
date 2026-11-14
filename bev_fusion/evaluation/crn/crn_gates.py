"""crn_gates - the two CPU gates of the CRN export that need no devkit (card t26, G0, export gate (a) and (b)). Env `gcar`, from the bev_fusion folder.

    python evaluation/crn/crn_gates.py overlay RECORDS.pkl OUT.png [--n 20] [--min-score 0.3] [--x-max 50]
    python evaluation/crn/crn_gates.py tf32 DUMP_A.pkl DUMP_B.pkl [--thr 0.1] [--min-score 0.3] [--match 1.0]

overlay (a): `n` frames spread evenly over the record that crn_convert.py wrote (front_compare format: CRN's boxes and our GT boxes of the same frames in one ego frame), drawn from above
(forward up, left on the left): GT black, CRN coloured (vehicle red, two-wheeler orange, pedestrian magenta) with a tick at the front edge, the field of view of the camera and the
50 m sector. Printed next to it: per frame the GT boxes of the sector and how many of them have a CRN box of the same group within 2 m; overall the share. By eye: boxes on the cars, ticks
pointing the way they drive; a mirrored or rotated picture is the bug this gate is for.
tf32 (b): two `crn_dump.pkl` of the SAME samples (crn_export.py with and without --tf32): for every box of A with score >= min-score the nearest box of B of the same class (greedy by A's
score, B's boxes of any score, within --match m); PASS when no box is without a partner and the largest centre difference (x y z, CRN's ego frame) is <= --thr (0.1 m).
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "..", "autoresearch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from evaluation.metrics import FRONT_HALF_FOV, in_front  # noqa: E402

COLOURS = {"vehicle": "tab:red", "two_wheeler": "tab:orange", "pedestrian": "tab:purple"}


# ------------------------------------------------------------------------------------------------ (b) TF32 on / off
def compare_dumps(a, b, min_score=0.3, match=1.0):
    """a, b = the `samples` of two dumps (same tokens) -> (centre differences of the matched boxes, number of boxes of A compared, number without a partner)"""
    if set(a) != set(b):
        raise ValueError(f"the dumps hold different samples ({len(set(a) & set(b))} common of {len(a)} / {len(b)})")
    diffs, n_cmp, n_lost = [], 0, 0
    for tok in sorted(a):
        A, B = a[tok], b[tok]
        for lab in np.unique(A["labels"]):
            ia = np.flatnonzero((A["labels"] == lab) & (A["scores"] >= min_score))
            ia = ia[np.argsort(-A["scores"][ia])]
            cb = B["boxes"][B["labels"] == lab][:, :3].astype(float)
            free = np.ones(len(cb), bool)
            for i in ia:
                n_cmp += 1
                d = np.linalg.norm(cb - A["boxes"][i, :3].astype(float), axis=1) if len(cb) else np.zeros(0)
                d = np.where(free, d, np.inf)
                j = int(np.argmin(d)) if len(d) else -1
                if j < 0 or d[j] > match:
                    n_lost += 1
                    continue
                free[j] = False
                diffs.append(float(d[j]))
    return np.array(diffs), n_cmp, n_lost


def tf32_gate(path_a, path_b, thr=0.1, min_score=0.3, match=1.0):
    with open(path_a, "rb") as f:
        A = pickle.load(f)
    with open(path_b, "rb") as f:
        B = pickle.load(f)
    ta, tb = A["meta"].get("tf32"), B["meta"].get("tf32")
    print(f"A: {path_a} (TF32 {ta}), B: {path_b} (TF32 {tb}), {len(A['samples'])} samples" + ("" if ta != tb else "   NOTE: the same TF32 setting in both: this measures the run-to-run noise"))
    d, n_cmp, n_lost = compare_dumps(A["samples"], B["samples"], min_score, match)
    worst = float(d.max()) if len(d) else float("nan")
    ok = n_lost == 0 and len(d) > 0 and worst <= thr
    print(f"{n_cmp} boxes of A with score >= {min_score:g}: {len(d)} matched within {match:g} m, {n_lost} without a partner")
    if len(d):
        print(f"centre difference of the matched: median {np.median(d):.4f} m, p99 {np.percentile(d, 99):.4f} m, max {worst:.4f} m")
    print(f"gate (b): {'PASS' if ok else 'FAIL'} - largest centre difference {worst:.4f} m against {thr:g} m, {n_lost} boxes lost")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ (a) boxes over our GT
def corners(b):
    """4 corners (x, y) of a det / GT box: a circle (pedestrian GT has a radius) is returned as None"""
    if "radius" in b:
        return None
    c, s, l, w = math.cos(b["yaw"]), math.sin(b["yaw"]), b["length"] / 2, b["width"] / 2
    return [(b["cx"] + c * x - s * y, b["cy"] + s * x + c * y) for x, y in ((l, w), (l, -w), (-l, -w), (-l, w))]


def hits(dets, gts, x_max, min_score, radius=2.0):
    """(GT boxes of the sector, how many have a det of the same group within `radius` m)"""
    g = [b for b in gts if in_front(b, x_max)]
    d = [b for b in dets if b["score"] >= min_score]
    return len(g), sum(any(x["group"] == b["group"] and math.hypot(x["cx"] - b["cx"], x["cy"] - b["cy"]) <= radius for x in d) for b in g)


def draw_box(ax, b, colour, lw):
    """one box from above (horizontal axis = y, vertical = x): outline, and a tick from the centre to the middle of the front edge; a circle for a pedestrian"""
    from matplotlib.patches import Circle, Polygon
    pts = corners(b)
    if pts is None or b["group"] == "pedestrian":
        ax.add_patch(Circle((b["cy"], b["cx"]), b["radius"] if "radius" in b else max(b["length"], b["width"]) / 2, fill=False, ec=colour, lw=lw))
        return
    ax.add_patch(Polygon([(y, x) for x, y in pts], closed=True, fill=False, ec=colour, lw=lw))
    ax.plot([b["cy"], (pts[0][1] + pts[1][1]) / 2], [b["cx"], (pts[0][0] + pts[1][0]) / 2], "-", color=colour, lw=lw)


def draw(ax, dets, gts, min_score, x_max):
    for b in gts:
        draw_box(ax, b, "k", 2.0)
    for b in dets:
        if b["score"] >= min_score:
            draw_box(ax, b, COLOURS[b["group"]], 1.0)
    for sign in (-1, 1):                                                       # the camera's field of view
        ax.plot([0, sign * x_max * math.tan(FRONT_HALF_FOV)], [0, x_max], ":", color="grey", lw=0.6)
    ax.plot([-x_max * math.tan(FRONT_HALF_FOV), x_max * math.tan(FRONT_HALF_FOV)], [x_max, x_max], ":", color="grey", lw=0.6)


def overlay(records, out_png, n=20, min_score=0.3, x_max=50.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(records, "rb") as f:
        rec = pickle.load(f)
    frames = [(s, k, fr) for s, lst in rec["det"].items() for k, fr in enumerate(lst[0.0])]
    m = min(n, len(frames))
    sel = [frames[int(round(i * (len(frames) - 1) / max(1, m - 1)))] for i in range(m)]            # m frames spread evenly, all of them when there are no more than n
    cols = 5
    rows = -(-len(sel) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 4.4 * rows), squeeze=False)
    tot_g = tot_h = 0
    for ax, (scene, k, (dets, gts)) in zip(axes.ravel(), sel):
        draw(ax, dets, gts, min_score, x_max)
        ng, nh = hits(dets, gts, x_max, min_score)
        tot_g, tot_h = tot_g + ng, tot_h + nh
        ax.set_xlim(28, -28)
        ax.set_ylim(-3, 62)
        ax.set_aspect("equal")
        ax.set_title(f"{scene}[{k}] GT {ng} hit {nh} CRN {sum(d['score'] >= min_score for d in dets)}", fontsize=8)
        ax.tick_params(labelsize=6)
        print(f"{scene}[{k}]: GT boxes in the sector {ng}, with a CRN box of the same group within 2 m {nh}")
    for ax in axes.ravel()[len(sel):]:
        ax.axis("off")
    fig.suptitle(f"CRN (coloured, score >= {min_score:g}) over our GT (black), {len(sel)} frames; sector x < {x_max:g} m; forward up, left on the left", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    print(f"{len(sel)} frames: {tot_h} of {tot_g} GT boxes of the sector have a CRN box within 2 m ({tot_h / max(tot_g, 1):.0%}) -> {out_png}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overlay")
    o.add_argument("records")
    o.add_argument("out")
    o.add_argument("--n", type=int, default=20)
    o.add_argument("--min-score", type=float, default=0.3)
    o.add_argument("--x-max", type=float, default=50.0)
    t = sub.add_parser("tf32")
    t.add_argument("dump_a")
    t.add_argument("dump_b")
    t.add_argument("--thr", type=float, default=0.1)
    t.add_argument("--min-score", type=float, default=0.3)
    t.add_argument("--match", type=float, default=1.0)
    a = p.parse_args(argv)
    if a.cmd == "overlay":
        return overlay(a.records, a.out, a.n, a.min_score, a.x_max)
    return tf32_gate(a.dump_a, a.dump_b, a.thr, a.min_score, a.match)


if __name__ == "__main__":
    raise SystemExit(main())
