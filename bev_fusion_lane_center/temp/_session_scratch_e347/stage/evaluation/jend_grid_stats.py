"""E347 (E344 4b as a committed check): how do the GT entry / exit ends of the junctions sit on the grid of a head? Ground truth only: no model, no features, no tuning.

    wsl bash tools/env/_py14.sh jend_grid_stats.py [--split val_clean] [--res 2.0 1.0] [--out file.json]      # the lane GT of the split's frames, the ends of the junction targets (`jgt.junction_gt`: what `jgt.fields` / `jgt.end_cells` rasterise)
    wsl bash tools/env/_py14.sh jend_grid_stats.py selftest

Per kind (entry / exit) and per cell size (2 m = the 2 m head of E347, 1 m = the head of E307), over the junction frames: `ends` = all GT ends of the kind (what the scorer's recall counts), `off_grid` = those outside the 100 m map (no cell: no target can hold them);
among the ends ON the grid the ends that SHARE a cell with another end of their kind (`shared`, every member of a shared cell) and the ends beyond the first per cell (`dropped`: what a one-to-one target loses; `shared` is up to twice as large); and for the NEAREST other end of
the kind in the frame (all ends, off-grid ones too: a physical distance): its distance (median, the shares below 1 / 2 / 3.5 m) and where its cell lies relative to the end's cell: the `same` cell, an `adjacent` one (Chebyshev distance 1: the 3 x 3 peak rule of the heat keeps
only one of two ends there), or `further`. The scratch of E344 4b (`gt_end_grid.py`) counted the off-grid ends in the cell statistics too (about twice the `dropped` of the on-grid ends); its distances are the ones printed here.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jgt as JG  # noqa: E402


def frame_stats(E, res, half=50.0):
    """the ends E (m, >= 2) of one kind in one frame on the grid of cell size `res` -> dict(n = all ends, off_grid, n_in = the ends on the grid, shared, dropped (both among the ends on the grid), nn (m,) the distance of every end to its nearest other end
    (empty when m < 2), rel (m,) 0 = that end's cell is the cell of its nearest end, 1 = an adjacent cell (Chebyshev 1), 2 = further; an end off the grid has a virtual cell beyond the map)"""
    E = np.asarray(E, float).reshape(-1, E.shape[-1] if np.ndim(E) > 1 else 4)
    cells, inside = JG.end_cell_index(E, int(round(2 * half / res)), res, half)
    n = len(E)
    _, inv, cnt = np.unique(cells[inside, 0] * 100000 + cells[inside, 1], return_inverse=True, return_counts=True)
    out = dict(n=n, off_grid=int((~inside).sum()), n_in=int(inside.sum()), shared=int((cnt[inv] > 1).sum()), dropped=int((cnt - 1).sum()), nn=np.zeros(0), rel=np.zeros(0, int))
    if n > 1:
        D = np.linalg.norm(E[:, None, :2] - E[None, :, :2], axis=-1) + np.eye(n) * 1e9
        j = D.argmin(1)
        out["nn"] = D[np.arange(n), j]
        cheb = np.abs(cells - cells[j]).max(1)
        out["rel"] = np.where(cheb == 0, 0, np.where(cheb == 1, 1, 2))
    return out


def aggregate(per_frame):
    """list of `frame_stats` of one kind -> the totals and shares"""
    n, n_in = sum(f["n"] for f in per_frame), sum(f["n_in"] for f in per_frame)
    nn = np.concatenate([f["nn"] for f in per_frame]) if per_frame else np.zeros(0)
    rel = np.concatenate([f["rel"] for f in per_frame]) if per_frame else np.zeros(0, int)
    sh = lambda m: float(m.mean()) if len(m) else float("nan")  # noqa: E731
    shared, dropped = sum(f["shared"] for f in per_frame), sum(f["dropped"] for f in per_frame)
    return dict(ends=n, frames=len(per_frame), off_grid=n - n_in, off_grid_share=(n - n_in) / max(n, 1), shared=shared, shared_share=shared / max(n_in, 1), dropped=dropped, dropped_share=dropped / max(n_in, 1),
                nearest_median_m=float(np.median(nn)) if len(nn) else float("nan"), nearest_lt1=sh(nn < 1.0), nearest_lt2=sh(nn < 2.0), nearest_lt3_5=sh(nn < 3.5), nearest_same_cell=sh(rel == 0), nearest_adjacent_cell=sh(rel == 1),
                nearest_further=sh(rel == 2))


def stats(frames_ends, ress=(2.0, 1.0)):
    """frames_ends: list of (ent (m, 4), ext (n, 4)) per junction frame -> {res: {"ent": aggregate, "ext": aggregate}}"""
    return {str(res): {k: aggregate([frame_stats(fe[i], res) for fe in frames_ends if len(fe[i])]) for i, k in enumerate(("ent", "ext"))} for res in ress}


def load_ends(split):
    """the ends of the junction targets of every frame of the split with a junction and a stored lane GT -> list of (ent, ext), the number of frames"""
    import lane_gt_data as LD
    ds = LD.QData(split, map_mode="error", pixels=False)
    out = []
    for k in ds.ks:
        jg = JG.junction_gt(ds.lg.load(ds.tokens[int(ds.bd.rows[k])]))
        if jg is not None:
            out.append((jg["ent"], jg["ext"]))
    return out, len(ds)


def show(res):
    for r, d in res.items():
        for k, a in d.items():
            print(f"  {r} m grid, {k}: {a['ends']} ends in {a['frames']} junction frames, {a['off_grid']} ({100 * a['off_grid_share']:.1f} %) off the map; of the others sharing a cell with another end {a['shared']} ({100 * a['shared_share']:.2f} %), beyond the first per cell {a['dropped']} ({100 * a['dropped_share']:.2f} %); "
                  f"nearest other end: median {a['nearest_median_m']:.2f} m, < 1 m {100 * a['nearest_lt1']:.1f} %, < 2 m {100 * a['nearest_lt2']:.1f} %, < 3.5 m {100 * a['nearest_lt3_5']:.1f} %; its cell: same {100 * a['nearest_same_cell']:.1f} %, adjacent {100 * a['nearest_adjacent_cell']:.1f} %, further {100 * a['nearest_further']:.1f} %")


def selftest():
    E = np.array([[0.5, 0.5, 0.0, 1.0], [1.5, 0.5, 0.0, 1.0], [3.5, 0.5, 0.0, 1.0], [70.0, 0.0, 0.0, 1.0]])                  # the last one is off the 100 m map
    f2, f1 = frame_stats(E, 2.0), frame_stats(E, 1.0)
    assert f2["n"] == 4 and f2["off_grid"] == 1 and f2["n_in"] == 3 and f2["shared"] == 2 and f2["dropped"] == 1 and f1["shared"] == 0 and f1["dropped"] == 0, (f2, f1)
    assert np.allclose(f2["nn"], [1.0, 1.0, 2.0, 66.5], atol=0.01) and f2["rel"].tolist() == [0, 0, 1, 2], f2                             # 2 m cells 25, 25, 26 along x: end 3 -> end 2 is an adjacent cell; the off-grid end counts as a neighbour (all ends), its own nearest end is 66.5 m away
    assert f1["rel"].tolist() == [1, 1, 2, 2], f1                                                                               # 1 m cells 50, 51, 53: ends 1, 2 are adjacent, end 3's nearest (end 2) is two cells away
    a = aggregate([f2, frame_stats(E[:1], 2.0)])
    assert a["ends"] == 5 and a["off_grid"] == 1 and a["shared"] == 2 and a["dropped"] == 1 and abs(a["shared_share"] - 2 / 4) < 1e-9 and a["frames"] == 2 and abs(a["nearest_median_m"] - 1.5) < 1e-9 and abs(a["nearest_lt2"] - 2 / 4) < 1e-9 \
        and abs(a["nearest_adjacent_cell"] - 1 / 4) < 1e-9, a                                                                   # distances 1, 1, 2, 66.5: median 1.5, two of four below 2 m
    r = stats([(E, E[:2])])
    assert r["2.0"]["ent"]["ends"] == 4 and r["2.0"]["ent"]["off_grid"] == 1 and r["2.0"]["ext"]["shared"] == 2 and r["1.0"]["ext"]["shared"] == 0
    print("jend_grid_stats selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="stats", choices=("stats", "selftest"))
    ap.add_argument("--split", default="val_clean")
    ap.add_argument("--res", type=float, nargs="+", default=[2.0, 1.0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        fe, n_frames = load_ends(a.split)
        res = stats(fe, a.res)
        print(f"{a.split}: {n_frames} frames, {len(fe)} with a junction")
        show(res)
        if a.out:                                                                                      # relative to the problem folder (the launcher runs from the folder of the script)
            json.dump(dict(split=a.split, frames=n_frames, junction_frames=len(fe), stats=res), open(os.path.join(ROOT, a.out), "w"), indent=1)
