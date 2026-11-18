"""Where do the errors of a lane graph come from? The error of one frame is split into the metres of lane that are missing
(`fn.*`: ground-truth points with no prediction within tau) and the metres that are false (`fp.*`: prediction points with no
ground truth within tau), and every metre is given ONE cause, by the first rule that fits (order = priority). The pieces add up to the
frame's total error, so they can be summed over frames, scenes or situations: an error budget.

fn (ground truth missed)                                   fp (prediction false)
  unobservable  outside the view of the input (mask given)    shift        the best shift of the whole prediction fixes it
  shift         the best shift of the whole prediction        duplicate    parallel copy of a correct lane
                fixes it                                      spur         short stray piece (< spur_len)
  offset        a parallel prediction within near_factor * tau  offset     parallel ground truth within near_factor * tau
                (displaced, not absent; a lane that only        overshoot  a correct lane continues past the end of the truth
                crosses the missing one does not count)         hallucinated nothing parallel within near_factor * tau: invented
  gap           hole inside a lane that is found elsewhere                 lane (or ground truth that is missing there)
  truncated     prediction stops short of the end / starts late
  missed_near   lane absent within far_m of the ego
  missed_far    lane absent beyond far_m
duplicate and offset need a neighbour with the same direction (|cos| >= `parallel`, directions from the points 4 m before / after).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .geometry import LaneGraph
from .measures import nn_dist, nn_index

CAUSES: Dict[str, Tuple[str, str]] = {          # cause -> (what it is, what to improve)
    "fn.unobservable": ("missed part lies outside the input's view", "evaluation: mask the unobservable part"),
    "fn.shift": ("missed only because the whole prediction is displaced", "model / calibration: registration"),
    "fn.offset": ("a prediction is there but displaced beyond tau (localisation)", "model: localisation accuracy"),
    "fn.gap": ("hole inside a lane that is found elsewhere (broken lane)", "assembly code: linking / pruning"),
    "fn.truncated": ("lane found but the prediction stops short / starts late", "model range or assembly extension"),
    "fn.missed_near": ("lane absent within the near range", "model: recall"),
    "fn.missed_far": ("lane absent beyond the far range", "model: range / resolution at distance"),
    "fp.shift": ("false only because the whole prediction is displaced", "model / calibration: registration"),
    "fp.duplicate": ("parallel copy of a correct lane", "assembly code: de-duplication"),
    "fp.spur": ("short stray piece", "assembly code: pruning"),
    "fp.offset": ("ground truth nearby but displaced beyond tau", "model: localisation accuracy"),
    "fp.overshoot": ("a correct lane continues past the end of the ground truth", "model or assembly extension; or GT ends early"),
    "fp.hallucinated": ("no ground truth nearby: invented lane or ground truth missing there", "model precision, or evaluation: GT incomplete?"),
}
FN = [k for k in CAUSES if k.startswith("fn.")]
FP = [k for k in CAUSES if k.startswith("fp.")]


def target_of(fix: str) -> str:
    """where to work on a cause: 'model' | 'code' | 'evaluation' (or 'model/evaluation' when the fix text names both), read from the lane
    words of its fix text (moved here from Evaluation/probing.py: the core takes the place from Task.causes(), see LaneTask.causes)"""
    f = fix.lower()
    found = [t for t, keys in (("evaluation", ("evaluation", "gt ", "ground truth")), ("code", ("assembly", "code", "pruning", "linking")),
                               ("model", ("model", "calibration"))) if any(k in f for k in keys)]
    return "/".join(found) if found else "model"


def _kernel_scores(tg, tp, P: np.ndarray, G: np.ndarray, shifts: np.ndarray, tau: float, chunk: int = 600_000) -> np.ndarray:
    """for every shift d: sum of clip(1 - dist(P + d, G) / tau, 0, 1) + sum of clip(1 - dist(G - d, P) / tau, 0, 1). All shifts go through
    one KD-tree query per chunk of points instead of one query per shift, and a point farther than tau from the other set is not searched
    for (its kernel is 0 either way: the query answers inf, and clip(1 - inf) = 0). The same distances and the same row sums as the
    loop over shifts, so the same scores bit for bit (checked against the loop in tests/test_lane_task.py)."""
    out = np.zeros(len(shifts))
    for sign, tree, A in ((1.0, tg, P), (-1.0, tp, G)):
        per = max(chunk // max(len(A), 1), 1)                                # shifts per query
        for i in range(0, len(shifts), per):
            sh = shifts[i:i + per]
            d = tree.query((A[None, :, :] + sign * sh[:, None, :]).reshape(-1, 2), distance_upper_bound=tau)[0].reshape(len(sh), len(A))
            out[i:i + per] += np.clip(1 - d / tau, 0, 1).sum(axis=1)
    return out


def best_shift(P: np.ndarray, G: np.ndarray, tau: float, smax: float, step: float):
    """translation (du, dv) of the prediction that fits both sets best (triangular kernel of width tau: the peak is the true displacement,
    a count within tau would be flat over a whole range of shifts)
    -> (du, dv, matched P at the best shift, matched G at the best shift, matched P at 0, matched G at 0), 'matched' within tau"""
    from scipy.spatial import cKDTree
    if len(P) == 0 or len(G) == 0:
        return 0.0, 0.0, 0, 0, 0, 0
    tg, tp = cKDTree(G), cKDTree(P)
    ks = np.arange(-smax, smax + 1e-9, step)
    shifts = np.array([(du, dv) for du in ks for dv in ks])                 # the order of the double loop: the tie rule below depends on it
    scores = _kernel_scores(tg, tp, P, G, shifts, tau)
    best = (-1.0, 0.0, 0.0)
    for (du, dv), s in zip(shifts, scores):
        if s > best[0] + 1e-9 or (abs(s - best[0]) <= 1e-9 and np.hypot(du, dv) < np.hypot(best[1], best[2])):
            best = (s, du, dv)
    counts = lambda d: (int((tg.query(P + d)[0] <= tau).sum()), int((tp.query(G - d)[0] <= tau).sum()))
    mp, mg = counts(np.array(best[1:]))
    mp0, mg0 = counts(np.zeros(2))
    return best[1], best[2], mp, mg, mp0, mg0


def _prediction_ends_there(dp, dg, idx: np.ndarray, a: int, b: int, near: float, ds: float) -> bool:
    """a missed run at one end of a ground-truth lane is a TRUNCATION when the prediction that covers the neighbouring points ends
    (starts) right there; if that prediction piece runs on elsewhere, the run is a missed branch / lane"""
    if len(dp) == 0:
        return False
    at_end = b == len(idx)
    j = idx[a - 1] if at_end else idx[b]                       # last covered ground-truth point next to the run
    k = int(np.argmin(np.linalg.norm(dp.xy - dg.xy[j], axis=1)))
    ln = dp.line[k]
    s_all = dp.s[dp.line == ln]
    return bool((s_all.max() - dp.s[k]) <= near + ds) if at_end else bool(dp.s[k] <= near + ds)


def _ends(d, ds: float) -> np.ndarray:
    """per dense point: is it (within a spacing of) the start or end of its polyline"""
    if not len(d):
        return np.zeros(0, bool)
    top = {int(l): d.s[d.line == l].max() for l in np.unique(d.line)}
    return np.array([d.s[i] <= 0.75 * ds or d.s[i] >= top[int(d.line[i])] - 0.75 * ds for i in range(len(d))])


def _secant(d, ds: float, half: float = 4.0) -> np.ndarray:
    """direction of every dense point from the points `half` metres before and after it on its own polyline (steady against jitter of single vertices)"""
    out = np.zeros((len(d), 2))
    k = max(int(round(half / ds)), 1)
    for ln in np.unique(d.line):
        idx = np.flatnonzero(d.line == ln)
        idx = idx[np.argsort(d.s[idx])]
        p = d.xy[idx]
        v = p[np.minimum(np.arange(len(p)) + k, len(p) - 1)] - p[np.maximum(np.arange(len(p)) - k, 0)]
        out[idx] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return out


def _lateral(A: np.ndarray, B, j: np.ndarray, is_end: np.ndarray) -> np.ndarray:
    """is the offset of the points A from their nearest point of B (index j) sideways? False when that nearest point is an end of its
    polyline and A lies beyond it along the track: then the difference is the missing / extra LENGTH, not a displacement"""
    if len(A) == 0 or len(B) == 0:
        return np.zeros(len(A), bool)
    w, t = A - B.xy[j], B.tan[j]
    along = np.abs((w * t).sum(1))
    side = np.abs(w[:, 0] * t[:, 1] - w[:, 1] * t[:, 0])
    return ~(is_end[j] & (along > side))


def _runs(flag: np.ndarray):
    """(start, stop) of the runs of True in a 1-d bool array"""
    if not len(flag):
        return []
    d = np.diff(np.r_[0, flag.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def error_budget(pred: LaneGraph, gt: LaneGraph, ds: float, tau: float, near_factor: float = 3.0, far_m: float = 40.0,
                 box=None, visible=None, grid=None, shift_max: float = 2.0, shift_step: float = 0.25, shift_min: float = 0.5,
                 shift_gain: float = 0.10, dup_r: float = 1.5, spur_len: float = 6.0, origin=(0.0, 0.0), parallel: float = 0.8) -> Dict[str, float]:
    """metres of lane per cause for one frame (all CAUSES keys, zero when absent) + 'shift_u', 'shift_v', 'shift_gain'.
    `visible` (bool image on `grid`): cells the input can observe; ground-truth points outside are 'unobservable'."""
    dg, dp = gt.dense(ds, box), pred.dense(ds, box)
    G, P = dg.xy, dp.xy
    out = {k: 0.0 for k in CAUSES}
    near = near_factor * tau
    d_gp, j_gp = nn_index(G, P)
    d_pg, j_pg = nn_index(P, G)
    fn, fp = d_gp > tau, d_pg > tau
    lat_g, lat_p = _lateral(G, dp, j_gp, _ends(dp, ds)), _lateral(P, dg, j_pg, _ends(dg, ds))
    # 'displaced' needs a neighbour that runs the same way: a lane that only crosses the missing one (an intersection) is no localisation error
    tg, tp_ = _secant(dg, ds), _secant(dp, ds)
    par_g = np.abs((tg * tp_[j_gp]).sum(1)) >= parallel if len(P) and len(G) else np.zeros(len(G), bool)
    par_p = np.abs((tp_ * tg[j_pg]).sum(1)) >= parallel if len(P) and len(G) else np.zeros(len(P), bool)
    # a displaced prediction: the best rigid translation of this frame
    du, dv, mp, mg, mp0, mg0 = best_shift(P, G, tau, shift_max, shift_step)
    tot = max(len(P) + len(G), 1)
    gain = (mp + mg - mp0 - mg0) / tot
    # a displacement explains the errors only if BOTH sets match better afterwards (a duplicate or an extra lane straddling the
    # truth lets one set match better at the cost of the other)
    shifted = (np.hypot(du, dv) >= shift_min and gain >= shift_gain and (mp - mp0) >= 0.05 * max(len(P), 1)
               and (mg - mg0) >= 0.05 * max(len(G), 1))
    out.update(shift_u=float(du) if shifted else 0.0, shift_v=float(dv) if shifted else 0.0, shift_gain=float(gain) if shifted else 0.0)
    sh_g = (nn_dist(G, P + [du, dv]) <= tau) if shifted else np.zeros(len(G), bool)
    sh_p = (nn_dist(P + [du, dv], G) <= tau) if shifted else np.zeros(len(P), bool)

    # ---- missed ground truth
    cause = np.full(len(G), "", object)
    if visible is not None and grid is not None and len(G):
        r, c, ok = grid.cell(G)
        vis = np.ones(len(G), bool)                              # outside the image: not known to be unobservable
        vis[ok] = np.asarray(visible, bool)[r[ok], c[ok]]
        cause[fn & ~vis] = "fn.unobservable"
    cause[fn & (cause == "") & sh_g] = "fn.shift"
    cause[fn & (cause == "") & (d_gp <= near) & lat_g & par_g] = "fn.offset"
    miss = fn & (cause == "")                                  # absent: nothing within near
    covered = ~fn | (cause == "fn.shift") | (cause == "fn.offset")      # a prediction is there (right, displaced or shifted)
    rng = np.hypot(G[:, 0] - origin[0], G[:, 1] - origin[1]) if len(G) else np.zeros(0)
    for ln in np.unique(dg.line):
        idx = np.flatnonzero(dg.line == ln)
        idx = idx[np.argsort(dg.s[idx])]
        idx = idx[cause[idx] != "fn.unobservable"]               # the unobservable points are neutral for the lane's structure
        mk, cv = miss[idx], covered[idx]
        for a, b in _runs(mk):
            interior = a > 0 and b < len(idx)
            touches = (a == 0) != (b == len(idx))                # exactly one end
            if interior and cv[:a].any() and cv[b:].any():
                lab = "fn.gap"
            elif touches and cv.any() and _prediction_ends_there(dp, dg, idx, a, b, near, ds):
                lab = "fn.truncated"
            else:
                lab = None                                       # absent (a missed branch included): near or far by range
            for j in idx[a:b]:
                cause[j] = lab or ("fn.missed_far" if rng[j] >= far_m else "fn.missed_near")
    for k in FN:
        out[k] = float((cause == k).sum() * ds)

    # ---- false prediction
    cp = np.full(len(P), "", object)
    cp[fp & sh_p] = "fp.shift"
    tp_pts = ~fp
    if tp_pts.any() and fp.any():
        from scipy.spatial import cKDTree
        t = cKDTree(P[tp_pts])
        lid, ttp = dp.line[tp_pts], tp_[tp_pts]
        for j in np.flatnonzero(fp & (cp == "")):
            for q in t.query_ball_point(P[j], dup_r):
                if lid[q] != dp.line[j] and abs(float(tp_[j] @ ttp[q])) >= parallel:
                    cp[j] = "fp.duplicate"
                    break
    plen = {int(l): float((dp.line == l).sum() * ds) for l in np.unique(dp.line)}
    for j in np.flatnonzero(fp & (cp == "")):
        if plen[int(dp.line[j])] < spur_len:
            cp[j] = "fp.spur"
    cp[fp & (cp == "") & (d_pg <= near) & lat_p & par_p] = "fp.offset"
    for ln in np.unique(dp.line):
        idx = np.flatnonzero(dp.line == ln)
        idx = idx[np.argsort(dp.s[idx])]
        free = fp[idx] & (cp[idx] == "")
        for a, b in _runs(free):
            if (a == 0) != (b == len(idx)) and (~fp[idx]).any():          # a run at one end of a line that is right elsewhere
                cp[idx[a:b]] = "fp.overshoot"
    cp[fp & (cp == "")] = "fp.hallucinated"
    for k in FP:
        out[k] = float((cp == k).sum() * ds)
    out["fn_total"], out["fp_total"] = float(fn.sum() * ds), float(fp.sum() * ds)
    return out
