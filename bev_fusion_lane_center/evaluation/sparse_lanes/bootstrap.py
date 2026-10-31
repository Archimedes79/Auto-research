"""Scene-level bootstrap for sparse-graph metrics: how large is the noise of S, F1@1, J, vertices per 100 m - and is the difference between two systems real?

    rec_a = SB.frame_records(preds_a, gts)          # per-frame counts, computed once
    rec_b = SB.frame_records(preds_b, gts)
    SB.pooled(rec_a)                                # == sparse_metrics.card_any(preds_a, gts) for the same frames (tested)
    SB.interval(rec_a, scenes, B=1000)              # mean and 95 % interval of every metric over resampled scenes
    SB.paired(rec_a, rec_b, scenes, B=1000)         # difference a - b (a = variant, b = control) with 95 % interval, share of draws a > b, verdict against the noise band

Frames of one scene are correlated, so whole scenes are drawn with replacement (the same draw for both systems = paired). Pooled metrics are re-aggregated from the per-frame counts of every draw, exactly as `graph_eval.aggregate` does for the full set.
The draws and the intervals are the ONE implementation of the repository, `autoresearch/evalsuite/stats.py` (scene_draws, paired_draws, interval_draws; D24, 2026-10-31); its draw sequence is the one this module always used, so stored numbers are reproduced exactly. Draws that cannot be scored (non-finite) are dropped and a result then carries `n_valid`.
Verdict on S: "better" if the interval of the S difference lies above the noise band (0.010), "worse" if below minus the band, otherwise a "tie (...)" that says whether the interval is positive, negative or contains 0.
"""
import os
import sys

import numpy as np

import bflc_paths as BP  # project paths from the location of this checkout
import graph_eval as GE
import sparse_metrics as SM

if os.path.join(BP.REPO, "autoresearch") not in sys.path:
    sys.path.append(os.path.join(BP.REPO, "autoresearch"))
from evalsuite import stats as ST  # noqa: E402  (the shared scene bootstrap)

BAND = 0.010
KEYS = ("S", "Q", "F1@1.0", "F1@0.5", "F1o2o@1.0", "J", "Sp", "vert100", "len_ratio", "false_len_frac", "lanes_found", "split6_F1", "merge6_F1", "R@1.0", "P@1.0")


def _vstats(g, use_nodes, eps=0.3):
    """(vertices, polyline length in m) of one graph: unique node ids when the graphs carry them, otherwise Douglas-Peucker vertices (same rules as vert100 / vert100_dp)"""
    nv, tot = 0, 0.0
    if use_nodes:
        ids = set()
        for k, p in enumerate(g["pts"]):
            q = np.asarray(p, np.float64)
            if len(q) < 2:
                continue
            tot += float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
            nd = g["nodes"][k] if g.get("nodes") is not None and k < len(g["nodes"]) else None
            ids.update((k, j) for j in range(len(q))) if nd is None else ids.update(int(i) for i in nd)
        return len(ids), tot
    for p in g["pts"]:
        q = GE.resample(np.asarray(p, np.float64), 0.5)
        if len(q) < 3:
            continue
        q = q[(np.abs(q) < 48.0).all(1)]
        if len(q) < 3:
            continue
        tot += float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
        nv += len(SM._dp_keep(q, eps))
    return nv, tot


def frame_records(preds, gts):
    use_nodes = all(g.get("nodes") is not None for g in preds)
    rec = []
    for i, (p, g) in enumerate(zip(preds, gts)):
        nv, tot = _vstats(p, use_nodes)
        rec.append(dict(rs=GE.eval_sample(p, g, i, with_route=False), nv=nv, length=tot))
    return rec


def pooled(rec, idx=None):
    """metrics of the frames `idx` (default all), pooled like card()"""
    sel = rec if idx is None else [rec[i] for i in idx]
    r = GE.aggregate([x["rs"] for x in sel])
    m = {k: float(r[k]) for k in SM.KEEP if k in r}
    m["vert100"] = 100.0 * sum(x["nv"] for x in sel) / max(sum(x["length"] for x in sel), 1e-9)
    m["len_ratio"] = float(sum(x["rs"]["n_p"] for x in sel) / max(sum(x["rs"]["n_g"] for x in sel), 1))  # predicted / ground-truth length: parallel duplicates show up here and in F1o2o, not in S
    m["S"], m["J"], m["Sp"] = SM.composite(m)
    m["Q"] = SM.quality(m)
    return m


def _lane_fields(v, names):
    """the fields this module has always returned, plus n_valid when some draws were not scorable (non-finite: the interval is then
    conditional on the n_valid scorable draws)"""
    out = {k: v[k] for k in names}
    if v["n_valid"] < v["n_draws"]:
        out["n_valid"] = v["n_valid"]
    return out


def interval(rec, scenes, B=1000, seed=0, keys=KEYS):
    """mean and 95 % interval of every metric over resampled scenes (one system)"""
    res = ST.interval_draws(lambda idx: pooled(rec, idx), ST.scene_draws(scenes, B, seed), keys)
    return {k: _lane_fields(v, ("value", "lo", "hi", "sd")) for k, v in res.items()}


def verdict(lo, hi, band=BAND):
    if lo > band:
        return "better"
    if hi < -band:
        return "worse"
    if lo > 0:
        return "tie (positive, inside the noise band)"
    if hi < 0:
        return "tie (negative, inside the noise band)"
    return "tie (interval contains 0)"


def paired(rec_a, rec_b, scenes, B=1000, seed=0, keys=KEYS):
    """a - b per metric (a = variant, b = control): value on all frames, 95 % interval over scene draws, p_a_better = share of draws
    with a - b > 0 (NOT direction-aware: for vert100 / false_len_frac a share near 1 means a is worse); verdict on S"""
    res = ST.paired_draws(lambda idx: pooled(rec_a, idx), lambda idx: pooled(rec_b, idx), ST.scene_draws(scenes, B, seed), keys)
    out = {k: dict(_lane_fields(v, ("a", "b", "diff", "lo", "hi")), p_a_better=v["p_gt0"]) for k, v in res.items()}
    if "S" in out:
        out["S"]["verdict"] = verdict(out["S"]["lo"], out["S"]["hi"])
    return out
