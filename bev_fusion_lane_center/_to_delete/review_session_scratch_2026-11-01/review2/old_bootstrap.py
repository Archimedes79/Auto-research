"""Scene-level bootstrap for sparse-graph metrics: how large is the noise of S, F1@1, J, vertices per 100 m - and is the difference between two systems real?

    rec_a = SB.frame_records(preds_a, gts)          # per-frame counts, computed once
    rec_b = SB.frame_records(preds_b, gts)
    SB.pooled(rec_a)                                # == sparse_metrics.card_any(preds_a, gts) for the same frames (tested)
    SB.interval(rec_a, scenes, B=1000)              # mean and 95 % interval of every metric over resampled scenes
    SB.paired(rec_a, rec_b, scenes, B=1000)         # difference a - b with 95 % interval, share of draws a > b, verdict against the noise band

Frames of one scene are correlated, so whole scenes are drawn with replacement (the same draw for both systems = paired). Pooled metrics are re-aggregated from the per-frame counts of every draw, exactly as `graph_eval.aggregate` does for the full set.
Verdict: "better" if the interval of the S difference lies above the noise band (0.010), "worse" if below minus the band, "tie" if the interval contains values within the band, otherwise "within noise band but resolvable" is reported as "tie (resolved, small)".
"""
import numpy as np

import graph_eval as GE
import sparse_metrics as SM

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


def _scene_draws(scenes, B, seed):
    scenes = np.asarray(scenes)
    ids, inv = np.unique(scenes, return_inverse=True)
    members = [np.nonzero(inv == s)[0] for s in range(len(ids))]
    rng = np.random.RandomState(seed)
    for _ in range(B):
        pick = rng.randint(0, len(ids), len(ids))
        yield np.concatenate([members[s] for s in pick])


def interval(rec, scenes, B=1000, seed=0, keys=KEYS):
    base = pooled(rec)
    draws = [pooled(rec, idx) for idx in _scene_draws(scenes, B, seed)]
    return {k: dict(value=base[k], lo=float(np.percentile([d[k] for d in draws], 2.5)), hi=float(np.percentile([d[k] for d in draws], 97.5)), sd=float(np.std([d[k] for d in draws]))) for k in keys if k in base}


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
    """a - b per metric: value on all frames, 95 % interval over scene draws, share of draws with a > b"""
    base_a, base_b = pooled(rec_a), pooled(rec_b)
    da = {k: [] for k in keys}
    for idx in _scene_draws(scenes, B, seed):
        ma, mb = pooled(rec_a, idx), pooled(rec_b, idx)
        for k in keys:
            if k in ma:
                da[k].append(ma[k] - mb[k])
    out = {}
    for k in keys:
        if k not in base_a or not da[k]:
            continue
        d = np.array(da[k])
        out[k] = dict(a=base_a[k], b=base_b[k], diff=base_a[k] - base_b[k], lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p_a_better=float((d > 0).mean()))
    if "S" in out:
        out["S"]["verdict"] = verdict(out["S"]["lo"], out["S"]["hi"])
    return out
