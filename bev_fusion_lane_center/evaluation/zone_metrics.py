"""E282: the ZONE metrics of a sparse node row (card E282, "New metrics"). Pure numpy / scipy, no model, no torch.

    zs = qzones.zones(g, sparse=(0.3, 7.5))                     # the zones of one lane GT frame (training/grid/qzones.py)
    c = frame_counts(zs, qgraph_eval.LaneRef(g).length, P)      # P (n, 2) = the delivered nodes in metres (x right, y forward) -> the ADDITIVE counts of the frame
    pool([c, ...])                                              # the numbers of a set of frames (counts summed, then ratios: the project's micro pooling)
    bootstrap_diff(counts_a, counts_b, "free_zone_exact")       # a - b, paired frame bootstrap (same frames, same order)
    matched_threshold(frames_e, ref_lens)                       # the read-out threshold at which a head delivers about 14.4 nodes per 100 m of GT lane

Zones (qzones): a FIXED zone is an anchor of the sparse target (event node, first / last node of a piece: one cell), a FREE zone is the stretch of lane of a Douglas-Peucker vertex (any cell of it may
carry the node). Membership of a DELIVERED node is by POSITION, not by its slot cell: it belongs to the zone of the nearest DENSE GT node of its frame if that node is within ZONE_TOL (2.0 m, the
scorer's node tolerance); no dense node within 2.0 m, or the nearest one in no zone (a lane cell where no node is demanded), means OFF-zone.
Numbers (from the pooled counts; NaN when the denominator is 0, an empty head has off_zone NaN):
  free_zone_recall   share of the free zones with at least one delivered node         free_zone_exact   share with exactly one
  fixed_zone_recall  the same for the fixed zones (anchors)                           fixed_zone_exact
  zone_recall / zone_exact   the same for all zones together
  off_zone           share of the delivered nodes that are off-zone
  nodes_per_100m     delivered nodes per 100 m of GT lane, the convention of E276 (`qgraph_eval.node_counts` `dens`): ALL delivered nodes (also those outside the 48 m box of the metrics) over the
                     GT lane length inside the box, `qgraph_eval.LaneRef(g).length`, passed in as `ref_len`
frame_counts returns dict(n_free, n_fixed (zones), free_rec, free_exact, fixed_rec, fixed_exact (zones with >= 1 / exactly 1 delivered node), n_deliv, n_off, dens = [n_deliv, ref_len]).
Tests: evaluation/tests/test_zone_metrics.py (hand cases, additivity, bootstrap, matched threshold, the density against qgraph_eval; wsl bash tools/env/_py14.sh test_zone_metrics.py).
"""
import numpy as np
from scipy.spatial import cKDTree

ZONE_TOL = 2.0                                                                     # m: a delivered node is in the zone of the nearest dense GT node within this distance (qgraph_eval.NODE_TOL)
COUNTS = ("n_free", "n_fixed", "free_rec", "free_exact", "fixed_rec", "fixed_exact", "n_deliv", "n_off")
COLS = COUNTS + ("dens_nodes", "dens_len")                                         # the columns of the additive count table


def frame_counts(zs, ref_len, P):
    """additive counts of one frame; zs = qzones.zones(g, sparse), ref_len = qgraph_eval.LaneRef(g).length, P = the delivered node positions (n, 2)"""
    P = np.asarray(P, float).reshape(-1, 2)
    pos, zone_of, kind = zs["nd"]["pos"], zs["zone_of"], zs["kind"]
    zone = -np.ones(len(P), int)                                                   # the zone of every delivered node, -1 = off-zone
    if len(P) and len(pos):
        d, i = cKDTree(pos).query(P)
        zone = np.where(d <= ZONE_TOL, zone_of[i], -1)
    n = np.bincount(zone[zone >= 0], minlength=zs["n_zones"])                      # delivered nodes per zone
    free, fixed = kind == 1, kind == 0
    return dict(n_free=int(free.sum()), n_fixed=int(fixed.sum()), free_rec=int((n[free] >= 1).sum()), free_exact=int((n[free] == 1).sum()),
                fixed_rec=int((n[fixed] >= 1).sum()), fixed_exact=int((n[fixed] == 1).sum()), n_deliv=len(P), n_off=int((zone < 0).sum()), dens=[len(P), float(ref_len)])


def _table(cs):
    """(frames, len(COLS)) float table of a list of count dicts"""
    return np.array([[c[k] for k in COUNTS] + list(c["dens"]) for c in cs], float).reshape(len(cs), len(COLS))


def _numbers(v):
    """the numbers from one row of summed counts"""
    c = dict(zip(COLS, v))
    r = lambda a, b: float(a) / float(b) if b > 0 else float("nan")  # noqa: E731
    nz = c["n_free"] + c["n_fixed"]
    return dict(free_zone_recall=r(c["free_rec"], c["n_free"]), free_zone_exact=r(c["free_exact"], c["n_free"]),
                fixed_zone_recall=r(c["fixed_rec"], c["n_fixed"]), fixed_zone_exact=r(c["fixed_exact"], c["n_fixed"]),
                zone_recall=r(c["free_rec"] + c["fixed_rec"], nz), zone_exact=r(c["free_exact"] + c["fixed_exact"], nz),
                off_zone=r(c["n_off"], c["n_deliv"]), nodes_per_100m=100.0 * r(c["dens_nodes"], c["dens_len"]))


def pool(cs):
    """the numbers of a set of frames: the counts are summed, then the ratios taken (an empty set: all NaN)"""
    return _numbers(_table(cs).sum(0))


def bootstrap_diff(counts_a, counts_b, metric, n=2000, seed=0):
    """a - b of a `pool` key with a paired frame bootstrap: the frames are resampled with replacement (the same draw for both lists, which must hold the same frames in the same order), the pool of
    both is recomputed. -> dict(mean, median of the n differences; lo90, hi90 = their 5th and 95th percentile (a two-sided 90 % interval); point = the difference on the full set). Resamples in which
    the metric is NaN (a denominator of 0) are left out"""
    ta, tb = _table(counts_a), _table(counts_b)
    assert len(ta) == len(tb) and len(ta) > 0, "paired bootstrap: the same frames in the same order"
    point = _numbers(ta.sum(0))[metric] - _numbers(tb.sum(0))[metric]
    idx = np.random.RandomState(seed).randint(0, len(ta), size=(n, len(ta)))
    d = np.array([_numbers(ta[i].sum(0))[metric] - _numbers(tb[i].sum(0))[metric] for i in idx])
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(mean=float("nan"), lo90=float("nan"), median=float("nan"), hi90=float("nan"), point=float(point))
    return dict(mean=float(d.mean()), lo90=float(np.percentile(d, 5)), median=float(np.median(d)), hi90=float(np.percentile(d, 95)), point=float(point))


def matched_threshold(frames_e, ref_lens, target=14.4, tol=2.0, grid=np.arange(0.05, 0.96, 0.01)):
    """the read-out threshold of the grid at which the head delivers the density closest to `target` nodes per 100 m of GT lane (nodes with e >= thr over the summed GT lane lengths; equal
    distances: the lowest threshold). frames_e = the existence probabilities of every frame (any shape), ref_lens = the GT lane length of every frame (`LaneRef(g).length`).
    -> dict(thr, dens, ok): ok = the density is within `tol` of the target (False: no threshold of the grid matches, thr is still the closest)"""
    e = np.sort(np.concatenate([np.asarray(x, float).reshape(-1) for x in frames_e]))
    thr = np.round(np.asarray(grid, float), 9)                                     # 0.3 must stay 0.3 (the scorer reads e >= thr)
    dens = 100.0 * (len(e) - np.searchsorted(e, thr, side="left")) / float(np.sum(ref_lens))
    i = int(np.argmin(np.abs(dens - target)))
    return dict(thr=float(thr[i]), dens=float(dens[i]), ok=bool(abs(dens[i] - target) <= tol))
