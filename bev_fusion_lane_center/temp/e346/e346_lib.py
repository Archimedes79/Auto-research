"""E346 measurement code (card experiments/E346_one-graph-code-ceiling): the ceiling of B's node + link code (qtargets / qformat / qgraph_eval) on the VAL ground truth. Scratch: it only READS the
frozen lane files (E343 digest), the cell size is changed at runtime, never in a file. The driver is `e346_arm.py`.

    with cell_size(1.0): ...                      # qtargets.N / RES for the block (restored after); check_cell_size(res) asserts that every consumer sees it
    rec = frame_record(arm, frame)                # one frame of arm g0 / g1 / l2 / l1: scorer counts + the breakdown of the card (a) - (e) + the collision count
    metrics = numbers(pool(recs))                 # flat {name: float}: the card's numbers

Arms: g0 = the old graph GT scored against itself; g1 = the UNDECODED lane GT polylines (edges = the lane GT's own `succ`); l2 / l1 = `qformat.oracle_prediction(frame.lane())` -> `qgraph_eval.readout`
on the 2 m / 1 m grid. All are scored by `qgraph_eval.graph_counts` against the old graph GT (the scorer's reference), so numbers are those of `graph_metrics`.
Kinds: 0 lane, 1 connector. A lane GT point is a connector point when its `segments` entry says so; an old GT polyline has a `kind`.
"""
import contextlib
import importlib.util
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import graph_eval as GE
import qformat
import qgraph_eval as QE
import qtargets as QT

ARMS = ("g0", "g1", "l2", "l1")
GRID = {"l2": 2.0, "l1": 1.0}                      # metres per cell of the decoded arms
TAU = 1.0                                          # the F1@1.0 radius
FOUND = 0.8                                        # lanes found: share of a GT lane within TAU of the graph (graph_eval.eval_sample)
KINDS = ("lane", "connector")
CLASSES = ("lane_short", "lane_long", "conn_short", "conn_long")     # links by the kind of their GT path (more than half connector points = connector) and by the 8 m chord rule
COLL_DIST, COLL_ANG = 0.3, 30.0                    # card: two pieces in a cell are not one chain node when their points are farther apart than this or their headings differ by more than this


# ------------------------------------------------------------------------------------------------------------------------------ cell size
@contextlib.contextmanager
def cell_size(res):
    """qtargets on a `res` m grid for the block: N and RES are module constants that `cell_of` / `centre_of` read at call time, so setting them on the module is seen by every consumer that holds the
    module (`import qtargets as QT`: qformat, qgraph_eval, lane_gt_data, qzones); a default argument bound at import (`grid_targets(n=N)`, `zone_maps(n=QT.N)`) is NOT: pass n explicitly"""
    n = 2 * QT.HALF / res
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"cell size {res} m does not divide the {2 * QT.HALF} m window")
    old = QT.N, QT.RES
    QT.N, QT.RES = int(round(n)), float(res)
    try:
        yield
    finally:
        QT.N, QT.RES = old


def check_cell_size(res):
    """AssertionError unless every consumer of the grid sees `res` m cells: no stale second copy of qtargets, the cell function and the two consumers on the path (qformat.oracle_prediction,
    qgraph_eval.LaneRef) give the cells of `res` on a toy lane, `grid_targets(n=N)` has N x N cells"""
    n = int(round(2 * QT.HALF / res))
    assert (QT.N, QT.RES) == (n, float(res)), (QT.N, QT.RES, n, res)
    held = {name: m.QT for name, m in list(sys.modules.items()) if m is not None and getattr(getattr(m, "QT", None), "__name__", "").startswith("qtargets")}
    stale = [name for name, q in held.items() if q is not QT]
    assert not stale, f"consumers hold another copy of qtargets: {stale}"
    copies = [name for name, m in list(sys.modules.items()) if m is not None and os.path.basename(getattr(m, "__file__", "") or "") == "qtargets.py" and m is not QT]
    assert not copies, f"a second module object for qtargets.py: {copies}"
    xy = np.array([[-49.9, -49.9], [0.0, 0.0], [49.9, 49.9], [0.3, -0.7]])
    assert np.array_equal(QT.cell_of(xy), np.floor((xy + QT.HALF) / res).astype(int)), "cell_of does not use the cell size"
    assert np.abs(QT.centre_of(QT.cell_of(xy)) - xy).max() <= res / 2 + 1e-9, "centre_of does not use the cell size"
    g, _ = QE.toy_lane_gt()
    pred = qformat.oracle_prediction(g)                                             # qformat -> qtargets.node_table
    for who, nt in (("qformat.oracle_prediction", pred["table"]), ("qgraph_eval.LaneRef", QE.LaneRef(g).nt)):
        pos, cell = nt["pos"], nt["cell"]
        assert np.array_equal(cell, np.clip(np.floor((pos + QT.HALF) / res).astype(int), 0, n - 1)), f"{who}: node cells are not the cells of {res} m"
        assert abs(len(pos) - 80.0 / res) <= 6, f"{who}: {len(pos)} nodes on the toy lane (about {80.0 / res:.0f} expected at {res} m)"
    gt = QT.grid_targets(pred["table"], n=QT.N)
    assert gt["exists"].shape == (n, n) and gt["exists"].sum() == len(pred["table"]["pos"]), "grid_targets: wrong grid"


def stale_qtargets():
    """a second copy of qtargets.py at its defaults (a consumer that bound the constants at import): the mutation of the cell-size check"""
    spec = importlib.util.spec_from_file_location("qtargets_stale", QT.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------------------------------------ small geometry
def poly_dist(q, a):
    """(n,) distance of every point of q (n, 2) to the polyline a (m, 2)"""
    q, a = np.asarray(q, float).reshape(-1, 2), np.asarray(a, float).reshape(-1, 2)
    if len(a) == 0 or len(q) == 0:
        return np.full(len(q), np.inf)
    if len(a) == 1:
        return np.linalg.norm(q - a[0], axis=1)
    p0, d = a[:-1], np.diff(a, axis=0)
    t = np.clip(((q[:, None] - p0[None]) * d[None]).sum(2) / np.maximum((d * d).sum(1), 1e-12)[None], 0.0, 1.0)
    return np.linalg.norm(q[:, None] - (p0[None] + t[..., None] * d[None]), axis=2).min(1)


def point_kinds(lane):
    """per piece an (n,) int array: 1 where a `connector` segment of the lane GT covers the point, else 0"""
    out = []
    for p in lane["pieces"]:
        k = np.zeros(len(p["pts"]), int)
        for s in p["segments"]:
            if s["kind"] == "connector":
                k[s["i0"]: s["i1"] + 1] = 1
        out.append(k)
    return out


def point_weights(q):
    """metres of polyline each point stands for (half of each adjacent segment)"""
    w = np.zeros(len(q))
    if len(q) > 1:
        s = np.linalg.norm(np.diff(q, axis=0), axis=1)
        w[:-1] += s / 2
        w[1:] += s / 2
    return w


# ------------------------------------------------------------------------------------------------------------------------------ the arms' graphs
def lane_polylines(lane):
    """arm g1: the lane GT as a raw graph (pts, edges); an edge (i, j) is the lane GT's own `succ` relation of piece i (ids that are not in the frame are dropped)"""
    idx = {p["id"]: i for i, p in enumerate(lane["pieces"])}
    return dict(pts=[np.asarray(p["pts"], float).reshape(-1, 2) for p in lane["pieces"]], edges=[(i, idx[q]) for i, p in enumerate(lane["pieces"]) for q in p["succ"] if q in idx])


def decoded(lane, pred=None):
    """arms l2 / l1 (the grid in force in qtargets): the oracle prediction of the lane GT and its read-out -> (pred, LaneGraph, seconds of the oracle, seconds of the read-out); `pred` replaces the oracle
    (the planted degradations of the self-test)"""
    t0 = time.perf_counter()
    if pred is None:
        pred = qformat.oracle_prediction(lane)
    t1 = time.perf_counter()
    graph, _ = QE.readout(pred)
    return pred, graph, t1 - t0, time.perf_counter() - t1


# ------------------------------------------------------------------------------------------------------------------------------ (b) lane / connector parts
def split_counts(raw, gt, delivered_kind):
    """additive counts of F1@1.0 and lanes found by kind: p (2 kinds, [delivered points within TAU of a GT point, delivered points]) by the kind of the DELIVERED point, r (2, [GT points within TAU of the
    graph, GT points]) and lanes (2, [GT lanes found, GT lanes]) by the `kind` of the old-GT polyline; the sums over the kinds are the scorer's own counts (checked by the driver).
    delivered_kind(P, I) -> (n,) kinds of the delivered dense points P (polyline numbers I)"""
    P, _, IP = GE.dense_points(raw)
    G, _, IG = GE.dense_points(gt)
    gk = np.asarray([k == "connector" for k in gt["kind"]], int)[IG] if len(G) else np.zeros(0, int)
    out = dict(p=np.zeros((2, 2)), r=np.zeros((2, 2)), lanes=np.zeros((2, 2)))
    both = len(P) and len(G)
    dg = cKDTree(P).query(G)[0] if both else np.full(len(G), np.inf)
    dp = cKDTree(G).query(P)[0] if both else np.full(len(P), np.inf)
    kp = delivered_kind(P, IP) if len(P) else np.zeros(0, int)
    for k in (0, 1):
        out["p"][k] = ((dp[kp == k] <= TAU).sum(), (kp == k).sum())
        out["r"][k] = ((dg[gk == k] <= TAU).sum(), (gk == k).sum())
    for i in np.unique(IG):
        m = IG == i
        out["lanes"][gk[m][0]] += ((dg[m] <= TAU).mean() >= FOUND, 1)
    return out


def kind_labeller(arm, lane, raw):
    """kinds of delivered points: g0 = the kinds of the delivered (old GT) polylines themselves, else the kind of the nearest lane GT point (its segment)"""
    if arm == "g0":
        pk = np.asarray([k == "connector" for k in raw["kind"]], int)
        return lambda P, I: pk[I]
    pts = [np.asarray(p["pts"], float).reshape(-1, 2) for p in lane["pieces"]]
    if not sum(len(q) for q in pts):
        return lambda P, I: np.zeros(len(P), int)
    kk, tree = np.concatenate(point_kinds(lane)), cKDTree(np.concatenate(pts))
    return lambda P, I: kk[tree.query(P)[1]]


# ------------------------------------------------------------------------------------------------------------------------------ (c) links, (d) suppression, (e) outputs
def link_counts(nt, pred, gk):
    """per link class (CLASSES): its curve sampled every 0.5 m against its GT polyline (the lane GT points of its path): distance curve -> GT (`d`) and GT -> curve (`rev`), pooled as arrays; n links"""
    out = {c: dict(n=0, d=[], rev=[]) for c in CLASSES}
    curves = pred["links"]["curves"]
    for l, (path, lg) in enumerate(zip(nt["path"], nt["long"])):
        if l >= len(curves):                                                       # a prediction with fewer links than the table (planted degradations)
            break
        gtp = nt["pts"][path]
        c = ("conn_" if gk[path].mean() > 0.5 else "lane_") + ("long" if lg else "short")
        out[c]["n"] += 1
        out[c]["d"].append(poly_dist(curves[l], gtp))
        out[c]["rev"].append(poly_dist(gtp, curves[l]))
    return out


def suppression_counts(nt, gk, w):
    """GT connector length (m, inside the 48 m box of the scorer) = [all, suppressed inside an intersection (no node), of that on the path of a long link, only of short links, of no link]"""
    box = (np.abs(nt["pts"]) < QE.BOX).all(1)
    on = {lg: np.zeros(len(gk), bool) for lg in (True, False)}
    for path, lg in zip(nt["path"], nt["long"]):
        on[bool(lg)][path] = True
    conn, supp = (gk == 1) & box, nt["supp"] & (gk == 1) & box
    sl = supp & on[True]
    ss = supp & ~on[True] & on[False]
    return np.array([w[conn].sum(), w[supp].sum(), w[sl].sum(), w[ss].sum(), w[supp & ~sl & ~ss].sum()])


def output_counts(nt):
    """[nodes, cells, short links, long links, cells with two or more nodes, channels per cell, candidate nodes dropped (a cell claimed by two pieces)] of one frame (the head's per-cell outputs: exists, 8 flags, offset 2, direction 2, section start, long-link
    out / in = the channels of `qnodes.CH`)"""
    t = QT.grid_targets(nt, n=QT.N)
    ch = sum(t[k].shape[0] if t[k].ndim == 3 else 1 for k in ("exists", "flags", "offset", "dirv", "sstart", "uout", "uin"))
    cell = nt["cell"][:, 0] * QT.N + nt["cell"][:, 1] if len(nt["cell"]) else np.zeros(0, int)
    return np.array([len(nt["pos"]), QT.N * QT.N, int((~nt["long"]).sum()), int(nt["long"].sum()), len(cell) - len(np.unique(cell)), ch, nt["stats"]["dropped"]], float)


# ------------------------------------------------------------------------------------------------------------------------------ collisions of a chain code
def collision_counts(lane):
    """the chain code's cell collisions on the lane GT, connector points kept (no intersection suppression), on the grid in force. Two pieces with points in one cell are ONE node when their closest
    two points there are within COLL_DIST m and COLL_ANG degrees in heading (pieces that run together: a shared entry or exit, a join: consecutive pieces share their end point), else a collision.
    -> [occupied cells, cells with two or more pieces, cells with two pieces whose closest points differ in heading by more than COLL_ANG degrees (crossing), collision cells, connectors (`connector`
    segments of the lane GT), connectors with a collision cell along them (a cell of theirs in which their piece collides), the same two numbers for the connector segments inside an intersection,
    connectors with a CROSSING cell along them (the part of the share that does not depend on COLL_DIST)]"""
    cells, keys, aff, affx, pts, tans = {}, [], [], [], [], []
    for i, p in enumerate(lane["pieces"]):
        q = np.asarray(p["pts"], float).reshape(-1, 2)
        pts.append(q)
        tans.append(QT._tangents(q))
        aff.append(set())
        affx.append(set())
        c = QT.cell_of(q)
        key = c[:, 0] * QT.N + c[:, 1]
        keys.append(key)
        for k in np.unique(key):
            cells.setdefault(int(k), []).append((i, np.flatnonzero(key == k)))
    out = np.zeros(9)
    out[0] = len(cells)
    for key, members in cells.items():
        if len(members) < 2:
            continue
        out[1] += 1
        coll = cross = False
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                (ia, ja), (ib, jb) = members[x], members[y]
                d = np.linalg.norm(pts[ia][ja][:, None] - pts[ib][jb][None], axis=2)
                u, v = np.unravel_index(d.argmin(), d.shape)
                ang = np.degrees(np.arccos(np.clip(float(tans[ia][ja[u]] @ tans[ib][jb[v]]), -1.0, 1.0)))
                bad = d[u, v] > COLL_DIST or ang > COLL_ANG
                cross |= ang > COLL_ANG
                coll |= bad
                if bad:
                    aff[ia].add(key)
                    aff[ib].add(key)
                if ang > COLL_ANG:
                    affx[ia].add(key)
                    affx[ib].add(key)
        out[2] += cross
        out[3] += coll
    for i, p in enumerate(lane["pieces"]):
        for s in p["segments"]:
            if s["kind"] != "connector":
                continue
            along = set(keys[i][s["i0"]: s["i1"] + 1].tolist())
            hit = bool(aff[i] & along)
            out[4] += 1
            out[5] += hit
            out[8] += bool(affx[i] & along)
            if s["in_intersection"]:
                out[6] += 1
                out[7] += hit
    return out


def old_as_pieces(gt):
    """the old graph GT (pts, kind) in the shape of a lane GT for collision_counts: polylines every 0.5 m, one segment of the piece's kind (E273's 18.4 % / 3.7 % were counted on these pieces)"""
    pieces = []
    for p, k in zip(gt["pts"], gt["kind"]):
        q = GE.resample(np.asarray(p, float), 0.5)
        pieces.append(dict(pts=q, segments=[dict(kind=k, i0=0, i1=len(q) - 1, in_intersection=False)]))
    return dict(pieces=pieces)


# ------------------------------------------------------------------------------------------------------------------------------ one frame
def frame_record(arm, f, override=None):
    """everything one frame of `arm` contributes: core = the scorer's counts (qgraph_eval.graph_counts against the old graph GT), split (b), and for the decoded arms links (c), supp (d), out (e),
    coll, time (oracle, read-out seconds). `override` replaces the arm's graph (g0 / g1: a raw graph dict) or the oracle prediction (l2 / l1: a prediction dict with its `table`): the planted
    degradations of the self-test. The lane GT is loaded here and let go again."""
    gt = f.gt
    lane = None if arm == "g0" else f.lane()
    rec = {}
    try:
        if arm == "g0":
            graph = raw = gt if override is None else override
        elif arm == "g1":
            graph = raw = lane_polylines(lane) if override is None else override
        else:
            pred, graph, t_or, t_ro = decoded(lane, override)
            raw = QE.to_raw(graph)
            nt = pred["table"]
            gk = np.concatenate([np.zeros(0, int)] + point_kinds(lane))                 # aligned with nt["pts"] (the pieces' points in order)
            w = np.concatenate([np.zeros(0)] + [point_weights(np.asarray(p["pts"], float).reshape(-1, 2)) for p in lane["pieces"]])
            rec.update(links=link_counts(nt, pred, gk), supp=suppression_counts(nt, gk, w), out=output_counts(nt), coll=collision_counts(lane), coll_old=collision_counts(old_as_pieces(gt)), time=np.array([t_or, t_ro]))
        rec["core"] = QE.graph_counts(graph, gt)
        rec["split"] = split_counts(raw, gt, kind_labeller(arm, lane, raw))
        cnt, rs = rec["split"], rec["core"]["rec"]["rs"]                           # the parts add up to the scorer's own counts
        assert np.array_equal(cnt["p"].sum(0), rs["cnt"][1][:2]) and np.array_equal(cnt["r"].sum(0), rs["cnt"][1][2:]), (cnt, rs["cnt"][1])
        assert cnt["lanes"].sum(0).tolist() == [rs["lanes_found"], rs["lanes_total"]], (cnt["lanes"], rs["lanes_found"], rs["lanes_total"])
    finally:
        f._lane = f._ref = None
    return rec


# ------------------------------------------------------------------------------------------------------------------------------ pooling and numbers
def pool(recs):
    """the records of a set of frames -> dict: core (list), split / supp / out / coll / time summed (time: kept as an (n, 2) array), link distances concatenated per class"""
    out = dict(core=[r["core"] for r in recs], frames=len(recs))
    out["split"] = {k: sum(r["split"][k] for r in recs) for k in ("p", "r", "lanes")}
    if "links" in recs[0]:
        out["supp"], out["out"], out["coll"], out["coll_old"] = (sum(r[k] for r in recs) for k in ("supp", "out", "coll", "coll_old"))
        out["time"] = np.stack([r["time"] for r in recs])
        out["links"] = {c: dict(n=sum(r["links"][c]["n"] for r in recs), **{k: np.concatenate([x for r in recs for x in r["links"][c][k]] or [np.zeros(0)]) for k in ("d", "rev")}) for c in CLASSES}
    return out


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _q(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def numbers(pl):
    """flat {name: float} of the pooled records (NaN included: the driver moves them out of the card's metrics)"""
    m = {k: float(v) for k, v in QE.graph_numbers(pl["core"]).items()}
    sp = pl["split"]
    for k, name in enumerate(KINDS):
        p, r = sp["p"][k][0] / max(sp["p"][k][1], 1), sp["r"][k][0] / max(sp["r"][k][1], 1)
        m[f"P@1.0_{name}"], m[f"R@1.0_{name}"], m[f"F1@1.0_{name}"] = float(p), float(r), float(_f1(p, r))
        m[f"lanes_found_{name}"], m[f"n_gt_lanes_{name}"] = float(sp["lanes"][k][0] / max(sp["lanes"][k][1], 1)), float(sp["lanes"][k][1])
    if "links" not in pl:
        return m
    n = pl["frames"]
    for c in CLASSES:
        L = pl["links"][c]
        m[f"link_{c}_n"] = float(L["n"])
        m[f"link_{c}_d_med"], m[f"link_{c}_d_p90"] = _q(L["d"], 50), _q(L["d"], 90)
        m[f"link_{c}_rev_med"], m[f"link_{c}_rev_p90"] = _q(L["rev"], 50), _q(L["rev"], 90)
    s = pl["supp"]
    m["conn_len_m"], m["conn_supp_share"] = float(s[0]), float(s[1] / max(s[0], 1e-9))
    for k, name in ((2, "on_long"), (3, "on_short_only"), (4, "on_no_link")):
        m[f"conn_supp_{name}"] = float(s[k] / max(s[1], 1e-9))
    o = pl["out"]
    m.update(nodes_per_frame=o[0] / n, cells_per_frame=o[1] / n, short_links_per_frame=o[2] / n, long_links_per_frame=o[3] / n, cells_2plus_nodes_per_frame=o[4] / n, channels_per_cell=o[5] / n)
    m["outputs_per_frame"] = m["cells_per_frame"] * m["channels_per_cell"]
    m["node_cell_share"], m["dropped_candidates_per_node"] = float(o[0] / max(o[1], 1)), float(o[6] / max(o[0], 1))
    m["oracle_ms_mean"], m["readout_ms_mean"], m["readout_ms_p90"] = 1e3 * pl["time"][:, 0].mean(), 1e3 * pl["time"][:, 1].mean(), 1e3 * _q(pl["time"][:, 1], 90)
    for key, pre in (("coll", "coll"), ("coll_old", "coll_old")):                  # on the lane GT (the code's source) and on the old graph GT (E273's pieces)
        c = pl[key]
        m.update({f"{pre}_occupied_cells_per_frame": c[0] / n, f"{pre}_raw_share": c[1] / max(c[0], 1), f"{pre}_crossing_share": c[2] / max(c[0], 1), f"{pre}_cell_share": c[3] / max(c[0], 1),
                  f"{pre}_connectors_n": c[4], f"{pre}_connector_share": c[5] / max(c[4], 1), f"{pre}_inter_connectors_n": c[6], f"{pre}_inter_connector_share": c[7] / max(c[6], 1),
                  f"{pre}_connector_cross_share": c[8] / max(c[4], 1)})
    return {k: float(v) for k, v in m.items()}
