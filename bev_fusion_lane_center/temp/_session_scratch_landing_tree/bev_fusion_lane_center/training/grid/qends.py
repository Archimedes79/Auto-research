"""Junction ends and pairs of the lane GT (E284; copied into this project from the E283 scratch `temp/e284/pair_geometry.py`, which stays the record of the E283 numbers).

    et = end_table(g)       # g = the lane GT graph of a frame (GtGenerator lane_gt.load)
    et["ent"], et["ext"]    (E, 4) / (X, 4): entry / exit points (x, y, tx, ty); the cluster centres of the starts / ends of the connector runs
    et["cand"]              (C, 2) int: the geometrically possible pairs (entry a, exit b); et["y"] (C,) 1 for a true pair (a connector run from a to b)
    et["X"]                 (C, 20) the geometry features of a candidate pair (`pair_features`)
    curve(pa, ta, pb, tb)   the cubic Hermite curve of a pair (tangent length = k x chord), the shape of the E283 oracle

An ENTRY is the first point of a connector run of a piece (the lane GT's own entry into an intersection: the stop / event point, 1 - 3 m before the polygon edge, E283 section 5); an EXIT is its last point.
Runs of fewer than 3 points are not ends. Starts that lie within 1.5 m and 40 degrees of each other are one entry (a cluster; likewise for the exits). A pair is possible when the turn between the two
headings is at most 130 degrees, the chord is at least 2 m and the exit lies ahead of the entry and the entry behind the exit (chord . heading > 0 at both). 98 % of the true pairs pass this rule, 4.3 candidates per true
pair (E283 section 1). Nothing here uses the image, the map or the network.
Run the self-test: python qends.py
"""
from collections import Counter

import numpy as np

import qtargets as QT

MAX_TURN = 130.0
MIN_CHORD = 2.0
NF = 20                                                                    # features of a candidate pair


def cluster(items, tol_pos=1.5, tol_ang=40.0):
    """greedy clustering of (position, tangent) items -> (list of representatives, cluster id of every item)"""
    reps, ids = [], []
    for pos, tan in items:
        for j, (rp, rt) in enumerate(reps):
            if np.linalg.norm(pos - rp) <= tol_pos and np.degrees(np.arccos(np.clip(tan @ rt, -1, 1))) <= tol_ang:
                ids.append(j)
                break
        else:
            reps.append((pos, tan))
            ids.append(len(reps) - 1)
    return reps, ids


def allowed(a, ta, b, tb):
    """can an entry at a (heading ta) be joined to an exit at b (heading tb)?"""
    ch = b - a
    return np.degrees(np.arccos(np.clip(ta @ tb, -1, 1))) <= MAX_TURN and np.linalg.norm(ch) >= MIN_CHORD and (ch @ ta) > 0 and (ch @ tb) > 0


def sang(u, v):
    """signed angle (degrees, + = left) from u to v"""
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)))


def side(P, Tn):
    """(left, right) counts of the parallel ends (heading within 30 degrees, within 12 m, more than 1 m to the side) of every end"""
    out = []
    for i in range(len(P)):
        n = np.array([-Tn[i][1], Tn[i][0]])
        d = P - P[i]
        ang = np.degrees(np.arccos(np.clip(Tn @ Tn[i], -1, 1)))
        m = (np.linalg.norm(d, axis=1) <= 12.0) & (ang <= 30.0)
        m[i] = False
        l = d[m] @ n
        out.append((int((l > 1.0).sum()), int((l < -1.0).sum())))
    return out


def pair_features(ent, ext, cand):
    """(C, 20) features of the candidate pairs from the ends alone: the turn angle (signed, and its size), the chord (length, offsets in the entry's and in the exit's frame, the angles between the chord and
    both headings), the position of the entry / exit among the parallel ones (left / right counts), the number of candidates of the entry and of the exit, the rank of the pair by turn angle and chord length
    among the candidates of its entry (and by turn angle among those of its exit), the numbers of entries and exits. ent / ext: sequences of (point (2,), heading (2,))."""
    if not len(cand):
        return np.zeros((0, NF))
    E, TE = np.array([e[0] for e in ent]), np.array([e[1] for e in ent])
    X_, TX = np.array([x[0] for x in ext]), np.array([x[1] for x in ext])
    sa, sb = side(E, TE), side(X_, TX)
    na, nb = Counter(a for a, b in cand), Counter(b for a, b in cand)
    turn = np.array([sang(ent[a][1], ext[b][1]) for a, b in cand])
    L = np.array([np.linalg.norm(ext[b][0] - ent[a][0]) for a, b in cand])
    rk_turn_a, rk_len_a, rk_turn_b = np.zeros(len(cand)), np.zeros(len(cand)), np.zeros(len(cand))
    for a in set(x[0] for x in cand):
        idx = [i for i, (aa, bb) in enumerate(cand) if aa == a]
        for arr, vals in ((rk_turn_a, np.abs(turn[idx])), (rk_len_a, L[idx])):
            r = np.argsort(np.argsort(vals))
            arr[idx] = r / max(len(idx) - 1, 1)
    for b in set(x[1] for x in cand):
        idx = [i for i, (aa, bb) in enumerate(cand) if bb == b]
        r = np.argsort(np.argsort(np.abs(turn[idx])))
        rk_turn_b[idx] = r / max(len(idx) - 1, 1)
    F = []
    for i, (a, b) in enumerate(cand):
        pa, ta, pb, tb = ent[a][0], ent[a][1], ext[b][0], ext[b][1]
        ch = pb - pa
        na_, nb_ = np.array([-ta[1], ta[0]]), np.array([-tb[1], tb[0]])
        F.append([turn[i], abs(turn[i]), L[i], ch @ ta, ch @ na_, ch @ tb, ch @ nb_, sang(ta, ch), sang(tb, ch), sa[a][0], sa[a][1], sb[b][0], sb[b][1], na[a], nb[b],
                  rk_turn_a[i], rk_len_a[i], rk_turn_b[i], len(ent), len(ext)])
    return np.array(F, float)


def end_table(g):
    """the ends and candidate pairs of the lane GT graph of one frame; None when the frame has no connector run of 3 or more points (no junction) or no possible pair"""
    runs = []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        tan = QT._tangents(q)
        conn = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
        k = 0
        while k < len(q):
            if conn[k]:
                j = k
                while j + 1 < len(q) and conn[j + 1]:
                    j += 1
                if j - k + 1 >= 3:
                    runs.append((q[k], tan[k], q[j], tan[j]))
                k = j + 1
            else:
                k += 1
    if not runs:
        return None                                                        # (the arrays of a frame without a junction: `empty_table()`)
    ent, eid = cluster([(r[0], r[1]) for r in runs])
    ext, xid = cluster([(r[2], r[3]) for r in runs])
    true_pairs = set(zip(eid, xid))
    cand = [(a, b) for a in range(len(ent)) for b in range(len(ext)) if allowed(ent[a][0], ent[a][1], ext[b][0], ext[b][1])]
    if not cand:
        return None
    y = np.array([(a, b) in true_pairs for a, b in cand], int)
    return dict(ent=np.array([np.r_[e[0], e[1]] for e in ent]), ext=np.array([np.r_[x[0], x[1]] for x in ext]), cand=np.array(cand, int), y=y, X=pair_features(ent, ext, cand),
                pairs=np.array(sorted(true_pairs), int), n_true=len(true_pairs))


def empty_table():
    """what `end_table` stands for in a batch when a frame has no junction: no ends, no pairs"""
    return dict(ent=np.zeros((0, 4)), ext=np.zeros((0, 4)), cand=np.zeros((0, 2), int), y=np.zeros(0, int), X=np.zeros((0, NF)), pairs=np.zeros((0, 2), int), n_true=0)


def curve(a, ta, b, tb, k=1.0, step=0.25):
    """cubic Hermite curve from a (heading ta) to b (heading tb), handles k x chord (the shape of the E283 oracle: k = 1.0), points about `step` metres apart"""
    L = float(np.linalg.norm(b - a))
    n = max(8, int(1.6 * L / step))
    s = np.linspace(0, 1, n)[:, None]
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * a + h10 * (k * L * ta) + h01 * b + h11 * (k * L * tb)


def curve_s(a, ta, b, tb, sa=1.0 / 3.0, sb=1.0 / 3.0, step=0.25):
    """the cubic Bezier / Hermite curve from a (heading ta) to b (heading tb) with the handle scales sa, sb in units of the chord (the control points a + sa x chord x ta and b - sb x chord x tb; sa = sb = 1/3 is `curve` with k = 1;
    the scales of `qtargets.fit_long_scales` / E303 are these), points about `step` metres apart"""
    L = float(np.linalg.norm(b - a))
    n = max(8, int(1.6 * L / step))
    s = np.linspace(0, 1, n)[:, None]
    q1, q2 = a + sa * L * ta, b - sb * L * tb
    return (1 - s) ** 3 * a + 3 * (1 - s) ** 2 * s * q1 + 3 * (1 - s) * s ** 2 * q2 + s ** 3 * b


def _selftest():
    # a T-junction: the lane comes from the south (0, -20) -> (0, -6), a connector turns left to (-6, 0) -> and the lane continues west; a second one goes straight on north
    def piece(pts, conn_from, conn_to):
        pts = np.asarray(pts, float)
        return dict(pts=pts, segments=[dict(kind="lane", i0=0, i1=conn_from - 1), dict(kind="connector", i0=conn_from, i1=conn_to), dict(kind="lane", i0=conn_to + 1, i1=len(pts) - 1)])
    south = [(0.0, y) for y in np.arange(-20, -6, 1.0)]
    left = [(-t, -6 + t) for t in np.linspace(0, 6, 9)[1:-1]]
    west = [(-6.0 - x, 0.0) for x in np.arange(1, 10, 1.0)]
    north = [(0.0, y) for y in np.arange(-5, 20, 1.0)]
    g = dict(pieces=[piece(south + [(0, -6)] + left + west, len(south), len(south) + len(left)),
                     piece(south + [(0, -6)] + north, len(south), len(south) + 6)])
    et = end_table(g)
    assert et is not None and len(et["ent"]) == 1 and len(et["ext"]) == 2, (len(et["ent"]), len(et["ext"]))
    assert len(et["cand"]) == 2 and et["y"].sum() == 2 and et["X"].shape == (2, NF), (et["cand"], et["y"], et["X"].shape)
    assert end_table(dict(pieces=[piece(south, 5, 6)])) is None
    c = curve(np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([4.0, 4.0]), np.array([1.0, 0.0]))
    c2 = curve_s(np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([4.0, 4.0]), np.array([1.0, 0.0]))
    assert np.allclose(c, c2), 'curve_s with the scales 1/3 is curve with k = 1'
    assert np.allclose(c[0], [0, 0]) and np.allclose(c[-1], [4, 4]) and len(c) >= 8
    print("qends selftest ok")


if __name__ == "__main__":
    _selftest()
