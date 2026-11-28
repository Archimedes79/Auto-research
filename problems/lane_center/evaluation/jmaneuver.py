"""Maneuver coverage of a delivered lane graph (E310): the number that asks what the planner needs and what the pictures show; `jtoken_eval.py graphs` writes it
per frame.

    mans = maneuvers(lane_gt, qends.cluster); n_ok, n, hit, near, real = frame_counts(raw, mans, ref_tree)
    python -P -m lane_center.evaluation.jmaneuver      # the selftest (the E310 command `score` on the junction pair head's dumps is in the tag v1)

A GT MANEUVER is one entry -> exit connection of a junction: the connector runs of the lane GT grouped by their entry / exit cluster (`qends.cluster`, as
`qends.end_table`), the longest run per (entry, exit). It is REALISED in a delivered graph when at least 90 % of the points of its GT run (the GT points,
spacing 0.5 m, inside the 100 m box; at least 3 points) have a delivered polyline point within 1.5 m whose direction differs by at most 40 degrees: 'is the lane
drawn there, in that direction', no pairing and no tolerance at the ends. The junction PRECISION is the share of the delivered points within 8 m of a GT
connector run that lie within 1.5 m of a GT lane point. The maneuver denominator (826 frames, 10991 maneuvers on VAL) is smaller than the 11083 GT pairs of the
junction pair numbers: the box and the 3-point rule drop 0.8 %.
"""
import numpy as np
from scipy.spatial import cKDTree

BOX, TOL, ANG, NEED, JUNC = 50.0, 1.5, 40.0, 0.9, 8.0
COS = np.cos(np.radians(ANG))


def _dense(p, step):
    """a polyline resampled so that no two neighbouring points are farther apart than `step` (the vertices stay)"""
    p = np.asarray(p, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    n = np.maximum(np.ceil(seg / step - 1e-9).astype(int), 1)
    return np.concatenate([p[:1]] + [p[i] + (p[i + 1] - p[i]) * (np.arange(1, n[i] + 1) / n[i])[:, None] for i in range(len(seg))])


def _tangents(p):
    """unit tangents of a polyline (central differences, the ends one-sided)"""
    p = np.asarray(p, float)
    t = np.zeros_like(p)
    if len(p) < 2:
        t[:] = [0.0, 1.0]
        return t
    t[1:-1] = p[2:] - p[:-2]
    t[0], t[-1] = p[1] - p[0], p[-1] - p[-2]
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(n, 1e-9)


def maneuvers(g, cluster):
    """[(entry cluster, exit cluster, run points every 1 m inside the box (n, 2), tangents (n, 2))] of the connector runs of a lane GT graph, the longest run
    per (entry, exit); `cluster` = `qends.cluster`"""
    runs = []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        tan = _tangents(q)
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
                    runs.append((q[k], tan[k], q[j], tan[j], q[k: j + 1]))
                k = j + 1
            else:
                k += 1
    if not runs:
        return []
    _, eid = cluster([(r[0], r[1]) for r in runs])
    _, xid = cluster([(r[2], r[3]) for r in runs])
    best = {}
    for r, a, b in zip(runs, eid, xid):
        if (a, b) not in best or len(r[4]) > len(best[(a, b)][4]):
            best[(a, b)] = r
    out = []
    for (a, b), r in best.items():
        P = _dense(r[4], 1.0)
        m = (np.abs(P) < BOX).all(1)
        if m.sum() >= 3:
            out.append((a, b, P[m], _tangents(P)[m]))
    return out


def delivered(raw):
    """the densified points (0.5 m) of a delivered graph (dict with `pts`) and their tangents"""
    Ps, Ts = [], []
    for p in raw["pts"]:
        if len(p) < 2:
            continue
        d = _dense(p, 0.5)
        Ps.append(d)
        Ts.append(_tangents(d))
    if not Ps:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(Ps), np.concatenate(Ts)


def realised_share(tree, tang, run_pts, run_tan):
    """share of the GT run points that have a delivered point within TOL (ALL of them, not the nearest few: a crowded graph must not hide the right line) whose
    direction differs by at most ANG"""
    if tree is None:
        return 0.0
    ok = np.zeros(len(run_pts), bool)
    for i, nb in enumerate(tree.query_ball_point(run_pts, TOL)):
        if nb:
            ok[i] = bool(((tang[nb] * run_tan[i]).sum(1) >= COS).any())
    return float(ok.mean())


def frame_counts(raw, mans, ref_tree):
    """(maneuvers realised, maneuvers, delivered points near the junction that are on a GT lane, delivered points near the junction, realised flags) of one
    delivered graph"""
    P, T = delivered(raw)
    tree = cKDTree(P) if len(P) else None
    real = [realised_share(tree, T, m[2], m[3]) >= NEED for m in mans]
    near = P[(np.abs(P) < BOX).all(1)] if len(P) else P
    if len(near):
        near = near[cKDTree(np.concatenate([m[2] for m in mans])).query(near)[0] <= JUNC]
    hit = int((ref_tree.query(near)[0] <= TOL).sum()) if len(near) else 0
    return int(sum(real)), len(mans), hit, len(near), real


def selftest():
    # a GT run: a quarter circle of radius 15 m (a left turn) from (0, -15) heading +y to (-15, 0) heading -x
    th = np.linspace(0, np.pi / 2, 60)
    arc = np.stack([-(15 - 15 * np.cos(th)), -15 + 15 * np.sin(th)], 1)
    g = dict(pieces=[dict(pts=arc, segments=[dict(kind="connector", i0=0, i1=len(arc) - 1)])])
    from lane_center.net.prep import qends as QE
    mans = maneuvers(g, QE.cluster)
    assert len(mans) == 1 and len(mans[0][2]) >= 20, mans
    ref_tree = cKDTree(arc)
    ok = dict(pts=[arc])
    # the centre of the circle: radial offsets keep the direction and move the lane sideways everywhere
    c = np.array([-15.0, -15.0])
    shifted = dict(pts=[c + (arc - c) * (19.0 / 15.0)])                               # 4 m aside: not realised
    reverse = dict(pts=[arc[::-1]])                                                    # the same lane the other way round: the direction rule rejects it
    half = dict(pts=[arc[: len(arc) // 2]])                                            # half of it: 50 % < 90 %
    near = dict(pts=[c + (arc - c) * (15.6 / 15.0)])                                   # 0.6 m aside: realised
    for raw, want in ((ok, 1), (shifted, 0), (reverse, 0), (half, 0), (near, 1), (dict(pts=[]), 0)):
        n_ok, n, hit, n_near, _ = frame_counts(raw, mans, ref_tree)
        assert n == 1 and n_ok == want, (n_ok, want)
    # precision: the delivered points near the junction that lie on a GT lane (the shifted copy lies 4 m away: none)
    assert frame_counts(ok, mans, ref_tree)[2] == frame_counts(ok, mans, ref_tree)[3] > 0
    assert frame_counts(shifted, mans, ref_tree)[2] == 0
    # a frame without a connector run has no maneuver
    assert maneuvers(dict(pieces=[dict(pts=arc, segments=[dict(kind="lane", i0=0, i1=len(arc) - 1)])]), QE.cluster) == []
    # the thresholds are pinned: a straight GT run along x (101 points, 0.5 m apart), a delivered line parallel to it
    x = np.arange(-25.0, 25.01, 0.5)
    run = np.stack([x, np.zeros_like(x)], 1)
    gs = dict(pieces=[dict(pts=run, segments=[dict(kind="connector", i0=0, i1=len(run) - 1)])])
    ms = maneuvers(gs, QE.cluster)
    assert len(ms) == 1 and len(ms[0][2]) == 101
    tr = cKDTree(run)

    def line(y, x0=-25.0, x1=25.0, rev=False):
        q = np.stack([np.arange(x0, x1 + 1e-9, 0.5), np.full(int((x1 - x0) / 0.5) + 1, float(y))], 1)
        return q[::-1] if rev else q

    for y, want in ((1.4, 1), (1.6, 0)):                                                  # TOL = 1.5 m
        assert frame_counts(dict(pts=[line(y)]), ms, tr)[0] == want, y
    for x1, want in ((19.0, 1), (17.0, 0)):                                               # NEED = 0.9: 92 % of the run covered, 87 % covered
        assert frame_counts(dict(pts=[line(0.0, -25.0, x1)]), ms, tr)[0] == want, x1
    t = _tangents(run)
    rot = lambda deg: np.stack([np.cos(np.radians(deg)) * t[:, 0] - np.sin(np.radians(deg)) * t[:, 1],
                                np.sin(np.radians(deg)) * t[:, 0] + np.cos(np.radians(deg)) * t[:, 1]], 1)
    for deg, want in ((35, 1.0), (45, 0.0)):                                              # ANG = 40 degrees
        assert realised_share(tr, rot(deg), run, t) == want, deg
    # a crowded graph does not hide the right line: the correct line 1.2 m aside plus eight reversed duplicates and a crossing line
    # the right line comes LAST in the point order
    crowd = [line(0.05 * k, rev=True) for k in range(1, 9)] + [np.stack([np.zeros(41), np.linspace(-10, 10, 41)], 1)] + [line(1.2)]
    assert frame_counts(dict(pts=crowd), ms, tr)[0] == 1
    # the junction precision looks only at delivered points within 8 m of a GT run: a far line is not counted
    assert frame_counts(dict(pts=[line(30.0)]), ms, tr)[3] == 0 and frame_counts(dict(pts=[line(5.0)]), ms, tr)[3] > 0
    # two runs of one (entry, exit) pair: the longer one is the maneuver, a second pair is a second maneuver
    short, long_ = run[40:61], run
    g2 = dict(pieces=[dict(pts=short, segments=[dict(kind="connector", i0=0, i1=len(short) - 1)]),
                      dict(pts=long_, segments=[dict(kind="connector", i0=0, i1=len(long_) - 1)])])
    m2 = maneuvers(g2, QE.cluster)
    assert len(m2) <= 2 and max(len(m[2]) for m in m2) == 101, [len(m[2]) for m in m2]
    print("jmaneuver selftest ok")


if __name__ == "__main__":
    selftest()
