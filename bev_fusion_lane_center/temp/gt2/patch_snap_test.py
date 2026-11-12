p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()
old = '    base = dict(topo_F1=0.60, **{"F1@1.0": 0.80}'
assert old in s
new = '''    # E311 decoys (critic pass): which piece is chosen. Entry keypoint (0.9, -2.3) heading +y; piece ends: 0 heading +x at (2.5, -2.2) (wrong direction), 1 heading +y at (0, -0.5) (right direction, 2.0 m away, FIRST in the list),
    # 2 heading +x at (0.5, -2.3) (nearest, wrong direction), 3 heading +y at (0, -2) (the nearest right one, 0.95 m); exit keypoint (8.5, 2.3): piece starts 4 heading +y at (8, 2.9) (0.78 m, first in the list), 5 heading +y at (8, 2) (0.58 m, the nearest),
    # 6 heading +x at (9.6, 2.2) (wrong direction)
    def ln(a, b, n=10):
        return np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)]
    dl = [ln((-7.5, -2.2), (2.5, -2.2)), ln((0, -9), (0, -0.5)), ln((-9.5, -2.3), (0.5, -2.3)), ln((0, -12), (0, -2)), ln((8, 2.9), (8, 12)), ln((8, 2), (8, 12)), ln((9.6, 2.2), (19.6, 2.2))]
    dk, dlk, dm = ["lane"] * 7, [(i,) for i in range(7)], np.zeros(7, bool)
    pr = [(np.array([0.9, -2.3]), np.array([0.0, 1.0]), np.array([8.5, 2.3]), np.array([0.0, 1.0]))]
    rd, nd, wd = add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, pr, snap=True)
    cv = rd["pts"][-1]
    assert wd == 1 and np.allclose(cv[0], (0, -2)) and np.allclose(cv[-1], (8, 2)), (wd, cv[[0, -1]])                      # from the end of the nearest right-direction piece to the start of the nearest right-direction piece
    assert sorted(e for e in rd["edges"] if 7 in e) == [(3, 7), (7, 5)], rd["edges"]                                      # exactly one piece per end
    ra, na, wa = add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, pr, snap=True, snap_all=True)
    assert np.allclose(ra["pts"][-1][0], (0, -2)) and {(1, 7), (3, 7), (7, 4), (7, 5)} <= set(ra["edges"]) and (0, 7) not in ra["edges"] and (2, 7) not in ra["edges"], ra["edges"]   # snap_all: every right-direction piece in reach of the snapped end
    # snap_guard: two abutting pieces make a chord of 0.5 m: the guard keeps the keypoints, without it the curve has no length
    ab = [ln((0, -10), (0, 0)), ln((0, 0.5), (0, 10))]
    pa_ = [(np.array([0.3, 0.0]), np.array([0.0, 1.0]), np.array([0.3, 0.4]), np.array([0.0, 1.0]))]
    r0, _, _ = add_connectors(ab, [], [[0], [1]], ["lane", "lane"], [(0,), (1,)], np.zeros(2, bool), pa_, snap=True)
    r1, _, _ = add_connectors(ab, [], [[0], [1]], ["lane", "lane"], [(0,), (1,)], np.zeros(2, bool), pa_, snap=True, snap_guard=True)
    assert np.allclose(r0["pts"][-1][0], (0, 0)) and np.allclose(r1["pts"][-1][0], (0.3, 0.0)) and np.isfinite(r0["pts"][-1]).all(), (r0["pts"][-1][0], r1["pts"][-1][0])
    # snap_dedup: two pairs that snap to the same two pieces give one curve
    two = pr + [(np.array([0.7, -2.4]), np.array([0.0, 1.0]), np.array([8.3, 2.4]), np.array([0.0, 1.0]))]
    assert len(add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, two, snap=True)[0]["pts"]) == 9 and len(add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, two, snap=True, snap_dedup=True)[0]["pts"]) == 8
    base = dict(topo_F1=0.60, **{"F1@1.0": 0.80}'''
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
