p = "evaluation/jmaneuver.py"
s = open(p, encoding="utf-8").read()
old = '    print("jmaneuver selftest ok")'
assert old in s
new = '''    # the thresholds are pinned: a straight GT run along x (101 points, 0.5 m apart), a delivered line parallel to it
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
    rot = lambda deg: np.stack([np.cos(np.radians(deg)) * t[:, 0] - np.sin(np.radians(deg)) * t[:, 1], np.sin(np.radians(deg)) * t[:, 0] + np.cos(np.radians(deg)) * t[:, 1]], 1)
    for deg, want in ((35, 1.0), (45, 0.0)):                                              # ANG = 40 degrees
        assert realised_share(tr, rot(deg), run, t) == want, deg
    # a crowded graph does not hide the right line: the correct line 1.2 m aside plus eight reversed duplicates and a crossing line
    crowd = [line(1.2)] + [line(0.05 * k, rev=True) for k in range(1, 9)] + [np.stack([np.zeros(41), np.linspace(-10, 10, 41)], 1)]
    assert frame_counts(dict(pts=crowd), ms, tr)[0] == 1
    # the junction precision looks only at delivered points within 8 m of a GT run: a far line is not counted
    assert frame_counts(dict(pts=[line(30.0)]), ms, tr)[3] == 0 and frame_counts(dict(pts=[line(5.0)]), ms, tr)[3] > 0
    # two runs of one (entry, exit) pair: the longer one is the maneuver, a second pair is a second maneuver
    short, long_ = run[40:61], run
    g2 = dict(pieces=[dict(pts=short, segments=[dict(kind="connector", i0=0, i1=len(short) - 1)]), dict(pts=long_, segments=[dict(kind="connector", i0=0, i1=len(long_) - 1)])])
    m2 = maneuvers(g2, QE.cluster)
    assert len(m2) <= 2 and max(len(m[2]) for m in m2) == 101, [len(m[2]) for m in m2]
    print("jmaneuver selftest ok")'''
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
