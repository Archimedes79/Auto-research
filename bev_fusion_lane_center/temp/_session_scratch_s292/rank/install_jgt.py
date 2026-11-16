"""E342 part A: jgt._groups vectorised (same labels as the python loop: single linkage, a component is numbered by the order of its first member); the loop stays as the oracle of the selftest."""


def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:90]
    return s.replace(a, b)


p = "training/grid/jgt.py"
s = rd(p)
if "def groups_loop(E):" in s:
    print("already installed")
    raise SystemExit(0)
i0 = s.index("def _groups(E):")
i1 = s.index("def _ranks(E, grp):")
old = s[i0:i1]
new = '''def _groups(E):
    """single-linkage groups of ends (E, 4): heading within ARM_ANG, along <= ARM_ALONG, across <= ARM_ACROSS. All pairs at once (E342: the python double loop cost 0.8 ms per set of 13 ends, this 0.1 ms); the labels are the
    ones of the loop (a group is numbered by the order of its first member), the selftest compares both on random sets"""
    n = len(E)
    if n < 2:
        return np.zeros(n, int)
    P, T = E[:, :2], E[:, 2:]
    ti, tj = T[:, None, :], T[None, :, :]
    ang = np.degrees(np.arccos(np.clip(ti[..., 0] * tj[..., 0] + ti[..., 1] * tj[..., 1], -1, 1)))
    s = ti + tj
    sn = s + 1e-9
    t = s / np.sqrt(sn[..., 0] * sn[..., 0] + sn[..., 1] * sn[..., 1])[..., None]                      # the mean heading of the pair
    d = P[None, :, :] - P[:, None, :]                                                                  # d[i, j] = end j - end i
    along = np.abs(d[..., 0] * t[..., 0] + d[..., 1] * t[..., 1])
    across = np.abs(d[..., 0] * (-t[..., 1]) + d[..., 1] * t[..., 0])
    A = (ang <= ARM_ANG) & (along <= ARM_ALONG) & (across <= ARM_ACROSS)
    A |= A.T
    A[np.arange(n), np.arange(n)] = True
    R = A.astype(np.float32)
    for _ in range(int(np.ceil(np.log2(n)))):                                                           # squaring: after k rounds all paths of up to 2^k links are reached, a path has at most n - 1 links
        R = ((R @ R) > 0).astype(np.float32)
    return np.unique(R.argmax(1), return_inverse=True)[1].astype(int)                                   # R.argmax(1): the smallest member of the component


'''
s = s[:i0] + new + s[i1:]
# selftest: the oracle loop and the comparison
s = sub(s, '''    print("jgt selftest ok")''', '''    def groups_loop(E):                                                                                   # E342: the python loop that `_groups` replaced, kept here as the oracle
        n = len(E)
        par = list(range(n))

        def find(i):
            while par[i] != i:
                par[i] = par[par[i]]
                i = par[i]
            return i
        for i in range(n):
            for j in range(i + 1, n):
                ti, tj = E[i, 2:], E[j, 2:]
                if _angle(ti, tj) > ARM_ANG:
                    continue
                t = (ti + tj) / np.linalg.norm(ti + tj + 1e-9)
                d = E[j, :2] - E[i, :2]
                if abs(d @ t) <= ARM_ALONG and abs(d @ np.array([-t[1], t[0]])) <= ARM_ACROSS:
                    par[find(i)] = find(j)
        roots = {}
        return np.array([roots.setdefault(find(i), len(roots)) for i in range(n)], int)
    rng = np.random.default_rng(1)
    n_multi = 0
    for it in range(1500):                                                                                # lane-like clusters (several ends 1 - 4 m apart, headings within a few degrees) so that most sets have groups of 2 - 6 ends
        n = int(rng.integers(0, 33))
        base = np.resize(np.repeat(rng.uniform(-30, 30, (max(n // 3, 1), 2)), 3, 0), (n, 2)) + rng.normal(0, 2.5, (n, 2))
        ang = rng.normal(0, 0.3, n) + rng.integers(0, 4, n) * np.pi / 2
        E = np.concatenate([base, np.stack([np.cos(ang), np.sin(ang)], 1)], 1)
        for dt in (np.float64, np.float32):
            g_old, g_new = groups_loop(E.astype(dt)), _groups(E.astype(dt))
            assert np.array_equal(g_old, g_new), (it, dt, g_old, g_new)
        n_multi += n >= 2 and len(set(g_new.tolist())) < n
    assert n_multi > 500, n_multi                                                                         # the groups are really exercised: more than a third of the sets have a group of 2 or more
    assert _groups(np.zeros((0, 4))).shape == (0,) and _groups(np.array([[0.0, 0.0, 1.0, 0.0]])).tolist() == [0]
    # a chain of 20 ends, each 5 m after the other: single linkage joins them all although the first and the last are 95 m apart
    chain = np.array([[5.0 * i, 0.0, 1.0, 0.0] for i in range(20)])
    assert _groups(chain).tolist() == [0] * 20
    # the edge semantics (the critic's mutants M2, M3, M9, M10 survived the first selftest): the boundary is <=, a zero-heading row is its own group, float32 headings with a self-dot above 1 still link
    cases = [([[0, 0, 1, 0], [6, 0, 1, 0]], [0, 0]), ([[0, 0, 1, 0], [0, 6, 1, 0]], [0, 0]), ([[0, 0, 1, 0], [3, 0, 0, 0], [6, 0, 1, 0]], [0, 1, 0])]
    for rows, want in cases:
        Ec = np.array(rows, float)
        assert groups_loop(Ec).tolist() == want == _groups(Ec).tolist(), (rows, want)
    h = (0.9431353807449341, 0.33240893483161926)
    E32 = np.array([[0, 0, *h], [2, 0, *h], [4, 0, *h]], np.float32)
    assert groups_loop(E32).tolist() == [0, 0, 0] == _groups(E32).tolist()
    # a shuffled path of 32 ends 5.9 m apart is one group (the squaring count: 5 rounds are needed)
    perm = rng.permutation(32)
    Ep = np.array([[5.9 * i, 0.0, 1.0, 0.0] for i in perm])
    assert groups_loop(Ep).tolist() == _groups(Ep).tolist() == [0] * 32
    print("jgt selftest ok")''')
import os
tmp = os.path.join(os.path.dirname(p), ".jgt.py.new")
with open(tmp, "w", encoding="utf-8", newline="\n") as f:
    f.write(s)
os.replace(tmp, p)
print("ok")
