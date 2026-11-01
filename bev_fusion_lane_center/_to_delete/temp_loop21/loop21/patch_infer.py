p = "training/pipeline/p16_infer.py"
t = open(p, encoding="utf-8").read()

sig_old = "fork_prune=0.0, fork_prune_m=-1.0, tau_decay=0.0):\n    \"\"\"tree -> DAG by path union"
assert t.count(sig_old) == 1
t = t.replace(sig_old, "fork_prune=0.0, fork_prune_m=-1.0, tau_decay=0.0, dp_eps=0.0):\n    \"\"\"tree -> DAG by path union")

ret_old = "    g = C.graph_from_vertices(V, ve)\n    if aux:  # vertex positions"
assert t.count(ret_old) == 1
t = t.replace(ret_old, "    g = C.graph_from_vertices(V, ve)\n    if dp_eps > 0 and not aux:  # output stage (E257 / D20): drop interior vertices of straight runs, geometry kept within dp_eps; off by default\n        g = simplify_dp(g, dp_eps)\n    if aux:  # vertex positions")

fn = '''def _dp_keep(q, eps):
    """indices kept by Douglas-Peucker (iterative) on the polyline q [n,2] with tolerance eps (m); the ends are always kept"""
    n = len(q)
    if n <= 2:
        return list(range(n))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ab = q[b] - q[a]
        s = np.clip(((q[a + 1:b] - q[a]) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(q[a + 1:b] - (q[a] + s[:, None] * ab), axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return [int(i) for i in np.nonzero(keep)[0]]


def simplify_dp(g, eps):
    """output stage (E257, decision D20): remove the interior vertices of every polyline that the Douglas-Peucker rule (tolerance eps, m) does not need.
    Kept: both ends of every polyline, every node id that occurs in more than one polyline (junctions, links), and the DP vertices between those anchors, so connectivity,
    the polyline pairs of `edges` and the junction metrics are unchanged and no removed vertex is farther than eps from the new polyline. Returns a new graph dict (pts, edges, nodes)."""
    from collections import Counter

    cnt = Counter(i for nd in g["nodes"] for i in nd)
    pts, nodes = [], []
    for q, nd in zip(g["pts"], g["nodes"]):
        q = np.asarray(q, np.float64)
        n = len(q)
        if n < 3:
            pts.append(q)
            nodes.append(list(nd))
            continue
        anchors = sorted({0, n - 1} | {j for j in range(n) if cnt[nd[j]] > 1})
        keep = set(anchors)
        for a, b in zip(anchors, anchors[1:]):
            keep.update(a + k for k in _dp_keep(q[a:b + 1], eps))
        idx = sorted(keep)
        pts.append(q[idx])
        nodes.append([nd[j] for j in idx])
    return dict(pts=pts, edges=list(g["edges"]), nodes=nodes)


'''
key = "def assemble(r, b, tau, cfg, cov_r=1.0"
assert t.count(key) == 1
t = t.replace(key, fn + key)
open(p, "w", encoding="utf-8").write(t)
print("patched")
