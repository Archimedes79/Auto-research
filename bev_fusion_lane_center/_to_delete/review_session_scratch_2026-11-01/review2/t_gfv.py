import numpy as np, itertools, random
import sparse_metrics as SM
def check(edges, n=None, show=False):
    n = n or (max(max(e) for e in edges) + 1)
    xy = np.random.RandomState(0).rand(n, 2) * 10
    g = SM.graph_from_vertices(xy, edges)
    got = []
    for c in g["nodes"]:
        got += list(zip(c[:-1], c[1:]))
    want = sorted({(a, b) for a, b in edges if a != b})
    ok = sorted(got) == want
    # interior vertices of a chain must have in=out=1
    ins = {v: 0 for v in range(n)}; outs = {v: 0 for v in range(n)}
    for a, b in want: outs[a] += 1; ins[b] += 1
    maximal = all(ins[v] == 1 and outs[v] == 1 for c in g["nodes"] for v in c[1:-1])
    if show or not ok or not maximal:
        print(edges, "->", g["nodes"], "ok" if ok else "EDGE MISMATCH", "maximal" if maximal else "NOT MAXIMAL")
    return ok and maximal
print(check([(0,1),(1,2),(2,3),(3,4),(2,5),(5,6)], show=True))
print(check([(0,1),(1,2),(2,0)], show=True))
print(check([(0,1),(1,0)], show=True))
print(check([(0,1),(1,2),(2,0),(1,3)], show=True))
print(check([(3,0),(0,1),(1,2),(2,0)], show=True))
print(check([(0,2),(1,2),(2,3),(2,4)], show=True))
print(check([(0,0),(0,1),(0,1),(1,2)], show=True))
print(check([(0,1),(2,3)], n=5, show=True))
# random graphs
random.seed(0); bad = 0
for t in range(3000):
    n = random.randint(2, 9); m = random.randint(1, 14)
    E = [(random.randrange(n), random.randrange(n)) for _ in range(m)]
    if not check(E, n): bad += 1
print("random failures", bad)
