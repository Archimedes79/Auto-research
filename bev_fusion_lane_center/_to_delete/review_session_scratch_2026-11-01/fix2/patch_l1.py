import sys
sm, inf = sys.argv[1], sys.argv[2]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
patch(sm, [
('''def graph_from_vertices(xy, edges, order=None):''', '''def graph_from_vertices(xy, edges, order=None, legacy_cut=False):'''),
('''    xy [n,2]; edges list of (a, b); `order` is unused (kept for old callers). Returns dict(pts, edges, nodes)."""''',
 '''    legacy_cut=True reproduces that old cut exactly: the product path (p16_infer.assemble) keeps it until route success and the consumer,
    whose heads were trained on its piece ids, are re-measured on the GPU (D24, open item); the vertices and edges are the same either way.
    xy [n,2]; edges list of (a, b); `order` is unused (kept for old callers). Returns dict(pts, edges, nodes)."""'''),
('''    starts = [v for v in range(n) if outs[v] and (len(ins[v]) != 1 or len(outs[v]) != 1)]
    for s in starts:
        for nxt in outs[s]:
            if (s, nxt) not in used:
                walk(s, nxt)
    for a, b in sorted(seen):  # only cycles of interior vertices are left: one closed piece each
        if (a, b) not in used:
            walk(a, b)''', '''    if legacy_cut:  # the cut before 2026-10-31: a split vertex with one predecessor starts no chain; leftover edges are 2-vertex pieces
        starts = [v for v in range(n) if outs[v] and (len(ins[v]) != 1 or len(outs[ins[v][0]]) != 1)]
    else:
        starts = [v for v in range(n) if outs[v] and (len(ins[v]) != 1 or len(outs[v]) != 1)]
    for s in starts:
        for nxt in outs[s]:
            if (s, nxt) not in used:
                walk(s, nxt)
    for a, b in sorted(seen):  # only cycles of interior vertices are left: one closed piece each (legacy: every leftover edge alone)
        if (a, b) not in used:
            if legacy_cut:
                pieces.append([a, b])
                used.add((a, b))
            else:
                walk(a, b)'''),
])
patch(inf, [
('''    g = C.graph_from_vertices(V, ve)
    if dp_eps > 0:''', '''    g = C.graph_from_vertices(V, ve, legacy_cut=True)  # D24 open item: the consumer / route heads were trained on this piece cut; the
    # fixed cut (legacy_cut=False) moves F1o2o +0.001, dup_frac up to +0.005, vert100 of the *_dp names -0.16 and was not yet measured on route success
    if dp_eps > 0:'''),
])
print("ok")
