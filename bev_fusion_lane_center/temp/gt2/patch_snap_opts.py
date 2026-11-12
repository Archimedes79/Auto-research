p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


sub("def add_connectors(lines, edges, nodes, kinds, links, long_mask, pairs, snap=False):", "def add_connectors(lines, edges, nodes, kinds, links, long_mask, pairs, snap=False, snap_all=False, snap_guard=False, snap_dedup=False):")
sub("ends on average: the curve then ends beside the pieces and the topology scores see a gap); an end with no piece in reach keeps its keypoint; only that one piece is wired at each end\"\"\"",
    "ends on average: the curve then ends beside the pieces and the topology scores see a gap); an end with no piece in reach keeps its keypoint; only that one piece is wired at each end.\n"
    "    Options of the snapped read-out (each only with snap, all default off; E311 refinements, the E311 verdict is the plain `snap`): snap_all = wire EVERY piece in reach of the snapped end (the default wiring loop on the snapped ends: separates the effect of the geometry from the\n"
    "    effect of wiring one piece only), snap_guard = a pair whose snapped ends make a degenerate curve (chord under 1 m or a chord that runs against one of the two headings) keeps its keypoints, snap_dedup = a second curve between the same two snapped pieces is not added\"\"\"")
sub("    wired = 0\n    for n, pr in enumerate(pairs):", "    wired = 0\n    seen = set()\n    for n, pr in enumerate(pairs):")
sub('''            if w_in is not None:
                pa, ta = ends[w_in][0], ends[w_in][1]
            if w_out is not None:
                pb, tb = ends[w_out][2], ends[w_out][3]
        m = len(L2)''', '''            if snap_dedup and w_in is not None and w_out is not None:
                if (w_in, w_out) in seen:
                    continue
                seen.add((w_in, w_out))
            pk = (pa, ta, pb, tb)
            if w_in is not None:
                pa, ta = ends[w_in][0], ends[w_in][1]
            if w_out is not None:
                pb, tb = ends[w_out][2], ends[w_out][3]
            if snap_guard:
                ch = pb - pa
                if np.linalg.norm(ch) < 1.0 or ch @ ta <= 0 or ch @ tb <= 0:
                    pa, ta, pb, tb = pk
                    w_in = w_out = None
        m = len(L2)''')
sub('''        if snap:
            if w_in is not None:
                E2.append((w_in, m))''', '''        if snap and not snap_all:
            if w_in is not None:
                E2.append((w_in, m))''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
