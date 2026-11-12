p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


sub("def add_connectors(lines, edges, nodes, kinds, links, long_mask, pairs):", "def add_connectors(lines, edges, nodes, kinds, links, long_mask, pairs, snap=False):")
sub("    pairs = [(entry xy, entry heading, exit xy, exit heading)] -> (raw graph dict for the scorer, curves added, curves wired at both ends)\"\"\"",
    "    pairs = [(entry xy, entry heading, exit xy, exit heading)] -> (raw graph dict for the scorer, curves added, curves wired at both ends).\n"
    "    snap (E311): the connector is drawn from the end of the NEAREST delivered piece wired to the entry (position and direction of that piece end) to the start of the nearest piece wired to the exit, instead of from the keypoints (1.4 m off the GT\n"
    "    ends on average: the curve then ends beside the pieces and the topology scores see a gap); an end with no piece in reach keeps its keypoint; only that one piece is wired at each end\"\"\"")
sub('''        pa, ta, pb, tb = pr[:4]                                                                       # E303: a pair may carry the two handle scales (pr[4], pr[5])
        m = len(L2)''', '''        pa, ta, pb, tb = pr[:4]                                                                       # E303: a pair may carry the two handle scales (pr[4], pr[5])
        w_in = w_out = None
        if snap:
            pa, ta, pb, tb = (np.asarray(x, float) for x in (pa, ta, pb, tb))
            d_in = d_out = np.inf
            for i, (le, te, fs, ts) in enumerate(ends):
                d = np.linalg.norm(le - pa)
                if d <= WIRE_POS and _ang(te, ta) <= WIRE_ANG and d < d_in:
                    w_in, d_in = i, d
                d = np.linalg.norm(fs - pb)
                if d <= WIRE_POS and _ang(ts, tb) <= WIRE_ANG and d < d_out:
                    w_out, d_out = i, d
            if w_in is not None:
                pa, ta = ends[w_in][0], ends[w_in][1]
            if w_out is not None:
                pb, tb = ends[w_out][2], ends[w_out][3]
        m = len(L2)''')
sub('''        N2.append([10000 + n, 20000 + n])
        w_in = w_out = False
        for i, (le, te, fs, ts) in enumerate(ends):
            if np.linalg.norm(le - pa) <= WIRE_POS and _ang(te, ta) <= WIRE_ANG:
                E2.append((i, m))
                w_in = True
            if np.linalg.norm(fs - pb) <= WIRE_POS and _ang(ts, tb) <= WIRE_ANG:
                E2.append((m, i))
                w_out = True
        wired += int(w_in and w_out)''', '''        N2.append([10000 + n, 20000 + n])
        if snap:
            if w_in is not None:
                E2.append((w_in, m))
            if w_out is not None:
                E2.append((m, w_out))
            wired += int(w_in is not None and w_out is not None)
            continue
        w_in = w_out = False
        for i, (le, te, fs, ts) in enumerate(ends):
            if np.linalg.norm(le - pa) <= WIRE_POS and _ang(te, ta) <= WIRE_ANG:
                E2.append((i, m))
                w_in = True
            if np.linalg.norm(fs - pb) <= WIRE_POS and _ang(ts, tb) <= WIRE_ANG:
                E2.append((m, i))
                w_out = True
        wired += int(w_in and w_out)''')
# the curve of the snapped pair is drawn from the snapped ends: the curve construction must come after the snapping (it does: it follows `m = len(L2)`), check the order in the file
sub("        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk[\"kind\"], bk[\"piece_links\"], np.asarray(pred[\"links\"][\"long\"], bool), pairs)",
    "        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk[\"kind\"], bk[\"piece_links\"], np.asarray(pred[\"links\"][\"long\"], bool), pairs, snap=bool(getattr(a, \"snap\", False)))")
sub('    s.add_argument("--chord", action="store_true", help="E303: use the chord tangent for the connectors even when the dump carries learned handle scales (the control)")\n    sub.add_parser("selftest")',
    '    s.add_argument("--chord", action="store_true", help="E303: use the chord tangent for the connectors even when the dump carries learned handle scales (the control)")\n    s.add_argument("--snap", action="store_true", help="E311: draw the connectors from the end of the delivered piece wired to the entry to the start of the piece wired to the exit (the keypoints only decide which entry joins which exit)")\n    sub.add_parser("selftest")')
# selftest: a snapped connector starts at the piece end, not at the keypoint
sub('''    base = dict(topo_F1=0.60, **{"F1@1.0": 0.80}''', '''    off = [(np.array([1.2, -2.5]), np.array([0.0, 1.0]), np.array([8.9, 2.5]), np.array([0.0, 1.0]))]            # keypoints 1.2 / 0.9 m beside the piece ends
    raw3, n3, w3 = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, off, snap=True)
    assert w3 == 1 and np.allclose(raw3["pts"][-1][0], lines[0][-1]) and np.allclose(raw3["pts"][-1][-1], lines[1][0]), (w3, raw3["pts"][-1][[0, -1]])      # the connector runs from the end of piece 0 to the start of piece 1
    assert (0, len(raw3["pts"]) - 1) in raw3["edges"] and (len(raw3["pts"]) - 1, 1) in raw3["edges"] and len([e for e in raw3["edges"] if len(raw3["pts"]) - 1 in e]) == 2, raw3["edges"]
    raw4, n4, w4 = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, [(np.array([30.0, 0.0]), np.array([0.0, 1.0]), np.array([40.0, 10.0]), np.array([0.0, 1.0]))], snap=True)
    assert w4 == 0 and np.allclose(raw4["pts"][-1][0], [30.0, 0.0]), w4                                    # no piece in reach: the keypoints stay the ends
    base = dict(topo_F1=0.60, **{"F1@1.0": 0.80}''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
