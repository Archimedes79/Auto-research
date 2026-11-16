p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
# generic matcher
s = s.replace('''def match_nodes(a, b, tol=0.05):
    """the nodes of two B dump entries as one set: mutual nearest neighbours in (position, 5 x heading) within `tol` -> (ia, ib) indices into the node lists. The order of the active nodes is the top-K order of the existence, where equal values tie differently on the CPU and on the
    GPU, and a node near e = 0.2 is in one dump and not in the other; the scorer reads nodes as a set (the links name them by index), so the dumps are compared as sets"""
    fa = np.c_[a["pos"], 5.0 * a["dirv"]]
    fb = np.c_[b["pos"], 5.0 * b["dirv"]]
    d = np.sqrt''', '''def match_points(pa, da, pb, db, tol=0.05):
    """mutual nearest neighbours in (position, 5 x heading) within `tol` -> (ia, ib) indices. The order of the active nodes / keypoints is a top-K order where equal values tie differently on the CPU and on the GPU, and an item near its threshold (node e = 0.2, keypoint p) is in one dump
    and not in the other; the scorer reads them as sets (the links and the pair logits name them by index), so the dumps are compared as sets"""
    fa = np.c_[pa, 5.0 * da]
    fb = np.c_[pb, 5.0 * db]
    d = np.sqrt''')
s = s.replace('''    return ia, ja[ia]


def compare_dumps''', '''    return ia, ja[ia]


def match_nodes(a, b, tol=0.05):
    return match_points(a["pos"], a["dirv"], b["pos"], b["dirv"], tol)


def compare_dumps''')
# junction compare
a = s.index("def compare_frames(ref, new):")
b = s.index("def check(seed")
new = '''def compare_frames(ref, new):
    """junction dumps frame by frame, the valid keypoints of each kind as sets (`match_points`): -> dict(frames, keypoints_matched / _only_in_ref / _only_in_new, p_max_unmatched (the probability of the keypoints in one dump only), max_abs per key over the matched keypoints and over the logit /
    scale entries of the matched pairs, allowed_equal)"""
    r = dict(frames=len(ref), keypoints_matched=0, keypoints_only_in_ref=0, keypoints_only_in_new=0, p_max_unmatched=0.0, max_abs={}, allowed_equal=True)

    def put(k, v):
        r["max_abs"][k] = max(r["max_abs"].get(k, 0.0), float(v))
    for a, b in zip(ref, new):
        assert a["token"] == b["token"] and a["row"] == b["row"]
        K = a["valid"].shape[1]
        m = []
        for kind in (0, 1):
            ja, jb = np.flatnonzero(a["valid"][kind]), np.flatnonzero(b["valid"][kind])
            ia, ib = match_points(a["kp_pos"][kind * K + ja], a["kp_dir"][kind * K + ja], b["kp_pos"][kind * K + jb], b["kp_dir"][kind * K + jb])
            m.append((ja[ia], jb[ib]))
            r["keypoints_matched"] += len(ia)
            ua, ub = np.setdiff1d(np.arange(len(ja)), ia), np.setdiff1d(np.arange(len(jb)), ib)
            r["keypoints_only_in_ref"] += len(ua)
            r["keypoints_only_in_new"] += len(ub)
            r["p_max_unmatched"] = max([r["p_max_unmatched"], *a["kp_p"][kind * K + ja[ua]].tolist(), *b["kp_p"][kind * K + jb[ub]].tolist()])
        ra, rb = np.r_[m[0][0], K + m[1][0]], np.r_[m[0][1], K + m[1][1]]
        for k in ("kp_pos", "kp_dir", "kp_p", "att_pos", "att_dir", "att_p", "logsig"):
            if isinstance(a.get(k), np.ndarray) and len(ra):
                put(k, np.abs(a[k][ra] - b[k][rb]).max())
        (ea, eb), (xa, xb) = m
        if len(ea) and len(xa):
            for k in ("logit", "scale"):
                if isinstance(a.get(k), np.ndarray):
                    put(k, np.abs(a[k][np.ix_(ea, xa)] - b[k][np.ix_(eb, xb)]).max())
            r["allowed_equal"] &= bool(np.array_equal(a["allowed"][np.ix_(ea, xa)], b["allowed"][np.ix_(eb, xb)]))
    return r


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
