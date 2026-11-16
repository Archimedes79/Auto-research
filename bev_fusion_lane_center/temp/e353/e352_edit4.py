p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
a = s.index("SCORED = (")
b = s.index("def check(seed")
new = '''def canon_b(d):
    """a B dump entry with the nodes in a canonical order (by rounded position and heading): the order of the active nodes is the top-K order of the existence, where equal values tie differently on the CPU and on the GPU; the scorer reads the nodes as a set (the links name them by
    index), so the comparison permutes the nodes, renumbers the links and the section starts, and sorts the links"""
    pos, dv = np.round(d["pos"], 2), np.round(d["dirv"], 2)
    order = np.lexsort((dv[:, 1], dv[:, 0], pos[:, 1], pos[:, 0]))
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    src, dst = rank[d["link_src"]], rank[d["link_dst"]]
    lo = np.argsort(src * len(order) + dst, kind="stable")
    return dict(nodes={k: np.asarray(d[k])[order] for k in ("e", "pos", "dirv", "flags", "s") if d.get(k) is not None}, section=rank[np.asarray(d["section"])[order]],
                links=dict(key=(src * len(order) + dst)[lo], p=d["link_p"][lo], a=d["link_a"][lo], b=d["link_b"][lo], long=d["link_long"][lo]))


def compare_dumps(ref, new):
    """B' dumps frame by frame, nodes and links as sets (`canon_b`): -> dict(frames, node_count_differs (frames whose node count at e >= 0.2 differs), nodes (max abs per key over the frames of equal node count), section_equal, links_only_in_ref / _in_new / _common (totals; the links
    are those at p >= 0.05: the ones near the threshold flip between devices), links (max abs of p / a / b on the common links), long_equal). The assignment rows are not compared: they depend on the map layers of the loader, and the stored dump of 2026-11-08 predates the GT merge of
    2026-11-11 (v9 -> v13); the scorer does not read them"""
    nodes, links, bad, only_r, only_n, both, sec, lng = {}, {}, [], 0, 0, 0, True, True
    for i, (a, b) in enumerate(zip(ref, new)):
        assert a["row"] == b["row"]
        ca, cb = canon_b(a), canon_b(b)
        ka, kb = np.intersect1d(ca["links"]["key"], cb["links"]["key"], return_indices=True), None
        _, ia, ib = ka
        only_r += len(ca["links"]["key"]) - len(ia)
        only_n += len(cb["links"]["key"]) - len(ib)
        both += len(ia)
        if len(a["e"]) != len(b["e"]):
            bad.append(i)
            continue
        for k, v in ca["nodes"].items():
            nodes[k] = max(nodes.get(k, 0.0), float(np.abs(v.astype(float) - cb["nodes"][k].astype(float)).max()))
        sec &= bool(np.array_equal(ca["section"], cb["section"]))
        for k in ("p", "a", "b"):
            links[k] = max(links.get(k, 0.0), float(np.abs(ca["links"][k][ia] - cb["links"][k][ib]).max()) if len(ia) else 0.0)
        lng &= bool(np.array_equal(ca["links"]["long"][ia], cb["links"]["long"][ib]))
    return dict(frames=len(ref), node_count_differs=bad, nodes=nodes, section_equal=sec, links_only_in_ref=only_r, links_only_in_new=only_n, links_common=both, links=links, long_equal=lng)


def compare_frames(ref, new):
    """junction dumps frame by frame with the valid keypoints of each kind in a canonical order (rounded position): -> dict(keypoints_differ (frames whose valid keypoints differ in number), max_abs per key over the valid keypoints and over the logit / scale entries of the valid pairs,
    allowed_equal)"""
    mx, bad, allowed = {}, [], True

    def order(fr, kind):
        K = fr["valid"].shape[1]
        j = np.flatnonzero(fr["valid"][kind])
        pos = np.round(fr["kp_pos"][kind * K + j], 2)
        return j[np.lexsort((pos[:, 1], pos[:, 0]))]
    for i, (a, b) in enumerate(zip(ref, new)):
        assert a["token"] == b["token"] and a["row"] == b["row"]
        K = a["valid"].shape[1]
        pa, pb = [order(a, k) for k in (0, 1)], [order(b, k) for k in (0, 1)]
        if any(len(x) != len(y) for x, y in zip(pa, pb)):
            bad.append(i)
            continue
        ra, rb = np.r_[pa[0], K + pa[1]], np.r_[pb[0], K + pb[1]]
        for k in ("kp_pos", "kp_dir", "kp_p", "att_pos", "att_dir", "att_p", "logsig"):
            if isinstance(a.get(k), np.ndarray):
                mx[k] = max(mx.get(k, 0.0), float(np.abs(a[k][ra] - b[k][rb]).max()) if len(ra) else 0.0)
        for k in ("logit", "scale"):
            if isinstance(a.get(k), np.ndarray) and len(pa[0]) and len(pa[1]):
                mx[k] = max(mx.get(k, 0.0), float(np.abs(a[k][np.ix_(pa[0], pa[1])] - b[k][np.ix_(pb[0], pb[1])]).max()))
        allowed &= bool(np.array_equal(a["allowed"][np.ix_(pa[0], pa[1])], b["allowed"][np.ix_(pb[0], pb[1])]))
    return dict(frames=len(ref), keypoints_differ=bad, max_abs=mx, allowed_equal=allowed)


'''
s = s[:a] + new + s[b:]
s = s.replace('FRAME_KEYS = ("kp_pos", "kp_dir", "kp_p", "valid", "logit", "allowed", "scale", "logsig", "att_pos", "att_dir", "att_p")\n', '')
open(p, "w", encoding="utf-8").write(s)
