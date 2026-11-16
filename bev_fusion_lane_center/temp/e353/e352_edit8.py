p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
a = s.index("def canon_b(d):")
b = s.index("def compare_frames(ref, new):")
new = '''def match_nodes(a, b, tol=0.05):
    """the nodes of two B dump entries as one set: mutual nearest neighbours in (position, 5 x heading) within `tol` -> (ia, ib) indices into the node lists. The order of the active nodes is the top-K order of the existence, where equal values tie differently on the CPU and on the
    GPU, and a node near e = 0.2 is in one dump and not in the other; the scorer reads nodes as a set (the links name them by index), so the dumps are compared as sets"""
    fa = np.c_[a["pos"], 5.0 * a["dirv"]]
    fb = np.c_[b["pos"], 5.0 * b["dirv"]]
    d = np.sqrt(((fa[:, None, :] - fb[None, :, :]) ** 2).sum(-1))
    ja, jb = d.argmin(1), d.argmin(0)
    ia = np.flatnonzero((jb[ja] == np.arange(len(fa))) & (d[np.arange(len(fa)), ja] < tol))
    return ia, ja[ia]


def compare_dumps(ref, new):
    """B' dumps frame by frame, nodes and links as sets (`match_nodes`): -> dict(frames, nodes_matched / _only_in_ref / _only_in_new, e_max_unmatched (the existence of the nodes in one dump only: they sit at the threshold e >= 0.2), nodes (max abs per key over the matched nodes),
    section_equal (the section starts of matched nodes are matched nodes), links_matched / _only_in_ref / _only_in_new (among the links of matched nodes; p_max_one_sided: their probability sits at the threshold p >= 0.05), links (max abs of p / a / b), long_equal). The assignment rows
    are not compared: they depend on the map layers of the loader, and the stored dump of 2026-11-08 predates the GT merge of 2026-11-11 (v9 -> v13); the scorer does not read them"""
    r = dict(frames=len(ref), nodes_matched=0, nodes_only_in_ref=0, nodes_only_in_new=0, e_max_unmatched=0.0, nodes={}, section_equal=True, links_matched=0, links_only_in_ref=0, links_only_in_new=0, p_max_one_sided=0.0, links={}, long_equal=True)
    for x, y in zip(ref, new):
        assert x["row"] == y["row"]
        ia, ib = match_nodes(x, y)
        r["nodes_matched"] += len(ia)
        um_a, um_b = np.setdiff1d(np.arange(len(x["e"])), ia), np.setdiff1d(np.arange(len(y["e"])), ib)
        r["nodes_only_in_ref"] += len(um_a)
        r["nodes_only_in_new"] += len(um_b)
        r["e_max_unmatched"] = max([r["e_max_unmatched"], *x["e"][um_a].tolist(), *y["e"][um_b].tolist()])
        for k in ("e", "pos", "dirv", "flags", "s"):
            if x.get(k) is not None:
                r["nodes"][k] = max(r["nodes"].get(k, 0.0), float(np.abs(np.asarray(x[k])[ia].astype(float) - np.asarray(y[k])[ib].astype(float)).max()) if len(ia) else 0.0)
        mapa, mapb = -np.ones(len(x["e"]), int), -np.ones(len(y["e"]), int)
        mapa[ia] = ib
        mapb[ib] = ia
        where_a = -np.ones(int(max(x["idx"].max(initial=0), np.asarray(x["section"]).max(initial=0))) + 1, int)         # `dump_full` keeps the section start as an index of the unfiltered node list (`idx` holds the kept ones)
        where_a[x["idx"]] = np.arange(len(x["idx"]))
        where_b = -np.ones(int(max(y["idx"].max(initial=0), np.asarray(y["section"]).max(initial=0))) + 1, int)
        where_b[y["idx"]] = np.arange(len(y["idx"]))
        sa, sb = where_a[np.asarray(x["section"])[ia]], where_b[np.asarray(y["section"])[ib]]
        both = (sa >= 0) & (sb >= 0)
        r["section_equal"] &= bool(np.array_equal(mapa[sa[both]], sb[both]) and np.array_equal(sa >= 0, sb >= 0))
        la = {(int(s), int(d)): j for j, (s, d) in enumerate(zip(mapa[x["link_src"]], mapa[x["link_dst"]])) if s >= 0 and d >= 0}
        lb = {(int(s), int(d)): j for j, (s, d) in enumerate(zip(y["link_src"], y["link_dst"])) if mapb[s] >= 0 and mapb[d] >= 0}
        com = [(la[k], lb[k]) for k in la if k in lb]
        ja, jb = np.array([c[0] for c in com], int), np.array([c[1] for c in com], int)
        r["links_matched"] += len(com)
        one_a, one_b = [j for k, j in la.items() if k not in lb], [j for k, j in lb.items() if k not in la]
        r["links_only_in_ref"] += len(one_a)
        r["links_only_in_new"] += len(one_b)
        r["p_max_one_sided"] = max([r["p_max_one_sided"], *x["link_p"][one_a].tolist(), *y["link_p"][one_b].tolist()])
        for k, key in (("p", "link_p"), ("a", "link_a"), ("b", "link_b")):
            r["links"][k] = max(r["links"].get(k, 0.0), float(np.abs(x[key][ja] - y[key][jb]).max()) if len(com) else 0.0)
        r["long_equal"] &= bool(np.array_equal(x["link_long"][ja], y["link_long"][jb]))
    return r


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
