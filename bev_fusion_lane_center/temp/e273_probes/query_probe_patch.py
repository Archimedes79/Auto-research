"""patch of temp/query_probe.py: variant 'lanes_win' (window ends of suppressed connectors are kept as nodes), handles fitted to the traversed connector pieces, recall by GT kind"""
p = "temp/query_probe.py"
s = open(p, encoding="utf-8").read()

# 1. nodes_of gets the open flags
s = s.replace("def nodes_of(P, suppress):", "def nodes_of(P, suppress, g=None, keep_open=False):")
s = s.replace("""        if suppress and kd == "connector":
            continue
        cell =""", """        if suppress and kd == "connector":
            if not (keep_open and g is not None and (g["start_open"][pi] or g["end_open"][pi])):
                continue
        cell =""")
s = s.replace("""        for _, k in sorted(best.values(), key=lambda e: e[1]):
            nodes.append(""", """        for _, k in sorted(best.values(), key=lambda e: e[1]):
            if suppress and kd == "connector" and not ((k == 0 and g["start_open"][pi]) or (k == len(p) - 1 and g["end_open"][pi])):
                continue
            nodes.append(""")
s = s.replace("""        for nd in nodes[-len(best):]:
            nd["cell"] = int(key[nd["k"]])""", """        for nd in nodes:
            if nd["piece"] == pi:
                nd["cell"] = int(key[nd["k"]])""")

# 2. links carry the traversed pieces
s = s.replace("""    L = set()
    for pi, lst in by_piece.items():
        for a, b in zip(lst[:-1], lst[1:]):
            L.add((a, b))
        # across the piece end: through pieces without nodes
        stack, seen = [(s, 0) for s in succ.get(pi, [])], set()
        while stack:
            q, dep = stack.pop()
            if (q, dep) in seen or dep > 8:
                continue
            seen.add((q, dep))
            if q in by_piece:
                L.add((lst[-1], by_piece[q][0]))
            else:
                stack += [(s, dep + 1) for s in succ.get(q, [])]
    return sorted(L)""", """    L = {}
    for pi, lst in by_piece.items():
        for a, b in zip(lst[:-1], lst[1:]):
            L[(a, b)] = ()
        # across the piece end: through pieces without nodes (their points are what a long link has to reproduce)
        stack, seen = [(s, 0, ()) for s in succ.get(pi, [])], set()
        while stack:
            q, dep, path = stack.pop()
            if (q, dep) in seen or dep > 8:
                continue
            seen.add((q, dep))
            if q in by_piece:
                key = (lst[-1], by_piece[q][0])
                if key not in L or len(path) < len(L[key]):
                    L[key] = path
            else:
                stack += [(s, dep + 1, path + (q,)) for s in succ.get(q, [])]
    return L""")

# 3. variants and the loop over links
s = s.replace("""def run(variant):
    suppress = variant == "lanes\"""", """def run(variant):
    suppress = variant in ("lanes", "lanes_win")""")
s = s.replace("nodes = nodes_of(P, suppress)", "nodes = nodes_of(P, suppress, g, keep_open=variant == \"lanes_win\")")
s = s.replace("L = links_of(nodes, P, g[\"edges\"])", "LD = links_of(nodes, P, g[\"edges\"])\n        L = sorted(LD)")
s = s.replace("""        dec = []
        for a, b in L:""", """        dec = []
        kind_pts = {kd_: np.concatenate([pc[0] for pc in P if pc is not None and pc[2] == kd_] or [np.zeros((0, 2))]) for kd_ in ("lane", "connector")}
        for a, b in L:""")
s = s.replace("""                best, bc = 1e9, None
                for sa in""", """                tgt = np.concatenate([P[q][0] for q in LD[(a, b)] if P[q] is not None] or [gt_all])
                tt = cKDTree(tgt)
                best, bc = 1e9, None
                for sa in""")
s = s.replace("m = tree_gt.query(cur)[0].mean()", "m = tt.query(cur)[0].mean() + 0.2 * cKDTree(cur).query(tgt)[0].mean()")
s = s.replace("""            rec.append(float((td.query(gt_all)[0] < 0.3).mean()))""", """            rec.append(float((td.query(gt_all)[0] < 0.3).mean()))
            for kd_ in ("lane", "connector"):
                if len(kind_pts[kd_]):
                    kind_rec.setdefault(kd_, []).append(float((td.query(kind_pts[kd_])[0] < 0.3).mean()))""")
s = s.replace("    rec, prec_, fit_a = [], [], []", "    rec, prec_, fit_a, kind_rec = [], [], [], {}")
s = s.replace("""    print("section membership""", """    print("   recall by GT kind: " + ", ".join(f"{k_}: {np.mean(v):.4f}" for k_, v in kind_rec.items()))
    print("section membership""")
s = s.replace("""for v in ("all", "lanes"):
    run(v)""", """for v in sys.argv[1:] or ("all", "lanes"):
    run(v)""")
open(p, "w", encoding="utf-8").write(s)
print("patched")
