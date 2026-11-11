"""one-off: which connector points does the lanes_win oracle NOT reproduce, and why? (VAL, 1 frame in 4)"""
import sys, collections
sys.path.insert(0, "temp/e273_probes")
import numpy as np
from scipy.spatial import cKDTree
import query_probe as Q

why = collections.Counter()
pts = collections.Counter()
for k, g in enumerate(Q.G):
    if int(Q.sc[k]) not in Q.VAL:
        continue
    why["_n"] += 1
    if why["_n"] % 4:
        continue
    P = Q.pieces_of(g)
    nodes = Q.nodes_of(P, True, g, keep_open=True)
    if len(nodes) < 3:
        continue
    LD = Q.links_of(nodes, P, g["edges"])
    pos = np.array([n["pos"] for n in nodes]); hd = np.array([n["hdg"] for n in nodes])
    covered = {}      # piece -> is it traversed by a link path
    for (a, b), path in LD.items():
        for q in path:
            covered[q] = True
    for (a, b), path in LD.items():
        if nodes[a]["piece"] in covered or True:
            pass
    nodepieces = {n["piece"] for n in nodes}
    # decode all links like the probe and measure per connector piece
    allgt = np.concatenate([pc[0] for pc in P if pc is not None])
    dec = []
    for (a, b), path in LD.items():
        d = np.linalg.norm(pos[a] - pos[b])
        if d <= Q.SHORT:
            dec.append(Q.hermite(pos[a], hd[a], pos[b], hd[b]))
        else:
            tgt = np.concatenate([P[q][0] for q in path if P[q] is not None] or [allgt])
            tt = cKDTree(tgt)
            best, bc = 1e9, None
            for sa in (0.1, 0.25, 0.4, 0.55, 0.7):
                for sb in (0.1, 0.25, 0.4, 0.55, 0.7):
                    cur = Q.hermite(pos[a], hd[a], pos[b], hd[b], sa, sb)
                    m = tt.query(cur)[0].mean() + 0.2 * cKDTree(cur).query(tgt)[0].mean()
                    if m < best:
                        best, bc = m, cur
            dec.append(bc)
    td = cKDTree(np.concatenate(dec)) if dec else None
    succ, pred = collections.defaultdict(list), collections.defaultdict(list)
    for i, j in g["edges"]:
        succ[i].append(j); pred[j].append(i)
    for pi, pc in enumerate(P):
        if pc is None or pc[2] != "connector":
            continue
        miss = (td.query(pc[0])[0] > 0.3) if td is not None else np.ones(len(pc[0]), bool)
        if not miss.any():
            continue
        n = int(miss.sum())
        so, eo = bool(g["start_open"][pi]), bool(g["end_open"][pi])
        has_node = pi in nodepieces
        trav = pi in covered
        lane_pred = any(P[q] is not None and P[q][2] == "lane" for q in pred[pi])
        lane_succ = any(P[q] is not None and P[q][2] == "lane" for q in succ[pi])
        if trav:
            key = "traversed by a long link, curve does not follow it (fit / turn)"
        elif has_node:
            key = "has window node but no link reproduces it"
        elif (so or eo) and not (lane_pred or lane_succ):
            key = "open at the window, no lane next to it inside"
        elif not lane_pred and not lane_succ:
            key = "chained connectors only (no lane at either end)"
        elif (lane_pred and not lane_succ and not eo) or (lane_succ and not lane_pred and not so):
            key = "lane at one end, the other end dead (no successor / predecessor)"
        else:
            key = "other"
        why[key] += n
        pts["_all"] += n
tot = pts["_all"]
for k_, v in why.most_common():
    if not k_.startswith("_"):
        print(f"{v / tot:.3f}  {k_}")
