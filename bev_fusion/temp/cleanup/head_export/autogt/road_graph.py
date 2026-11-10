"""Neighbours, lane counts and roads of the lane graph (`autogt.lane_graph`): which lanes are side by side, and the road graph above the lanes.

The nuScenes map has NO road graph: a lane's edge-line tokens match no road block (0 of 3,662 lanes in the four maps), two of the four maps (singapore-
queenstown, singapore-hollandvillage) have road segments only for intersections and road blocks that do not point at a segment, and nothing says how
many lanes a road has. So everything here is derived from the lane centre-lines.

NEIGHBOURS. At every metre of a lane the nearest lane on its left (and right) is searched: its centre-line point lies 1.5 - 6 m to the side, abreast
(within 1 m along the lane) and parallel (within 25 degrees: co-directional; antiparallel: opposite). A lane is a neighbour of another when this holds
along at least `min_overlap` metres. `lane_count` at a point = 1 + the number of lanes found by following left and right neighbours.
ROADS. A road is a set of lanes joined by the neighbour relation and by 1-in-1-out lane-to-lane links (it ends where lanes split, merge or reach an
intersection). An INTERSECTION is a connected area of road segments flagged is_intersection (the connectors inside it belong to it). A connector outside
every intersection belongs to the road of its predecessor lane. The road graph has roads and intersections as nodes and the lane successors as edges.
LINKS. Every joined piece gets `road` (the road or intersection under its middle), `roads_along` (all of them in travel order), `left`, `right`
and `opposite` (piece ids), `lane_count` and `lane_index` (lanes to its left); the crop lists the roads and intersections of the window.
Open with Mirko: the neighbour distances, the minimum overlap and whether an opposite road counts as part of the same road.
"""
import numpy as np

from .lane_graph import _length

DEFAULTS = dict(lat_min=1.5, lat_max=6.0, lon_max=1.0, par_deg=25.0, min_overlap=3, hops=8)


def _samples(polys, kind, step=1.0, base=0.5):
    """lane centre-lines every `step` metres: (xy (n, 2), heading (n, 2), lane index (n,), tokens)"""
    xs, hs, ids, toks = [], [], [], []
    for t in sorted(polys):
        if kind[t] != "lane":
            continue
        P = polys[t]
        k = max(1, int(round(step / base)))
        Q = P[::k] if (len(P) - 1) % k == 0 else np.vstack([P[::k], P[-1:]])
        if len(Q) < 2:
            continue
        d = np.diff(Q, axis=0)
        h = np.vstack([d, d[-1:]])
        h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
        xs.append(Q)
        hs.append(h)
        ids.append(np.full(len(Q), len(toks)))
        toks.append(t)
    return np.concatenate(xs), np.concatenate(hs), np.concatenate(ids), toks


def neighbours(polys, kind, lat_min=1.5, lat_max=6.0, lon_max=1.0, par_deg=25.0, min_overlap=3, hops=8):
    """-> dict(left, right, opposite: token -> [tokens]; lane_count, lane_min, lane_max, lane_index: token -> int)"""
    X, H, lane, toks = _samples(polys, kind)
    n = len(X)
    cos_par = np.cos(np.radians(par_deg))
    cell = np.floor(X / (lat_max + lon_max)).astype(np.int64)             # a cell is as wide as the search reaches: the 3 x 3 cells around it hold every candidate
    key = cell[:, 0] * 1000003 + cell[:, 1]
    order = np.argsort(key)
    skey = key[order]
    uniq, start = np.unique(skey, return_index=True)
    stop = np.r_[start[1:], len(skey)]
    where = {int(k): (int(a), int(b)) for k, a, b in zip(uniq, start, stop)}
    left_of, right_of, opp_l, opp_r = (np.full(n, -1, np.int64) for _ in range(4))
    for k, (a, b) in where.items():
        mine = order[a:b]
        cx, cy = divmod(k, 1000003)
        if cy > 500000:                                                   # undo the carry of a negative cy in the key
            cx, cy = cx + 1, cy - 1000003
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                kk = (cx + dx) * 1000003 + (cy + dy)
                if kk in where:
                    cand.append(order[where[kk][0]:where[kk][1]])
        if not cand:
            continue
        cand = np.concatenate(cand)
        d = X[cand][None, :, :] - X[mine][:, None, :]                   # (mine, cand, 2)
        h = H[mine][:, None, :]
        lon = (d * h).sum(-1)
        lat = -d[..., 0] * h[..., 1] + d[..., 1] * h[..., 0]            # + = left of the heading
        cosang = (H[cand][None, :, :] * h).sum(-1)
        other = lane[cand][None, :] != lane[mine][:, None]
        abeam = (np.abs(lon) <= lon_max) & other
        for sign, target, par in ((1, left_of, True), (-1, right_of, True), (1, opp_l, False), (-1, opp_r, False)):
            side = lat * sign
            ok = abeam & (side >= lat_min) & (side <= lat_max) & ((cosang >= cos_par) if par else (cosang <= -cos_par))
            cost = np.where(ok, side, np.inf)
            j = cost.argmin(axis=1)
            has = np.isfinite(cost[np.arange(len(mine)), j])
            target[mine[has]] = cand[j[has]]
    pair = {}
    for name, arr in (("left", left_of), ("right", right_of), ("opposite", opp_l), ("opposite", opp_r)):         # the nearest opposite lane on each side
        for i in np.nonzero(arr >= 0)[0]:
            pair[(name, lane[i], lane[arr[i]])] = pair.get((name, lane[i], lane[arr[i]]), 0) + 1
    out = {name: {t: [] for t in toks} for name in ("left", "right", "opposite")}
    for (name, a, b), c in sorted(pair.items()):
        if c >= min_overlap:
            out[name][toks[a]].append(toks[b])
    # lane count: follow the left / right neighbours point by point
    count = np.ones(n, np.int64)
    idx_left = np.zeros(n, np.int64)
    for arr, is_left in ((left_of, True), (right_of, False)):
        cur = arr.copy()
        for _ in range(hops):
            m = cur >= 0
            count += m
            if is_left:
                idx_left += m
            cur = np.where(m, arr[np.maximum(cur, 0)], -1)
    for name in ("lane_count", "lane_min", "lane_max", "lane_index"):
        out[name] = {}
    for li, t in enumerate(toks):
        m = lane == li
        out["lane_count"][t] = int(np.median(count[m]))
        out["lane_min"][t], out["lane_max"][t] = int(count[m].min()), int(count[m].max())
        out["lane_index"][t] = int(np.median(idx_left[m]))
    return out


def build_road_graph(mg, **params):
    """roads, intersections and the road graph of a map graph (`lane_graph.build_map_graph`) in global coordinates"""
    import shapely
    from shapely.strtree import STRtree
    p = {**DEFAULTS, **params}
    polys, kind, succ, pred, piece_of, pieces = mg["polys"], mg["kind"], mg["succ"], mg["pred"], mg["piece_of"], mg["pieces"]
    nb = neighbours(polys, kind, **p)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    lanes = [t for t in polys if kind[t] == "lane"]
    for t in lanes:
        find(t)
        for name in ("left", "right"):
            for b in nb[name][t]:
                parent[find(t)] = find(b)
        if len(succ[t]) == 1 and kind[succ[t][0]] == "lane" and len(pred[succ[t][0]]) == 1:
            parent[find(t)] = find(succ[t][0])
    comp = {}
    for t in lanes:
        comp.setdefault(find(t), []).append(t)
    road_of = {}
    roads = {}
    for members in comp.values():
        rid = "r" + min(members)[:8]
        roads[rid] = dict(id=rid, kind="road", lanes=sorted(members), lane_max=max(nb["lane_max"][t] for t in members))
        for t in members:
            road_of[t] = rid
    # intersections: the connected areas of the intersection segments, the connectors inside them
    area_of, areas, area_geoms = {}, {}, []
    inter = mg.get("inter")
    if inter is not None:
        geoms = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
        tree = STRtree(geoms)
        for t, flag in mg["in_inter"].items():
            if flag:
                P = polys[t]
                mid = shapely.Point(*P[len(P) // 2])
                hit = tree.query(mid, predicate="intersects")
                if len(hit):
                    area_of[t] = int(hit[0])
        by_area = {}
        for t, a in area_of.items():
            by_area.setdefault(a, []).append(t)
        names = {a: "x" + min(ts)[:8] for a, ts in by_area.items()}
        for a, ts in by_area.items():
            areas[names[a]] = dict(id=names[a], kind="intersection", connectors=sorted(ts))
        area_of = {t: names[a] for t, a in area_of.items()}
        area_geoms = [(names.get(a), g) for a, g in enumerate(geoms)]               # every intersection area, with the id of its node (None: no connector in it)
    for t in polys:                                                       # connectors outside every intersection: the road of the lane before / after
        if kind[t] == "connector" and t not in area_of:
            ref = [u for u in pred[t] if u in road_of] or [v for v in succ[t] if v in road_of]
            if ref:
                road_of[t] = road_of[ref[0]]
    node_of = {**road_of, **area_of}
    edges = {}
    for t, vs in succ.items():
        for v in vs:
            a, b = node_of.get(t), node_of.get(v)
            if a is not None and b is not None and a != b:
                edges.setdefault(a, set()).add(b)
    opposite = {}
    for t in lanes:
        for b in nb["opposite"][t]:
            if b in road_of and road_of[b] != road_of[t]:
                opposite.setdefault(road_of[t], set()).add(road_of[b])
    # per piece: the nodes along it, the neighbours as piece ids, the lane count
    plinks = {}
    for pid, pc in pieces.items():
        seq, tokens_lane = [], []
        for tok, i0, i1 in pc["spans"]:
            nid = node_of.get(tok)
            if nid is not None and (not seq or seq[-1][0] != nid):
                seq.append((nid, i0, i1))
            elif nid is not None:
                seq[-1] = (nid, seq[-1][1], i1)
            if kind[tok] == "lane":
                tokens_lane.append(tok)
        link = dict(along=seq, left=set(), right=set(), opposite=set())
        for tok in tokens_lane:
            for name in ("left", "right", "opposite"):
                link[name].update(piece_of[b] for b in nb[name][tok] if piece_of[b] != pid)
        link["lane_count"] = int(np.median([nb["lane_count"][t] for t in tokens_lane])) if tokens_lane else 0
        link["lane_index"] = int(np.median([nb["lane_index"][t] for t in tokens_lane])) if tokens_lane else 0
        plinks[pid] = link
    return dict(nb=nb, roads=roads, areas=areas, area_geoms=area_geoms, road_of=road_of, area_of=area_of, node_of=node_of, edges=edges, opposite=opposite, plinks=plinks,
                params=p)


def attach(graph, mg, rg):
    """add the road structure to a crop (`lane_graph.crop` calls this with rg = build_road_graph(mg)): per part `road`, `roads_along`, `left`,
    `right`, `opposite`, `lane_count`, `lane_index`; and graph["roads"], the roads and intersections present in the window with their edges"""
    parts_of = {}
    for part in graph["pieces"]:
        parts_of.setdefault(part["piece"], []).append(part)
    present = set()
    for part in graph["pieces"]:
        link = rg["plinks"][part["piece"]]
        first, last = part["seg"]
        mid = (first + last) // 2
        along = [a for a in link["along"] if a[2] >= first and a[1] <= last + 1]
        mid_node = next((a[0] for a in link["along"] if a[1] <= mid <= a[2]), along[0][0] if along else None)
        part["road"] = mid_node
        part["roads_along"] = [a[0] for a in along]
        for name in ("left", "right", "opposite"):
            part[name] = sorted(q["id"] for pid in link[name] for q in parts_of.get(pid, []))
        part["lane_count"], part["lane_index"] = link["lane_count"], link["lane_index"]
        present.update(part["roads_along"])
    nodes = []
    for nid in sorted(present):
        node = rg["roads"].get(nid) or rg["areas"][nid]
        nodes.append(dict(id=nid, kind=node["kind"], lane_max=node.get("lane_max", 0),
                          pieces=sorted(q["id"] for q in graph["pieces"] if nid in q["roads_along"]),
                          succ=sorted(b for b in rg["edges"].get(nid, ()) if b in present),
                          opposite=sorted(b for b in rg["opposite"].get(nid, ()) if b in present)))
    graph["roads"] = nodes
    return graph
