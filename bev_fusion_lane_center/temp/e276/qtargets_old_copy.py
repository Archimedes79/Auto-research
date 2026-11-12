"""Targets of the lane head (E273) from the lane GT of autogt (`autogt.lane_gt`, frame "lane": x right, y forward, |x|, |y| <= 50 m): nodes on a 2 m grid, the links between them,
the section starts, and the oracle that decodes nodes + links back to curves (the representation check of the card, section 3.4).

    nt = node_table(g)                       # g = one lane GT graph (dict); -> nodes, links, counts
    gt = grid_targets(nt)                    # the 50 x 50 maps of the node head: exists, flags, offset, direction, section start, long-link endpoints
    polys = decode(nt)                       # Hermite curves of all links (short: handles chord / 3; long: two handle scales fitted to the GT path between the nodes)
    cov = coverage(polys, g)                 # GT points within 0.3 m of the decoded curves (lane / connector), decoded points within 0.3 m of the GT

The rules (card E273, section 3.3), all by LOCATION, no matching:
  node of a cell      per piece and 2 m half-open cell the point nearest to the cell centre; event points win their cell; at most one node per cell
  suppression         points of connector segments inside an intersection give no node (unless a painted divider lies within `marker_dist`, default 0: never);
                      event points (stop, exit, split, merge, start, end, window) are always nodes
  flags               stop, exit, split, merge, start, end, win_in, win_out (events at the same place are one node with several flags)
  piece of a node     the piece that ENDS at the event for stop / split / end / win_out, the piece that STARTS there for merge / exit / start / win_in
  links               a node to every node reached next along the points of the pieces (through suppressed points and over the joins of succ); distance above `short` = long link
  section start       the first node (smallest point index) of a piece: s = 1 (the ids change only at splits and merges, so a piece is a section)
Frame of the maps: row = x, column = y, cell (i, j) has its centre at -50 + 2 (i + 0.5) / -50 + 2 (j + 0.5), as the feature raster of the project.
Run the self-test: python qtargets.py     (real frames of the lane GT store: `test_qtargets.py`)
"""
import numpy as np

N, RES, HALF = 50, 2.0, 50.0
FLAGS = ("stop", "exit", "split", "merge", "start", "end", "win_in", "win_out")
FI = {f: i for i, f in enumerate(FLAGS)}
SHORT = 8.0
STARTERS = ("merge", "exit", "start", "win_in")                    # the piece STARTS at the event


def _tangents(p):
    if len(p) < 2:
        return np.tile([0.0, 1.0], (len(p), 1))
    d = np.gradient(p, axis=0)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


def cell_of(xy):
    return np.clip(np.floor((np.asarray(xy, float) + HALF) / RES).astype(int), 0, N - 1)


def centre_of(cell):
    return -HALF + (np.asarray(cell, float) + 0.5) * RES


def _groups(events, tol=0.3):
    """events at the same place (a stop is also a split, an exit a merge, ...) -> one group each"""
    groups = []
    for e in events:
        pos = np.asarray(e["pos"], float)
        gr = next((g for g in groups if np.linalg.norm(g["pos"] - pos) < tol), None)
        if gr is None:
            gr = dict(pos=pos, types=set(), ins=[], outs=[], on=None)
            groups.append(gr)
        t = e["type"]
        if t == "window":
            gr["types"].add("win_in" if e.get("io") == "in" else "win_out")
        elif t == "cross":
            gr["types"].update(("split", "merge"))
        else:
            gr["types"].add(t)
        for k in e.get("in", []):
            if k not in gr["ins"]:
                gr["ins"].append(k)
        for k in e.get("out", []):
            if k not in gr["outs"]:
                gr["outs"].append(k)
        if e.get("on") is not None:
            gr["on"] = e["on"]
    return groups


def node_table(g, marker_dist=0.0, short=SHORT):
    """-> dict: nodes (M): cell (M, 2) int, pos (M, 2), dirv (M, 2), flags (M, 8) bool, piece (M,) index into g["pieces"], j (M,) point index in the piece, ev (M,) bool, sstart (M,) bool;
    links: src (L,), dst (L,) node indices, long (L,) bool, path (L,) lists of global point ids; points: pts (P, 2), pid (P,), pj (P,), supp (P,) bool, off (n pieces + 1); stats"""
    pieces = g["pieces"]
    idx = {p["id"]: i for i, p in enumerate(pieces)}
    off = np.zeros(len(pieces) + 1, int)
    for i, p in enumerate(pieces):
        off[i + 1] = off[i] + len(p["pts"])
    P = off[-1]
    pts = np.zeros((P, 2))
    tan = np.zeros((P, 2))
    pid = np.zeros(P, int)
    pj = np.zeros(P, int)
    supp = np.zeros(P, bool)
    div = None
    if marker_dist > 0:
        from scipy.spatial import cKDTree
        lines = [l["pts"] for k in ("lane_divider", "road_divider") for l in g["lines"].get(k, []) if len(l["pts"])]
        div = cKDTree(np.concatenate(lines)) if lines else None
    for i, p in enumerate(pieces):
        q = np.asarray(p["pts"], float)
        a, b = off[i], off[i + 1]
        pts[a:b], tan[a:b], pid[a:b], pj[a:b] = q, _tangents(q), i, np.arange(len(q))
        for s in p["segments"]:
            if s["kind"] == "connector" and s["in_intersection"]:
                supp[a + s["i0"]: a + s["i1"] + 1] = True
        if div is not None and supp[a:b].any():
            near = div.query(q)[0] <= marker_dist
            supp[a:b] &= ~near
    # successors of every point: the next point of the piece; the last point -> the first point of every succ part
    succ = [[] for _ in range(P)]
    for i, p in enumerate(pieces):
        for k in range(off[i], off[i + 1] - 1):
            succ[k].append(k + 1)
        for q in p["succ"]:
            if q in idx:
                succ[off[i + 1] - 1].append(off[idx[q]])
    # event nodes
    nodes = {}            # gid -> dict
    cell_event = {}
    n_ev_merged = 0
    for gr in _groups(g["events"]):
        types = gr["types"]
        cand = (gr["outs"] or ([gr["on"]] if gr["on"] else [])) if types & set(STARTERS) else (gr["ins"] or ([gr["on"]] if gr["on"] else []))
        cand = [c for c in cand if c in idx]
        if not cand:
            continue
        pi = idx[cand[0]]
        a, b = off[pi], off[pi + 1]
        gid = a + int(np.argmin(np.linalg.norm(pts[a:b] - gr["pos"], axis=1)))
        c = tuple(cell_of(gr["pos"]))
        if c in cell_event:                                         # two event places in one cell: one node, the flags add up
            nodes[cell_event[c]]["flags"] |= set(types)
            n_ev_merged += 1
            continue
        cell_event[c] = gid
        nodes[gid] = dict(flags=set(types), pos=gr["pos"].copy(), ev=True)
    # normal nodes: per piece and cell the free point nearest to the cell centre
    best = {}
    for i in range(len(pieces)):
        a, b = off[i], off[i + 1]
        cells = cell_of(pts[a:b])
        for k in range(b - a):
            gid = a + k
            if supp[gid] or gid in nodes:
                continue
            c = (int(cells[k][0]), int(cells[k][1]))
            d = float(np.linalg.norm(pts[gid] - centre_of(c)))
            key = (i, c)
            if key not in best or d < best[key][0]:
                best[key] = (d, gid)
    per_cell = {}
    for (i, c), (d, gid) in best.items():
        if c in cell_event:
            continue
        if c not in per_cell or d < per_cell[c][0]:
            per_cell[c] = (d, gid)
    n_cand = len(best)
    for c, (d, gid) in per_cell.items():
        nodes[gid] = dict(flags=set(), pos=pts[gid].copy(), ev=False)
    n_dropped = n_cand - len(per_cell) - sum(1 for (i, c) in best if c in cell_event)
    gids = sorted(nodes)
    M = len(gids)
    node_of = {gid: m for m, gid in enumerate(gids)}
    cell = cell_of(np.array([nodes[gd]["pos"] for gd in gids])) if M else np.zeros((0, 2), int)
    flags = np.zeros((M, len(FLAGS)), bool)
    for m, gd in enumerate(gids):
        for f in nodes[gd]["flags"]:
            flags[m, FI[f]] = True
    piece = pid[gids] if M else np.zeros(0, int)
    j = pj[gids] if M else np.zeros(0, int)
    sstart = np.zeros(M, bool)
    first = {}
    for m in range(M):
        if piece[m] not in first or j[m] < j[first[piece[m]]]:
            first[piece[m]] = m
    for m in first.values():
        sstart[m] = True
    # links by contraction: from every node along the points until the next node
    src, dst, paths = [], [], []
    for m, gd in enumerate(gids):
        parent = {gd: None}
        queue = list(succ[gd])
        for s in queue:
            parent.setdefault(s, gd)
        head = 0
        while head < len(queue):
            s = queue[head]
            head += 1
            if s in node_of and s != gd:
                path = [s]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                src.append(m)
                dst.append(node_of[s])
                paths.append(np.array(path[::-1]))
                continue
            for t in succ[s]:
                if t not in parent:
                    parent[t] = s
                    queue.append(t)
    src, dst = np.array(src, int), np.array(dst, int)
    pos = np.array([nodes[gd]["pos"] for gd in gids]) if M else np.zeros((0, 2))
    dist = np.linalg.norm(pos[src] - pos[dst], axis=1) if len(src) else np.zeros(0)
    return dict(cell=cell, pos=pos, dirv=tan[gids] if M else np.zeros((0, 2)), flags=flags, piece=piece, j=j, ev=np.array([nodes[gd]["ev"] for gd in gids], bool), sstart=sstart,
                src=src, dst=dst, long=dist > short, dist=dist, path=paths, pts=pts, tan=tan, pid=pid, pj=pj, supp=supp, off=off, gid=np.array(gids, int),
                stats=dict(nodes=M, events=int(sum(nodes[gd]["ev"] for gd in gids)), dropped=int(n_dropped), cand=int(n_cand), ev_merged=n_ev_merged, links=len(src),
                           long=int((dist > short).sum())))


def grid_targets(nt, n=N):
    """the maps of the node head from a node table: exists (n, n), flags (8, n, n), offset (2, n, n) metres from the cell centre, dirv (2, n, n), sstart (n, n), piece (n, n) int (-1 = none),
    uout / uin (n, n): the node starts / ends a long link"""
    t = dict(exists=np.zeros((n, n), np.float32), flags=np.zeros((len(FLAGS), n, n), np.float32), offset=np.zeros((2, n, n), np.float32), dirv=np.zeros((2, n, n), np.float32),
             sstart=np.zeros((n, n), np.float32), piece=-np.ones((n, n), np.int32), uout=np.zeros((n, n), np.float32), uin=np.zeros((n, n), np.float32))
    for m in range(len(nt["pos"])):
        i, j = nt["cell"][m]
        t["exists"][i, j] = 1
        t["flags"][:, i, j] = nt["flags"][m]
        t["offset"][:, i, j] = nt["pos"][m] - centre_of((i, j))
        t["dirv"][:, i, j] = nt["dirv"][m]
        t["sstart"][i, j] = nt["sstart"][m]
        t["piece"][i, j] = nt["piece"][m]
    for a, b, lg in zip(nt["src"], nt["dst"], nt["long"]):
        if lg:
            t["uout"][tuple(nt["cell"][a])] = 1
            t["uin"][tuple(nt["cell"][b])] = 1
    return t


def link_array(nt):
    """-> (L, 4) int cells (iu, ju, iv, jv) of the links and (L,) bool long"""
    c = nt["cell"]
    return np.concatenate([c[nt["src"]], c[nt["dst"]]], 1).astype(np.int64) if len(nt["src"]) else np.zeros((0, 4), np.int64), nt["long"].copy()


def hermite(p0, h0, p1, h1, a=1 / 3, b=1 / 3, step=0.5):
    c = float(np.linalg.norm(p1 - p0))
    n = max(int(c / step) + 2, 3)
    s = np.linspace(0, 1, n)[:, None]
    q1, q2 = p0 + a * c * h0, p1 - b * c * h1
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * q1 + 3 * (1 - s) * s ** 2 * q2 + s ** 3 * p1


def fit_long_scales(nt, scales=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85)):
    """the two Hermite handle scales (a, b) of every LONG link, fitted to the true path between its two nodes (grid search): (L, 2), NaN for the short links.
    This is what the learned scales of the link head are trained towards"""
    from scipy.spatial import cKDTree
    out = np.full((len(nt["src"]), 2), np.nan)
    for l, (a, b, lg, path) in enumerate(zip(nt["src"], nt["dst"], nt["long"], nt["path"])):
        if not lg:
            continue
        p0, h0, p1, h1 = nt["pos"][a], nt["dirv"][a], nt["pos"][b], nt["dirv"][b]
        tgt = nt["pts"][path]
        tt = cKDTree(tgt)
        best = 1e9
        for sa in scales:
            for sb in scales:
                cur = hermite(p0, h0, p1, h1, sa, sb)
                m = tt.query(cur)[0].mean() + 0.2 * cKDTree(cur).query(tgt)[0].mean()
                if m < best:
                    best, out[l] = m, (sa, sb)
    return out


def decode(nt, fit_long=True, fit=None):
    """Hermite curve of every link -> list of (K, 2) arrays; long links: the handle scales are the fitted ones (`fit` = `fit_long_scales(nt)`, computed here if not given), the oracle of what
    the learned scales can reach"""
    if fit is None and fit_long:
        fit = fit_long_scales(nt)
    out = []
    for l, (a, b, lg) in enumerate(zip(nt["src"], nt["dst"], nt["long"])):
        p0, h0, p1, h1 = nt["pos"][a], nt["dirv"][a], nt["pos"][b], nt["dirv"][b]
        if lg and fit is not None and not np.isnan(fit[l, 0]):
            out.append(hermite(p0, h0, p1, h1, fit[l, 0], fit[l, 1]))
        else:
            out.append(hermite(p0, h0, p1, h1))
    return out


def coverage(polys, g, tol=0.3):
    """-> dict: share of the GT points (lane / connector / all) within `tol` of the decoded curves, share of the decoded points within `tol` of a GT point; the keys `*_1m` the same at 1 m"""
    from scipy.spatial import cKDTree
    gp, kd = [], []
    for p in g["pieces"]:
        q = np.asarray(p["pts"], float)
        k = np.array(["lane"] * len(q), dtype=object)
        for s in p["segments"]:
            k[s["i0"]: s["i1"] + 1] = s["kind"]
        gp.append(q)
        kd.append(k)
    gp, kd = np.concatenate(gp), np.concatenate(kd)
    if not len(polys):
        return dict(all=0.0, lane=0.0, connector=0.0, prec=1.0, all_1m=0.0, lane_1m=0.0, connector_1m=0.0, prec_1m=1.0)
    dec = np.concatenate(polys)
    dg = cKDTree(dec).query(gp)[0]
    dd = cKDTree(gp).query(dec)[0]
    out = {}
    for suf, t in (("", tol), ("_1m", 1.0)):
        r = dg <= t
        out["all" + suf], out["prec" + suf] = float(r.mean()), float((dd <= t).mean())
        for k in ("lane", "connector"):
            out[k + suf] = float(r[kd == k].mean()) if (kd == k).any() else 1.0
    return out


def _piece(pid, pts, kind="lane", succ=(), pred=(), start_open=False, end_open=False, inter=False):
    pts = np.asarray(pts, float)
    return dict(id=pid, piece=pid, part=0, kind=kind, pts=pts, start_open=start_open, end_open=end_open, length=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), tokens=[pid],
                succ=list(succ), pred=list(pred), segments=[dict(token=pid, kind=kind, i0=0, i1=len(pts) - 1, in_intersection=inter)], road="r", roads_along=["r"], left=[], right=[],
                opposite=[], lane_count=1, lane_index=0)


def _selftest():
    ys = lambda a, b: np.stack([np.zeros(int((b - a) / 0.5) + 1), np.arange(a, b + 1e-9, 0.5)], 1)
    A = _piece("A", ys(-40, 0), succ=["B", "C"], start_open=True)
    B = _piece("B", ys(0, 40), end_open=True, pred=["A"])
    ang = np.linspace(0, 1, 61)[:, None]
    Cpts = np.concatenate([np.stack([0 * ang[:, 0] + 0, 0 * ang[:, 0]], 1)[:0], np.stack([20 * ang[:, 0] ** 2, 30 * ang[:, 0]], 1)], 0)
    C = _piece("C", Cpts, kind="connector", pred=["A"], end_open=True, inter=True)
    ev = [dict(type="window", io="in", pos=[0.0, -40.0], **{"in": [], "out": ["A"]}, dirs_in=[], dirs_out=[[0, 1]], partial=False),
          dict(type="split", pos=[0.0, 0.0], **{"in": ["A"], "out": ["B", "C"]}, dirs_in=[[0, 1]], dirs_out=[[0, 1], [0, 1]], partial=False),
          dict(type="stop", pos=[0.0, 0.0], on="A", s=40.0, n_succ=2, **{"in": ["A"], "out": ["B", "C"]}, dirs_in=[[0, 1]], dirs_out=[[0, 1], [0, 1]], partial=False),
          dict(type="window", io="out", pos=[0.0, 40.0], **{"in": ["B"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False),
          dict(type="window", io="out", pos=[float(Cpts[-1, 0]), float(Cpts[-1, 1])], **{"in": ["C"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False)]
    g = dict(pieces=[A, B, C], events=ev, lines=dict(lane_divider=[], road_divider=[], boundary=[]))
    nt = node_table(g)
    st = nt["stats"]
    assert st["nodes"] >= 20 and nt["flags"][:, FI["split"]].sum() == 1 and nt["flags"][:, FI["stop"]].sum() == 1, st
    assert nt["flags"][:, FI["win_in"]].sum() == 1 and nt["flags"][:, FI["win_out"]].sum() == 2, "window events"
    # the connector C is inside an intersection: no node of it except its window end
    assert (nt["piece"] == 2).sum() == 1, "a suppressed connector keeps only its window event node"
    # the split node belongs to A (the piece that ends there); it has two out-links (to the next node of B and the window node of C)
    m = int(np.where(nt["flags"][:, FI["split"]])[0][0])
    assert nt["piece"][m] == 0 and (nt["src"] == m).sum() == 2, ((nt["src"] == m).sum(), nt["piece"][m])
    assert nt["long"].sum() == 1, "the link over the suppressed connector is the one long link"
    # sections: the first node of each piece is its start
    assert nt["sstart"].sum() == 3 - 1 or nt["sstart"].sum() == 3, nt["sstart"].sum()
    gt = grid_targets(nt)
    assert gt["exists"].sum() == st["nodes"] and gt["uout"].sum() == 1 and gt["uin"].sum() == 1
    assert np.abs(gt["offset"]).max() <= 1.0 + 1e-6, "an offset leaves its cell"
    cov = coverage(decode(nt), g)
    assert cov["lane"] > 0.999 and cov["prec"] > 0.97, cov
    print("qtargets selftest ok", st, {k: round(v, 4) for k, v in cov.items()})


if __name__ == "__main__":
    _selftest()
