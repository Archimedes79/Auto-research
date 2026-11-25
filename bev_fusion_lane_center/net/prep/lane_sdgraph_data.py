"""E390 step 1 (card experiments/E390_sd-map-logical-road-graph): the SD map as a LOGICAL ROAD GRAPH, from the lane GT's road graph (`g['roads']`, GtGenerator/road_graph.py), no perception input.

    wsl bash tools/env/_py14.sh lane_sdgraph_data.py build --split val|train [--limit N]   # runs/e390_sdgraph/<split>.pkl (+ .json manifest): the exact graphs of the frames (derived data, not in git; never overwritten)
    wsl bash tools/env/_py14.sh lane_sdgraph_data.py stats --split val|train               # nodes / links / forks per frame, the share of true links inside the P2 gap
    wsl bash tools/env/_py14.sh lane_sdgraph_data.py selftest                              # hand graphs; tests with mutants: net/tests/test_sdgraph.py

A node = one node of g['roads'] in the +-50 m window: a ROAD (one direction of travel: its co-directional lanes; `lane_max` is NOT used) or an INTERSECTION. Geometry (card v2, critic B2: no lane-level geometry that
encodes the count): a road = its DIVIDER-SIDE EDGE line, i.e. the chain of its lanes next to the opposite direction (the pieces with an `opposite` link, else the pieces without a neighbour on the divider side:
left in right-hand traffic (Boston), right in left-hand traffic (Singapore)), offset by half a lane (1.75 m) towards the divider, so a two-way road's two directions share one line up to GT noise and the line does
not move with the lane count; an intersection = its connectors (the polylines of its pieces), with their entry (start) and exit (end) points. Attributes: lanes in the direction of travel = the length-weighted median
of the road's pieces' lane_count (classes 1, 2, 3, 4+), the opposite road (link), the route (a node holding a tier-1 route piece, `lane_route_data.route_of`, its order = the driving order of its first route piece;
the route is the simulated navigation input as in E389, D127 / D128). Links: successor (g['roads'] succ), opposite. No lane-level link, lane index, piece id or road id leaves this module.

`frame_graph(g)` -> the exact graph; `perturb(G, rng, ...)` -> (the input graph, the targets): count errors (+-1 at a rate, 'unknown' at p), dropped successor links (the P2 positives), hidden route flags (the P3
positives), a SMOOTH deformation of the whole map (sigma m: joints stay joined), Douglas-Peucker, a rigid transform of the frame; `featurize(Gin)` -> the encoder's arrays, ALL derived from the INPUT graph (critic B1):
node features in the node's OWN frame, the relation types (succ, pred, opp, junction sibling, near = within NEAR m whether linked or not), the relative geometry of every pair (rigid-motion invariant), and the ego-
frame pose (kept apart, for step 2). Rule nulls: P1 copy / neighbour vote (and the Bayes ceiling in evaluation/sdgraph_probe.py), P2 the geometric link rule, P3 the link-to-route rule.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "training/data/prep", "net/prep", "net/model", "net/loss", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))

VERSION = "v6"
CACHE = os.path.join(ROOT, "runs", "e390_sdgraph")
N_LANE = 4                      # lane classes 1, 2, 3, 4+ (index 0 - 3); input class 4 = unknown
N_OPP = 6                       # opposite lanes: 0 = no opposite road, 1 - 4+, 5 = unknown
GAP = 10.0                      # m: P2 candidate pairs = an exit of a -> an entry of b within this gap
NEAR = 30.0                     # m: the 'near' relation (any two nodes, linked or not)
HALF_LANE = 1.75                # m: the divider-side edge = the divider-side lane centre offset by this
MAX_NODES = 64                  # nodes per frame kept (the nearest the ego); `stats` counts the frames at the cap
WAVE = 60.0                     # m: the wavelength of the smooth deformation
PROX_STEP = 8                   # lane points (0.5 m apart) between two perception proxies: one every 4 m
MAX_PROX = 160                  # proxies per frame kept (the nearest the ego)
PA_F = 8
TIE_TOL = 1e-6                  # m: candidates within this of the minimum are a tie, decided by the stored order (card v6: exact ties of the 1.75 m offset were decided by float noise)
APPROACH = 6.0                  # m: intersection entries (exits) closer than this are one approach (code critic: a count per connector carries the lane counts)
REL = ("succ", "pred", "opp", "sib", "near")
NODE_F = 15 + N_LANE + 1 + N_OPP
EDGE_F = 11


# ------------------------------------------------------------------------------------------------------------------------------ geometry helpers
def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _heading(p, end=False):
    q = p[-min(len(p), 6):] if end else p[: min(len(p), 6)]
    d = q[-1] - q[0] if len(q) > 1 else np.array([1.0, 0.0])
    return float(np.arctan2(d[1], d[0]))


def _offset(p, dist):
    """the polyline p moved sideways by dist (> 0 = to the left of the direction of travel)"""
    if len(p) < 2:
        return p
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return p + dist * np.stack([-t[:, 1], t[:, 0]], 1)


def _chain(pieces, ks, order=False):
    """the longest chain (by length) of the pieces ks along their succ links -> the concatenated points (travel order), or with order=True the piece indices of the chain"""
    idx = {pieces[k]["id"]: k for k in ks}
    nxt = {k: [idx[s] for s in pieces[k]["succ"] if s in idx] for k in ks}
    has_pred = {j for k in ks for j in nxt[k]}
    best, best_len = None, -1.0
    for s in [k for k in ks if k not in has_pred] or list(ks):
        seq, seen = [s], {s}
        while nxt[seq[-1]] and nxt[seq[-1]][0] not in seen:
            seq.append(nxt[seq[-1]][0])
            seen.add(seq[-1])
        ln = sum(pieces[k]["length"] for k in seq)
        if ln > best_len:
            best, best_len = seq, ln
    if order:
        return best
    return np.concatenate([np.asarray(pieces[k]["pts"], float) for k in best])


def _inside(pt, poly):
    """point in polygon (ray casting); poly (n, 2) the exterior ring"""
    x, y = float(pt[0]), float(pt[1])
    n, c = len(poly), False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            c = not c
    return c


def _cut(piece, inside, area=None):
    """the point runs of a piece whose segments lie (inside=True) / do not lie in an intersection (`segments[i].in_intersection`); a piece without segments counts as outside; with `area` (the exterior ring of ONE
    intersection) an inner run is kept only when its midpoint lies in that area (the critic of v4: a piece crossing two intersections is listed on both)"""
    q = np.asarray(piece["pts"], float)
    segs = piece.get("segments") or [dict(i0=0, i1=len(q) - 1, in_intersection=False)]
    out = []
    for s in segs:
        if bool(s.get("in_intersection")) == inside:
            a, b = int(s["i0"]), int(s["i1"]) + 1
            if b - a >= 2 and (area is None or _inside(q[(a + b - 1) // 2], area)):
                out.append(q[a:b])
    return out


def right_hand(location):
    return str(location).startswith("boston")


# ------------------------------------------------------------------------------------------------------------------------------ the exact graph of a frame
def frame_graph(g, route=None):
    """the exact SD road graph of a lane-GT frame -> dict(kind (N,) 0 road / 1 intersection, geo [N lists of polylines (n, 2)], lanes (N,) class 0 - 3 (-1 intersection), route (N,) order >= 0 or -1, succ [(i, j)],
    oppl [(i, j)] both ways); `route` = `lane_route_data.route_of(g)` (piece indices in driving order) or None (computed)"""
    if route is None:
        import lane_route_data as LR
        route = LR.route_of(g)
    rh = right_hand(g.get("location", ""))
    side = "left" if rh else "right"                                          # the divider side of a road
    pieces = g["pieces"]
    pidx = {p["id"]: k for k, p in enumerate(pieces)}
    rorder = {}
    for o, k in enumerate(route):
        rorder.setdefault(pieces[k]["road"], o)
    nodes = [r for r in g["roads"] if any(p in pidx for p in r.get("pieces", []))]
    nid = {r["id"]: i for i, r in enumerate(nodes)}
    memb = {}                                                                 # piece index -> the nodes that list it (a piece runs along up to five nodes: the critic of v3)
    for i, r in enumerate(nodes):
        for p in r["pieces"]:
            if p in pidx:
                memb.setdefault(pidx[p], []).append(i)
    kind, geo, lanes = [], [], []
    for r in nodes:
        ks = [pidx[p] for p in r["pieces"] if p in pidx]
        if r["kind"] == "road":
            own = [k for k in ks if sum(nodes[j]["kind"] == "road" for j in memb[k]) == 1] or ks      # the pieces of no other road (a piece through a split is listed on both)
            div = [k for k in own if not pieces[k].get(side)] or own                  # the lanes without a co-directional neighbour on the divider side
            runs = [q for k in _chain(pieces, div, order=True) for q in _cut(pieces[k], False)]           # cut at the intersections: their stretches belong to the intersection node
            line = np.concatenate(runs) if runs else _chain(pieces, div)
            geo.append([_offset(line, HALF_LANE if rh else -HALF_LANE)])
            w = np.array([pieces[k]["length"] for k in ks], float)
            c = np.array([pieces[k]["lane_count"] for k in ks], float)
            o = np.argsort(c, kind="stable")
            med = c[o][min(len(c) - 1, int(np.searchsorted(np.cumsum(w[o]), 0.5 * w.sum())))] if w.sum() > 0 else c.max()
            kind.append(0)
            lanes.append(min(int(med), N_LANE) - 1)
        else:                                                                 # an intersection: only the stretches of its pieces INSIDE it (the critic of v3: whole road pieces sat in the token)
            area = next((np.asarray(a["exterior"], float) for a in g.get("areas", {}).get("intersection", []) if a.get("id") == r["id"] and len(a.get("exterior", [])) >= 3), None)
            runs = [q for k in ks for q in _cut(pieces[k], True, area)]
            geo.append(runs or [np.asarray(pieces[k]["pts"], float) for k in ks if len(memb[k]) == 1 and len(pieces[k]["pts"]) >= 2])
            kind.append(1)
            lanes.append(-1)
    succ = sorted({(nid[r["id"]], nid[s]) for r in nodes for s in r.get("succ", []) if s in nid and s != r["id"]})
    oppl = sorted({(nid[r["id"]], nid[s]) for r in nodes for s in r.get("opposite", []) if s in nid} | {(nid[s], nid[r["id"]]) for r in nodes for s in r.get("opposite", []) if s in nid})
    # D135 (P6): the perception PROXIES = the true lane centre points of the frame (every PROX_STEP-th point of every piece of a node, with its travel direction), labelled with their node (the segment the
    # perceived element belongs to): the 'perceived' side of the association, in the true geometry (the map errors move only the map)
    pp, pd, ps = [], [], []
    n_all = n_one = 0
    for k, p in enumerate(pieces):
        if k not in memb or len(p["pts"]) < 2:
            continue
        n_all += 1
        if len(memb[k]) != 1:                                                 # a piece listed on several nodes: its proxies would have several right answers: left out (counted)
            continue
        n_one += 1
        q = np.asarray(p["pts"], float)
        t = np.gradient(q, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        sel = np.arange(PROX_STEP // 2, len(q), PROX_STEP)
        pp.append(q[sel]), pd.append(t[sel]), ps.append(np.full(len(sel), memb[k][0]))
    prox = np.concatenate(pp) if pp else np.zeros((0, 2))
    pdir = np.concatenate(pd) if pd else np.zeros((0, 2))
    pseg = np.concatenate(ps).astype(int) if ps else np.zeros(0, int)
    if len(prox) > MAX_PROX:
        o = np.sort(np.argsort(np.linalg.norm(prox, axis=1), kind="stable")[:MAX_PROX])
        prox, pdir, pseg = prox[o], pdir[o], pseg[o]
    G = dict(kind=np.array(kind, int), geo=geo, lanes=np.array(lanes, int), route=np.array([rorder.get(r["id"], -1) for r in nodes], int), succ=succ, oppl=oppl, prox=prox, pdir=pdir, pseg=pseg,
             prox_pieces=(n_one, n_all))
    return keep_nearest(G)


def ends(G, i):
    """(entries (m, 2), entry headings (m,), exits (m, 2), exit headings (m,)) of node i: a road its first / last point, an intersection those of its connectors"""
    gs = [p for p in G["geo"][i] if len(p) >= 2] or [np.zeros((2, 2))]
    return (np.array([p[0] for p in gs]), np.array([_heading(p) for p in gs]), np.array([p[-1] for p in gs]), np.array([_heading(p, True) for p in gs]))


def keep_nearest(G, n=MAX_NODES):
    """at most n nodes: the nearest the ego (the distance of the geometry to the origin); the links among the kept, renumbered"""
    N = len(G["kind"])
    if N <= n:
        return G
    d = np.array([min(np.linalg.norm(p, axis=1).min() for p in gs) if gs else 1e9 for gs in G["geo"]])
    keep = np.sort(np.argsort(d, kind="stable")[:n])
    m = {int(k): i for i, k in enumerate(keep)}
    out = dict(kind=G["kind"][keep], geo=[G["geo"][k] for k in keep], lanes=G["lanes"][keep], route=G["route"][keep], prox_pieces=G.get("prox_pieces"))
    if "prox" in G:
        ok = np.array([int(s) in m for s in G["pseg"]], bool)
        out.update(prox=G["prox"][ok], pdir=G["pdir"][ok], pseg=np.array([m[int(s)] for s in G["pseg"][ok]], int))
    out["succ"] = [(m[a], m[b]) for a, b in G["succ"] if a in m and b in m]
    out["oppl"] = [(m[a], m[b]) for a, b in G["oppl"] if a in m and b in m]
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the map errors of step 1
def first_min(d, tol=TIE_TOL):
    """the index (unravelled) of the FIRST entry within tol of the minimum of d, in d's stored order: rigid-motion stable where float noise would decide an exact tie"""
    flat = np.asarray(d).ravel()
    return np.unravel_index(int(np.flatnonzero(flat <= flat.min() + tol)[0]), np.shape(d))


def rdp(p, eps):
    """Douglas-Peucker of a polyline (eps m); eps <= 0: unchanged"""
    if eps <= 0 or len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    d = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / n if n > 1e-9 else np.linalg.norm(p - a, axis=1)
    k = int(np.flatnonzero(d >= d.max() - TIE_TOL)[0])
    if d[k] <= eps:
        return np.stack([a, b])
    return np.concatenate([rdp(p[: k + 1], eps)[:-1], rdp(p[k:], eps)])


def smooth_field(rng, sigma):
    """a smooth random displacement field of the plane (two plane waves per axis, wavelength WAVE m, amplitude sigma m in total): the same point always moves the same way, so linked ends stay joined"""
    if sigma <= 0:
        return None
    th = rng.uniform(0, 2 * np.pi, 2)
    ph = rng.uniform(0, 2 * np.pi, (2, 2))
    amp = rng.normal(0, 1, (2, 2))
    amp *= sigma / max(np.sqrt((amp ** 2).sum() / 2), 1e-9)
    k = 2 * np.pi / WAVE * np.stack([np.cos(th), np.sin(th)], 1)

    def f(p):
        s = np.sin(p @ k.T + ph[:, 0])                                        # (n, 2 waves)
        c = np.cos(p @ k.T + ph[:, 1])
        return p + np.stack([s @ amp[:, 0], c @ amp[:, 1]], 1) / np.sqrt(2)
    return f


def perturb(G, rng, count_rate=0.0, unknown_p=0.0, drop=0.0, hide=0.0, sigma=0.0, eps=0.0, rigid=None, pose=0.0, node_drop=0.0):
    """-> (Gin, T). Gin = the input graph: lanes_in (0 - 3, 4 = unknown), the KEPT succ links, route_in (hidden flags -1), the geometry after the errors. T = the targets (exact lanes, succ, route; the dropped links;
    the hidden route nodes). count_rate: a road's count +-1 (clipped: 1 -> 2, 4+ -> 3); unknown_p: the count unknown; drop: a successor link removed; hide: a route node's flag removed (order -1); sigma: the smooth
    deformation (m); eps: Douglas-Peucker of the road lines; rigid = (dx, dy, angle) of the whole FRAME (map AND proxies) or None; D135 (P6): pose = a map-only pose error of this size (a shift of pose m in a
    random direction and a turn of pose degrees, random sign, about the ego: the map against the perceived proxies); node_drop = a node missing from the map (present False: masked in the encoder, its links and
    opposite links gone, its proxies' target 'unassigned' (-1)). The uniforms are drawn in a fixed order (count, sign, unknown, drop, hide, node drop, pose, then the deformation), so the levels of one generator
    are nested"""
    N = len(G["kind"])
    road = G["kind"] == 0
    flip = road & (rng.random(N) < count_rate)
    moved = G["lanes"] + np.where(rng.random(N) < 0.5, -1, 1)
    moved = np.where(moved < 0, 1, np.where(moved > N_LANE - 1, N_LANE - 2, moved))
    lanes_in = np.where(flip, moved, G["lanes"])
    lanes_in = np.where(road & (rng.random(N) < unknown_p), N_LANE, lanes_in)
    keep = rng.random(len(G["succ"])) >= drop
    hid = (G["route"] >= 0) & (rng.random(N) < hide)
    present = (rng.random(N) >= node_drop) & np.array([any(len(p) >= 2 for p in gs) for gs in G["geo"]], bool)   # card v6: a node without geometry is absent
    pang, psgn = rng.uniform(0, 2 * np.pi), (1.0 if rng.random() < 0.5 else -1.0)
    f = smooth_field(rng, sigma)
    geo = [[(f(p) if f else p.copy()) for p in gs] for gs in G["geo"]]
    geo = [[rdp(p, eps) for p in gs] if k == 0 else gs for gs, k in zip(geo, G["kind"])]
    present &= np.array([any(len(p) >= 2 and np.ptp(p, 0).max() > 1e-6 for p in gs) for gs in geo], bool)    # card v6 critic: a zero-length polyline (e.g. after Douglas-Peucker) has no heading: absent too
    if pose > 0:
        Rp, tp = rot(np.deg2rad(psgn * pose)), pose * np.array([np.cos(pang), np.sin(pang)])
        geo = [[p @ Rp.T + tp for p in gs] for gs in geo]
    prox, pdir = G.get("prox", np.zeros((0, 2))).copy(), G.get("pdir", np.zeros((0, 2))).copy()
    if rigid is not None:
        R, t = rot(rigid[2]), np.array(rigid[:2])
        geo = [[p @ R.T + t for p in gs] for gs in geo]
        prox, pdir = prox @ R.T + t, pdir @ R.T
    succ_in = [e for e, k in zip(G["succ"], keep) if k and present[e[0]] and present[e[1]]]
    oppl_in = [e for e in G["oppl"] if present[e[0]] and present[e[1]]]
    route_in = route_ranks(np.where(hid | ~present, -1, G["route"]))
    pseg = G.get("pseg", np.zeros(0, int))
    Gin = dict(G, geo=geo, lanes_in=lanes_in, succ=succ_in, oppl=oppl_in, route_in=route_in, present=present, prox=prox, pdir=pdir, count_rate=float(count_rate))
    return Gin, dict(lanes=G["lanes"], succ=G["succ"], dropped=[e for e, k in zip(G["succ"], keep) if not k and present[e[0]] and present[e[1]]], kind=G["kind"], route=G["route"],
                     hidden=np.flatnonzero(hid & present), present=present, pseg=np.where(present[pseg] if len(pseg) else np.zeros(0, bool), pseg, -1))


def route_ranks(route):
    """the route orders as ranks 0 .. m-1 of the flagged nodes (-1 stays -1): no gap in the input order (a hidden node leaves no hole) and no count of route pieces per node"""
    out = np.full(len(route), -1, int)
    f = np.flatnonzero(route >= 0)
    out[f[np.argsort(route[f], kind="stable")]] = np.arange(len(f))
    return out


def approaches(pts, tol=APPROACH):
    """the number of groups of points closer than tol (single linkage): the approaches of an intersection"""
    lab = np.arange(len(pts))
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < tol:
                lab[lab == lab[j]] = lab[i]
    return len(set(lab.tolist()))


# ------------------------------------------------------------------------------------------------------------------------------ the features of the encoder (from the INPUT graph only)
def _local(p, origin, h):
    return (np.asarray(p, float) - origin) @ rot(-h).T


def opp_in(G):
    """the input lane class of each road's opposite road: 0 = no opposite road, 1 - 4 = classes 1, 2, 3, 4+, 5 = unknown"""
    lin = G.get("lanes_in", G["lanes"])
    out = np.zeros(len(G["kind"]), int)
    for a, b in G["oppl"]:
        if G["kind"][a] == 0 and G["kind"][b] == 0:
            out[a] = 5 if lin[b] == N_LANE else lin[b] + 1
    return out


def node_features(G):
    """(N, NODE_F) float32, in the node's OWN frame: length / 50, heading change start -> end (cos, sin), curvature, kind, entries / exits (/ 4, intersections), extent / 20, in / out degree of the INPUT links (/ 4),
    the number of near nodes (/ 10), on-route flag, route order (/ 20), 2 zeros (slot 13: the map's stated count-error rate when `featurize(stated_rate=True)`, card v7; slot 14 spare), the input lane class one-hot
    (4 + unknown), the opposite road's input class one-hot (6)"""
    N = len(G["kind"])
    deg_in, deg_out = np.zeros(N), np.zeros(N)
    for a, b in G["succ"]:
        deg_out[a] += 1
        deg_in[b] += 1
    lin = G.get("lanes_in", G["lanes"])
    rin = G["route_in"] if "route_in" in G else route_ranks(G["route"])
    oin = opp_in(G)
    near = relations(G)[4].sum(1)
    F = np.zeros((N, NODE_F), np.float32)
    for i in range(N):
        gs = G["geo"][i]
        en, eh, ex, xh = ends(G, i)
        ln = float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in gs))
        pts = np.concatenate(gs) if gs else np.zeros((1, 2))
        ext = float(np.linalg.norm(pts - pts.mean(0), axis=1).max())
        if G["kind"][i] == 0:
            dh = float(np.angle(np.exp(1j * (xh[0] - eh[0]))))
            row = [ln / 50.0, np.cos(dh), np.sin(dh), np.clip(dh / max(ln, 1.0) * 10.0, -3, 3), 0.0, 0.25, 0.25]
        else:
            row = [ext / 50.0, 1.0, 0.0, 0.0, 1.0, approaches(en) / 4.0, approaches(ex) / 4.0]      # an intersection: its extent and approaches, not its connector count / length (they grow with the lanes)
        row += [ext / 20.0, deg_in[i] / 4.0, deg_out[i] / 4.0, near[i] / 10.0, float(rin[i] >= 0), rin[i] / 20.0 if rin[i] >= 0 else -0.05, 0.0, 0.0]
        l1h = np.zeros(N_LANE + 1)
        o1h = np.zeros(N_OPP)
        if G["kind"][i] == 0:
            l1h[int(lin[i])] = 1.0
            o1h[int(oin[i])] = 1.0
        F[i] = row + list(l1h) + list(o1h)
    return F


def relations(G):
    """(len(REL), N, N) bool from the INPUT graph only: succ a -> b, pred (b -> a), opposite, junction sibling (a common input successor or predecessor), near (the geometries within NEAR m, linked or not)"""
    N = len(G["kind"])
    Rm = np.zeros((len(REL), N, N), bool)
    for a, b in G["succ"]:
        Rm[0, a, b] = True
        Rm[1, b, a] = True
    for a, b in G["oppl"]:
        Rm[2, a, b] = True
    S = Rm[0].astype(int)
    Rm[3] = ((S @ S.T) > 0) | ((S.T @ S) > 0)
    np.fill_diagonal(Rm[3], False)
    pres = G.get("present", np.ones(N, bool))
    sub = [np.concatenate([p[:: max(1, len(p) // 10)] for p in gs]) if gs else np.zeros((1, 2)) for gs in G["geo"]]
    for i in range(N):
        for j in range(i + 1, N):
            if pres[i] and pres[j] and np.sqrt(((sub[i][:, None] - sub[j][None]) ** 2).sum(-1)).min() <= NEAR:
                Rm[4, i, j] = Rm[4, j, i] = True
    return Rm


def _dense(gs):
    """all points of a node's geometry with their arc position (0 - 1 along each polyline) and local heading"""
    pts, frac, hd = [], [], []
    for p in gs:
        if len(p) < 2:
            continue
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        t = np.gradient(p, axis=0)
        pts.append(p)
        frac.append(s / max(s[-1], 1e-9))
        hd.append(np.arctan2(t[:, 1], t[:, 0]))
    if not pts:
        return np.zeros((1, 2)), np.zeros(1), np.zeros(1)
    return np.concatenate(pts), np.concatenate(frac), np.concatenate(hd)


def pair_features(G):
    """(N, N, EDGE_F) float32 for the ordered pair (a, b), rigid-motion invariant (card v2: a successor may branch off the SIDE of a's carriageway at a split, so the link geometry is 'b starts on / next to a, or a
    ends on / next to b', not only exit -> entry): the attach point = the closest pair of (an entry of b, a point of a) or (an exit of a, a point of b), whichever is closer; in the frame at the attach point of a
    (a's heading there): the point of b / 20 m, b's position 10 m along its heading there / 20 m; b's heading relative to a's (cos, sin); the gap / 10 m (clipped at 5); gap <= GAP; the distance between the
    geometries / 30 m; the attach position along a (0 start - 1 end); the exit -> entry gap / 10 m (clipped at 5)"""
    N = len(G["kind"])
    E = [ends(G, i) for i in range(N)]
    D = [_dense(gs) for gs in G["geo"]]
    P = np.zeros((N, N, EDGE_F), np.float32)
    for a in range(N):
        pa, fa, ha = D[a]
        ex, xh = E[a][2], E[a][3]
        for b in range(N):
            if a == b:
                continue
            pb, fb, hb = D[b]
            en, eh = E[b][0], E[b][1]
            d1 = np.linalg.norm(en[:, None] - pa[None], axis=-1)                # entries of b against the points of a
            d2 = np.linalg.norm(ex[:, None] - pb[None], axis=-1)                # exits of a against the points of b
            i1, i2 = first_min(d1), first_min(d2)
            if d1[i1] <= d2[i2] + TIE_TOL:
                gap, oa, h_a, ob, h_b, fr = float(d1[i1]), pa[i1[1]], ha[i1[1]], en[i1[0]], eh[i1[0]], fa[i1[1]]
            else:
                gap, oa, h_a, ob, h_b, fr = float(d2[i2]), ex[i2[0]], xh[i2[0]], pb[i2[1]], hb[i2[1]], 1.0
            loc = _local(ob, oa, h_a)
            far = _local(ob + 10.0 * np.array([np.cos(h_b), np.sin(h_b)]), oa, h_a)
            dh = h_b - h_a
            xe = float(np.linalg.norm(ex[:, None] - en[None], axis=-1).min())
            dist = float(np.sqrt(((pa[:: max(1, len(pa) // 40), None] - pb[None, :: max(1, len(pb) // 40)]) ** 2).sum(-1)).min())
            P[a, b] = [*np.clip(loc / 20.0, -3, 3), *np.clip(far / 20.0, -3, 3), np.cos(dh), np.sin(dh), min(gap / GAP, 5.0), float(gap <= GAP), min(dist / NEAR, 5.0), fr, min(xe / GAP, 5.0)]
    return P


def ego_pose(G):
    """(N, 6): the ego-frame pose of every node (the mean entry xy, the mean exit xy, the first entry heading cos / sin): NOT an encoder input (step 2's positional input)"""
    out = np.zeros((len(G["kind"]), 6), np.float32)
    for i in range(len(G["kind"])):
        en, eh, ex, _ = ends(G, i)
        out[i] = [*en.mean(0), *ex.mean(0), np.cos(eh[0]), np.sin(eh[0])]
    return out


def prox_features(G):
    """D135 (P6): (M, N, PA_F) float32: proxy m against node n of the INPUT map: the distance to the node's nearest point / 10 m (clipped at 5), the cos / sin of the proxy direction against the node's local
    heading there, the proxy in the frame of that point (along, lateral) / 10 m (clipped), the attach position along the node, the kind of the node, 1 = present; rigid-motion invariant (a joint transform of map and
    proxies); absent nodes: zeros"""
    M, N = len(G["prox"]), len(G["kind"])
    out = np.zeros((M, N, PA_F), np.float32)
    if M == 0:
        return out
    pres = G.get("present", np.ones(N, bool))
    ph = np.arctan2(G["pdir"][:, 1], G["pdir"][:, 0])
    for n in range(N):
        if not pres[n]:
            continue
        pts, fr, hd = _dense(G["geo"][n])
        d = np.linalg.norm(G["prox"][:, None] - pts[None], axis=-1)
        j = (d <= d.min(1, keepdims=True) + TIE_TOL).argmax(1)                  # the first point within the tie tolerance (card v6)
        dh = ph - hd[j]
        rel = G["prox"] - pts[j]
        c, s = np.cos(-hd[j]), np.sin(-hd[j])
        along, lat = rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c
        out[:, n] = np.stack([np.minimum(d[np.arange(M), j] / 10.0, 5.0), np.cos(dh), np.sin(dh), np.clip(along / 10.0, -3, 3), np.clip(lat / 10.0, -3, 3), fr[j], np.full(M, float(G["kind"][n])), np.ones(M)], 1)
    return out


STATED_F = 13                   # card v7: the node-feature slot (a spare zero before v7) that carries the map's STATED count-error rate when featurize(stated_rate=True)


def featurize(Gin, stated_rate=False):
    P = pair_features(Gin)
    pres = Gin.get("present", np.ones(len(Gin["kind"]), bool))
    x = node_features(Gin)
    if stated_rate:                                                           # card v7: the map's stated count-error rate (one scalar per frame, the quality a map provider states), on every present road
        x[:, STATED_F] = np.where(Gin["kind"] == 0, Gin.get("count_rate", 0.0), 0.0)
    x[~pres] = 0.0
    P[~pres] = 0.0
    P[:, ~pres] = 0.0
    out = dict(x=x, rel=relations(Gin), pair=P, pose=ego_pose(Gin), present=pres)
    if "prox" in Gin:
        out["pa"] = prox_features(Gin)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ probe sets and rule nulls
def candidates(P):
    """P2: the ordered pairs (a, b), a != b, with the gap exit(a) -> entry(b) within GAP in the INPUT geometry -> (A, B)"""
    return np.nonzero(P[..., 7] > 0.5)


def null_p1(Gin):
    """(copy, neighbour vote) predictions of the lane class of every node: copy = the input class (unknown -> class 1 = 2 lanes); vote = the majority of the node's and its input successor / predecessor roads' known
    classes (ties: the node's own; all unknown: 2 lanes)"""
    lin = Gin["lanes_in"]
    copy = np.where(lin == N_LANE, 1, lin)
    nb = {i: [] for i in range(len(lin))}
    for a, b in Gin["succ"]:
        nb[a].append(b)
        nb[b].append(a)
    vote = copy.copy()
    for i in range(len(lin)):
        if Gin["kind"][i] != 0:
            continue
        vals = [lin[j] for j in [i] + nb[i] if Gin["kind"][j] == 0 and lin[j] < N_LANE]
        if vals:
            c = np.bincount(vals, minlength=N_LANE)
            top = np.flatnonzero(c == c.max())
            vote[i] = lin[i] if lin[i] in top else top[0]
    return copy, vote


def null_p2(P):
    """the geometric link rule: score of (a, b) = -gap - 2 m x |heading change| (rad)"""
    return -P[..., 6] * GAP - 2.0 * np.abs(np.arctan2(P[..., 5], P[..., 4]))


def p3_set(Gin):
    """P3 candidates: the nodes whose input route flag is -1 and that lie within NEAR of a node with an input route flag"""
    rin = Gin["route_in"]
    near = relations(Gin)[4]
    flagged = rin >= 0
    return np.flatnonzero((~flagged) & (near[:, flagged].any(1) if flagged.any() else np.zeros(len(rin), bool)))


def null_p3(Gin, P):
    """the link-to-route rule: a candidate's score = 1 when an INPUT succ / pred link joins it to a flagged route node, + the straightness of that link (cos of the heading change) / 10, else -gap to the nearest flagged
    node / 100"""
    rin = Gin["route_in"]
    flagged = set(np.flatnonzero(rin >= 0).tolist())
    score = {}
    for c in p3_set(Gin).tolist():
        best = None
        for a, b in Gin["succ"]:
            if (a in flagged and b == c) or (b in flagged and a == c):
                s = 1.0 + float(P[a, b, 4]) / 10.0
                best = s if best is None else max(best, s)
        if best is None:
            best = -min(float(P[f, c, 6]) for f in flagged) * GAP / 100.0 if flagged else -1.0
        score[c] = best
    return score


# ------------------------------------------------------------------------------------------------------------------------------ the cache
def cache_path(split, limit=None, cache=CACHE):
    return os.path.join(cache, f"{split}{'_n%d' % limit if limit else ''}.pkl")


def build(split, limit=None, cache=CACHE, log=print):
    """the exact graphs of the frames of the token caches (VAL = data/splits.json['val'] in order, TRAIN = splits['train']) -> runs/e390_sdgraph/<split>.pkl + .json manifest (never overwritten)"""
    import lane_gt_data as LD
    import lane_ptok_data as LP
    import lane_route_data as LR
    path = cache_path(split, limit, cache)
    assert not os.path.exists(path), f"a stored cache is never overwritten: {path}"
    lg, sdir = LD.lane_store()
    tdir = LP.cache_dir(split, 0, "e292_a")
    fr = json.load(open(os.path.join(tdir, "frames.json")))
    LR.rows_guard(split, fr["rows"])
    toks = fr["tokens"][: limit or None]
    scene = [int(s) for s in LR.LPD.scene_ids(LR.LPD.load_ext(tdir)["prev"])][: limit or None]
    graphs = []
    for k, t in enumerate(toks):
        graphs.append(frame_graph(lg.load(t)))
        if k % 500 == 0:
            log(f"  {k}/{len(toks)}", flush=True)
    os.makedirs(cache, exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(dict(tokens=toks, scene=scene, graphs=graphs), fh)
    os.replace(path + ".tmp", path)
    man = dict(version=VERSION, split=split, frames=len(toks), lane_gt=os.path.basename(sdir), tokens_sha=hashlib.sha256("\n".join(toks).encode()).hexdigest()[:16], max_nodes=MAX_NODES,
               scenes=len(set(scene)))
    json.dump(man, open(path[:-4] + ".json", "w"), indent=1)
    log(f"written {os.path.relpath(path, ROOT)}: {len(toks)} frames, {len(set(scene))} scenes")
    return path


def load(split, limit=None, cache=CACHE):
    path = cache_path(split, limit, cache)
    man = json.load(open(path[:-4] + ".json"))
    assert man["version"] == VERSION, f"{path}: version {man['version']}, the code {VERSION}: rebuild"
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    assert len(d["tokens"]) == man["frames"]
    return d


def stats(split, log=print):
    """nodes / links / forks per frame, the frames at the node cap, the share of true links inside the P2 gap, the lane classes, the route nodes"""
    d = load(split)
    n = np.array([len(G["kind"]) for G in d["graphs"]])
    links = inside = 0
    for G in d["graphs"]:
        P = pair_features(G)
        links += len(G["succ"])
        inside += sum(P[a, b, 7] > 0.5 for a, b in G["succ"])
    roads = np.concatenate([G["lanes"][G["kind"] == 0] for G in d["graphs"]])
    pp = np.array([G.get("prox_pieces") or (0, 0) for G in d["graphs"]]).sum(0)
    res = dict(prox_pieces_one_node=int(pp[0]), prox_pieces_all=int(pp[1]), proxies=int(sum(len(G["pseg"]) for G in d["graphs"])), proxies_on_intersections=int(sum((G["kind"][G["pseg"]] == 1).sum() for G in d["graphs"])),
               frames=len(n), scenes=len(set(d["scene"])), nodes_mean=round(float(n.mean()), 2), nodes_max=int(n.max()), at_cap=int((n >= MAX_NODES).sum()), links=links, links_inside_gap=round(inside / max(links, 1), 4),
               route_nodes=int(sum((G["route"] >= 0).sum() for G in d["graphs"])), lane_classes=np.bincount(roads, minlength=N_LANE).tolist())
    log(json.dumps(res))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ selftest (a hand graph)
def toy():
    """road 0 (2 lanes) -> intersection 1 (two connectors) -> road 2 (2 lanes, straight on) and road 3 (1 lane, turning left); road 4 = the opposite of 0 (1 lane); the route 0 -> 1 -> 3"""
    s = np.linspace(0, 40, 41)
    r0 = np.stack([np.zeros_like(s), s - 45.0], 1)
    c1 = np.stack([np.zeros(9), np.linspace(-5, 3, 9)], 1)
    c2 = np.stack([-np.sin(np.linspace(0, np.pi / 2, 9)) * 5.0, -5 + 5 * (1 - np.cos(np.linspace(0, np.pi / 2, 9))) + 0.0], 1)
    r2 = np.stack([np.zeros_like(s), s + 4.0], 1)
    r3 = np.stack([-(s + 6.0), np.zeros_like(s)], 1)
    r4 = np.stack([np.full_like(s, 0.2), -s - 6.0], 1)
    geo = [[r0], [c1, c2], [r2], [r3], [r4]]
    pp, pd, ps = [], [], []
    for n, gs in enumerate(geo):                                              # proxies: every 4th point of every polyline, 1 m to the side (a lane centre next to the map line)
        for p in gs:
            t = np.gradient(p, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            q = p + np.stack([t[:, 1], -t[:, 0]], 1)
            pp.append(q[2::4]), pd.append(t[2::4]), ps.append(np.full(len(q[2::4]), n))
    return dict(kind=np.array([0, 1, 0, 0, 0]), geo=geo, lanes=np.array([1, -1, 1, 0, 0]), route=np.array([0, 1, -1, 2, -1]),
                succ=[(0, 1), (1, 2), (1, 3)], oppl=[(0, 4), (4, 0)], prox=np.concatenate(pp), pdir=np.concatenate(pd), pseg=np.concatenate(ps).astype(int))


def selftest():
    G = toy()
    Gin, T = perturb(G, np.random.default_rng(0))
    assert (Gin["lanes_in"] == G["lanes"]).all() and Gin["succ"] == G["succ"] and T["dropped"] == [] and (Gin["route_in"] == G["route"]).all()
    F = featurize(Gin)
    assert F["x"].shape == (5, NODE_F) and F["rel"].shape == (len(REL), 5, 5) and F["pair"].shape == (5, 5, EDGE_F)
    assert F["rel"][0, 0, 1] and F["rel"][1, 1, 0] and F["rel"][2, 0, 4] and F["rel"][3, 2, 3] and not F["rel"][3, 0, 2], "succ / pred / opposite / sibling"
    a, b = candidates(F["pair"])
    assert {(0, 1), (1, 2), (1, 3)} <= set(zip(a.tolist(), b.tolist())), "the true links lie inside the gap"
    assert opp_in(Gin)[0] == 1 and opp_in(Gin)[4] == 2 and opp_in(Gin)[2] == 0, "the opposite road's input class (1 lane -> 1, 2 lanes -> 2, none -> 0)"
    Gr, _ = perturb(G, np.random.default_rng(0), rigid=(7.0, -3.0, 0.6))
    Fr = featurize(Gr)
    assert np.allclose(Fr["x"], F["x"], atol=1e-5) and np.allclose(Fr["pair"], F["pair"], atol=1e-5) and (Fr["rel"] == F["rel"]).all(), "the features are not rigid-motion invariant"
    assert not np.allclose(Fr["pose"], F["pose"]), "the ego pose must follow the transform"
    Gc, _ = perturb(G, np.random.default_rng(1), count_rate=1.0)
    assert (Gc["lanes_in"][G["kind"] == 0] != G["lanes"][G["kind"] == 0]).all()
    Gd, Td = perturb(G, np.random.default_rng(1), drop=1.0, hide=1.0)
    assert Gd["succ"] == [] and Td["dropped"] == G["succ"] and (Gd["route_in"] == -1).all() and sorted(Td["hidden"].tolist()) == [0, 1, 3]
    Rd = relations(Gd)
    assert not Rd[0].any() and not Rd[1].any() and not Rd[3].any() and Rd[4][0, 1], "a dropped link leaves no succ / pred / sibling relation; the pair stays 'near'"
    Fd = featurize(Gd)
    assert (Fd["x"][:, 8:10] == 0).all(), "the degrees come from the input links"
    # the smooth deformation keeps a joint joined: two coincident points move together
    f = smooth_field(np.random.default_rng(3), 2.0)
    q = f(np.array([[1.0, 2.0], [1.0, 2.0], [30.0, -4.0]]))
    assert np.allclose(q[0], q[1]) and np.linalg.norm(q[2] - [30.0, -4.0]) > 0
    assert route_ranks(np.array([-1, 4, 0, 9, -1])).tolist() == [-1, 1, 0, 2, -1]
    assert approaches(np.array([[0.0, 0.0], [3.5, 0.0], [20.0, 0.0]])) == 2
    assert len(rdp(np.stack([s := np.linspace(0, 10, 11), 0.01 * s], 1), 0.5)) == 2
    print("lane_sdgraph_data selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "stats", "selftest"))
    ap.add_argument("--split", choices=("val", "train"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "build":
        return build(a.split, a.limit)
    return stats(a.split)


if __name__ == "__main__":
    main()
