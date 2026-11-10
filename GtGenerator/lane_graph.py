"""The lane graph of a keyframe: pieces with ids, successors, events (split, merge, stop, start, end, window edge), cut to a square window.

Source: the nuScenes map (`lane` and `lane_connector` with their exact arc paths and the connectivity), nothing else: no image, no radar, no box.
The map is turned into a graph once per location (`build_map_graph`), a keyframe takes a crop of it (`crop`).

PIECES. A map piece is a lane or a lane connector. A chain of map pieces is joined into one piece while every join is 1 in, 1 out (the previous piece
has one successor and the next one has one predecessor): a piece id is therefore kept along a road and changes only at a split or a merge (Mirko's
id rule). The id of a piece is the token of its first map piece; `tokens` lists the map pieces it covers and `spans` where (point indices). The points of a piece are the points of its map pieces
without the repeats of the map (the arc paths repeat the joint of two arcs: no two consecutive points are closer than 1e-3 m).

EVENTS. Where pieces meet or end, in the window:
  split    one piece ends, two or more start            (in = [1 piece], out = [2+ pieces]); `pos` is the end of the one piece (not between the ends of a gap in the map)
  merge    two or more end, one starts                  `pos` is the start of the one piece
  cross    two or more end, two or more start           (a node that is both; `pos` the mean of the ends and starts)
  stop     the end of a LANE whose successor connector lies inside an intersection (road_segment.is_intersection, at least `inter_frac` of the
           connector's points inside); it can lie inside a joined piece (`on` = the piece, `s` = metres from the start of the piece part);
           `out` are the stop branches, `n_succ` the number of successors of the lane in the map (n_succ > len(out): a branch that is not a stop)
  exit     the mirror of stop: the START of a LANE whose predecessor connector lies inside an intersection (same fields: `on`, `s`; `n_pred` = predecessors of the lane
           in the map, `in` = the intersection branches that lead into it, `out` = the piece it starts); `stop_rule` decides here the same way
  start    a piece without a predecessor in the map;  end   a piece without a successor in the map
A stop or an exit can lie inside a joined piece (a lane followed by one connector): then `out` of the stop (`in` of the exit) is that piece itself, the continuation after `s`.
`dirs_in` / `dirs_out` have one row per id of `in` / `out`, in the same order.
  window   the polyline crosses the window border (io "in" / "out"); per frame
`stop_rule` decides a lane with several successors: "intersection_branch" (default: a stop point for the intersection branches only), "any_successor"
(if one is an intersection connector, all branches), "all_successors", "none". `dirs_in` / `dirs_out` are the unit directions of the pieces at the event (UNIT vectors in every frame, also when the pose
has pitch or roll). In a window an event keeps only the ids of the pieces that are inside (`partial`); a split / merge / cross whose point lies outside the window (a gap of the map, at most MAP_GAP m) while a
branch of it is inside keeps its event, at the branches that are inside; a stop or an exit with nothing on one side is not an event, so every stop / exit names its in and its out, and every natural end
(no successor, not cut by the window) of a part is a successor, an event that names the part at that end (split, merge, cross, end), or the window.

FRAME. frame="lane" (default): the lane project's frame, ego (rear axle) at the front-camera time, x to the RIGHT, y forward; frame="ego": the frame of
the other GtGenerator layers, x forward, y left. The window is the square |x|, |y| <= half (default 50 m), or the rectangle `window` = (x0, x1, y0, y1) of a rig
(lane_gt.RIGS: BEV fusion x 0 .. 60 m, y -26 .. 26 m, which then carries the key `window`). Polylines are cut exactly at the border; a
part shorter than `min_len` is dropped, but only when the window cuts it at both ends (a short part with a natural end keeps its links and its events); a piece that leaves the window and comes back is several parts (`part` 0, 1, ...; id "<piece>#<part>").
Every part has `segments`: [{token, kind, i0, i1, in_intersection}], the index ranges of its `pts` per map piece, contiguous (a connector inside an intersection can be dropped by a
consumer that has no nodes there, a connector outside one is a plain continuation, also on a piece of kind "mixed"). A point of a part stands for a vertex of its map piece (`crop` gives it as `src`):
a cut point stands for the vertex at the END of the segment it lies on, so a part that starts right after the joint of two map pieces has no segment of the first one.
`piece` is the id of the piece and is the same in every frame; `id` equals it except for a part that comes back into the window ("<piece>#1": its number depends on the
window, so use `piece` for anything that must be stable from frame to frame).
A part's `succ` / `pred` lists the ids of the parts that continue it inside the window (a part that ends at the border has none: its window event says so).

The result is plain data (dicts, lists, float32 arrays). `graph` -> {"frame", "half", "location", "pieces": [...], "events": [...], ...}.
Only vectors and events are stored; rasters are drawn from them where they are needed.
"""
import math

import numpy as np

STOP_RULES = ("intersection_branch", "any_successor", "all_successors", "none")
MAP_GAP = 10.0                                              # m: how far outside the window a junction can lie and still keep an event for the branches that are inside (the largest gap of the four maps is 6.3 m)
REACH_MARGIN = 25.0                                        # m: slack of the pre-filter `reach` (the filter is exact without it: a piece outside the circle cannot reach the window; 1 m instead of 25 m saved 2 - 4 % of the time, below the 10 % that a change of this kind must earn)


# ---------------------------------------------------------------------------------------------------------------- small geometry
def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _tangent(P, at_end, base=1.0, step=0.5):
    """unit direction of a polyline at its end (or start), over about `base` metres"""
    k = max(1, int(round(base / step)))
    d = (P[-1] - P[max(-1 - k, -len(P))]) if at_end else (P[min(k, len(P) - 1)] - P[0])
    return _unit(d)


def _length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def _distinct(P, eps=1e-3):
    """the polyline without the points that repeat the previous one: the arc paths of the map repeat the joint of two arcs (a zero-length segment, 31 % of the pieces; the repeats are 0 or
    below 1e-13 m on the four maps, the other segments are 0.2 m or longer)"""
    return P[np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) >= eps]]


def to_frame(xy, T_e2g, frame):
    """global points (n, 2) -> the frame of the keyframe ("ego": x forward, y left; "lane": x right, y forward)"""
    e = (np.asarray(xy, float) - T_e2g[:2, 3]) @ T_e2g[:2, :2]
    return e if frame == "ego" else np.c_[-e[:, 1], e[:, 0]]


def dirs_to_frame(d, T_e2g, frame):
    """global direction vectors (n, 2) -> UNIT vectors in the frame (no translation). The 2 x 2 block of a pose with pitch or roll is not a rotation (it shortens a vector by cos(pitch)); the
    rows are normalised, which is the direction of the polyline that `to_frame` makes of the same points"""
    e = np.asarray(d, float).reshape(-1, 2) @ T_e2g[:2, :2]
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    return e if frame == "ego" else np.c_[-e[:, 1], e[:, 0]]


def clip_rect(P, x0, x1, y0, y1):
    """cut a polyline (n, 2) to the rectangle x0 <= x <= x1, y0 <= y <= y1 exactly at the border (the square |x|, |y| <= h is the case (-h, h, -h, h)) ->
    [(pts, cut_start, cut_end, seg_first, seg_last)]: the parts in travel order; seg_first / seg_last are the indices of the first / last original segment the part uses"""
    if len(P) < 2:
        return []
    a, b = P[:-1], P[1:]
    d = b - a
    t0, t1 = np.zeros(len(d)), np.ones(len(d))
    ok = np.ones(len(d), bool)
    for ax, (blo, bhi) in enumerate(((x0, x1), (y0, y1))):
        da, pa = d[:, ax], a[:, ax]
        zero = np.abs(da) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            tl, th = (blo - pa) / da, (bhi - pa) / da
        lo, hi = np.where(da > 0, tl, th), np.where(da > 0, th, tl)
        lo, hi = np.where(zero, 0.0, lo), np.where(zero, 1.0, hi)
        ok &= ~(zero & ((pa < blo) | (pa > bhi)))
        t0, t1 = np.maximum(t0, lo), np.minimum(t1, hi)
    ok &= t0 < t1 - 1e-9
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    entry, exit_ = a[idx] + t0[idx][:, None] * d[idx], a[idx] + t1[idx][:, None] * d[idx]
    joined = (idx[1:] == idx[:-1] + 1) & (t1[idx[:-1]] >= 1.0 - 1e-9) & (t0[idx[1:]] <= 1e-9)           # a segment goes on from the one before when that ends on its vertex and this one starts there
    starts = np.flatnonzero(np.r_[True, ~joined])
    out = []
    for s, e in zip(starts, np.r_[starts[1:], len(idx)]):
        first, last = int(idx[s]), int(idx[e - 1])
        # a part is cut where the polyline goes on outside the window, also when a vertex lies exactly on the border (t0 = 0 / t1 = 1)
        out.append((np.vstack([entry[s:s + 1], exit_[s:e]]), t0[idx[s]] > 1e-9 or first > 0, t1[idx[e - 1]] < 1.0 - 1e-9 or last < len(d) - 1, first, last))
    return out


def window_of(half, window=None):
    """the window (x0, x1, y0, y1) in the frame of the graph: `window` when given, else the square |x|, |y| <= half"""
    return (-half, half, -half, half) if window is None else tuple(float(v) for v in window)


def reach(half, window=None):
    """the radius around the ego position that holds the window plus a margin: the pre-filter for the map pieces that can reach it (the frame is a rotation of the global frame)"""
    if window is None:
        return half * math.sqrt(2.0) + REACH_MARGIN
    x0, x1, y0, y1 = window
    return math.hypot(max(abs(x0), abs(x1)), max(abs(y0), abs(y1))) + REACH_MARGIN


def project(P, p):
    """(s, dist): the arc length of the polyline P at the foot of point p, and its distance"""
    a, b = P[:-1], P[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
    foot = a + t[:, None] * ab
    dist = np.linalg.norm(foot - p, axis=1)
    i = int(np.argmin(dist))
    seg = np.sqrt(L2)
    return float(seg[:i].sum() + t[i] * seg[i]), float(dist[i])


# ---------------------------------------------------------------------------------------------------------------- the map graph
def build_map_graph(nmap, step=0.5, inter_frac=0.5, stop_rule="intersection_branch"):
    """the graph of a nuScenes map (global coordinates): joined pieces, junction / start / end / stop events"""
    import shapely
    if stop_rule not in STOP_RULES:
        raise ValueError(f"stop_rule must be one of {STOP_RULES}, got {stop_rule!r}")
    lanes = [r["token"] for r in nmap.lane]
    conns = [r["token"] for r in nmap.lane_connector]
    disc = nmap.discretize_lanes(lanes + conns, step)
    polys = {t: np.asarray(v, float)[:, :2] for t, v in disc.items() if len(v) >= 2}
    kind = {t: "lane" for t in lanes}
    kind.update({t: "connector" for t in conns})
    succ = {t: [] for t in polys}
    for t in sorted(polys):
        c = nmap.connectivity.get(t, {})
        for v in c.get("outgoing", []):
            if v in polys and v not in succ[t]:
                succ[t].append(v)
        for u in c.get("incoming", []):
            if u in polys and t not in succ[u]:
                succ[u].append(t)
    pred = {t: [] for t in polys}
    for u, vs in succ.items():
        for v in vs:
            pred[v].append(u)
    # connectors inside an intersection
    segs = [nmap.extract_polygon(r["polygon_token"]) for r in nmap.road_segment if r["is_intersection"]]
    segs = [g for g in segs if not g.is_empty]
    inter = shapely.union_all(segs) if segs else None
    in_inter = {}
    if inter is not None:
        shapely.prepare(inter)
        for t in conns:
            if t in polys:
                P = polys[t]
                in_inter[t] = bool(shapely.contains_xy(inter, P[:, 0], P[:, 1]).mean() >= inter_frac)
    # joined pieces: follow the chain while the join is 1 in, 1 out
    nxt = {t: (succ[t][0] if len(succ[t]) == 1 and len(pred[succ[t][0]]) == 1 else None) for t in polys}
    has_prev = {v for v in nxt.values() if v is not None}
    chains, seen = [], set()

    def follow(t):
        chain = [t]
        seen.add(t)
        while nxt[chain[-1]] is not None and nxt[chain[-1]] not in seen:
            chain.append(nxt[chain[-1]])
            seen.add(chain[-1])
        return chain
    for t in sorted(polys):
        if t not in has_prev:
            chains.append(follow(t))
    for t in sorted(polys):                       # a closed loop of 1 in, 1 out joins has no start
        if t not in seen:
            chains.append(follow(t))
    piece_of, pieces = {}, {}
    for chain in chains:
        pts, spans, n = [], [], 0
        for tok in chain:
            P = _distinct(polys[tok])               # the points of a piece; `polys` stays as the map has it (the road graph and the intersection test were made on it: 5 connectors and 4 neighbour pairs would move)
            if pts and np.linalg.norm(P[0] - pts[-1][-1]) < 0.05:
                P = P[1:]                         # the joint point once
            spans.append((tok, n, n + len(P) - 1))
            pts.append(P)
            n += len(P)
        P = np.concatenate(pts)
        kinds = {kind[t] for t in chain}
        pid = chain[0]
        pieces[pid] = dict(id=pid, tokens=chain, kind=kinds.pop() if len(kinds) == 1 else "mixed", pts=P, spans=spans, starts=np.array([s[1] for s in spans]),
                           bbox=np.r_[P.min(0), P.max(0)])
        for tok in chain:
            piece_of[tok] = pid
    for pid, p in pieces.items():
        p["succ"] = [piece_of[v] for v in succ[p["tokens"][-1]]]
        p["pred"] = [piece_of[u] for u in pred[p["tokens"][0]]]
    events = _map_events(pieces, piece_of, succ, pred, kind, in_inter, stop_rule, step)
    n = 8                                                      # ids in the output are the first n characters of the token (the shortest unique prefix, at least 8)
    while len({pid[:n] for pid in pieces}) < len(pieces):
        n += 2
    short = {pid: pid[:n] for pid in pieces}
    return dict(short=short, pieces=pieces, piece_of=piece_of, succ=succ, pred=pred, kind=kind, in_inter=in_inter, events=events, event_pos=np.array([e["pos"] for e in events]).reshape(-1, 2),
                polys=polys, inter=inter, params=dict(inter_frac=inter_frac, stop_rule=stop_rule))


def _map_events(pieces, piece_of, succ, pred, kind, in_inter, stop_rule, step):
    ev = []
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for pid, p in pieces.items():
        for q in p["succ"]:
            parent[find(("end", pid))] = find(("start", q))
    comps = {}
    for node in list(parent):
        comps.setdefault(find(node), []).append(node)
    for nodes in comps.values():
        ins = sorted({n[1] for n in nodes if n[0] == "end"})
        outs = sorted({n[1] for n in nodes if n[0] == "start"})
        if len(ins) == 1 and len(outs) >= 2:
            typ = "split"
        elif len(ins) >= 2 and len(outs) == 1:
            typ = "merge"
        elif len(ins) >= 2 and len(outs) >= 2:
            typ = "cross"
        else:
            continue
        if typ == "split":                                      # at the branch end, not between the ends of a gap in the map: where the one piece ends / where the one piece starts
            pos = pieces[ins[0]]["pts"][-1]
        elif typ == "merge":
            pos = pieces[outs[0]]["pts"][0]
        else:
            pos = np.mean([pieces[i]["pts"][-1] for i in ins] + [pieces[o]["pts"][0] for o in outs], axis=0)
        ev.append(dict(type=typ, pos=pos, **{"in": ins, "out": outs},
                       dirs_in=np.array([_tangent(pieces[i]["pts"], True, step=step) for i in ins]),
                       dirs_out=np.array([_tangent(pieces[o]["pts"], False, step=step) for o in outs])))
    for pid, p in sorted(pieces.items()):
        if not p["pred"]:
            ev.append(dict(type="start", pos=p["pts"][0], **{"in": [], "out": [pid]}, dirs_in=np.zeros((0, 2)),
                           dirs_out=np.array([_tangent(p["pts"], False, step=step)])))
        if not p["succ"]:
            ev.append(dict(type="end", pos=p["pts"][-1], **{"in": [pid], "out": []}, dirs_in=np.array([_tangent(p["pts"], True, step=step)]),
                           dirs_out=np.zeros((0, 2))))
    if stop_rule != "none":
        def branches(tokens, at_end):
            """[(piece id, unit direction)] sorted by piece id: the map pieces `tokens` with their direction at the end (at_end) or at the start"""
            out = []
            for c in tokens:
                pc = piece_of[c]
                sp = _span(pieces, pc, c)
                P = pieces[pc]["pts"]
                out.append((pc, _tangent(P[:sp[2] + 1], True, step=step) if at_end else _tangent(P[sp[1]:], False, step=step)))
            return sorted(out, key=lambda b: b[0])

        def chosen(neigh, tok):
            """the successors (or predecessors) of a lane that count for a stop (an exit) under the rule"""
            into = [c for c in neigh[tok] if kind[c] == "connector" and in_inter.get(c)]
            if stop_rule == "intersection_branch":
                return into
            if stop_rule == "any_successor":
                return neigh[tok] if into else []
            return neigh[tok] if len(into) == len(neigh[tok]) else []
        for tok in sorted(succ):
            if kind[tok] != "lane" or not succ[tok] or not chosen(succ, tok):
                continue
            pid = piece_of[tok]
            sp = _span(pieces, pid, tok)
            P = pieces[pid]["pts"]
            br = branches(chosen(succ, tok), False)
            ev.append(dict(type="stop", pos=P[sp[2]], on=pid, idx=sp[2], n_succ=len(succ[tok]), **{"in": [pid], "out": [b[0] for b in br]},
                           dirs_in=np.array([_tangent(P[:sp[2] + 1], True, step=step)]), dirs_out=np.array([b[1] for b in br])))
        for tok in sorted(pred):                                 # the mirror: the start of a lane that an intersection connector leads into
            if kind[tok] != "lane" or not pred[tok] or not chosen(pred, tok):
                continue
            pid = piece_of[tok]
            sp = _span(pieces, pid, tok)
            P = pieces[pid]["pts"]
            br = branches(chosen(pred, tok), True)
            j = sp[1] - 1 if sp[1] > 0 else 0                    # the joint vertex belongs to the map piece before (the duplicate is dropped from the lane)
            ev.append(dict(type="exit", pos=P[j], on=pid, idx=j, n_pred=len(pred[tok]), **{"in": [b[0] for b in br], "out": [pid]},
                           dirs_in=np.array([b[1] for b in br]), dirs_out=np.array([_tangent(P[j:], False, step=step)])))
    return ev


def _span(pieces, pid, tok):
    """(token, first index, last index) of a map piece inside its joined piece"""
    return next(s for s in pieces[pid]["spans"] if s[0] == tok)


# ---------------------------------------------------------------------------------------------------------------- a keyframe's crop
def _segments(p, src, mg):
    """the index ranges of a part's points per map piece: [{token, kind, i0, i1, in_intersection}], contiguous, one range per map piece the points stand for (`src`: the vertex of the joined
    piece `p` each point stands for, non-decreasing; the spans of `p` partition its vertices)"""
    owner = np.searchsorted(p["starts"], src, side="right") - 1
    cuts = np.flatnonzero(np.diff(owner)) + 1
    out = []
    for i0, i1 in zip(np.r_[0, cuts], np.r_[cuts, len(src)] - 1):
        tok = p["spans"][owner[i0]][0]
        out.append(dict(token=tok, kind=mg["kind"][tok], i0=int(i0), i1=int(i1), in_intersection=bool(mg["in_inter"].get(tok, False))))
    return out


def crop(mg, T_e2g, half=50.0, min_len=2.0, frame="lane", rg=None, window=None, full=True):
    """the graph of one keyframe: the map graph `mg` cut to the window around the ego pose T_e2g (4 x 4, ego -> global); with `rg` =
    road_graph.build_road_graph(mg) the neighbours, lane counts and roads are added (see road_graph.py).
    The window is the square |x|, |y| <= half, or the rectangle `window` = (x0, x1, y0, y1) in the frame of the graph (then `half` is only the bound that is stored).
    `full=False` leaves out the events and the segments (the pieces and their links only: the wide window of the tiers needs no more).
    Every part also carries `src` (n,): the vertex of its map piece that each point stands for (what the labels of lane_tiers are carried over by); `lane_gt.LaneGT.graph` drops it"""
    t = T_e2g[:2, 3]
    pieces, short = mg["pieces"], mg["short"]
    W = window_of(half, window)
    r = reach(half, window)
    keep = [pid for pid, p in pieces.items()
            if p["bbox"][2] > t[0] - r and p["bbox"][0] < t[0] + r and p["bbox"][3] > t[1] - r and p["bbox"][1] < t[1] + r]
    out_parts, parts_of = [], {}
    for pid in sorted(keep):
        p = pieces[pid]
        Q = to_frame(p["pts"], T_e2g, frame)
        k = 0
        for pts, cut0, cut1, first, last in clip_rect(Q, *W):
            length = _length(pts)
            if length < (min_len if (cut0 and cut1) else 0.05):                   # a fragment cut at both ends is dropped when short; a natural end keeps its links
                continue
            src = np.arange(first, last + 2)                                      # point j stands for the original vertex src[j] of the piece; the cut points stand for the vertex at the END of the segment they lie on
            if cut0 and not np.array_equal(pts[0], Q[first]):                     # (an entry cut lies on the segment first -> first + 1; the exit cut is already on last -> last + 1)
                src[0] += 1
            part = dict(id=short[pid] if k == 0 else f"{short[pid]}#{k}", piece=pid, part=k, kind=p["kind"], pts=pts.astype(np.float32), start_open=bool(cut0),
                        end_open=bool(cut1), length=length, succ=[], pred=[], src=src)
            if full:
                part["segments"] = _segments(p, src, mg)
            out_parts.append(part)
            parts_of.setdefault(pid, []).append(part)
            k += 1
    head = {pid: next((q for q in ps if not q["start_open"]), None) for pid, ps in parts_of.items()}
    tail = {pid: next((q for q in ps if not q["end_open"]), None) for pid, ps in parts_of.items()}
    for part in out_parts:
        if not part["end_open"]:
            for q in pieces[part["piece"]]["succ"]:
                h = head.get(q)
                if h is not None:
                    part["succ"].append(h["id"])
        if not part["start_open"]:
            for q in pieces[part["piece"]]["pred"]:
                tl = tail.get(q)
                if tl is not None:
                    part["pred"].append(tl["id"])
    for part in out_parts:
        part["succ"], part["pred"] = sorted(part["succ"]), sorted(part["pred"])
    events = []

    def pick(ids, dirs, table, on=None, q=None):
        """the ids that exist in this window (the part of the piece through `table`; the part `q` for the piece `on` that carries the event) with their direction rows"""
        keep = [k for k, i in enumerate(ids) if (q is not None and i == on) or table.get(i) is not None]
        return [q["id"] if (q is not None and ids[k] == on) else table[ids[k]]["id"] for k in keep], np.asarray(dirs).reshape(-1, 2)[keep]
    epos = to_frame(mg["event_pos"], T_e2g, frame) if full else np.zeros((0, 2))
    inside = (W[0] - 1e-6 <= epos[:, 0]) & (epos[:, 0] <= W[1] + 1e-6) & (W[2] - 1e-6 <= epos[:, 1]) & (epos[:, 1] <= W[3] + 1e-6)
    near_out = (W[0] - MAP_GAP <= epos[:, 0]) & (epos[:, 0] <= W[1] + MAP_GAP) & (W[2] - MAP_GAP <= epos[:, 1]) & (epos[:, 1] <= W[3] + MAP_GAP)          # a gap of the map can put a junction outside while a branch is inside
    for i in np.flatnonzero(near_out):
        e, pos = mg["events"][i], epos[i]
        if not inside[i] and (e["type"] not in ("split", "merge", "cross") or not (any(tail.get(j) for j in e["in"]) or any(head.get(j) for j in e["out"]))):
            continue
        extra, q = {}, None
        if e["type"] in ("stop", "exit"):                                   # they lie on a piece, possibly inside it: the part that carries the point
            ps = parts_of.get(e["on"], [])
            best = min(((project(q_["pts"].astype(float), pos), q_) for q_ in ps), key=lambda x: x[0][1], default=None)
            if best is None or best[0][1] > 0.6:
                continue
            (s, _), q = best
            extra = dict(on=q["id"], s=s, **({"n_succ": e["n_succ"]} if e["type"] == "stop" else {"n_pred": e["n_pred"]}))
        if e["type"] == "stop":
            ids_in, d_in, (ids_out, d_out) = [q["id"]], np.asarray(e["dirs_in"]).reshape(-1, 2), pick(e["out"], e["dirs_out"], head, e["on"], q)
        elif e["type"] == "exit":
            (ids_in, d_in), ids_out, d_out = pick(e["in"], e["dirs_in"], tail, e["on"], q), [q["id"]], np.asarray(e["dirs_out"]).reshape(-1, 2)
        else:
            (ids_in, d_in), (ids_out, d_out) = pick(e["in"], e["dirs_in"], tail), pick(e["out"], e["dirs_out"], head)
            if not inside[i]:                                                # its point is outside the window, a branch of it is inside: the event is at the branches that are
                pos = np.mean([tail[j]["pts"][-1] for j in e["in"] if tail.get(j)] + [head[j]["pts"][0] for j in e["out"] if head.get(j)], axis=0)
        if len(ids_in) + len(ids_out) == 0 or (e["type"] in ("stop", "exit") and not (ids_in and ids_out)):          # a stop / exit always says where it leads from / to
            continue
        events.append(dict(type=e["type"], pos=pos.astype(np.float32), **extra, **{"in": ids_in, "out": ids_out},
                           dirs_in=dirs_to_frame(d_in, T_e2g, frame).astype(np.float32), dirs_out=dirs_to_frame(d_out, T_e2g, frame).astype(np.float32),
                           partial=bool(len(ids_in) != len(e["in"]) or len(ids_out) != len(e["out"]))))
    for part in out_parts if full else ():                   # the window border
        if part["start_open"]:
            events.append(dict(type="window", io="in", pos=part["pts"][0], **{"in": [], "out": [part["id"]]}, dirs_in=np.zeros((0, 2), np.float32),
                               dirs_out=_tangent(part["pts"], False)[None].astype(np.float32), partial=False))
        if part["end_open"]:
            events.append(dict(type="window", io="out", pos=part["pts"][-1], **{"in": [part["id"]], "out": []},
                               dirs_in=_tangent(part["pts"], True)[None].astype(np.float32), dirs_out=np.zeros((0, 2), np.float32), partial=False))
    graph = dict(frame=frame, half=half, pieces=out_parts, events=events, params=dict(min_len=min_len, **mg["params"]))
    if window is not None:
        graph["window"] = W                                  # only a graph with a rectangular window carries the key: the square graphs (lane_center) are unchanged
    if rg is not None:
        from . import road_graph
        road_graph.attach(graph, rg)
    for part in out_parts:
        part["piece"] = short[part["piece"]]
    return graph


def in_frame(graph, frame):
    """the same graph in another frame ("lane": x right, y forward; "ego": x forward, y left): every point and direction rotated, nothing else"""
    if graph["frame"] == frame:
        return graph

    def turn(a):
        a = np.asarray(a)
        return np.stack([a[..., 1], -a[..., 0]], -1) if frame == "ego" else np.stack([-a[..., 1], a[..., 0]], -1)       # lane -> ego, ego -> lane

    def walk(x):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k in ("pts", "pos", "dirs_in", "dirs_out", "dir_in", "dir_out", "exterior") and len(np.asarray(v)):
                    out[k] = turn(v)
                elif k == "holes":
                    out[k] = [turn(h) for h in v]
                elif k == "window" and v is not None and len(v) == 4:                 # the rectangle of a rig (x0, x1, y0, y1): a rotation of the frame turns it too
                    x0, x1, y0, y1 = v
                    out[k] = (y0, y1, -x1, -x0) if frame == "ego" else (-y1, -y0, x0, x1)       # lane -> ego, ego -> lane
                else:
                    out[k] = walk(v) if not isinstance(v, np.ndarray) else v
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    out = walk(graph)
    out["frame"] = frame
    return out
