"""The lane graph of a keyframe: pieces with ids, successors, events (split, merge, stop, start, end, window edge), cut to a square window.

Source: the nuScenes map (`lane` and `lane_connector` with their exact arc paths and the connectivity), nothing else: no image, no radar, no box.
The map is turned into a graph once per location (`build_map_graph`), a keyframe takes a crop of it (`crop`).

PIECES. A map piece is a lane or a lane connector. A chain of map pieces is joined into one piece while every join is 1 in, 1 out (the previous piece
has one successor and the next one has one predecessor): a piece id is therefore kept along a road and changes only at a split or a merge (Mirko's
id rule). The id of a piece is the token of its first map piece; `tokens` lists the map pieces it covers and `spans` where (point indices).

EVENTS. Where pieces meet or end, in the window:
  split    one piece ends, two or more start            (in = [1 piece], out = [2+ pieces])
  merge    two or more end, one starts
  cross    two or more end, two or more start           (a node that is both)
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
(if one is an intersection connector, all branches), "all_successors", "none". `dirs_in` / `dirs_out` are the unit directions of the pieces at the event.

FRAME. frame="lane" (default): the lane project's frame, ego (rear axle) at the front-camera time, x to the RIGHT, y forward; frame="ego": the frame of
the other autogt layers, x forward, y left. The window is the square |x|, |y| <= half (default 50 m). Polylines are cut exactly at the border; a
part shorter than `min_len` is dropped, but only when the window cuts it at both ends (a short part with a natural end keeps its links and its events); a piece that leaves the window and comes back is several parts (`part` 0, 1, ...; id "<piece>#<part>").
Every part has `segments`: [{token, kind, i0, i1, in_intersection}], the index ranges of its `pts` per map piece (a connector inside an intersection can be dropped by a
consumer that has no nodes there, a connector outside one is a plain continuation, also on a piece of kind "mixed").
`piece` is the id of the piece and is the same in every frame; `id` equals it except for a part that comes back into the window ("<piece>#1": its number depends on the
window, so use `piece` for anything that must be stable from frame to frame).
A part's `succ` / `pred` lists the ids of the parts that continue it inside the window (a part that ends at the border has none: its window event says so).

The result is plain data (dicts, lists, float32 arrays). `graph` -> {"frame", "half", "location", "pieces": [...], "events": [...], ...}.
Only vectors and events are stored; rasters are drawn from them where they are needed.
"""
import math

import numpy as np

DEFAULTS = dict(step=0.5, half=50.0, min_len=2.0, inter_frac=0.5, stop_rule="intersection_branch", frame="lane")
STOP_RULES = ("intersection_branch", "any_successor", "all_successors", "none")


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


def to_frame(xy, T_e2g, frame):
    """global points (n, 2) -> the frame of the keyframe ("ego": x forward, y left; "lane": x right, y forward)"""
    e = (np.asarray(xy, float) - T_e2g[:2, 3]) @ T_e2g[:2, :2]
    return e if frame == "ego" else np.c_[-e[:, 1], e[:, 0]]


def dirs_to_frame(d, T_e2g, frame):
    """global direction vectors (n, 2) -> the frame (no translation)"""
    e = np.asarray(d, float).reshape(-1, 2) @ T_e2g[:2, :2]
    return e if frame == "ego" else np.c_[-e[:, 1], e[:, 0]]


def clip_square(P, half):
    """cut a polyline (n, 2) to the square |x|, |y| <= half exactly at the border -> [(pts, cut_start, cut_end, seg_first, seg_last)]
    (parts in travel order; seg_first / seg_last are the indices of the first / last original segment the part uses)"""
    if len(P) < 2:
        return []
    a, b = P[:-1], P[1:]
    d = b - a
    t0, t1 = np.zeros(len(d)), np.ones(len(d))
    ok = np.ones(len(d), bool)
    for ax in (0, 1):
        da, pa = d[:, ax], a[:, ax]
        zero = np.abs(da) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            tl, th = (-half - pa) / da, (half - pa) / da
        lo, hi = np.where(da > 0, tl, th), np.where(da > 0, th, tl)
        lo, hi = np.where(zero, 0.0, lo), np.where(zero, 1.0, hi)
        ok &= ~(zero & (np.abs(pa) > half))
        t0, t1 = np.maximum(t0, lo), np.minimum(t1, hi)
    ok &= t0 < t1 - 1e-9
    parts, cur = [], None
    for i in np.nonzero(ok)[0]:
        i = int(i)
        entry, exit_ = a[i] + t0[i] * d[i], a[i] + t1[i] * d[i]
        if cur is not None and cur["last"] == i - 1 and cur["t1"] >= 1.0 - 1e-9 and t0[i] <= 1e-9:
            cur["pts"].append(exit_)
            cur["last"], cur["t1"] = i, t1[i]
        else:
            if cur is not None:
                parts.append(cur)
            cur = dict(pts=[entry, exit_], first=i, last=i, t0=t0[i], t1=t1[i])
    if cur is not None:
        parts.append(cur)
    # a part is cut where the polyline goes on outside the square, also when a vertex lies exactly on the border (t0 = 0 / t1 = 1)
    return [(np.asarray(c["pts"]), c["t0"] > 1e-9 or c["first"] > 0, c["t1"] < 1.0 - 1e-9 or c["last"] < len(d) - 1, c["first"], c["last"]) for c in parts]


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
            P = polys[tok]
            if pts and np.linalg.norm(P[0] - pts[-1][-1]) < 0.05:
                P = P[1:]                         # the joint point once
            spans.append((tok, n, n + len(P) - 1))
            pts.append(P)
            n += len(P)
        P = np.concatenate(pts)
        kinds = {kind[t] for t in chain}
        pid = chain[0]
        pieces[pid] = dict(id=pid, tokens=chain, kind=kinds.pop() if len(kinds) == 1 else "mixed", pts=P, spans=spans,
                           bbox=np.r_[P.min(0), P.max(0)])
        for tok in chain:
            piece_of[tok] = pid
    for pid, p in pieces.items():
        p["succ"] = [piece_of[v] for v in succ[p["tokens"][-1]]]
        p["pred"] = [piece_of[u] for u in pred[p["tokens"][0]]]
    events = _map_events(pieces, piece_of, succ, pred, kind, in_inter, stop_rule, step)
    n = 8                                                       # ids in the output are the first n characters of the token (the shortest unique prefix, at least 8)
    while len({pid[:n] for pid in pieces}) < len(pieces):
        n += 2
    short = {pid: pid[:n] for pid in pieces}
    return dict(short=short, pieces=pieces, piece_of=piece_of, succ=succ, pred=pred, kind=kind, in_inter=in_inter, events=events, step=step, polys=polys, inter=inter,
                params=dict(inter_frac=inter_frac, stop_rule=stop_rule))


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
def crop(mg, T_e2g, half=50.0, min_len=2.0, frame="lane", margin=25.0, rg=None):
    """the graph of one keyframe: the map graph `mg` cut to the window around the ego pose T_e2g (4 x 4, ego -> global); with `rg` =
    road_graph.build_road_graph(mg) the neighbours, lane counts and roads are added (see autogt.road_graph)"""
    t = T_e2g[:2, 3]
    pieces, short = mg["pieces"], mg["short"]
    r = half * math.sqrt(2.0) + margin
    keep = [pid for pid, p in pieces.items()
            if p["bbox"][2] > t[0] - r and p["bbox"][0] < t[0] + r and p["bbox"][3] > t[1] - r and p["bbox"][1] < t[1] + r]
    out_parts, parts_of = [], {}
    for pid in sorted(keep):
        p = pieces[pid]
        Q = to_frame(p["pts"], T_e2g, frame)
        k = 0
        for pts, cut0, cut1, first, last in clip_square(Q, half):
            if _length(pts) < (min_len if (cut0 and cut1) else 0.05):             # a fragment cut at both ends is dropped when short; a natural end keeps its links
                continue
            part = dict(id=short[pid] if k == 0 else f"{short[pid]}#{k}", piece=pid, part=k, kind=p["kind"], pts=pts.astype(np.float32), start_open=bool(cut0),
                        end_open=bool(cut1), length=_length(pts), tokens=[s[0] for s in p["spans"] if s[1] <= last + 1 and s[2] >= first],
                        succ=[], pred=[], seg=(first, last))
            # the index ranges of the part's points per map piece: point j is the original vertex first + j (the first and the last one may be cut points)
            part["segments"] = [dict(token=s[0], kind=mg["kind"][s[0]], i0=int(max(s[1] - first, 0)), i1=int(min(s[2] - first, len(pts) - 1)),
                                     in_intersection=bool(mg["in_inter"].get(s[0], False)))
                                for s in p["spans"] if s[1] <= last + 1 and s[2] >= first and min(s[2] - first, len(pts) - 1) >= max(s[1] - first, 0)]
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

    def inside(p):
        return abs(p[0]) <= half + 1e-6 and abs(p[1]) <= half + 1e-6
    def pick(ids, dirs, table, on=None, q=None):
        """the ids that exist in this window (the part of the piece through `table`; the part `q` for the piece `on` that carries the event) with their direction rows"""
        keep = [k for k, i in enumerate(ids) if (q is not None and i == on) or table.get(i) is not None]
        return [q["id"] if (q is not None and ids[k] == on) else table[ids[k]]["id"] for k in keep], np.asarray(dirs).reshape(-1, 2)[keep]
    for e in mg["events"]:
        pos = to_frame(e["pos"][None], T_e2g, frame)[0]
        if not inside(pos):
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
        if len(ids_in) + len(ids_out) == 0:
            continue
        events.append(dict(type=e["type"], pos=pos.astype(np.float32), **extra, **{"in": ids_in, "out": ids_out},
                           dirs_in=dirs_to_frame(d_in, T_e2g, frame).astype(np.float32), dirs_out=dirs_to_frame(d_out, T_e2g, frame).astype(np.float32),
                           partial=bool(len(ids_in) != len(e["in"]) or len(ids_out) != len(e["out"]))))
    for part in out_parts:                                   # the window border
        if part["start_open"]:
            events.append(dict(type="window", io="in", pos=part["pts"][0], **{"in": [], "out": [part["id"]]}, dirs_in=np.zeros((0, 2), np.float32),
                               dirs_out=_tangent(part["pts"], False)[None].astype(np.float32), partial=False))
        if part["end_open"]:
            events.append(dict(type="window", io="out", pos=part["pts"][-1], **{"in": [part["id"]], "out": []},
                               dirs_in=_tangent(part["pts"], True)[None].astype(np.float32), dirs_out=np.zeros((0, 2), np.float32), partial=False))
    graph = dict(frame=frame, half=half, pieces=out_parts, events=events, params=dict(min_len=min_len, **mg["params"]))
    if rg is not None:
        from . import road_graph
        road_graph.attach(graph, mg, rg)
    for part in out_parts:
        del part["seg"]                                      # the index range in the map piece: only needed to attach the roads
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
                else:
                    out[k] = walk(v) if not isinstance(v, np.ndarray) else v
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    out = walk(graph)
    out["frame"] = frame
    return out
