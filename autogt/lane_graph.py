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
  start    a piece without a predecessor in the map;  end   a piece without a successor in the map
  window   the polyline crosses the window border (io "in" / "out"); per frame
`stop_rule` decides a lane with several successors: "intersection_branch" (default: a stop point for the intersection branches only), "any_successor"
(if one is an intersection connector, all branches), "all_successors", "none". `dirs_in` / `dirs_out` are the unit directions of the pieces at the event.

FRAME. frame="lane" (default): the lane project's frame, ego (rear axle) at the front-camera time, x to the RIGHT, y forward; frame="ego": the frame of
the other autogt layers, x forward, y left. The window is the square |x|, |y| <= half (default 50 m). Polylines are cut exactly at the border; a
part shorter than `min_len` is dropped; a piece that leaves the window and comes back is several parts (`part` 0, 1, ...; id "<piece>#<part>").
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
    return [(np.asarray(c["pts"]), c["t0"] > 1e-9, c["t1"] < 1.0 - 1e-9, c["first"], c["last"]) for c in parts]


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
    events = _map_events(pieces, piece_of, succ, kind, in_inter, stop_rule, step)
    return dict(pieces=pieces, piece_of=piece_of, succ=succ, pred=pred, kind=kind, in_inter=in_inter, events=events, step=step,
                params=dict(inter_frac=inter_frac, stop_rule=stop_rule))


def _map_events(pieces, piece_of, succ, kind, in_inter, stop_rule, step):
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
        for tok in sorted(succ):
            if kind[tok] != "lane" or not succ[tok]:
                continue
            into = [c for c in succ[tok] if kind[c] == "connector" and in_inter.get(c)]
            if stop_rule == "intersection_branch":
                out = into
            elif stop_rule == "any_successor":
                out = succ[tok] if into else []
            else:
                out = succ[tok] if len(into) == len(succ[tok]) else []
            if not out:
                continue
            pid = piece_of[tok]
            sp = _span(pieces, pid, tok)
            P = pieces[pid]["pts"]
            dirs_out = []
            for c in out:
                pc = piece_of[c]
                dirs_out.append(_tangent(pieces[pc]["pts"][_span(pieces, pc, c)[1]:], False, step=step))
            ev.append(dict(type="stop", pos=P[sp[2]], on=pid, idx=sp[2], n_succ=len(succ[tok]), **{"in": [pid], "out": sorted({piece_of[c] for c in out})},
                           dirs_in=np.array([_tangent(P[:sp[2] + 1], True, step=step)]), dirs_out=np.array(dirs_out)))
    return ev


def _span(pieces, pid, tok):
    """(token, first index, last index) of a map piece inside its joined piece"""
    return next(s for s in pieces[pid]["spans"] if s[0] == tok)


# ---------------------------------------------------------------------------------------------------------------- a keyframe's crop
def crop(mg, T_e2g, half=50.0, min_len=2.0, frame="lane", margin=25.0):
    """the graph of one keyframe: the map graph `mg` cut to the window around the ego pose T_e2g (4 x 4, ego -> global)"""
    t = T_e2g[:2, 3]
    pieces = mg["pieces"]
    r = half * math.sqrt(2.0) + margin
    keep = [pid for pid, p in pieces.items()
            if p["bbox"][2] > t[0] - r and p["bbox"][0] < t[0] + r and p["bbox"][3] > t[1] - r and p["bbox"][1] < t[1] + r]
    out_parts, parts_of = [], {}
    for pid in sorted(keep):
        p = pieces[pid]
        Q = to_frame(p["pts"], T_e2g, frame)
        k = 0
        for pts, cut0, cut1, first, last in clip_square(Q, half):
            if _length(pts) < min_len:
                continue
            part = dict(id=pid if k == 0 else f"{pid}#{k}", piece=pid, part=k, kind=p["kind"], pts=pts.astype(np.float32), start_open=bool(cut0),
                        end_open=bool(cut1), length=_length(pts), tokens=[s[0] for s in p["spans"] if s[1] <= last + 1 and s[2] >= first],
                        succ=[], pred=[])
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
    events = []

    def inside(p):
        return abs(p[0]) <= half + 1e-6 and abs(p[1]) <= half + 1e-6
    for e in mg["events"]:
        pos = to_frame(e["pos"][None], T_e2g, frame)[0]
        if not inside(pos):
            continue
        if e["type"] == "stop":
            ps = parts_of.get(e["on"], [])
            best = min(((project(q["pts"].astype(float), pos), q) for q in ps), key=lambda x: x[0][1], default=None)
            if best is None or best[0][1] > 0.6:
                continue
            (s, _), q = best
            ids_in, ids_out = [q["id"]], [head[o]["id"] for o in e["out"] if head.get(o) is not None]
            events.append(dict(type="stop", pos=pos.astype(np.float32), on=q["id"], s=s, n_succ=e["n_succ"], **{"in": ids_in, "out": ids_out},
                               dirs_in=dirs_to_frame(e["dirs_in"], T_e2g, frame).astype(np.float32),
                               dirs_out=dirs_to_frame(e["dirs_out"], T_e2g, frame).astype(np.float32), partial=False))
            continue
        ids_in = [tail[i]["id"] for i in e["in"] if tail.get(i) is not None]
        ids_out = [head[o]["id"] for o in e["out"] if head.get(o) is not None]
        if len(ids_in) + len(ids_out) == 0:
            continue
        events.append(dict(type=e["type"], pos=pos.astype(np.float32), **{"in": ids_in, "out": ids_out},
                           dirs_in=dirs_to_frame(e["dirs_in"], T_e2g, frame).astype(np.float32),
                           dirs_out=dirs_to_frame(e["dirs_out"], T_e2g, frame).astype(np.float32),
                           partial=bool(len(ids_in) != len(e["in"]) or len(ids_out) != len(e["out"]))))
    for part in out_parts:                                   # the window border
        if part["start_open"]:
            events.append(dict(type="window", io="in", pos=part["pts"][0], **{"in": [], "out": [part["id"]]}, dirs_in=np.zeros((0, 2), np.float32),
                               dirs_out=_tangent(part["pts"], False)[None].astype(np.float32), partial=False))
        if part["end_open"]:
            events.append(dict(type="window", io="out", pos=part["pts"][-1], **{"in": [part["id"]], "out": []},
                               dirs_in=_tangent(part["pts"], True)[None].astype(np.float32), dirs_out=np.zeros((0, 2), np.float32), partial=False))
    return dict(frame=frame, half=half, pieces=out_parts, events=events, params=dict(min_len=min_len, **mg["params"]))
