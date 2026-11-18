"""The NAMED HARD OUTPUT STEPS: from the soft outputs of one frame to the delivered lane graph (numpy, no gradient goes through here).

    graph = deliver(lanes, pairs, cfg)       # lanes: LaneFrame, pairs: PairFrame (the contract below; `lg_model.per_frame` fills it)
Steps, in order (each a selection on the output, D55 / D88):
  0. section of a node = the argmax of its soft section membership (computed in `lg_lane_head.section_start`)
  1. close-node suppression (E275): nodes with e >= node_thr, strongest first; a node is dropped when a kept node lies closer than close_radius
  2. link threshold: links of B with p >= link_thr between kept nodes
  3. sections: maximal chains of the kept links inside one section are the lane pieces; a kept link between two sections is a link piece
  4. pair threshold: a connector for every allowed keypoint pair with p >= pair_thr
  5. cover: a long link of B whose first and last point lie within cover_radius of a connector's entry and exit end is dropped
  6. wiring (E311): each connector end is wired to the nearest kept piece end within wire_radius and wire_angle; the card of record draws the
     connector between those piece ends (the snap's geometry); with attached_ends the connector runs between the learned attached ends and the
     same proximity rule wires it
  7. Douglas-Peucker (`lg_graph.simplify`, applied by the JSON writer) thins the written polylines; the scorer reads the dense ones
"""
from typing import TypedDict

import numpy as np

from lg_config import Config
from lg_graph import Graph


class LaneFrame(TypedDict):
    """B's soft outputs of one frame (K active nodes, L candidate links), float64"""
    pos: np.ndarray        # (K, 2) node position (m)
    dir: np.ndarray        # (K, 2) unit heading
    e: np.ndarray          # (K,) existence probability
    valid: np.ndarray      # (K,) bool, a prefix of the slots
    start: np.ndarray      # (K,) the slot of the node its section starts at (step 0)
    src: np.ndarray        # (L,) link source slot
    dst: np.ndarray        # (L,) link target slot
    p: np.ndarray          # (L,) link probability
    sa: np.ndarray         # (L,) handle scales (chords)
    sb: np.ndarray
    long: np.ndarray       # (L,) bool, a long link


class PairFrame(TypedDict, total=False):
    """the junction head's soft outputs of one frame (K entry + K exit keypoints), float32 as the network made them"""
    kp_pos: np.ndarray     # (2K, 2) keypoint position: entries first
    kp_dir: np.ndarray     # (2K, 2) unit heading
    logit: np.ndarray      # (K, K) pair logit entry i -> exit j
    allowed: np.ndarray    # (K, K) bool
    scale: np.ndarray      # (K, K, 2) handle scales of the connector
    att_pos: np.ndarray    # (2K, 2) attached ends (only with Config.attached_ends)
    att_dir: np.ndarray


def cubic(p0, h0, p1, h1, a: float, b: float, n: int) -> np.ndarray:
    """(n, 2) points of the cubic curve p0 -> p1 with control points p0 + a c h0 and p1 - b c h1 (c = the chord)"""
    c = float(np.linalg.norm(p1 - p0))
    s = np.linspace(0, 1, n)[:, None]
    q1, q2 = p0 + a * c * h0, p1 - b * c * h1
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * q1 + 3 * (1 - s) * s ** 2 * q2 + s ** 3 * p1


def suppress_close_nodes(pos: np.ndarray, e: np.ndarray, radius: float, thr: float) -> np.ndarray:
    """(M,) bool: greedy by e descending (ties: lower index first), keep a node unless a kept node is strictly closer than radius"""
    keep = np.zeros(len(e), bool)
    cand = np.flatnonzero(e >= thr)
    kept, n = np.empty((len(cand), 2)), 0
    for i in cand[np.argsort(-e[cand], kind="stable")]:
        if n == 0 or np.min(((kept[:n] - pos[i]) ** 2).sum(1)) >= radius ** 2:
            kept[n], n, keep[i] = pos[i], n + 1, True
    return keep


def chains(edges: list) -> list:
    """maximal vertex chains of a directed graph: a chain starts at every out-edge of a vertex with in- or out-degree != 1 and runs through vertices
    with in = out = 1; what is left are cycles, one closed chain each (started at its smallest edge)"""
    outs, ins = {}, {}
    for a, b in edges:
        outs.setdefault(a, []).append(b)
        ins.setdefault(b, []).append(a)
    used, out = set(), []

    def walk(s, nxt):
        c = [s, nxt]
        used.add((s, nxt))
        while len(outs.get(c[-1], ())) == 1 and len(ins.get(c[-1], ())) == 1 and (c[-1], outs[c[-1]][0]) not in used:
            used.add((c[-1], outs[c[-1]][0]))
            c.append(outs[c[-1]][0])
        out.append(c)
    for s in sorted(outs):
        if len(ins.get(s, ())) != 1 or len(outs[s]) != 1:
            for nxt in outs[s]:
                if (s, nxt) not in used:
                    walk(s, nxt)
    for a, b in sorted(set(edges)):
        if (a, b) not in used:
            walk(a, b)
    return out


def lane_pieces(lanes: LaneFrame, cfg: Config) -> tuple:
    """steps 1 - 3: -> (pieces [(polyline, node ids, kind, is a long link)], edges [(i, j)]) of B's delivered graph"""
    pos, dirv, e = lanes["pos"], lanes["dir"], lanes["e"]
    keep = suppress_close_nodes(pos, e, cfg.close_radius, cfg.node_thr)
    src, dst = lanes["src"], lanes["dst"]
    L = np.flatnonzero((lanes["p"] >= cfg.link_thr) & keep[src] & keep[dst])
    sec = np.where(lanes["valid"][lanes["start"]], lanes["start"], np.arange(len(e)))
    cross = sec[src[L]] != sec[dst[L]]
    by_pair = {(int(src[l]), int(dst[l])): int(l) for l in L}

    def curve(l):
        a, b = src[l], dst[l]
        n = max(int(np.linalg.norm(pos[b] - pos[a]) / cfg.lane_step) + 2, 3)
        return cubic(pos[a], dirv[a], pos[b], dirv[b], float(lanes["sa"][l]), float(lanes["sb"][l]), n)
    pieces = []
    for c in chains([(int(src[l]), int(dst[l])) for l in L[~cross]]):
        ls = [by_pair[ab] for ab in zip(c[:-1], c[1:])]
        pieces.append((np.concatenate([curve(l)[(0 if i == 0 else 1):] for i, l in enumerate(ls)]), c, "lane", False))
    for l in L[cross]:
        pieces.append((curve(l), [int(src[l]), int(dst[l])], "link", bool(lanes["long"][l])))
    starts = {}
    for i, pc in enumerate(pieces):
        starts.setdefault(pc[1][0], []).append(i)
    return pieces, [(i, j) for i, pc in enumerate(pieces) for j in starts.get(pc[1][-1], []) if j != i]


def _tangents(lines: list, at_end: bool) -> np.ndarray:
    """(n, 2) unit heading over the last (first) three points of every polyline"""
    out = []
    for P in lines:
        q = P[-3:] if at_end else P[:3]
        d = q[-1] - q[0]
        n = np.linalg.norm(d)
        out.append(d / n if n > 1e-9 else np.array([0.0, 1.0]))
    return np.asarray(out, float).reshape(-1, 2)


def _nearest(Q: np.ndarray, T: np.ndarray, p: np.ndarray, t: np.ndarray, cfg: Config):
    """index of the nearest point of Q within wire_radius whose heading T is within wire_angle of t (the first on a tie), or None"""
    d = np.linalg.norm(Q - p, axis=1)
    ok = np.flatnonzero((d <= cfg.wire_radius) & (np.degrees(np.arccos(np.clip(T @ t, -1.0, 1.0))) <= cfg.wire_angle))
    return int(ok[np.argmin(d[ok])]) if len(ok) else None


def deliver(lanes: LaneFrame, pairs: PairFrame, cfg: Config) -> Graph:
    """the delivered graph of one frame: B's pieces (steps 1 - 3) + the connectors of the junction head (steps 4 - 6)"""
    pieces, edges = lane_pieces(lanes, cfg)
    K = pairs["logit"].shape[0]
    prob = 1.0 / (1.0 + np.exp(-pairs["logit"])) * pairs["allowed"]                             # float32, as the card's read-out
    ia, ib = np.nonzero(prob >= cfg.pair_thr)
    ep, ed = (pairs["att_pos"], pairs["att_dir"]) if cfg.attached_ends else (pairs["kp_pos"], pairs["kp_dir"])
    cand = [(ep[i], ed[i], ep[K + j], ed[K + j], pairs["scale"][i, j]) for i, j in zip(ia, ib)]
    covered = lambda P: any(np.linalg.norm(P[0] - c[0]) <= cfg.cover_radius and np.linalg.norm(P[-1] - c[2]) <= cfg.cover_radius for c in cand)  # noqa: E731
    keep = [i for i, (P, _, _, long) in enumerate(pieces) if not (long and covered(P))]
    new = {old: k for k, old in enumerate(keep)}
    out = Graph([np.asarray(pieces[i][0], float) for i in keep], [(new[a], new[b]) for a, b in edges if a in new and b in new],
                [list(pieces[i][1]) for i in keep], [pieces[i][2] for i in keep])
    last, first = np.array([P[-1] for P in out.lines]).reshape(-1, 2), np.array([P[0] for P in out.lines]).reshape(-1, 2)
    t_last, t_first = _tangents(out.lines, True), _tangents(out.lines, False)
    for n, (pa, ta, pb, tb, sc) in enumerate(cand):
        pa, ta, pb, tb = (np.asarray(x, float) for x in (pa, ta, pb, tb))
        w_in, w_out = _nearest(last, t_last, pa, ta, cfg), _nearest(first, t_first, pb, tb, cfg)
        if not cfg.attached_ends:                                                                 # the snap's geometry: the curve runs between the wired piece ends
            pa, ta = (last[w_in], t_last[w_in]) if w_in is not None else (pa, ta)
            pb, tb = (first[w_out], t_first[w_out]) if w_out is not None else (pb, tb)
        m = len(out.lines)
        npts = max(cfg.connector_min_points, int(cfg.connector_density * float(np.linalg.norm(pb - pa)) / cfg.connector_step))
        out.lines.append(cubic(pa, ta, pb, tb, float(sc[0]), float(sc[1]), npts))
        out.nodes.append([cfg.connector_ids[0] + n, cfg.connector_ids[1] + n])
        out.kinds.append("connector")
        out.edges += ([(w_in, m)] if w_in is not None else []) + ([(m, w_out)] if w_out is not None else [])
    return out
