"""The local scorer of the lane head (E273, card section 4): what the head delivers per frame -> the delivered lane graph -> the numbers. Local to the card: `evaluate.py` and the global
protocol are untouched; the graph part is scored by the SAME functions as `evaluate.py` and the lane suite (`lane_suite.py graphs:<name>`), so the reference graph d and the new graph p are
read by one ruler (tested: the stored graphs of recall_sp0_dp give the counts of the stored suite run exactly).

    pred = qformat.oracle_prediction(lane_gt_graph)         # or the head's output: the dict of training/grid/qformat.py (nodes, links, assign)
    graph, bk = readout(pred)                                # the output step: e >= 0.5, p >= 0.5, one piece per section -> LaneGraph + bookkeeping
    frames = val_frames()                                    # the 873 VAL frames (lane suite order): frame.gt = old graph GT, frame.lane() = lane GT of autogt
    graph_metrics(graph, frames[0])                          # HEADLINE: Q, F1@1.0, lanes_found, topo P / R / F1, junction P / R / F1, dup_frac, R@1_far, vert100, pieces_per_gt_piece ...
    node_metrics(pred, frames[0].lane()); assign_metrics(pred, frames[0].lane())
    out = score_frames(preds, frames)                        # {'graph', 'nodes', 'assign'} pooled over frames the project's way (micro counts); records=True: + the per-frame counts for `paired`

Readout (the named OUTPUT step, no loss goes through it). Kept nodes: e >= thr_e; kept links: p >= thr_link between kept nodes (one link per node pair, the most probable). A SECTION (the
delivered lane piece) is a set of nodes with one `section` value; its polyline is the chain of Hermite curves (qtargets.hermite, handles a, b; `curves` if the prediction carries them) of the kept
links INSIDE the section, sampled every 0.5 m (a section whose kept links branch or break gives one polyline per chain, all with that section id). A kept link between two sections is a connector
piece of its own (its Hermite curve: the connectors through the intersections carry no node, so their geometry is exactly these links); edges of the graph join a piece to every piece that
starts at its end node. `bk` = the bookkeeping (also graph.meta): kept nodes, kept links, `cross` (link joins two sections), per piece kind / section / links / nodes, the kept nodes with sections.

Numbers (from ADDITIVE counts per frame, pooled by summing, so any set of frames can be scored again; `*_counts` -> `pool_counts` -> `*_numbers`):
  graph_metrics   `sparse_lanes/bootstrap.py` (Q, S, J, F1@t, lanes_found, split6 / merge6; vert100 = unique kept node ids per 100 m of delivered polyline, vert100_dp = Douglas-Peucker vertices
                  of the same polylines), `topo_metrics.py` (topo P / R / F1, dup_frac, R@1 by range), `lane_task/measures.py` topo_counts (junction P / R / F1 within 3 m: the lane suite's topo
                  metric), all against the OLD graph GT of the frame (`graph_gt_val.pkl`: the GT the lane suite scores d with; the lane GT agrees with it within 0.3 m); plus pieces_per_gt_piece
                  (distinct sections among the nodes on one GT piece, mean over GT pieces with a node) and id_switches_per_100m (changes of the section along a GT piece, per 100 m of GT lane).
                  A node belongs to the GT piece of its nearest GT lane point when that is within 2 m and no other piece has a point within TIE (0.3 m) of it (shared junction points are left out).
  node_metrics    F1 of node existence within 2 m (greedy one-to-one matching, nearest pairs first: a metric only), median position / direction error of the matched nodes, event P / R per flag
                  within 3 m, short / long link P / R and out-degree accuracy on matched nodes, nodes per 100 m of GT lane.
  assign_metrics  lane_acc (e-weighted share of the nodes within 1 m of a GT lane whose most probable map piece is a piece of that lane; a GT piece the map lacks means the dustbin), dust P / R
                  (nodes farther than 2 m from every GT lane, or on a piece the map lacks, belong to the dustbin), purity (mean over GT lanes of the share on the majority map piece: consistency,
                  not correctness), road_acc (the same at road level). A node at a point shared by several GT pieces accepts any of them.
A ratio with an empty denominator is 0.0 when the other side is not empty (an empty prediction scores 0, not 1), NaN when both are empty. Q of an empty graph is 0.15 / 0.85 (its false-length term).
Self-test: python qgraph_eval.py (a toy frame); real frames: evaluation/tests/test_qgraph_eval.py (wsl bash tools/env/_py14.sh test_qgraph_eval.py).
"""
import json
import os
import pickle
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

import bflc_paths as BP  # project paths from the location of this checkout
import bootstrap as SB
import lane_suite as LS
import qtargets as QT
import sparse_metrics as SM
import topo_metrics as TM
from lane_task import LaneGraph
from lane_task import measures as MS

CFG = LS.config()                 # the lane suite's settings: box 48 m, topo seeds every 5 m, junctions within 3 m
BOX = CFG.box[1]
STEP = 0.5                        # sampling of the delivered curves (m)
TIE = 0.3                         # GT points this much farther than the nearest one still tie with it: a point shared by pieces (qtargets groups events within 0.3 m too)
NODE_SPARSE = None                # E276: None = the GT nodes of E273 (a node in every 2 m cell); (eps, max chord) = the sparse node target the model was trained on (qtargets.node_table)
TIER_HIT, TIER_ZONE = 1.0, 2.5    # E276 tier metrics: a point is right within 1.0 m of the other graph; a delivered point belongs to the zone of its nearest GT point within 2.5 m
NODE_TOL, EVENT_TOL = 2.0, 3.0    # node / event matching tolerances (m)
SEC_NEAR, LANE_NEAR, DUST_FAR = 2.0, 1.0, 2.0
RAGGED = ("pos_err", "dir_err")   # counts that are concatenated, not summed
HEADLINE = ("Q", "F1@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "dup_frac", "R@1_far", "vert100", "vert100_dp", "pieces_per_gt_piece",
            "id_switches_per_100m")


# ------------------------------------------------------------------------------------------------------------------------------ readout
def kept(pred, thr_e=0.5, thr_link=0.5):
    """-> (keep (M,) bool, L (n,) int): the kept nodes and the ids of the kept links (between kept nodes, one per node pair, the most probable)"""
    nd, lk = pred["nodes"], pred["links"]
    keep = np.asarray(nd["e"], float).reshape(-1) >= thr_e
    src, dst = (np.asarray(lk[k]).reshape(-1).astype(int) for k in ("src", "dst"))
    p = np.asarray(lk["p"], float).reshape(-1)
    ok = (p >= thr_link) & keep[src] & keep[dst] & (src != dst)
    best = {}
    for l in np.flatnonzero(ok):
        key = (int(src[l]), int(dst[l]))
        if key not in best or p[l] > p[best[key]]:
            best[key] = int(l)
    return keep, np.array(sorted(best.values()), int)


def readout(pred, thr_e=0.5, thr_link=0.5, step=STEP):
    """the output step (module docstring) -> (LaneGraph, bookkeeping dict); the graph carries one node-id list per polyline (vert100 counts them) and the bookkeeping as `meta`"""
    nd, lk = pred["nodes"], pred["links"]
    pos, dirv = np.asarray(nd["pos"], float).reshape(-1, 2), np.asarray(nd["dirv"], float).reshape(-1, 2)
    sec = np.asarray(nd["section"]).reshape(-1)
    src, dst = (np.asarray(lk[k]).reshape(-1).astype(int) for k in ("src", "dst"))
    keep, L = kept(pred, thr_e, thr_link)
    cross = sec[src[L]] != sec[dst[L]] if len(L) else np.zeros(0, bool)
    by_pair = {(int(src[l]), int(dst[l])): int(l) for l in L}
    curves = lk.get("curves")

    def curve(l):
        if curves is not None and curves[l] is not None:
            return np.asarray(curves[l], float)
        a, b = src[l], dst[l]
        return QT.hermite(pos[a], dirv[a], pos[b], dirv[b], float(lk["a"][l]), float(lk["b"][l]), step)

    inside = [(int(src[l]), int(dst[l])) for l, c in zip(L, cross) if not c]
    lines, nodes, kind, section, links = [], [], [], [], []
    for chain in SM.graph_from_vertices(pos, inside)["nodes"]:                       # maximal chains of the kept links inside a section
        ls = [by_pair[(a, b)] for a, b in zip(chain[:-1], chain[1:])]
        lines.append(np.concatenate([curve(l)[(0 if i == 0 else 1):] for i, l in enumerate(ls)]))
        nodes.append([int(v) for v in chain])
        kind.append("section"); section.append(sec[chain[0]]); links.append(ls)
    for l in L[cross]:                                                                 # a link between two sections is a connector piece
        lines.append(curve(l)); nodes.append([int(src[l]), int(dst[l])])
        kind.append("link"); section.append(-1); links.append([int(l)])
    starts = {}
    for i, n in enumerate(nodes):
        starts.setdefault(n[0], []).append(i)
    edges = [(i, j) for i, n in enumerate(nodes) for j in starts.get(n[-1], []) if j != i]
    idx = np.flatnonzero(keep)
    bk = dict(keep=keep, links=L, cross=cross, kind=kind, section=section, piece_links=links, piece_nodes=nodes, nodes=dict(idx=idx, pos=pos[idx], section=sec[idx]))
    return LaneGraph(lines, edges, True, nodes, bk), bk


def to_raw(graph):
    """LaneGraph -> the sparse-lane contract dict(pts, edges, nodes) of graph_eval / bootstrap / topo_metrics (a raw graph dict passes through)"""
    if isinstance(graph, dict):
        return graph
    raw = dict(pts=[np.asarray(p, float) for p in graph.lines], edges=[tuple(e) for e in graph.edges])
    if graph.nodes is not None:
        raw["nodes"] = graph.nodes
    return raw


# ------------------------------------------------------------------------------------------------------------------------------ frames
@dataclass
class Frame:
    """one VAL frame: k position in the 873 (the order of the lane suite and of E271 / E272), row in the split file / feature files, token = nuScenes sample token, scene number,
    gt = the old graph GT (dict pts, edges) that the lane suite scores against; lane() = the lane GT of autogt (FileNotFoundError while it is not stored), ref() = it prepared for lookups"""
    k: int
    row: int
    token: str
    scene: int
    gt: dict
    _lane: object = field(default=None, repr=False, compare=False)
    _ref: object = field(default=None, repr=False, compare=False)

    def lane(self):
        if self._lane is None:
            sys.path.insert(0, BP.REPO)
            from autogt import lane_gt
            self._lane = lane_gt.load(self.token)
        return self._lane

    def ref(self):
        if self._ref is None:
            self._ref = LaneRef(self.lane())
        return self._ref


def val_frames():
    """the 873 VAL frames in the order of data/splits.json['val'] (= the lane suite, BuilderData('val_clean').rows); the tokens are val_tokens.json[row] (that file also holds the TEST rows)"""
    rows = np.asarray(json.load(open(f"{BP.ROOT}/data/splits.json"))["val"], int)
    tokens = json.load(open(f"{LS.B123}/feats/val_tokens.json"))
    scene = np.load(f"{LS.B123}/poses_val.npz")["scene"]
    gt = pickle.load(open(f"{LS.B123}/graph_gt_val.pkl", "rb"))
    return [Frame(k, int(r), tokens[r], int(scene[r]), gt[r]) for k, r in enumerate(rows)]


def stored_share(frames):
    """share of the frames whose lane GT is in the store of this checkout's lane signature (the store is built in the background)"""
    sys.path.insert(0, BP.REPO)
    from autogt import lane_gt
    held = lane_gt.held(os.path.join(lane_gt.DATA, lane_gt.signature()))
    return sum(f.token in held for f in frames) / max(len(frames), 1)


class LaneRef:
    """the lane GT of a frame prepared for lookups: the GT nodes / links (`qtargets.node_table`), the points of all pieces in a KD-tree, the GT lane length inside the metrics' box"""

    def __init__(self, g):
        self.nt = nt = QT.node_table(g, sparse=NODE_SPARSE)
        self.pts, self.pid, self.pj = nt["pts"], nt["pid"], nt["pj"]
        self.ptier, self.pvis = nt["ptier"], nt["pvis"]                                  # E276: the l4 labels of the GT points (tier 0 / visible when the graph has none)
        self.labelled = bool((self.ptier > 0).any())
        self.tree = cKDTree(self.pts) if len(self.pts) else None
        self.length = 0.5 * float((np.abs(self.pts) < BOX).all(1).sum()) if len(self.pts) else 0.0       # points every 0.5 m
        self.ids = [p["piece"] for p in g["pieces"]]
        self.roads = [p["road"] for p in g["pieces"]]

    def near(self, xy):
        """-> d (n,) distance to the nearest GT lane point, tied (n,) arrays of the GT pieces with a point within d + TIE, pj (n,) index along its piece of the nearest point"""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.tree is None or not len(xy):
            return np.full(len(xy), np.inf), [np.zeros(0, int) for _ in xy], np.full(len(xy), -1)
        d, i = self.tree.query(xy)
        tied = [np.unique(self.pid[self.tree.query_ball_point(x, di + TIE)]) for x, di in zip(xy, d)]
        return d, tied, self.pj[i]


def _ref(g):
    return g if isinstance(g, LaneRef) else LaneRef(g)


# ------------------------------------------------------------------------------------------------------------------------------ counts -> numbers
def _ratio(a, b, other):
    """a / b; an empty denominator is 0.0 when the other side has something, NaN when both are empty"""
    return float(a) / float(b) if b > 0 else (0.0 if other > 0 else float("nan"))


def _prf(ok, n_pred, n_gt):
    p, r = _ratio(ok, n_pred, n_gt), _ratio(ok, n_gt, n_pred)
    if p != p or r != r:
        return p, r, float("nan")
    return p, r, 2 * p * r / (p + r) if p + r > 0 else 0.0


def pool_counts(cs):
    """sum the count dicts of a set of frames (the ragged error lists are concatenated)"""
    out = {}
    for c in cs:
        for k, v in c.items():
            v = np.asarray(v, float)
            out[k] = (np.concatenate([out[k], v]) if k in RAGGED else out[k] + v) if k in out else v.copy()
    return out


def _med(x):
    return float(np.median(x)) if len(x) else float("nan")


# ------------------------------------------------------------------------------------------------------------------------------ nodes
def node_counts(pred, g, thr_e=0.5, thr_link=0.5):
    ref = _ref(g)
    nt = ref.nt
    nd, lk = pred["nodes"], pred["links"]
    keep, L = kept(pred, thr_e, thr_link)
    idx = np.flatnonzero(keep)
    P, H = np.asarray(nd["pos"], float).reshape(-1, 2)[idx], np.asarray(nd["dirv"], float).reshape(-1, 2)[idx]
    F = np.asarray(nd["flags"], float).reshape(len(keep), len(QT.FLAGS))[idx]
    G = nt["pos"]
    pairs = MS.match_1to1(P, G, NODE_TOL)                                              # greedy one-to-one, nearest pairs first (a metric only)
    pi, gi = pairs[:, 0], pairs[:, 1]
    pos_err = np.linalg.norm(P[pi] - G[gi], axis=1)
    dir_err = np.degrees(np.arccos(np.clip((H[pi] * nt["dirv"][gi]).sum(1), -1.0, 1.0)))
    ev = np.zeros((len(QT.FLAGS), 3))                                                  # per flag: matched, delivered, GT
    evt = np.zeros((len(QT.FLAGS), 3, 2))                                              # E276, per flag and tier of the GT event node (demanded ones only): matched, GT
    tg, vg = np.where(nt["tier"] > 0, nt["tier"], 3), nt["vis"]
    for f in range(len(QT.FLAGS)):
        sp, sg = np.flatnonzero(F[:, f] >= 0.5), np.flatnonzero(nt["flags"][:, f])
        pr = MS.match_1to1(P[sp], G[sg], EVENT_TOL)
        ev[f] = (len(pr), len(sp), len(sg))
        if ref.labelled and len(sg):
            hitg = np.zeros(len(sg), bool)
            if len(pr):
                hitg[pr[:, 1]] = True
            for k in (1, 2, 3):
                m = (tg[sg] == k) & vg[sg]
                evt[f, k - 1] = [(m & hitg).sum(), m.sum()]
    # links on matched nodes: a delivered link counts when both ends are matched; correct when the two GT nodes are joined by a GT link of the same kind
    g_of_p = -np.ones(len(P), int)
    g_of_p[pi] = gi
    p_of_g = -np.ones(len(G), int)
    p_of_g[gi] = pi
    local = -np.ones(len(keep), int)
    local[idx] = np.arange(len(idx))
    gt_link = {(int(a), int(b)): bool(lg) for a, b, lg in zip(nt["src"], nt["dst"], nt["long"])}
    lkc = np.zeros((2, 3))                                                             # [short, long] x (correct, delivered, GT)
    out_deg = np.zeros(len(P), int)
    for l in L:
        a, b = local[int(lk["src"][l])], local[int(lk["dst"][l])]
        out_deg[a] += 1
        if g_of_p[a] >= 0 and g_of_p[b] >= 0:
            t = int(bool(lk["long"][l]))
            lkc[t, 1] += 1
            lkc[t, 0] += gt_link.get((int(g_of_p[a]), int(g_of_p[b])), None) == bool(t)
    for (a, b), lg in gt_link.items():
        if p_of_g[a] >= 0 and p_of_g[b] >= 0:
            lkc[int(lg), 2] += 1
    gt_out = np.bincount(nt["src"], minlength=len(G)) if len(nt["src"]) else np.zeros(len(G), int)
    out = dict(node=[len(pi), len(P), len(G)], pos_err=pos_err, dir_err=dir_err, ev=ev, link=lkc, deg=[int((out_deg[pi] == gt_out[gi]).sum()), len(pi)], dens=[len(P), ref.length])
    if ref.labelled:                                                                   # E276: the node numbers by tier (GT nodes) and by zone (delivered nodes)
        tn, zn = np.zeros((3, 4)), np.zeros((4, 2))
        hit = p_of_g >= 0
        for k in (1, 2, 3):
            m = tg == k
            tn[k - 1] = [(m & vg & hit).sum(), (m & vg).sum(), (m & hit).sum(), m.sum()]
        if len(P) and ref.tree is not None:
            dz, iz = ref.tree.query(P)
            zone = np.where(dz <= TIER_ZONE, np.where(ref.ptier[iz] > 0, ref.ptier[iz], 3), 0)
            for k in range(4):
                zn[k] = [((zone == k) & (g_of_p >= 0)).sum(), (zone == k).sum()]
        out.update(tn=tn, zn=zn, evt=evt)
    return out


def node_numbers(c):
    out = {}
    out["node_P"], out["node_R"], out["node_F1"] = _prf(*c["node"])
    out["pos_err_med"], out["dir_err_med"] = _med(c["pos_err"]), _med(c["dir_err"])
    for f, name in enumerate(QT.FLAGS):
        out[f"ev_{name}_P"], out[f"ev_{name}_R"], _ = _prf(*c["ev"][f])
        out[f"ev_{name}_n"] = float(c["ev"][f][2])
    for t, name in enumerate(("short", "long")):
        out[f"link_{name}_P"], out[f"link_{name}_R"], _ = _prf(*c["link"][t])
    out["degree_acc"] = _ratio(c["deg"][0], c["deg"][1], 1.0)
    out["nodes_per_100m"] = 100.0 * c["dens"][0] / c["dens"][1] if c["dens"][1] > 0 else float("nan")
    return out


def node_metrics(pred, g, thr_e=0.5, thr_link=0.5):
    """the node numbers of one frame (against the lane GT dict `g`, or a LaneRef)"""
    return node_numbers(pool_counts([node_counts(pred, g, thr_e, thr_link)]))


# ------------------------------------------------------------------------------------------------------------------------------ tiers (E276)
def _densify(p, step=STEP):
    """a polyline resampled so that no two neighbouring points are farther apart than `step` (the vertices stay)"""
    p = np.asarray(p, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    n = np.maximum(np.ceil(seg / step - 1e-9).astype(int), 1)
    return np.concatenate([p[:1]] + [p[i] + (p[i + 1] - p[i]) * (np.arange(1, n[i] + 1) / n[i])[:, None] for i in range(len(seg))])


def tier_counts(raw, ref):
    """E276, additive counts of one frame by the l4 tier of the GT lane points (None when the lane GT carries no labels): `rec` (3, 4): per tier 1 - 3 [visible GT points within 1.0 m of a
    delivered polyline point, visible GT points, the same for all GT points], `prec` (4, 2): per ZONE 0 - 3 (the tier of the nearest GT point within 2.5 m, 0 = none) [delivered polyline points
    within 1.0 m of a GT point, delivered polyline points]; only points inside the metrics' box; delivered and GT points are every 0.5 m, so counts are half-metres"""
    if not ref.labelled or ref.tree is None:
        return None
    dp = [_densify(p) for p in raw["pts"] if len(p)]                                 # a raw graph (the reference d) may have vertices farther apart than 0.5 m
    P = np.concatenate(dp) if dp else np.zeros((0, 2))
    P = P[(np.abs(P) < BOX).all(1)]
    G = ref.pts
    t = np.where(ref.ptier > 0, ref.ptier, 3)
    inbox = (np.abs(G) < BOX).all(1)
    hit = (cKDTree(P).query(G)[0] <= TIER_HIT) if len(P) else np.zeros(len(G), bool)
    rec, prec = np.zeros((3, 4)), np.zeros((4, 2))
    for k in (1, 2, 3):
        m = inbox & (t == k)
        rec[k - 1] = [(m & ref.pvis & hit).sum(), (m & ref.pvis).sum(), (m & hit).sum(), m.sum()]
    if len(P):
        dg, ig = ref.tree.query(P)
        zone = np.where(dg <= TIER_ZONE, t[ig], 0)
        for k in range(4):
            prec[k] = [((zone == k) & (dg <= TIER_HIT)).sum(), (zone == k).sum()]
    return dict(rec=rec, prec=prec)


def tier_numbers(tc, nc=None):
    """the E276 numbers from pooled tier counts `tc` (dict rec, prec) and pooled node counts `nc` (dict tn, zn; may be empty): t<k>_R_1 / t<k>_R_1all (recall of the visible / all GT points of tier k),
    z<k>_P_1 (precision of the delivered points of zone k; z0 = off the lanes), t12_R_1 / t12_P_1 / t12_F1_1 (tiers and zones 1 + 2 pooled; the primary numbers of the E276 card), t<k>_len (visible GT
    metres of tier k), and the node numbers t<k>_node_R / z<k>_node_P / t12_node_F1 (within 2 m, against the GT nodes of this scorer: dense for the E273 target, sparse for NODE_SPARSE)"""
    out = {}
    rec, prec = np.asarray(tc["rec"], float), np.asarray(tc["prec"], float)
    for k in (1, 2, 3):
        r = rec[k - 1]
        out[f"t{k}_R_1"], out[f"t{k}_R_1all"], out[f"t{k}_len"] = _ratio(r[0], r[1], 1.0), _ratio(r[2], r[3], 1.0), 0.5 * float(r[1])
    for k in range(4):
        if k:                                                                               # zone 0 (no GT lane within 2.5 m) can never be "right within 1 m": its precision is identically 0, only the count is reported
            out[f"z{k}_P_1"] = _ratio(prec[k][0], prec[k][1], 1.0)
        out[f"z{k}_n"] = 0.5 * float(prec[k][1])
    r12, p12 = rec[:2, :2].sum(0), prec[1:3].sum(0)
    out["t12_R_1"], out["t12_P_1"] = _ratio(r12[0], r12[1], 1.0), _ratio(p12[0], p12[1], 1.0)
    out["t12_F1_1"] = 2 * out["t12_P_1"] * out["t12_R_1"] / (out["t12_P_1"] + out["t12_R_1"]) if out["t12_P_1"] + out["t12_R_1"] > 0 else 0.0
    if nc and "tn" in nc:
        tn, zn = np.asarray(nc["tn"], float), np.asarray(nc["zn"], float)
        for k in (1, 2, 3):
            out[f"t{k}_node_R"] = _ratio(tn[k - 1][0], tn[k - 1][1], 1.0)
        for k in (1, 2, 3):                                                                 # zone 0 (off the lanes) has no matched node by construction: only its count and false nodes are reported
            out[f"z{k}_node_P"] = _ratio(zn[k][0], zn[k][1], 1.0)
        pr, rc = _ratio(zn[1][0] + zn[2][0], zn[1][1] + zn[2][1], 1.0), _ratio(tn[0][0] + tn[1][0], tn[0][1] + tn[1][1], 1.0)
        out["t12_node_P"], out["t12_node_R"] = pr, rc
        out["t12_node_F1"] = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0
        for k in range(4):                                                                  # delivered nodes per zone and how many of them have no GT node within 2 m (prediction 5 of the card)
            out[f"z{k}_node_n"], out[f"z{k}_node_false"] = float(zn[k][1]), float(zn[k][1] - zn[k][0])
        if "evt" in nc:                                                                      # recall of the GT event nodes (demanded ones) per tier, within the event tolerance (prediction 6)
            evt = np.asarray(nc["evt"], float)
            for f, name in enumerate(QT.FLAGS):
                for k in (1, 2, 3):
                    out[f"t{k}_ev_{name}_R"] = float(evt[f][k - 1][0] / evt[f][k - 1][1]) if evt[f][k - 1][1] > 0 else float("nan")
                    out[f"t{k}_ev_{name}_n"] = float(evt[f][k - 1][1])
    return out


# ------------------------------------------------------------------------------------------------------------------------------ assignment
def assign_counts(pred, g, thr_e=0.5):
    """None when the prediction carries no assignment"""
    a = pred.get("assign")
    if a is None:
        return None
    ref = _ref(g)
    nd = pred["nodes"]
    e = np.asarray(nd["e"], float).reshape(-1)
    keep = np.flatnonzero(e >= thr_e)
    w = e[keep]
    P = np.asarray(a["P"], float).reshape(len(e), len(a["map_ids"]) + 1)[keep]
    R = np.asarray(a["road"], float).reshape(len(e), len(a["road_ids"]) + 1)[keep] if a.get("road") is not None else None
    d, tied, _ = ref.near(np.asarray(nd["pos"], float).reshape(-1, 2)[keep])
    K = len(a["map_ids"])
    id2k = {}                                                                          # a map id may stand in several columns (the parts of a piece that leaves and re-enters the window)
    for k, i in enumerate(a["map_ids"]):
        id2k.setdefault(i, set()).add(k)
    id2r = {i: k for k, i in enumerate(a.get("road_ids", []))}
    amax = P.argmax(1)
    lane = np.zeros(2)                                                                 # e-weighted (right, nodes within 1 m of a GT lane)
    road = np.zeros(3)                                                                 # the same at road level; [2] = 1 when the prediction has a road assignment
    road[2] = R is not None
    dust = np.zeros(3)                                                                 # e-weighted (dustbin delivered and right, dustbin delivered, dustbin right)
    by_gt = {}
    for n in range(len(keep)):
        if LANE_NEAR < d[n] <= DUST_FAR:
            continue
        true_dust = True                                                               # farther than DUST_FAR from every GT lane
        if d[n] <= LANE_NEAR:
            accept = set().union(*(id2k.get(ref.ids[i], {K}) for i in tied[n]))        # the map pieces of the GT lanes at this point; a piece the map lacks = the dustbin
            true_dust = accept == {K}
            lane += w[n] * np.array([amax[n] in accept, 1.0])
            if R is not None:
                road += w[n] * np.array([R[n].argmax() in {id2r.get(ref.roads[i], len(id2r)) for i in tied[n]}, 1.0, 0.0])
            if len(tied[n]) == 1:
                by_gt.setdefault(int(tied[n][0]), []).append((int(amax[n]), w[n]))
        dust += w[n] * np.array([true_dust and amax[n] == K, amax[n] == K, true_dust])
    pur = np.zeros(2)                                                                  # (sum of the majority shares, GT lanes with a node)
    for v in by_gt.values():
        votes = {}
        for k, x in v:
            votes[k] = votes.get(k, 0.0) + x
        pur += [max(votes.values()) / max(sum(votes.values()), 1e-12), 1.0]
    return dict(lane=lane, road=road, dust=dust, purity=pur)


def assign_numbers(c):
    """lane_acc / road_acc / purity: nothing delivered near a GT lane is nothing right (0.0); road_acc is NaN when there is no road assignment"""
    out = dict(lane_acc=_ratio(c["lane"][0], c["lane"][1], 1.0), road_acc=_ratio(c["road"][0], c["road"][1], 1.0) if c["road"][2] > 0 else float("nan"))
    out["dust_P"], out["dust_R"], _ = _prf(*c["dust"])
    out["purity"] = _ratio(c["purity"][0], c["purity"][1], 1.0)
    return out


def assign_metrics(pred, g, thr_e=0.5):
    c = assign_counts(pred, g, thr_e)
    return {} if c is None else assign_numbers(pool_counts([c]))


# ------------------------------------------------------------------------------------------------------------------------------ graph
def section_counts(meta, ref):
    """pieces_per_gt_piece / id switches from the kept nodes (meta['nodes'] of readout) and the lane GT: (distinct sections, switches, GT pieces with a node, GT lane length)"""
    nodes = meta["nodes"]
    d, tied, pj = ref.near(nodes["pos"])
    by_gt = {}
    for n in range(len(d)):
        if d[n] <= SEC_NEAR and len(tied[n]) == 1:
            by_gt.setdefault(int(tied[n][0]), []).append((int(pj[n]), nodes["section"][n]))
    distinct = switches = 0
    for v in by_gt.values():
        s = [x for _, x in sorted(v, key=lambda t: t[0])]
        distinct += len(set(s))
        switches += sum(a != b for a, b in zip(s[:-1], s[1:]))
    return np.array([distinct, switches, len(by_gt), ref.length], float)


def graph_counts(graph, frame):
    """the additive counts of one frame: rec (bootstrap.frame_records: Q, S, J, F1, lanes found ... as `lane_suite.LcrMetric`), topo (topo_metrics.frame_record: as `LcrTopoMetric`),
    lane (lane_task topo_counts: junctions, as `LaneTopoMetric`), dp (Douglas-Peucker vertices and length of the polylines, for vert100_dp), sec (section counts; only for a readout graph
    and a frame with a stored lane GT)"""
    raw = to_raw(graph)
    gt = getattr(frame, "gt", frame)
    c = dict(rec=SB.frame_records([raw], [gt])[0], topo=TM.frame_record(raw, gt), dp=np.array(SB._vstats(raw, False)),
             lane=MS.topo_counts(LaneGraph.of(raw), LaneGraph.of({"pts": gt["pts"], "edges": gt.get("edges")}), box=CFG.box, **CFG.topo))
    meta = getattr(graph, "meta", None)
    if meta and "nodes" in meta and hasattr(frame, "ref"):
        try:
            c["sec"] = section_counts(meta, frame.ref())
        except FileNotFoundError:
            pass
    if hasattr(frame, "ref"):                                                          # E276: the tier counts (graphs of any origin, d included; frames without l4 labels: none)
        try:
            tc = tier_counts(raw, frame.ref())
        except FileNotFoundError:
            tc = None
        if tc is not None:
            c["tier"] = tc
    return c


def graph_numbers(cs):
    out = SB.pooled([c["rec"] for c in cs])
    out.update(TM.pooled([c["topo"] for c in cs]))
    lane = {k: sum(c["lane"][k] for c in cs) for k in ("marb", "junc")}
    out.update({k: float(v) for k, v in MS.topo_numbers(lane).items() if k.startswith("junction")})
    dp = np.sum([c["dp"] for c in cs], 0)
    out["vert100_dp"] = float(100.0 * dp[0] / max(dp[1], 1e-9))              # the vertices the delivered polylines need at 0.3 m (the project's measure for graphs without node ids)
    sec = [c["sec"] for c in cs if "sec" in c]
    if sec:
        s = np.sum(sec, 0)
        out["pieces_per_gt_piece"] = float(s[0] / s[2]) if s[2] > 0 else float("nan")
        out["id_switches_per_100m"] = float(100.0 * s[1] / s[3]) if s[3] > 0 else float("nan")
    return out


def graph_metrics(graph, frame):
    """the numbers of one frame's graph (a LaneGraph from readout, or a raw graph dict such as the reference d); `frame` = a Frame, or the old graph GT dict itself"""
    return graph_numbers([graph_counts(graph, frame)])


# ------------------------------------------------------------------------------------------------------------------------------ frames
def score_frames(preds, frames, thr_e=0.5, thr_link=0.5, records=False):
    """pooled numbers over frames: preds[i] is a prediction dict (qformat) or a delivered raw graph dict (pts, edges: then only the graph numbers). Frames whose lane GT is not stored
    count for the graph numbers but not for the node / assignment numbers (`lane_missing`). records=True also returns the per-frame counts (for `paired`)"""
    rec = dict(graph=[], node=[], assign=[])
    miss = 0
    for p, f in zip(preds, frames):
        cached = getattr(f, "_ref", None) is not None                        # a frame loaded before keeps its lane GT; one loaded here lets it go (873 of them are large)
        if "pts" in p:
            rec["graph"].append(graph_counts(p, f))
            if not cached:                                                       # graph_counts now loads the lane GT for the tier counts: let it go again
                f._lane = f._ref = None
            continue
        graph, _ = readout(p, thr_e, thr_link)
        rec["graph"].append(graph_counts(graph, f))
        try:
            ref = f.ref()
        except FileNotFoundError:
            miss += 1
            continue
        rec["node"].append(node_counts(p, ref, thr_e, thr_link))
        a = assign_counts(p, ref, thr_e)
        if a is not None:
            rec["assign"].append(a)
        if not cached:
            f._lane = f._ref = None
    tg = [c["tier"] for c in rec["graph"] if c.get("tier") is not None]
    out = dict(frames=len(frames), lane_missing=miss, graph=graph_numbers(rec["graph"]) if rec["graph"] else {},
               nodes=node_numbers(pool_counts(rec["node"])) if rec["node"] else {}, assign=assign_numbers(pool_counts(rec["assign"])) if rec["assign"] else {},
               tier=tier_numbers(pool_counts(tg), pool_counts([{k: c[k] for k in ("tn", "zn", "evt")} for c in rec["node"] if "tn" in c])) if tg else {})
    return (out, rec) if records else out


def paired(rec_a, rec_b, frames, B=1000, seed=0):
    """a - b per graph metric with a 95 % interval over resampled scenes (the project's paired scene bootstrap: bootstrap.paired + topo_metrics.paired, as `evaluate.py --compare`);
    rec_a / rec_b = the `records` of score_frames for the same frames"""
    sc = np.array([f.scene for f in frames])
    out = SB.paired([c["rec"] for c in rec_a["graph"]], [c["rec"] for c in rec_b["graph"]], sc, B=B, seed=seed)
    out.update(TM.paired([c["topo"] for c in rec_a["graph"]], [c["topo"] for c in rec_b["graph"]], sc, B=B, seed=seed))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ toy + self-test
def toy_lane_gt():
    """a lane GT graph of one lane that splits into a straight piece and a curved connector (as the qtargets self-test), and the old-GT style graph made from it"""
    ys = lambda a, b: np.stack([np.zeros(int((b - a) / 0.5) + 1), np.arange(a, b + 1e-9, 0.5)], 1)
    A = QT._piece("A", ys(-40, 0), succ=["B", "C"], start_open=True)
    B = QT._piece("B", ys(0, 40), end_open=True, pred=["A"])
    t = np.linspace(0, 1, 61)
    C = QT._piece("C", np.stack([20 * t ** 2, 30 * t], 1), kind="connector", pred=["A"], end_open=True, inter=True)
    ev = [dict(type="window", io="in", pos=[0.0, -40.0], **{"in": [], "out": ["A"]}, dirs_in=[], dirs_out=[[0, 1]], partial=False),
          dict(type="split", pos=[0.0, 0.0], **{"in": ["A"], "out": ["B", "C"]}, dirs_in=[[0, 1]], dirs_out=[[0, 1], [0, 1]], partial=False),
          dict(type="stop", pos=[0.0, 0.0], on="A", s=40.0, n_succ=2, **{"in": ["A"], "out": ["B", "C"]}, dirs_in=[[0, 1]], dirs_out=[[0, 1], [0, 1]], partial=False),
          dict(type="window", io="out", pos=[0.0, 40.0], **{"in": ["B"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False),
          dict(type="window", io="out", pos=[float(C["pts"][-1, 0]), float(C["pts"][-1, 1])], **{"in": ["C"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False)]
    g = dict(pieces=[A, B, C], events=ev, lines=dict(lane_divider=[], road_divider=[], boundary=[]))
    old = dict(pts=[np.asarray(p["pts"]) for p in g["pieces"]], edges=[(0, 1), (0, 2)])
    return g, old


def _selftest():
    import qformat
    g, old = toy_lane_gt()
    pred = qformat.oracle_prediction(g)
    graph, bk = readout(pred)
    m = graph_metrics(graph, old)
    assert m["F1@1.0"] > 0.99 and m["topo_F1"] > 0.95 and m["lanes_found"] == 1.0 and m["split6_F1"] == 1.0 and m["junction_F1"] == 1.0, m
    n = node_metrics(pred, g)
    assert n["node_F1"] == 1.0 and n["link_short_P"] == 1.0 and n["link_short_R"] == 1.0 and n["pos_err_med"] < 1e-9 and n["degree_acc"] == 1.0, n
    a = assign_metrics(pred, g)
    assert a["lane_acc"] == 1.0 and a["purity"] == 1.0 and a["road_acc"] == 1.0, a
    # a shift of 1.5 m of every node (and curve) is seen; an empty prediction scores 0
    sh = qformat.oracle_prediction(g)
    sh["nodes"]["pos"] = sh["nodes"]["pos"] + [1.5, 0.0]
    sh["links"]["curves"] = [c + [1.5, 0.0] for c in sh["links"]["curves"]]
    assert graph_metrics(readout(sh)[0], old)["F1@1.0"] < 0.2 and abs(node_metrics(sh, g)["pos_err_med"] - 1.5) < 1e-9
    sh["nodes"]["e"] = np.zeros_like(sh["nodes"]["e"])
    z, zn = graph_metrics(readout(sh)[0], old), node_metrics(sh, g)
    assert z["F1@1.0"] == 0.0 and z["topo_F1"] == 0.0 and zn["node_F1"] == 0.0 and assign_metrics(sh, g)["lane_acc"] == 0.0
    print("qgraph_eval selftest ok: oracle F1@1.0", round(m["F1@1.0"], 4), "topo F1", round(m["topo_F1"], 4), "pieces", len(graph.lines), "edges", len(graph.edges))


if __name__ == "__main__":
    _selftest()
