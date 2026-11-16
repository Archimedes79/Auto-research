"""E355 step 0 (card experiments/E355_one-token-set-design, section 0): variants of B's node + link code, replayed on the lane GT and scored like the arms of E346 / E356. Scratch next to `e346_lib.py`;
it only READS the frozen lane files (E343 digest): the lane GT is copied where a variant changes it, `qtargets` / `qformat` are called as they are.

    pred = oracle_variant(lane, "m25")              # qformat's oracle prediction of the arm on the grid in force (e346_lib.cell_size); pred["merge"] = the merge counts (None for an arm without a merge)
    w = wanted_lost(lane, "v25")                    # the cell contest against the node set the arm wants: the entries of WBL (connectors, with a wanted-but-lost node, ..., candidates, dropped)
    j = junction_nodes(lane, nt)                    # kept nodes on the connector segments of the lane GT

Arms (`VARIANTS`; the grid of an arm is in e346_lib.GRID): s = suppression ON + the sparse node target of E276 (the references of v); v = suppression OFF (nodes inside the junction) + the sparse
target; m = B's dense code + a MERGE of the node table (radius = the lattice); c = B's dense code + node positions at the cell centres. The arms of DENSE (E346 l2 / l1, E356 l25 / l31) are the dense
code of B itself; any other arm name raises.
"""
import numpy as np

import qformat
import qtargets as QT

SPARSE = (0.3, 7.5)                                # E276: eps (m), longest chord (m)
DEFAULT = dict(sparse=None, suppress=True, merge=None, centre=False)
DENSE = ("l2", "l1", "l25", "l31")                 # the arms of B's dense code (E346 / E356): the only arms that are not in VARIANTS and have no variant
VARIANTS = {**{f"s{g}": dict(sparse=SPARSE) for g in (20, 25, 31)}, **{f"v{g}": dict(sparse=SPARSE, suppress=False) for g in (20, 25, 31)},
            "m20": dict(merge=2.0), "m25": dict(merge=2.5), "c25": dict(centre=True)}
MERGE = ("ev_into_ev", "norm_into_ev", "norm_into_norm", "unlinked", "unlinked_kin", "selfloops", "dups", "nodes_before", "events_before")     # the counts of one merge, in this order
NODE_KEYS = ("cell", "pos", "dirv", "flags", "piece", "j", "ev", "sstart", "tier", "vis", "gid")                                           # the per-node arrays of a node table
WBL = ("connectors", "hit", "inter", "inter_hit", "wanted", "lost", "hit_excl_adj", "inter_hit_excl_adj", "lost_excl_adj", "cand", "dropped")   # the entries of `wanted_lost`, in this order


def variant(arm):
    """the settings of an arm: DEFAULT (B's dense code) for the arms of DENSE, VARIANTS[arm] over it for an E355 arm; any other name is an error (a typo must not fall back to the dense code)"""
    if arm not in VARIANTS and arm not in DENSE:
        raise ValueError(f"unknown decoded arm {arm!r}: {sorted(DENSE)} or {sorted(VARIANTS)}")
    return {**DEFAULT, **VARIANTS.get(arm, {})}


# ------------------------------------------------------------------------------------------------------------------------------ the variants
def suppress_off(lane):
    """a copy of the lane GT with `in_intersection = False` on every connector segment (`node_table` then keeps the connector points inside intersections as candidates; the frozen file is not
    touched). The point arrays are shared, only the segment dicts are new: `node_table` reads the lane GT and writes nothing"""
    pieces = [dict(p, segments=[dict(s, in_intersection=False) if s["kind"] == "connector" else s for s in p["segments"]]) for p in lane["pieces"]]
    return dict(lane, pieces=pieces)


def piece_relations(lane):
    """-> (adjacent, kin): sets of piece-index pairs (both orders) of the lane GT: adjacent = one is a successor of the other (`succ`, as the link walk of `node_table` reads it); kin = not adjacent
    but a common successor or a common predecessor (the branches of a split, the arms of a merge)"""
    idx = {p["id"]: i for i, p in enumerate(lane["pieces"])}
    succ = [{idx[q] for q in p["succ"] if q in idx} for p in lane["pieces"]]
    adj = {(i, j) for i, s in enumerate(succ) for j in s}
    adj |= {(j, i) for i, j in adj}
    pred = {}
    for i, s in enumerate(succ):
        for j in s:
            pred.setdefault(j, set()).add(i)
    kin = {(a, b) for grp in [*succ, *pred.values()] for a in grp for b in grp if a != b and (a, b) not in adj}
    return adj, kin


def prediction_of(g, nt):
    """qformat.oracle_prediction made from the node table `nt` of the lane GT `g`: the same lines (tested: equal to qformat.oracle_prediction on an unprocessed table), so that a post-processed
    table is decoded and assigned exactly as the oracle of B is"""
    M, L = len(nt["pos"]), len(nt["src"])
    nodes = dict(pos=nt["pos"].copy(), dirv=nt["dirv"].copy(), e=np.ones(M), flags=nt["flags"].astype(float), s=nt["sstart"].astype(float), cell=nt["cell"].copy(), section=nt["piece"].astype(int).copy())
    links = dict(src=nt["src"].copy(), dst=nt["dst"].copy(), p=np.ones(L), long=nt["long"].copy(), a=np.full(L, 1 / 3), b=np.full(L, 1 / 3), curves=QT.decode(nt))
    pred = dict(nodes=nodes, links=links, assign=None, maps=None, table=nt)
    ids = [p["piece"] for p in g["pieces"]]
    P = np.zeros((M, len(ids) + 1))
    P[np.arange(M), nt["piece"]] = 1.0
    roads = sorted({p["road"] for p in g["pieces"]})
    R = np.zeros((M, len(roads) + 1))
    for m in range(M):
        R[m, roads.index(g["pieces"][nt["piece"][m]]["road"])] = 1.0
    pred["assign"] = dict(P=P, map_ids=ids, road=R, road_ids=roads)
    return pred


def merge_nodes(nt, radius, adjacent=frozenset(), kin=frozenset()):
    """E355 merge of a node table (post-processing; the dense code of B, suppression ON): the nodes in priority order (event nodes first, among equals the lower cell index); a node is kept when no
    kept node lies within `radius` m (Euclidean, true positions, strictly closer = within), else it is merged into the NEAREST kept node; the survivor keeps its own position and type. Links are
    re-pointed to the survivors, self-loops and duplicates (the first link of a pair stays) are dropped, `long` is re-evaluated by the chord (> SHORT), the path of global point ids of a link stays
    (only its ends change: `decode` fits the handle scales of the long links against it). -> (new table, counts in the order of MERGE: event into event, normal into event, normal into normal,
    merges between pieces that are not adjacent / of those between kin, self-loops, duplicates, nodes and event nodes before)"""
    M, pos, ev = len(nt["pos"]), nt["pos"], nt["ev"]
    order = np.lexsort((nt["cell"][:, 0] * QT.N + nt["cell"][:, 1], ~ev))                    # the last key is the first: events, then the cell index
    rep, kept, c = np.arange(M), [], dict.fromkeys(MERGE, 0)
    for m in order:
        d = np.linalg.norm(pos[kept] - pos[m], axis=1) if kept else np.zeros(0)
        if not len(d) or d.min() >= radius:
            kept.append(m)
            continue
        s = kept[int(d.argmin())]
        rep[m] = s
        assert ev[s] or not ev[m], "an event node is merged into a normal node"             # the events came first
        c["ev_into_ev" if ev[m] else "norm_into_ev" if ev[s] else "norm_into_norm"] += 1
        pm, ps = int(nt["piece"][m]), int(nt["piece"][s])
        if pm != ps and (pm, ps) not in adjacent:
            c["unlinked"] += 1
            c["unlinked_kin"] += (pm, ps) in kin
    src, dst = rep[nt["src"]], rep[nt["dst"]]
    take, seen = [], set()
    for l in range(len(src)):
        if src[l] == dst[l]:
            c["selfloops"] += 1
        elif (int(src[l]), int(dst[l])) in seen:
            c["dups"] += 1
        else:
            seen.add((int(src[l]), int(dst[l])))
            take.append(l)
    alive = np.array(sorted(kept), int)
    new = np.full(M, -1, int)
    new[alive] = np.arange(len(alive))
    out = dict(nt, **{k: nt[k][alive] for k in NODE_KEYS})
    out["src"], out["dst"], out["path"] = new[src[take]], new[dst[take]], [nt["path"][l] for l in take]
    out["dist"] = np.linalg.norm(out["pos"][out["src"]] - out["pos"][out["dst"]], axis=1)
    out["long"] = out["dist"] > QT.SHORT
    out["stats"] = dict(nt["stats"], nodes=len(alive), events=int(out["ev"].sum()), links=len(take), long=int(out["long"].sum()), merged=M - len(alive))
    c["nodes_before"], c["events_before"] = M, int(ev.sum())
    return out, np.array([c[k] for k in MERGE], float)


def at_cell_centres(nt):
    """E355 c arm: every node position set to the centre of its cell (the minimum distance between nodes is then the lattice); `long` re-evaluated by the new chord, the paths stay"""
    out = dict(nt, pos=QT.centre_of(nt["cell"]))
    out["dist"] = np.linalg.norm(out["pos"][nt["src"]] - out["pos"][nt["dst"]], axis=1)
    out["long"] = out["dist"] > QT.SHORT
    out["stats"] = dict(nt["stats"], long=int(out["long"].sum()))
    return out


def oracle_variant(lane, arm):
    """the oracle prediction of `arm` from the lane GT on the grid in force: suppression, sparse target, then the merge or the cell-centre positions of the arm. An arm without a merge goes through
    `qformat.oracle_prediction` itself"""
    v = variant(arm)
    g = lane if v["suppress"] else suppress_off(lane)
    if v["merge"] is None and not v["centre"]:
        return qformat.oracle_prediction(g, sparse=v["sparse"])
    nt, mg = QT.node_table(g, sparse=v["sparse"]), None
    if v["merge"] is not None:
        nt, mg = merge_nodes(nt, v["merge"], *piece_relations(lane))
    if v["centre"]:
        nt = at_cell_centres(nt)
    pred = prediction_of(g, nt)
    pred["merge"] = mg
    return pred


# ------------------------------------------------------------------------------------------------------------------------------ counts of the variants
def connector_segments(lane, off):
    """(S, 3) int: first and last global point id and 'inside an intersection' (the lane GT's own flag) of every connector segment of the lane GT"""
    return np.array([(off[i] + s["i0"], off[i] + s["i1"], bool(s["in_intersection"])) for i, p in enumerate(lane["pieces"]) for s in p["segments"] if s["kind"] == "connector"], int).reshape(-1, 3)


def n_between(ids, lo, hi):
    """how many of the sorted ids lie in [lo, hi] (arrays lo, hi)"""
    return np.searchsorted(ids, hi, "right") - np.searchsorted(ids, lo, "left")


def junction_nodes(lane, nt):
    """[kept nodes on connector segments of the lane GT, those segments, kept nodes on connector segments inside an intersection, those segments] of one frame (a node lies on the segment that holds
    its point `gid`; the flag is the lane GT's own, so the arms with the suppression off are counted against the same connectors)"""
    seg = connector_segments(lane, nt["off"])
    n = n_between(np.sort(nt["gid"]), seg[:, 0], seg[:, 1])
    inter = seg[:, 2].astype(bool)
    return np.array([n.sum(), len(seg), n[inter].sum(), inter.sum()], float)


def candidates(nt):
    """global point ids of the normal-node candidates of `node_table` before the cell contest: per piece and cell the point nearest to the cell centre (the first of equals) among the points that
    are neither suppressed nor an event node"""
    free = ~nt["supp"]
    free[nt["gid"][nt["ev"]]] = False
    gid = np.flatnonzero(free)
    c = QT.cell_of(nt["pts"][gid])
    cell = c[:, 0] * QT.N + c[:, 1]
    order = np.lexsort((gid, np.linalg.norm(nt["pts"][gid] - QT.centre_of(c), axis=1), cell, nt["pid"][gid]))
    key = np.stack([nt["pid"][gid], cell], 1)[order]
    first = np.ones(len(order), bool)
    first[1:] = (key[1:] != key[:-1]).any(1)
    return gid[order][first]


def wanted_lost(lane, arm, nt=None):
    """E355 collision clause: `node_table` runs the cell contest (one node per cell, the loser's candidate is dropped) BEFORE the sparse pruning, so the quantity is 'wanted-but-lost'. -> the entries of
    WBL for one frame: connector segments, with a wanted-but-lost node, connector segments inside an intersection, with one, wanted candidates, wanted and lost; then the same three counts of the
    losses to a piece that is NOT adjacent (a contest lost to the previous / next piece in a shared cell is a benign join: `_excl_adj`); then the candidates and the dropped candidates of the dense
    contest. A candidate is WANTED when the arm's own node set contains it: all candidates (dense arms) or `_sparse_keep` on the piece's full candidate list plus its event nodes (sparse arms); it is
    LOST when its cell is held by a node of ANOTHER piece. A connector segment counts when a lost wanted candidate lies on it. `nt` = the dense table of the arm's lane (computed here when None)"""
    v = variant(arm)
    if nt is None:
        nt = QT.node_table(lane if v["suppress"] else suppress_off(lane))
    cand = candidates(nt)
    assert len(cand) == nt["stats"]["cand"] and np.isin(nt["gid"][~nt["ev"]], cand).all(), "the candidate list is not the one of node_table"
    if v["sparse"] is None:
        wanted = cand
    else:
        nodes = {**{int(g): dict(ev=False) for g in cand}, **{int(g): dict(ev=True) for g in nt["gid"][nt["ev"]]}}
        wanted = np.array(sorted(QT._sparse_keep(nodes, nt["pts"], nt["off"], float(v["sparse"][0]), float(v["sparse"][1]))), int)
    held = {(int(a), int(b)): int(p) for (a, b), p in zip(nt["cell"], nt["piece"])}               # one node per cell: the piece that holds it
    holder = np.array([held[(int(a), int(b))] for a, b in QT.cell_of(nt["pts"][wanted])], int)
    own = nt["pid"][wanted]
    adjacent = piece_relations(lane)[0]
    lost = holder != own
    lost_x = lost & np.array([(int(o), int(h)) not in adjacent for o, h in zip(own, holder)], bool)
    seg = connector_segments(lane, nt["off"])
    inter = seg[:, 2].astype(bool)
    hit, hit_x = (n_between(np.sort(wanted[l]), seg[:, 0], seg[:, 1]) > 0 for l in (lost, lost_x))
    return np.array([len(seg), hit.sum(), inter.sum(), (hit & inter).sum(), len(wanted), lost.sum(), hit_x.sum(), (hit_x & inter).sum(), lost_x.sum(), len(cand), nt["stats"]["dropped"]], float)
