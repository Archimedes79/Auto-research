"""throw-away: replace wanted_lost in e355_lib.py (text edit, no run of the tool)"""
import io
p = "C:/Auto-research/bev_fusion_lane_center/temp/e346/e355_lib.py"
s = io.open(p, encoding="utf-8").read()
i = s.index("def wanted_lost(lane, arm, nt=None):")
new = '''def wanted_lost(lane, arm, nt=None):
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
'''
io.open(p, "w", encoding="utf-8", newline="\n").write(s[:i] + new)
print("ok", len(s[:i] + new))
