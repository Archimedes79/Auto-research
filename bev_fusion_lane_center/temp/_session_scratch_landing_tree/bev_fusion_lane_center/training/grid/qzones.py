"""E282: the ZONES of the sparse node target. A zone is a stretch of lane in which the head may place ONE node anywhere (the zone-matching existence loss, `qgraph.node_loss_zone`), instead of at the one
cell that the Douglas-Peucker vertex happens to fall in (E276 arm s: the per-cell loss cannot decide that cell, the existence probability smears). The loss itself is `qgraph.zone_targets` (the winners)
+ `qgraph.node_loss` with `CFG["node_loss"] = "zone"`.

    zs = zones(g, sparse=(0.3, 7.5))        # g = one lane GT graph (dict)
    zm = zone_maps(zs)                       # the 50 x 50 maps of the batch

The sparse nodes of `qtargets.node_table(g, sparse=...)` are a SUBSET of the dense nodes (a node in every 2 m cell, E273): the ANCHORS (event nodes, the first and the last dense node of a piece) and the
Douglas-Peucker VERTICES in between. Every sparse node is a zone:
  fixed zone   an anchor: its own cell only (the cell is fixed, the per-cell loss as before);
  free zone    a vertex: the cells of the dense nodes of its piece that are nearest to it in point index (ties: the earlier sparse node); ANY of these cells may carry the node.
A dense node that is nearest to an ANCHOR but is not the anchor itself belongs to no zone (a lane cell where no node is demanded: a negative, as in the per-cell sparse target).
`zones` returns dict(nd = the dense node table, zone_of (M,) int zone of every dense node or -1, kind (Z,) 0 fixed / 1 free, gt_node (Z,) dense index of the zone's sparse node, members list of (k,) dense indices,
n_zones). `zone_maps` returns dict(zone (50, 50) int32 zone id of a cell or -1, zone_kind (50, 50) int8 -1 / 0 / 1, zone_gt (50, 50) float32 1 at the cell of a zone's sparse node, exists_dense / offset_dense /
dirv_dense = the DENSE targets of `qtargets.grid_targets`: the offset and direction of a winner other than the GT cell come from its own dense node).
Run the self-test: python qzones.py     (real frames: `test_zones.py`)
"""
import numpy as np

import qtargets as QT


def zones(g, sparse=(0.3, 7.5)):
    nd = QT.node_table(g)
    ns = QT.node_table(g, sparse=sparse)
    gd, gs = nd["gid"], ns["gid"]
    dense_of = {int(x): m for m, x in enumerate(gd)}
    assert all(int(x) in dense_of for x in gs), "the sparse nodes must be dense nodes"
    piece, off = nd["piece"], nd["off"]
    anchors = {int(x) for x in gd[nd["ev"]]}
    for p in np.unique(piece):
        ids = np.flatnonzero(piece == p)
        anchors.add(int(gd[ids[0]]))
        anchors.add(int(gd[ids[-1]]))
    zid, gt_node, kind, members = {}, [], [], []
    for s in sorted(int(x) for x in gs):
        zid[s] = len(gt_node)
        gt_node.append(dense_of[s])
        kind.append(0 if s in anchors else 1)
        members.append([])
    zone_of = -np.ones(len(gd), int)
    for p in np.unique(piece):
        sp = sorted(s for s in zid if off[p] <= s < off[p + 1])
        if not sp:
            continue
        for m in np.flatnonzero(piece == p):
            gid = int(gd[m])
            s = min(sp, key=lambda q: (abs(gid - q), q))
            if gid != s and s in anchors:
                continue
            zone_of[m] = zid[s]
            members[zid[s]].append(int(m))
    return dict(nd=nd, zone_of=zone_of, kind=np.array(kind, int), gt_node=np.array(gt_node, int), members=[np.array(v, int) for v in members], n_zones=len(gt_node))


def zone_maps(zs, n=QT.N):
    nd = zs["nd"]
    cell = nd["cell"]
    zone = -np.ones((n, n), np.int32)
    zkind = -np.ones((n, n), np.int8)
    zgt = np.zeros((n, n), np.float32)
    for z, mem in enumerate(zs["members"]):
        for m in mem:
            zone[cell[m][0], cell[m][1]] = z
            zkind[cell[m][0], cell[m][1]] = zs["kind"][z]
        zgt[cell[zs["gt_node"][z]][0], cell[zs["gt_node"][z]][1]] = 1.0
    t = QT.grid_targets(nd, n)
    return dict(zone=zone, zone_kind=zkind, zone_gt=zgt, exists_dense=t["exists"], offset_dense=t["offset"], dirv_dense=t["dirv"])


def _selftest():
    ys = lambda a, b: np.stack([np.zeros(int((b - a) / 0.5) + 1), np.arange(a, b + 1e-9, 0.5)], 1)
    A = QT._piece("A", ys(-40, 40), start_open=True, end_open=True)
    ev = [dict(type="window", io="in", pos=[0.0, -40.0], **{"in": [], "out": ["A"]}, dirs_in=[], dirs_out=[[0, 1]], partial=False),
          dict(type="window", io="out", pos=[0.0, 40.0], **{"in": ["A"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False)]
    g = dict(pieces=[A], events=ev, lines=dict(lane_divider=[], road_divider=[], boundary=[]))
    zs = zones(g, (0.3, 7.5))
    nd = zs["nd"]
    Z = zs["n_zones"]
    assert Z == len(QT.node_table(g, sparse=(0.3, 7.5))["gid"]), "one zone per sparse node"
    assert (zs["kind"] == 0).sum() == 2 and (zs["kind"] == 1).sum() >= 8, ("a straight 80 m piece: two anchors and chord vertices", zs["kind"])
    for z in range(Z):
        assert zs["gt_node"][z] in zs["members"][z], "the sparse node is in its own zone"
        assert (zs["zone_of"][zs["members"][z]] == z).all()
        if zs["kind"][z] == 0:
            assert len(zs["members"][z]) == 1, "a fixed zone is one cell"
        else:
            mm = zs["members"][z]
            assert (np.diff(mm) == 1).all(), "a free zone is a connected stretch of the piece"
    assert (zs["zone_of"] >= 0).sum() == sum(len(v) for v in zs["members"]) and len(set(np.concatenate(zs["members"]).tolist())) == (zs["zone_of"] >= 0).sum()
    # the dense nodes in no zone are exactly the non-anchor nodes that are nearer to an anchor than to any free vertex
    gid = nd["gid"]
    sp = sorted(int(gid[v]) for v in zs["gt_node"])
    anc = {int(gid[zs["gt_node"][z]]) for z in range(Z) if zs["kind"][z] == 0}
    for m in range(len(gid)):
        s = min(sp, key=lambda q: (abs(int(gid[m]) - q), q))
        expect_none = (int(gid[m]) != s) and (s in anc)
        assert (zs["zone_of"][m] < 0) == expect_none, (m, s)
    zm = zone_maps(zs)
    assert (zm["zone"] >= 0).sum() == (zs["zone_of"] >= 0).sum() and zm["zone_gt"].sum() == Z
    assert ((zm["zone_kind"] >= 0) == (zm["zone"] >= 0)).all() and zm["exists_dense"].sum() == len(gid)
    for z in range(Z):
        c = nd["cell"][zs["gt_node"][z]]
        assert zm["zone"][c[0], c[1]] == z and zm["zone_gt"][c[0], c[1]] == 1.0
    print("qzones selftest ok: zones", Z, "fixed", int((zs["kind"] == 0).sum()), "free", int((zs["kind"] == 1).sum()), "dense nodes", len(gid), "in no zone", int((zs["zone_of"] < 0).sum()))


if __name__ == "__main__":
    _selftest()
