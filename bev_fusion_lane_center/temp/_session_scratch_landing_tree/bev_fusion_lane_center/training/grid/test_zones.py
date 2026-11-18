"""E282 tests (CPU): the zones of the sparse node target (`qzones`), the zone-matching existence loss (`qgraph.zone_targets`, `qgraph.node_loss` with CFG node_loss = "zone") and the data path.
Run: python test_zones.py   (`_py14.sh test_zones.py`; the last test reads real frames of the lane GT store and is skipped with a message when none is stored)"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for sub in ("training/grid", "training/data", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph as QG  # noqa: E402
import qnodes as QN  # noqa: E402
import qtargets as QT  # noqa: E402
import qzones as QZ  # noqa: E402
from test_tiers_sparse import toy  # noqa: E402

SPARSE = (0.3, 7.5)


def batch_of(graphs):
    """a batch dict like the loader's, from lane GT graphs (sparse targets + the zone maps), B = len(graphs)"""
    items = []
    for g in graphs:
        t = QT.grid_targets(QT.node_table(g, sparse=SPARSE))
        t.update(QZ.zone_maps(QZ.zones(g, SPARSE)))
        items.append(t)
    return {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in items[0]}


def test_zones_toy():
    g = toy(labelled=False)
    zs = QZ.zones(g, SPARSE)
    nd = zs["nd"]
    sp = QT.node_table(g, sparse=SPARSE)
    assert zs["n_zones"] == len(sp["gid"]), "one zone per sparse node"
    gid = nd["gid"]
    # every sparse node is the GT node of exactly one zone and a member of it; fixed zones are one cell
    assert sorted(int(gid[v]) for v in zs["gt_node"]) == sorted(int(x) for x in sp["gid"])
    for z in range(zs["n_zones"]):
        assert zs["gt_node"][z] in zs["members"][z]
        if zs["kind"][z] == 0:
            assert len(zs["members"][z]) == 1
        else:
            assert (np.diff(zs["members"][z]) == 1).all(), "a free zone is a stretch of consecutive dense nodes of one piece"
            assert len(set(nd["piece"][zs["members"][z]])) == 1
    # a dense node is in at most one zone; the ones in no zone are exactly those nearer to an anchor than to any free vertex
    allm = np.concatenate(zs["members"])
    assert len(allm) == len(set(allm.tolist())) == int((zs["zone_of"] >= 0).sum())
    anchors = {int(gid[zs["gt_node"][z]]) for z in range(zs["n_zones"]) if zs["kind"][z] == 0}
    for p in np.unique(nd["piece"]):
        sps = sorted(int(gid[zs["gt_node"][z]]) for z in range(zs["n_zones"]) if nd["piece"][zs["gt_node"][z]] == p)
        for m in np.flatnonzero(nd["piece"] == p):
            s = min(sps, key=lambda q: (abs(int(gid[m]) - q), q))
            assert (zs["zone_of"][m] < 0) == (int(gid[m]) != s and s in anchors)
    # events and the first / last nodes of a piece are anchors (fixed zones)
    ev = {int(gid[m]) for m in np.flatnonzero(nd["ev"])}
    assert ev <= anchors
    for p in np.unique(nd["piece"]):
        ids = np.flatnonzero(nd["piece"] == p)
        assert int(gid[ids[0]]) in anchors and int(gid[ids[-1]]) in anchors
    # the zone maps: the cell of a zone's sparse node holds the GT flag, the sparse exists map equals the GT flags
    zm = QZ.zone_maps(zs)
    ex_sparse = QT.grid_targets(sp)["exists"]
    assert (zm["zone_gt"] == ex_sparse).all()
    assert (zm["exists_dense"].sum() == len(gid)) and ((zm["zone"] >= 0) == (zm["zone_kind"] >= 0)).all()
    print("test_zones_toy ok: zones", zs["n_zones"], "fixed", int((zs["kind"] == 0).sum()), "free", int((zs["kind"] == 1).sum()))


def test_zone_targets_rules():
    g = toy(labelled=False)
    b = batch_of([g, g])
    torch.manual_seed(0)
    e = torch.randn(2, 50, 50) * 3
    ex, info = QG.zone_targets(e, b, bias=0.0)
    zone = b["zone"].long()
    nz = int(info["n_zones"])
    assert int(ex.sum()) == nz == sum(int(zone[i].max()) + 1 for i in range(2)), "one winner per zone"
    for i in range(2):
        for z in range(int(zone[i].max()) + 1):
            cells = zone[i] == z
            assert ex[i][cells].sum() == 1, "exactly one positive inside a zone"
            assert ex[i][cells].nonzero().shape[0] == 1
            win = (e[i] * 1.0)[cells].argmax()
            flat_idx = cells.flatten().nonzero()[:, 0]
            assert ex[i].flatten()[flat_idx[win]] == 1.0, "the winner is the cell with the highest logit"
    assert float(ex[zone < 0].sum()) == 0.0, "no positive outside the zones"
    # exact ties go to the lowest flat index
    e0 = torch.zeros(2, 50, 50)
    ex0, _ = QG.zone_targets(e0, b, bias=0.0)
    for i in range(2):
        for z in range(int(zone[i].max()) + 1):
            first = (zone[i] == z).flatten().nonzero()[0, 0]
            assert ex0[i].flatten()[first] == 1.0, "a tie goes to the lowest cell index"
    # the bias: a gap below the bias keeps the GT cell, a gap above it does not
    gz = b["zone_gt"]
    z_free = [(i, z) for i in range(1) for z in range(int(zone[i].max()) + 1) if int((zone[i] == z).sum()) >= 2]
    assert z_free, "the toy has a free zone with several cells"
    i, z = z_free[0]
    cells = (zone[i] == z)
    gt_cell = (cells & (gz[i] > 0.5)).flatten().nonzero()[0, 0]
    other = [c for c in cells.flatten().nonzero()[:, 0].tolist() if c != int(gt_cell)][0]
    for gap, expect_gt in ((0.04, True), (0.06, False)):
        ee = torch.zeros(2, 50, 50)
        ee.view(2, -1)[i, other] = gap
        exb, inf = QG.zone_targets(ee, b, bias=0.05)
        assert (exb.view(2, -1)[i, int(gt_cell)] == 1.0) == expect_gt, (gap, expect_gt)
        assert 0.0 <= float(inf["win_off_gt"]) <= 1.0
    # the share of free zones whose winner is not the GT cell: bias large -> 0, a logit pattern that pushes every winner away -> 1
    exa, inf_a = QG.zone_targets(torch.zeros(2, 50, 50), b, bias=10.0)
    assert float(inf_a["win_off_gt"]) == 0.0
    away = (-1.0) * gz * 5.0
    exb, inf_b = QG.zone_targets(away, b, bias=0.0)
    multi = [(i, z) for i in range(2) for z in range(int(zone[i].max()) + 1) if int((zone[i] == z).sum()) >= 2 and int(b["zone_kind"][i][zone[i] == z][0]) == 1]
    assert multi and float(inf_b["win_off_gt"]) > 0.5
    print("test_zone_targets_rules ok")


def fake_no(batch, e=None):
    B = batch["zone"].shape[0]
    z = lambda *s: torch.zeros(B, *s, requires_grad=True)  # noqa: E731
    return dict(e=(e if e is not None else z(50, 50)), flags=z(8, 50, 50), off=z(2, 50, 50), dir_raw=z(2, 50, 50), s=z(50, 50), uout=z(50, 50), uin=z(50, 50))


def test_node_loss_modes():
    g = toy(labelled=False)
    b = batch_of([g])
    torch.manual_seed(1)
    # 1. the default (CFG node_loss = e273) is the E273 loss: recompute it from its pieces
    QG.CFG.update(node_loss="e273")
    no = fake_no(b, e=(torch.randn(1, 50, 50) * 2).requires_grad_())
    l_ref, _ = QG.node_loss(no, b)
    ex = b["exists"].float()
    m = ex > 0.5
    npos = ex.sum().clamp_min(1.0)
    l_e = QG.focal_bce(no["e"], ex, norm=npos)
    fl = b["flags"].float()
    l_f = (torch.nn.functional.binary_cross_entropy_with_logits(no["flags"], fl, reduction="none", pos_weight=torch.tensor(6.0)) * m[:, None]).sum() / npos
    l_o = ((no["off"] - b["offset"]).abs().sum(1) * m).sum() / npos
    dr = QN.unit(no["dir_raw"])
    l_d = ((1 - (dr * b["dirv"]).sum(1)) * m).sum() / npos
    l_s = (torch.nn.functional.binary_cross_entropy_with_logits(no["s"], b["sstart"].float(), reduction="none", pos_weight=torch.tensor(3.0)) * m).sum() / npos
    l_u = sum((torch.nn.functional.binary_cross_entropy_with_logits(no[k], b[k].float(), reduction="none", pos_weight=torch.tensor(8.0)) * m).sum() / npos for k in ("uout", "uin"))
    l_hand = l_e + 0.5 * l_f + l_o + 0.5 * l_d + 0.5 * l_s + 2.0 * l_u
    assert abs(float(l_ref) - float(l_hand)) < 1e-5, (float(l_ref), float(l_hand))
    # 2. the zone loss: finite, the offset and direction terms use the winner's DENSE node (zero when the head predicts exactly that), the winners carry no gradient
    QG.CFG.update(node_loss="zone", zone_bias=0.05)
    try:
        e = (torch.randn(1, 50, 50) * 2).requires_grad_()
        no = fake_no(b, e=e)
        no["off"] = b["offset_dense"].clone().requires_grad_()
        no["dir_raw"] = b["dirv_dense"].clone().requires_grad_()
        l_z, info = QG.node_loss(no, b)
        assert torch.isfinite(l_z) and info["off"] < 1e-6 and info["dir"] < 1e-5, info
        assert int(info["zones"]) == int(b["zone"].max()) + 1
        l_z.backward()
        assert e.grad is not None and torch.isfinite(e.grad).all() and float(e.grad.abs().sum()) > 0
        # 3. with the SPARSE offset instead (the e273 targets) the same head would be wrong at every winner that is not the GT cell: check the dense map differs there
        winners, _ = QG.zone_targets(e.detach(), b, 0.05)
        off_cells = (winners > 0.5) & (b["zone_gt"] < 0.5)
        assert off_cells.any(), "a random head chooses winners other than the GT cell"
        assert float((b["offset_dense"].abs().sum(1) * off_cells).sum()) > 0
    finally:
        QG.CFG.update(node_loss="e273")
    print("test_node_loss_modes ok")


def test_gradient_reaches_head():
    g = toy(labelled=False)
    b = batch_of([g])
    head = QN.NodeHead(c2=16, c1=8, c05=8, d=16)
    torch.manual_seed(2)
    P = dict(feat2=torch.randn(1, 16, 50, 50), feat1=torch.randn(1, 8, 100, 100), feat05=torch.randn(1, 8, 200, 200), road2=torch.randn(1, 2, 50, 50), centre1=torch.randn(1, 12, 100, 100))
    QG.CFG.update(node_loss="zone", zone_bias=0.05)
    try:
        no = head(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
        loss, info = QG.node_loss(no, b)
        loss.backward()
    finally:
        QG.CFG.update(node_loss="e273")
    gs = [p.grad for p in head.parameters() if p.grad is not None]
    assert gs and all(torch.isfinite(x).all() for x in gs) and sum(float(x.abs().sum()) for x in gs) > 0
    print("test_gradient_reaches_head ok: loss", round(float(loss), 3), {k: round(v, 3) for k, v in info.items()})


def test_real_frames():
    try:
        import lane_gt_data as LD
        ds = LD.QData("val_clean", pixels=False, map_mode="exact", sparse=SPARSE, zone=True)
    except Exception as ex:                                                              # no lane GT store / no feature files: say so, do not fake it
        print("test_real_frames SKIPPED:", type(ex).__name__, str(ex)[:120])
        return
    n = min(6, len(ds))
    items = [ds[i] for i in range(0, len(ds), max(1, len(ds) // n))][:n]
    for it in items:
        assert (it["zone_gt"] == it["exists"]).all(), "the sparse nodes are the GT cells of the zones"
        assert (it["zone"] >= 0).sum() >= it["exists"].sum() and it["exists_dense"].sum() >= it["exists"].sum()
        z = it["zone"].numpy()
        assert z.max() + 1 == int(it["exists"].sum()), "one zone per sparse node"
    b = LD.collate(items[:2])
    for k in LD.ZONE_NAMES:
        assert k in b and b[k].shape[0] == 2, k
    torch.manual_seed(3)
    no = fake_no(b, e=(torch.randn(2, 50, 50) * 2).requires_grad_())                      # the default loss is the E273 loss on REAL frames (recomputed from its pieces)
    assert abs(float(QG.node_loss(no, b)[0]) - float(hand_e273_loss(no, b))) < 1e-4
    QG.CFG.update(node_loss="zone")
    try:
        no = fake_no(b)
        l, info = QG.node_loss(no, b)
        assert torch.isfinite(l)
    finally:
        QG.CFG.update(node_loss="e273")
    print("test_real_frames ok: frames", len(items), "zones", [int(it["exists"].sum()) for it in items], "dense nodes", [int(it["exists_dense"].sum()) for it in items])


def hand_e273_loss(no, b):
    """the E273 node loss recomputed from its pieces (the reference of the bit-for-bit claim)"""
    ex = b["exists"].float()
    m = ex > 0.5
    npos = ex.sum().clamp_min(1.0)
    l_e = QG.focal_bce(no["e"], ex, norm=npos)
    l_f = (torch.nn.functional.binary_cross_entropy_with_logits(no["flags"], b["flags"].float(), reduction="none", pos_weight=torch.tensor(6.0)) * m[:, None]).sum() / npos
    l_o = ((no["off"] - b["offset"]).abs().sum(1) * m).sum() / npos
    l_d = ((1 - (QN.unit(no["dir_raw"]) * b["dirv"]).sum(1)) * m).sum() / npos
    l_s = (torch.nn.functional.binary_cross_entropy_with_logits(no["s"], b["sstart"].float(), reduction="none", pos_weight=torch.tensor(3.0)) * m).sum() / npos
    l_u = sum((torch.nn.functional.binary_cross_entropy_with_logits(no[k], b[k].float(), reduction="none", pos_weight=torch.tensor(8.0)) * m).sum() / npos for k in ("uout", "uin"))
    return l_e + 0.5 * l_f + l_o + 0.5 * l_d + 0.5 * l_s + 2.0 * l_u


def test_loss_masks_and_normalisation():
    """the pieces of the zone loss one by one (critic pass 2026-11-08): normalisation by the number of zones, the offset term only on the winners, win_off_gt over FREE zones only"""
    g = toy(labelled=False)
    b = batch_of([g, g])
    zone, gz, kind = b["zone"].long(), b["zone_gt"], b["zone_kind"]
    QG.CFG.update(node_loss="zone", zone_bias=0.0)
    try:
        e = -5.0 * gz                                                                  # every zone prefers a cell other than the GT cell (where it has one)
        win, info = QG.zone_targets(e, b, 0.0)
        nz = int(win.sum())
        no = fake_no(b, e=e.clone().requires_grad_())
        _, inf = QG.node_loss(no, b)
        assert abs(inf["e"] - float(QG.focal_bce(e, win, norm=float(nz)))) < 1e-5, ("the existence loss is normalised by the number of zones", inf["e"])
        offw = (win > 0.5) & (gz < 0.5)                                                # winners that are not the GT cell
        k = int(offw.sum())
        assert k > 0
        no = fake_no(b, e=e.clone().requires_grad_())
        no["off"] = (b["offset_dense"] + 0.3 * offw[:, None].float()).clone().requires_grad_()
        no["dir_raw"] = b["dirv_dense"].clone().requires_grad_()
        _, inf = QG.node_loss(no, b)
        assert abs(inf["off"] - 0.6 * k / nz) < 1e-5, ("the offset error counts on the winners", inf["off"], 0.6 * k / nz)
        nonwin = (zone >= 0) & (win < 0.5)
        no = fake_no(b, e=e.clone().requires_grad_())
        no["off"] = (b["offset_dense"] + 0.3 * nonwin[:, None].float()).clone().requires_grad_()
        no["dir_raw"] = b["dirv_dense"].clone().requires_grad_()
        _, inf = QG.node_loss(no, b)
        assert inf["off"] < 1e-6, ("an offset error on a non-winner zone cell must not count", inf["off"])
        free_zones = [(i, z) for i in range(2) for z in range(int(zone[i].max()) + 1) if int(kind[i][zone[i] == z][0]) == 1]
        multi = [(i, z) for (i, z) in free_zones if int((zone[i] == z).sum()) >= 2]
        expect = len(multi) / len(free_zones)
        assert 0 < expect < 1 and abs(float(info["win_off_gt"]) - expect) < 1e-6, ("win_off_gt is the share over the FREE zones", float(info["win_off_gt"]), expect)
    finally:
        QG.CFG.update(node_loss="e273", zone_bias=0.05)
    print("test_loss_masks_and_normalisation ok")


def test_tier_switch_refused_in_zone_mode():
    b = batch_of([toy(labelled=False)])
    QG.CFG.update(node_loss="zone", tier_w=(3.0, 1.5, 1.0))
    try:
        try:
            QG.node_loss(fake_no(b), b)
        except AssertionError:
            pass
        else:
            raise AssertionError("the zone loss must refuse the E276 tier switches")
    finally:
        QG.CFG.update(node_loss="e273", tier_w=None)
    print("test_tier_switch_refused_in_zone_mode ok")


def test_zone_assignment_on_real_graphs():
    """the zone of a dense node is the nearest sparse node by POINT INDEX (not by metres): an independent recomputation on real graphs, and the two rules differ somewhere (so the test can tell them apart)"""
    try:
        import lane_gt_data as LD
        ds = LD.QData("val_clean", pixels=False, map_mode="exact", sparse=SPARSE, zone=True)
    except Exception as ex:
        print("test_zone_assignment_on_real_graphs SKIPPED:", type(ex).__name__, str(ex)[:120])
        return
    differ = total = 0
    for i in list(range(0, len(ds), max(1, len(ds) // 12)))[:12]:
        g = ds.lg.load(ds.tokens[int(ds.bd.rows[ds.ks[i]])])
        zs = QZ.zones(g, SPARSE)
        nd = zs["nd"]
        gid, piece, off = nd["gid"], nd["piece"], nd["off"]
        sp = [int(gid[v]) for v in zs["gt_node"]]
        anc = {int(gid[zs["gt_node"][z]]) for z in range(zs["n_zones"]) if zs["kind"][z] == 0}
        spos = {int(gid[v]): nd["pos"][v] for v in zs["gt_node"]}
        for m in range(len(gid)):
            cand = [s for s in sp if off[piece[m]] <= s < off[piece[m] + 1]]
            s_idx = min(cand, key=lambda q: (abs(int(gid[m]) - q), q))
            s_euc = min(cand, key=lambda q: (float(np.linalg.norm(nd["pos"][m] - spos[q])), q))
            expect = -1 if (int(gid[m]) != s_idx and s_idx in anc) else s_idx
            got = int(gid[zs["gt_node"][zs["zone_of"][m]]]) if zs["zone_of"][m] >= 0 else -1
            assert got == expect, (i, m, got, expect)
            total += 1
            differ += int(s_euc != s_idx)
    assert differ > 0, "the real graphs never distinguish nearest-by-index from nearest-in-metres: this test could not catch that mistake"
    print(f"test_zone_assignment_on_real_graphs ok: {total} dense nodes, {differ} would be assigned differently by metres")


if __name__ == "__main__":
    test_zones_toy()
    test_zone_targets_rules()
    test_node_loss_modes()
    test_loss_masks_and_normalisation()
    test_tier_switch_refused_in_zone_mode()
    test_gradient_reaches_head()
    test_real_frames()
    test_zone_assignment_on_real_graphs()
    print("test_zones: all ok")
