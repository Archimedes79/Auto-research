"""E347 tests (CPU, torch): the junction maps head with its output on the 2 m grid (`jmaps.JMaps` ctx '+r2' / '+r2+onehot', `jmaps_train.py --no-paint --r2 [--onehot]`) and the pair head on it (`jends.JEnds(n1, offset, peak)`).
 - the one-to-one end targets: the cell that holds an end, its offset from the cell centre, two ends in one cell (the nearer to the centre keeps it), ends off the grid; the zone / core raster on the 2 m grid follows the area rule (and not the half-cell shifted legacy fill);
   the targets of a toy junction on the 2 m grid, the default call unchanged (golden numbers of the code before E347);
 - the maps loss: the offset term is the L1 in metres at the end cells only (weight, normalisation, no gradient elsewhere), the heat of the Gaussian arm is the control's loss, the heat of the one-to-one arm is the focal loss (hand formula) with no positive weight;
 - the read-out: the keypoint of a 2 m head is the cell centre + its offset; the 3 x 3 peak keeps one of two adjacent cells, the threshold alone keeps both; the 1 m read-out is unchanged (golden numbers);
 - the pair head: grid as a parameter (a 2 m soft-argmax moves by the cell size, not by 1 m), the position from the offset, the peak step switchable, a head / grid mismatch refused; the pair loss reaches the offset head; `JEndModel` builds the pair head from the ctx;
 - flags off: the command lines of E312 (maps ctx 'dil') and E341 ('dil+nopaint') build the networks of the code before E347 bit for bit (sha256 of the state at seed 0; the stored checkpoints load strictly), the pair head, the maps loss, the keypoints and the junction
   targets reproduce golden numbers of that code.
Run: python test_jmaps_r2.py [--real] [--print-golden]   (`_py14.sh grid/test_jmaps_r2.py`); `--real`: a few real VAL frames and the real pyramid on the CPU (the data path, the targets against the GT ends, 40 optimiser steps overfit a fixed batch).
"""
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jgt as JG  # noqa: E402
import jmaps as J  # noqa: E402
import jends as JE  # noqa: E402
import qjunction as QJ  # noqa: E402
import test_jmaps_dirx as TD  # noqa: E402

torch.set_num_threads(2)
# numbers of the code before E347 (real files, flags off), printed by `--print-golden`
GOLDEN = {
    "state_e312": (136, "ce5dfe6a0b037df0"), "state_e341": (128, "ef57847a5bb3d0c5"),
    "loss": dict(loss=3.1578919887542725, zone=1.1235294342041016, heat=1.016376256942749, dir=0.0019498169422149658, paint=1.0170114040374756), "loss_far": 3.1801280975341797,
    "kp": dict(ent=(2230, 829.5408998903295), ext=(2251, 2165.4532022292187)),
    "targets": dict(jz=68.0, jheat=28.274333963045827, jdir=54.0, jpaint=83.0),
    "jends": dict(logit=98.36999170482159, kp_pos=-645.8732805550098, kp_p=34.75095495581627, scale=2748.026376605034, allowed=1003, valid=128),
}


def fingerprint(m):
    """(number of tensors, sha256 prefix over keys, shapes and bytes in state order): equal fingerprints = bit-identical networks"""
    h = hashlib.sha256()
    sd = m.state_dict()
    for k, v in sd.items():
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return len(sd), h.hexdigest()[:16]


def toy_junction(x0=0.4):
    """one lane north through a junction (entry at (x0, -8.3), exit at (x0, 8.3)) as a lane GT graph with a paint sidecar, the toy of `qjunction._selftest` shifted off the cell borders"""
    def run(a, b, n=12):
        return np.stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)], 1)
    s, nth = run((x0, -30), (x0, -8.3)), run((x0, 8.3), (x0, 30))
    conn = run(s[-1], nth[0], 16)
    pts = np.vstack([s, conn, nth])
    g = dict(pieces=[dict(piece="p0", pts=pts, tier=np.full(len(pts), 3),
                          segments=[dict(kind="lane", i0=0, i1=len(s) - 1), dict(kind="connector", i0=len(s), i1=len(s) + len(conn) - 1), dict(kind="lane", i0=len(s) + len(conn), i1=len(pts) - 1)])])
    paint = dict(stop_line=[dict(poly=np.array([[-2, -8.6], [2, -8.6], [2, -8.0], [-2, -8.0]], np.float32), type=3)], ped_crossing=[np.array([[-3, 6], [3, 6], [3, 8], [-3, 8]], np.float32)])
    return g, paint


def golden_numbers():
    """the numbers that the flags-off code of the five touched modules must reproduce"""
    out = {}
    for name, ctx in (("e312", "dil"), ("e341", "dil+nopaint")):                                         # the command lines of E312 / E341: `jends_train.py --e2e --scales` on the maps run of ctx 'dil' / 'dil+nopaint'
        import jends_train as JT
        torch.manual_seed(0)
        out["state_" + name] = fingerprint(JT.JEndModel(dict(d=64, ctx=ctx), order=True, evidence=True, scales=True, e2e=True))
    m = TD.build("dil")
    o = m(TD.mk_P())
    b = TD.mk_batch()
    loss, info = J.jmaps_loss(o, b)
    out["loss"] = dict(loss=float(loss), **info)
    out["loss_far"] = float(J.jmaps_loss(o, b, far_w=2.0)[0])
    g = torch.Generator().manual_seed(3)
    kp = J.keypoints(dict(j1=torch.randn(2, 8, 100, 100, generator=g) * 2.0 - 0.5), thr=0.3)
    out["kp"] = {k: (int(sum(len(f[k]) for f in kp)), float(sum(f[k].double().sum() for f in kp))) for k in ("ent", "ext")}
    gt, paint = toy_junction()
    t, _ = QJ.junction_targets(gt, paint)
    out["targets"] = {k: float(v.astype(np.float64).sum()) for k, v in t.items()}
    gg = torch.Generator().manual_seed(7)
    jout = dict(j1=torch.randn(2, 8, 100, 100, generator=gg) * 0.5 - 2.5, j05=torch.randn(2, 2, 200, 200, generator=gg) - 2.0, f1=torch.randn(2, 64, 100, 100, generator=gg))
    P = dict(centre1=torch.randn(2, 12, 100, 100, generator=gg))
    torch.manual_seed(0)
    with torch.no_grad():
        je = JE.JEnds(d=64, order=True, evidence=True, scales=True)(jout, P, gt=None)
    out["jends"] = {k: float(je[k].double().sum()) for k in ("logit", "kp_pos", "kp_p", "scale")} | {"allowed": int(je["allowed"].sum()), "valid": int(je["valid"].sum())}
    return out


def close(a, b, tol=1e-4):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_flags_off():
    got = golden_numbers()
    for k in ("state_e312", "state_e341"):
        assert got[k] == GOLDEN[k], (k, got[k], GOLDEN[k], "the E312 / E341 command line no longer builds the same network")
    for grp in ("loss", "kp", "targets", "jends"):
        assert set(got[grp]) == set(GOLDEN[grp]), (grp, sorted(got[grp]), sorted(GOLDEN[grp]))
        for k, v in got[grp].items():
            ref = GOLDEN[grp][k]
            same = v == ref if isinstance(v, int) else (v[0] == ref[0] and close(v[1], ref[1], 1e-9)) if isinstance(v, tuple) else close(v, ref, 1e-5)           # counts exactly, sums to 1e-9 / 1e-5
            assert same, (grp, k, v, ref)
    assert close(got["loss_far"], GOLDEN["loss_far"], 1e-5)
    root = os.environ.get("BFLC_ROOT")                                                                    # the stored checkpoints of E312 / E341 load into the model of the new code (same keys, same shapes)
    import jends_train as JT
    for name, run, ctx in (("e312", "e312_p_s0", "dil"), ("e341", "e341_p_s0", "dil+nopaint")):
        ck = os.path.join(root or "", "runs", run, "model.pt")
        if root and os.path.exists(ck):
            m = JT.JEndModel(dict(d=64, ctx=ctx), order=True, evidence=True, scales=True, e2e=True)
            m.load_state_dict(torch.load(ck, map_location="cpu")["state"], strict=True)
    print("flags off: the E312 / E341 networks (sha256 of the state), the pair head, the maps loss, the keypoints and the junction targets reproduce the code before E347: ok")


def test_end_cells():
    # the end (3.2, -7.4) lies in cell (26, 21) of the 2 m grid (centre (3.0, -7.0)): offset (0.2, -0.4); (3.9, -7.1) is in the same cell but farther from its centre: the nearer end keeps the cell, whatever the order
    ent = np.array([[3.2, -7.4, 0.0, 1.0], [3.9, -7.1, 0.0, 1.0], [-49.9, 49.9, 1.0, 0.0], [60.0, 0.0, 1.0, 0.0]])
    ext = np.array([[-20.5, 10.1, 0.0, 1.0]])
    for order in ((0, 1, 2, 3), (3, 2, 1, 0)):
        oh, off = JG.end_cells(dict(ent=ent[list(order)], ext=ext), n=50, res=2.0)
        assert oh.shape == (2, 50, 50) and off.shape == (4, 50, 50) and oh.dtype == np.float32
        assert oh[0].sum() == 2 and oh[0, 26, 21] == 1 and oh[0, 0, 49] == 1, np.argwhere(oh[0])           # the end off the grid (x = 60) has no cell; (-49.9, 49.9) is the corner cell (0, 49)
        assert np.allclose(off[0:2, 26, 21], (0.2, -0.4), atol=1e-5), off[0:2, 26, 21]                      # the nearer end's offset, not the other one's (0.9, -0.1)
        assert np.allclose(off[0:2, 0, 49], (-49.9 - (-49.0), 49.9 - 49.0), atol=1e-5)
        assert oh[1].sum() == 1 and oh[1, 14, 30] == 1 and np.allclose(off[2:4, 14, 30], (-20.5 - (-21.0), 10.1 - 11.0), atol=1e-5), np.argwhere(oh[1])
        assert off[:, oh.sum(0) == 0].any() == False  # noqa: E712                                          # the offset target lives at the end cells only
    z, zo = JG.end_cells(None, n=50, res=2.0)
    assert z.shape == (2, 50, 50) and not z.any() and not zo.any() and JG.end_cells(dict(ent=np.zeros((0, 4)), ext=np.zeros((0, 4))), n=50, res=2.0)[0].sum() == 0
    cells, inside = JG.end_cell_index(ent, 50, 2.0)
    assert cells[:3].tolist() == [[26, 21], [26, 21], [0, 49]] and inside.tolist() == [True, True, True, False], (cells, inside)
    c1, _ = JG.end_cell_index(ent, 100, 1.0)                                                               # the 1 m grid: the same rule
    assert c1[0].tolist() == [53, 42]
    print("end targets: the cell that holds an end, its offset, two ends in one cell (the nearer keeps it), off-grid ends: ok")


def test_raster_2m():
    from shapely.geometry import Polygon, box as sbox
    rng = np.random.RandomState(0)
    old = os.environ.pop("BFLC_RASTER", None)
    try:
        bad_exact = bad_legacy = n_ref = n_poly = 0
        for _ in range(24):                                                                                  # random convex polygons: the 2 m cells with more than half their area inside agree with shapely
            ang = np.sort(rng.uniform(0, 2 * np.pi, 6))
            poly = np.stack([rng.uniform(-20, 20) + rng.uniform(4, 12) * np.cos(ang), rng.uniform(-20, 20) + rng.uniform(4, 12) * np.sin(ang)], 1)
            sp = Polygon(poly)
            if not sp.is_valid or sp.area < 16:
                continue
            jg = dict(zones=[dict(poly=poly)], core=None, ent=np.zeros((0, 4)), ext=np.zeros((0, 4)))
            ex = JG.fields(jg, n=50, res=2.0, exact=True)[0][0] > 0
            leg = JG.fields(jg, n=50, res=2.0)[0][0] > 0                                                      # no exact flag, no environment: the legacy fill (half a cell = 1 m off)
            lo = np.floor((np.array(sp.bounds[:2]) + 50) / 2).astype(int).clip(0, 49) - 1
            hi = np.ceil((np.array(sp.bounds[2:]) + 50) / 2).astype(int).clip(0, 49) + 1
            ref = np.zeros((50, 50), bool)
            for i in range(max(lo[0], 0), min(hi[0], 49) + 1):
                for j in range(max(lo[1], 0), min(hi[1], 49) + 1):
                    ref[i, j] = sp.intersection(sbox(-50 + 2 * i, -50 + 2 * j, -48 + 2 * i, -48 + 2 * j)).area > 2.0
            bad_exact += int((ex != ref).sum())
            bad_legacy += int((leg != ref).sum())
            n_ref += int(ref.sum())
            n_poly += 1
            os.environ["BFLC_RASTER"] = "exact"
            assert (JG.fields(jg, n=50, res=2.0)[0][0] > 0).tolist() == ex.tolist(), "BFLC_RASTER=exact and exact=True must agree"
            os.environ["BFLC_RASTER"] = "legacy"
            assert (JG.fields(jg, n=50, res=2.0, exact=True)[0][0] > 0).tolist() == ex.tolist(), "exact=True must not depend on the environment"
            assert (JG.fields(jg, n=50, res=2.0)[0][0] > 0).tolist() == leg.tolist()
            os.environ.pop("BFLC_RASTER")
        assert n_poly >= 12 and bad_exact <= 0.06 * n_ref and bad_exact < 0.4 * bad_legacy, (n_poly, bad_exact, bad_legacy, n_ref)
    finally:
        os.environ.pop("BFLC_RASTER", None)
        if old is not None:
            os.environ["BFLC_RASTER"] = old
    print("zone / core raster on the 2 m grid: %d polygons, %d cells differ from the area rule with the exact fill, %d with the legacy fill (of %d cells): ok" % (n_poly, bad_exact, bad_legacy, n_ref))


def test_targets():
    g, paint = toy_junction()
    t, jg = QJ.junction_targets(g, paint, r2=True)
    assert jg is not None and set(t) == {"jz", "jheat", "jdir", "jpaint", "jend", "joff"}
    assert t["jz"].shape == (2, 50, 50) and t["jheat"].shape == (2, 50, 50) and t["jdir"].shape == (4, 50, 50) and t["jpaint"].shape == (2, 200, 200) and t["jend"].shape == (2, 50, 50) and t["joff"].shape == (4, 50, 50)
    assert t["jz"][0].sum() > 4 and t["jheat"].max() > 0.6 and abs(np.hypot(t["jdir"][0], t["jdir"][1]).max() - 1.0) < 1e-5
    # entry (0.4, -8.3): cell (25, 20) of centre (1.0, -9.0): offset (-0.6, 0.7); exit (0.4, 8.3): cell (25, 29) of centre (1.0, 9.0): offset (-0.6, -0.7)
    assert t["jend"][0].sum() == 1 and t["jend"][0, 25, 20] == 1 and np.allclose(t["joff"][0:2, 25, 20], (-0.6, 0.7), atol=1e-4), np.argwhere(t["jend"][0])
    assert t["jend"][1].sum() == 1 and t["jend"][1, 25, 29] == 1 and np.allclose(t["joff"][2:4, 25, 29], (-0.6, -0.7), atol=1e-4), np.argwhere(t["jend"][1])
    assert t["jheat"][0, 25, 20] == t["jheat"][0].max() and t["jdir"][1, 25, 20] > 0.99                      # the heat peaks at the end cell, the heading target (0, 1) is set there
    t1, _ = QJ.junction_targets(g, paint)                                                                  # the default call: the 1 m targets and no end-cell keys
    assert set(t1) == {"jz", "jheat", "jdir", "jpaint"} and t1["jz"].shape == (2, 100, 100) and t1["jheat"].shape == (2, 100, 100)
    t0, jg0 = QJ.junction_targets(dict(pieces=[]), None, r2=True)
    assert jg0 is None and t0["jz"].shape == (2, 50, 50) and not t0["jend"].any() and not t0["joff"].any() and not t0["jheat"].any()
    import lane_gt_data as LD
    assert LD.QData.DEFAULT_R2 is False and "jend" in LD.JUNCTION_NAMES and "joff" in LD.JUNCTION_NAMES
    print("targets on the 2 m grid: shapes, the end cells and offsets of a toy junction, the default call unchanged: ok")


def mk_r2_batch(B=2):
    """end cells: entry (26, 21) offset (0.2, -0.4) and exit (5, 5) offset (-0.7, 0.9) in every frame"""
    b = dict(jz=torch.zeros(B, 2, 50, 50), jheat=torch.zeros(B, 2, 50, 50), jdir=torch.zeros(B, 4, 50, 50), jpaint=torch.zeros(B, 2, 200, 200), jend=torch.zeros(B, 2, 50, 50), joff=torch.zeros(B, 4, 50, 50))
    b["jend"][:, 0, 26, 21], b["jend"][:, 1, 5, 5] = 1.0, 1.0
    b["joff"][:, 0, 26, 21], b["joff"][:, 1, 26, 21] = 0.2, -0.4
    b["joff"][:, 2, 5, 5], b["joff"][:, 3, 5, 5] = -0.7, 0.9
    b["jheat"][:, 0, 26, 21], b["jheat"][:, 1, 5, 5] = 1.0, 1.0
    return b


def mk_r2_out(B=2, onehot=False):
    j1 = torch.full((B, 12, 50, 50), -4.0)
    j1[:, 4:12] = 0.0
    j1[:, 5], j1[:, 7] = 1.0, 1.0
    out = dict(j1=j1, f1=torch.zeros(B, 64, 50, 50), r2=True)
    if onehot:
        out["onehot"] = True
    return out


def focal_ref(logit, onehot, n):
    """the penalty-reduced focal loss of `qendnet._focal` for a one-hot target, in numpy"""
    p = 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))
    pos = onehot >= 1.0
    return (-((1 - p) ** 2 * np.log(p))[pos].sum() - (p ** 2 * np.log(1 - p))[~pos].sum()) / n


def test_loss():
    b = mk_r2_batch()
    # the offset term: |dx| + |dy| in metres at the end cells, mean over the end cells of both kinds, weight OFF_W; the offset channels are the only thing that moves
    out = mk_r2_out()
    out["j1"][:, 8:10, 26, 21] = torch.tensor([0.2, -0.4])[None, :]                                          # right
    out["j1"][:, 10:12, 5, 5] = torch.tensor([-0.7, 0.9])[None, :]
    l0, i0 = J.jmaps_loss(out, b)
    assert i0["off"] < 1e-6, i0
    out["j1"][:, 8:10, 26, 21] = 0.0                                                                       # entry offset off by (0.2, 0.4) = 0.6 in each of the 2 frames; 4 end cells: 1.2 / 4
    l1, i1 = J.jmaps_loss(out, b)
    assert close(i1["off"], 0.3) and close(float(l1) - float(l0), J.OFF_W * 0.3, 1e-4) and abs(i1["heat"] - i0["heat"]) < 1e-9 and abs(i1["dir"] - i0["dir"]) < 1e-9, (i1, i0, float(l1), float(l0))
    j1 = mk_r2_out()["j1"].requires_grad_(True)                                                             # the gradient of the offset term sits at the end cells and nowhere else
    bo = dict(b, jheat=torch.zeros_like(b["jheat"]))
    lg, _ = J.jmaps_loss(dict(j1=j1, f1=None, r2=True), bo)
    lg.backward()
    go = j1.grad[:, 8:12].abs().sum(1)
    mask = b["jend"].amax(1) > 0
    assert float(go[mask].min()) > 0 and float(go[~mask].max()) == 0.0, (float(go[mask].min()), float(go[~mask].max()))
    # the heat of the Gaussian arm is the control's loss on the same maps: the 12-channel 2 m output and the 8-channel one give the same heat / zone / direction terms
    lr, ir = J.jmaps_loss(out, b)
    o8 = dict(j1=out["j1"][:, :8].clone(), f1=None)
    l8, i8 = J.jmaps_loss(o8, b)
    assert ir["heat"] == i8["heat"] and ir["zone"] == i8["zone"] and ir["dir"] == i8["dir"], (ir, i8)
    # the one-to-one arm: the focal loss of the hand formula on the one-hot target, normalised by the number of end cells of the kind, no positive weight, no dice
    oo = mk_r2_out(onehot=True)
    oo["j1"][:, 2, 26, 21] = 2.0
    oo["j1"][0, 2, 26, 22] = 1.0                                                                            # a firing neighbour of an end is a negative
    oo["j1"][:, 3, 5, 5] = 0.5
    lo, io = J.jmaps_loss(oo, b)
    want = focal_ref(oo["j1"][:, 2].numpy(), b["jend"][:, 0].numpy(), 2.0) + focal_ref(oo["j1"][:, 3].numpy(), b["jend"][:, 1].numpy(), 2.0)
    assert close(io["heat"], want, 1e-4), (io["heat"], want)
    oo2 = mk_r2_out(onehot=True)
    oo2["j1"][:, 2, 26, 21] = 2.0
    oo2["j1"][0, 2, 26, 22] = 4.0                                                                           # a more confident neighbour costs more
    assert J.jmaps_loss(oo2, b)[1]["heat"] > io["heat"] + 0.3
    assert abs(io["heat"] - ir["heat"]) > 0.1, "the one-to-one heat loss must not be the BCE + dice of the control"
    nb = dict(b, jend=torch.zeros_like(b["jend"]), joff=torch.zeros_like(b["joff"]))                         # a batch without ends: the focal loss has only negatives, finite, normalised by 1
    ln, inn = J.jmaps_loss(oo, nb)
    assert np.isfinite(float(ln)) and inn["off"] == 0.0
    try:
        J.jmaps_loss(oo, b, far_w=1.0)
        raise RuntimeError("far_w on a 2 m head must be refused")
    except AssertionError:
        pass
    print("loss: the offset L1 (weight, normalisation, gradient only at the end cells), the control's heat loss for the Gaussian arm, the focal loss for the one-to-one arm: ok")


def test_keypoints():
    j1 = torch.full((1, 12, 50, 50), -6.0)
    j1[0, 4:12] = 0.0
    j1[0, 5] = 3.0
    j1[0, 2, 26, 21], j1[0, 2, 26, 22] = 3.0, 1.5                                                           # two adjacent cells above the threshold: p = 0.95 and 0.82
    j1[0, 8:10, 26, 21] = torch.tensor([0.2, -0.4])
    j1[0, 8:10, 26, 22] = torch.tensor([0.2, -1.0])
    out = dict(j1=j1, r2=True)
    kp = J.keypoints(out, thr=0.5)[0]
    assert len(kp["ent"]) == 1 and abs(kp["ent"][0, 0] - 3.2) < 1e-5 and abs(kp["ent"][0, 1] - (-7.4)) < 1e-5 and kp["ent"][0, 3] > 0.99, kp["ent"]      # 3 x 3 peak; cell (26, 21): centre (3, -7) + offset
    kpa = J.keypoints(out, thr=0.5, peak=False)[0]
    ys = sorted(kpa["ent"][:, 1].tolist())
    assert len(kpa["ent"]) == 2 and np.allclose(ys, [-7.4, -6.0], atol=1e-5), ys                                               # the neighbour (26, 22): centre y = -5.0, offset -1.0 -> -6.0
    out1 = dict(j1=j1, r2=True, onehot=True)                                                                # the one-to-one head's own rule: no peak step
    assert len(J.keypoints(out1, thr=0.5)[0]["ent"]) == 2 and len(J.keypoints(out1, thr=0.5, peak=True)[0]["ent"]) == 1
    assert len(J.keypoints(out, thr=0.5)[0]["ext"]) == 0
    # the soft-argmax of an 8-channel map on a 2 m grid moves by the cell size: a peak 0.9 with a right neighbour 0.3 -> +0.25 cells = 0.5 m
    j8 = torch.full((1, 8, 50, 50), -9.0)
    j8[0, 2, 26, 21], j8[0, 2, 26, 22] = float(np.log(0.9 / 0.1)), float(np.log(0.3 / 0.7))
    k8 = J.keypoints(dict(j1=j8), thr=0.5)[0]["ent"]
    assert abs(float(k8[0, 0]) - 3.0) < 1e-4 and abs(float(k8[0, 1]) - (-7.0 + 0.5)) < 0.01 and len(k8) == 1, k8
    print("read-out: the keypoint of a 2 m head is the cell centre + offset, the peak step keeps one of two adjacent cells, the threshold alone both; the soft-argmax follows the cell size: ok")


def mk_pair_inputs(B=2):
    ent = torch.tensor([[[-1.75, -8.3, 0.0, 1.0], [1.75, -8.3, 0.0, 1.0]]] * B)
    ext = torch.tensor([[[-1.75, 8.3, 0.0, 1.0], [1.75, 8.3, 0.0, 1.0]]] * B)
    gt = dict(ent=ent, ext=ext, ent_m=torch.ones(B, 2, dtype=torch.bool), ext_m=torch.ones(B, 2, dtype=torch.bool))
    batch = dict(ent=ent, ext=ext, ent_m=gt["ent_m"], ext_m=gt["ext_m"], pairs=torch.tensor([[[0, 0], [1, 1]]] * B))
    return gt, batch


def test_pair_head():
    B, d = 2, 64
    gt, batch = mk_pair_inputs(B)
    # the position from the offset: a peak at cell (24, 20) (centre (-1.0, -9.0)) with offset (0.2, 0.3) -> (-0.8, -8.7); the neighbour cell (24, 21) is a second peak only without the local-maximum step
    j1 = torch.full((B, 12, 50, 50), -6.0)
    j1[:, 4:12] = 0.0
    j1[:, 5], j1[:, 7] = 4.0, 4.0
    j1[:, 2, 24, 20], j1[:, 2, 24, 21] = 2.0, -1.0                                                          # p = 0.88 and 0.27
    j1[:, 8, 24, 20], j1[:, 9, 24, 20] = 0.2, 0.3
    jout = dict(j1=j1, f1=torch.randn(B, d, 50, 50), r2=True)
    P = dict(centre1=torch.randn(B, 12, 100, 100))
    torch.manual_seed(0)
    on = JE.JEnds(d=d, order=True, evidence=True, paint_ev=False, n1=50, offset=True)(jout, P, gt=None)
    assert on["valid"][0, 0].sum() == 1 and abs(float(on["kp_pos"][0, 0, 0]) + 0.8) < 1e-5 and abs(float(on["kp_pos"][0, 0, 1]) + 8.7) < 1e-5, (on["valid"][0, 0].sum(), on["kp_pos"][0, 0])
    torch.manual_seed(0)
    onp = JE.JEnds(d=d, order=True, evidence=True, paint_ev=False, n1=50, offset=True, peak=False)(jout, P, gt=None)
    assert onp["valid"][0, 0].sum() == 2, "without the peak step both cells above PEAK_MIN are candidates"
    first = onp["kp_pos"][0, :JE.K][onp["valid"][0, 0]]                                                     # the valid entry keypoints of frame 0
    assert any(torch.allclose(r, torch.tensor([-0.8, -8.7]), atol=1e-5) for r in first) and any(torch.allclose(r, torch.tensor([-1.0, -7.0]), atol=1e-5) for r in first), first   # the second candidate: its cell centre (offset 0)
    # a head and a pair head that disagree about the grid are refused
    for kw, jo in ((dict(), jout), (dict(n1=50, offset=True), dict(j1=j1[:, :8], f1=jout["f1"])), (dict(n1=50), jout)):
        try:
            JE.JEnds(d=d, order=True, evidence=True, paint_ev=False, **kw)(jo, P, gt=None)
            raise RuntimeError(f"JEnds{kw} on a {tuple(jo['j1'].shape)} map must be refused")
        except AssertionError as ex:
            assert "do not match" in str(ex), ex
    # the soft-argmax on the 2 m grid moves by the cell size (a peak 0.9, right neighbour 0.3: +0.25 cells)
    j8 = torch.full((B, 8, 50, 50), -9.0)
    j8[:, 4:8] = 0.0
    j8[:, 5], j8[:, 7] = 4.0, 4.0
    j8[:, 2, 26, 21], j8[:, 2, 26, 22] = float(np.log(0.9 / 0.1)), float(np.log(0.3 / 0.7))
    os8 = JE.JEnds(d=d, order=True, evidence=False, n1=50)(dict(j1=j8, f1=jout["f1"]), P, gt=None)
    assert abs(float(os8["kp_pos"][0, 0, 0]) - 3.0) < 1e-4 and abs(float(os8["kp_pos"][0, 0, 1]) - (-7.0 + 0.5)) < 0.01, os8["kp_pos"][0, 0]
    # the pair loss reaches the offset head (and the heat head through the probabilities), through a real 2 m maps head
    for ctx in ("dil+nopaint+r2", "dil+nopaint+r2+onehot"):
        torch.manual_seed(0)
        jm = J.JMaps(ctx=ctx)
        jo = jm(TD.mk_P(B=B))
        je = JE.JEnds(d=d, order=True, evidence=True, paint_ev=False, n1=50, offset=True, peak="onehot" not in ctx)
        out = je(jo, P, gt=gt)
        loss, info = JE.jends_loss(out, batch)
        loss.backward()
        go = float(jm.head2.weight.grad[8:12].abs().sum()) + float(jm.head2.bias.grad[8:12].abs().sum())
        gh = float(jm.head2.weight.grad[2:4].abs().sum())
        assert go > 0 and gh > 0 and info["pos_pairs"] >= 2 * B, (ctx, go, gh, info)
        assert jo["j1"].shape == (B, 12, 50, 50) and out["kp_pos"].shape == (B, 2 * JE.K, 2) and out["logit"].shape == (B, JE.K, JE.K)
    print("pair head: the grid as a parameter (the 2 m soft-argmax moves by the cell size), the position from the offset, the peak step switchable, a head / grid mismatch refused, the pair loss reaches the offset head: ok")


def test_glue():
    import jends_train as JT
    for ctx, n1, off, peak in (("dil+nopaint", 100, False, True), ("dil+nopaint+r2", 50, True, True), ("dil+nopaint+r2+onehot", 50, True, False)):
        m = JT.JEndModel(dict(d=64, ctx=ctx), order=True, evidence=True, scales=True, e2e=True)
        assert (m.je.n1, m.je.offset, m.je.peak) == (n1, off, peak) and m.je.paint_ev is False, (ctx, m.je.n1, m.je.offset, m.je.peak)
        assert m.je.pair[0].in_features == 150 and m.jmodel.jm.r2 == off and m.jmodel.jm.onehot == (not peak)
    gt, batch = mk_pair_inputs(1)
    m = JT.JEndModel(dict(d=64, ctx="dil+nopaint+r2"), order=True, evidence=True, scales=True, e2e=True)                  # the whole model: pyramid (frozen) -> 2 m maps head -> pair head, one random frame
    P, out, oe = m(torch.randn(1, 39, 200, 200), gt=gt)
    assert out["j1"].shape == (1, 12, 50, 50) and out["f1"].shape == (1, 64, 50, 50) and oe["kp_pos"].shape == (1, 2 * JE.K, 2) and "j05" not in out
    loss, _ = JE.jends_loss(oe, batch)
    lm, _ = J.jmaps_loss(out, dict(mk_r2_batch(1), jheat=torch.zeros(1, 2, 50, 50)))
    (loss + lm).backward()
    assert float(m.jmodel.jm.head2.weight.grad[8:12].abs().sum()) > 0 and all(p.grad is None for p in m.jmodel.pyr.parameters()), "the offset head trains, the frozen pyramid gets nothing"
    print("JEndModel builds the pair head from the maps ctx; the whole model runs (pyramid -> 2 m maps head -> pair head) and the e2e loss reaches the offset head: ok")


def smoke_real(n_frames=4, steps=40):
    """a few real VAL frames: the data path on the 2 m grid, the end targets against the GT ends, the real pyramid and 40 Adam steps on one fixed batch (the loss falls, the offset error falls, the read-out finds the ends)"""
    import lane_gt_data as LD
    os.environ["BFLC_SIG"] = "1.0"
    LD.QData.DEFAULT_JUNCTION, LD.QData.DEFAULT_R2 = True, True
    ds = LD.QData("val_clean", map_mode="error", pixels=False)
    items = []
    for i in range(0, len(ds), 5):
        it = ds[i]
        if len(it["ent"]) >= 3:
            items.append(it)
        if len(items) == n_frames:
            break
    assert len(items) == n_frames
    b = LD.collate(items)
    assert b["jz"].shape == (n_frames, 2, 50, 50) and b["jheat"].shape == (n_frames, 2, 50, 50) and b["jend"].shape == (n_frames, 2, 50, 50) and b["joff"].shape == (n_frames, 4, 50, 50) and b["jdir"].shape == (n_frames, 4, 50, 50)
    cc = -50.0 + (torch.arange(50).float() + 0.5) * 2.0
    n_ok = n_end = 0
    for i, it in enumerate(items):                                                                          # every end cell: centre + offset = a GT end of the frame (entries and exits)
        for k, E in enumerate((it["ent"], it["ext"])):
            for (r, c) in b["jend"][i, k].nonzero().tolist():
                pos = torch.stack([cc[r] + b["joff"][i, 2 * k, r, c], cc[c] + b["joff"][i, 2 * k + 1, r, c]])
                n_end += 1
                n_ok += int((E[:, :2] - pos).norm(dim=1).min() < 1e-4)
    assert n_end >= n_frames and n_ok == n_end, (n_ok, n_end)
    P = None
    x = b["x"].float()
    root = os.environ["BFLC_ROOT"]
    model = J.JModel(width=48, d=64, ctx="dil+nopaint+r2")
    model.load_base(os.path.join(root, "runs", "e273_p_s0", "model.pt"))
    with torch.no_grad():
        P = model.pyr(x)
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.jm.parameters(), lr=2e-3)
    hist = []
    for it in range(steps):
        loss, info = J.jmaps_loss(model.jm(P), b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append((float(loss), info["off"], info["heat"]))
    assert hist[-1][0] < 0.8 * hist[0][0] and hist[-1][1] < 0.8 * hist[0][1], hist[::10]
    with torch.no_grad():
        kp = J.keypoints(model.jm(P), thr=0.3)
    found = tot = 0
    for i, it in enumerate(items):
        E = it["ent"][:, :2].numpy()
        tot += len(E)
        found += sum(float(np.linalg.norm(kp[i]["ent"][:, :2].numpy() - e, axis=1).min()) < 3.0 for e in E) if len(kp[i]["ent"]) else 0
    print(f"real frames: {n_frames} VAL frames on the 2 m grid, {n_end} end cells = GT ends; {steps} steps on one batch: loss {hist[0][0]:.3f} -> {hist[-1][0]:.3f}, offset error {hist[0][1]:.3f} -> {hist[-1][1]:.3f} m; entries found within 3 m {found} of {tot}")


def main():
    if "--print-golden" in sys.argv:
        import pprint
        pprint.pprint(golden_numbers(), sort_dicts=False)
        return
    test_end_cells()
    test_raster_2m()
    test_targets()
    test_loss()
    test_keypoints()
    test_pair_head()
    test_glue()
    test_flags_off()
    if "--real" in sys.argv:
        smoke_real()
    print("test_jmaps_r2 ok")


if __name__ == "__main__":
    main()
