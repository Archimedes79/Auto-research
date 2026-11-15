"""E357 tests of the token pair head on GT nodes (`jtok.py`, `data/lane_tok_data.py`, `train/jtok_train.py`, `evaluation/jtok_eval.py`); CPU only (the GPU is hidden), 2 threads.

    wsl bash tools/env/_py14.sh training/grid/test_jtok.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

No data: geometry (the 13 pair features, the gate and the lane evidence = `JEnds`'s on the same pairs; the gate's own bounds by hand), lattice (restored, 3 m refused, a stale copy of qtargets caught, right inside a loader
worker), rules (lo / lo_near / lo_all on a hand-made frame), model (the loss by hand, a gradient into every part, no flag / piece id / node order reaches the logits, a frame alone = the frame in a padded batch), links (the link
tables, the node removal and the contraction of the GT links on hand-made tables).
Real frames: writer (3 VAL frames: this file's scoring path = `score_frames`, the GT edge list through the writer = `oracle_prediction`, what the two read-outs do), gate (50 TRAIN frames: the share of the GT links outside
the allowed gate, split short / long; RULE of the card: above 0.5 % the gate is widened before any training, this test fails), data (x / token of a frame = QData's, VAL = the 873 rows, the cache round trip, the batch),
pipeline (20 training steps on the CPU into a temporary folder -> model.pt -> dump of 3 VAL frames -> scores; the refusal of another recipe in the same folder; the degraded-node dump).
"""
import importlib.util
import math
import os
import pickle
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                      # this test never touches the GPU
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in (".", "..", "../data", "../data/prep", "../train", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
import jends as JE  # noqa: E402
import jtok as JT  # noqa: E402
import jtok_eval as JV  # noqa: E402
import jtok_train as JTT  # noqa: E402
import lane_gt_data as LD  # noqa: E402
import lane_tok_data as TD  # noqa: E402
import qendnet as QEN  # noqa: E402
import qformat  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qtargets as QT  # noqa: E402

torch.set_num_threads(2)
SCRATCH = os.path.join(JT.ROOT, "temp", "e357")
GATE_MAX = 0.005                                                             # the card's rule: more than 0.5 % of the GT links outside the allowed gate = widen the gate before training


def must_fail(fn, *a, **k):
    """a mutation check: fn must fail its assertion"""
    try:
        fn(*a, **k)
    except AssertionError:
        return
    raise RuntimeError(f"{getattr(fn, '__name__', fn)} passed with the mutated input")


def rot(deg):
    """the heading +y turned by `deg` degrees counter-clockwise"""
    return [-math.sin(math.radians(deg)), math.cos(math.radians(deg))]


def table(pos, dirv=None, links=(), conn=(), scale=None, res=2.5):
    """a hand-made frame table (the keys of `lane_tok_data.TABLE`): headings +y unless given, `conn` = the indices of the links on a connector path, `scale` = {link index: (a, b)}"""
    pos = np.asarray(pos, float)
    M, L = len(pos), len(links)
    src, dst = (np.array([l[i] for l in links], np.int32) for i in (0, 1))
    sc, cn = np.full((L, 2), np.nan), np.zeros(L, bool)
    for l, v in (scale or {}).items():
        sc[l] = v
    cn[list(conn)] = True
    return dict(pos=pos, dirv=np.tile([0.0, 1.0], (M, 1)) if dirv is None else np.asarray(dirv, float), cell=np.clip(np.floor((pos + 50.0) / res), 0, round(100 / res) - 1).astype(np.int16), flags=np.zeros((M, 8), bool),
                piece=np.zeros(M, np.int32), sstart=np.zeros(M, bool), src=src, dst=dst, long=np.linalg.norm(pos[src] - pos[dst], axis=1) > JT.SHORT, scale=sc, conn=cn)


def item(t, token="t", row=0):
    return dict(t, token=token, row=row)


# ------------------------------------------------------------------------------------------------------------------------------ no data
def test_geometry():
    torch.manual_seed(0)
    B, K, d = 2, JE.K, 64
    jout = dict(j1=torch.randn(B, 8, 100, 100) * 0.1 - 2.0, f1=torch.randn(B, d, 100, 100))
    jout["j1"][:, 4:8] = torch.randn(B, 4, 100, 100)                         # random headings
    P = dict(centre1=torch.randn(B, 12, 100, 100))
    je, seen = JE.JEnds(d=d, order=False, evidence=True, paint_ev=False), {}
    je.pair.register_forward_pre_hook(lambda _m, a_: seen.__setitem__("f", a_[0].detach()))      # the input of JEnds's pair MLP: [token i, token j, 13 geometric features, 4 evidence features (lane mean, lane min, zone, core)]
    with torch.no_grad():
        out = je(jout, P)
    pos, dirv, valid = out["kp_pos"], out["kp_dir"], out["valid"].flatten(1)                     # ONE node set: the K entries, then the K exits; JEnds's pairs are the block [entries, exits]
    assert valid.all() and float((pos[:, :K, None] - pos[:, None, K:]).norm(dim=-1).max()) > JT.MAX_CHORD
    g, allowed = JT.pair_geometry(pos, dirv, valid, out["kp_p"], min_chord=QEN.MIN_CHORD, max_chord=float("inf"))
    assert g.shape == (B, 2 * K, 2 * K, JT.NG) and torch.allclose(g[:, :K, K:], seen["f"][..., 2 * d:2 * d + JT.NG], atol=1e-6), "the 13 geometric pair features are not JEnds's"
    assert torch.equal(allowed[:, :K, K:], out["allowed"]) and 0 < int(out["allowed"].sum()) < B * K * K, "the gate (with JEnds's chord bounds) is not JEnds's"
    ev = JT.lane_evidence(P["centre1"], pos, dirv)
    assert torch.allclose(ev[:, :K, K:], seen["f"][..., 2 * d + JT.NG:2 * d + JT.NG + 2], atol=1e-5), "the lane evidence is not JEnds's lane mean / minimum"
    assert not JT.pair_geometry(pos, dirv, valid)[1][:, :K, K:][(pos[:, :K, None] - pos[:, None, K:]).norm(dim=-1) > JT.MAX_CHORD].any()
    # the gate's own bounds by hand. Node 0 at the origin heading +y; 1: 5 m ahead; 2: 0.4 m ahead (too close); 3: 0.6 m ahead; 4: 101 m ahead (too far); 5: 99 m ahead; 6: an invalid node 3 m ahead
    p = torch.tensor([[[0.0, 0.0], [0.0, 5.0], [0.0, 0.4], [0.0, 0.6], [0.0, 101.0], [0.0, 99.0], [0.0, 3.0]]])
    h = torch.tensor([[[0.0, 1.0]] * 7])
    v = torch.tensor([[True] * 6 + [False]])
    a = JT.pair_geometry(p, h, v)[1][0]
    assert a[0].tolist() == [False, True, False, True, False, True, False], a[0].tolist()
    assert not a[1, 0] and not a.diagonal().any() and not a[6].any(), "a node behind, the node itself and an invalid node are no candidates"
    # the turn bound: the chord 0 -> 1 points to +y, heading 0 is turned by +w, heading 1 by -w: both look forward along the chord, the turn is 2 w (130 degrees is the bound)
    for w, ok in ((64.0, True), (66.0, False)):
        a = JT.pair_geometry(torch.tensor([[[0.0, 0.0], [0.0, 5.0]]]), torch.tensor([[rot(w), rot(-w)]]))[1][0]
        assert bool(a[0, 1]) == ok, (w, a)
    print(f"geometry: 13 features, gate and lane evidence = JEnds's on {B * K * K} pairs; chord 0.5 .. 100 m, forward, turn <= 130 degrees, valid, i != j by hand: ok")


class _ToyFrames(torch.utils.data.Dataset):
    """what a loader worker sees: the lattice constants before, the table of the toy lane on the 2.5 m lattice, the constants after"""

    def __len__(self):
        return 2

    def __getitem__(self, i):
        before = (QT.N, QT.RES)
        t = TD.frame_table(QE.toy_lane_gt()[0], 2.5)
        return dict(before=before, after=(QT.N, QT.RES), n=len(t["pos"]), cells=bool(np.array_equal(t["cell"], np.floor((t["pos"] + 50.0) / 2.5))), worker=torch.utils.data.get_worker_info() is not None)


def test_lattice():
    assert (QT.N, QT.RES) == (50, 2.0)
    with JT.lattice(2.5):
        assert (QT.N, QT.RES) == (40, 2.5)
        JT.check_lattice(2.5)
        must_fail(JT.check_lattice, 2.0)
    for bad in (KeyError, ValueError):                                       # restored after an exception inside the block; 3 m does not divide the window
        try:
            with JT.lattice(2.5 if bad is KeyError else 3.0):
                raise KeyError
        except bad:
            pass
        assert (QT.N, QT.RES) == (50, 2.0), "the lattice was not restored"
    g = QE.toy_lane_gt()[0]                                                  # 80 m of lane + a suppressed connector
    t25, t2 = TD.frame_table(g, 2.5), TD.frame_table(g, 2.0)
    assert (QT.N, QT.RES) == (50, 2.0) and abs(len(t25["pos"]) - 32) <= 3 and abs(len(t2["pos"]) - 40) <= 3, (len(t25["pos"]), len(t2["pos"]))
    assert np.array_equal(t25["cell"], np.floor((t25["pos"] + 50.0) / 2.5)) and t25["pos"].dtype == np.float64 and t25["scale"].shape == (len(t25["src"]), 2)
    assert int(t25["long"].sum()) == 1 and bool(t25["conn"][t25["long"]][0]) and np.isfinite(t25["scale"][t25["long"]]).all() and np.isnan(t25["scale"][~t25["long"]]).all(), "the link over the suppressed connector is the one junction link"
    spec = importlib.util.spec_from_file_location("qtargets_stale", QT.__file__)                 # a second copy of qtargets.py (it would stay on 2 m) and a consumer that holds it
    stale = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stale)
    for name, mod in (("qtargets_stale", stale), ("jtok_stale_holder", types.SimpleNamespace(QT=stale, __file__="holder.py"))):
        sys.modules[name] = mod
        try:
            with JT.lattice(2.5):
                must_fail(JT.check_lattice, 2.5)
        finally:
            del sys.modules[name]
    rows = list(torch.utils.data.DataLoader(_ToyFrames(), batch_size=None, num_workers=1))       # inside a worker: 2 m before, the table of the 2.5 m lattice, 2 m after
    assert len(rows) == 2 and all(r["worker"] and tuple(r["before"]) == (50, 2.0) == tuple(r["after"]) and r["n"] == len(t25["pos"]) and r["cells"] for r in rows), rows
    print(f"lattice: restored (also after an exception), 3 m refused, a stale copy caught, toy lane {len(t25['pos'])} nodes at 2.5 m / {len(t2['pos'])} at 2 m, the same table inside a loader worker: ok")


def test_rules():
    # 0 -> 1 -> 2 on a line (2.5 m apart); 3 beside the line (63 degrees off node 0's heading: outside its cone; its own nearest node ahead is 2); 4 between 0 and 1 heading the other way; 5: 9 m ahead of 2 (beyond 8 m);
    # 6: ahead of 2 at 2.2 m but turned by 70 degrees (more than 60); 7: 46 m ahead of 5
    t = table([[0, 0], [0, 2.5], [0, 5], [2, 1], [0, 1], [0, 14], [1, 7], [0, 60]], dirv=[rot(0)] * 4 + [rot(180), rot(0), rot(-70), rot(0)])
    links = lambda arm: sorted(zip(*(JV.rule_prediction(t, arm)["links"][k].tolist() for k in ("src", "dst"))))  # noqa: E731
    lo = [(0, 1), (1, 2), (3, 2)]
    assert links("lo") == lo, links("lo")
    # without a successor: 2, 4, 5, 6, 7; without a predecessor: 0, 3, 4, 5, 6, 7. In the gate: 2 -> 6 (2.2 m), 2 -> 5 (9 m), 2 -> 7 (55 m), 6 -> 5 (7.1 m), 6 -> 7, 5 -> 7 (46 m: beyond the 40 m of lo_near)
    assert links("lo_near") == sorted(lo + [(2, 6), (6, 5)]), links("lo_near")
    assert links("lo_all") == sorted(lo + [(2, 5), (2, 6), (2, 7), (5, 7), (6, 5), (6, 7)]), links("lo_all")
    p = JV.rule_prediction(t, "lo_all")
    lk = p["links"]
    assert lk["curves"] is None and (lk["p"] == 1).all() and np.allclose(lk["a"], 1 / 3) and np.allclose(lk["b"], 1 / 3) and lk["long"].tolist() == [bool(np.linalg.norm(t["pos"][a] - t["pos"][b]) > 8) for a, b in zip(lk["src"], lk["dst"])]
    assert (p["nodes"]["e"] == 1).all() and len(p["nodes"]["section"]) == 8 and JV.rule_links(t["pos"][:1], t["dirv"][:1])[0].size == 0
    print("rules: lo (cone 45 degrees, 8 m, heading 60 degrees, one link per node), lo_near (nearest open start in the gate within 40 m), lo_all (every open start in the gate) on a hand-made frame: ok")


def _bce(z, y):
    return math.log1p(math.exp(-z if y else z))


def test_model():
    # the loss by hand, one frame of 3 nodes. Allowed: all pairs but (2, 0) and the diagonal; chords: 0-1 3 m (short), 0-2 10 m, 1-2 9 m (long); GT links 0 -> 1, 1 -> 2 (fitted scales 0.3, 0.5) and 2 -> 0 (outside the gate)
    z = torch.tensor([[[0.0, 1.0, -2.0], [0.5, 0.0, 3.0], [-1.0, 0.2, 0.0]]])
    al = torch.tensor([[[False, True, True], [True, False, True], [False, True, False]]])
    ch = torch.tensor([[[0.0, 3.0, 10.0], [3.0, 0.0, 9.0], [10.0, 9.0, 0.0]]])
    y = torch.zeros(1, 3, 3)
    y[0, 0, 1] = y[0, 1, 2] = y[0, 2, 0] = 1.0
    ys = torch.full((1, 3, 3, 2), float("nan"))
    ys[0, 1, 2] = torch.tensor([0.3, 0.5])
    out = dict(logit=z, allowed=al, chord=ch, scale=torch.full((1, 3, 3, 2), 0.4))
    loss, info = JT.tok_loss(out, dict(y=y, ys=ys))
    l_s = 0.5 * (_bce(1.0, 1) + _bce(0.5, 0))                                # short group: one positive (0, 1), one negative (1, 0)
    l_l = 0.5 * (_bce(3.0, 1) + (_bce(-2.0, 0) + _bce(0.2, 0)) / 2)         # long group: one positive (1, 2), two negatives (0, 2), (2, 1)
    dual = float(QEN._dual_softmax(z, y, al))
    assert abs(info["bce_s"] - l_s) < 1e-6 and abs(info["bce_l"] - l_l) < 1e-6 and abs(info["scale_l1"] - 0.1) < 1e-6 and abs(float(loss) - (l_s + l_l + dual + 0.5 * 0.1)) < 1e-5, (info, l_s, l_l, dual)
    assert (info["gt_links"], info["pos_pairs"], info["gt_outside"], info["allowed"]) == (3.0, 2.0, 1.0, 5.0), info
    y2 = y.clone()                                                           # a frame without a long positive: its long negatives keep the weight 1/2; no scale term
    y2[0, 1, 2] = 0.0
    _, i2 = JT.tok_loss(out, dict(y=y2, ys=torch.full((1, 3, 3, 2), float("nan"))))
    assert abs(i2["bce_l"] - 0.5 * (_bce(3.0, 0) + _bce(-2.0, 0) + _bce(0.2, 0)) / 3) < 1e-6 and "scale_l1" not in i2, i2
    # the model on two hand-made frames (9 and 6 nodes): shapes, ranges, a gradient into every part
    torch.manual_seed(0)
    ta = table([[0, -10 + 2.5 * k] for k in range(5)] + [[3, 12], [-8, 14], [10, 20], [-20, -20]], links=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6)], conn=[4, 5], scale={4: (0.3, 0.5), 5: (0.4, 0.6)})
    tb = table([[5 + 2.5 * k, 3.0 * k] for k in range(6)], dirv=[rot(-40)] * 6, links=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    b = TD.collate([item(ta, "a"), item(tb, "b")])
    assert b["pos"].shape == (2, 9, 2) and b["valid"].sum(1).tolist() == [9, 6] and b["y"].sum((1, 2)).tolist() == [6.0, 5.0] and int(torch.isfinite(b["ys"][..., 0]).sum()) == 2 and b["pos"].dtype == torch.float32
    model = JT.JTokModel()
    x = torch.randn(2, 39, 200, 200)
    out = JT.forward_batch(model, dict(b, x=x))
    assert out["logit"].shape == (2, 9, 9) and out["scale"].shape == (2, 9, 9, 2) and 0.05 <= float(out["scale"].min()) and float(out["scale"].max()) <= 1.25 and not out["allowed"][1, 6:].any() and not out["allowed"][1, :, 6:].any()
    loss, info = JT.tok_loss(out, b)
    loss.backward()
    assert info["gt_outside"] == 0 and info["pos_pairs"] == 11 and "scale_l1" in info, info
    grad = lambda mod: [float(p.grad.abs().sum()) if p.grad is not None else None for p in mod.parameters()]  # noqa: E731
    assert all(v is not None and v > 0 for v in grad(model.tok.enc)), "the loss does not reach the token encoder"
    assert all(v is not None for v in grad(model.tok)) and all(v > 0 for mod in (model.tok.pair, model.tok.scale) for v in grad(mod)) and sum(grad(model.tok.blocks)) > 0, "a part of the head gets no gradient"
    assert all(p.grad is None for p in model.pyr.parameters()), "the frozen pyramid must not get a gradient"
    # no node type, piece id or section id reaches the head: scrambling them in the batch changes nothing
    model.eval()
    rng = np.random.RandomState(0)
    scr = lambda t: dict(t, flags=rng.rand(*t["flags"].shape) > 0.5, piece=rng.permutation(len(t["pos"])).astype(np.int32), sstart=rng.rand(len(t["pos"])) > 0.5, cell=rng.permutation(t["cell"]),  # noqa: E731
                         conn=~t["conn"])
    with torch.no_grad():
        o1 = JT.forward_batch(model, dict(b, x=x))
        b2 = TD.collate([item(scr(ta), "a"), item(scr(tb), "b")])
        o2 = JT.forward_batch(model, dict(b2, x=x))
        assert torch.equal(o1["logit"], o2["logit"]) and torch.equal(o1["scale"], o2["scale"]) and float(JT.tok_loss(o1, b)[0]) == float(JT.tok_loss(o2, b2)[0]), "flags / piece ids / cells reach the logits"
        # ... and not through the ORDER of the nodes (the table lists them piece by piece): the head is a function of the set
        perm = rng.permutation(9)
        inv = np.argsort(perm)
        tp = dict(ta, pos=ta["pos"][perm], dirv=ta["dirv"][perm], cell=ta["cell"][perm], src=inv[ta["src"]].astype(np.int32), dst=inv[ta["dst"]].astype(np.int32))
        op = JT.forward_batch(model, dict(TD.collate([item(tp, "a"), item(tb, "b")]), x=x))
        pm = torch.from_numpy(perm)
        assert torch.equal(op["allowed"][0], o1["allowed"][0][pm][:, pm]) and float((op["logit"][0] - o1["logit"][0][pm][:, pm]).abs().max()) < 1e-4, "the order of the nodes reaches the logits"
        # padding: frame b alone (6 nodes) = frame b in the batch padded to 9
        o_alone = JT.forward_batch(model, dict(TD.collate([item(tb, "b")]), x=x[1:]))
        assert o_alone["logit"].shape == (1, 6, 6) and torch.equal(o_alone["allowed"][0], o1["allowed"][1, :6, :6])
        assert float((o_alone["logit"][0] - o1["logit"][1, :6, :6]).abs().max()) < 1e-4 and float((o_alone["scale"][0] - o1["scale"][1, :6, :6]).abs().max()) < 1e-4, "a padded batch changes the frame's logits"
        must_fail(lambda: (_ for _ in ()).throw(AssertionError()) if float((JT.forward_batch(model, dict(b, x=x, valid=torch.ones(2, 9, dtype=torch.bool)))["logit"][1, :6, :6] - o1["logit"][1, :6, :6]).abs().max()) > 1e-4 else None)
    print(f"model: the loss by hand (two balanced groups + dual softmax + scale L1; GT links outside the gate counted), gradient into encoder / set blocks / pair / scale head and none into the pyramid, "
          f"scrambled flags / piece ids / node order and padding leave the logits alone (and an unmasked padding does not): ok; head parameters {sum(p.numel() for p in model.tok.parameters()) / 1e6:.3f} M")


def test_links():
    # GT on a line heading +y: short 0 -> 1, 1 -> 2; lane_long 2 -> 3 (10 m); junction 3 -> 4 (12 m, connector path, fitted scales 0.3 / 0.5); conn_short 4 -> 5; short 5 -> 6 with a chord of 0.4 m (outside the gate)
    t = table([[0, 0], [0, 2.5], [0, 5], [0, 15], [0, 27], [0, 29.5], [0, 29.9]], links=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], conn=[3, 4], scale={2: (0.33, 0.33), 3: (0.3, 0.5)})
    # predicted: 0 -> 1 (true short), 1 -> 0 (false short), 3 -> 4 (true junction, scales 0.4 / 0.5), 2 -> 4 (false long link of 22 m)
    src, dst = np.array([0, 1, 3, 2]), np.array([1, 0, 4, 4])
    n = JV.link_numbers([JV.link_counts(t, src, dst, np.array([1 / 3, 1 / 3, 0.4, 1 / 3]), np.array([1 / 3, 1 / 3, 0.5, 1 / 3]))])
    want = dict(link_short_P=0.5, link_short_R=0.25, link_short_n_gt=4, link_short_gate_out=0.25, link_lane_long_P=0.0, link_lane_long_R=0.0, link_lane_long_n_pred=1, link_junction_P=0.5, link_junction_R=1.0,
                link_junction_F1=2 / 3, link_junction_n_gt=1, link_junction_gate_out=0.0, link_conn_short_R=0.0, link_conn_short_n_gt=1, link_c0_8_P=0.5, link_c0_8_R=0.25, link_c8_20_P=1.0, link_c8_20_R=0.5,
                link_c20_n_gt=0, link_c20_n_pred=1, scale_err_a=0.1, scale_err_b=0.0, scale_n=1)
    assert all(abs(n[k] - v) < 1e-9 for k, v in want.items()), {k: (n[k], v) for k, v in want.items() if abs(n[k] - v) >= 1e-9}
    assert "link_conn_short_P" not in n and math.isnan(n["link_c20_R"])
    full = JV.gt_prediction(t)                                               # the GT links through the writer: fitted scales on the long links, recall 1 except the link outside the gate is still a link
    assert full["links"]["long"].tolist() == [False, False, True, True, False, False] and np.allclose(full["links"]["a"], [1 / 3, 1 / 3, 0.33, 0.3, 1 / 3, 1 / 3]) and np.allclose(full["links"]["b"][3], 0.5)
    n1 = JV.link_numbers([JV.link_counts(t, full["links"]["src"], full["links"]["dst"], full["links"]["a"], full["links"]["b"])])
    assert all(n1[f"link_{c}_{m}"] == 1.0 for c in JV.CLASSES for m in ("P", "R", "F1")) and n1["scale_err_a"] == 0.0 and n1["scale_n"] == 2
    # the node removal: the same for a token and a seed, another one for another seed, about the asked share, none at 0
    k1, k2 = JV.keep_nodes("tok", 20000, 0.17, 0), JV.keep_nodes("tok", 20000, 0.17, 0)
    assert np.array_equal(k1, k2) and abs(1 - k1.mean() - 0.17) < 0.01 and not np.array_equal(k1, JV.keep_nodes("tok", 20000, 0.17, 1)) and not np.array_equal(k1, JV.keep_nodes("tok2", 20000, 0.17, 0))
    assert JV.keep_nodes("tok", 7, 0.0, 0).all()
    # the contraction: without nodes 1 and 5 the GT is 0 -> 2 (one short link of 5 m, no fit), 2 -> 3, 3 -> 4 unchanged (scale and class kept), 4 -> 6 (through 5: on a connector path because 4 -> 5 is)
    s = JV.sub_table(t, np.array([True, False, True, True, True, False, True]))
    got = {(int(a), int(b)): (bool(lg), bool(c), None if np.isnan(sc[0]) else tuple(sc)) for a, b, lg, c, sc in zip(s["src"], s["dst"], s["long"], s["conn"], s["scale"])}
    assert got == {(0, 1): (False, False, None), (1, 2): (True, False, (0.33, 0.33)), (2, 3): (True, True, (0.3, 0.5)), (3, 4): (False, True, None)}, got
    assert len(s["pos"]) == 5 and np.array_equal(s["pos"][4], t["pos"][6]) and all(len(s[k]) == 5 for k in ("dirv", "cell", "flags", "piece", "sstart"))
    s2 = JV.sub_table(t, np.array([True, True, True, True, False, True, True]))                  # without node 4 the junction link 3 -> 4 and 4 -> 5 become 3 -> 5: long, on a connector path = a junction link, no fit
    j = [(int(a), int(b)) for a, b, lg, c in zip(s2["src"], s2["dst"], s2["long"], s2["conn"]) if lg and c]
    assert j == [(3, 4)] and np.isnan(s2["scale"][[(int(a), int(b)) for a, b in zip(s2["src"], s2["dst"])].index((3, 4))]).all(), (j, s2)
    assert JV.with_readout(full, "gtsec") is full and (JV.with_readout(full, "sec0")["nodes"]["section"] == 0).all() and JV.with_readout(full, "sec0")["links"] is full["links"]
    print("links: the tables by class and chord (true / false positives by the card's rule, GT outside the gate, scale error), the seeded node removal, the contraction of the GT links over removed nodes by hand: ok")


# ------------------------------------------------------------------------------------------------------------------------------ real frames
def _same(a, b, keys=None):
    """the two metric dicts agree on `keys` (all of a): equal, or both NaN"""
    return [k for k in (keys or a) if not (a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]))]


def test_writer():
    frames = [QE.val_frames()[k] for k in (0, 200, 400)]
    with JT.lattice(2.5):
        orc = [qformat.oracle_prediction(f.lane()) for f in frames]
        tabs = [TD.frame_table(f.lane(), 2.5) for f in frames]
        ref = QE.score_frames(orc, frames)["graph"]                           # the standard scorer on the oracle, on the lattice (the frames keep their lane GT: loaded before)
    k_of = {f.k: i for i, f in enumerate(frames)}
    res = JV.score_arms(frames, lambda f: (tabs[k_of[f.k]], dict(hi=orc[k_of[f.k]], w=JV.gt_prediction(tabs[k_of[f.k]]))), 2.5, log=None)
    assert (QT.N, QT.RES) == (50, 2.0)
    bad = _same(ref, res["hi"]["gtsec"])
    assert not bad and ref["topo_F1"] > 0.9, f"this file's scoring path is not score_frames: {bad}"
    for r in JV.READOUTS:
        bad = _same(res["hi"][r], res["w"][r], [k for k in res["hi"][r] if not k.startswith("scale_")])
        assert not bad, f"read-out {r}: the GT edge list through the writer does not score as oracle_prediction: {bad}"
    w = res["w"]["sec0"]
    assert all(w[f"link_{c}_{m}"] == 1.0 for c in JV.CLASSES for m in ("P", "R", "F1")) and w["scale_err_a"] == 0.0 and w["link_junction_n_gt"] > 0 and math.isnan(res["hi"]["sec0"]["scale_err_a"])
    # what the two read-outs do: with the GT piece ids a link across pieces is a piece of its own; with one section for all nodes there is none and every piece is a chain cut at the branch nodes
    p = JV.gt_prediction(tabs[0])
    k_gt, k_0 = (QE.readout(JV.with_readout(p, r))[1]["kind"] for r in ("gtsec", "sec0"))
    assert "link" in k_gt and set(k_0) == {"section"}, (set(k_gt), set(k_0))
    cut = dict(p, links={k: (v[1:] if v is not None else None) for k, v in p["links"].items()})                  # a mutation: one GT link less is seen by the link table
    m = JV.link_numbers([JV.link_counts(tabs[0], cut["links"]["src"], cut["links"]["dst"], cut["links"]["a"], cut["links"]["b"])])
    assert min(m[f"link_{c}_R"] for c in JV.CLASSES) < 1.0
    print(f"writer (3 VAL frames, 2.5 m): scoring path = score_frames, GT edge list through the writer = oracle_prediction under both read-outs (topo F1 gtsec {res['w']['gtsec']['topo_F1']:.4f}, sec0 {w['topo_F1']:.4f}; "
          f"junction F1 gtsec {res['w']['gtsec']['junction_F1']:.4f}, sec0 {w['junction_F1']:.4f}), sec0 has no link pieces: ok")


def test_gate(n=50, res=2.5):
    ds = TD.TokData("train", res, cache=None, feats=False)
    cnt = dict(short=[0, 0], long=[0, 0], junction=[0, 0])
    why, pairs, nodes, below = dict(chord_lt=0, chord_gt=0, forward=0, turn=0), [0, 0], [], []
    for i in np.linspace(0, len(ds) - 1, n).astype(int):
        t = ds.table(int(i))
        pos, dirv = torch.from_numpy(t["pos"]).float()[None], torch.from_numpy(t["dirv"]).float()[None]          # float32: the gate as the head computes it
        g, allowed = JT.pair_geometry(pos, dirv)
        g, allowed, M = g[0].numpy(), allowed[0].numpy(), len(t["pos"])
        out = ~allowed[t["src"], t["dst"]]
        for name, m in (("short", ~t["long"]), ("long", t["long"]), ("junction", t["long"] & t["conn"])):
            cnt[name][0] += int(out[m].sum())
            cnt[name][1] += int(m.sum())
        gl = g[t["src"], t["dst"]][out]
        why["chord_lt"] += int((gl[:, 2] * 20 < JT.MIN_CHORD).sum())
        why["chord_gt"] += int((gl[:, 2] * 20 > JT.MAX_CHORD).sum())
        why["forward"] += int(((gl[:, 3] <= 0) | (gl[:, 5] <= 0)).sum())
        why["turn"] += int((gl[:, 0] < QEN.COS_TURN).sum())
        below += (gl[:, 2] * 20).tolist()
        pairs[0] += int(allowed.sum())
        pairs[1] += M * (M - 1)
        nodes.append(M)
    tot = [cnt["short"][0] + cnt["long"][0], cnt["short"][1] + cnt["long"][1]]
    share = tot[0] / max(tot[1], 1)
    print(f"gate ({n} TRAIN frames, {res:g} m; chord {JT.MIN_CHORD:g} .. {JT.MAX_CHORD:g} m): GT links outside {tot[0]} of {tot[1]} = {100 * share:.3f} %; short {cnt['short'][0]} of {cnt['short'][1]} "
          f"({100 * cnt['short'][0] / max(cnt['short'][1], 1):.3f} %), long {cnt['long'][0]} of {cnt['long'][1]} ({100 * cnt['long'][0] / max(cnt['long'][1], 1):.3f} %), of the long the junction links {cnt['junction'][0]} of "
          f"{cnt['junction'][1]}; reasons (a link can have several) {why}; chords of the links outside: {np.round(np.percentile(below, [0, 50, 100]), 3).tolist() if below else []} m (min / median / max); "
          f"allowed pairs {100 * pairs[0] / max(pairs[1], 1):.1f} % of all ordered pairs; nodes per frame mean {np.mean(nodes):.0f}, max {max(nodes)}")
    assert share <= GATE_MAX, f"RULE of the card: {100 * share:.3f} % of the GT links are outside the allowed gate (limit {100 * GATE_MAX:g} %): stop and report, the gate is widened before any training"
    print("gate: ok")


def test_data():
    ds = TD.TokData("val", 2.5, cache=None)                                  # the init asserts: the 873 rows of data/splits.json['val'], in order, none without lane GT
    fr = QE.val_frames()
    assert len(ds) == 873 and ds.tokens == [f.token for f in fr] and ds.rows == [f.row for f in fr]
    o, q = ds[5], LD.QData("val_clean", map_mode="error", pixels=False)[5]   # QData's own item (on its 2 m defaults: outside every lattice block)
    assert o["token"] == q["token"] and o["row"] == q["row"] and o["x"].dtype == torch.float16 and torch.equal(o["x"], q["x"]), "x / token of a frame are not QData's"
    with JT.lattice(2.5):
        nt = qformat.oracle_prediction(fr[5].lane())["table"]
    assert np.array_equal(o["pos"], nt["pos"]) and np.array_equal(o["src"], nt["src"]) and np.array_equal(o["piece"], nt["piece"]) and np.array_equal(o["long"], nt["long"]) and (QT.N, QT.RES) == (50, 2.0)
    b = TD.collate([o, ds[6]])
    n0, n1 = len(o["pos"]), len(ds[6]["pos"])
    assert b["x"].shape == (2, 39, 200, 200) and b["pos"].shape == (2, max(n0, n1), 2) and b["valid"].sum(1).tolist() == [n0, n1] and len(b["table"]) == 2 and b["token"] == [ds.tokens[5], ds.tokens[6]]
    assert int(b["y"][0].sum()) == len(set(zip(o["src"].tolist(), o["dst"].tolist()))) and bool((b["y"][0, o["src"], o["dst"]] == 1).all()) and int(torch.isfinite(b["ys"][0, ..., 0]).sum()) <= int(o["long"].sum())
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:                    # the cache: written once, read back bit for bit, a table of another lattice refused
        dc = TD.TokData("val", 2.5, cache=tmp, feats=False)
        assert dc.cached() == 0 and os.path.basename(os.path.dirname(dc.dir)) == "l2.5_" + os.path.basename(LD.lane_store()[1])
        t1 = dc.table(5)
        assert dc.cached() == 1 and os.listdir(dc.dir) == [ds.tokens[5] + ".npz"]
        t2 = dc.table(5)
        assert all(np.array_equal(t1[k], t2[k], equal_nan=True) and t1[k].dtype == t2[k].dtype for k in TD.TABLE) and all(np.array_equal(t1[k], o[k], equal_nan=True) for k in TD.TABLE)
        d2 = TD.TokData("val", 2.0, cache=tmp, feats=False)
        shutil.copy(os.path.join(dc.dir, ds.tokens[5] + ".npz"), os.path.join(d2.dir, ds.tokens[5] + ".npz"))
        must_fail(d2.table, 5)
        assert TD.build_cache("val", 2.5, n=7, cache=tmp, log=None) == (7, 6) and dc.cached() == 7
    print(f"data: VAL = the 873 rows of the split file, x / token = QData's, the table = the oracle's node table at 2.5 m ({n0} nodes), batch padded to {max(n0, n1)} nodes with masks, cache round trip bit for bit: ok")


def test_pipeline():
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        argv = ["--name", "t_s0", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]
        JTT.main(argv, runs=runs, cache=None)
        rd = os.path.join(runs, "t_s0")
        assert all(os.path.exists(os.path.join(rd, f)) for f in (".done", "model.pt", "ckpt.pt", "config.json", "log.jsonl", "train.log"))
        must_fail(JTT.main, argv + ["--lr", "5e-4"], runs=runs, cache=None)    # another recipe in the same folder
        m, args = JT.load_model("t_s0", "cpu", runs)
        assert (args["seed"], args["lattice"], args["d"], args["steps"]) == (0, 2.5, JT.D, 20)
        base = torch.load(args["base"], map_location="cpu")["state"]
        assert all(torch.equal(v, base["pyr." + k]) for k, v in m.pyr.state_dict().items()), "the pyramid of the trained model is not the baseline's"
        fresh = JT.JTokModel(d=JT.D, base=args["base"])
        assert any(not torch.equal(a, b) for a, b in zip(m.tok.parameters(), fresh.tok.parameters())), "20 steps did not move the head"
        frames, tab = JV.val_tables(2.5)
        for drop in (0.0, 0.17):
            path = JV.dump(SimpleNamespace(run="t_s0", seed=0, drop=drop, drop_seed=0, cpu=True, max=3), runs)
            assert os.path.basename(path) == ("tok_val_n3.pkl" if drop == 0 else "tok_val_drop0.17_s0_n3.pkl")
            preds = pickle.load(open(path, "rb"))
            _, tabd = JV.val_tables(2.5, drop, 0)
            assert [p["token"] for p in preds] == [f.token for f in frames[:3]] and all(len(p["nodes"]["pos"]) == len(tabd(k)["pos"]) for k, p in enumerate(preds))
            assert drop == 0 or all(len(tabd(k)["pos"]) == int(JV.keep_nodes(frames[k].token, len(tab(k)["pos"]), drop, 0).sum()) < len(tab(k)["pos"]) for k in range(3))
            for k, p in enumerate(preds):                                    # every dumped link is an allowed pair with p >= 0.05; long = chord above 8 m; the short links carry 1/3
                lk, t = p["links"], tabd(k)
                al = JT.pair_geometry(torch.from_numpy(t["pos"]).float()[None], torch.from_numpy(t["dirv"]).float()[None])[1][0].numpy()
                assert len(lk["src"]) > 0 and al[lk["src"], lk["dst"]].all() and float(lk["p"].min()) >= JV.KEEP_P and np.allclose(lk["a"][~lk["long"]], 1 / 3) and (np.abs(lk["a"][lk["long"]] - 1 / 3) > 1e-6).all()
            res = JV.score_arms(frames[:3], lambda f: (tabd(f.k), {"p": preds[f.k]}), 2.5, 0.5, log=None)["p"]
            assert all(0.0 <= res[r][k] <= 1.0 for r in JV.READOUTS for k in ("topo_F1", "F1@1.0", "link_short_R", "link_junction_R")), res
            print(f"   pipeline, drop {drop:g}: 20 steps on the CPU, 3 VAL frames: topo F1 sec0 {res['sec0']['topo_F1']:.3f} / gtsec {res['gtsec']['topo_F1']:.3f}, short-link recall {res['sec0']['link_short_R']:.3f} (a model of 20 steps: no result)")
        card = JV.make_card("t", 3, res["sec0"], "sec0", run="t_s0", seed=0)
        assert card["metrics"]["topo_F1"] == res["sec0"]["topo_F1"] and all(math.isfinite(v) for v in card["metrics"].values()) and "sec0" in card["protocol"]["readout"]
    print("pipeline: trainer (smoke, resume contract, another recipe refused) -> model.pt (frozen pyramid = the baseline's, head moved) -> dump (also with 17 % of the nodes removed) -> scores -> card: ok")


TESTS = [test_geometry, test_lattice, test_rules, test_model, test_links, test_writer, test_gate, test_data, test_pipeline]

if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtok ok")
