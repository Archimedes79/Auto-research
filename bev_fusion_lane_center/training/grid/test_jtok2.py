"""E359 tests of the token head with the node-type head and the per-source softmax (`jtok2.py`, `train/jtok2_train.py`, `evaluation/jtok2_eval.py`); CPU only (the GPU is hidden), 2 threads. Reuses the helpers of test_jtok.py.

    wsl bash tools/env/_py14.sh training/grid/test_jtok2.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

No data: likelihood (the loss of the module docstring by hand on a 6-node frame: a source with two GT successors, one with none, one whose only GT link is outside the gate, nodes without a candidate masked; a pair
logit outside the gate or of a source without a GT link gets no gradient), probs (sums to 1 per source and class, 0 outside the gate, the edge probability is at most the type probability, long_in changes the long ranking),
model (gradient into the encoder from the type head AND from the softmax, none into the frozen pyramid; flags / piece ids / node order / padding leave the outputs alone), decodes (record with the tie rule, cond03, top-3,
the writer, by hand), diag (the attribution numbers on a two-frame dump by hand), overfit (500 CPU steps on three hand-made frames: the decode of record reproduces the GT links of the fork-free frames exactly, P = R = 1,
0 against the transposed GT; the fork frame puts its probability on the two GT successors, which only the multi-successor decode can deliver).
Real frames: pipeline (20 training steps on the CPU -> model.pt -> dump of 3 VAL frames, also degraded -> scores of both decodes -> diagnosis; a stored dump is never overwritten).
"""
import json
import math
import os
import pickle
import sys
import tempfile
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                      # this test never touches the GPU
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in (".", "..", "../data", "../data/prep", "../train", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
import jtok as JT  # noqa: E402
import jtok2 as J2  # noqa: E402
import jtok2_eval as E2  # noqa: E402
import jtok2_train as J2T  # noqa: E402
import jtok_eval as JV  # noqa: E402
import lane_tok_data as TD  # noqa: E402
import test_jtok as TJ  # noqa: E402
from test_jtok import item, must_fail, rot, table  # noqa: E402

torch.set_num_threads(2)
SCRATCH = os.path.join(JT.ROOT, "temp", "e359")
os.makedirs(SCRATCH, exist_ok=True)
lsig = lambda x: -math.log1p(math.exp(-x))  # noqa: E731
lse = lambda *v: math.log(sum(math.exp(x) for x in v))  # noqa: E731


def _hand():
    """the 6-node frame of the likelihood test: -> (out dict with requires_grad leaves, batch)"""
    short = {(0, 1): 3.0, (0, 2): 4.0, (0, 3): 5.0, (1, 2): 3.0, (1, 3): 4.0, (2, 3): 3.0, (3, 4): 3.0}
    long = {(0, 4): 10.0, (0, 5): 14.0, (1, 4): 12.0}
    al, ch = torch.zeros(1, 6, 6, dtype=torch.bool), torch.zeros(1, 6, 6)
    for (i, j), c in {**short, **long}.items():
        al[0, i, j], ch[0, i, j] = True, c
    z = torch.full((1, 6, 6), 7.0)                                           # 7.0 outside the gate: a leak would show
    for (i, j), v in zip([*short, *long], [0.3, -0.2, 0.9, 0.5, -0.4, 0.1, 0.7, 1.1, -0.6, 0.2]):
        z[0, i, j] = v
    ty = torch.tensor([[[0.4, 0.1, 0.8], [-0.3, 0.2, -0.5], [0.6, 0.3, 0.2], [-1.0, 0.4, 1.3], [2.0, -0.7, -0.9], [0.5, 1.5, 0.9]]])     # long_out, long_in, short_out of the 6 nodes
    sc = torch.full((1, 6, 6, 2), 0.4)
    y = torch.zeros(1, 6, 6)
    for i, j in ((0, 1), (0, 2), (0, 4), (2, 0), (3, 4)):
        y[0, i, j] = 1.0
    ys = torch.full((1, 6, 6, 2), float("nan"))
    ys[0, 0, 4] = torch.tensor([0.3, 0.5])
    out = dict(logit=z.requires_grad_(), types=ty.requires_grad_(), allowed=al, chord=ch, scale=sc)
    return out, dict(y=y, ys=ys, valid=torch.ones(1, 6, dtype=torch.bool))


def test_likelihood():
    out, b = _hand()
    z, ty = out["logit"][0].detach().tolist(), out["types"][0].detach().tolist()
    lo, li, so = ([ty[n][k] for n in range(6)] for k in range(3))
    # short class, sources 0 1 2 3 (4 and 5 have no short candidate: masked). 0: GT {1, 2} of {1, 2, 3}; 1: none; 2: none (its GT link 2 -> 0 is outside the gate); 3: GT {4}, the only candidate
    l0 = -(lsig(so[0]) + lse(z[0][1], z[0][2]) - lse(z[0][1], z[0][2], z[0][3]))
    nll_s = (l0 - lsig(-so[1]) - lsig(-so[2]) - lsig(so[3])) / 4
    # long class, sources 0 and 1. z' = z + logsigmoid(long_in of the destination). 0: GT {4} of {4, 5}; 1: none
    z04, z05 = z[0][4] + lsig(li[4]), z[0][5] + lsig(li[5])
    nll_l = (-(lsig(lo[0]) + z04 - lse(z04, z05)) - lsig(-lo[1])) / 2
    bce = lambda x, t: -(lsig(x) if t else lsig(-x))  # noqa: E731
    bce_in = (bce(li[4], 1) + bce(li[5], 0)) / 2                             # the nodes that are a long candidate destination: 4 and 5
    loss, info = J2.tok2_loss(out, b)
    want = nll_s + nll_l + bce_in + 0.5 * 0.1
    assert abs(info["nll_s"] - nll_s) < 1e-5 and abs(info["nll_l"] - nll_l) < 1e-5 and abs(info["bce_in"] - bce_in) < 1e-5 and abs(info["scale_l1"] - 0.1) < 1e-6 and abs(float(loss) - want) < 1e-5, (info, nll_s, nll_l, bce_in)
    assert (info["gt_links"], info["pos_pairs"], info["gt_outside"], info["allowed"]) == (5.0, 4.0, 1.0, 10.0), info
    # nodes without a candidate of a class are masked from it entirely: changing their type logits (not the ones that enter a loss term) changes nothing
    out2 = dict(out, types=out["types"].detach().clone())
    out2["types"][0, 4, 0] = out2["types"][0, 5, 0] = out2["types"][0, 4, 2] = out2["types"][0, 5, 2] = out2["types"][0, 2, 0] = out2["types"][0, 3, 0] = -9.0      # long_out of 2 3 4 5, short_out of 4 5
    out2["logit"] = out["logit"].detach().clone()
    out2["logit"][0, 4, 0] = out2["logit"][0, 5, 3] = -9.0                   # pairs outside the gate
    assert abs(float(J2.tok2_loss(out2, b)[0]) - float(loss)) < 1e-6, "a masked node or a pair outside the gate reaches the loss"
    loss.backward()
    zg = out["logit"].grad[0].abs()
    assert float(zg[1].sum()) == 0.0 and float(zg[2].sum()) == 0.0 and float(zg[3, 0:4].sum()) == 0.0, "a source without a GT link must not move the pair logits (only its type logit)"
    assert float(zg[0, 1:5].min()) > 0.0 and float(zg[0, 4:6].min()) > 0.0 and float(zg[4:].sum()) == 0.0 and float(out["types"].grad[0, 1, 2].abs()) > 0.0
    assert float(out["logit"].grad[0, 3, 4].abs()) == 0.0, "a source with ONE candidate has a constant softmax: no gradient on its pair logit"
    print("likelihood: the loss by hand (a source with two GT successors, one with none, one whose GT link is outside the gate, nodes without a candidate masked, long_in in the ranking and its BCE), no gradient "
          "into the pair logits of sources without a GT link: ok")


def test_probs():
    out, _ = _hand()
    pr = J2.probs({k: v.detach() for k, v in out.items()})
    s, tp = pr["s"][0], pr["tp"][0]
    assert abs(float(s[0, 1:4].sum()) - 1) < 1e-6 and abs(float(s[0, 4:6].sum()) - 1) < 1e-6 and abs(float(s[1, 2:4].sum()) - 1) < 1e-6 and abs(float(s[1, 4]) - 1) < 1e-6, "a class of a source sums to 1"
    z = out["logit"][0].detach()
    assert abs(float(s[0, 1]) - float(torch.softmax(z[0, 1:4], 0)[0])) < 1e-6, "short: the plain softmax of z"
    zp = z[0, 4:6] + F.logsigmoid(out["types"][0, 4:6, 1].detach())
    assert abs(float(s[0, 4]) - float(torch.softmax(zp, 0)[0])) < 1e-6, "long: the softmax of z + logsigmoid(long_in)"
    assert float(s[2, 0]) == 0.0 and float(s[4].abs().sum()) == 0.0 and float(s[0, 0]) == 0.0 and float(s[3, 5]) == 0.0, "no probability outside the gate / for a node without candidates"
    pe = tp[:, 0][:, None] * s * pr["longm"][0] + tp[:, 2][:, None] * s * ~pr["longm"][0]
    assert (pe <= torch.where(pr["longm"][0], tp[:, 0][:, None], tp[:, 2][:, None]) + 1e-6).all() and float(pe[0, 1:4].sum()) <= float(tp[0, 2]) + 1e-6
    # long_in changes the RANKING: z of 0 -> 4 above 0 -> 5, then the destination 5 is pulled up and 4 pushed down
    det = {k: v.detach() for k, v in out.items()}

    def top(b4, b5):
        ty = det["types"].clone()
        ty[0, 4, 1], ty[0, 5, 1] = b4, b5
        return int(J2.probs(dict(det, types=ty))["s"][0, 0, 4:6].argmax())
    assert (top(5.0, -5.0), top(-5.0, 5.0)) == (0, 1), "long_in does not change the ranking of the long candidates"
    print("probs: the softmax of each class of a source sums to 1, 0 outside the gate, edge probability <= type probability, long_in turns the long ranking: ok")


def _frames():
    ta = table([[0, -10 + 2.5 * k] for k in range(5)] + [[3, 12], [-8, 14], [10, 20], [-20, -20]], links=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 5), (3, 6)], conn=[4, 5, 6], scale={4: (0.2, 0.7), 5: (0.6, 0.3), 6: (0.4, 0.5)})
    tb = table([[5 + 2.5 * k, 3.0 * k] for k in range(6)], dirv=[rot(-40)] * 6, links=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    tf = table([[0, -10 + 2.5 * k] for k in range(5)] + [[3, 12], [-8, 14], [10, 20], [-20, -20]], links=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6)], conn=[4, 5], scale={4: (0.3, 0.5), 5: (0.4, 0.6)})
    return ta, tb, tf


def test_model():
    torch.manual_seed(0)
    ta, tb, _ = _frames()
    b = TD.collate([item(ta, "a"), item(tb, "b")])
    model = J2.JTok2Model()
    x = torch.randn(2, 39, 200, 200)
    out = J2.forward_batch(model, dict(b, x=x))
    assert out["types"].shape == (2, 9, 3) and out["logit"].shape == (2, 9, 9) and out["scale"].shape == (2, 9, 9, 2) and out["types"].dtype == torch.float32
    grad = lambda mod: [float(p.grad.abs().sum()) if p.grad is not None else None for p in mod.parameters()]  # noqa: E731
    # the type head alone: a loss on the types reaches the encoder and the set blocks, not the pair head
    F.binary_cross_entropy_with_logits(out["types"], torch.ones_like(out["types"])).backward(retain_graph=True)
    assert all(v is not None and v > 0 for v in grad(model.tok.types)) and all(v is not None and v > 0 for v in grad(model.tok.enc)) and sum(v or 0 for v in grad(model.tok.blocks)) > 0
    assert all(v is None for v in grad(model.tok.pair)), "the type loss must not reach the pair head"
    model.zero_grad(set_to_none=True)
    # the softmax part alone: types detached, the gradient reaches the pair head and the encoder (through the pair logits), not the type head
    loss, info = J2.tok2_loss(dict(out, types=out["types"].detach()), b)
    loss.backward(retain_graph=True)
    assert all(v is not None and v > 0 for v in grad(model.tok.pair)) and all(v is not None and v > 0 for v in grad(model.tok.enc)) and all(v is None for v in grad(model.tok.types)), "the softmax part"
    model.zero_grad(set_to_none=True)
    loss, info = J2.tok2_loss(out, b)                                       # the whole loss: every parameter of the head (the scale head through the L1 on the long GT links)
    loss.backward()
    assert all(v is not None and v > 0 for mod in (model.tok.pair, model.tok.scale, model.tok.types, model.tok.enc) for v in grad(mod)), "a part of the head gets no gradient"
    assert all(p.grad is None for p in model.pyr.parameters()), "the frozen pyramid must not get a gradient"
    # no flags / piece ids / cells, no node order, no padding reach the outputs
    model.eval()
    rng = np.random.RandomState(0)
    scr = lambda t: dict(t, flags=rng.rand(*t["flags"].shape) > 0.5, piece=rng.permutation(len(t["pos"])).astype(np.int32), sstart=rng.rand(len(t["pos"])) > 0.5, cell=rng.permutation(t["cell"]), conn=~t["conn"])  # noqa: E731
    with torch.no_grad():
        o1 = J2.forward_batch(model, dict(b, x=x))
        o2 = J2.forward_batch(model, dict(TD.collate([item(scr(ta), "a"), item(scr(tb), "b")]), x=x))
        assert all(torch.equal(o1[k], o2[k]) for k in ("logit", "scale", "types", "allowed")), "flags / piece ids / cells reach the outputs"
        perm = rng.permutation(9)
        tp_ = dict(ta, pos=ta["pos"][perm], dirv=ta["dirv"][perm], cell=ta["cell"][perm], src=np.argsort(perm)[ta["src"]].astype(np.int32), dst=np.argsort(perm)[ta["dst"]].astype(np.int32))
        op = J2.forward_batch(model, dict(TD.collate([item(tp_, "a"), item(tb, "b")]), x=x))
        pm = torch.from_numpy(perm)
        assert float((op["logit"][0] - o1["logit"][0][pm][:, pm]).abs().max()) < 1e-4 and float((op["types"][0] - o1["types"][0][pm]).abs().max()) < 1e-4, "the order of the nodes reaches the outputs"
        o_alone = J2.forward_batch(model, dict(TD.collate([item(tb, "b")]), x=x[1:]))
        assert float((o_alone["types"][0] - o1["types"][1, :6]).abs().max()) < 1e-4 and float((o_alone["logit"][0] - o1["logit"][1, :6, :6]).abs().max()) < 1e-4, "a padded batch changes the frame's outputs"
        o_open = J2.forward_batch(model, dict(b, x=x, valid=torch.ones(2, 9, dtype=torch.bool)))
        assert float((o_open["types"][1, :6] - o1["types"][1, :6]).abs().max()) > 1e-4, "the padding mask does nothing"
    print(f"model: gradient into the encoder from the type head and from the softmax part, the type loss does not reach the pair head, none into the pyramid; scrambled flags / piece ids / node order and padding "
          f"leave the outputs alone: ok; head parameters {sum(p.numel() for p in model.tok.parameters()) / 1e6:.3f} M")


def _toy_decode():
    """5 nodes: 0 short {1, 2} (tie 0.5 / 0.5), long {3, 4} (0.6 / 0.4); 1 short {2}; 2 short {3, 4} (0.2 / 0.8); positions chosen so that the classes are the chords"""
    M = 5
    s, cand, longm = np.zeros((M, M), np.float32), np.zeros((M, M), bool), np.zeros((M, M), bool)
    for (i, j), v, lg in (((0, 1), 0.5, False), ((0, 2), 0.5, False), ((0, 3), 0.6, True), ((0, 4), 0.4, True), ((1, 2), 1.0, False), ((2, 3), 0.2, False), ((2, 4), 0.8, False)):
        s[i, j], cand[i, j], longm[i, j] = v, True, lg
    tp = np.array([[0.9, 0.1, 0.9], [0.1, 0.1, 0.3], [0.1, 0.1, 0.5], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], np.float32)
    pos = np.array([[0, 0], [0, 3], [0, 6], [0, 12], [0, 8.5]], float)
    return s, cand, longm, tp, table(pos)


def test_decode():
    s, cand, longm, tp, t = _toy_decode()
    pairs = lambda r: sorted(zip(*(x.tolist() for x in r)))  # noqa: E731
    # rec: per source and class the highest s (the tie 0 -> 1 / 0 -> 2 goes to the LOWER index), kept at the type probability >= 0.5 (node 2: exactly 0.5 is kept, node 1: 0.3 is not)
    assert pairs(E2.decode_record(s, cand, longm, tp)) == [(0, 1), (0, 3), (2, 4)]
    assert pairs(E2.decode_cond(s, cand, longm, tp)) == [(0, 1), (0, 2), (0, 3), (0, 4), (2, 4)]                          # s >= 0.3 of the gated sources: node 1 is not gated, 2 -> 3 (0.2) is below
    assert pairs(E2.decode_cond(s, cand, longm, tp, 0.45)) == [(0, 1), (0, 2), (0, 3), (2, 4)]
    top = E2.top_candidates(s, cand, longm)
    got = sorted(zip(top["src"].tolist(), top["dst"].tolist(), top["long"].tolist(), [round(v, 2) for v in top["s"].tolist()]))
    assert got == [(0, 1, False, 0.5), (0, 2, False, 0.5), (0, 3, True, 0.6), (0, 4, True, 0.4), (1, 2, False, 1.0), (2, 3, False, 0.2), (2, 4, False, 0.8)]
    t1 = E2.top_candidates(s, cand, longm, k=1)
    assert sorted(zip(t1["src"].tolist(), t1["dst"].tolist())) == [(0, 1), (0, 3), (1, 2), (2, 4)] and t1["s"].shape == (4,)
    sc = np.full((5, 5, 2), 0.4, np.float32)
    sc[0, 3] = (0.2, 0.7)
    fd = E2.frame_dump(t, s, cand, longm, tp, sc)
    lk = fd["rec"]["links"]
    assert (lk["src"].tolist(), lk["dst"].tolist(), lk["long"].tolist()) == ([0, 2, 0], [1, 4, 3], [False, False, True]) and (lk["p"] == 1).all()
    assert np.allclose(lk["a"], [1 / 3, 1 / 3, 0.2]) and np.allclose(lk["b"], [1 / 3, 1 / 3, 0.7]) and np.allclose(fd["rec"]["p_edge"], [0.9 * 0.5, 0.5 * 0.8, 0.9 * 0.6]) and fd["types"].shape == (5, 3)
    assert len(fd["cond03"]["links"]["src"]) == 5 and (fd["rec"]["nodes"]["pos"] == t["pos"]).all()
    # the padded batch is cropped to the frame
    pad = lambda a_, k: np.pad(a_, [(0, k)] * 2 + [(0, 0)] * (a_.ndim - 2))  # noqa: E731
    fp = E2.frame_dump(t, pad(s, 2), pad(cand, 2), pad(longm, 2), np.pad(tp, [(0, 2), (0, 0)], constant_values=0.9), pad(sc, 2))
    assert fp["rec"]["links"]["src"].tolist() == lk["src"].tolist() and len(fp["types"]) == 5
    assert E2.auc(np.array([1, 2, 2, 3.0]), np.array([0, 1, 0, 1])) == 0.875 and math.isnan(E2.auc(np.ones(3), np.ones(3, bool))) and E2.auc(np.array([np.inf, np.inf, 1.0]), np.array([1, 0, 0])) == 0.75
    print("decode: rec (argmax per source and class, the tie goes to the lower index, 0.5 on the type probability kept), cond03, top-3, the writer (scales, p = 1, p_edge), padding cropped, AUC with ties: ok")


def test_diag():
    # frame A: a line of 6 nodes (2.5 m apart, then 15.3 m to node 5): GT short 0-1-2-3-4, long 4 -> 5. frame B: 0 (0,0), 1 (0,12), 2 (-3,16), 3 (0,30): GT long 0 -> 1, 1 -> 3
    ta = table([[0, 0], [0, 2.5], [0, 5], [0, 7.5], [0, 10], [3, 25]], links=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    tb = table([[0, 0], [0, 12], [-3, 16], [0, 30]], links=[(0, 1), (1, 3)])
    f32 = lambda rows: np.array(rows, np.float32)  # noqa: E731
    cd = lambda rows: dict(src=np.array([r[0] for r in rows]), dst=np.array([r[1] for r in rows]), s=np.array([r[2] for r in rows], np.float32), long=np.array([r[3] for r in rows], bool))  # noqa: E731
    pa = dict(types=f32([[.1, .1, .9], [.1, .1, .8], [.1, .1, .4], [.1, .1, .9], [.9, .2, .2], [.1, .9, .1]]), cand=cd([(0, 1, 1, 0), (1, 2, 1, 0), (3, 4, 1, 0), (4, 5, 1, 1)]),
              rec=JV.edge_prediction(ta, [0, 1, 3, 4], [1, 2, 4, 5]))
    pb = dict(types=f32([[.8, .1, .1], [.75, .8, .1], [.8, .1, .1], [.1, .9, .1]]), cand=cd([(0, 2, .5, 1), (0, 1, .3, 1), (0, 3, .2, 1), (1, 3, .9, 1), (1, 2, .1, 0)]),
              rec=JV.edge_prediction(tb, [0, 1, 2], [2, 3, 3]))
    d = E2.diagnose([pa, pb], [ta, tb])
    lo, li, so = (d["type"][k] for k in J2.TYPES)
    assert d["nodes"] == 10 and (lo["n_pos"], li["n_pos"], so["n_pos"]) == (3, 3, 4)
    assert abs(lo["precision@0.5"] - 0.75) < 1e-9 and lo["recall@0.5"] == 1.0 and abs(lo["f1@0.7"] - 6 / 7) < 1e-9 and abs(lo["auc"] - 19.5 / 21) < 1e-9, lo
    assert li["auc"] == 1.0 and so["auc"] == 1.0 and so["precision@0.5"] == 1.0 and so["recall@0.5"] == 0.75 and so["recall@0.3"] == 1.0, (li, so)
    assert d["top1"]["long"]["sources"] == 3 and d["top1"]["long"]["hits"] == 2 and d["top1"]["short"]["sources"] == 4 and d["top1"]["short"]["hits"] == 3, d["top1"]       # source 2 has no stored candidate: a miss
    c = d["composition"]
    assert (c["delivered_long"], c["true_long"], c["gt_long"], c["gt_long_sources"]) == (4, 2, 3, 3) and c["P_link"] == 0.5 and abs(c["R_link"] - 2 / 3) < 1e-9 and abs(c["F1_src"] - 4 / 7) < 1e-9, c
    assert d["gate"]["outside"] == 0 and d["gate"]["gt_links"] == 7
    g = d["geometry_null"]                                                   # nearest short candidate ahead: 2.5 m for nodes 0 - 3, none for 4 and 5 (node 5 is 15.3 m away); frame B: none inside 8 m except 1 -> 2 (5 m)
    assert g["nodes_without_short_candidate"] == 5 and abs(g["auc_long_out"] - 15 / 21) < 1e-9 and g["auc_no_short_successor"] == 1.0, g
    assert abs(g["rule_no_short_candidate_precision"] - 0.4) < 1e-9 and abs(g["rule_no_short_candidate_recall"] - 2 / 3) < 1e-9, g
    assert d["attribution"]["type_head_works"] is False and d["attribution"]["softmax_works"] is False
    pa2 = dict(pa, types=pa["types"][:5])
    must_fail(E2.diagnose, [pa2, pb], [ta, tb])
    print(f"diag: type precision / recall / F1 / AUC, oracle-source top-1 (a source without a stored candidate is a miss), composition, gate, geometry null on a two-frame dump by hand: ok (null AUC {g['auc_long_out']:.3f})")


def _pred_numbers(t, fd, which="rec"):
    lk = fd[which]["links"]
    return JV.link_numbers([JV.link_counts(t, lk["src"], lk["dst"], lk["a"], lk["b"])])


def test_overfit():
    torch.manual_seed(0)
    ta, tb, tf = _frames()
    tabs = [ta, tb, tf]
    b = TD.collate([item(t, k) for t, k in zip(tabs, "abf")])
    P = dict(feat2=torch.randn(3, 144, 50, 50), feat1=torch.randn(3, 48, 100, 100), centre1=torch.randn(3, 12, 100, 100))
    model = J2.JTok2()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(500):
        loss, info = J2.tok2_loss(model(P, b["pos"], b["dirv"], b["valid"]), b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    model.eval()
    with torch.no_grad():
        out = model(P, b["pos"], b["dirv"], b["valid"])
    pr = J2.probs(out)
    arr = lambda x: x.detach().cpu().numpy()  # noqa: E731
    s, tp, lm, al, sc = arr(pr["s"]), arr(pr["tp"]), arr(pr["longm"]), arr(out["allowed"]), arr(out["scale"])
    fds = [E2.frame_dump(t, s[i], al[i], lm[i], tp[i], sc[i]) for i, t in enumerate(tabs)]
    for i in (0, 1):                                                         # the fork-free frames: the argmax decode of record reproduces every GT link and no other
        n = _pred_numbers(tabs[i], fds[i])
        assert all(n[f"link_{c}_{m}"] == 1.0 for c in ("short", "junction") for m in ("P", "R", "F1") if n[f"link_{c}_n_gt"] > 0) and n["link_short_n_gt"] > 0, (i, n)
        assert all(n[f"link_{c}_n_pred"] == n[f"link_{c}_n_gt"] for c in ("short", "junction")), (i, n)             # and no link that is not a GT link
    n0 = _pred_numbers(ta, fds[0])
    assert n0["link_junction_n_gt"] == 3 and n0["scale_n"] == 3 and n0["scale_err_a"] < 0.1 and n0["scale_err_b"] < 0.1, n0
    rev = lambda t_: dict(t_, src=t_["dst"], dst=t_["src"])  # noqa: E731
    for i in (0, 1):                                                         # the mutation: against the TRANSPOSED GT nothing is right
        lk = fds[i]["rec"]["links"]
        nr = JV.link_numbers([JV.link_counts(rev(tabs[i]), lk["src"], lk["dst"], lk["a"], lk["b"])])
        assert nr["link_short_F1"] == 0.0 and (nr["link_junction_F1"] == 0.0 or (i == 1 and math.isnan(nr["link_junction_F1"]))), nr       # frame b has no junction link at all: NaN
    # the fork frame: node 4 has TWO long GT successors; its probability mass sits on them (the multi-positive likelihood does not force an even split: the argmax delivers one of them, so recall 1/2)
    mass = float(s[2, 4, 5] + s[2, 4, 6])
    nf = _pred_numbers(tabs[2], fds[2])
    assert mass > 0.95 and nf["link_junction_P"] == 1.0 and abs(nf["link_junction_R"] - 0.5) < 1e-9 and nf["link_short_F1"] == 1.0, (mass, nf)
    print(f"overfit: 500 CPU steps on three hand-made frames: loss {first:.3f} -> {float(loss):.3f}; the decode of record gives P = R = F1 = 1.0 (short and junction, merge and a node with a short AND a long successor included) "
          f"on the fork-free frames, 0.0 against the transposed GT; the fork frame carries a mass of {mass:.3f} on its two successors (argmax recall 0.5): ok")


def test_pipeline():
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        argv = ["--name", "t2_s0", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]
        must_fail(J2T.main, argv[:4], runs=runs, cache=None)
        J2T.main(argv, runs=runs, cache=None)
        rd = os.path.join(runs, "t2_s0")
        assert all(os.path.exists(os.path.join(rd, f)) for f in (".done", "model.pt", "ckpt.pt", "config.json", "log.jsonl", "train.log"))
        must_fail(J2T.main, argv + ["--lr", "5e-4"], runs=runs, cache=None)
        m, args = J2.load_model("t2_s0", "cpu", runs)
        assert (args["seed"], args["lattice"], args["steps"]) == (0, 2.5, 20) and hasattr(m.tok, "types")
        rows = [json.loads(r) for r in open(os.path.join(rd, "log.jsonl"))]
        assert all(k in rows[0] for k in ("nll_s", "nll_l", "bce_in")), rows[0]
        frames, tab = JV.val_tables(2.5)
        for drop in (0.0, 0.17):
            path = E2.dump(SimpleNamespace(run="t2_s0", seed=0, drop=drop, drop_seed=0, cpu=True, max=3), runs)
            assert os.path.basename(path) == ("tok2_val_n3.pkl" if drop == 0 else "tok2_val_drop0.17_s0_n3.pkl")
            must_fail(E2.dump, SimpleNamespace(run="t2_s0", seed=0, drop=drop, drop_seed=0, cpu=True, max=3), runs)       # a stored dump is never overwritten
            preds = pickle.load(open(path, "rb"))
            _, tabd = JV.val_tables(2.5, drop, 0)
            tabs = [tabd(k) for k in range(3)]
            assert [p["token"] for p in preds] == [f.token for f in frames[:3]] and all(p["types"].shape == (len(t["pos"]), 3) for p, t in zip(preds, tabs))
            for p, t in zip(preds, tabs):
                al = JT.pair_geometry(torch.from_numpy(t["pos"]).float()[None], torch.from_numpy(t["dirv"]).float()[None])[1][0].numpy()
                for d_ in E2.DECODES:
                    lk = p[d_]["links"]
                    assert al[lk["src"], lk["dst"]].all() and (lk["p"] == 1).all() and np.allclose(lk["a"][~lk["long"]], 1 / 3)
                assert len(p["rec"]["links"]["src"]) <= 2 * len(t["pos"]) and p["cand"]["s"].max() <= 1.0 + 1e-6
            res = JV.score_arms(frames[:3], lambda f: (tabd(f.k), {d_: preds[f.k][d_] for d_ in E2.DECODES}), 2.5, 0.5, log=None)
            assert all(0.0 <= res[d_][r][k] <= 1.0 for d_ in E2.DECODES for r in JV.READOUTS for k in ("topo_F1", "link_short_R")), res
            dg = E2.diagnose(preds, tabs)
            assert dg["nodes"] == sum(len(t["pos"]) for t in tabs) and set(dg["type"]) == set(J2.TYPES)
            print(f"   pipeline, drop {drop:g}: 20 steps, 3 VAL frames: rec topo F1 sec0 {res['rec']['sec0']['topo_F1']:.3f}, cond03 {res['cond03']['sec0']['topo_F1']:.3f}; diag long_out AUC {dg['type']['long_out']['auc']:.3f} (a model of 20 steps: no result)")
    print("pipeline: trainer (smoke: 20 updates, another recipe refused, the four loss parts logged) -> model.pt -> dump (also degraded; never overwritten) -> scores of both decodes -> diagnosis: ok")


TESTS = [test_likelihood, test_probs, test_model, test_decode, test_diag, test_overfit, test_pipeline]

if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtok2 ok")
