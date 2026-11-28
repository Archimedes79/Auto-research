"""Tests of the token head on predicted nodes (`jtoken.py`, `loss_jtoken.py`, `train/jtoken_train.py`); CPU only (the GPU is hidden), 2 threads, SYNTHETIC batches that follow the batch dict of the card (2a): no cache needed.

    python net/tests/test_jtoken.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

likelihood (the loss by hand on two frames of 5 tokens, the second with 2 padded tokens, other source counts and no long source: a source with three GT successors (one of them with `ylong` against the token
chord), a source with none, a source whose GT link lies outside the gate, sources without a candidate masked, the per-frame normalisation; mutants: no none slot, a swapped link direction, no gate, a global mean over
the sources, no valid mask; the none slot trains the empty sources), none_weight (W by hand, mutants), wiring_logp (a gate and padding mask, sums to 1), decode (the thresholds, the none rule, the kept ends, a padded source and destination;
mutants: other thresholds / ends / no valid mask), head (shapes, every parameter gets gradient, the gate and the chord classes against an independent distance, order and padding invariance with a padding mutant),
key_bias (the head's own p_none gets gradient from layer 2 on after the switch and not before; the step mirror survives load_state_dict; no gradient reaches the cached inputs), writer_free (the forward reads no GT
key), overfit (8 fixed frames, <= 300 steps, loss below 0.5 x its start, the decode of record reproduces the GT links), trainer (the smoke run on synthetic batches, kill, resume, another recipe refused, no loader
import; a non-smoke run through a fake loader that returns (None, batches) with the ident argument), arch_options (defaults bit for bit; a stored run's removed options (`jtoken_train.REMOVED`: D117 and the closed experiments of
the tag v1) at their off value resume and load, another value is refused with the option's name and its tag), live_inputs (S4: the default = the forward of before bit for bit, outputs and parameter gradients, no input gradient; live: the same values, a gradient on every float
input, the full gradient against float64 central differences; textual mutants; build_head / load_model read it, absent = False).
head_seed (--seed 0 --head-seed 2 reads the seed-0 cache, initialises (the trainer's step-0 head = build_head after seed_all(2)) and orders the data as --seed 2, also through a shuffling DataLoader as the loader; --head-seed = --seed is the default run).
champion (the default path against RECORDED numbers: the forward, the loss and the gradient sums of a seeded head at two head steps and of the stored E373 head; the 20 smoke steps of the champion recipe; one step on the real
TRAIN cache starts as E373 did); the soft output layers, each against a brute-force or hand case (skip_suppress, close_gaps, fork_gate, refine_positions: off = the head bit for bit, build_head) and layers_train (the loss
behind the wiring layers and every gradient finite, the none slot never emptied). The tests of the removed switches (E376 pos_loss max, E379 distillation, E381 noise, E392 rounds, E395 position head) are in the tag v1.
"""
import json
import math
import os
import sys
import tempfile
import types

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                      # this test never touches the GPU
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in ("../model", "../prep", "../loss", "../../training", "../../training/data", "../../training/data/prep", "../../training/train", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
import bflc_paths as BP  # noqa: E402
import jtok as JT  # noqa: E402
import jtoken as JK  # noqa: E402
import loss_jtoken as LJK  # noqa: E402
from merged_source import merged  # noqa: E402
# the checks below take "the model module" as a parameter (jtoken or a textual mutant of it) and call its losses on it: jtoken gets the names of its
# losses (D152 moved them to net/loss; the same objects). A patch of a loss goes to LJK, which the trainer reads.
for _n, _v in vars(LJK).items():
    if _n not in vars(JK):
        setattr(JK, _n, _v)
import jtoken_train as JKT  # noqa: E402
import lane_net as LN  # noqa: E402
import lane_readout as RO  # noqa: E402
import lane_ptok_data  # noqa: E402  (lane_net reads it; a test that swaps the trainer's loader puts this module back)
from merged_source import must_fail  # noqa: E402

torch.set_num_threads(2)
SCRATCH = os.path.join(BP.SCRATCH, "e365")
os.makedirs(SCRATCH, exist_ok=True)
lse = lambda *v: math.log(sum(math.exp(x) for x in v))  # noqa: E731


def holds(cond):
    """`must_fail(holds, <mutant's check>)`: the check of a mutant must be False"""
    assert cond


def patched(module, name, repl, fn, *a, **k):
    """fn(*a, **k) with module.name replaced by repl (a mutant), restored afterwards"""
    old = getattr(module, name)
    setattr(module, name, repl)
    try:
        return fn(*a, **k)
    finally:
        setattr(module, name, old)


# ------------------------------------------------------------------------------------------------------------------------------ hand frames
def _frame(K, short, long, gt, ylong, ytype, valid, scale=None):
    """one hand frame: the candidates {source: [destinations]} of the two classes (chord 3 m / 12 m), the GT links [(i, j)], the pairs flagged in `ylong`, `ytype`, `valid`, `scale` {(i, j): (a, b)} -> a spec dict"""
    return dict(K=K, short=short, long=long, gt=gt, ylong=ylong, ytype=ytype, valid=valid, scale=scale or {})


F1 = _frame(5, {0: [1, 2, 3], 1: [2], 3: [4]}, {0: [4], 1: [4]}, [(0, 1), (0, 2), (0, 3), (0, 4), (3, 0)], [(0, 3), (0, 4)], [1, 0, 1, 3, 0], [True] * 5, {(0, 4): (0.3, 0.5)})
F2 = _frame(5, {0: [1], 1: [2]}, {}, [(0, 1)], [], [1, 1, 0, 0, 0], [True] * 3 + [False] * 2)       # other source counts, two padded tokens, no long source


def _tensors(f):
    K = f["K"]
    cand, chord, y, yl = torch.zeros(2, K, K, dtype=torch.bool), torch.full((K, K), 3.0), torch.zeros(K, K), torch.zeros(K, K, dtype=torch.bool)
    ys = torch.full((K, K, 2), float("nan"))
    for c, cs in enumerate((f["short"], f["long"])):
        for i, js in cs.items():
            for j in js:
                cand[c, i, j] = True
                chord[i, j] = 12.0 if c else 3.0
    for i, j in f["gt"]:
        y[i, j] = 1
    for i, j in f["ylong"]:
        yl[i, j] = True
    for (i, j), v in f["scale"].items():
        ys[i, j] = torch.tensor(v)
    return dict(cand=cand, chord=chord, y=y, ylong=yl, yscale=ys, ytype=torch.tensor(f["ytype"]), valid=torch.tensor(f["valid"]))


def _hand(frames=(F1, F2)):
    """-> (out with leaf logits z / zn, batch, frames)"""
    K = frames[0]["K"]
    ts = [_tensors(f) for f in frames]
    st = lambda k: torch.stack([t[k] for t in ts])  # noqa: E731
    g = torch.Generator().manual_seed(1)
    B = len(frames)
    z, zn, ty = torch.randn(B, 2, K, K, generator=g), torch.randn(B, 2, K, generator=g), torch.randn(B, K, 7, generator=g)
    out = dict(z=z.requires_grad_(), zn=zn.requires_grad_(), cand=st("cand"), chord=st("chord"), valid=st("valid"), scale=torch.full((B, K, K, 2), 0.4), types=ty, types1=ty.flip(-1))
    return out, dict(y=st("y"), ylong=st("ylong"), yscale=st("yscale"), ytype=st("ytype")), frames


def _with_logp(out, cand=None):
    c = out["cand"] if cand is None else cand
    return dict(out, cand=c, logp=JK.wiring_logp(out["z"], out["zn"], c))


def _hand_loss(f, z, zn, t1, t3, w=1.0):
    """the per-frame numbers by hand: (nll_s, nll_l or None, ce1, ce3); w = the none weight (empty-G sources x w, the none class of the CE x w, normalised by the sum of the weights);
    the likelihood of a source with GT successors is -log sum of p over G(i). G(i) = the GT successors of i among its candidates of the class (the class of a pair = its candidate list, i.e. the token chord)"""
    res = []
    for c, cs in enumerate((f["short"], f["long"])):
        rows = []
        for i, js in cs.items():
            G = [j for a, j in f["gt"] if a == i and j in js]
            den = lse(*[z[c][i][j] for j in js], zn[c][i])
            zg = [z[c][i][j] for j in G]
            rows.append(-(lse(*zg) - den) if G else -w * (zn[c][i] - den))
        res.append(sum(rows) / len(rows) if rows else None)
    wt = lambda n: w if f["ytype"][n] == 0 else 1.0  # noqa: E731
    ce = lambda t: sum(wt(n) * -(t[n][f["ytype"][n]] - lse(*t[n])) for n in range(f["K"]) if f["valid"][n]) / sum(wt(n) for n in range(f["K"]) if f["valid"][n])  # noqa: E731
    return res[0], res[1], ce(t1), ce(t3)


def _check_hand(out, batch, frames, w=1.0):
    """asserts the numbers of `jtoken_loss` against the hand values of every frame, averaged per frame (a class that a frame lacks adds nothing)"""
    z, zn, t1, t3 = (x.detach().tolist() for x in (out["z"], out["zn"], out["types1"], out["types"]))
    per = [_hand_loss(f, z[b], zn[b], t1[b], t3[b], w) for b, f in enumerate(frames)]
    mean = lambda k: (lambda v: sum(v) / len(v))([p[k] for p in per if p[k] is not None])  # noqa: E731
    nll_s, nll_l, ce1, ce3 = (mean(k) for k in range(4))
    loss, info = LJK.jtoken_loss(out, batch, none_weight=w)
    assert abs(info["nll_s"] - nll_s) < 1e-5 and abs(info["nll_l"] - nll_l) < 1e-5 and abs(info["ce1"] - ce1) < 1e-5 and abs(info["ce3"] - ce3) < 1e-5 and abs(info["scale_l1"] - 0.1) < 1e-6, (info, nll_s, nll_l, ce1, ce3)
    assert abs(float(loss) - (nll_s + nll_l + 0.5 * (ce1 + ce3) + 0.5 * 0.1)) < 1e-5
    assert (info["gt_links"], info["pos_pairs"], info["gt_outside"], info["gt_class"]) == (6.0, 5.0, 1.0, 1.0), info
    return loss


def test_likelihood():
    out, b, frames = _hand()
    out = _with_logp(out)
    loss = _check_hand(out, b, frames)
    # the second frame has another source count, padding and no long source: nll_l is the value of frame 1 alone (not halved), the CE averages 5 and 3 valid tokens per frame; the GT link 0 -> 3 has `ylong` = True
    # against its token chord (3 m): it stays a positive of the short class (G(0) = {1, 2, 3}, checked by the hand value)
    must_fail(_check_hand, _with_logp(dict(out, zn=torch.full_like(out["zn"], -1e4))), b, frames)                       # no none slot
    must_fail(_check_hand, out, dict(b, y=b["y"].transpose(1, 2), ylong=b["ylong"].transpose(1, 2)), frames)            # a swapped link direction
    must_fail(_check_hand, _with_logp(out, torch.ones_like(out["cand"])), b, frames)                                    # no gate
    must_fail(patched, LJK, "_frame_mean", lambda x, m, w=None: (x * m).sum() / m.sum().clamp_min(1), _check_hand, out, b, frames)   # a global mean over the sources / tokens instead of per frame
    must_fail(_check_hand, dict(out, valid=torch.ones_like(out["valid"])), b, frames)                                    # no valid mask (the padded tokens of frame 2 in the type CE)
    # sources without a candidate are masked: their none logit changes nothing; the empty sources are trained by the none slot (and push their candidates down)
    out2 = dict(out, zn=out["zn"].detach().clone())
    out2["zn"][0, 0, (2, 4)] = out2["zn"][0, 1, (2, 3, 4)] = out2["zn"][1, 0, (2, 3, 4)] = out2["zn"][1, 1] = 9.0
    assert abs(float(LJK.jtoken_loss(_with_logp(out2), b)[0]) - float(loss)) < 1e-6, "a source without a candidate reaches the loss"
    loss.backward()
    zn_g, z_g = out["zn"].grad[0], out["z"].grad[0]
    assert float(zn_g[0, 1].abs()) > 0 and float(zn_g[0, 3].abs()) > 0 and float(zn_g[1, 1].abs()) > 0 and float(zn_g[0, 0]) > 0, "the none slot gets no gradient (source 1, 3 empty G; source 0 has G: the none mass is pushed down)"
    assert float(z_g[0, 1, 2]) > 0 and float(z_g[0, 3, 4]) > 0, "the candidate of an empty source must be pushed down (positive gradient on its logit)"
    assert float(zn_g[0, 2].abs()) == 0.0 and float(z_g[0, 0, 1]) < 0 and float(z_g[0, 0, 3]) < 0, "masked source / the GT candidates of a source with G(i) go up (negative gradient), also 0 -> 3 with ylong against its chord"
    assert float(z_g[~out["cand"][0]].abs().max()) == 0.0, "a logit outside the gate gets gradient"
    print("likelihood: the loss by hand on two frames (three GT successors incl. one with ylong against the token chord, empty G -> none, GT link outside the gate -> none, masked sources, padding, a frame without a long "
          "source, the per-frame normalisation), mutants (no none slot, swapped direction, no gate, global mean, no valid mask) fail, the none slot trains the empty sources and pushes their candidates down: ok")


def test_none_weight():
    out, b, frames = _hand()
    out = _with_logp(out)
    l1 = _check_hand(out, b, frames, 1.0)
    l5 = _check_hand(out, b, frames, 0.5)                                                       # W = 0.5 by hand: the empty-G sources x 0.5 in both classes, the none class of the CE x 0.5, normalised by the weights
    assert abs(float(l1) - float(l5)) > 1e-3 and float(LJK.jtoken_loss(out, b)[0]) == float(LJK.jtoken_loss(out, b, none_weight=1.0)[0]), "W changes the loss, W = 1 is the loss of before"
    assert any(not [j for a, j in f["gt"] if a == i and j in js] for f in frames for i, js in f["short"].items()) and any(f["ytype"][n] == 0 and f["valid"][n] for f in frames for n in range(f["K"])), "the frames have empty-G sources and none tokens"
    orig = LJK._source_nll
    must_fail(patched, LJK, "_source_nll", lambda lp, G, w: w * orig(lp, G, 1.0), _check_hand, out, b, frames, 0.5)                      # the weight on the sources WITH a GT successor too
    must_fail(patched, LJK, "_source_nll", lambda lp, G, w: orig(lp, G, 1.0), _check_hand, out, b, frames, 0.5)                          # no weight on the none sources
    x, mk, wk = torch.full((1, 2), 0.1), torch.ones(1, 2, dtype=torch.bool), torch.full((1, 2), 0.1)       # weights summing below 1: sum(w x) / sum(w) = 0.2 / 0.2 = 1, not 0.2 / 1
    assert abs(float(LJK._frame_mean(x, mk, wk)) - 1.0) < 1e-6 and abs(float(LJK._frame_mean(x, mk)) - 0.1) < 1e-6 and float(LJK._frame_mean(x, torch.zeros_like(mk), wk)) == 0.0
    orig_fm = LJK._frame_mean
    must_fail(patched, LJK, "_frame_mean", lambda x, m, w=None: orig_fm(x, m), _check_hand, out, b, frames, 0.5)                                     # weighted terms divided by the token count instead of the sum of the weights
    # the sources with a GT successor are not touched by W: the gradient of the row of frame 0, source 0 (G non-empty) is identical, the none logit of source 1 (G empty) moves
    for w_ in (0.5, 2.0):
        o2 = _with_logp(dict(out, z=out["z"].detach().clone().requires_grad_(), zn=out["zn"].detach().clone().requires_grad_()))
        LJK.jtoken_loss(o2, b, none_weight=w_)[0].backward()
        o1 = _with_logp(dict(out, z=out["z"].detach().clone().requires_grad_(), zn=out["zn"].detach().clone().requires_grad_()))
        LJK.jtoken_loss(o1, b)[0].backward()
        assert torch.equal(o2["z"].grad[0, 0, 0], o1["z"].grad[0, 0, 0]) and not torch.equal(o2["zn"].grad[0, 0, 1], o1["zn"].grad[0, 0, 1]), "frame 1 source 0 has a GT successor (short): untouched; source 1 (long class: empty) moves"
    print("none_weight: the loss by hand with W = 0.5 on the two hand frames (empty-G sources x W in both classes, the none class of the CE x W, normalised by the sum of the weights), W = 1 = the loss of before, "
          "mutants (the weight on the non-empty sources too, no weight on the none sources, the CE divided by the token count) fail, a source with a GT successor is untouched: ok")


def _skip_brute(q, hops):
    """the largest product of q along a walk of 2 .. hops links, by hand (loops over every intermediate node)"""
    K = q.shape[0]
    reach, indirect = q.copy(), np.zeros_like(q)
    for _ in range(hops - 1):
        nxt = np.array([[max(q[i, n] * reach[n, j] for n in range(K)) for j in range(K)] for i in range(K)])
        indirect = np.maximum(indirect, nxt)
        reach = np.maximum(q, nxt)
    return indirect


def test_skip_suppress():
    """E399 the soft skip-link layer: off (the default) = the head bit for bit and no new parameter or state key; on a chain 0 -> 1 -> 2 with the jump 0 -> 2 only the jump loses its mass, to the none slot; rows stay distributions;
    against a brute-force walk enumeration on a random graph; the gradient reaches the input log-probabilities; the head builds it from its args"""
    cfg = dict(sigma=0.03, hops=5, m=8, t_short=0.3, t_long=0.2, keep=0.6)
    K = 4
    cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
    for c, i, j in ((0, 0, 1), (0, 1, 2), (1, 0, 2), (0, 2, 3)):
        cand[0, c, i, j] = True
    chord = torch.full((1, K, K), 2.0)
    chord[0, 0, 2] = 12.0
    z = torch.full((1, 2, K, K), 2.0)
    zn = torch.zeros(1, 2, K)
    logp = JK.wiring_logp(z, zn, cand).requires_grad_()
    pnone = torch.zeros(1, K)
    out = JK.skip_suppress(logp, cand, pnone, chord, cfg)
    p0, p1 = logp.exp()[0], out.exp()[0]
    assert torch.allclose(p1.sum(-1), torch.ones(2, K), atol=1e-5), "every row stays a distribution"
    for c, i, j in ((0, 0, 1), (0, 1, 2), (0, 2, 3)):
        assert abs(float(p1[c, i, j] - p0[c, i, j])) < 1e-3, ("a link of the chain is not damped", c, i, j)
    assert float(p1[1, 0, 2]) < 0.05 * float(p0[1, 0, 2]), ("the jump 0 -> 2 is damped", float(p0[1, 0, 2]), float(p1[1, 0, 2]))
    assert abs(float(p1[1, 0, K] - (p0[1, 0, K] + p0[1, 0, 2] - p1[1, 0, 2]))) < 1e-5, "the removed mass goes to the none slot"
    near = JK.wiring_logp(torch.zeros(1, 2, K, K), torch.full((1, 2, K), 0.6), cand).requires_grad_()       # every link at p 0.35, near the thresholds
    JK.skip_suppress(near, cand, pnone, chord, cfg).exp()[0, 1, 0, K].backward()
    assert float(near.grad[0, 0, 1, 2].abs()) > 0, "near the thresholds the gradient reaches the chain's links (far from them the layer is saturated)"
    gone = JK.skip_suppress(logp.detach(), cand, torch.tensor([[0.0, 0.99, 0.0, 0.0]]), chord, cfg).exp()[0]
    assert abs(float(gone[1, 0, 2] - p0[1, 0, 2])) < 1e-3, "a path over a node that is not kept covers nothing"
    rng = np.random.RandomState(3)
    K = 7
    cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
    cand[0, 0] = torch.from_numpy((rng.rand(K, K) < 0.45) & ~np.eye(K, dtype=bool))
    chord = torch.full((1, K, K), 2.0)
    logp = JK.wiring_logp(torch.from_numpy(rng.randn(1, 2, K, K).astype(np.float32) * 2), torch.zeros(1, 2, K), cand)
    pn = torch.from_numpy(rng.rand(1, K).astype(np.float32) * 0.5)
    got = JK.skip_suppress(logp, cand, pn, chord, dict(cfg, m=K, sigma=0.2, hops=4)).exp()[0, 0, :, :K].numpy()
    p = (logp.exp()[0, 0, :, :K] * cand[0, 0]).numpy()
    sg = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    k = sg((0.6 - pn[0].numpy()) / 0.2)
    q = sg((p - 0.3) / 0.2) * cand[0].any(0).numpy() * k[:, None] * k[None]
    want = p * (1 - (1 - JK.SKIP_FLOOR) * _skip_brute(q.astype(np.float64), 4))
    assert np.abs(got - want)[cand[0, 0].numpy()].max() < 1e-5, np.abs(got - want).max()
    torch.manual_seed(7)
    plain = JK.JTokenHead().eval()
    torch.manual_seed(7)
    sk = JK.JTokenHead(skip=cfg).eval()
    assert list(plain.state_dict()) == list(sk.state_dict()) and all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), sk.state_dict().values()))
    b = JK.synthetic_batch(2, 64, seed=5)
    with torch.no_grad():
        o0, o1 = plain(b), sk(b)
    assert torch.equal(o0["types"], o1["types"]) and not torch.equal(o0["logp"], o1["logp"])
    assert torch.allclose(o1["logp"], JK.skip_suppress(o0["logp"], o0["cand"], o0["types"].softmax(-1)[..., 0], o0["chord"], cfg)), "the head applies the layer to its final wiring"
    assert LN.build_head(dict(skip_suppress=cfg)).skip == cfg and LN.build_head({}).skip is None
    must_fail(JK.JTokenHead, skip=dict(sigma=0.03))
    print("skip_suppress: chain with a jump (only the jump damped, mass to none, gradient flows, a dropped node covers nothing), brute-force walks on a random graph, head off = bit for bit, same state keys, build_head: ok")


def _gap_brute(p, cand, pnone, chord, cfg):
    """`close_gaps` on one frame by hand: p (2, K, K) link probabilities per class, loops over every pair -> (2, K, K)"""
    K = cand.shape[-1]
    sg = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    P, any_ = (p * cand).sum(0), cand.any(0)
    k = sg((cfg["keep"] - pnone) / cfg["sigma"])
    ends = [[any_[i, j] * k[i] * k[j] for j in range(K)] for i in range(K)]
    t = lambda i, j: cfg["t_long"] if chord[i, j] > 8 else cfg["t_short"]  # noqa: E731
    lo = lambda i, j: cfg["lo_long"] if chord[i, j] > 8 else cfg["lo"]  # noqa: E731
    q = [[sg((P[i, j] - t(i, j)) / cfg["sigma"]) * ends[i][j] for j in range(K)] for i in range(K)]
    row = lambda i, j: max(q[i][n] for n in range(K) if n != j)  # noqa: E731
    col = lambda i, j: max(q[n][j] for n in range(K) if n != i)  # noqa: E731
    gate = np.array([[(1 - row(i, j)) * (1 - col(i, j)) * sg((P[i, j] - lo(i, j)) / cfg["sigma_lo"]) * ends[i][j] for j in range(K)] for i in range(K)])
    m = P * gate
    lift = np.zeros_like(p)
    for i in range(K):
        for j in range(K):
            rel = (m[i, j] / max(m[i].max(), 1e-12)) ** cfg["gamma"] * (m[i, j] / max(m[:, j].max(), 1e-12)) ** cfg["gamma"]
            lift[:, i, j] = cand[:, i, j] * gate[i, j] * rel * t(i, j) * max(1 - P[i, j], 0.0)
    out = np.zeros_like(p)
    for c in range(2):
        for i in range(K):
            total = lift[c, i].sum()
            used = min(total, 0.5)
            out[c, i] = p[c, i] * cand[c, i] * (1 - used) + lift[c, i] * (used / max(total, 1e-12))
    return out


def test_layers_train():
    """E425 a head that trains with the layers: on a source whose none probability is smaller than its lift the none slot keeps at least half of its mass; the token loss behind all three wiring layers is finite and so is every
    gradient, also with a source that has no true successor"""
    cfg = dict(sigma=0.01, sigma_lo=0.01, gamma=16.0, lo=0.03, lo_long=0.02, t_short=0.3, t_long=0.2, keep=0.6)
    K = 4
    p, cand = torch.zeros(1, 2, K, K), torch.zeros(1, 2, K, K, dtype=torch.bool)
    for j in (1, 2, 3):
        p[0, 0, 0, j], cand[0, 0, 0, j] = 0.29, True                         # three weak links of one source, none 0.13: their lifts exceed it
    logp = torch.cat([p.clamp_min(1e-30).log(), (1 - p.sum(-1, keepdim=True)).clamp_min(1e-30).log()], -1).requires_grad_(True)
    out = JK.close_gaps(logp, cand, torch.zeros(1, K), torch.full((1, K, K), 2.0), cfg)
    none = float(out.exp()[0, 0, 0, K])
    assert 0.065 - 1e-4 <= none <= 0.13 and float(out.exp()[0, 0, 0, 1]) >= 0.3, ("the none slot keeps half of its mass and the links are delivered", none, out.exp()[0, 0, 0])
    (-out[0, 0, 0, K]).backward()
    assert torch.isfinite(logp.grad).all(), "a finite gradient through the none slot of a lifted source"
    layers = JK.OUTPUT_LAYERS["soft_output"]
    for seed in (3, 11):
        torch.manual_seed(seed)
        head = JK.JTokenHead(skip=layers["skip_suppress"], gap=layers["close_gaps"], fork=layers["fork_gate"], refine=layers["refine_positions"]).train()
        head.set_step(JK.SWITCH + 1)
        b = JK.synthetic_batch(4, 64, seed=seed)
        loss, info = LJK.jtoken_loss(head(b), b, none_weight=0.25)
        loss.backward()
        bad = [n for n, q in head.named_parameters() if q.grad is None or not torch.isfinite(q.grad).all()]
        assert torch.isfinite(loss) and float(loss) < 30 and not bad, (seed, float(loss), bad)
    print("layers_train: the none slot keeps half its mass under a lift; the loss behind skip, gap and fork layers is finite, every gradient finite: ok")


def test_close_gaps():
    """E420 the soft gap-closing layer: off (the default) = the head bit for bit and no new state key; a weak link between an open end and an open start is delivered (only the best one), none when its probability is below the lower
    threshold, when an end already has another delivered link or when a node is not kept; rows stay distributions; the output never falls when the pair's own probability rises (both classes, below, at and above the thresholds);
    against a brute-force form on a random graph; the head builds it from its args"""
    cfg = dict(sigma=0.03, sigma_lo=0.01, gamma=16.0, lo=0.05, lo_long=0.05, t_short=0.3, t_long=0.2, keep=0.6)
    K = 4
    pairs = ((0, 1), (1, 2), (1, 3), (2, 3))

    def logp_of(vals):
        p, cand = torch.zeros(1, 2, K, K), torch.zeros(1, 2, K, K, dtype=torch.bool)
        for (i, j), v in zip(pairs, vals):
            p[0, 0, i, j], cand[0, 0, i, j] = v, True
        return torch.cat([p.clamp_min(1e-30).log(), (1 - p.sum(-1, keepdim=True)).clamp_min(1e-30).log()], -1), cand

    chord, kept = torch.full((1, K, K), 2.0), torch.zeros(1, K)

    def run(vals, pnone=kept):
        logp, cand = logp_of(vals)
        out = JK.close_gaps(logp, cand, pnone, chord, cfg).exp()[0, 0]
        assert torch.allclose(out.sum(-1), torch.ones(K), atol=1e-5), "every row stays a distribution"
        return out

    out = run((0.8, 0.10, 0.07, 0.8))
    assert float(out[1, 2]) >= 0.3, ("the gap 1 -> 2 is closed", float(out[1, 2]))
    assert float(out[1, 3]) < 0.1, ("the weaker candidate of the same source stays", float(out[1, 3]))
    for i, j in ((0, 1), (2, 3)):
        assert 0.8 <= float(out[i, j]) <= 0.8 + 0.3 * 0.2 + 1e-5, ("a delivered link stays delivered and rises by at most t (1 - p)", i, j, float(out[i, j]))
    assert abs(float(out[1, K]) - (1 - float(out[1, :K].sum()))) < 1e-5 and float(out[1, K]) < 1 - 0.17, "the lifted mass comes from the source's other slots, mostly none"
    assert float(run((0.8, 0.03, 0.02, 0.8))[1, 2]) < 0.1, "below the lower threshold nothing is delivered"
    assert float(run((0.8, 0.10, 0.60, 0.1))[1, 2]) < 0.12, "a source with a delivered successor is not open"
    assert float(run((0.8, 0.10, 0.07, 0.8), torch.tensor([[0.0, 0.99, 0.0, 0.0]]))[1, 2]) < 0.12, "a node that is not kept opens no gap"
    two = run((0.8, 0.22, 0.22, 0.1))                                        # two equal candidates of one source
    assert float(two[1, :K].sum()) <= 1.0 + 1e-5 and float(two[1, 2:4].min()) >= 0.3
    for far in (2.0, 12.0):                                                  # the pair 1 -> 2 in the short (t 0.3) and the long class (t 0.2)
        ch = torch.full((1, K, K), 2.0)
        ch[0, 1, 2] = far
        for others in ((0.8, 0.07, 0.8), (0.05, 0.20, 0.1), (0.4, 0.28, 0.29), (0.8, 0.6, 0.1)):
            last = -1.0
            for x in (0.001, 0.02, 0.04, 0.05, 0.06, 0.07, 0.10, 0.15, 0.18, 0.2, 0.25, 0.28, 0.29, 0.3, 0.31, 0.4, 0.5, 0.65):
                leaf = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                p = torch.zeros(1, 2, K, K, dtype=torch.float64)
                cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
                for (i, j), v in zip(pairs, (others[0], None, others[1], others[2])):
                    c = int(float(ch[0, i, j]) > 8)
                    p[0, c, i, j] = leaf if v is None else v
                    cand[0, c, i, j] = True
                logp = torch.cat([p.clamp_min(1e-30).log(), (1 - p.sum(-1, keepdim=True)).clamp_min(1e-30).log()], -1)
                got = JK.close_gaps(logp, cand, kept, ch, cfg).exp()[0, int(far > 8), 1, 2]
                got.backward()
                assert float(leaf.grad) >= -1e-9 and float(got) >= last - 1e-9, ("the output falls when the pair's own probability rises", far, others, x, float(got), float(leaf.grad))
                last = float(got)
    rng = np.random.RandomState(5)
    K = 7
    cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
    pick = rng.rand(K, K)
    cand[0, 0] = torch.from_numpy((pick < 0.3) & ~np.eye(K, dtype=bool))
    cand[0, 1] = torch.from_numpy((pick > 0.8) & ~np.eye(K, dtype=bool))
    chord = torch.from_numpy(np.where(pick > 0.8, 12.0, 3.0).astype(np.float32))[None]
    soft = dict(cfg, sigma=0.2, sigma_lo=0.1, gamma=3.0, lo_long=0.03)
    logp = JK.wiring_logp(torch.from_numpy(rng.randn(1, 2, K, K).astype(np.float32) * 2), torch.from_numpy(rng.randn(1, 2, K).astype(np.float32) + 1.0), cand)
    pn = torch.from_numpy(rng.rand(1, K).astype(np.float32) * 0.8)
    got = JK.close_gaps(logp, cand, pn, chord, soft).exp()[0, :, :, :K].numpy()
    p = (logp.exp()[0, :, :, :K] * cand[0]).numpy().astype(np.float64)
    want = _gap_brute(p, cand[0].numpy(), pn[0].numpy().astype(np.float64), chord[0].numpy(), soft)
    assert np.abs(got - want)[cand[0].numpy()].max() < 1e-5, np.abs(got - want).max()
    assert np.abs(want - p).max() > 1e-3, "the random graph exercises the layer"
    torch.manual_seed(13)
    plain = JK.JTokenHead().eval()
    torch.manual_seed(13)
    closed = JK.JTokenHead(gap=cfg).eval()
    torch.manual_seed(13)
    both = JK.JTokenHead(gap=cfg, skip={k: v for k, v in dict(cfg, hops=5, m=8).items() if k in JK.SKIP_KEYS}).eval()
    assert list(plain.state_dict()) == list(closed.state_dict()) and all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), closed.state_dict().values()))
    b = JK.synthetic_batch(2, 64, seed=5)
    with torch.no_grad():
        o0, o1, o2 = plain(b), closed(b), both(b)
    pnone = o0["types"].softmax(-1)[..., 0]
    assert torch.equal(o0["types"], o1["types"])
    assert torch.allclose(o1["logp"], JK.close_gaps(o0["logp"], o0["cand"], pnone, o0["chord"], cfg)), "the head applies the layer to its final wiring"
    assert torch.allclose(o2["logp"], JK.close_gaps(JK.skip_suppress(o0["logp"], o0["cand"], pnone, o0["chord"], both.skip), o0["cand"], pnone, o0["chord"], cfg)), "the skip layer first, then the gaps"
    assert LN.build_head(dict(close_gaps=cfg)).gap == cfg and LN.build_head({}).gap is None
    must_fail(JK.JTokenHead, gap=dict(lo=0.05))
    print("close_gaps: a gap between an open end and an open start is closed (best candidate only; not below the lower threshold, not at a source with a successor, not over a dropped node), rows stay distributions, never falls with the pair's own probability, brute force on a random graph, head off = bit for bit, skip then gaps, build_head: ok")


def _fork_brute(p, cand, pt, chord, cfg):
    """`fork_gate` on one frame by hand: p (2, K, K) link probabilities per class, pt (K, 7), loops over every pair -> (2, K, K)"""
    K = cand.shape[-1]
    sg = lambda x: 1 / (1 + np.exp(-np.clip(x, -60, 60)))  # noqa: E731
    P, any_ = (p * cand).sum(0), cand.any(0)
    k = sg((cfg["keep"] - pt[:, 0]) / cfg["sigma"])
    s = np.array([[P[i, j] / (cfg["t_long"] if chord[i, j] > 8 else cfg["t_short"]) * any_[i, j] * k[i] * k[j] for j in range(K)] for i in range(K)])
    out = np.zeros_like(p)
    for i in range(K):
        for j in range(K):
            m = max(s[i, n] for n in range(K) if n != j)
            w = sg((m - s[i, j]) / cfg["sigma_rel"]) * sg((m - 1) / cfg["sigma_rel"])
            f = sg((pt[i, JK.FORK] - cfg["tau"]) / cfg["sigma_f"])
            out[:, i, j] = p[:, i, j] * cand[:, i, j] * (1 - (1 - f) * w)
    return out


def test_fork_gate():
    """E423 the soft fork gate: a source that is no fork keeps its strongest successor and loses a further delivered one (its mass to none); a fork keeps both; one successor is never touched; rows stay distributions; never
    falling in a pair's own probability; the type softmax gets the gradient; brute force on a random graph; the head applies it after the other layers and builds it from its args"""
    cfg = dict(sigma=0.03, sigma_rel=0.05, tau=0.05, sigma_f=0.01, t_short=0.3, t_long=0.2, keep=0.6)
    K = 4
    p, cand = torch.zeros(1, 2, K, K), torch.zeros(1, 2, K, K, dtype=torch.bool)
    for (i, j), v in {(0, 1): 0.55, (0, 2): 0.40, (1, 3): 0.45}.items():
        p[0, 0, i, j], cand[0, 0, i, j] = v, True
    logp = torch.cat([p.clamp_min(1e-30).log(), (1 - p.sum(-1, keepdim=True)).clamp_min(1e-30).log()], -1)
    chord = torch.full((1, K, K), 2.0)

    def types(fork):
        pt = torch.zeros(1, K, 7)
        pt[..., 1], pt[..., 2] = 1 - fork, fork
        return pt

    lone = JK.fork_gate(logp, cand, types(0.0), chord, cfg).exp()[0, 0]
    assert float(lone[0, 1]) > 0.54 and float(lone[0, 2]) < 0.05 and abs(float(lone[1, 3]) - 0.45) < 1e-3, ("the second successor of a source that is no fork goes, the others stay", lone[0], lone[1])
    assert torch.allclose(lone.sum(-1), torch.ones(K), atol=1e-5) and abs(float(lone[0, K]) - (1 - float(lone[0, :K].sum()))) < 1e-5, "the removed mass goes to none"
    forked = JK.fork_gate(logp, cand, types(0.5), chord, cfg).exp()[0, 0]
    assert abs(float(forked[0, 1]) - 0.55) < 1e-3 and abs(float(forked[0, 2]) - 0.40) < 1e-3, ("a fork keeps both", forked[0])
    pt = types(0.05).requires_grad_(True)
    JK.fork_gate(logp, cand, pt, chord, cfg).exp()[0, 0, 0, 2].backward()
    assert float(pt.grad[0, 0, JK.FORK]) > 0, "the fork probability of the source raises its second successor"
    last = -1.0
    for x in (0.05, 0.2, 0.29, 0.31, 0.4, 0.5, 0.54, 0.56, 0.6):
        q = p.clone()
        q[0, 0, 0, 2], q[0, 0, 0, 1] = x, 0.38
        lq = torch.cat([q.clamp_min(1e-30).log(), (1 - q.sum(-1, keepdim=True)).clamp_min(1e-30).log()], -1)
        got = float(JK.fork_gate(lq, cand, types(0.0), chord, cfg).exp()[0, 0, 0, 2])
        assert got >= last - 1e-6, ("the output falls when the pair's own probability rises", x, got, last)
        last = got
    rng = np.random.RandomState(6)
    K = 7
    cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
    pick = rng.rand(K, K)
    cand[0, 0] = torch.from_numpy((pick < 0.4) & ~np.eye(K, dtype=bool))
    cand[0, 1] = torch.from_numpy((pick > 0.7) & ~np.eye(K, dtype=bool))
    chord = torch.from_numpy(np.where(pick > 0.7, 12.0, 3.0).astype(np.float32))[None]
    soft = dict(cfg, sigma=0.2, sigma_rel=0.5, tau=0.2, sigma_f=0.1)
    logp = JK.wiring_logp(torch.from_numpy(rng.randn(1, 2, K, K).astype(np.float32) * 2), torch.from_numpy(rng.randn(1, 2, K).astype(np.float32)), cand)
    pt = torch.from_numpy(rng.dirichlet(np.ones(7), (1, K)).astype(np.float32))
    got = JK.fork_gate(logp, cand, pt, chord, soft).exp()[0, :, :, :K].numpy()
    p = (logp.exp()[0, :, :, :K] * cand[0]).numpy().astype(np.float64)
    want = _fork_brute(p, cand[0].numpy(), pt[0].numpy().astype(np.float64), chord[0].numpy(), soft)
    assert np.abs(got - want)[cand[0].numpy()].max() < 1e-5 and np.abs(want - p).max() > 1e-3, np.abs(got - want).max()
    torch.manual_seed(13)
    plain = JK.JTokenHead().eval()
    torch.manual_seed(13)
    gated = JK.JTokenHead(fork=cfg).eval()
    assert list(plain.state_dict()) == list(gated.state_dict())
    b = JK.synthetic_batch(2, 64, seed=5)
    with torch.no_grad():
        o0, o1 = plain(b), gated(b)
    assert torch.allclose(o1["logp"], JK.fork_gate(o0["logp"], o0["cand"], o0["types"].softmax(-1), o0["chord"], cfg)) and "pos" not in o1, "the head applies the layer to its final wiring"
    assert LN.build_head(dict(fork_gate=cfg)).fork == cfg and LN.build_head({}).fork is None
    must_fail(JK.JTokenHead, fork=dict(tau=0.05))
    print("fork_gate: a second successor only at a fork (mass to none), one successor untouched, rows stay distributions, never falls with the pair's own probability, gradient to the fork probability, brute force on a random graph, head, build_head: ok")


def test_refine_positions():
    """E424 the soft position layer: straight lanes and circular arcs are fixed points; a zig-zag shrinks; only delivered short links move nodes; the offset is scaled by `gain` inside the cell; padding stays; gradients reach positions and
    the wiring; the head writes `pos`, the dump record carries it and the delivered graph is written from it"""
    cfg = dict(gain=1.0, lam=1.0, sigma=0.03, t_short=0.3, keep=0.6)
    K = 8

    def chain(pos, dirv, p_link=0.9, long=False):
        p, cand = torch.zeros(1, 2, K, K), torch.zeros(1, 2, K, K, dtype=torch.bool)
        for i in range(K - 2):
            p[0, int(long), i, i + 1], cand[0, int(long), i, i + 1] = p_link, True
        logp = torch.cat([p.clamp_min(1e-30).log(), (1 - p.sum(-1, keepdim=True)).clamp_min(1e-30).log()], -1)
        valid = torch.arange(K)[None] < K - 1
        return JK.refine_positions(pos[None], dirv[None], valid, logp, cand, torch.zeros(1, K), cfg)[0]

    x = torch.arange(K, dtype=torch.float32) * 2.0 - 7.3
    straight, up = torch.stack([torch.full((K,), 0.4), x], -1), torch.tensor([[0.0, 1.0]]).expand(K, 2)
    assert (chain(straight, up) - straight).abs().max() < 1e-5, "a straight lane stays"
    th = torch.arange(K, dtype=torch.float32) * 0.15
    arc, tan = 15.0 * torch.stack([th.cos(), th.sin()], -1), torch.stack([-th.sin(), th.cos()], -1)
    assert (chain(arc, tan) - arc).abs().max() < 1e-4, ("a circular arc stays", (chain(arc, tan) - arc).abs().max())
    zig = straight.clone()
    zig[:, 0] += 0.3 * (-1.0) ** torch.arange(K)
    out = chain(zig, up)
    amp = lambda q: float((q[1:K - 2, 0] - 0.4).abs().mean())  # noqa: E731
    assert amp(out) < 0.4 * amp(zig) and (out[:, 1] - zig[:, 1]).abs().max() < 1e-5 and torch.equal(out[K - 1], zig[K - 1]), ("a zig-zag shrinks sideways; the padding token stays", amp(zig), amp(out))
    assert (chain(zig, up, p_link=0.01) - zig).abs().max() < 1e-4 and (chain(zig, up, long=True) - zig).abs().max() < 1e-6, "an undelivered or a long link moves nothing"
    node = torch.tensor([[1.4, 1.0], [-0.2, 3.9]])                            # offsets (0.4, 0) in the cell [0, 2) x [0, 2) and (-1.2 + 1, ..) in its cell
    none = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    lp = torch.cat([torch.full((1, 2, 2, 2), -69.0), torch.zeros(1, 2, 2, 1)], -1)
    got = JK.refine_positions(node[None], up[None, :2], torch.ones(1, 2, dtype=torch.bool), lp, none, torch.zeros(1, 2), dict(cfg, gain=3.0))[0]
    centre = torch.tensor([[1.0, 1.0], [-1.0, 3.0]])
    assert torch.allclose(got, centre + (3.0 * (node - centre)).clamp(-1, 1), atol=1e-5) and torch.allclose(got[0], torch.tensor([2.0, 1.0]), atol=1e-5), ("the offset is scaled and stays inside the cell", got)
    pos = (zig[None] + 0.0).requires_grad_(True)
    p, cand = torch.zeros(1, 2, K, K), torch.zeros(1, 2, K, K, dtype=torch.bool)
    for i in range(K - 2):
        p[0, 0, i, i + 1], cand[0, 0, i, i + 1] = 0.31, True
    logp = torch.cat([p.clamp_min(1e-30).log(), (1 - p.sum(-1, keepdim=True)).clamp_min(1e-30).log()], -1).requires_grad_(True)
    JK.refine_positions(pos, up[None], torch.ones(1, K, dtype=torch.bool), logp, cand, torch.zeros(1, K), cfg)[..., 0].square().sum().backward()
    assert float(pos.grad.abs().sum()) > 0 and float(logp.grad.abs().sum()) > 0, "gradients reach positions and wiring"
    torch.manual_seed(13)
    plain = JK.JTokenHead().eval()
    torch.manual_seed(13)
    moved = JK.JTokenHead(refine=dict(cfg, gain=1.2)).eval()
    b = JK.synthetic_batch(2, 64, seed=5)
    with torch.no_grad():
        o0, o1 = plain(b), moved(b)
    assert torch.equal(o0["logp"], o1["logp"]) and "pos" not in o0 and o1["pos"].shape == b["pos"].shape and torch.isfinite(o1["pos"]).all() and (o1["pos"] - b["pos"])[b["valid"]].abs().max() > 1e-3
    assert torch.equal(o1["pos"][~b["valid"]], b["pos"][~b["valid"]])
    cell = lambda q: (((q[:, 0] + 50) // 2) * 50 + (q[:, 1] + 50) // 2).astype(np.int64)  # noqa: E731
    toks = [dict(pos=b["pos"][i].numpy(), dirv=b["dirv"][i].numpy(), e_p=b["e"][i].numpy(), flags_p=b["flags"][i].numpy(), s_p=b["s"][i].numpy(), cell=cell(b["pos"][i].numpy()), valid=b["valid"][i].numpy()) for i in range(2)]
    recs = RO.batch_dump(o1, dict(token=["a", "b"], row=[0, 1], tokens_np=toks))
    graph = RO.graph(toks[0], recs[0], toks[0]["valid"], RO.Decode(link_thr=0.05, keep_thr=0.99, none_gate=False))
    want = np.asarray(recs[0]["pos"], float)[toks[0]["valid"] & (recs[0]["pnone"] < np.float32(0.99))]
    assert len(want) and np.array_equal(graph["nodes"]["pos"], want), "the delivered graph is written from the refined positions"
    assert "pos" not in RO.batch_dump(o0, dict(token=["a", "b"], row=[0, 1], tokens_np=toks))[0], "a head without the layer writes the records of before"
    assert LN.build_head(dict(refine_positions=cfg)).refine == cfg and LN.build_head({}).refine is None
    must_fail(JK.JTokenHead, refine=dict(gain=1.2))
    print("refine_positions: straight lanes and arcs stay, a zig-zag shrinks, only delivered short links move nodes, the offset scaled inside its cell, gradients, head / dump / graph carry the refined positions: ok")


def test_wiring_logp():
    out, _, _ = _hand()
    lp = JK.wiring_logp(out["z"].detach(), out["zn"].detach(), out["cand"])
    p = lp.exp()
    assert float((p.sum(-1) - 1).abs().max()) < 1e-6, "a source must sum to 1 over the candidates + none"
    assert float(p[..., :5][~out["cand"]].max()) == 0.0, "probability outside the gate"
    assert float(p[0, 0, 2, 5]) == 1.0 and float(p[0, 1, 4, 5]) == 1.0 and float(p[1, 0, 4, 5]) == 1.0, "a source without a candidate (also a padded one) has p(none) = 1"
    assert abs(float(p[0, 0, 1, 2]) - math.exp(float(out["z"][0, 0, 1, 2]) - lse(float(out["z"][0, 0, 1, 2]), float(out["zn"][0, 0, 1])))) < 1e-6
    bad = JK.wiring_logp(out["z"].detach(), out["zn"].detach(), torch.ones_like(out["cand"])).exp()
    must_fail(holds, float(bad[..., :5][~out["cand"]].max()) == 0.0)                                         # the mutant (no gate) puts probability outside it
    print("wiring_logp: sums to 1 over {candidates} + none, exactly 0 outside the gate, p(none) = 1 without a candidate; the ungated mutant leaks: ok")


# ------------------------------------------------------------------------------------------------------------------------------ decode
def _decode_out(p_none4):
    """6 tokens (5 is padding), short candidates 0->1 0->2 1->2 1->3 1->4 2->3 3->4 4->5 5->3, long 0->4: the per-source probabilities of `test_decode`; token 4 has p_none = p_none4. The padded token 5 is
    given probability mass as a source and as a destination (and a p_none of 0.1) on purpose: the decode must still never link it"""
    K = 6
    cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
    P = torch.zeros(1, 2, K, K + 1)
    P[..., K] = 1.0
    for c, i, row in ((0, 0, {1: .36, 2: .34}), (0, 1, {2: .27, 3: .26, 4: .25}), (0, 2, {3: .31}), (0, 3, {4: .55}), (0, 4, {5: .9}), (0, 5, {3: .9}), (1, 0, {4: .6})):
        for j, v in row.items():
            cand[0, c, i, j], P[0, c, i, j] = True, v
        P[0, c, i, K] = 1.0 - sum(row.values())
    chord = torch.where(cand[0, 1].any(-1)[:, None] & cand[0, 1], 12.0, 3.0)[None]
    pt = torch.zeros(1, K, 7)
    pn = torch.tensor([0.1, 0.1, 0.1, 0.2, p_none4, 0.1])
    pt[0, :, 0], pt[0, :, 1] = pn, 1 - pn
    valid = torch.tensor([[True] * 5 + [False]])
    return dict(logp=P.clamp_min(1e-9).log(), cand=cand, chord=chord, valid=valid, types=pt.clamp_min(1e-9).log(), types1=pt.clamp_min(1e-9).log())


def _pairs(d, key="link"):
    return sorted((int(i), int(j)) for i, j in d[key][0].nonzero().tolist())


def test_decode():
    a, bk = _decode_out(0.49), _decode_out(0.5)
    rec = lambda o, **k: _pairs(JK.decode(o, **k))  # noqa: E731
    # 0 -> 1 (.36) and 0 -> 2 (.34): both kept (a fork source), >= 0.3 and above p_none .30; source 1: .27 / .26 / .25 against p_none .22: all below 0.3; 2 -> 3 (.31): >= 0.3 but p_none .69 wins; 3 -> 4 (.55);
    # long 0 -> 4 (.6); 4 -> 5 (.9) ends in the padded token and 5 -> 3 (.9) starts in it: never a link
    exp = [(0, 1), (0, 2), (0, 4), (3, 4)]
    assert rec(a) == exp and _pairs(JK.decode(a), "long") == [(0, 4)]
    assert rec(bk) == [(0, 1), (0, 2)], "a link into a node with p_none >= 0.5 must be dropped (p_none 0.49 keeps it)"
    assert rec(a, thr=0.35) == [(0, 1), (0, 4), (3, 4)] and rec(a, thr=0.2) == sorted(exp + [(1, 2), (1, 3), (1, 4)]), "the threshold (the 0.2 mutant also keeps source 1; 2 -> 3 stays out: p_none wins)"
    open_ = dict(a, valid=torch.ones_like(a["valid"]))                                                         # mutant: no valid mask
    assert rec(open_) == sorted(exp + [(4, 5), (5, 3)])
    for mut in (rec(a, thr=0.2), rec(bk), rec(open_)):
        must_fail(holds, mut == exp)                                                                        # each mutant (threshold, kept ends, no valid mask) differs from the record
    pr = JK.probs(a)
    assert abs(float(pr["p"][0, 0, 2]) - 0.34) < 1e-6 and abs(float(pr["pnone"][0, 0, 2]) - 0.30) < 1e-6 and bool(pr["keep"][0, 4]) and not bool(pr["keep"][0, 5]) and not bool(JK.probs(bk)["keep"][0, 4])
    print("decode: p >= 0.3 and p >= p(none), both ends kept (p_none < 0.5, 0.5 itself dropped), forks keep both branches, a padded source / destination never forms a link; "
          "mutants (thr 0.2, no end rule, no valid mask) differ: ok")


# ------------------------------------------------------------------------------------------------------------------------------ the head
def _batch(B=3, K=48, seed=0):
    return JK.synthetic_batch(B, K, seed)


def test_head():
    torch.manual_seed(0)
    b = _batch()
    m = JK.JTokenHead()
    m.set_step(JK.SWITCH + 5)                                                # the learned key weights: every parameter is on the path
    out = m(b)
    B, K = b["pos"].shape[:2]
    assert out["logp"].shape == (B, 2, K, K + 1) and out["scale"].shape == (B, K, K, 2) and out["types"].shape == (B, K, 7) and out["types1"].shape == (B, K, 7)
    pr = JK.probs(out)
    assert float((out["logp"].exp().sum(-1) - 1).abs().max()) < 1e-5
    _, al = JT.pair_geometry(b["pos"], b["dirv"], b["valid"], e=b["e"].sigmoid())
    cd = torch.cdist(b["pos"].double(), b["pos"].double()).float()          # an independent chord length (the head reads it from the pair features)
    off = (cd - JK.SHORT).abs() > 1e-4                                      # tokens 4 apart on a lane are exactly 8 m: the class of such a pair is a float tie (the head reads L / 20 * 20)
    assert torch.equal(out["cand"][:, 0][off], (al & (cd <= JK.SHORT))[off]) and torch.equal(out["cand"][:, 1][off], (al & (cd > JK.SHORT))[off]), "the chord classes of the candidates are <= 8 m / > 8 m of the token distance"
    assert float((out["chord"][al] - cd[al]).abs().max()) < 1e-3 and bool((out["cand"][:, 1]).any()) and bool((out["cand"][:, 0]).any()), "the chord of the head is the token distance"
    assert float((pr["p"] * ~al).abs().max()) == 0.0, "the candidates are the allowed gate of `jtok.pair_geometry`"
    nv = ~b["valid"]
    assert float(pr["p"][nv[:, :, None].expand_as(al)].abs().max()) == 0.0 and float(pr["p"][nv[:, None, :].expand_as(al)].abs().max()) == 0.0, "padding is a key or a source"
    assert float((out["logp"].exp()[..., -1][nv[:, None, :].expand(-1, 2, -1)] - 1).abs().max()) < 1e-6, "a padding source has p(none) = 1"
    loss, info = LJK.jtoken_loss(out, b)
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"no gradient: {dead}"
    # a head is a function of the SET: permuting the tokens permutes the outputs; padding tokens do not reach the valid ones
    m.eval()
    perm = torch.randperm(K)
    with torch.no_grad():
        o1 = m(b)
        o2 = m({k: (v[:, perm] if v.dim() >= 2 and v.shape[1] == K else v) for k, v in b.items() if k in JK.INPUTS})
        assert float((o2["logp"][..., :K] - o1["logp"][..., :K][:, :, perm][:, :, :, perm]).abs().max()) < 1e-4 and float((o2["types"] - o1["types"][:, perm]).abs().max()) < 1e-4, "the token order reaches the outputs"
        junk = {k: (torch.randn_like(v) * 5 if v.is_floating_point() and k != "valid" else v) for k, v in b.items() if k in JK.INPUTS}
        for k, v in junk.items():
            if k != "valid":
                junk[k] = torch.where(b["valid"].view(*b["valid"].shape, *[1] * (v.dim() - 2)), b[k], v)
        o3 = m(junk)
        vv = b["valid"]
        pad_ok = lambda o: float((o["logp"][..., :K][vv[:, None, :, None].expand(-1, 2, -1, K)] - o1["logp"][..., :K][vv[:, None, :, None].expand(-1, 2, -1, K)]).abs().max()) < 1e-4 and \
            float((o["types"][vv] - o1["types"][vv]).abs().max()) < 1e-4  # noqa: E731
        assert pad_ok(o3), "a padding token changes the outputs of the valid tokens"
        o_open = m(dict(junk, valid=torch.ones_like(vv)))
        must_fail(holds, pad_ok(o_open))                                                                    # mutant: the padding mask off
    print(f"head: shapes, probabilities sum to 1, the candidates are the allowed gate split by an independent token distance (padding neither key nor source), gradient reaches every parameter, order and padding "
          f"leave the valid outputs alone (padding mutant fails): ok; parameters {JK.count_params(m)} ({JK.count_params(m) / 1e6:.3f} M)")


def test_key_bias():
    """what it proves: (1) after the switch the head's own p_none read after layer 1 AND after the last layer receives gradient from a loss on the links only (type CE off), i.e. through the key bias of layers 2, 3 and the
    wiring layer; before the switch it receives none (the keys use sigmoid(e)); patching SWITCH to 0 (a head that is always in the learned mode) breaks the second half. (2) The Python mirror of the step follows
    load_state_dict. (3) no gradient reaches any cached input (e, e_raw, flags, tok, ...): B' is frozen and the hard top-K is cut"""
    torch.manual_seed(0)
    b = _batch(2)

    def grads(step):
        m = JK.JTokenHead()
        m.set_step(step)
        reads = []

        def tap(mod, inp, o):                                                # keeps the gradient of each reading of the type Linear (layer 1, last layer)
            o.retain_grad()
            reads.append(o)
        m.types.register_forward_hook(tap)
        out = m(b)
        LJK.jtoken_loss(out, b, type_w=0.0, scale_w=0.0)[0].backward()
        return [float(r.grad.abs().sum()) if r.grad is not None else 0.0 for r in reads], m

    def before_switch_is_silent():
        (z1, z3), m0 = grads(0)
        assert z1 == 0.0 and z3 == 0.0 and (m0.types.weight.grad is None or float(m0.types.weight.grad.abs().sum()) == 0.0), "before the switch the links must not reach the type head"

    (g1, g3), m = grads(JK.SWITCH + 1)
    assert g1 > 0 and g3 > 0, (g1, g3)
    assert float(m.types.weight.grad.abs().sum()) > 0
    before_switch_is_silent()
    must_fail(patched, JK, "SWITCH", 0, before_switch_is_silent)                                            # mutant: always in the learned mode
    m_on = JK.JTokenHead().eval()
    m_on.set_step(JK.SWITCH + 1)
    m_load = JK.JTokenHead().eval()
    m_load.load_state_dict(m_on.state_dict())
    m_off = JK.JTokenHead().eval()
    m_off.load_state_dict(m_on.state_dict())
    m_off.set_step(0)
    with torch.no_grad():
        assert m_load._step_py == JK.SWITCH + 1 and torch.equal(m_load(b)["logp"], m_on(b)["logp"]) and not torch.equal(m_off(b)["logp"], m_on(b)["logp"]), "the step mirror does not follow load_state_dict"
    live = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in b.items()}
    m1 = JK.JTokenHead()
    m1.set_step(JK.SWITCH + 1)
    LJK.jtoken_loss(m1(live), live)[0].backward()
    assert all(live[k].grad is None for k in JK.INPUTS if live[k].is_floating_point()), "a gradient reached a cached input"

    def inputs_cut():                                                        # mutant: the graph is not cut at the inputs
        live2 = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in b.items()}
        LJK.jtoken_loss(m1(live2), live2)[0].backward()
        return all(live2[k].grad is None for k in JK.INPUTS if live2[k].is_floating_point())
    must_fail(holds, patched(JK, "_const", lambda t: t, inputs_cut))
    print(f"key_bias: after the switch a links-only loss moves the head's own type reading (layer 1 grad {g1:.2e}, last layer {g3:.2e}), before it not (SWITCH = 0 mutant fails); the step mirror follows "
          f"load_state_dict; no gradient into the cached inputs (cut mutant fails): ok")


def test_writer_free():
    b = _batch(2)
    m = JK.JTokenHead().eval()

    class NoGT(dict):
        def __getitem__(self, k):
            if k in JK.GT_KEYS or k.startswith("gt_"):
                raise KeyError(f"the forward read the GT key {k}")
            return super().__getitem__(k)

    clean = {k: v for k, v in b.items() if k in JK.INPUTS}
    with torch.no_grad():
        o_min = m(NoGT(clean))
        poison = dict(b, y=torch.full_like(b["y"], 1), ylong=~b["ylong"], yscale=torch.full_like(b["yscale"], 7.0), ytype=torch.full_like(b["ytype"], 3), gt_out=torch.ones(2))
        assert torch.equal(m(NoGT(poison))["logp"], o_min["logp"]) and torch.equal(m(poison)["types"], o_min["types"]), "the forward depends on a GT key"

        def leaky(bb):                                                       # mutant: a forward that reads a GT key
            return m(bb)["types"] + bb["y"].float().sum()
        try:
            leaky(NoGT(clean))
        except KeyError:
            pass
        else:
            raise RuntimeError("the mutant read a GT key without being caught")
        assert not torch.equal(leaky(poison), m(clean)["types"])
    print("writer_free: the forward runs on a batch without y / ylong / yscale / ytype / gt_* and ignores them when present; a forward that reads one is caught: ok")


def test_overfit():
    for step0 in (0, JK.SWITCH):                                             # the input key weights, then the head's own
        torch.manual_seed(0)
        b = _batch(8, 48, seed=3)
        m = JK.JTokenHead()
        m.set_step(step0)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        first, n = None, 0
        for n in range(1, 301):
            loss, info = LJK.jtoken_loss(m(b), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = float(loss) if first is None else first
            if float(loss) < 0.25 * first and n >= 100:
                break
        m.eval()
        with torch.no_grad():
            out = m(b)
            last = LJK.jtoken_loss(out, b)[0]
            dec = JK.decode(out)
        link, y = dec["link"], b["y"] > 0
        has = y.any(-1) & b["valid"]
        recall = float((link & y).any(-1)[has].float().mean())
        prec = float((link & y).sum() / link.sum().clamp_min(1))
        transposed = int((link & y.transpose(1, 2)).sum())
        assert float(last) < 0.5 * first, (step0, first, float(last))
        assert recall > 0.85 and prec > 0.85 and transposed == 0, (recall, prec, transposed)
        print(f"overfit (key weights {'own' if step0 else 'input'}): loss {first:.3f} -> {float(last):.3f} ({float(last) / first:.2f} x) in {n} steps on 8 frames; decode of record: source recall {recall:.3f}, link precision {prec:.3f}, "
              f"hits against the transposed GT {transposed}")


def test_trainer():
    kw = dict(smoke_k=48)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        argv = ["--name", "t_s0", "--seed", "0", "--smoke", "--bs", "2"]
        must_fail(JKT.main, argv[:4], runs=runs)                                                              # no GPU, no smoke: refused
        calls = [0]
        real = LJK.jtoken_loss

        def dying(out, batch, **k):                                          # a kill at the 12th step (the checkpoint of step 10 exists)
            calls[0] += 1
            if calls[0] == 12:
                raise KeyboardInterrupt("kill")
            return real(out, batch, **k)
        LJK.jtoken_loss = dying
        try:
            try:
                JKT.train(JKT.parse_args(argv), runs=runs, ckpt_every=5, **kw)
            except KeyboardInterrupt:
                pass
            else:
                raise RuntimeError("the kill did not happen")
        finally:
            LJK.jtoken_loss = real
        rd = os.path.join(runs, "t_s0")
        assert os.path.exists(os.path.join(rd, "ckpt.pt")) and not os.path.exists(os.path.join(rd, ".done"))
        JKT.train(JKT.parse_args(argv), runs=runs, ckpt_every=5, **kw)
        log = open(os.path.join(rd, "train.log"), encoding="utf-8").read()
        assert "resumed from" in log and "at step 10" in log and all(os.path.exists(os.path.join(rd, f)) for f in (".done", "model.pt", "ckpt.pt", "config.json", "log.jsonl", "train.log")), log[-600:]
        m, args = LN.load_head("t_s0", "cpu", runs)
        assert (args["seed"], args["steps"], args["bs"]) == (0, 20, 2) and int(m.step) == 20 and m._step_py == 20, "the step buffer of the head must follow the run through the kill and the resume"
        rows = [json.loads(r) for r in open(os.path.join(rd, "log.jsonl"))]
        assert all(k in rows[-1] for k in ("nll_s", "nll_l", "ce1", "ce3", "gpu_gb")), rows[-1]
        for other in (["--lr", "5e-4"], ["--ident", "e273_p"], ["--none-weight", "0.5"]):
            must_fail(JKT.train, JKT.parse_args(argv + other), runs=runs, ckpt_every=5, **kw)                # another recipe in the same folder
        assert "t_s0" in argv
        sys.modules["lane_ptok_data"] = types.SimpleNamespace()                                              # --smoke must not read the loader: a module without one stands in its place
        try:
            JKT.train(JKT.parse_args(["t_noload" if x == "t_s0" else x for x in argv]), runs=runs, ckpt_every=5, **kw)
        finally:
            sys.modules["lane_ptok_data"] = lane_ptok_data
        # a non-smoke run through a fake loader that returns (None, batches) like `lane_ptok_data.loader`
        seen = {}

        def fake_loader(split, seed, bs, workers, **k):
            seen.update(split=split, seed=seed, bs=bs, workers=workers, **k)
            return None, [JK.synthetic_batch(bs, 48, seed=s) for s in range(3)]
        sys.modules["lane_ptok_data"] = types.SimpleNamespace(loader=fake_loader)
        try:
            JKT.train(JKT.parse_args(["--name", "t_real", "--seed", "1", "--steps", "6", "--bs", "2", "--workers", "0", "--ident", "e273_p"]), runs=runs, ckpt_every=3)
        finally:
            sys.modules["lane_ptok_data"] = lane_ptok_data
        assert seen == dict(split="train", seed=1, bs=2, workers=0, ident="e273_p"), seen
        m2, a2 = LN.load_head("t_real", "cpu", runs)
        assert a2["steps"] == 6 and a2["ident"] == "e273_p" and m2._step_py == 6 and os.path.exists(os.path.join(runs, "t_real", ".done"))
        sys.modules["lane_ptok_data"] = types.SimpleNamespace(loader=fake_loader)
        seen.clear()
        try:
            JKT.train(JKT.parse_args(["--name", "t_def", "--steps", "2", "--bs", "2", "--workers", "0"]), runs=runs, ckpt_every=3)
        finally:
            sys.modules["lane_ptok_data"] = lane_ptok_data
        assert seen == dict(split="train", seed=0, bs=2, workers=0, ident="e292_a"), seen                      # the default: the one cache of the lane head of the seed
    print("trainer: smoke on synthetic batches (20 steps, kill at 12, resume from the checkpoint of step 10, the step buffer = 20), run-directory files, loss parts logged, another recipe (lr / ident / none weight) refused, "
          "no loader import; a non-smoke run through a fake loader (ident passed, (None, batches) unpacked): ok")


def test_arch_options():
    # defaults = the head of the first build, bit for bit
    dflt = JKT.parse_args(["--name", "x", "--none-weight", "1.0"])
    torch.manual_seed(0)
    new = LN.build_head(vars(dflt))
    torch.manual_seed(0)
    old = JK.JTokenHead()
    torch.manual_seed(0)
    expl = JK.JTokenHead(live_inputs=False)                                                                 # the constructor argument at its default
    for h in (new, expl):
        assert list(h.state_dict()) == list(old.state_dict()) and all(torch.equal(v, old.state_dict()[k]) for k, v in h.state_dict().items()), "default options must build the old head"
    assert JK.count_params(new) == 127131 and LN.build_head({}).state_dict().keys() == old.state_dict().keys(), "an old run's args (no options) mean the defaults"
    # 20 synthetic steps with the defaults reproduce what the committed code did (recorded from the committed code before the changes, 2 threads: seed 0, bs 2, K 48). THESE CONSTANTS HOLD IN THIS TEST'S PROCESS ONLY (this file
    # sets 2 threads; a fresh process with another thread count / BLAS order gives 2.8011269 for the same run): the portable check is the same-process one below (defaults against the explicit default arguments)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        JKT.train(JKT.parse_args(["--name", "r", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]), runs, 500, 48)
        row = [json.loads(r) for r in open(os.path.join(runs, "r", "log.jsonl"))][-1]
        sd = torch.load(os.path.join(runs, "r", "model.pt"), map_location="cpu", weights_only=False)["state"]
        tot = sum(float(v.double().abs().sum()) for v in sd.values() if v.dtype.is_floating_point)
        assert abs(row["total"] - 2.8015905141830446) < 1e-12 and abs(row["nll_s"] - 1.288527250289917) < 1e-12 and abs(tot / 8250.530362168049 - 1) < 1e-14, (row["total"], tot)
        JKT.train(JKT.parse_args(["--name", "r2", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "1.0"]), runs, 500, 48)
        rows2 = [json.loads(r) for r in open(os.path.join(runs, "r2", "log.jsonl"))]
        sd2 = torch.load(os.path.join(runs, "r2", "model.pt"), map_location="cpu", weights_only=False)["state"]
        assert [r["total"] for r in rows2] == [json.loads(r)["total"] for r in open(os.path.join(runs, "r", "log.jsonl"))] and list(sd2) == list(sd) and all(torch.equal(v, sd[k]) for k, v in sd2.items()), "explicit default arguments = the defaults, same process, bit for bit"
        assert LN.arch({}) == (64, 4, 3, 128), "the champion's architecture"
        os.makedirs(os.path.join(runs, "old"))
        json.dump(dict(design="jtoken", args=dict(seed=0, cache=None, ident="e292_a", K=None, bs=2, steps=20, lr=1e-3)), open(os.path.join(runs, "old", "config.json"), "w"))     # a config of before the options
        a_old = vars(JKT.parse_args(["--name", "old", "--bs", "2", "--steps", "20"]))
        JKT.check_start(a_old, os.path.join(runs, "old"))
        must_fail(JKT.check_start, dict(a_old, none_weight=0.5), os.path.join(runs, "old"))
        ck = torch.load(os.path.join(runs, "r", "model.pt"), map_location="cpu", weights_only=False)       # `check_removed`: a stored run's removed options at their off value resume and load as before, another value is refused by name
        bad = dict(prior="rule", teacher="rule", init="runs/x/model.pt", token_drop=0.1, none_weight_near=0.73, d=96, heads=2, layers=4, ffn=512,
                   pos_loss="max", distill_from="runs/a,runs/b", distill_weight=0.5, distill_sha256=["0"], noise_pos=0.2, noise_e=0.5, cache="runs/k384", K=384,
                   rounds=2, wire_weight=0.5, msg="uniform", pos_head=True, curve_w=0.5, node_w=0.5, min_e=0.2)
        off = {k: v for options in LN.REMOVED.values() for k, v in options.items()}
        tag = {k: t for t, options in LN.REMOVED.items() for k in options}
        assert set(bad) == set(off) and set(tag.values()) == {"v1", "archive/jtoken-before-simplify-2026-11-17", "archive/e399-thinning-2026-11-28"} and tag["rounds"] == "v1", "a refusal test for every removed option"
        for name, extra in [("o_dflt", off)] + [(f"o_{k}", {k: v}) for k, v in bad.items()]:
            os.makedirs(os.path.join(runs, name))
            json.dump(dict(design="jtoken", args=dict(ck["args"], **extra)), open(os.path.join(runs, name, "config.json"), "w"))
            torch.save(dict(ck, args=dict(ck["args"], **extra)), os.path.join(runs, name, "model.pt"))
            if name == "o_dflt":
                JKT.check_start(dict(ck["args"], name=name), os.path.join(runs, name))
                assert all(torch.equal(v, sd[k]) for k, v in LN.load_head(name, "cpu", runs)[0].state_dict().items())
                continue
            for fn in (lambda: JKT.check_start(dict(ck["args"], name=name), os.path.join(runs, name)), lambda: LN.load_head(name, "cpu", runs)):
                try:
                    fn()
                except AssertionError as e:
                    (k,) = extra
                    assert f"'{k}'" in str(e) and f"the tag {tag[k]} keeps them" in str(e), e                 # the message names the option and the tag that keeps its code
                else:
                    raise RuntimeError(f"a run with the removed option {extra} was accepted")
    print("arch_options: defaults = the old head (127131 parameters, 20 smoke steps reproduce the recorded loss 2.8015905 and the weights, also with --none-weight 1), the architecture is the constant (64, 4, 3, 128); "
          f"an old config resumes with the defaults and another recipe is refused; a stored run's removed options ({', '.join(off)}) at their off value resume and load, another value is refused with its name and tag: ok")


# ------------------------------------------------------------------------------------------------------------------------------ the head seed, the live inputs (S4)
def _cache_batch(n=3):
    """the first n VAL frames of the predicted-node cache of B' seed 0, or None when it is not there"""
    import lane_ptok_data as LP
    try:
        ds = LP.PTokData("val", 0, "e292_a")
    except (AssertionError, FileNotFoundError) as e:
        if isinstance(e, AssertionError) and "finished caches match" not in str(e):
            raise
        return None
    return LP.collate([ds[i] for i in range(n)])


def _mutant(*edits, double=False, of=JK):
    """jtoken.py (of=JKT: jtoken_train.py) with each (old, new) text replaced (old found exactly once), loaded in this process as its own module: a textual mutant; double: then every `.float()` becomes `.double()` (the float64
    copy of `_live_fd_ok`)"""
    src = merged(of.__file__)                                                    # jtoken: with its losses (D152)
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    if double:
        src = src.replace(".float()", ".double()")
    mod, path = types.ModuleType(of.__name__ + "_mutant"), list(sys.path)
    mod.__file__ = of.__file__
    try:
        exec(compile(src, of.__file__, "exec"), mod.__dict__)
    finally:
        sys.path[:] = path
    return mod


def _live_ok(mod, before):
    """S4 (E372) `live_inputs` of `mod.JTokenHead` (jtoken itself or a textual mutant): a synthetic batch, the head after the switch (the champion's state), the champion's loss (none weight 0.25), the float INPUTS as leaves
    with requires_grad. `before` = the module with the forward's input line of before (`_const` on every input). The default = before bit for bit (every output, every parameter gradient), no gradient into an input;
    live_inputs=True: the same outputs and parameter gradients, a nonzero gradient on every float input -> {input: the sum of |grad|} of the live head"""
    b = _batch(2)

    def run(m_mod, **kw):
        torch.manual_seed(0)
        m = m_mod.JTokenHead(**kw)
        m.set_step(JK.SWITCH + 1)
        x = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in b.items()}
        out = m(x)
        m_mod.jtoken_loss(out, x, none_weight=0.25)[0].backward()
        return out, {n: p.grad for n, p in m.named_parameters()}, {k: x[k].grad for k in JK.INPUTS if x[k].is_floating_point()}
    ref, dflt, live = run(before), run(mod), run(mod, live_inputs=True)
    same = lambda r: set(r[0]) == set(ref[0]) and all(torch.equal(ref[0][k], r[0][k]) for k in ref[0]) and all(torch.equal(g, r[1][n]) for n, g in ref[1].items())  # noqa: E731
    assert same(dflt) and all(g is None for g in dflt[2].values()), "the default is not the head of before (outputs, parameter gradients), or a gradient reached an input"
    assert same(live) and all(g is not None and float(g.abs().sum()) > 0 for g in live[2].values()), {k: None if g is None else float(g.abs().sum()) for k, g in live[2].items()}
    return {k: float(g.abs().sum()) for k, g in live[2].items()}


FD_H, FD_TOL = 1e-6, 1e-4                                                     # `_live_fd_ok`: the step of the central difference, the tolerance relative to the analytic directional derivative


def _live_fd_ok(mod):
    """the FULL live gradient of `mod` (the float64 copy of jtoken, `_mutant(..., double=True)`, or a mutant of it): a synthetic batch of 2 frames x 64 tokens with the valid positions jittered by about 1 cm (the synthetic lanes
    have pairs at exactly 8.0 m = SHORT, where the class flips with rounding, and duplicates exactly beside their token), the live head at step 0 and after the switch, the champion's loss; per float input a central difference
    with h = FD_H along a random direction on the valid tokens against the analytic directional derivative (|fd - an| <= FD_TOL |an|: a zero analytic derivative fails too) -> the largest relative error"""
    b = {k: (v.double() if v.is_floating_point() else v) for k, v in JK.synthetic_batch(2, 64, seed=0).items()}
    gen, vm = torch.Generator().manual_seed(7), b["valid"].double()
    b["pos"] = b["pos"] + 0.01 * torch.randn(b["pos"].shape, generator=gen, dtype=torch.float64) * vm[..., None]
    keys, worst = [k for k in JK.INPUTS if b[k].is_floating_point()], 0.0
    for step in (0, JK.SWITCH + 1):
        torch.manual_seed(0)
        m = mod.JTokenHead(live_inputs=True).to(torch.float64).eval()      # eval: the step counter stays where it is set
        m.set_step(step)
        loss = lambda bb: mod.jtoken_loss(m(bb), bb, none_weight=0.25)[0]  # noqa: E731
        x = dict(b, **{k: b[k].clone().requires_grad_(True) for k in keys})
        loss(x).backward()
        for k in keys:
            v = torch.randn(b[k].shape, generator=gen, dtype=torch.float64) * vm.view(*vm.shape, *[1] * (b[k].dim() - 2))
            with torch.no_grad():
                fd = float(loss(dict(b, **{k: b[k] + FD_H * v})) - loss(dict(b, **{k: b[k] - FD_H * v}))) / (2 * FD_H)
            an = float((x[k].grad * v).sum())
            assert abs(fd - an) <= FD_TOL * abs(an), (step, k, fd, an)
            worst = max(worst, abs(fd - an) / abs(an))
    return worst


def test_live_inputs():
    """S4 (E372) `JTokenHead(live_inputs=)`: `_live_ok` on jtoken and three textual mutants (the detach kept under live_inputs, the option ignored, the detach dropped under the default); `_live_fd_ok` (the full gradient of
    every float input, float64 finite differences) on jtoken and two textual mutants (the geometry path cut, the key weight of layers 2 - 3 cut before the switch); `build_head` reads it from a run's args (absent = False,
    also through `load_model`); `arch` and the state keys do not hold it"""
    line = "b = {k: batch[k] if self.live_inputs else _const(batch[k]) for k in INPUTS}"
    before = _mutant((line, "b = {k: _const(batch[k]) for k in INPUTS}"))
    g = _live_ok(JK, before)
    mutants = {"the detach kept under live_inputs": [("batch[k] if self.live_inputs", "batch[k].detach() if self.live_inputs")], "the option ignored": [("self.live_inputs = live_inputs", "self.live_inputs = False")],
               "the detach dropped under the default": [("else _const(batch[k])", "else batch[k]")]}
    for edits in mutants.values():
        must_fail(_live_ok, _mutant(*edits), before)
    fd = _live_fd_ok(_mutant(double=True))
    fd_mutants = {"the geometry path cut": [("JT.pair_geometry(pos, dirv, valid, e=w_in)", "JT.pair_geometry(pos.detach(), dirv.detach(), valid, e=w_in.detach())")],
                  "the key weight of layers 2 - 3 cut before the switch": [("x = layer(x, g, lw(w1 if learned else w_in), valid)", "x = layer(x, g, lw(w1 if learned else w_in.detach()), valid)")]}
    for edits in fd_mutants.values():
        must_fail(_live_fd_ok, _mutant(*edits, double=True))
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        os.makedirs(os.path.join(runs, "live"))
        torch.save(dict(state=JK.JTokenHead().state_dict(), args=dict(live_inputs=True), steps=0), os.path.join(runs, "live", "model.pt"))
        assert LN.load_head("live", "cpu", runs)[0].live_inputs is True
    assert LN.build_head({}).live_inputs is False and LN.build_head(vars(JKT.parse_args(["--name", "x"]))).live_inputs is False and LN.arch(dict(live_inputs=True)) == LN.arch({})
    assert set(JK.JTokenHead(live_inputs=True).state_dict()) == set(JK.JTokenHead().state_dict())
    print(f"live_inputs: the default = the head of before bit for bit (outputs, every parameter gradient; the champion's loss after the switch), no gradient into the inputs; live_inputs=True: the same outputs and parameter "
          f"gradients, a gradient on every float input (sum |grad|: {', '.join(f'{k} {v:.2e}' for k, v in g.items())}); textual mutants ({', '.join(mutants)}) fail; the full live gradient of every float input against "
          f"float64 central differences (h {FD_H:g}, step 0 and after the switch): largest relative error {fd:.1e} <= {FD_TOL:g}, textual mutants ({', '.join(fd_mutants)}) fail; build_head / load_model read it from the args "
          f"(absent = False), not in arch or the state keys: ok")


def test_head_seed():
    """--head-seed N: the head's initialisation and the data order use N, --seed keeps choosing the cache (the loader's seed); head_seed in the recipe; --head-seed equal to --seed (and none) = the default run bit for bit"""
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        base = ["--smoke", "--bs", "2", "--workers", "0"]
        step0, real_bh = {}, LN.build_head
        for name, extra in (("s0", ["--seed", "0"]), ("s0h0", ["--seed", "0", "--head-seed", "0"]), ("s0h2", ["--seed", "0", "--head-seed", "2"]), ("s2", ["--seed", "2"])):
            def built(args, name=name):                                                                       # the trainer's step-0 head
                m = real_bh(args)
                step0[name] = {k: v.clone() for k, v in m.state_dict().items()}
                return m
            patched(LN, "build_head", built, JKT.train, JKT.parse_args(["--name", name] + extra + base), runs, 500, 48)
        st = lambda n: torch.load(os.path.join(runs, n, "model.pt"), map_location="cpu", weights_only=False)["state"]  # noqa: E731
        same = lambda x, y: all(torch.equal(v, y[k]) for k, v in x.items())  # noqa: E731
        seeded = lambda hs: (JKT.T.seed_all(hs), LN.build_head({}).state_dict())[1]  # noqa: E731
        tot = lambda n: [json.loads(r)["total"] for r in open(os.path.join(runs, n, "log.jsonl")) if "total" in r]  # noqa: E731
        assert same(step0["s0h2"], seeded(2)) and same(step0["s2"], seeded(2)) and same(step0["s0"], seeded(0)) and not same(step0["s0h2"], step0["s0"]), "the head seed initialises the head (build_head after seed_all)"
        assert tot("s0h2") == tot("s2") and tot("s0h2") != tot("s0"), "the data (smoke: the synthetic batches) follow the head seed"
        assert same(st("s0h0"), st("s0")) and tot("s0h0") == tot("s0"), "--head-seed = --seed is the default run"
        a = LN.load_head("s0h2", "cpu", runs)[1]
        assert (a["seed"], a["head_seed"]) == (0, 2) and LN.load_head("s0", "cpu", runs)[1]["head_seed"] is None
        must_fail(JKT.check_start, dict(a, head_seed=3), os.path.join(runs, "s0h2"))                           # another head seed in the same folder
        # cache-free: a fake loader that returns a SHUFFLING DataLoader over 16 named synthetic batches; --seed chooses the cache, the head seed the batch order
        seen, order, now, loss_real = {}, {}, [None], LJK.jtoken_loss

        def fake_loader(split, seed, bs, workers, **k):
            seen[now[0]] = dict(seed=seed, **k)
            return None, torch.utils.data.DataLoader([dict(JK.synthetic_batch(bs, 48, seed=s), token=[f"batch{s}"] * bs) for s in range(16)], batch_size=None, shuffle=True)

        def first_batch(out, batch, **k):
            order.setdefault(now[0], batch["token"][0])
            return loss_real(out, batch, **k)
        sys.modules["lane_ptok_data"] = types.SimpleNamespace(loader=fake_loader)
        try:
            for name, extra in (("f0", []), ("f0h0", ["--head-seed", "0"]), ("f0h2", ["--head-seed", "2"])):
                now[0] = name
                patched(LJK, "jtoken_loss", first_batch, JKT.train, JKT.parse_args(["--name", name, "--seed", "0", "--steps", "1", "--bs", "2", "--workers", "0"] + extra), runs, 3)
        finally:
            sys.modules["lane_ptok_data"] = lane_ptok_data
        assert all(s["seed"] == 0 and s["ident"] == "e292_a" for s in seen.values()) and len(seen) == 3, "--seed chooses the cache of B' seed 0"
        assert order["f0h0"] == order["f0"] != order["f0h2"], f"the batch order follows the head seed, --head-seed 0 = the default: {order}"
        log = open(os.path.join(runs, "s0h2", "train.log"), encoding="utf-8").read()
        assert "cache seed 0, head seed 2" in log and "trainable 127131 parameters" in log and "cache seed 0, head seed 0" in open(os.path.join(runs, "s0", "train.log"), encoding="utf-8").read(), log[:400]
        # the real loader of the cache (when it is there): the cache directory is B' seed 0's for both runs, the frames of the first batch follow the head seed
        real = "SKIPPED (no cache)"
        if _cache_batch(1) is not None:
            import lane_ptok_data as LP
            dirs, first, loss0 = {}, {}, LJK.jtoken_loss

            def spy_loader(*x, **k):
                ds, dl = LP_loader(*x, **k)
                dirs[cur[0]] = ds.dir
                return ds, dl

            def spy_loss(out, batch, **k):
                first.setdefault(cur[0], list(batch["token"]))
                return loss0(out, batch, **k)
            LP_loader, cur = LP.loader, [None]
            for name, extra in (("c0", []), ("c0h2", ["--head-seed", "2"])):
                cur[0] = name
                patched(LP, "loader", spy_loader, patched, LJK, "jtoken_loss", spy_loss, JKT.train,
                        JKT.parse_args(["--name", name, "--seed", "0", "--steps", "1", "--bs", "2", "--workers", "0"] + extra), runs, 500)
            assert all(f"be292_a_0_K" in d for d in dirs.values()) and len(dirs) == 2, dirs
            assert first["c0"] != first["c0h2"], first
            real = f"real loader: both runs read {os.path.basename(os.path.dirname(dirs['c0']))}, first batch {first['c0']} vs {first['c0h2']}"
    print(f"head_seed: --seed 0 --head-seed 2 reads the seed-0 cache, initialises and orders the data as --seed 2 (step-0 weights and the loss curve equal), not as --seed 0; --head-seed 0 = the default run bit for bit; "
          f"head_seed in the recipe (another one refused); the log names both seeds and 127131 parameters; a shuffling DataLoader as the loader: first batch {order['f0']} (default, --head-seed 0) vs {order['f0h2']} (--head-seed 2); {real}: ok")


# ------------------------------------------------------------------------------------------------------------------------------ the champion's recorded numbers
# Recorded in this test's process (2 threads, CPU; another process may differ in the last digits, as `arch_options`) from the code before the switches of E379 / E392 / E395 existed (commit 866ba519 and git HEAD before E392).
CHAMP = (4.647189617156982, 1.6313660144805908, 2.20220827460289, 8251.217476223794)        # the champion recipe (--none-weight 0.25, seed 0, smoke bs 2, K 48): the first and the last of the 20 losses, the log row, the sum of |weights|
# `_champ_numbers` of a seeded head (torch seed 0) at head step 0 and SWITCH + 5, and of the stored E373 head `runs/e373_cb_s0` (train mode)
CHAMP_HEAD = {0: (118588089.77677026, 9123.041142880917, 503.1498308633454, 525.4964824039489, 4.633667469024658, 928.3080354075339),
              JK.SWITCH + 5: (118588330.5475591, 9126.07040989399, 496.1637700991705, 525.4964824039489, 4.695507526397705, 921.0818347715707),
              "e373": (118604369.3126266, 6105.978199899197, 2566.0501022301614, 2319.9599598338827, 5.47215461730957, 4672.238252198448)}
E373 = "e373_cb_s0"                                                           # the champion's stage-1 head of seed 0 (head seed 2)


def _champ_numbers(head):
    """forward + loss (none weight 0.25) + backward of `head` (a `JTokenHead` in train mode, its step as set) on a synthetic batch (3 frames, K 48) -> (sum |x| in float64 of logp, scale, types, types1; the loss; sum |grad| of every parameter)"""
    b = JK.synthetic_batch(3, 48, seed=0)
    out = head(b)
    loss = LJK.jtoken_loss(out, b, none_weight=0.25)[0]
    loss.backward()
    s = lambda t: float(t.double().abs().sum())  # noqa: E731
    return tuple(s(out[k]) for k in ("logp", "scale", "types", "types1")) + (float(loss), sum(s(p.grad) for p in head.parameters()))


def _twin_step():
    """one step of the champion's stage-1 recipe of seed 0 on the real TRAIN cache (CPU): the head's init is E373's rebuilt (seed_all(2), build_head of its args), the first batch is the first of E373's order and the first loss is
    E373's logged one (the GPU in bf16 there: to 2e-3) -> a note; skipped where the run or the cache is not on this machine"""
    import re
    import lane_ptok_data as LP
    e373 = os.path.join(JKT.T.RUNS, E373)
    try:
        LP.cache_dir("train", 0, "e292_a")                                        # asserts unless the finished cache is there
    except AssertionError as e:
        if not str(e).startswith("no finished cache"):
            raise                                                                 # several caches match: a real problem, not a skip
        return "the step on the real cache SKIPPED (no TRAIN cache of B' seed 0)"
    if not os.path.exists(os.path.join(e373, "model.pt")):
        return f"the step on the real cache SKIPPED (no run {E373})"
    built, tokens, real_bh, real_loss = [], [], LN.build_head, LJK.jtoken_loss

    def rec(args):                                                             # the trainer's step-0 head
        m = real_bh(args)
        built.append({k: v.clone() for k, v in m.state_dict().items()})
        return m

    def first_batch(out, batch, **k):
        tokens.append(list(batch["token"]))
        return real_loss(out, batch, **k)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        patched(LN, "build_head", rec, patched, LJK, "jtoken_loss", first_batch, JKT.train,
                JKT.parse_args(["--name", "twin", "--seed", "0", "--head-seed", "2", "--ident", "e292_a", "--steps", "1", "--bs", "8", "--workers", "0", "--none-weight", "0.25"]), runs, 500)
        log = open(os.path.join(runs, "twin", "train.log"), encoding="utf-8").read()
    JKT.T.seed_all(2)
    init = real_bh(torch.load(os.path.join(e373, "model.pt"), map_location="cpu", weights_only=False)["args"]).state_dict()
    JKT.T.seed_all(2)
    first = next(iter(LP.loader("train", 0, 8, 0, ident="e292_a")[1]))
    assert list(built[-1]) == list(init) and all(torch.equal(v, init[k]) for k, v in built[-1].items()), "the head's init is not E373's"
    assert tokens[0] == list(first["token"]), "the first batch is not the first of E373's order"
    mine, theirs = (float(re.search(r"first loss of this process: ([0-9.]+)", t).group(1)) for t in (log, open(os.path.join(e373, "train.log"), encoding="utf-8").read()))
    assert abs(mine - theirs) < 2e-3, (mine, theirs)
    return f"one step on the real cache: E373's rebuilt init, the first batch of its order, first loss {mine:.4f} against E373's logged {theirs:.4f}"


def test_champion():
    """the default path against RECORDED numbers (the known answer of every change of the head, its loss and the stage-1 trainer): a seeded head at two head steps and the stored E373 head (`load_model`; skipped when the run is not
    there) give the forward, the loss and the gradient sums of `CHAMP_HEAD`; 127131 parameters, the champion's output keys; the 20 smoke steps of the champion recipe give `CHAMP`; `_twin_step` on the real cache"""
    keys = ["chord", "cand", "logp", "scale", "types", "types1", "valid"]
    for step in (0, JK.SWITCH + 5):
        torch.manual_seed(0)
        h = JK.JTokenHead()
        h.set_step(step)
        assert JK.count_params(h) == 127131 and sorted(h(_batch(2))) == sorted(keys), "the champion's parameters and output keys"
        torch.manual_seed(0)
        h = JK.JTokenHead()
        h.set_step(step)
        got = _champ_numbers(h)
        assert all(abs(g - w) <= 1e-9 * abs(w) for g, w in zip(got, CHAMP_HEAD[step])), (step, got, CHAMP_HEAD[step])
    note = f"the stored head {E373} SKIPPED (run not there)"
    if os.path.exists(os.path.join(JKT.T.RUNS, E373, "model.pt")):
        m, _ = LN.load_head(E373, "cpu")
        got = _champ_numbers(m.train())
        assert all(abs(g - w) <= 1e-9 * abs(w) for g, w in zip(got, CHAMP_HEAD["e373"])), (got, CHAMP_HEAD["e373"])
        note = f"the stored head {E373} (step {m._step_py - 1}) too"
    steps, real_ml = [], JKT.make_loss

    def recording(*a, **k):                                                    # the loss of every step of the run
        f = real_ml(*a, **k)

        def g(m, b):
            r = f(m, b)
            steps.append(float(r[0]))
            return r
        return g
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        patched(JKT, "make_loss", recording, JKT.train, JKT.parse_args(["--name", "champ", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "0.25"]), runs, 500, 48)
        sd = torch.load(os.path.join(runs, "champ", "model.pt"), map_location="cpu", weights_only=False)["state"]
        row = [json.loads(x) for x in open(os.path.join(runs, "champ", "log.jsonl"))][-1]
    got = (steps[0], steps[-1], row["total"], sum(float(v.double().abs().sum()) for v in sd.values() if v.dtype.is_floating_point))
    assert len(steps) == 20 and all(g == w for g, w in zip(got, CHAMP)), (len(steps), got, CHAMP)
    print(f"champion: the forward, the loss and the gradient sums of a seeded head at head step 0 / {JK.SWITCH + 5} are the recorded ones to 1e-9, {note}; 127131 parameters, the champion's output keys; the 20 smoke steps of "
          f"the champion recipe: first loss {got[0]:.6f}, last {got[1]:.6f}, the log row and the weights as recorded; {_twin_step()}: ok")


TESTS = [test_likelihood, test_none_weight, test_skip_suppress, test_close_gaps, test_fork_gate, test_refine_positions, test_layers_train, test_wiring_logp, test_decode, test_head, test_key_bias, test_writer_free, test_overfit, test_trainer, test_arch_options,
         test_live_inputs, test_head_seed, test_champion]
if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtoken ok")
