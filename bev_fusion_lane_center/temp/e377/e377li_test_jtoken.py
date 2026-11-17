"""E365 tests of the token head on predicted nodes (`jtoken.py`, `train/jtoken_train.py`); CPU only (the GPU is hidden), 2 threads, SYNTHETIC batches that follow the batch dict of the card (2a): no cache needed.

    wsl bash tools/env/_py14.sh training/grid/test_jtoken.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

likelihood (the loss by hand on two frames of 5 tokens, the second with 2 padded tokens, other source counts and no long source: a source with three GT successors (one of them with `ylong` against the token
chord), a source with none, a source whose GT link lies outside the gate, sources without a candidate masked, the per-frame normalisation; mutants: no none slot, a swapped link direction, no gate, a global mean over
the sources, no valid mask; the none slot trains the empty sources), pos_loss (E376: -log sum / -log max over G(i) by hand with duplicates and a fork, one member = the same loss, the max raises one member; textual
mutants), none_weight_near (E377: the near label and the loss by hand on a toy frame (a twin at 1.0 m, exactly 2.0 m, 2.01 m, a heading cosine 0.4, a duplicate in G(i), padding), n_near, unset / equal = the
loss of before, no near token = any W; textual mutants), none_weight_near_trainer (E377: 20 smoke steps bit for bit at W = the none weight, n_near logged, resume refusal, the teacher branch), wiring_logp (a gate and padding mask, sums to 1), decode (the thresholds, the none rule, the kept ends, a padded source and destination, argmax;
mutants: other thresholds / ends / no valid mask), head (shapes, every parameter gets gradient, the gate and the chord classes against an independent distance, order and padding invariance with a padding mutant),
key_bias (the head's own p_none gets gradient from layer 2 on after the switch and not before; the step mirror survives load_state_dict; no gradient reaches the cached inputs), writer_free (the forward reads no GT
key), overfit (8 fixed frames, <= 300 steps, loss below 0.5 x its start, the decode of record reproduces the GT links), trainer (the smoke run on synthetic batches, kill, resume, another recipe refused, no loader
import; a non-smoke run through a fake loader that returns (None, batches) with the ident / K / cache arguments), arch_options (defaults bit for bit, also with --prior none --teacher none), pos_loss_trainer (E376:
--pos-loss sum = the champion's 20 smoke steps bit for bit, max differs), prior (E369: the step-0 head = c-rule in the gate on a hand frame and 3 VAL cache frames, temperature, zero init + scales 1/3, gradient, the type prior inside read_types; mutants lam 0, mu 0, no type prior, the prior outside
read_types), past (the S2 hook), maps (the S3 hook: the plain head bit for bit without a block or maps, `{}` reaches the block, the call order with the S2 hook, the gradient into the block; textual mutants), live_inputs (S4: the default = the forward of before bit for bit, outputs and parameter gradients, no input gradient; live: the same values, a gradient on every float
input; textual mutants; build_head / load_model read it, absent = False), teacher (E370 stage A: targets = the scored c-rule in the gate, B''s types, out-of-gate sources left out, no GT key), init (weights + step buffer, ckpt first, sha256, architecture refused, snapshots with step buffer 20 + n, the hook's eval rows without 'total').
head_seed (--seed 0 --head-seed 2 reads the seed-0 cache, initialises and orders the data as --seed 2, also through a shuffling DataLoader as the loader; --head-seed = --seed is the default run).
"""
import json
import math
import os
import sys
import tempfile
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                      # this test never touches the GPU
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in (".", "..", "../data", "../data/prep", "../train", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
import jtok as JT  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_train as JKT  # noqa: E402
from test_jtok import must_fail  # noqa: E402

torch.set_num_threads(2)
SCRATCH = os.path.join(JT.ROOT, "temp", "e365")
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
    geo = {k: torch.tensor(f[k], dtype=torch.float32) for k in ("pos", "dirv") if k in f}                  # the token geometry of a frame that has one (the E377 near label)
    return dict(cand=cand, chord=chord, y=y, ylong=yl, yscale=ys, ytype=torch.tensor(f["ytype"]), valid=torch.tensor(f["valid"]), **geo)


def _hand(frames=(F1, F2)):
    """-> (out with leaf logits z / zn, batch, frames)"""
    K = frames[0]["K"]
    ts = [_tensors(f) for f in frames]
    st = lambda k: torch.stack([t[k] for t in ts])  # noqa: E731
    g = torch.Generator().manual_seed(1)
    B = len(frames)
    z, zn, ty = torch.randn(B, 2, K, K, generator=g), torch.randn(B, 2, K, generator=g), torch.randn(B, K, 7, generator=g)
    out = dict(z=z.requires_grad_(), zn=zn.requires_grad_(), cand=st("cand"), chord=st("chord"), valid=st("valid"), scale=torch.full((B, K, K, 2), 0.4), types=ty, types1=ty.flip(-1))
    return out, dict(y=st("y"), ylong=st("ylong"), yscale=st("yscale"), ytype=st("ytype"), **{k: st(k) for k in ("pos", "dirv") if k in ts[0]}), frames


def _with_logp(out, cand=None):
    c = out["cand"] if cand is None else cand
    return dict(out, cand=c, logp=JK.wiring_logp(out["z"], out["zn"], c))


def _hand_loss(f, z, zn, t1, t3, w=1.0, pos="sum"):
    """the per-frame numbers by hand: (nll_s, nll_l or None, ce1, ce3); w = the none weight (empty-G sources x w, the none class of the CE x w, normalised by the sum of the weights), a number or a list per token (E377);
    pos = the likelihood of a source with GT successors, -log sum ('sum') or -log max ('max') of p over G(i). G(i) = the GT successors of i among its candidates of the class (the class of a pair = its candidate list, i.e. the token chord)"""
    wn = lambda n: w[n] if isinstance(w, (list, tuple)) else w  # noqa: E731
    res = []
    for c, cs in enumerate((f["short"], f["long"])):
        rows = []
        for i, js in cs.items():
            G = [j for a, j in f["gt"] if a == i and j in js]
            den = lse(*[z[c][i][j] for j in js], zn[c][i])
            zg = [z[c][i][j] for j in G]
            rows.append(-((lse(*zg) if pos == "sum" else max(zg)) - den) if G else -wn(i) * (zn[c][i] - den))
        res.append(sum(rows) / len(rows) if rows else None)
    wt = lambda n: wn(n) if f["ytype"][n] == 0 else 1.0  # noqa: E731
    ce = lambda t: sum(wt(n) * -(t[n][f["ytype"][n]] - lse(*t[n])) for n in range(f["K"]) if f["valid"][n]) / sum(wt(n) for n in range(f["K"]) if f["valid"][n])  # noqa: E731
    return res[0], res[1], ce(t1), ce(t3)


def _check_hand(out, batch, frames, w=1.0):
    """asserts the numbers of `jtoken_loss` against the hand values of every frame, averaged per frame (a class that a frame lacks adds nothing)"""
    z, zn, t1, t3 = (x.detach().tolist() for x in (out["z"], out["zn"], out["types1"], out["types"]))
    per = [_hand_loss(f, z[b], zn[b], t1[b], t3[b], w) for b, f in enumerate(frames)]
    mean = lambda k: (lambda v: sum(v) / len(v))([p[k] for p in per if p[k] is not None])  # noqa: E731
    nll_s, nll_l, ce1, ce3 = (mean(k) for k in range(4))
    loss, info = JK.jtoken_loss(out, batch, none_weight=w)
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
    must_fail(patched, JK, "_frame_mean", lambda x, m, w=None: (x * m).sum() / m.sum().clamp_min(1), _check_hand, out, b, frames)   # a global mean over the sources / tokens instead of per frame
    must_fail(_check_hand, dict(out, valid=torch.ones_like(out["valid"])), b, frames)                                    # no valid mask (the padded tokens of frame 2 in the type CE)
    # sources without a candidate are masked: their none logit changes nothing; the empty sources are trained by the none slot (and push their candidates down)
    out2 = dict(out, zn=out["zn"].detach().clone())
    out2["zn"][0, 0, (2, 4)] = out2["zn"][0, 1, (2, 3, 4)] = out2["zn"][1, 0, (2, 3, 4)] = out2["zn"][1, 1] = 9.0
    assert abs(float(JK.jtoken_loss(_with_logp(out2), b)[0]) - float(loss)) < 1e-6, "a source without a candidate reaches the loss"
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
    assert abs(float(l1) - float(l5)) > 1e-3 and float(JK.jtoken_loss(out, b)[0]) == float(JK.jtoken_loss(out, b, none_weight=1.0)[0]), "W changes the loss, W = 1 is the loss of before"
    assert any(not [j for a, j in f["gt"] if a == i and j in js] for f in frames for i, js in f["short"].items()) and any(f["ytype"][n] == 0 and f["valid"][n] for f in frames for n in range(f["K"])), "the frames have empty-G sources and none tokens"
    orig = JK._source_nll
    must_fail(patched, JK, "_source_nll", lambda lp, G, w, pl="sum": w * orig(lp, G, 1.0, pl), _check_hand, out, b, frames, 0.5)        # the weight on the sources WITH a GT successor too
    must_fail(patched, JK, "_source_nll", lambda lp, G, w, pl="sum": orig(lp, G, 1.0, pl), _check_hand, out, b, frames, 0.5)            # no weight on the none sources
    x, mk, wk = torch.full((1, 2), 0.1), torch.ones(1, 2, dtype=torch.bool), torch.full((1, 2), 0.1)       # weights summing below 1: sum(w x) / sum(w) = 0.2 / 0.2 = 1, not 0.2 / 1
    assert abs(float(JK._frame_mean(x, mk, wk)) - 1.0) < 1e-6 and abs(float(JK._frame_mean(x, mk)) - 0.1) < 1e-6 and float(JK._frame_mean(x, torch.zeros_like(mk), wk)) == 0.0
    orig_fm = JK._frame_mean
    must_fail(patched, JK, "_frame_mean", lambda x, m, w=None: orig_fm(x, m), _check_hand, out, b, frames, 0.5)                                     # weighted terms divided by the token count instead of the sum of the weights
    # the sources with a GT successor are not touched by W: the gradient of the row of frame 0, source 0 (G non-empty) is identical, the none logit of source 1 (G empty) moves
    for w_ in (0.5, 2.0):
        o2 = _with_logp(dict(out, z=out["z"].detach().clone().requires_grad_(), zn=out["zn"].detach().clone().requires_grad_()))
        JK.jtoken_loss(o2, b, none_weight=w_)[0].backward()
        o1 = _with_logp(dict(out, z=out["z"].detach().clone().requires_grad_(), zn=out["zn"].detach().clone().requires_grad_()))
        JK.jtoken_loss(o1, b)[0].backward()
        assert torch.equal(o2["z"].grad[0, 0, 0], o1["z"].grad[0, 0, 0]) and not torch.equal(o2["zn"].grad[0, 0, 1], o1["zn"].grad[0, 0, 1]), "frame 1 source 0 has a GT successor (short): untouched; source 1 (long class: empty) moves"
    print("none_weight: the loss by hand with W = 0.5 on the two hand frames (empty-G sources x W in both classes, the none class of the CE x W, normalised by the sum of the weights), W = 1 = the loss of before, "
          "mutants (the weight on the non-empty sources too, no weight on the none sources, the CE divided by the token count) fail, a source with a GT successor is untouched: ok")


PF = (_frame(6, {0: [1, 2, 3], 1: [2], 3: [4, 5]}, {4: [0, 1], 2: [5]}, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 0), (4, 1)], [(4, 0), (4, 1)], [1] * 6, [True] * 6),
      _frame(6, {0: [1, 2], 2: [3]}, {1: [3, 4]}, [(0, 1), (1, 3)], [(1, 3)], [1] * 6, [True] * 6))
# frame 1, short: 0 -> {1, 2} two tokens near ONE GT successor (duplicates; 3 a candidate outside the group), 1 -> {2} one member, 3 -> {4, 5} a fork (both branches in ONE group); long: 4 -> {0, 1} duplicates, 2 -> none.
# frame 2: one member in every non-empty group, 2 -> none (short)


def _pos_out(frames=PF):
    """the hand frames with leaf logits; in frame 1 the candidate 0 -> 3 (outside the group) is the largest logit of source 0: a max over all candidates differs from the max over the group; and the none logit of
    source 0 (short) lies above every candidate of it: a max that takes the none slot in differs from the max over the group (code critic of the E376 option)"""
    out, b, frames = _hand(frames)
    if frames[0] is PF[0]:
        with torch.no_grad():
            out["z"][0, 0, 0, 3] = out["z"][0, 0, 0, :3].max() + 1.0
            out["zn"][0, 0, 0] = out["z"][0, 0, 0, :4].max() + 1.0
    return _with_logp(out), b, frames


def _check_pos(pos, mod=JK):
    """asserts nll_s / nll_l of `mod.jtoken_loss(none_weight=0.25, pos_loss=pos)` (jtoken or a textual mutant) on PF against the hand values, and that the rest of the loss is that of 'sum' -> (nll_s, nll_l)"""
    out, b, frames = _pos_out()
    z, zn, t1, t3 = (x.detach().tolist() for x in (out["z"], out["zn"], out["types1"], out["types"]))
    per = [_hand_loss(f, z[n], zn[n], t1[n], t3[n], 0.25, pos) for n, f in enumerate(frames)]
    nll_s, nll_l = (sum(p[k] for p in per if p[k] is not None) / sum(p[k] is not None for p in per) for k in range(2))
    (loss, info), (l_sum, i_sum) = mod.jtoken_loss(out, b, none_weight=0.25, pos_loss=pos), JK.jtoken_loss(out, b, none_weight=0.25)
    assert abs(info["nll_s"] - nll_s) < 1e-5 and abs(info["nll_l"] - nll_l) < 1e-5, (pos, info["nll_s"], nll_s, info["nll_l"], nll_l)
    assert abs(float(loss) - float(l_sum) - (info["nll_s"] + info["nll_l"] - i_sum["nll_s"] - i_sum["nll_l"])) < 1e-5, "only the likelihood of the sources with a GT successor may change"
    return nll_s, nll_l


def test_pos_loss():
    """E376 --pos-loss: -log sum (default) and -log max over G(i) by hand on a toy batch with duplicates and a fork in both classes (none weight 0.25: the empty-G sources keep it); one member in every group: max = sum bit
    for bit; under max the gradient raises the winning member only (the other member of a duplicate pair or of a fork is pushed down: the max keeps one branch); textual mutants fail"""
    hand = {pos: _check_pos(pos) for pos in JK.POS_LOSS}
    assert hand["max"][0] > hand["sum"][0] + 1e-3 and hand["max"][1] > hand["sum"][1] + 1e-3, hand
    one, b1, _ = _pos_out(PF[1:])
    assert torch.equal(JK.jtoken_loss(one, b1, none_weight=0.25)[0], JK.jtoken_loss(one, b1, none_weight=0.25, pos_loss="max")[0]), "one member in every G(i): max = sum"
    out, b, _ = _pos_out()
    gz = {pos: torch.autograd.grad(JK.jtoken_loss(_with_logp(out), b, none_weight=0.25, pos_loss=pos)[0], out["z"])[0][0, 0] for pos in JK.POS_LOSS}
    for i, (j1, j2) in ((0, (1, 2)), (3, (4, 5))):                                                          # duplicates, a fork (short class of frame 1)
        win, lose = (j1, j2) if out["z"][0, 0, i, j1] > out["z"][0, 0, i, j2] else (j2, j1)
        assert gz["sum"][i, win] < 0 and gz["sum"][i, lose] < 0 and gz["max"][i, win] < 0 and gz["max"][i, lose] > 0, (i, gz["sum"][i], gz["max"][i])
    must_fail(JK.jtoken_loss, out, b, pos_loss="mean")
    mutants = {"the max over the wrong axis": [("g.amax(-1)", "g.amax(-2)")], "the max over all candidates instead of the group": [("g.amax(-1)", "lp[..., :K].amax(-1)")],
               "the option ignored": [("_source_nll(lp, G, wn, pos_loss)", "_source_nll(lp, G, wn)")], "logsumexp kept under max": [("g.amax(-1)", "torch.logsumexp(g, -1)")],
               "the none slot inside the max": [("g.amax(-1)", "torch.cat([g, lp[..., K:]], -1).amax(-1)")]}
    for edits in mutants.values():
        must_fail(_check_pos, "max", _mutant(*edits))
    print(f"pos_loss: nll_s / nll_l by hand on two frames (duplicates and a fork in the short class, duplicates in the long class, one-member groups, empty-G sources at W 0.25): sum {hand['sum'][0]:.6f} / {hand['sum'][1]:.6f}, "
          f"max {hand['max'][0]:.6f} / {hand['max'][1]:.6f}, the rest of the loss unchanged; one member in every group: max = sum bit for bit; under max the losing member of a duplicate pair and of a fork is pushed down "
          f"(under sum both go up); an unknown mode refused; textual mutants ({', '.join(mutants)}) fail: ok")


# E377 toy frame (metres, heading +y unless said): 0 (type 1) at (0, 0) and 1 (type 2) at (20, 0) are assigned; GT 0 -> 1 and 0 -> 7. NEAR (by hand): 2 a twin 1.0 m beside 0 (in no G), 3 exactly 2.0 m ahead of 0,
# 7 a duplicate 0.5 m beside 1 and in G(0). FAR: 4 2.01 m behind 0, 5 1.0 m beside 0 with a heading cosine 0.4, 6 a far none token, 9 1.0 m beside 6 (near a NONE token only); 8 padding 0.5 m beside 0.
# Empty-G sources: short 1 (assigned: keeps the none weight), 2, 3, 7 (near), 5 (far); long 7 (near), 4, 9 (far). NF_FAR = NF with 2, 3, 7 moved away: no near token
NEAR_HAND = (2, 3, 7)
NF = dict(_frame(10, {0: [1, 2, 7], 1: [9], 2: [3], 3: [4], 5: [6], 7: [1]}, {4: [6], 7: [9], 9: [0]}, [(0, 1), (0, 7)], [], [1, 2] + [0] * 8, [True] * 8 + [False, True]),
          pos=[(0, 0), (20, 0), (1, 0), (0, 2), (0, -2.01), (-1, 0), (-15, 30), (20.5, 0), (0.5, 0), (-15, 31)],
          dirv=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (math.sqrt(1 - 0.4 ** 2), 0.4), (1, 0), (0, 1), (0, 1), (1, 0)])
NF_FAR = dict(NF, pos=[{2: (-30, -30), 3: (30, -30), 7: (30, 30)}.get(n, p) for n, p in enumerate(NF["pos"])])


def _check_near(mod, w, W=0.73):
    """asserts `mod.jtoken_loss(none_weight=w, none_weight_near=W)` (jtoken or a textual mutant) on the batch (NF, NF_FAR) against the hand values: the none weight of every token typed by hand (W on NEAR_HAND of NF,
    w on every other token, also on the assigned empty-G source 1 and on all of NF_FAR) in the empty-G source terms of both classes and in the type CE with its denominator; n_near = (3 + 0) / 2 -> (loss, hand parts)"""
    out, b, frames = _hand((NF, NF_FAR))
    out = _with_logp(out)
    wl = [[W if n in NEAR_HAND else w for n in range(NF["K"])], [w] * NF["K"]]
    z, zn, t1, t3 = (x.detach().tolist() for x in (out["z"], out["zn"], out["types1"], out["types"]))
    per = [_hand_loss(f, z[n], zn[n], t1[n], t3[n], wl[n]) for n, f in enumerate(frames)]
    hand = [sum(p[k] for p in per) / len(per) for k in range(4)]
    loss, info = mod.jtoken_loss(out, b, none_weight=w, none_weight_near=W)
    got = [info[k] for k in ("nll_s", "nll_l", "ce1", "ce3")]
    assert all(abs(g - h) < 1e-5 for g, h in zip(got, hand)) and abs(float(loss) - (hand[0] + hand[1] + 0.5 * (hand[2] + hand[3]))) < 1e-5, (w, W, got, hand)
    assert info["n_near"] == 1.5, info["n_near"]
    return loss, hand


def test_none_weight_near():
    """E377 --none-weight-near: the label `near_none` on the toy frame (NEAR_HAND, padding never near); the loss by hand on (NF, NF_FAR) at W 0.73 with none weight 0.25 and 1.0 (the CE weighting must switch on at 1);
    n_near by hand; unset or equal to the none weight = the loss of before bit for bit (also 1.0 / 1.0); a batch without near tokens has the same loss under any W; textual mutants fail"""
    out, b, _ = _hand((NF, NF_FAR))
    near = JK.near_none(b["pos"], b["dirv"], out["valid"], b["ytype"])
    assert near[0].nonzero().flatten().tolist() == list(NEAR_HAND) and not near[1].any(), near
    den = {w: sum(0.73 if n in NEAR_HAND else (w if NF["ytype"][n] == 0 else 1.0) for n in range(NF["K"]) if NF["valid"][n]) for w in (0.25, 1.0)}
    assert abs(den[0.25] - 5.19) < 1e-12 and abs(den[1.0] - 8.19) < 1e-12, den                               # the CE denominator of NF by hand: 2 assigned + 3 x 0.73 + 4 far none x w (champion 0.25: 3.75)
    (l25, h25), (l1, h1) = _check_near(JK, 0.25), _check_near(JK, 1.0)
    out = _with_logp(out)
    loss = lambda **k: JK.jtoken_loss(out, b, **k)[0]  # noqa: E731
    assert torch.equal(loss(none_weight=0.25), loss(none_weight=0.25, none_weight_near=0.25)) and torch.equal(loss(), loss(none_weight=1.0, none_weight_near=1.0)), "W = the none weight: the loss of before"
    assert abs(float(l25) - float(loss(none_weight=0.25))) > 1e-3 and abs(float(l1) - float(loss())) > 1e-3, "W != the none weight changes the loss (at none weight 1 too: the CE weighting on)"
    o0, b0, _ = _hand((NF_FAR,))
    o0 = _with_logp(o0)
    for w in (0.25, 1.0):
        ref = JK.jtoken_loss(o0, b0, none_weight=w)[0]
        assert all(torch.equal(JK.jtoken_loss(o0, b0, none_weight=w, none_weight_near=wn)[0], ref) for wn in (0.0, 0.25, 0.73, 1.0, 5.0)), f"no near token: W changes the loss at none weight {w}"
    label = 'near = near_none(batch["pos"].to(dev).float(), batch["dirv"].to(dev).float(), valid, yt)'
    mutants = {"the label from the column of y (v1's duplicates)": [(label, "near = valid & (yt == 0) & y.any(1)")], "the label from the row of y": [(label, "near = valid & (yt == 0) & y.any(-1)")],
               "the weight on all none tokens": [(label, "near = valid & (yt == 0)")],
               "on the type term only": [("_source_nll(lp, G, wn, pos_loss)", "_source_nll(lp, G, none_weight, pos_loss)")],
               "on the source term only": [("torch.where(yt == 0, wn, 1.0)", "torch.where(yt == 0, none_weight, 1.0)")],
               "the CE denominator of the scalar weight": [("valid, wt) for t in", "valid, None if wt is None else torch.where(yt == 0, none_weight, 1.0)) for t in")],
               "< instead of <= at 2.0 m": [("d2 <= NEAR_R ** 2", "d2 < NEAR_R ** 2")], "the heading test dropped": [(" & (cos >= NEAR_COS)", "")],
               "the assigned test dropped (near any other valid token)": [("asg[:, None, :]", "(valid[:, None, :] & ~torch.eye(valid.shape[-1], dtype=torch.bool))")],
               "padding counted": [("return valid & (ytype == 0) & (", "return (ytype == 0) & (")],
               "the option ignored": [("wn = torch.where(near, none_weight_near, none_weight)", "wn = none_weight")],
               "the CE weighting off at none weight 1": [("if torch.is_tensor(wn) or none_weight != 1.0 else None", "if none_weight != 1.0 else None")]}
    for edits in mutants.values():
        must_fail(lambda mod: (_check_near(mod, 0.25), _check_near(mod, 1.0)), _mutant(*edits))
    # the label's two remaining edges (code critic of the option): an opposite-heading twin is not near; a TYPED padding token is no anchor (frame: 0 assigned at (0, 0) heading +y, 1 none at (1, 0) heading -y,
    # 2 padding with ytype 1 at (10, 0), 3 none at (11, 0) heading +y next to the padding token only, 4 none at (0, 1) heading +y: the one near token)
    t = lambda x: torch.tensor([x], dtype=torch.float32)  # noqa: E731
    edge = (t([(0, 0), (1, 0), (10, 0), (11, 0), (0, 1)]), t([(0, 1), (0, -1), (0, 1), (0, 1), (0, 1)]), torch.tensor([[True, True, False, True, True]]), torch.tensor([[1, 0, 1, 0, 0]]))

    def check_edge(mod):
        assert mod.near_none(*edge)[0].tolist() == [False, False, False, False, True], mod.near_none(*edge)
    check_edge(JK)
    for edits in ([("(cos >= NEAR_COS)", "(cos.abs() >= NEAR_COS)")], [("asg = valid & (ytype != 0)", "asg = (ytype != 0)")]):
        must_fail(check_edge, _mutant(*edits))
    print(f"none_weight_near: the near label on the toy frame = tokens {list(NEAR_HAND)} (a twin at 1.0 m, exactly 2.0 m, a duplicate in G(0) at 0.5 m; 2.01 m, cos 0.4, a far token, a token near a none token only and padding: not near); "
          f"by hand at W 0.73 / none weight 0.25: nll_s {h25[0]:.6f} nll_l {h25[1]:.6f} ce1 {h25[2]:.6f} ce3 {h25[3]:.6f} (CE denominator of NF {den[0.25]:.2f}, champion 3.75), at W 0.73 / none weight 1.0: nll_s {h1[0]:.6f} "
          f"nll_l {h1[1]:.6f} ce1 {h1[2]:.6f} ce3 {h1[3]:.6f} (denominator {den[1.0]:.2f}), n_near 1.5 (3 + 0 over 2 frames); unset / W = the none weight (0.25, 1.0) = the loss of before bit for bit; no near token: "
          f"the same loss under W 0 - 5; textual mutants ({', '.join(mutants)}) fail; an opposite-heading twin and a token next to a typed padding token are not near (mutants |cos| and anchors without the valid mask fail): ok")


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
    am = [(0, 1), (0, 4), (1, 2), (3, 4)]
    assert _pairs(JK.decode_argmax(a)) == am, "argmax over {none} + keys: one link per source and class (source 1: .27 beats none .22), none wins for 2; the padded token is no end of a link"
    assert _pairs(JK.decode_argmax(bk)) == [(0, 1), (1, 2)]
    open_ = dict(a, valid=torch.ones_like(a["valid"]))                                                         # mutant: no valid mask
    assert rec(open_) == sorted(exp + [(4, 5), (5, 3)]) and _pairs(JK.decode_argmax(open_)) == sorted(am + [(4, 5), (5, 3)])
    for mut in (rec(a, thr=0.2), rec(bk), _pairs(JK.decode_argmax(a)), rec(open_)):
        must_fail(holds, mut == exp)                                                                        # each mutant (threshold, kept ends, argmax, no valid mask) differs from the record
    pr = JK.probs(a)
    assert abs(float(pr["p"][0, 0, 2]) - 0.34) < 1e-6 and abs(float(pr["pnone"][0, 0, 2]) - 0.30) < 1e-6 and bool(pr["keep"][0, 4]) and not bool(pr["keep"][0, 5]) and not bool(JK.probs(bk)["keep"][0, 4])
    print("decode: p >= 0.3 and p >= p(none), both ends kept (p_none < 0.5, 0.5 itself dropped), forks keep both branches, a padded source / destination never forms a link, argmax = one link per source; "
          "mutants (thr 0.2, no end rule, no valid mask, argmax) differ: ok")


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
    loss, info = JK.jtoken_loss(out, b)
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
        JK.jtoken_loss(out, b, type_w=0.0, scale_w=0.0)[0].backward()
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
    JK.jtoken_loss(m1(live), live)[0].backward()
    assert all(live[k].grad is None for k in JK.INPUTS if live[k].is_floating_point()), "a gradient reached a cached input"

    def inputs_cut():                                                        # mutant: the graph is not cut at the inputs
        live2 = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in b.items()}
        JK.jtoken_loss(m1(live2), live2)[0].backward()
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
            loss, info = JK.jtoken_loss(m(b), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = float(loss) if first is None else first
            if float(loss) < 0.25 * first and n >= 100:
                break
        m.eval()
        with torch.no_grad():
            out = m(b)
            last = JK.jtoken_loss(out, b)[0]
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
        real = JK.jtoken_loss

        def dying(out, batch, **k):                                          # a kill at the 12th step (the checkpoint of step 10 exists)
            calls[0] += 1
            if calls[0] == 12:
                raise KeyboardInterrupt("kill")
            return real(out, batch, **k)
        JK.jtoken_loss = dying
        try:
            try:
                JKT.train(JKT.parse_args(argv), runs=runs, ckpt_every=5, **kw)
            except KeyboardInterrupt:
                pass
            else:
                raise RuntimeError("the kill did not happen")
        finally:
            JK.jtoken_loss = real
        rd = os.path.join(runs, "t_s0")
        assert os.path.exists(os.path.join(rd, "ckpt.pt")) and not os.path.exists(os.path.join(rd, ".done"))
        JKT.train(JKT.parse_args(argv), runs=runs, ckpt_every=5, **kw)
        log = open(os.path.join(rd, "train.log"), encoding="utf-8").read()
        assert "resumed from" in log and "at step 10" in log and all(os.path.exists(os.path.join(rd, f)) for f in (".done", "model.pt", "ckpt.pt", "config.json", "log.jsonl", "train.log")), log[-600:]
        m, args = JKT.load_model("t_s0", "cpu", runs)
        assert (args["seed"], args["steps"], args["bs"]) == (0, 20, 2) and int(m.step) == 20 and m._step_py == 20, "the step buffer of the head must follow the run through the kill and the resume"
        rows = [json.loads(r) for r in open(os.path.join(rd, "log.jsonl"))]
        assert all(k in rows[-1] for k in ("nll_s", "nll_l", "ce1", "ce3", "gpu_gb")), rows[-1]
        for other in (["--lr", "5e-4"], ["--ident", "e273_p"], ["--K", "256"], ["--prior", "rule"], ["--teacher", "rule"], ["--pos-loss", "max"]):
            must_fail(JKT.train, JKT.parse_args(argv + other), runs=runs, ckpt_every=5, **kw)                # another recipe in the same folder
        assert "lane_ptok_data" not in sys.modules, "--smoke must not import the loader"
        # a non-smoke run through a fake loader that returns (None, batches) like `lane_ptok_data.loader`
        seen = {}

        def fake_loader(split, seed, bs, workers, **k):
            seen.update(split=split, seed=seed, bs=bs, workers=workers, **k)
            return None, [JK.synthetic_batch(bs, 48, seed=s) for s in range(3)]
        sys.modules["lane_ptok_data"] = types.SimpleNamespace(loader=fake_loader)
        try:
            JKT.train(JKT.parse_args(["--name", "t_real", "--seed", "1", "--steps", "6", "--bs", "2", "--workers", "0", "--ident", "e273_p", "--K", "384", "--cache", "somewhere"]), runs=runs, ckpt_every=3)
        finally:
            del sys.modules["lane_ptok_data"]
        assert seen == dict(split="train", seed=1, bs=2, workers=0, ident="e273_p", K=384, cache="somewhere"), seen
        m2, a2 = JKT.load_model("t_real", "cpu", runs)
        assert a2["steps"] == 6 and a2["ident"] == "e273_p" and m2._step_py == 6 and os.path.exists(os.path.join(runs, "t_real", ".done"))
        sys.modules["lane_ptok_data"] = types.SimpleNamespace(loader=fake_loader)
        seen.clear()
        try:
            JKT.train(JKT.parse_args(["--name", "t_def", "--steps", "2", "--bs", "2", "--workers", "0"]), runs=runs, ckpt_every=3)
        finally:
            del sys.modules["lane_ptok_data"]
        assert seen == dict(split="train", seed=0, bs=2, workers=0, ident="e292_a", K=None), seen              # the defaults: B', the default cache root, K found by the loader
    print("trainer: smoke on synthetic batches (20 steps, kill at 12, resume from the checkpoint of step 10, the step buffer = 20), run-directory files, loss parts logged, another recipe (lr / ident / K) refused, "
          "no loader import; a non-smoke run through a fake loader (ident, K, cache passed, (None, batches) unpacked): ok")


def test_arch_options():
    import numpy as np
    # defaults = the head of the first build, bit for bit
    dflt = JKT.parse_args(["--name", "x", "--none-weight", "1.0"])
    torch.manual_seed(0)
    new = JKT.build_head(vars(dflt))
    torch.manual_seed(0)
    old = JK.JTokenHead()
    torch.manual_seed(0)
    expl = JK.JTokenHead(prior=None, past_block=None, map_block=None, live_inputs=False)                   # the new constructor arguments at their defaults
    for h in (new, expl):
        assert list(h.state_dict()) == list(old.state_dict()) and all(torch.equal(v, old.state_dict()[k]) for k, v in h.state_dict().items()), "default options must build the old head"
    assert JK.count_params(new) == 127131 and JKT.build_head({}).state_dict().keys() == old.state_dict().keys(), "an old run's args (no options) mean the defaults"
    # 20 synthetic steps with the defaults reproduce what the committed code did (recorded from the committed code before the changes, 2 threads: seed 0, bs 2, K 48). THESE CONSTANTS HOLD IN THIS TEST'S PROCESS ONLY (this file
    # sets 2 threads; a fresh process with another thread count / BLAS order gives 2.8011269 for the same run): the portable check is the same-process one below (defaults against the explicit default arguments)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        JKT.train(JKT.parse_args(["--name", "r", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]), runs, 500, 48)
        row = [json.loads(r) for r in open(os.path.join(runs, "r", "log.jsonl"))][-1]
        sd = torch.load(os.path.join(runs, "r", "model.pt"), map_location="cpu", weights_only=False)["state"]
        tot = sum(float(v.double().abs().sum()) for v in sd.values() if v.dtype.is_floating_point)
        assert abs(row["total"] - 2.8015905141830446) < 1e-12 and abs(row["nll_s"] - 1.288527250289917) < 1e-12 and abs(tot / 8250.530362168049 - 1) < 1e-14, (row["total"], tot)
        JKT.train(JKT.parse_args(["--name", "r2", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--d", "64", "--layers", "3", "--heads", "4", "--ffn", "128", "--token-drop", "0.0", "--none-weight", "1.0", "--prior", "none", "--teacher", "none", "--snapshots", ""]), runs, 500, 48)
        rows2 = [json.loads(r) for r in open(os.path.join(runs, "r2", "log.jsonl"))]
        sd2 = torch.load(os.path.join(runs, "r2", "model.pt"), map_location="cpu", weights_only=False)["state"]
        assert [r["total"] for r in rows2] == [json.loads(r)["total"] for r in open(os.path.join(runs, "r", "log.jsonl"))] and list(sd2) == list(sd) and all(torch.equal(v, sd[k]) for k, v in sd2.items()), "explicit default arguments = the defaults, same process, bit for bit"
        # the other sizes: parameter counts, one smoke run, the rebuild by load_model, a recipe that differs is refused, an old run folder resumes with the defaults
        counts = {}
        for name, kw in (("d96l4", ["--d", "96", "--layers", "4"]), ("d96l4f", ["--d", "96", "--layers", "4", "--ffn", "512"]), ("d32l2h2", ["--d", "32", "--layers", "2", "--heads", "2"])):
            JKT.train(JKT.parse_args(["--name", name, "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"] + kw), runs, 500, 48)
            m, a = JKT.load_model(name, "cpu", runs)
            counts[name] = JK.count_params(m)
            assert a["d"] == int(kw[1]) and m.proj.out_features == a["d"] and len(m.layers) == a["layers"] and m.layers[0].ffn[0].out_features == (a["ffn"] or 2 * a["d"]) and counts[name] != 127131
        assert counts["d96l4f"] > counts["d96l4"]
        must_fail(JKT.build_head, dict(d=70, heads=4))                                                     # d must be a multiple of the heads
        must_fail(JKT.build_head, dict(ffn=0))                                                             # 0 is not "the default"
        for bad in (["--d", "70"], ["--d", "96", "--heads", "5"], ["--ffn", "0"], ["--ffn", "-4"], ["--layers", "0"], ["--token-drop", "-0.1"], ["--token-drop", "1.0"]):
            must_fail(JKT.parse_args, ["--name", "x"] + bad)                                               # refused before anything is built or loaded
        assert JKT.parse_args(["--name", "x", "--token-drop", "0.5", "--ffn", "96", "--d", "96", "--heads", "6"]).token_drop == 0.5
        os.makedirs(os.path.join(runs, "old"))
        json.dump(dict(design="jtoken", args=dict(seed=0, cache=None, ident="e292_a", K=None, bs=2, steps=20, lr=1e-3)), open(os.path.join(runs, "old", "config.json"), "w"))     # a config of before the options
        a_old = vars(JKT.parse_args(["--name", "old", "--bs", "2", "--steps", "20"]))
        JKT.check_start(a_old, os.path.join(runs, "old"))
        must_fail(JKT.check_start, dict(a_old, d=96), os.path.join(runs, "old"))
        must_fail(JKT.check_start, dict(a_old, token_drop=0.1), os.path.join(runs, "old"))
        os.makedirs(os.path.join(runs, "big"))                                                             # the reverse: a run of d 96 / layers 4 is not resumed by the default command line
        json.dump(dict(design="jtoken", args=dict(seed=0, cache=None, ident="e292_a", K=None, bs=2, steps=20, lr=1e-3, d=96, layers=4)), open(os.path.join(runs, "big", "config.json"), "w"))
        a_big = vars(JKT.parse_args(["--name", "big", "--bs", "2", "--steps", "20"]))
        must_fail(JKT.check_start, a_big, os.path.join(runs, "big"))
        JKT.check_start(dict(a_big, d=96, layers=4), os.path.join(runs, "big"))
        must_fail(JKT.check_start, dict(a_big, d=96, layers=4, ffn=192), os.path.join(runs, "big"))         # ffn None and the explicit 2 x d are the same head: refused all the same (conservative)
    # token drop: the effect on a batch, the draw per step
    sb = JK.synthetic_batch(2, 48, seed=1)
    keep = torch.ones(2, 48, dtype=torch.bool)
    keep[0, 3] = keep[1, 12] = False
    out = JKT.drop_tokens(sb, keep)
    assert not out["valid"][0, 3] and not out["valid"][1, 12] and out["valid"].sum() == sb["valid"].sum() - 2 and out["ytype"][0, 3] == 0 and out["ytype"][1, 12] == 0
    assert out["y"][0, :, 3].sum() == 0 and out["y"][0, 3, :].sum() == 0 and out["y"][0, 4, 5] == sb["y"][0, 4, 5] and out["y"][1, 12, :].sum() == 0
    assert out["ylong"][1, 11, 12] == 0 and torch.isnan(out["yscale"][0, 3]).all() and torch.equal(out["gt_out"] if "gt_out" in out else sb["ytype"], sb["gt_out"] if "gt_out" in sb else sb["ytype"])
    assert sb["y"][0, 2, 3] == 1 and out["y"][0, 2, 3] == 0, "a GT link into a removed token is lost, the link through it is not bridged"
    assert torch.equal(JKT.drop_tokens(sb, torch.ones(2, 48, dtype=torch.bool))["y"], sb["y"])
    toks = ["a", "b"]
    k1, k1b, k2, k3 = JKT.step_keep(toks, 48, 0.3, 0, 5), JKT.step_keep(toks, 48, 0.3, 0, 5), JKT.step_keep(toks, 48, 0.3, 0, 6), JKT.step_keep(toks, 48, 0.3, 1, 5)
    big = JKT.step_keep(["a"] * 1, 100000, 0.3, 0, 5)
    assert (k1 == k1b).all() and not (k1 == k2).all() and not (k1 == k3).all() and abs(big.mean() - 0.7) < 0.01, "reproducible per (seed, step, token), new draw per step, the share"
    assert JKT.step_keep(["a"], 48, 0.0, 0, 5).all()
    # the same effect as the loader's own token_drop on a cache frame (when the cache exists)
    import lane_ptok_data as LP                                                                              # an ImportError is a failure, not a skip
    from jtok_eval import keep_nodes
    try:
        base, dropped = LP.PTokData("val", 0, "e292_a", None, 0.0), LP.PTokData("val", 0, "e292_a", None, 0.17)
    except (AssertionError, FileNotFoundError) as e:                                                         # only the cache lookup may skip the comparison
        if isinstance(e, AssertionError) and "finished caches match" not in str(e):
            raise
        base = None
    if base is None:
        same = "loader comparison SKIPPED (no cache)"
    else:
        b = LP.collate([base[i] for i in range(2)])
        kp = torch.from_numpy(np.stack([keep_nodes(t, base.K, 0.17, 0) for t in b["token"]]))
        mine = JKT.drop_tokens(b, kp)
        ref = LP.collate([dropped[i] for i in range(2)])
        assert all(torch.equal(mine[k], ref[k]) or (k == "yscale" and torch.equal(mine[k].nan_to_num(-7), ref[k].nan_to_num(-7))) for k in ("valid", "ytype", "y", "ylong", "yscale")), "drop_tokens is the loader's token_drop on the batch"
        same = "equal to the loader's token_drop on 2 cache frames"
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:                                                    # a training run with token drop: finite, differs from the run without
        rows = {}
        for name, extra in (("p0", []), ("p3", ["--token-drop", "0.3"]), ("w5", ["--none-weight", "0.5"])):
            JKT.train(JKT.parse_args(["--name", name, "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"] + extra), runs, 500, 48)
            rows[name] = [json.loads(r) for r in open(os.path.join(runs, name, "log.jsonl"))][-1]["total"]
        assert all(math.isfinite(v) for v in rows.values()) and rows["p3"] != rows["p0"] and rows["w5"] != rows["p0"], rows
    print(f"arch_options: defaults = the old head (127131 parameters, 20 smoke steps reproduce the recorded loss 2.8015905 and the weights, also with --none-weight 1), d96 / layers 4: {counts['d96l4']} parameters (ffn 512: {counts['d96l4f']}), d32 / l2 / h2: {counts['d32l2h2']}, "
          f"rebuilt by load_model, an old config resumes with the defaults and another recipe is refused; token drop: targets of removed tokens go, no re-contraction, a new draw per step, {same}: ok")


def test_pos_loss_trainer():
    """E376: the default is --pos-loss sum: an explicit `--pos-loss sum` equals the champion command line (--none-weight 0.25, no --pos-loss) bit for bit (the loss of each of 20 smoke steps and the final weights, same
    process); this pins the DEFAULT, not that the sum path equals the code of before: that is pinned by the recorded constants of `arch_options` (W 1) and the hand values of `likelihood` / `none_weight` / `pos_loss`
    (W 1, 0.5, 0.25, to 1e-5); --pos-loss max trains finite and differs; the value is in model.pt's args (another value is refused at a resume: `trainer`)"""
    losses, real, now = {}, JK.jtoken_loss, [None]

    def recording(out, batch, **k):                                           # the loss of every step
        r = real(out, batch, **k)
        losses[now[0]].append(float(r[0]))
        return r
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        base = ["--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "0.25"]
        for name, extra in (("champ", []), ("sum", ["--pos-loss", "sum"]), ("max", ["--pos-loss", "max"])):
            now[0], losses[name] = name, []
            patched(JK, "jtoken_loss", recording, JKT.train, JKT.parse_args(["--name", name] + base + extra), runs, 500, 48)
        ck = {n: torch.load(os.path.join(runs, n, "model.pt"), map_location="cpu", weights_only=False) for n in losses}
    assert len(losses["champ"]) == 20 and losses["sum"] == losses["champ"] and all(torch.equal(v, ck["champ"]["state"][k]) for k, v in ck["sum"]["state"].items()), "--pos-loss sum changes the champion recipe"
    assert all(math.isfinite(v) for v in losses["max"]) and losses["max"] != losses["champ"] and losses["max"][0] > losses["champ"][0], losses
    assert (ck["champ"]["args"]["pos_loss"], ck["max"]["args"]["pos_loss"]) == ("sum", "max")
    print(f"pos_loss_trainer: the default is sum: --pos-loss sum = the champion command line --none-weight 0.25 bit for bit (the loss of each of 20 smoke steps, last {losses['champ'][-1]:.6f}, the final weights); --pos-loss max differs "
          f"(first loss {losses['max'][0]:.6f} against {losses['champ'][0]:.6f}, last {losses['max'][-1]:.6f}), finite; pos_loss in model.pt's args: ok")


def test_none_weight_near_trainer():
    """E377 in the trainer: --none-weight-near unset or equal to --none-weight = the run without it bit for bit (the loss of each of 20 smoke steps and the final weights, same process) at the champion's 0.25 and at 1.0;
    0.73 differs (also at none weight 1), `n_near` in every log.jsonl row (>= the 4 duplicates of a synthetic frame), in the first-loss line, the start line names W; the value in model.pt's args, a resume with another value
    refused, W < 0 refused; the teacher branch passes it on (`_rule_frame` under B''s types: n_near 1 = token 10, 1 m ahead of token 5)"""
    losses, real, now = {}, JK.jtoken_loss, [None]

    def recording(out, batch, **k):                                           # the loss of every step
        r = real(out, batch, **k)
        losses[now[0]].append(float(r[0]))
        return r
    nw = lambda w, wn=None: ["--none-weight", w] + ([] if wn is None else ["--none-weight-near", wn])  # noqa: E731
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        base = ["--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]
        for name, extra in (("champ", nw("0.25")), ("champ_n", nw("0.25", "0.25")), ("near", nw("0.25", "0.73")), ("w1", []), ("w1_n", nw("1.0", "1.0")), ("w1_near", nw("1.0", "0.73"))):
            now[0], losses[name] = name, []
            patched(JK, "jtoken_loss", recording, JKT.train, JKT.parse_args(["--name", name] + base + extra), runs, 500, 48)
        ck = {n: torch.load(os.path.join(runs, n, "model.pt"), map_location="cpu", weights_only=False) for n in losses}
        rows = {n: [r for r in map(json.loads, open(os.path.join(runs, n, "log.jsonl"))) if "total" in r] for n in ("near", "champ")}
        log = open(os.path.join(runs, "near", "train.log"), encoding="utf-8").read()
        JKT.check_start(ck["near"]["args"], os.path.join(runs, "near"))
        for other in (None, 0.25, 0.5):
            must_fail(JKT.check_start, dict(ck["near"]["args"], none_weight_near=other), os.path.join(runs, "near"))       # a resume with another value (or without the option) is refused
        must_fail(JKT.check_start, dict(ck["champ"]["args"], none_weight_near=0.73), os.path.join(runs, "champ"))
    same = lambda a, b: losses[a] == losses[b] and all(torch.equal(v, ck[b]["state"][k]) for k, v in ck[a]["state"].items())  # noqa: E731
    assert len(losses["champ"]) == 20 and same("champ_n", "champ") and same("w1_n", "w1"), "W = the none weight changes the run"
    assert losses["near"] != losses["champ"] and losses["w1_near"] != losses["w1"] and all(math.isfinite(v) for v in losses["near"] + losses["w1_near"]), losses
    assert rows["near"] and all(r["n_near"] >= 4 for r in rows["near"]) and not any("n_near" in r for r in rows["champ"]), rows["near"][:2]
    assert "none weight near 0.73 (none tokens within 2.0 m / cos 0.5 of an assigned token)" in log and "near none tokens per frame in this batch (n_near)" in log, log[:800]
    assert (ck["near"]["args"]["none_weight_near"], ck["champ"]["args"]["none_weight_near"]) == (0.73, None)
    must_fail(JKT.parse_args, ["--name", "x", "--none-weight-near", "-0.1"])
    b = _rule_frame()
    gt_free = {k: v for k, v in b.items() if k not in JK.GT_KEYS}
    torch.manual_seed(0)
    m = JK.JTokenHead()
    (l_n, i_n), (l_0, i_0) = (JKT.make_loss(None, none_weight=0.25, teacher="rule", **k)(m, gt_free) for k in (dict(none_weight_near=0.73), {}))
    assert i_n["n_near"] == 1.0 and "n_near" not in i_0 and float(l_n) != float(l_0), (i_n, float(l_n), float(l_0))
    print(f"none_weight_near_trainer: unset / = --none-weight (0.25 and 1.0) = the run without it bit for bit (20 smoke steps, final weights); 0.73 differs (last loss {losses['near'][-1]:.6f} against {losses['champ'][-1]:.6f}; "
          f"at none weight 1: {losses['w1_near'][-1]:.6f} against {losses['w1'][-1]:.6f}); n_near in the log rows ({min(r['n_near'] for r in rows['near']):.1f} - {max(r['n_near'] for r in rows['near']):.1f} per synthetic frame), "
          f"in the first-loss line (the start line names W), not in a run without the option; in model.pt's args, another value refused at a resume, W < 0 refused; the teacher branch: n_near 1 on the hand frame: ok")


# ------------------------------------------------------------------------------------------------------------------------------ c-rule: E369 prior, E370 teacher, S2 hook, --init
RULE_HAND = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 12), (12, 13)}    # c-rule's links of `_rule_frame` by hand; (9, 12) is 0.2 m long: outside the gate (MIN_CHORD 0.3 m)
TYPE_HAND = [6, 1, 1, 2, 2, 1, 1, 4, 1, 1, 0, 1, 1, 1, 0, 0]                                                 # B''s own types of `_rule_frame` (e_p < 0.5 and padding: 0)


def _rule_frame(K=16):
    """one hand frame: lane A tokens 0-5 (x 0, y 0 .. 10 m, heading +y), token 10 at y 11 m with e < 0 (no node), lane B tokens 6-9 (y 22 .. 28 m: 5 -> 6 is lo_near's long link), token 11 at (5, 0) heading +x (a node
    without links), token 12 0.2 m ahead of token 9 (9 -> 12 is c-rule's link, outside the gate; 9 has the short candidate 13), token 13 at y 31 m, padding 14-15; flags: 0 start, 3 split, 4 split + merge, 7 stop, 10 exit (no node)
    -> a batch of one frame with `tokens_np`"""
    pos = np.zeros((1, K, 2), np.float32)
    dirv = np.tile(np.array([0.0, 1.0], np.float32), (1, K, 1))
    for i in range(6):
        pos[0, i] = (0.0, 2.0 * i)
    for n, i in enumerate(range(6, 10)):
        pos[0, i] = (0.0, 22.0 + 2.0 * n)
    pos[0, 10], pos[0, 11], pos[0, 12], pos[0, 13] = (0.0, 11.0), (5.0, 0.0), (0.0, 28.2), (0.0, 31.0)
    dirv[0, 11] = (1.0, 0.0)
    pos[0, 14:] = 30.0
    valid = np.arange(K) < 14
    e = np.where(valid, 2.0, -6.0).astype(np.float32)[None]
    e[0, 10], e[0, 11] = -1.0, 0.5
    fl = np.full((1, K, 8), -3.0, np.float32)
    for i, cols in ((0, (4,)), (3, (2,)), (4, (2, 3)), (7, (0,)), (10, (1,))):
        fl[0, i, list(cols)] = 3.0
    g = torch.Generator().manual_seed(5)
    t = lambda a: torch.from_numpy(a)  # noqa: E731
    z = torch.zeros(1, K)
    return JK.with_tokens_np(dict(tok=torch.randn(1, K, JK.TOK, generator=g), pos=t(pos), dirv=t(dirv), e=t(e), e_raw=t(e), flags=t(fl), s=z, uout=z, uin=z, valid=t(valid[None])))


def _cache_batch(n=3):
    """the first n VAL frames of the predicted-node cache of B' seed 0, or None when it is not there"""
    import lane_ptok_data as LP
    try:
        ds = LP.PTokData("val", 0, "e292_a", None, 0.0)
    except (AssertionError, FileNotFoundError) as e:
        if isinstance(e, AssertionError) and "finished caches match" not in str(e):
            raise
        return None
    return LP.collate([ds[i] for i in range(n)])


def _pairs_of(m):
    return {(int(i), int(j)) for i, j in m.nonzero().tolist()}


def _ctl_rule(t):
    """the links of the arm `rule` of `jtoken_eval.py ctl` (the control as it is scored)"""
    import jtoken_eval as E
    _, _, lk = E.rule_frame(t)
    return set(zip(np.asarray(lk["src"]).tolist(), np.asarray(lk["dst"]).tolist()))


def _delivered(m, b):
    with torch.no_grad():
        out = m(b)
    return out, JK.decode(out)


def test_prior():
    """E369: the step-0 head with the prior delivers c-rule (inside the gate) and keeps the nodes with e >= 0, on the hand frame and on 3 VAL cache frames; the temperature (a rule link at p 0.65 - 0.9); the zero-init list and the
    rest of the head as the plain one; gradient reaches the learned part (every parameter after one step), not lam / mu; the type prior inside `read_types` (the key weights after the switch equal the prior's; the mutant that adds
    it outside fails); mutants: lam 0, mu 0, no type prior"""
    b = _rule_frame()
    assert _pairs_of(JK.rule_matrix(b["pos"], b["dirv"], b["valid"] & (b["e"].sigmoid() >= 0.5))[0]) == RULE_HAND == _ctl_rule(b["tokens_np"][0]), "c-rule by hand"

    def step0_is_rule(m, bb, ref):
        out, dec = _delivered(m, bb)
        gate = out["cand"].any(1)
        for f in range(len(ref)):
            want = {(i, j) for i, j in ref[f] if gate[f, i, j]}
            assert _pairs_of(dec["link"][f]) == want, (f, sorted(_pairs_of(dec["link"][f]) ^ want)[:6])
            assert torch.equal(dec["keep"][f], bb["valid"][f] & (bb["e"][f] > 0)), f
        return out, dec
    torch.manual_seed(0)
    m = JK.JTokenHead(prior="rule").eval()
    out, dec = step0_is_rule(m, b, [RULE_HAND])
    assert not dec["link"][0, 9, 12] and out["cand"][0, :, 9, 12].sum() == 0, "the rule link of 0.2 m is outside the gate"
    p = JK.probs(out)["p"][0]
    pr = torch.stack([p[i, j] for i, j in RULE_HAND - {(9, 12)}])
    assert 0.65 < float(pr.min()) and float(pr.max()) < 0.9, pr                                               # the temperature: not one-hot
    for name, kw in (("lam 0", dict(lam=0.0)), ("mu 0", dict(mu=0.0))):
        torch.manual_seed(0)
        mm = JK.JTokenHead(prior="rule", **kw).eval()
        must_fail(lambda: (step0_is_rule(mm, b, [RULE_HAND]), holds(0.65 < float(torch.stack([JK.probs(mm(b))["p"][0][i, j] for i, j in RULE_HAND - {(9, 12)}]).min()) < 0.9)))
    real_rt = JK.JTokenHead.read_types
    must_fail(patched, JK.JTokenHead, "read_types", lambda self, x, none_prior=None: real_rt(self, x, None), step0_is_rule, m, b, [RULE_HAND])                                   # mutant: no type prior
    cb = _cache_batch(3)
    n_cache = "SKIPPED (no cache)"
    if cb is not None:
        ref = [_ctl_rule(t) for t in cb["tokens_np"]]
        o, d = step0_is_rule(m, cb, ref)
        n_cache = f"{int(d['link'].sum())} links of {sum(len(r) for r in ref)} rule links on 3 VAL cache frames"
        assert torch.equal(d["keep"], cb["valid"] & torch.from_numpy(np.stack([t["e_p"] for t in cb["tokens_np"]]) >= 0.5)), "the kept nodes are c-rule's (e_p >= 0.5)"
    # zero init: the four outputs zero, the rest is the plain head of the same seed; lam / mu are buffers
    torch.manual_seed(0)
    plain = JK.JTokenHead()
    zero = ("types.", "wk.", "wbias.2.", "dest.", "sc_o.weight")
    sp, sm = plain.state_dict(), m.state_dict()
    assert set(sm) - set(sp) == {"lam", "mu"} and float(m.lam) == JK.LAM and float(m.mu) == float(torch.tensor(JK.MU)) and not any(n in ("lam", "mu") for n, _ in m.named_parameters())
    assert all(float(sm[k].abs().max()) == 0.0 for k in sp if k.startswith(zero)) and all(torch.equal(sm[k], sp[k]) for k in sp if not k.startswith(zero + ("sc_o.bias",))), "zero init: only the four outputs (+ the scale head)"
    assert float((out["scale"] - 1 / 3).abs().max()) < 1e-6, "the handle scales start at c-rule's 1/3"
    # p_none = sigmoid(-e) at step 0; the switch keeps the key weights (the prior inside read_types); the mutant that adds the prior outside fails after the switch
    assert float((JK.probs(out)["pt"][0, :, 0] - torch.sigmoid(-b["e"][0]))[b["valid"][0]].abs().max()) < 1e-6

    def same_after_switch(mm):
        mm.set_step(0)
        o0 = mm(b)
        mm.set_step(JK.SWITCH + 1)
        o1 = mm(b)
        mm.set_step(0)
        assert float((o0["logp"].exp() - o1["logp"].exp()).abs().max()) < 1e-5 and float((o0["types"] - o1["types"]).abs().max()) < 1e-4

    def outside(self, x, none_prior=None):                                                                    # mutant: the key weight read before the prior is added
        t = self.types(self.tnorm(x)).float()
        w = 1.0 - t.softmax(-1)[..., 0]
        return (t if none_prior is None else torch.cat([t[..., :1] + none_prior[..., None], t[..., 1:]], -1)), w
    with torch.no_grad():
        same_after_switch(m)
        must_fail(patched, JK.JTokenHead, "read_types", outside, same_after_switch, m)
    # gradient: the zero-init outputs at step 0, every parameter after one step; none into lam / mu or the inputs
    torch.manual_seed(0)
    mt = JK.JTokenHead(prior="rule")
    sb = _batch(3)
    live = {k: (v.clone().requires_grad_(True) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sb.items()}
    JK.jtoken_loss(mt(live), live)[0].backward()
    assert all(float(getattr(mt, n).weight.grad.abs().sum()) > 0 for n in ("types", "wk", "dest")) and float(mt.wbias[2].weight.grad.abs().sum()) > 0, "the zero-init outputs get gradient at step 0"
    assert all(live[k].grad is None for k in JK.INPUTS if live[k].is_floating_point()) and mt.lam.grad is None and not mt.lam.requires_grad, "no gradient into the inputs or the prior"
    opt = torch.optim.Adam(mt.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()
    JK.jtoken_loss(mt(sb), sb)[0].backward()
    dead = [n for n, q in mt.named_parameters() if q.grad is None or float(q.grad.abs().sum()) == 0.0]
    assert not dead and float(mt.lam) == JK.LAM and float(mt.mu) == float(torch.tensor(JK.MU)), dead
    print(f"prior: the step-0 head delivers c-rule inside the gate and keeps e >= 0 (hand frame: 10 links, the 0.2 m rule link outside the gate; {n_cache}), rule links at p {float(pr.min()):.2f} - {float(pr.max()):.2f}, p_none = sigmoid(-e); "
          "mutants lam 0 / mu 0 / no type prior fail; zero init of types / wk / wbias[-1] / dest, handle scales 1/3, the rest = the plain head of the seed, lam / mu buffers; the switch keeps the outputs (the prior added outside read_types fails); "
          "gradient: the zero outputs at step 0, every parameter after one step, none into lam / mu / the inputs: ok")


class _Hook(torch.nn.Module):
    """a dummy hook block (S2 past / S3 maps): x + Linear(the mean of ctx['x']); the identity at zero init and on an empty or None ctx (`{}` = no map)"""

    def __init__(self, zero=True):
        super().__init__()
        self.lin = torch.nn.Linear(JK.D, JK.D)
        self.seen = None
        if zero:
            torch.nn.init.zeros_(self.lin.weight)
            torch.nn.init.zeros_(self.lin.bias)

    def forward(self, x, pos, dirv, valid, ctx):
        self.seen = ctx
        return x + self.lin(ctx["x"].mean(1, keepdim=True)) if ctx else x


def test_past():
    b = _batch(2)
    torch.manual_seed(0)
    base = JK.JTokenHead().eval()
    m = JK.JTokenHead(past_block=_Hook()).eval()
    m.load_state_dict(dict(base.state_dict(), **{f"past_block.{k}": v for k, v in m.past_block.state_dict().items()}))
    with torch.no_grad():
        o0, o1 = base(b), m(b)
        past = dict(x=o1["x"], pos=b["pos"], dirv=b["dirv"], valid=b["valid"], w=o1["w"], e=b["e"])
        o2 = m(b, past)
        assert m.past_block.seen is past, "the past dict reaches the block untouched"
        assert all(torch.equal(o0[k], o1[k]) and torch.equal(o0[k], o2[k]) for k in ("logp", "types", "types1", "scale", "x", "w")), "no past / a zero block: the output of the plain head"
        assert o0["x"].shape == (2, 48, JK.D) and o0["w"].shape == (2, 48) and torch.allclose(o0["w"], 1 - o0["types"].softmax(-1)[..., 0])
        m.past_block = _Hook(zero=False)
        must_fail(holds, torch.equal(m(b, past)["logp"], o0["logp"]))                                          # a live block changes the output: it is called
        m.past_block = _Hook(zero=True)
        m.past_block.lin = torch.nn.Identity()
        m.past_block.forward = lambda x, pos, dirv, valid, past: x                                             # a block that ignores past: the head reads none of its keys
        assert torch.equal(m(b, {"anything": 1})["logp"], o0["logp"])
    print("past: the S2 hook after layer 1 (no past or a zero block = the plain head bit for bit, a live block changes the output, the past dict reaches the block untouched and the head reads none of its keys), out['x'] (B, K, D), out['w'] = 1 - p_none: ok")


def _order(m, b, *a):
    """m(b, *a) without gradient -> the names of its layers and hook blocks in the order they ran; asserts the chain: every call got the x that the call before it returned"""
    log = []
    mods = [(f"layer{i}", l) for i, l in enumerate(m.layers, 1)] + [(n, getattr(m, f"{n}_block")) for n in ("past", "map") if getattr(m, f"{n}_block") is not None]
    hs = [mod.register_forward_hook(lambda _m, inp, out, n=n: log.append((n, inp[0], out))) for n, mod in mods]
    try:
        with torch.no_grad():
            m(b, *a)
    finally:
        for h in hs:
            h.remove()
    assert all(torch.equal(o, i) for (_, _, o), (_, i, _) in zip(log, log[1:])), [n for n, _, _ in log]
    return [n for n, _, _ in log]


def _maps_hook_ok(mod):
    """the S3 hook of `mod.JTokenHead` (jtoken itself or a textual mutant of it)"""
    b = _batch(2)
    torch.manual_seed(0)
    base = mod.JTokenHead().eval()
    m = mod.JTokenHead(map_block=_Hook()).eval()
    m.load_state_dict(dict(base.state_dict(), **{f"map_block.{k}": v for k, v in m.map_block.state_dict().items()}))
    maps, past = dict(x=torch.randn(2, 5, JK.D)), dict(x=torch.randn(2, 48, JK.D))
    with torch.no_grad():
        o0 = base(b)
        same = lambda o: all(torch.equal(o0[k], o[k]) for k in ("logp", "types", "types1", "scale", "x", "w"))  # noqa: E731
        assert same(m(b)) and same(m(b, None, None)) and same(base(b, maps=maps)), "no maps / no block: the plain head bit for bit"
        assert same(m(b, maps=maps)) and m.map_block.seen is maps, "a zero block: the plain head bit for bit; the maps dict reaches the block untouched"
        live = m.map_block = _Hook(zero=False)
        assert not torch.equal(m(b, maps=maps)["logp"], o0["logp"]), "a live block changes the output"
        assert _order(m, b, None, maps) == ["layer1", "layer2", "map", "layer3"], "the map block once, after layer 2, before layer 3"
        empty = {}
        assert same(m(b, maps=empty)) and live.seen is empty and _order(m, b, None, empty) == ["layer1", "layer2", "map", "layer3"], "{} reaches the block (the head does not special-case it)"
        assert same(m(b)) and _order(m, b) == ["layer1", "layer2", "layer3"], "maps None: the block is not called"
        mp = mod.JTokenHead(past_block=_Hook(zero=False), map_block=_Hook(zero=False)).eval()
        assert _order(mp, b, past, maps) == ["layer1", "past", "layer2", "map", "layer3"] and mp.past_block.seen is past and mp.map_block.seen is maps, "both hooks: layer 1 -> past -> layer 2 -> map -> layer 3"
        assert _order(mp, b, past) == ["layer1", "past", "layer2", "layer3"] and _order(mp, b, None, maps) == ["layer1", "layer2", "map", "layer3"]
    mg = mod.JTokenHead(map_block=_Hook())                                                                    # a zero block (the start of a training): the loss reaches its parameters
    mod.jtoken_loss(mg(b, maps=maps), b)[0].backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in mg.map_block.parameters()), "no gradient into the map block"


def _mutant(*edits):
    """jtoken.py with each (old, new) text replaced (old found exactly once), loaded in this process as its own module: a textual mutant"""
    src = open(JK.__file__, encoding="utf-8").read()
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    mod, path = types.ModuleType("jtoken_mutant"), list(sys.path)
    mod.__file__ = JK.__file__
    try:
        exec(compile(src, JK.__file__, "exec"), mod.__dict__)
    finally:
        sys.path[:] = path
    return mod


def test_maps():
    _maps_hook_ok(JK)
    must_fail(JK.JTokenHead, layers=1, map_block=_Hook())                                                     # no layer 2: refused
    cond, call = "self.map_block is not None and maps is not None", "self.map_block(x, pos, dirv, valid, maps)"
    hook = f"if n == 2 and {cond}:"
    mutants = {"after layer 3": [(hook, hook.replace("n == 2", "n == 3"))], "after every layer from 2 on": [(hook, hook.replace("n == 2 and ", ""))],
               "before layer 2": [(hook, hook.replace("n == 2", "n == 0")), ("        types1, w1 = self.read_types(x, tp)\n", f"        if {cond}:\n            x = {call}\n        types1, w1 = self.read_types(x, tp)\n")],
               "{} skipped": [(hook, hook.replace("maps is not None", "maps"))], "maps None passed on": [(hook, hook.replace(" and maps is not None", ""))], "the block gets past": [(call, call.replace("maps)", "past)"))]}
    for edits in mutants.values():
        must_fail(_maps_hook_ok, _mutant(*edits))
    print(f"maps: the S3 hook after layer 2 (no block / maps None / a zero block = the plain head bit for bit, a live block changes the output, the maps dict reaches the block untouched and {{}} is passed on, called once "
          f"between layer 2 and layer 3, with the S2 hook layer 1 -> past -> layer 2 -> map -> layer 3, the loss reaches a zero block, a 1-layer head refused); {len(mutants)} textual mutants fail ({', '.join(mutants)}): ok")


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
    same = lambda r: all(torch.equal(ref[0][k], r[0][k]) for k in ("logp", "types", "types1", "scale", "x", "w")) and all(torch.equal(g, r[1][n]) for n, g in ref[1].items())  # noqa: E731
    assert same(dflt) and all(g is None for g in dflt[2].values()), "the default is not the head of before (outputs, parameter gradients), or a gradient reached an input"
    assert same(live) and all(g is not None and float(g.abs().sum()) > 0 for g in live[2].values()), {k: None if g is None else float(g.abs().sum()) for k, g in live[2].items()}
    return {k: float(g.abs().sum()) for k, g in live[2].items()}


def test_live_inputs():
    """S4 (E372) `JTokenHead(live_inputs=)`: `_live_ok` on jtoken and three textual mutants (the detach kept under live_inputs, the option ignored, the detach dropped under the default); `build_head` reads it from a run's
    args (absent = False, also through `load_model`); `arch` and the state keys do not hold it (an --init of a cached-input head is accepted)"""
    line = "b = {k: batch[k] if self.live_inputs else _const(batch[k]) for k in INPUTS}"
    before = _mutant((line, "b = {k: _const(batch[k]) for k in INPUTS}"))
    g = _live_ok(JK, before)
    mutants = {"the detach kept under live_inputs": [("batch[k] if self.live_inputs", "batch[k].detach() if self.live_inputs")], "the option ignored": [("self.live_inputs = live_inputs", "self.live_inputs = False")],
               "the detach dropped under the default": [("else _const(batch[k])", "else batch[k]")]}
    for edits in mutants.values():
        must_fail(_live_ok, _mutant(*edits), before)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        os.makedirs(os.path.join(runs, "live"))
        torch.save(dict(state=JK.JTokenHead().state_dict(), args=dict(live_inputs=True), steps=0), os.path.join(runs, "live", "model.pt"))
        assert JKT.load_model("live", "cpu", runs)[0].live_inputs is True
    assert JKT.build_head({}).live_inputs is False and JKT.build_head(vars(JKT.parse_args(["--name", "x"]))).live_inputs is False and JKT.arch(dict(live_inputs=True)) == JKT.arch({})
    assert set(JK.JTokenHead(live_inputs=True).state_dict()) == set(JK.JTokenHead().state_dict())
    print(f"live_inputs: the default = the head of before bit for bit (outputs, every parameter gradient; the champion's loss after the switch), no gradient into the inputs; live_inputs=True: the same outputs and parameter "
          f"gradients, a gradient on every float input (sum |grad|: {', '.join(f'{k} {v:.2e}' for k, v in g.items())}); textual mutants ({', '.join(mutants)}) fail; build_head / load_model read it from the args (absent = False), "
          f"not in arch or the state keys: ok")


def test_teacher():
    """E370 stage A: `rule_teacher` = c-rule's links inside the gate, B''s own types, the out-of-gate rule link left out of its class (not 'none'); the stage-A loss reads no GT key and no yscale; on 3 cache frames the targets
    are the links of the scored control `jtoken_eval.rule_frame` inside the gate"""
    b = _rule_frame()
    torch.manual_seed(0)
    m = JK.JTokenHead()
    out = m(b)
    tb = JK.rule_teacher(out, b)
    assert _pairs_of(tb["y"][0]) == RULE_HAND - {(9, 12)} and tb["ytype"][0].tolist() == TYPE_HAND and tb["rule_links"] == 11 and tb["rule_out_gate"] == 1, (sorted(_pairs_of(tb["y"][0])), tb["ytype"][0].tolist())
    assert _pairs_of(tb["ylong"][0]) == {(5, 6)} and tb["yskip"][0].nonzero().tolist() == [[0, 9]], tb["yskip"].nonzero()
    gt_free = {k: v for k, v in b.items() if k not in JK.GT_KEYS}
    loss_fn = JKT.make_loss(None, teacher="rule")
    loss, info = loss_fn(m, gt_free)                                                                          # no y / ylong / yscale / ytype / gt_* / label in the batch
    assert torch.isfinite(loss) and "scale_l1" not in info and info["rule_out_gate"] == 1.0

    def g9(tt):                                                                                               # the gradient on the short-class row of source 9 (its rule link is outside the gate)
        o = m(b)
        o["logp"].retain_grad()
        JK.jtoken_loss(o, tt, scale_w=0.0)[0].backward()
        return float(o["logp"].grad[0, 0, 9].abs().sum())
    assert g9(tb) == 0.0 and g9({k: v for k, v in tb.items() if k != "yskip"}) > 0, "source 9 is left out of the short class, not taught none"
    must_fail(holds, g9({k: v for k, v in tb.items() if k != "yskip"}) == 0.0)
    cb = _cache_batch(3)
    n_cache = "SKIPPED (no cache)"
    if cb is not None:
        with torch.no_grad():
            oc = m(cb)
        tc = JK.rule_teacher(oc, {k: v for k, v in cb.items() if k not in JK.GT_KEYS + ("label",)})
        gate = oc["cand"].any(1)
        for f, t in enumerate(cb["tokens_np"]):
            want = _ctl_rule(t)
            assert _pairs_of(tc["y"][f]) == {(i, j) for i, j in want if gate[f, i, j]}, f
            keep = t["valid"] & (t["e_p"] >= 0.5)
            import lane_tokcache as LC
            yt = np.zeros(len(keep), int)
            yt[keep] = LC.types_of(t["flags_p"][keep] >= 0.5)
            assert np.array_equal(tc["ytype"][f].numpy(), yt), f
        n_cache = f"{int(tc['y'].sum())} target links = the scored c-rule inside the gate on 3 VAL cache frames, {int(tc['rule_out_gate'])} outside"
    print(f"teacher: the targets = c-rule's links inside the gate (hand frame 10 of 11, {n_cache}), B''s own types (split > merge > stop > exit > boundary > lane, e_p < 0.5 -> none), the 0.2 m rule link left out of the short class "
          "(no gradient on its row; the mutant that teaches it none has one); stage A runs on a batch without any GT key, no scale term: ok")


def test_init():
    """E370 stage B: --init loads the weights and the step buffer bit for bit (the step-0 snapshot = the init), ckpt.pt takes precedence on a resume, the recipe holds the path and its sha256, another architecture or another init
    is refused; the stage-B model.pt loads with the plain load_model and has the S1 keys; snapshots model_step<N>.pt are written once and read by load_model"""
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        base = ["--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]
        JKT.train(JKT.parse_args(["--name", "a", "--teacher", "rule"] + base), runs, 500, 48)
        JKT.train(JKT.parse_args(["--name", "big", "--d", "96"] + base), runs, 500, 48)
        JKT.train(JKT.parse_args(["--name", "pri", "--prior", "rule"] + base), runs, 500, 48)
        a_pt = os.path.join(runs, "a", "model.pt")
        ma, aa = JKT.load_model("a", "cpu", runs)
        assert aa["teacher"] == "rule" and ma._step_py == 20 and set(ma.state_dict()) == set(JK.JTokenHead().state_dict()), "the teacher is no part of the head"
        argv = ["--name", "b", "--init", a_pt, "--snapshots", "0,5,10"] + base
        calls, real = [0], JK.jtoken_loss

        def dying(out, batch, **k):                                                                            # a kill at the 7th step (the checkpoint of step 5 exists)
            calls[0] += 1
            if calls[0] == 7:
                raise KeyboardInterrupt("kill")
            return real(out, batch, **k)
        JK.jtoken_loss = dying
        try:
            try:
                JKT.train(JKT.parse_args(argv), runs, 5, 48)
            except KeyboardInterrupt:
                pass
        finally:
            JK.jtoken_loss = real
        s0 = torch.load(os.path.join(runs, "b", "model_step0.pt"), map_location="cpu", weights_only=False)
        assert all(torch.equal(v, ma.state_dict()[k]) for k, v in s0["state"].items()) and s0["steps"] == 0, "the step-0 snapshot is the init, step buffer included"
        t5 = os.path.getmtime(os.path.join(runs, "b", "model_step5.pt"))
        JKT.train(JKT.parse_args(argv), runs, 5, 48)                                                           # the resume: ckpt.pt, not the init
        log = open(os.path.join(runs, "b", "train.log"), encoding="utf-8").read()
        assert "resumed from" in log and "not loaded: resumed from ckpt.pt" in log and os.path.getmtime(os.path.join(runs, "b", "model_step5.pt")) == t5, "a snapshot is never overwritten"
        mb, ab = JKT.load_model("b", "cpu", runs)
        assert ab["init"] == a_pt and ab["init_sha256"] == JKT.file_sha256(a_pt) and mb._step_py == 40 and set(mb.state_dict()) == set(JK.JTokenHead().state_dict()), (mb._step_py, ab["init"])
        for n in (0, 5, 10):
            assert JKT.load_model("b", "cpu", runs, name=f"model_step{n}.pt")[0]._step_py == 20 + n, n                    # the init's 20 forwards + n updates
        evals = {r["step"]: r["val"] for r in map(json.loads, open(os.path.join(runs, "b", "log.jsonl"))) if "total" not in r}
        assert evals == {5: {"snapshot": 5.0}, 10: {"snapshot": 10.0}, 15: {}, 20: {}}, evals                              # the eval rows of the snapshot hook (jtoken_train docstring)
        must_fail(JKT.train, JKT.parse_args(["--name", "b", "--init", os.path.join(runs, "big", "model.pt"), "--snapshots", "0,5,10"] + base), runs, 5, 48)      # another init in the same folder
        for other in ("big", "pri"):
            must_fail(JKT.train, JKT.parse_args(["--name", f"c_{other}", "--init", os.path.join(runs, other, "model.pt")] + base), runs, 5, 48)                 # another architecture (d 96, the prior)
        must_fail(JKT.parse_args, ["--name", "x", "--init", os.path.join(runs, "nothing.pt")])
    print("init: weights and step buffer bit for bit (step-0 snapshot = the stage-A head, step 20 -> 40 after 20 more), ckpt.pt first on a resume, path + sha256 in the recipe, another init / d 96 / the prior refused; the teacher run "
          "and the stage-B run have the S1 state keys and load with load_model; snapshots 0 / 5 / 10 written once, read by load_model with step buffer 20 + n; the eval rows of the hook (no 'total') at 5 / 10 / 15 / 20: ok")


def test_head_seed():
    """--head-seed N: the head's initialisation and the data order use N, --seed keeps choosing the cache (the loader's seed); head_seed in the recipe; --head-seed equal to --seed (and none) = the default run bit for bit"""
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        base = ["--smoke", "--bs", "2", "--workers", "0", "--snapshots", "0"]
        for name, extra in (("s0", ["--seed", "0"]), ("s0h0", ["--seed", "0", "--head-seed", "0"]), ("s0h2", ["--seed", "0", "--head-seed", "2"]), ("s2", ["--seed", "2"])):
            JKT.train(JKT.parse_args(["--name", name] + extra + base), runs, 500, 48)
        st = lambda n, f="model_step0.pt": torch.load(os.path.join(runs, n, f), map_location="cpu", weights_only=False)["state"]  # noqa: E731
        same = lambda x, y: all(torch.equal(v, y[k]) for k, v in x.items())  # noqa: E731
        tot = lambda n: [json.loads(r)["total"] for r in open(os.path.join(runs, n, "log.jsonl")) if "total" in r]  # noqa: E731
        assert same(st("s0h2"), st("s2")) and not same(st("s0h2"), st("s0")), "the head seed initialises the head"
        assert tot("s0h2") == tot("s2") and tot("s0h2") != tot("s0"), "the data (smoke: the synthetic batches) follow the head seed"
        assert same(st("s0h0", "model.pt"), st("s0", "model.pt")) and tot("s0h0") == tot("s0"), "--head-seed = --seed is the default run"
        a = JKT.load_model("s0h2", "cpu", runs)[1]
        assert (a["seed"], a["head_seed"]) == (0, 2) and JKT.load_model("s0", "cpu", runs)[1]["head_seed"] is None
        must_fail(JKT.check_start, dict(a, head_seed=3), os.path.join(runs, "s0h2"))                           # another head seed in the same folder
        # cache-free: a fake loader that returns a SHUFFLING DataLoader over 16 named synthetic batches; --seed chooses the cache, the head seed the batch order
        seen, order, now, loss_real = {}, {}, [None], JK.jtoken_loss

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
                patched(JK, "jtoken_loss", first_batch, JKT.train, JKT.parse_args(["--name", name, "--seed", "0", "--steps", "1", "--bs", "2", "--workers", "0"] + extra), runs, 3)
        finally:
            del sys.modules["lane_ptok_data"]
        assert all(s["seed"] == 0 and s["ident"] == "e292_a" for s in seen.values()) and len(seen) == 3, "--seed chooses the cache of B' seed 0"
        assert order["f0h0"] == order["f0"] != order["f0h2"], f"the batch order follows the head seed, --head-seed 0 = the default: {order}"
        log = open(os.path.join(runs, "s0h2", "train.log"), encoding="utf-8").read()
        assert "cache seed 0, head seed 2" in log and "trainable 127131 parameters" in log and "cache seed 0, head seed 0" in open(os.path.join(runs, "s0", "train.log"), encoding="utf-8").read(), log[:400]
        # the real loader of the cache (when it is there): the cache directory is B' seed 0's for both runs, the frames of the first batch follow the head seed
        real = "SKIPPED (no cache)"
        if _cache_batch(1) is not None:
            import lane_ptok_data as LP
            dirs, first, loss0 = {}, {}, JK.jtoken_loss

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
                patched(LP, "loader", spy_loader, patched, JK, "jtoken_loss", spy_loss, JKT.train,
                        JKT.parse_args(["--name", name, "--seed", "0", "--steps", "1", "--bs", "2", "--workers", "0"] + extra), runs, 500)
            assert all(f"be292_a_0_K" in d for d in dirs.values()) and len(dirs) == 2, dirs
            assert first["c0"] != first["c0h2"], first
            real = f"real loader: both runs read {os.path.basename(os.path.dirname(dirs['c0']))}, first batch {first['c0']} vs {first['c0h2']}"
    print(f"head_seed: --seed 0 --head-seed 2 reads the seed-0 cache, initialises and orders the data as --seed 2 (step-0 weights and the loss curve equal), not as --seed 0; --head-seed 0 = the default run bit for bit; "
          f"head_seed in the recipe (another one refused); the log names both seeds and 127131 parameters; a shuffling DataLoader as the loader: first batch {order['f0']} (default, --head-seed 0) vs {order['f0h2']} (--head-seed 2); {real}: ok")


TESTS = [test_likelihood, test_none_weight, test_pos_loss, test_none_weight_near, test_wiring_logp, test_decode, test_head, test_key_bias, test_writer_free, test_overfit, test_trainer, test_arch_options, test_pos_loss_trainer,
         test_none_weight_near_trainer, test_prior, test_past, test_maps, test_live_inputs, test_teacher, test_init, test_head_seed]

if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtoken ok")
