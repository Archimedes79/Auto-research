"""E365 tests of the token head on predicted nodes (`jtoken.py`, `train/jtoken_train.py`); CPU only (the GPU is hidden), 2 threads, SYNTHETIC batches that follow the batch dict of the card (2a): no cache needed.

    wsl bash tools/env/_py14.sh training/grid/test_jtoken.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

likelihood (the loss by hand on two frames of 5 tokens, the second with 2 padded tokens, other source counts and no long source: a source with three GT successors (one of them with `ylong` against the token
chord), a source with none, a source whose GT link lies outside the gate, sources without a candidate masked, the per-frame normalisation; mutants: no none slot, a swapped link direction, no gate, a global mean over
the sources, no valid mask; the none slot trains the empty sources), pos_loss (E376: -log sum / -log max over G(i) by hand with duplicates and a fork, one member = the same loss, the max raises one member; textual
mutants), none_weight_near (E377: the near label and the loss by hand on a toy frame (a twin at 1.0 m, exactly 2.0 m, 2.01 m, a heading cosine 0.4, a duplicate in G(i), padding), n_near, unset / equal = the
loss of before, no near token = any W; textual mutants), none_weight_near_trainer (E377: 20 smoke steps bit for bit at W = the none weight, n_near logged, resume refusal), wiring_logp (a gate and padding mask, sums to 1), decode (the thresholds, the none rule, the kept ends, a padded source and destination, argmax;
mutants: other thresholds / ends / no valid mask), head (shapes, every parameter gets gradient, the gate and the chord classes against an independent distance, order and padding invariance with a padding mutant),
key_bias (the head's own p_none gets gradient from layer 2 on after the switch and not before; the step mirror survives load_state_dict; no gradient reaches the cached inputs), writer_free (the forward reads no GT
key), overfit (8 fixed frames, <= 300 steps, loss below 0.5 x its start, the decode of record reproduces the GT links), trainer (the smoke run on synthetic batches, kill, resume, another recipe refused, no loader
import; a non-smoke run through a fake loader that returns (None, batches) with the ident / K / cache arguments), arch_options (defaults bit for bit; an old run's removed options (D117) at their default resume and load, another value is refused), pos_loss_trainer (E376:
--pos-loss sum = the champion's 20 smoke steps bit for bit, max differs), past (the S2 hook), maps (the S3 hook: the plain head bit for bit without a block or maps, `{}` reaches the block, the call order with the S2 hook, the gradient into the block; textual mutants), live_inputs (S4: the default = the forward of before bit for bit, outputs and parameter gradients, no input gradient; live: the same values, a gradient on every float
input, the full gradient against float64 central differences; textual mutants; build_head / load_model read it, absent = False).
head_seed (--seed 0 --head-seed 2 reads the seed-0 cache, initialises (the trainer's step-0 head = build_head after seed_all(2)) and orders the data as --seed 2, also through a shuffling DataLoader as the loader; --head-seed = --seed is the default run).
distill (E379 card v2 `distill_loss` = KL_s + KL_l by hand on two toy frames with two stub teachers incl. masked keys and an all-masked row, a teacher equal to the student gives 0 exactly, the gradient, no type term;
textual mutants), distill_trainer (E379 --distill-from / --distill-weight: unset = the champion's 20 smoke steps (recorded constants), W 0 = the champion bit for bit, W 1 moves the weights, step 1 = + W x kd at W 1 / 4,
gt_mean / kd_mean, the teacher key, the start check on the teachers' GT loss over the first 20 batches, the first batch through a persistent-worker DataLoader unchanged by the teachers, the first-batch log, float32
teachers inside autocast, the student initialised as its teacher has KD 0 at step 1, the teachers frozen with their own counters; the refusals; textual mutants of jtoken_train.py; the card's teachers on their VAL
dumps and the twin of E373 (2 loader workers) on the real cache when they are there).
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
        for other in (["--lr", "5e-4"], ["--ident", "e273_p"], ["--K", "256"], ["--pos-loss", "max"]):
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
    # defaults = the head of the first build, bit for bit
    dflt = JKT.parse_args(["--name", "x", "--none-weight", "1.0"])
    torch.manual_seed(0)
    new = JKT.build_head(vars(dflt))
    torch.manual_seed(0)
    old = JK.JTokenHead()
    torch.manual_seed(0)
    expl = JK.JTokenHead(past_block=None, map_block=None, live_inputs=False)                               # the new constructor arguments at their defaults
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
        JKT.train(JKT.parse_args(["--name", "r2", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "1.0"]), runs, 500, 48)
        rows2 = [json.loads(r) for r in open(os.path.join(runs, "r2", "log.jsonl"))]
        sd2 = torch.load(os.path.join(runs, "r2", "model.pt"), map_location="cpu", weights_only=False)["state"]
        assert [r["total"] for r in rows2] == [json.loads(r)["total"] for r in open(os.path.join(runs, "r", "log.jsonl"))] and list(sd2) == list(sd) and all(torch.equal(v, sd[k]) for k, v in sd2.items()), "explicit default arguments = the defaults, same process, bit for bit"
        assert JKT.arch({}) == (64, 4, 3, 128), "the champion's architecture"
        os.makedirs(os.path.join(runs, "old"))
        json.dump(dict(design="jtoken", args=dict(seed=0, cache=None, ident="e292_a", K=None, bs=2, steps=20, lr=1e-3)), open(os.path.join(runs, "old", "config.json"), "w"))     # a config of before the options
        a_old = vars(JKT.parse_args(["--name", "old", "--bs", "2", "--steps", "20"]))
        JKT.check_start(a_old, os.path.join(runs, "old"))
        must_fail(JKT.check_start, dict(a_old, none_weight=0.5), os.path.join(runs, "old"))
        ck = torch.load(os.path.join(runs, "r", "model.pt"), map_location="cpu", weights_only=False)       # D117 (`check_removed`): an old run's removed options at their default resume and load as before, another value is refused
        bad = dict(prior="rule", teacher="rule", init="runs/x/model.pt", token_drop=0.1, d=96, heads=2, layers=4, ffn=512)
        assert set(bad) == set(JKT.REMOVED), "a refusal test for every removed option"
        for name, extra in [("o_dflt", dict(JKT.REMOVED))] + [(f"o_{k}", {k: v}) for k, v in bad.items()]:
            os.makedirs(os.path.join(runs, name))
            json.dump(dict(design="jtoken", args=dict(ck["args"], **extra)), open(os.path.join(runs, name, "config.json"), "w"))
            torch.save(dict(ck, args=dict(ck["args"], **extra)), os.path.join(runs, name, "model.pt"))
            if name == "o_dflt":
                JKT.check_start(dict(ck["args"], name=name), os.path.join(runs, name))
                assert all(torch.equal(v, sd[k]) for k, v in JKT.load_model(name, "cpu", runs)[0].state_dict().items())
            else:
                must_fail(JKT.check_start, dict(ck["args"], name=name), os.path.join(runs, name))
                must_fail(JKT.load_model, name, "cpu", runs)
    print("arch_options: defaults = the old head (127131 parameters, 20 smoke steps reproduce the recorded loss 2.8015905 and the weights, also with --none-weight 1), the architecture is the constant (64, 4, 3, 128); "
          f"an old config resumes with the defaults and another recipe is refused; an old run's removed options (D117: {', '.join(JKT.REMOVED)}) at their default resume and load, another value is refused: ok")


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
    refused, W < 0 refused"""
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
    print(f"none_weight_near_trainer: unset / = --none-weight (0.25 and 1.0) = the run without it bit for bit (20 smoke steps, final weights); 0.73 differs (last loss {losses['near'][-1]:.6f} against {losses['champ'][-1]:.6f}; "
          f"at none weight 1: {losses['w1_near'][-1]:.6f} against {losses['w1'][-1]:.6f}); n_near in the log rows ({min(r['n_near'] for r in rows['near']):.1f} - {max(r['n_near'] for r in rows['near']):.1f} per synthetic frame), "
          f"in the first-loss line (the start line names W), not in a run without the option; in model.pt's args, another value refused at a resume, W < 0 refused: ok")


# ------------------------------------------------------------------------------------------------------------------------------ S2 hook
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


def _mutant(*edits, double=False, of=JK):
    """jtoken.py (of=JKT: jtoken_train.py) with each (old, new) text replaced (old found exactly once), loaded in this process as its own module: a textual mutant; double: then every `.float()` becomes `.double()` (the float64
    copy of `_live_fd_ok`)"""
    src = open(of.__file__, encoding="utf-8").read()
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


def test_maps():
    _maps_hook_ok(JK)
    must_fail(JK.JTokenHead, layers=1, map_block=_Hook())                                                     # no layer 2: refused
    cond, call = "self.map_block is not None and maps is not None", "self.map_block(x, pos, dirv, valid, maps)"
    hook = f"if n == 2 and {cond}:"
    mutants = {"after layer 3": [(hook, hook.replace("n == 2", "n == 3"))], "after every layer from 2 on": [(hook, hook.replace("n == 2 and ", ""))],
               "before layer 2": [(hook, hook.replace("n == 2", "n == 0")), ("        types1, w1 = self.read_types(x)\n", f"        if {cond}:\n            x = {call}\n        types1, w1 = self.read_types(x)\n")],
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
                  "the key weight of layers 2 - 3 cut before the switch": [("lw(w1 if learned else w_in)", "lw(w1 if learned else w_in.detach())")]}
    for edits in fd_mutants.values():
        must_fail(_live_fd_ok, _mutant(*edits, double=True))
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        os.makedirs(os.path.join(runs, "live"))
        torch.save(dict(state=JK.JTokenHead().state_dict(), args=dict(live_inputs=True), steps=0), os.path.join(runs, "live", "model.pt"))
        assert JKT.load_model("live", "cpu", runs)[0].live_inputs is True
    assert JKT.build_head({}).live_inputs is False and JKT.build_head(vars(JKT.parse_args(["--name", "x"]))).live_inputs is False and JKT.arch(dict(live_inputs=True)) == JKT.arch({})
    assert set(JK.JTokenHead(live_inputs=True).state_dict()) == set(JK.JTokenHead().state_dict())
    print(f"live_inputs: the default = the head of before bit for bit (outputs, every parameter gradient; the champion's loss after the switch), no gradient into the inputs; live_inputs=True: the same outputs and parameter "
          f"gradients, a gradient on every float input (sum |grad|: {', '.join(f'{k} {v:.2e}' for k, v in g.items())}); textual mutants ({', '.join(mutants)}) fail; the full live gradient of every float input against "
          f"float64 central differences (h {FD_H:g}, step 0 and after the switch): largest relative error {fd:.1e} <= {FD_TOL:g}, textual mutants ({', '.join(fd_mutants)}) fail; build_head / load_model read it from the args "
          f"(absent = False), not in arch or the state keys: ok")


def test_head_seed():
    """--head-seed N: the head's initialisation and the data order use N, --seed keeps choosing the cache (the loader's seed); head_seed in the recipe; --head-seed equal to --seed (and none) = the default run bit for bit"""
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        base = ["--smoke", "--bs", "2", "--workers", "0"]
        step0, real_bh = {}, JKT.build_head
        for name, extra in (("s0", ["--seed", "0"]), ("s0h0", ["--seed", "0", "--head-seed", "0"]), ("s0h2", ["--seed", "0", "--head-seed", "2"]), ("s2", ["--seed", "2"])):
            def built(args, name=name):                                                                       # the trainer's step-0 head
                m = real_bh(args)
                step0[name] = {k: v.clone() for k, v in m.state_dict().items()}
                return m
            patched(JKT, "build_head", built, JKT.train, JKT.parse_args(["--name", name] + extra + base), runs, 500, 48)
        st = lambda n: torch.load(os.path.join(runs, n, "model.pt"), map_location="cpu", weights_only=False)["state"]  # noqa: E731
        same = lambda x, y: all(torch.equal(v, y[k]) for k, v in x.items())  # noqa: E731
        seeded = lambda hs: (JKT.T.seed_all(hs), JKT.build_head({}).state_dict())[1]  # noqa: E731
        tot = lambda n: [json.loads(r)["total"] for r in open(os.path.join(runs, n, "log.jsonl")) if "total" in r]  # noqa: E731
        assert same(step0["s0h2"], seeded(2)) and same(step0["s2"], seeded(2)) and same(step0["s0"], seeded(0)) and not same(step0["s0h2"], step0["s0"]), "the head seed initialises the head (build_head after seed_all)"
        assert tot("s0h2") == tot("s2") and tot("s0h2") != tot("s0"), "the data (smoke: the synthetic batches) follow the head seed"
        assert same(st("s0h0"), st("s0")) and tot("s0h0") == tot("s0"), "--head-seed = --seed is the default run"
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


# ------------------------------------------------------------------------------------------------------------------------------ E379 ensemble distillation
# two toy frames of 4 tokens. Frame 1 (token 3 padding): short candidates 0 -> {1, 2}, 1 -> {2}, long 2 -> {0} (token 2 has no short candidate, tokens 0 and 1 no long one: valid tokens outside the class rows).
# Frame 2 (all valid): short 0 -> {1}, 1 -> {2, 3}, 3 -> {0}, no long source (other source counts: the per-frame normalisation). Every other key masked. p: per teacher {(class, source): p of its candidates in order, p(none) last}
KD_K = 4
KD_FRAMES = (dict(valid=[True, True, True, False], cand=({0: [1, 2], 1: [2]}, {2: [0]}),
                  p=({(0, 0): [.5, .3, .2], (0, 1): [.6, .4], (1, 2): [.7, .3]}, {(0, 0): [.2, .6, .2], (0, 1): [.1, .9], (1, 2): [.5, .5]})),
             dict(valid=[True] * 4, cand=({0: [1], 1: [2, 3], 3: [0]}, {}), p=({(0, 0): [.4, .6], (0, 1): [.3, .3, .4], (0, 3): [.9, .1]}, {(0, 0): [.8, .2], (0, 1): [.1, .5, .4], (0, 3): [.5, .5]})))


def _kd_frame():
    """-> (the student's out with leaf logits z / zn / types1 / types, the two stub teachers' outs: log of the hand probabilities, -inf at a masked key, p(none) 1 for a valid source without a candidate, the padding
    source's rows ALL -inf (every key and the none slot masked); random types (the KD does not read them))"""
    K, B = KD_K, len(KD_FRAMES)
    valid = torch.tensor([f["valid"] for f in KD_FRAMES])
    cand = torch.zeros(B, 2, K, K, dtype=torch.bool)
    for b, f in enumerate(KD_FRAMES):
        for c, cs in enumerate(f["cand"]):
            for i, js in cs.items():
                cand[b, c, i, js] = True
    g = torch.Generator().manual_seed(3)
    z, zn, t1, t3 = (torch.randn(*s, generator=g).requires_grad_() for s in ((B, 2, K, K), (B, 2, K), (B, K, 7), (B, K, 7)))
    out = dict(z=z, zn=zn, cand=cand, valid=valid, logp=JK.wiring_logp(z, zn, cand), types1=t1, types=t3)
    teachers = []
    for t in range(2):
        P = torch.zeros(B, 2, K, K + 1)
        for b, f in enumerate(KD_FRAMES):
            P[b, :, valid[b], K] = 1.0
            for (c, i), p in f["p"][t].items():
                P[b, c, i, f["cand"][c][i] + [K]] = torch.tensor(p)
        teachers.append(dict(cand=cand, valid=valid, logp=P.log(), types1=torch.randn(B, K, 7, generator=g), types=torch.randn(B, K, 7, generator=g)))
    return out, teachers


def _kd_hand(out):
    """the KD by hand (float64): per class and frame the mean over its sources of sum p_T log(p_T / p_S) over the candidates + none (p_T the mean of the teachers' hand probabilities), then the mean over the frames
    that have a source of the class; the entropy of p_T per source the same way -> dict(kd_s, kd_l, ent_s, ent_l, kd)"""
    z, zn = out["z"].detach().double(), out["zn"].detach().double()
    kl = lambda pt, ps: sum(a * math.log(a / b) for a, b in zip(pt, ps))  # noqa: E731
    soft = lambda v: [math.exp(x - lse(*v)) for x in v]  # noqa: E731
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    h = {}
    for c, name in ((0, "s"), (1, "l")):
        frames = [[([(x + y) / 2 for x, y in zip(f["p"][0][(c, i)], f["p"][1][(c, i)])], soft([float(z[b, c, i, j]) for j in js] + [float(zn[b, c, i])])) for i, js in f["cand"][c].items()]
                  for b, f in enumerate(KD_FRAMES)]
        h["kd_" + name] = mean([mean([kl(pt, ps) for pt, ps in rows]) for rows in frames if rows])
        h["ent_" + name] = mean([mean([-sum(a * math.log(a) for a in pt) for pt, _ in rows]) for rows in frames if rows])
    h["kd"] = h["kd_s"] + h["kd_l"]
    return h


def _check_kd(mod):
    """asserts `mod.distill_loss` (jtoken or a textual mutant) on the toy frame against `_kd_hand` (to 1e-6) -> (kd, info, hand)"""
    out, teachers = _kd_frame()
    kd, info = mod.distill_loss(out, teachers)
    h = _kd_hand(out)
    assert all(abs(info[k] - v) < 1e-6 for k, v in h.items()) and abs(float(kd) - h["kd"]) < 1e-6, (info, h)
    return kd, info, h


def test_distill():
    """E379 (card v2) `distill_loss` on two toy frames by hand (kd = kd_s + kd_l, the entropy; the mean of the teachers' PROBABILITIES, the none slot in, a masked key with p_T 0 and log p_T -inf adds exactly 0, a padding
    row all -inf and a valid source without a candidate of the class: finite, outside the rows; per frame, then over the frames); a teacher equal to the student gives 0 exactly; the gradient: finite, on the gate only, none
    into the type readings (not distilled); no teacher refused (other tokens: `distill_trainer`); textual mutants fail"""
    _, info, h = _check_kd(JK)
    out, teachers = _kd_frame()
    kd = JK.distill_loss(out, teachers)[0]
    kd.backward()
    gz = out["z"].grad
    assert torch.isfinite(kd) and torch.isfinite(gz).all() and float(gz[~out["cand"]].abs().max()) == 0.0 and float(gz[out["cand"]].abs().min()) > 0, gz
    assert out["types1"].grad is None and out["types"].grad is None and set(info) == {"kd", "kd_s", "kd_l", "ent_s", "ent_l"}, "the type readings are not distilled"
    me = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in out.items()}
    k0, i0 = JK.distill_loss(out, [me])
    assert float(k0) == 0.0 and i0["kd_s"] == 0.0 and i0["kd_l"] == 0.0, i0                                 # a teacher equal to the student: 0 exactly
    must_fail(JK.distill_loss, out, [])
    kl = "torch.where(pt > 0, pt * (lpt - lp), 0.0)"
    mutants = {"KL(p_S || p_T)": [(kl, "torch.where(lp.exp() > 0, lp.exp() * (lp - lpt), 0.0)")], "no 0 x log 0 guard": [(kl, "(pt * (lpt - lp))")],
               "a global mean over the sources": [("(_frame_mean(kl[:, c], src[:, c]) for c in range(2))", "((kl[:, c] * src[:, c]).sum() / src[:, c].sum() for c in range(2))")],
               "the mean of the logits": [("torch.logsumexp(torch.stack(lps), 0) - math.log(len(lps))", "torch.stack(lps).mean(0).log_softmax(-1)")],
               "the none slot left out": [('_kl(lpt, out["logp"])', '_kl(lpt[..., :-1], out["logp"][..., :-1])')],
               "the rows of all valid tokens": [("src = source_rows(out)", 'src = out["valid"][:, None].expand(-1, 2, -1)')],
               "the type KL kept": [("kd = kd_s + kd_l\n", 'kd = kd_s + kd_l + 0.5 * _frame_mean(_kl(teacher_logp([o["types"].float().log_softmax(-1) for o in outs_t]), out["types"].float().log_softmax(-1)), out["valid"])\n')]}
    for edits in mutants.values():
        must_fail(_check_kd, _mutant(*edits))
    print(f"distill: by hand on two toy frames (two stub teachers, masked keys at -inf, a padding row all -inf, a source without a candidate of the class, 2 and 3 short sources): kd_s {h['kd_s']:.6f} kd_l {h['kd_l']:.6f} "
          f"kd {h['kd']:.6f}, the teachers' entropy per source {h['ent_s']:.6f} / {h['ent_l']:.6f}; a teacher equal to the student: 0 exactly; the gradient finite, on the gate only, none into the type readings; no teacher "
          f"refused; textual mutants ({', '.join(mutants)}) fail: ok")


CHAMP = (4.647189617156982, 1.6313660144805908, 2.20220827460289, 8251.217476223794)        # the champion recipe (--none-weight 0.25, seed 0, smoke bs 2, K 48): the first and the last of the 20 losses, the log row, the sum of |weights|;
# recorded from the code before --distill-from (commit 866ba519) in this test's process (2 threads; another process may differ in the last digits, as `arch_options`)


KD_KEYS = ("gt", "kd", "kd_s", "kd_l", "ent_s", "ent_l", "gt_mean", "kd_mean")
BASE = ["--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "0.25"]                       # the champion recipe on synthetic batches


def _distill_runs(mod, runs, todo, band=(0.0, 100.0)):
    """trains each (name, extra argv) of `todo` with `mod` (jtoken_train or a textual mutant) on the champion recipe, `mod.TEACHER_GT` = band (the toy teachers' GT loss is not the card's) -> {name: (the (loss, info) of
    every step, the final state, the run's args, the teachers handed to make_loss)}"""
    got = {}
    for name, extra in todo:
        steps, held, real = [], [], mod.make_loss

        def rec(*a, **k):
            held.extend(a[4] if len(a) > 4 else k.get("teachers", ()))
            f = real(*a, **k)

            def g(m, b):
                r = f(m, b)
                steps.append((float(r[0]), dict(r[1])))
                return r
            return g
        patched(mod, "TEACHER_GT", band, patched, mod, "make_loss", rec, mod.train, mod.parse_args(["--name", name] + BASE + extra), runs, 500, 48)
        ck = torch.load(os.path.join(runs, name, "model.pt"), map_location="cpu", weights_only=False)
        got[name] = (steps, ck["state"], ck["args"], held)
    return got


def _check_distill_runs(mod, runs, td, sha, band):
    """`mod` = jtoken_train or a textual mutant; td = the two teachers (a at step 20, b one past the switch), sha = their sha256, band = a TEACHER_GT that holds the teachers' mean GT loss over the first 20 batches and
    not their first-batch loss: unset = the champion of before (CHAMP), W 0 = the champion bit for bit (every loss, the final weights), W 1 differs (finite, KD > 0 at every step, other weights, other step-2 GT parts);
    step 1 = the champion's first loss + W x kd at W 1 and 4; gt_mean / kd_mean = the means of the steps so far; the teacher key (cache seed, ident, K, cache root); the teachers keep their step counters, eval mode, frozen,
    and run in float32 inside an autocast region (bf16 only for the precision probe); the sha256 in the recipe (another one refused at a resume) -> the runs"""
    r = _distill_runs(mod, runs, [("champ", []), ("w0", ["--distill-from", td, "--distill-weight", "0"]), ("w1", ["--distill-from", td]), ("w4", ["--distill-from", td, "--distill-weight", "4"])], band)
    ls = {n: [x[0] for x in v[0]] for n, v in r.items()}
    tot = sum(float(v.double().abs().sum()) for v in r["champ"][1].values() if v.dtype.is_floating_point)
    row = [json.loads(x) for x in open(os.path.join(runs, "champ", "log.jsonl"))][-1]
    assert len(ls["champ"]) == 20 and (ls["champ"][0], ls["champ"][-1], row["total"], tot) == CHAMP, (ls["champ"][0], ls["champ"][-1], row["total"], tot)
    assert ls["w0"] == ls["champ"] and all(torch.equal(v, r["champ"][1][k]) for k, v in r["w0"][1].items()), "W 0: not the champion"
    assert ls["w1"] != ls["champ"] and all(math.isfinite(v) for v in ls["w1"]) and all(i["kd"] > 0 for _, i in r["w1"][0]), "W 1: the option ignored"
    (_, c2), (_, w2) = r["champ"][0][1], r["w1"][0][1]
    assert not all(torch.equal(v, r["champ"][1][k]) for k, v in r["w1"][1].items()) and c2["nll_s"] != w2["nll_s"] and c2["ce1"] != w2["ce1"], "W 1: the KD gradient does not reach the student"
    assert all(abs(ls[n][0] - (ls["champ"][0] + w * r[n][0][0][1]["kd"])) < 1e-5 for n, w in (("w1", 1.0), ("w4", 4.0))), "step 1: not the champion's first loss + W x kd"
    s1 = [i for _, i in r["w1"][0]]
    assert s1[-1]["gt_mean"] == float(np.mean([i["gt"] for i in s1])) and s1[-1]["kd_mean"] == float(np.mean([i["kd"] for i in s1])) and s1[0]["gt_mean"] == s1[0]["gt"], "gt_mean / kd_mean"
    key = dict(distill_from=td, seed=0, ident="e292_a", K=None, cache=None)
    assert len(mod.load_teachers(key, "cpu")) == 2
    for k, v in (("seed", 1), ("ident", "e273_p"), ("K", 384), ("cache", "elsewhere")):
        try:
            mod.load_teachers(dict(key, **{k: v}), "cpu")
        except AssertionError:
            continue
        raise AssertionError(f"a teacher of another {k} accepted")
    tt = r["w1"][3]
    assert [(t._step_py, int(t.step), t.training, any(p.requires_grad for p in t.parameters())) for t in tt] == [(20, 20, False, False), (JK.SWITCH + 7, JK.SWITCH + 7, False, False)], "the teachers moved"
    b, seen, real_dl = JK.synthetic_batch(2, 48, seed=0), [], JK.distill_loss
    with torch.no_grad():
        ref = [t(b)["logp"] for t in tt]

    def spy(o, outs_t):
        seen.append([x["logp"] for x in outs_t])
        return real_dl(o, outs_t)
    with torch.autocast("cpu", dtype=torch.bfloat16):                                                       # the trainer's autocast (bf16 on the GPU; CPU autocast here) around the loss
        patched(JK, "distill_loss", spy, mod.make_loss(None, none_weight=0.25, teachers=tt), JK.JTokenHead(), b)
        o16 = mod.teacher_forward(tt, b, amp=True)
    assert seen and all(all(torch.equal(x, q) for x, q in zip(s, ref)) for s in seen) and not torch.equal(o16[1]["logp"], ref[1]), "the teachers not in float32 inside autocast"
    a1 = r["w1"][2]
    assert a1["distill_sha256"] == sha and r["champ"][2]["distill_sha256"] is None
    mod.check_start(a1, os.path.join(runs, "w1"))
    try:
        mod.check_start(dict(a1, distill_sha256=sha[::-1]), os.path.join(runs, "w1"))
    except AssertionError:
        return r
    raise AssertionError("a resume with other teacher files accepted")


class _Frames(torch.utils.data.Dataset):
    """48 synthetic frames (K 48) with a frame token: the dataset of a real DataLoader with workers (`_check_first_batch`; 24 batches of 2 >= the 20 start batches)"""

    def __len__(self):
        return 48

    def __getitem__(self, i):
        return dict({k: v[0] for k, v in JK.synthetic_batch(1, 48, seed=i).items()}, token=f"frame{i}")


def _collate(items):
    return dict({k: torch.stack([it[k] for it in items]) for k in items[0] if k != "token"}, token=[it["token"] for it in items])


def _check_first_batch(mod, runs, td):
    """the code critic's B-1: a non-smoke run through a real DataLoader (shuffled, 1 PERSISTENT worker, as `lane_ptok_data.loader` with workers) over `_Frames`: the first training batch of the run with the teachers (W 0)
    = that of the run without them = `first_batches(...)[0]` (the start check must not iterate the run's own loader: fit's reset of a persistent iterator draws another shuffle), and the weights after the step are the same
    (the start batches draw from the RNG: the student is seeded again before its build, F6)"""
    first, real, old = {}, mod.make_loss, sys.modules.get("lane_ptok_data")

    def fake_loader(split, seed, bs, workers, **k):
        return None, torch.utils.data.DataLoader(_Frames(), batch_size=bs, shuffle=True, num_workers=1, persistent_workers=True, collate_fn=_collate, drop_last=True)

    def rec(name):
        def mk(*a, **k):
            f = real(*a, **k)

            def g(m, b):
                first.setdefault(name, list(b["token"]))
                return f(m, b)
            return g
        return mk
    sys.modules["lane_ptok_data"] = types.SimpleNamespace(loader=fake_loader)
    try:
        for name, extra in (("p", []), ("d", ["--distill-from", td, "--distill-weight", "0"])):
            patched(mod, "TEACHER_GT", (0.0, 100.0), patched, mod, "make_loss", rec(name), mod.train,
                    mod.parse_args(["--name", name, "--seed", "0", "--steps", "1", "--bs", "2", "--workers", "1", "--none-weight", "0.25"] + extra), runs, 500)
    finally:
        sys.modules.pop("lane_ptok_data")
        if old is not None:
            sys.modules["lane_ptok_data"] = old
    fb = list(mod.first_batches(fake_loader("train", 0, 2, 1)[1], None, 0)[0]["token"])
    st = {n: torch.load(os.path.join(runs, n, "model.pt"), map_location="cpu", weights_only=False)["state"] for n in ("p", "d")}
    assert first["p"] == first["d"] == fb and all(torch.equal(v, st["p"][k]) for k, v in st["d"].items()), (first, fb)    # W 0: the same init and batch give the same step
    return fb


REAL_TEACHERS = ("runs/e373_cb_s0", "runs/e367_b25_s0")                                                             # the teachers of E379 arm d, cache seed 0


def _real_teachers():
    """the card's teachers through `load_teachers` on the first 3 frames of their VAL dumps (the cache of B' seed 0): the step counter is the checkpoint's (> SWITCH) and does not move in a forward, the p_none of the valid
    tokens = the dump's to 1e-3 (as `jtoken_eval.concentration`) -> a note"""
    import pickle
    paths = [os.path.join(JKT.ROOT, p) for p in REAL_TEACHERS]
    if not all(os.path.exists(os.path.join(p, f)) for p in paths for f in ("model.pt", "token_val.pkl")) or _cache_batch(1) is None:
        return "real teachers SKIPPED (the E373 / E367 runs or the VAL cache of B' seed 0 are not there)"
    import lane_ptok_data as LP
    ds = LP.PTokData("val", 0, "e292_a", None, 0.0)
    heads = JKT.load_teachers(dict(distill_from=",".join(REAL_TEACHERS), seed=0, ident="e292_a", K=None), "cpu")
    err, n = 0.0, []
    for p, m in zip(paths, heads):
        n.append(m._step_py)
        assert m._step_py == int(m.step) == int(torch.load(os.path.join(p, "model.pt"), map_location="cpu", weights_only=False)["state"]["step"]) > JK.SWITCH, (p, m._step_py)
        d = pickle.load(open(os.path.join(p, "token_val.pkl"), "rb"))
        idx = d["meta"]["idx"][:3]
        b = LP.collate([ds[int(i)] for i in idx])
        with torch.no_grad():
            pn = JK.probs(m(b))["pt"][..., 0].numpy()
        assert m._step_py == int(m.step) == n[-1] and [f["token"] for f in d["frames"][:3]] == list(b["token"]), "the counter moved, or other frames"
        err = max(err, max(float(np.abs(pn[f] - fr["pnone"])[np.asarray(t["valid"], bool)].max()) for f, (fr, t) in enumerate(zip(d["frames"], b["tokens_np"]))))
    assert err < 1e-3, err
    return f"the card's teachers {', '.join(REAL_TEACHERS)} (step counters {n}) reproduce their VAL dumps' p_none on 3 frames (largest difference {err:.1e})"


def _real_twin():
    """the twin of the card (arm d, seed 0) on the real TRAIN cache, one step on the CPU with the card's teachers at W 0 and 2 loader workers (persistent) (its step-0 head recorded at its build): the start check passes the card's band TEACHER_GT; the student's step-0 state =
    E373 cb's init rebuilt (seed_all(2), build_head of its args); the frame tokens of the first batch in train.log = the first batch of E373's order (seed_all(2), the TRAIN loader), and the first GT loss = E373's logged
    'first loss' (its first batch on its init; the GPU in bf16 there: to 2e-3) -> a note"""
    import hashlib
    import re
    import lane_ptok_data as LP
    e373 = os.path.join(JKT.ROOT, REAL_TEACHERS[0])
    if not all(os.path.exists(os.path.join(JKT.ROOT, p, "model.pt")) for p in REAL_TEACHERS) or not os.path.exists(os.path.join(LP.cache_dir("train", 0, "e292_a"), "meta.json")):
        return "twin SKIPPED (the E373 / E367 runs or the TRAIN cache of B' seed 0 are not there)"
    ea = torch.load(os.path.join(e373, "model.pt"), map_location="cpu", weights_only=False)["args"]
    built, real_bh = [], JKT.build_head

    def rec(args):                                                                                            # every head the trainer builds: the teachers first, the student last
        m = real_bh(args)
        built.append({k: v.clone() for k, v in m.state_dict().items()})
        return m
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        patched(JKT, "build_head", rec, JKT.train, JKT.parse_args(["--name", "twin", "--seed", "0", "--head-seed", "2", "--ident", "e292_a", "--steps", "1", "--bs", "8", "--workers", "2", "--none-weight", "0.25",
                                                                   "--distill-from", ",".join(REAL_TEACHERS), "--distill-weight", "0"]), runs, 500)
        s0 = built[-1]
        log = open(os.path.join(runs, "twin", "train.log"), encoding="utf-8").read()
    JKT.T.seed_all(2)
    init = JKT.build_head(ea).state_dict()
    JKT.T.seed_all(2)
    first = next(iter(LP.loader("train", 0, 8, 0, ident="e292_a")[1]))
    digest = lambda sd: hashlib.sha256(b"".join(v.numpy().tobytes() for v in sd.values())).hexdigest()  # noqa: E731
    assert list(s0) == list(init) and digest(s0) == digest(init), "the student's init is not E373 cb's"
    assert f"frame tokens of this batch {list(first['token'])}" in log and "the band of the mean (0.3, 0.5): ok" in log, log[-3000:]
    mine, theirs = (float(re.search(r"first loss of this process: ([0-9.]+)", t).group(1)) for t in (log, open(os.path.join(e373, "train.log"), encoding="utf-8").read()))
    assert abs(mine - theirs) < 2e-3, (mine, theirs)
    gl = re.search(r"on the first 20 batches (.*?); the band", log).group(1)
    return f"the twin on the real cache: init hash {digest(s0)[:12]} = E373 cb's rebuilt init, first batch {list(first['token'])[:2]}... = E373's order, first GT loss {mine:.4f} against E373's logged {theirs:.4f}; the card's teachers' GT loss on the first 20 batches {gl}"


def test_distill_trainer():
    """E379 (card v2) --distill-from / --distill-weight in the trainer (the champion recipe on synthetic batches; teacher b one past the switch): `_check_distill_runs` with a band that holds the teachers' mean GT loss
    over the first 20 batches and not every first-batch loss; gt / kd / kd_s / kd_l in the info of every step and in the log row; the start line (the teachers' GT loss on the first 20 batches) and the first-loss line (the parts,
    the teachers' GT loss on the first batch, the precision KL, the frame tokens) by hand; the student initialised as its teacher has KD 0 at step 1; refusals (the card's band, another cache seed, a step counter that is
    not the run's steps, W < 0, W without teachers, no model.pt, another W / no teachers at a resume, a changed teacher file at a resume); textual mutants of jtoken_train.py; `_real_teachers`, `_real_twin`"""
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        for name, extra in (("ta", ["--head-seed", "5"]), ("tb", ["--head-seed", "6"]), ("tc", ["--seed", "1"])):
            JKT.train(JKT.parse_args(["--name", name] + BASE + extra), runs, 500, 48)
        pb = os.path.join(runs, "tb", "model.pt")
        ck = torch.load(pb, map_location="cpu", weights_only=False)
        ck["state"]["step"], ck["args"]["steps"] = torch.tensor(JK.SWITCH + 7), JK.SWITCH + 7
        torch.save(ck, pb)                                                                                    # teacher b past the switch (its key weights are its own 1 - p_none), its run's steps the same
        os.makedirs(os.path.join(runs, "te"))
        torch.save(dict(ck, state=dict(ck["state"], step=torch.tensor(JK.SWITCH + 8))), os.path.join(runs, "te", "model.pt"))      # a step counter that is not its run's steps
        td = ",".join(os.path.join(runs, n) for n in ("ta", "tb"))
        sha = [JKT.file_sha256(os.path.join(runs, n, "model.pt")) for n in ("ta", "tb")]
        nb = JKT.START_BATCHES
        with torch.no_grad():                                                                                 # the teachers' GT loss on the first batches of the smoke run (head seed 0) by hand
            hand = [[float(JK.jtoken_loss(m(b), b, none_weight=0.25)[0]) for b in (JK.synthetic_batch(2, 48, seed=s) for s in range(nb))] for m in (JKT.load_model(n, "cpu", runs)[0] for n in ("ta", "tb"))]
        means = [sum(h) / nb for h in hand]
        band = (min(means) - 1e-3, max(means) + 1e-3)
        assert nb == 20 and any(not band[0] <= h[0] <= band[1] for h in hand), (hand, band)                  # the band holds the means and not every first-batch loss
        r = _check_distill_runs(JKT, runs, td, sha, band)
        steps, a1 = r["w1"][0], r["w1"][2]
        rows = [x for x in map(json.loads, open(os.path.join(runs, "w1", "log.jsonl"))) if "total" in x]
        assert rows and all(all(k in x for k in KD_KEYS) for x in rows + [i for _, i in steps]) and (a1["distill_from"], a1["distill_weight"]) == (td, 1.0)
        log, i1 = open(os.path.join(runs, "w1", "train.log"), encoding="utf-8").read(), steps[0][1]
        start = "; ".join(f"{os.path.join(runs, n)}: mean {sum(h) / nb:.4f} (one batch {min(h):.4f} - {max(h):.4f})" for n, h in zip(("ta", "tb"), hand))
        assert f"step counters [20, {JK.SWITCH + 7}]): their own GT loss (the student's loss options) on the first 20 batches {start}" in log, log[:2500]
        assert f"GT loss {i1['gt']:.4f} + KD {i1['kd']:.4f} (x W 1.0) = kd_s {i1['kd_s']:.4f} + kd_l {i1['kd_l']:.4f}" in log and f"the teachers' own GT loss on this batch {hand[0][0]:.4f}, {hand[1][0]:.4f}" in log, log[:2500]
        assert i1["gt"] == r["champ"][0][0][0], "info['gt'] is the GT loss alone (the champion's first loss: the same head and batch)"
        assert "KL(the teachers' mean in float32 || under bf16 autocast) short " in log and "frame tokens of this batch []" in log and f"distill from {td}" in log, log[:2500]
        assert abs(steps[0][0] - r["champ"][0][0][0] - i1["kd"]) < 1e-5, "step 1: the champion's GT loss (the same head and batch) + 1 x KD"
        os.makedirs(os.path.join(runs, "t0"))                                                                # a teacher of 0 steps = the student's step-0 head (seed_all(head seed 0), build_head)
        ka = dict(torch.load(os.path.join(runs, "ta", "model.pt"), map_location="cpu", weights_only=False)["args"], steps=0, head_seed=None)
        JKT.T.seed_all(0)
        torch.save(dict(state=JKT.build_head(ka).state_dict(), args=ka, steps=0), os.path.join(runs, "t0", "model.pt"))
        si = _distill_runs(JKT, runs, [("self", ["--distill-from", os.path.join(runs, "t0")])])["self"][0]
        assert si[0][1]["kd"] == 0.0 and all(i["kd"] > 0 for _, i in si[1:]), [i["kd"] for _, i in si[:3]]      # the student = its teacher on the same batch: 0 exactly
        # refusals
        must_fail(JKT.train, JKT.parse_args(["--name", "c0"] + BASE + ["--distill-from", td]), runs, 500, 48)  # the card's band (the toy teachers are not the card's)
        must_fail(JKT.train, JKT.parse_args(["--name", "c1"] + BASE + ["--distill-from", f"{td},{os.path.join(runs, 'tc')}"]), runs, 500, 48)
        assert not os.path.exists(os.path.join(runs, "c0")) and not os.path.exists(os.path.join(runs, "c1")), "a refused start wrote the run folder"
        must_fail(JKT.load_teachers, dict(distill_from=os.path.join(runs, "te"), seed=0, ident="e292_a", K=None), "cpu")
        for k, v in (("distill_weight", 0.5), ("distill_from", None)):
            must_fail(JKT.check_start, dict(a1, **{k: v}), os.path.join(runs, "w1"))
        for bad in (["--distill-from", td, "--distill-weight", "-0.5"], ["--distill-weight", "0.5"], ["--distill-from", os.path.join(runs, "nothing")], ["--distill-from", ""]):
            must_fail(JKT.parse_args, ["--name", "x"] + bad)

        class Other(torch.nn.Module):                                                                         # a teacher on other tokens (another gate)
            def __init__(self, t):
                super().__init__()
                self.t = t

            def forward(self, bb):
                o = self.t(bb)
                return dict(o, cand=~o["cand"])
        must_fail(JKT.make_loss(None, none_weight=0.25, teachers=[r["w1"][3][0], Other(r["w1"][3][1])]), JK.JTokenHead(), JK.synthetic_batch(2, 48, seed=0))
        argv = JKT.parse_args(["--name", "w1"] + BASE + ["--distill-from", td])
        patched(JKT, "TEACHER_GT", band, JKT.train, argv, runs, 500, 48)                                     # the same teacher files: resumed
        torch.save(dict(ck, steps=21), pb)
        must_fail(patched, JKT, "TEACHER_GT", band, JKT.train, argv, runs, 500, 48)                          # a changed teacher file: refused
        torch.save(ck, pb)
        fb = _check_first_batch(JKT, runs, td)
        mutants = {"the option ignored": [("teachers = load_teachers(args, dev) if a.distill_from else []", "teachers = []")], "W ignored": [("loss = loss + distill_weight * kd", "loss = loss + kd")],
                   "W only on / off": [("loss = loss + distill_weight * kd", "loss = loss + (kd if distill_weight else 0.0 * kd)")],
                   "W on the sum": [("loss = loss + distill_weight * kd", "loss = distill_weight * (loss + kd)")],
                   "the KD gradient cut": [("loss = loss + distill_weight * kd", "loss = loss + distill_weight * kd.detach()")],
                   "the KD on a detached student": [("JK.distill_loss(out, outs_t)", "JK.distill_loss({k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}, outs_t)")],
                   "the teachers in train mode": [("heads.append(m.requires_grad_(False))", "heads.append(m.train().requires_grad_(False))")],
                   "the autocast-off of the teachers removed": [('with torch.no_grad(), torch.autocast(b["valid"].device.type, dtype=torch.bfloat16, enabled=amp):', "with torch.no_grad():")],
                   "the start check on the first batch only": [("for v in tl.mean(1))", "for v in tl[:, 0])")],
                   "the cache root out of the teacher key": [('x.get("K"), x.get("cache"))', 'x.get("K"))')],
                   "the sha256 out of the recipe": [('"distill_from", "distill_weight", "distill_sha256")', '"distill_from", "distill_weight")')]}
        for edits in mutants.values():
            with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
                must_fail(_check_distill_runs, _mutant(*edits, of=JKT), tmp, td, sha, band)
        own = {"the start batches from the run's own loader": [("it = iter(torch.utils.data.DataLoader(", "it = iter(loader) if True else iter(torch.utils.data.DataLoader(")],                   # smoke runs cannot see these two
               "no seed before the student's build": [("    T.seed_all(hs)                                                          # the student's", "    pass                                                                    # the student's")]}
        for edits in own.values():
            with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
                must_fail(_check_first_batch, _mutant(*edits, of=JKT), tmp, td)
        mutants.update(own)
    real, twin = _real_teachers(), _real_twin()
    print(f"distill_trainer: unset = the champion's 20 smoke steps (recorded: first {CHAMP[0]:.6f}, last {CHAMP[1]:.6f}), W 0 with two teachers = the champion bit for bit, W 1 differs (KD at step 1 {i1['kd']:.4f}: "
          f"kd_s {i1['kd_s']:.4f} kd_l {i1['kd_l']:.4f}; other final weights, other step-2 GT parts; step 1 = the champion's + W x kd at W 1 and 4), gt / kd / kd_s / kd_l / gt_mean / kd_mean in every step's info "
          f"and the log row; through a shuffled DataLoader with a persistent worker the first batch with teachers = without = first_batches ({fb}); the start check on the mean of the first 20 batches (toy "
          f"teachers {means[0]:.4f} / {means[1]:.4f}, first batch {hand[0][0]:.4f} / {hand[1][0]:.4f}) and the first-batch line (= by hand, the precision KL, the frame tokens) logged; the teachers keep their counters (20, "
          f"{JK.SWITCH + 7}: past the switch), eval, frozen, float32 inside the loss under bf16 autocast; the student initialised as its teacher: KD 0 at step 1; the sha256 in the recipe; refusals (the card's band, another "
          f"cache seed / ident / K / cache root, a step counter off its run's steps, a teacher on other tokens, W < 0, W without teachers, no model.pt, another W / sha / no teachers at a resume, a changed teacher file); textual mutants "
          f"({', '.join(mutants)}) fail; {real}; {twin}: ok")


TESTS = [test_likelihood, test_none_weight, test_pos_loss, test_none_weight_near, test_wiring_logp, test_decode, test_head, test_key_bias, test_writer_free, test_overfit, test_trainer, test_arch_options, test_pos_loss_trainer,
         test_none_weight_near_trainer, test_past, test_maps, test_live_inputs, test_head_seed, test_distill, test_distill_trainer]
if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtoken ok")
