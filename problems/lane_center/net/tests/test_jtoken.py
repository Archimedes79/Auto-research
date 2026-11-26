"""E365 tests of the token head on predicted nodes (`jtoken.py`, `train/jtoken_train.py`); CPU only (the GPU is hidden), 2 threads, SYNTHETIC batches that follow the batch dict of the card (2a): no cache needed.

    wsl bash tools/env/_py14.sh net/tests/test_jtoken.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

likelihood (the loss by hand on two frames of 5 tokens, the second with 2 padded tokens, other source counts and no long source: a source with three GT successors (one of them with `ylong` against the token
chord), a source with none, a source whose GT link lies outside the gate, sources without a candidate masked, the per-frame normalisation; mutants: no none slot, a swapped link direction, no gate, a global mean over
the sources, no valid mask; the none slot trains the empty sources), pos_loss (E376: -log sum / -log max over G(i) by hand with duplicates and a fork, one member = the same loss, the max raises one member; textual
mutants), wiring_logp (a gate and padding mask, sums to 1), decode (the thresholds, the none rule, the kept ends, a padded source and destination, argmax;
mutants: other thresholds / ends / no valid mask), head (shapes, every parameter gets gradient, the gate and the chord classes against an independent distance, order and padding invariance with a padding mutant),
key_bias (the head's own p_none gets gradient from layer 2 on after the switch and not before; the step mirror survives load_state_dict; no gradient reaches the cached inputs), writer_free (the forward reads no GT
key), overfit (8 fixed frames, <= 300 steps, loss below 0.5 x its start, the decode of record reproduces the GT links), trainer (the smoke run on synthetic batches, kill, resume, another recipe refused, no loader
import; a non-smoke run through a fake loader that returns (None, batches) with the ident / K / cache arguments), arch_options (defaults bit for bit; an old run's removed options (D117) at their default resume and load, another value is refused), pos_loss_trainer (E376:
--pos-loss sum = the champion's 20 smoke steps bit for bit, max differs), live_inputs (S4: the default = the forward of before bit for bit, outputs and parameter gradients, no input gradient; live: the same values, a gradient on every float
input, the full gradient against float64 central differences; textual mutants; build_head / load_model read it, absent = False).
head_seed (--seed 0 --head-seed 2 reads the seed-0 cache, initialises (the trainer's step-0 head = build_head after seed_all(2)) and orders the data as --seed 2, also through a shuffling DataLoader as the loader; --head-seed = --seed is the default run).
distill (E379 card v2 `distill_loss` = KL_s + KL_l by hand on two toy frames with two stub teachers incl. masked keys and an all-masked row, a teacher equal to the student gives 0 exactly, the gradient, no type term;
textual mutants), distill_trainer (E379 --distill-from / --distill-weight: unset = the champion's 20 smoke steps (recorded constants), W 0 = the champion bit for bit, W 1 moves the weights, step 1 = + W x kd at W 1 / 4,
gt_mean / kd_mean, the teacher key, the start check on the teachers' GT loss over the first 20 batches, the first batch through a persistent-worker DataLoader unchanged by the teachers, the first-batch log, float32
teachers inside autocast, the student initialised as its teacher has KD 0 at step 1, the teachers frozen with their own counters; the refusals; textual mutants of jtoken_train.py; the card's teachers on their VAL
dumps and the twin of E373 (2 loader workers) on the real cache when they are there).
noise (E381 card v2 `--noise-pos` / `--noise-e`: `add_noise` on valid tokens only, e / e_raw one draw, the labels the same objects, the global stream untouched, keyed on (seed, step), the logged statistics; textual mutants;
0 = the champion's 20 smoke steps bit for bit; the noise in the args and the log, the clean first loss; another noise at a resume refused; kill / resume repeats the draws by the restored step counter; no noise in the read-outs).
E395 (the position head and its curve loss): pos_head (off = the champion's recorded numbers, no posh / dpos; on: zero start, the other parameters as without it, no GT key in the forward, gradients reach posh and the trunk and never pos / dirv; mutants),
curve_loss (Catmull-Rom by hand and against a basis-matrix numpy form; the node loss and the curve loss by hand, the chord rule, a frame of left-out links adds nothing; numpy reference on random batches + gradcheck; mutants), pos_loss_weights (curve_w =
node_w = 0 is the call without them; known counts on the geometry toy; info keys; refusals), dump_dpos (`jtoken_eval.batch_dump`), pos_head_trainer (--pos-head / --curve-w / --node-w: refusals, smoke runs, recipe, loader geom, stage 2 refuses).
"""
import copy
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
for sub_ in ("../model", "../prep", "../loss", "../../training", "../../training/data", "../../training/data/prep", "../../training/train", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
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


def _hand_loss(f, z, zn, t1, t3, w=1.0, pos="sum"):
    """the per-frame numbers by hand: (nll_s, nll_l or None, ce1, ce3); w = the none weight (empty-G sources x w, the none class of the CE x w, normalised by the sum of the weights);
    pos = the likelihood of a source with GT successors, -log sum ('sum') or -log max ('max') of p over G(i). G(i) = the GT successors of i among its candidates of the class (the class of a pair = its candidate list, i.e. the token chord)"""
    res = []
    for c, cs in enumerate((f["short"], f["long"])):
        rows = []
        for i, js in cs.items():
            G = [j for a, j in f["gt"] if a == i and j in js]
            den = lse(*[z[c][i][j] for j in js], zn[c][i])
            zg = [z[c][i][j] for j in G]
            rows.append(-((lse(*zg) if pos == "sum" else max(zg)) - den) if G else -w * (zn[c][i] - den))
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
    must_fail(patched, LJK, "_source_nll", lambda lp, G, w, pl="sum": w * orig(lp, G, 1.0, pl), _check_hand, out, b, frames, 0.5)        # the weight on the sources WITH a GT successor too
    must_fail(patched, LJK, "_source_nll", lambda lp, G, w, pl="sum": orig(lp, G, 1.0, pl), _check_hand, out, b, frames, 0.5)            # no weight on the none sources
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
    (loss, info), (l_sum, i_sum) = mod.jtoken_loss(out, b, none_weight=0.25, pos_loss=pos), LJK.jtoken_loss(out, b, none_weight=0.25)
    assert abs(info["nll_s"] - nll_s) < 1e-5 and abs(info["nll_l"] - nll_l) < 1e-5, (pos, info["nll_s"], nll_s, info["nll_l"], nll_l)
    assert abs(float(loss) - float(l_sum) - (info["nll_s"] + info["nll_l"] - i_sum["nll_s"] - i_sum["nll_l"])) < 1e-5, "only the likelihood of the sources with a GT successor may change"
    return nll_s, nll_l


def test_pos_loss():
    """E376 --pos-loss: -log sum (default) and -log max over G(i) by hand on a toy batch with duplicates and a fork in both classes (none weight 0.25: the empty-G sources keep it); one member in every group: max = sum bit
    for bit; under max the gradient raises the winning member only (the other member of a duplicate pair or of a fork is pushed down: the max keeps one branch); textual mutants fail"""
    hand = {pos: _check_pos(pos) for pos in LJK.POS_LOSS}
    assert hand["max"][0] > hand["sum"][0] + 1e-3 and hand["max"][1] > hand["sum"][1] + 1e-3, hand
    one, b1, _ = _pos_out(PF[1:])
    assert torch.equal(LJK.jtoken_loss(one, b1, none_weight=0.25)[0], LJK.jtoken_loss(one, b1, none_weight=0.25, pos_loss="max")[0]), "one member in every G(i): max = sum"
    out, b, _ = _pos_out()
    gz = {pos: torch.autograd.grad(LJK.jtoken_loss(_with_logp(out), b, none_weight=0.25, pos_loss=pos)[0], out["z"])[0][0, 0] for pos in LJK.POS_LOSS}
    for i, (j1, j2) in ((0, (1, 2)), (3, (4, 5))):                                                          # duplicates, a fork (short class of frame 1)
        win, lose = (j1, j2) if out["z"][0, 0, i, j1] > out["z"][0, 0, i, j2] else (j2, j1)
        assert gz["sum"][i, win] < 0 and gz["sum"][i, lose] < 0 and gz["max"][i, win] < 0 and gz["max"][i, lose] > 0, (i, gz["sum"][i], gz["max"][i])
    must_fail(LJK.jtoken_loss, out, b, pos_loss="mean")
    mutants = {"the max over the wrong axis": [("g.amax(-1)", "g.amax(-2)")], "the max over all candidates instead of the group": [("g.amax(-1)", "lp[..., :K].amax(-1)")],
               "the option ignored": [("_source_nll(lp, G, none_weight, pos_loss)", "_source_nll(lp, G, none_weight)")], "logsumexp kept under max": [("g.amax(-1)", "torch.logsumexp(g, -1)")],
               "the none slot inside the max": [("g.amax(-1)", "torch.cat([g, lp[..., K:]], -1).amax(-1)")]}
    for edits in mutants.values():
        must_fail(_check_pos, "max", _mutant(*edits))
    print(f"pos_loss: nll_s / nll_l by hand on two frames (duplicates and a fork in the short class, duplicates in the long class, one-member groups, empty-G sources at W 0.25): sum {hand['sum'][0]:.6f} / {hand['sum'][1]:.6f}, "
          f"max {hand['max'][0]:.6f} / {hand['max'][1]:.6f}, the rest of the loss unchanged; one member in every group: max = sum bit for bit; under max the losing member of a duplicate pair and of a fork is pushed down "
          f"(under sum both go up); an unknown mode refused; textual mutants ({', '.join(mutants)}) fail: ok")


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
    expl = JK.JTokenHead(live_inputs=False)                                                                 # the constructor argument at its default
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
        bad = dict(prior="rule", teacher="rule", init="runs/x/model.pt", token_drop=0.1, none_weight_near=0.73, d=96, heads=2, layers=4, ffn=512)
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
    losses, real, now = {}, LJK.jtoken_loss, [None]

    def recording(out, batch, **k):                                           # the loss of every step
        r = real(out, batch, **k)
        losses[now[0]].append(float(r[0]))
        return r
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        base = ["--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "0.25"]
        for name, extra in (("champ", []), ("sum", ["--pos-loss", "sum"]), ("max", ["--pos-loss", "max"])):
            now[0], losses[name] = name, []
            patched(LJK, "jtoken_loss", recording, JKT.train, JKT.parse_args(["--name", name] + base + extra), runs, 500, 48)
        ck = {n: torch.load(os.path.join(runs, n, "model.pt"), map_location="cpu", weights_only=False) for n in losses}
    assert len(losses["champ"]) == 20 and losses["sum"] == losses["champ"] and all(torch.equal(v, ck["champ"]["state"][k]) for k, v in ck["sum"]["state"].items()), "--pos-loss sum changes the champion recipe"
    assert all(math.isfinite(v) for v in losses["max"]) and losses["max"] != losses["champ"] and losses["max"][0] > losses["champ"][0], losses
    assert (ck["champ"]["args"]["pos_loss"], ck["max"]["args"]["pos_loss"]) == ("sum", "max")
    print(f"pos_loss_trainer: the default is sum: --pos-loss sum = the champion command line --none-weight 0.25 bit for bit (the loss of each of 20 smoke steps, last {losses['champ'][-1]:.6f}, the final weights); --pos-loss max differs "
          f"(first loss {losses['max'][0]:.6f} against {losses['champ'][0]:.6f}, last {losses['max'][-1]:.6f}), finite; pos_loss in model.pt's args: ok")


# ------------------------------------------------------------------------------------------------------------------------------ the head seed, the live inputs (S4)
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
            del sys.modules["lane_ptok_data"]
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
    kd = LJK.distill_loss(out, teachers)[0]
    kd.backward()
    gz = out["z"].grad
    assert torch.isfinite(kd) and torch.isfinite(gz).all() and float(gz[~out["cand"]].abs().max()) == 0.0 and float(gz[out["cand"]].abs().min()) > 0, gz
    assert out["types1"].grad is None and out["types"].grad is None and set(info) == {"kd", "kd_s", "kd_l", "ent_s", "ent_l"}, "the type readings are not distilled"
    me = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in out.items()}
    k0, i0 = LJK.distill_loss(out, [me])
    assert float(k0) == 0.0 and i0["kd_s"] == 0.0 and i0["kd_l"] == 0.0, i0                                 # a teacher equal to the student: 0 exactly
    must_fail(LJK.distill_loss, out, [])
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
            held.extend(a[3] if len(a) > 3 else k.get("teachers", ()))
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
    b, seen, real_dl = JK.synthetic_batch(2, 48, seed=0), [], LJK.distill_loss
    with torch.no_grad():
        ref = [t(b)["logp"] for t in tt]

    def spy(o, outs_t):
        seen.append([x["logp"] for x in outs_t])
        return real_dl(o, outs_t)
    with torch.autocast("cpu", dtype=torch.bfloat16):                                                       # the trainer's autocast (bf16 on the GPU; CPU autocast here) around the loss
        patched(LJK, "distill_loss", spy, mod.make_loss(None, none_weight=0.25, teachers=tt), JK.JTokenHead(), b)
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
            hand = [[float(LJK.jtoken_loss(m(b), b, none_weight=0.25)[0]) for b in (JK.synthetic_batch(2, 48, seed=s) for s in range(nb))] for m in (JKT.load_model(n, "cpu", runs)[0] for n in ("ta", "tb"))]
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
                   "the KD on a detached student": [("LJK.distill_loss(out, outs_t)", "LJK.distill_loss({k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}, outs_t)")],
                   "the teachers in train mode": [("heads.append(m.requires_grad_(False))", "heads.append(m.train().requires_grad_(False))")],
                   "the autocast-off of the teachers removed": [('with torch.no_grad(), torch.autocast(b["valid"].device.type, dtype=torch.bfloat16, enabled=amp):', "with torch.no_grad():")],
                   "the start check on the first batch only": [("for v in tl.mean(1))", "for v in tl[:, 0])")],
                   "the cache root out of the teacher key": [('x.get("K"), x.get("cache"))', 'x.get("K"))')],
                   "the sha256 out of the recipe": [('"distill_from", "distill_weight", "distill_sha256", "noise_pos", "noise_e", "rounds"', '"distill_from", "distill_weight", "noise_pos", "noise_e", "rounds"')]}
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


# ------------------------------------------------------------------------------------------------------------------------------ E381: input noise
def _noise_ok(mod, b, sp=0.2, se=0.5, seed=2, step=7):
    """the checks of one `add_noise` (of `mod`, a mutant or the module) on batch b -> True / an AssertionError"""
    v = b["valid"].bool()
    state = torch.random.get_rng_state()
    nb, st = mod.add_noise(b, sp, se, seed, step)
    assert torch.equal(torch.random.get_rng_state(), state), "the global random stream (the data order) is touched"
    assert all(nb[k] is b[k] for k in b if k not in mod.NOISY), "a key other than pos / e / e_raw (a label) changed"
    dp, de, der = nb["pos"] - b["pos"], nb["e"] - b["e"], nb["e_raw"] - b["e_raw"]
    assert bool((dp[~v] == 0).all()) and bool((de[~v] == 0).all()) and bool((der[~v] == 0).all()), "a padding token got noise"
    assert bool((dp[v] != 0).all()) and bool((de[v] != 0).all()) and bool((der[v] != 0).all()), "a valid token got no noise"
    assert torch.allclose(de, der, atol=1e-5), "e and e_raw got different draws"
    nb2, _ = mod.add_noise(b, sp, se, seed, step)
    nb3, _ = mod.add_noise(b, sp, se, seed, step + 1)
    nb4, _ = mod.add_noise(b, sp, se, seed + 1, step)
    assert all(torch.equal(nb[k], nb2[k]) for k in mod.NOISY) and not torch.equal(nb["pos"], nb3["pos"]) and not torch.equal(nb["pos"], nb4["pos"]), "the draw is not a function of (seed, step)"
    assert abs(st[0] - float(dp.norm(dim=-1)[v].mean())) < 1e-5 and abs(st[1] - float(de.abs()[v].mean())) < 1e-5, ("the logged statistics are not the draw's", st)
    return True


def test_noise():
    """E381 card v2 (D126): `--noise-pos SP` / `--noise-e SE` (`jtoken_train.add_noise`): (1) per step, valid tokens only, pos += N(0, SP^2) per axis, e and e_raw += the SAME N(0, SE^2) draw, every other key (the labels) the
    same object, the global random stream untouched, the draw a function of (head seed, head step), the logged mean displacement / mean |noise e| are the draw's and near SP sqrt(pi/2) / SE sqrt(2/pi) on a large batch, zeros = the
    batch itself; textual mutants (e_raw an own draw, no valid mask, the global generator) fail; (2) the trainer: both flags at 0 = the champion command line's 20 smoke steps bit for bit (losses, weights), the noise moves the
    losses (finite) and is in model.pt's args and the log; another noise at a resume is refused; (3) kill at step 12 and resume: the draws are keyed on the restored step counter (steps 0 - 11, then 10 - 19); (4) no read-out takes a noise flag"""
    b = JK.synthetic_batch(4, 96, seed=3)
    assert bool((~b["valid"].bool()).any()), "the toy needs padding"
    assert _noise_ok(JKT, b)
    nb, st = JKT.add_noise(b, 0.0, 0.0, 2, 7)
    assert nb is b and st is None, "zero noise must give the batch itself"
    big = JK.synthetic_batch(16, 512, seed=5)
    _, st = JKT.add_noise(big, 0.2, 0.5, 0, 0)
    assert abs(st[0] / (0.2 * math.sqrt(math.pi / 2)) - 1) < 0.03 and abs(st[1] / (0.5 * math.sqrt(2 / math.pi)) - 1) < 0.03, st
    mutants = {"e_raw own draw": ('out["e_raw"] = b["e_raw"] + ne.to(b["e_raw"].device, b["e_raw"].dtype)',
                                  'out["e_raw"] = b["e_raw"] + (torch.randn(b["e"].shape, generator=g) * se * vf).to(b["e_raw"].device, b["e_raw"].dtype)'),
               "no valid mask": ("vf = valid.to(torch.float32)", "vf = torch.ones_like(valid, dtype=torch.float32)"),
               "global generator": ("npos = torch.randn(b[\"pos\"].shape, generator=g)", "npos = torch.randn(b[\"pos\"].shape)"),
               "no seed in the key": ("manual_seed(int(seed) * 1_000_003 + int(step))", "manual_seed(int(step))"),
               "e_raw without noise": ('out["e_raw"] = b["e_raw"] + ne.to(b["e_raw"].device, b["e_raw"].dtype)', 'out["e_raw"] = b["e_raw"] + 0 * ne.to(b["e_raw"].device, b["e_raw"].dtype)')}
    for name, edit in mutants.items():
        must_fail(_noise_ok, _mutant(edit, of=JKT), b)
    # (2) the trainer
    losses, real, now = {}, LJK.jtoken_loss, [None]

    def recording(out, batch, **k):
        r = real(out, batch, **k)
        losses[now[0]].append(float(r[0]))
        return r
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        base = ["--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "0.25"]
        for name, extra in (("champ", []), ("zero", ["--noise-pos", "0", "--noise-e", "0"]), ("noisy", ["--noise-pos", "0.2", "--noise-e", "0.5"])):
            now[0], losses[name] = name, []
            patched(LJK, "jtoken_loss", recording, JKT.train, JKT.parse_args(["--name", name] + base + extra), runs, 500, 48)
        ck = {n: torch.load(os.path.join(runs, n, "model.pt"), map_location="cpu", weights_only=False) for n in losses}
        log = open(os.path.join(runs, "noisy", "train.log"), encoding="utf-8").read()
        must_fail(JKT.train, JKT.parse_args(["--name", "noisy"] + base + ["--noise-pos", "0.3", "--noise-e", "0.5"]), runs=runs, ckpt_every=500, smoke_k=48)
    assert len(losses["champ"]) == 20 and losses["zero"] == losses["champ"] and all(torch.equal(v, ck["champ"]["state"][k]) for k, v in ck["zero"]["state"].items()), "noise 0 changes the champion recipe"
    assert len(losses["noisy"]) == 21 and losses["noisy"][0] == losses["champ"][0], "the clean first-batch loss (one call more) is not the champion's first loss"
    assert all(math.isfinite(x) for x in losses["noisy"]) and all(x != y for x, y in zip(losses["noisy"][1:], losses["champ"])), "the noise does not move every step's loss"
    assert (ck["noisy"]["args"]["noise_pos"], ck["noisy"]["args"]["noise_e"], ck["champ"]["args"]["noise_pos"]) == (0.2, 0.5, 0.0)
    assert "input noise" in log and "mean displacement" in log and f"the CLEAN loss of this batch {losses['champ'][0]:.4f}" in log, log[-800:]
    # the clean forward inside bf16 autocast (the GPU trainer): the training forward after it still gives every parameter a gradient (autocast's cast cache must not hold the no_grad copies; code critic B1)
    def grads_ok(mod):
        m = mod.build_head(vars(mod.parse_args(["--name", "x"])))
        fn = mod.make_loss(types.SimpleNamespace(log=lambda *a, **k: None), 0.25, "sum", (), 1.0, 0.2, 0.5, 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = fn(m, JK.synthetic_batch(2, 48, seed=1))
            loss.backward()
        missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"no gradient after the clean forward under autocast: {missing[:6]}"
        return True
    assert grads_ok(JKT)
    must_fail(grads_ok, _mutant(("                torch.clear_autocast_cache()", "                pass"), of=JKT))
    mutants["autocast cache kept"] = None
    # (3) kill and resume: the draws follow the head's restored step counter
    steps, real_noise = [], JKT.add_noise

    def rec_noise(b, sp, se, seed, step):
        steps.append(step)
        return real_noise(b, sp, se, seed, step)
    calls = [0]

    def dying(out, batch, **k):
        calls[0] += 1
        if calls[0] == 12:
            raise KeyboardInterrupt("kill")
        return real(out, batch, **k)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        argv = JKT.parse_args(["--name", "k"] + base + ["--noise-pos", "0.2", "--noise-e", "0.5"])
        JKT.add_noise, LJK.jtoken_loss = rec_noise, dying
        try:
            try:
                JKT.train(argv, runs=runs, ckpt_every=5, smoke_k=48)
            except KeyboardInterrupt:
                pass
            else:
                raise RuntimeError("the kill did not happen")
            LJK.jtoken_loss = real
            JKT.train(argv, runs=runs, ckpt_every=5, smoke_k=48)
        finally:
            JKT.add_noise, LJK.jtoken_loss = real_noise, real
        ck_k = torch.load(os.path.join(runs, "k", "model.pt"), map_location="cpu", weights_only=False)
    assert steps == list(range(11)) + list(range(10, 20)), steps                # the 12th loss call is step 10's (the clean first-batch loss is one more call): killed there, resumed from the checkpoint of step 10
    assert all(torch.equal(v, ck["noisy"]["state"][k]) for k, v in ck_k["state"].items()), "the killed and resumed noisy run is not the uninterrupted one bit for bit"
    # (4) the read-outs take no noise
    for f in ("jtoken_eval.py", "jthresh.py"):
        assert "noise" not in open(os.path.join(JT.ROOT, "evaluation", f), encoding="utf-8").read(), f
    print(f"noise: add_noise on valid tokens only, e / e_raw one draw, labels the same objects, the global stream untouched, keyed on (seed, step), logged statistics = the draw's (large batch {st[0]:.4f} m / {st[1]:.4f} "
          f"against {0.2 * math.sqrt(math.pi / 2):.4f} / {0.5 * math.sqrt(2 / math.pi):.4f}); mutants ({', '.join(mutants)}) fail; trainer: noise 0 = the champion's 20 smoke steps bit for bit, noise moves the losses (finite), in the args and "
          f"the log, another noise at a resume refused; kill / resume: draws at steps {steps[0]}..{steps[10]} then {steps[11]}..{steps[-1]}, the final weights = the uninterrupted run's bit for bit; no noise in the read-outs: ok")


# ------------------------------------------------------------------------------------------------------------------------------ E392: message rounds and the early wiring loss
def _champ_numbers(mod, head):
    """forward + loss (none weight 0.25) + backward of `head` (a `mod.JTokenHead` in train mode, its step as set) on a synthetic batch (3 frames, K 48) -> (sum |x| in float64 of logp, scale, types, types1; the loss; sum |grad| of every parameter)"""
    b = mod.synthetic_batch(3, 48, seed=0)
    out = head(b)
    loss = mod.jtoken_loss(out, b, none_weight=0.25)[0]
    loss.backward()
    s = lambda t: float(t.double().abs().sum())  # noqa: E731
    return tuple(s(out[k]) for k in ("logp", "scale", "types", "types1")) + (float(loss), sum(s(p.grad) for p in head.parameters()))


# `_champ_numbers` of the champion head of git HEAD before E392 (recorded from that code in this test's process: 2 threads, CPU): a seeded head (torch seed 0) at head step 0 and SWITCH + 5, and the stored E373 head `runs/e373_cb_s0` (train mode)
CHAMP_HEAD = {0: (118588089.77677026, 9123.041142880917, 503.1498308633454, 525.4964824039489, 4.633667469024658, 928.3080354075339),
              JK.SWITCH + 5: (118588330.5475591, 9126.07040989399, 496.1637700991705, 525.4964824039489, 4.695507526397705, 921.0818347715707),
              "e373": (118604369.3126266, 6105.978199899197, 2566.0501022301614, 2319.9599598338827, 5.47215461730957, 4672.238252198448)}
RS = ("lnw.", "wq.", "wk.", "wbias.", "dest.", "none_k", "none_b")                    # the parameters of the wiring read: shared by the early reads and the final one


def _rounds_head(rounds, msg="real", step=0, perturb=False, mod=JK):
    """a seeded rounds head of `mod`; perturb: the zero last layer of every round moved off zero (at the zero start the messages are exactly 0: a gradient test needs them live)"""
    torch.manual_seed(0)
    m = mod.JTokenHead(rounds=rounds, msg=msg)
    m.set_step(step)
    if perturb:
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for r in m.rounds:
                for p in r.mlp[2].parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return m


def test_rounds_zero():
    """E392 `rounds` 0 (the default) is the champion bit for bit: a seeded head at two head steps and the stored E373 head (`load_model`: its args hold no rounds; skipped when the run is not there) give the forward, the loss and the gradient sums recorded from
    the code of git HEAD before the option (`CHAMP_HEAD`); no extra parameter or state key (127131 parameters); the output dict has the champion's keys. And the round modules are built LAST and start at zero: a rounds head of the same seed has the champion's
    parameters bit for bit, its keys come after the champion's, and its forward and loss are the champion's bit for bit (arm x starts as the control c), for both message modes"""
    keys = ["chord", "cand", "logp", "scale", "types", "types1", "valid"]
    for step in (0, JK.SWITCH + 5):
        torch.manual_seed(0)
        h = JK.JTokenHead()
        h.set_step(step)
        assert JK.count_params(h) == 127131 and len(h.rounds) == 0 and not any(k.startswith("rounds") for k in h.state_dict())
        assert sorted(h(_batch(2))) == sorted(keys), "the champion's output keys"
        torch.manual_seed(0)
        h = JK.JTokenHead()
        h.set_step(step)
        got = _champ_numbers(JK, h)
        assert all(abs(g - w) <= 1e-9 * abs(w) for g, w in zip(got, CHAMP_HEAD[step])), (step, got, CHAMP_HEAD[step])
    note = "stored E373 head SKIPPED (run not there)"
    if os.path.exists(os.path.join(JKT.T.RUNS, "e373_cb_s0", "model.pt")):
        m, args = JKT.load_model("e373_cb_s0", "cpu")
        assert "rounds" not in args and len(m.rounds) == 0
        got = _champ_numbers(JK, m.train())
        assert all(abs(g - w) <= 1e-9 * abs(w) for g, w in zip(got, CHAMP_HEAD["e373"])), (got, CHAMP_HEAD["e373"])
        note = f"the stored E373 head (step {m._step_py - 1}) too"
    for R in (1, 2):
        for msg in JK.EXCHANGE:
            for step in (0, JK.SWITCH + 5):
                torch.manual_seed(0)
                ch = JK.JTokenHead()
                ch.set_step(step)
                rh = _rounds_head(R, msg, step)
                assert list(rh.state_dict())[:len(ch.state_dict())] == list(ch.state_dict()) and all(k.startswith("rounds.") for k in list(rh.state_dict())[len(ch.state_dict()):])
                assert all(torch.equal(v, rh.state_dict()[k]) for k, v in ch.state_dict().items() if k != "step"), "a round module changed the init of an old parameter"
                assert all(float(p.abs().sum()) == 0.0 for r in rh.rounds for p in r.mlp[2].parameters()), "the last layer of a round starts at zero"
                b = _batch(3)
                o1, o2 = ch(b), rh(b)
                assert set(o2) == set(o1) | {"logp_rounds"} and all(torch.equal(o1[k], o2[k]) for k in o1), "a rounds head does not start as the champion function"
                assert torch.equal(LJK.jtoken_loss(o1, b, none_weight=0.25)[0], LJK.jtoken_loss(o2, b, none_weight=0.25)[0])
    print(f"rounds_zero: rounds 0 = the champion of git HEAD (the forward, the loss and the gradient sums of a seeded head at head step 0 / {JK.SWITCH + 5} to 1e-9, {note}); 127131 parameters, the champion's keys; a rounds head "
          f"(1 / 2 rounds, both message modes, both key-weight modes) has the champion's parameters bit for bit, its round keys last, and starts as the champion function (outputs, loss): ok")


def _trace(m, b):
    """one forward of head m with its events in call order: (name, the input x, the output x) of every layer ('layer<i>') and round ('round<k>'), ('read', the x read, None) of every wiring read -> (events, out)"""
    ev, hs = [], []
    for i, layer in enumerate(m.layers):
        hs.append(layer.register_forward_hook(lambda mod, inp, o, i=i: ev.append((f"layer{i + 1}", inp[0], o))))
    for k, r in enumerate(m.rounds):
        hs.append(r.register_forward_hook(lambda mod, inp, o, k=k: ev.append((f"round{k + 1}", inp[0], o))))
    real = m.read_wiring
    m.read_wiring = lambda x, *a: (ev.append(("read", x, None)), real(x, *a))[1]
    try:
        out = m(b)
    finally:
        del m.read_wiring
        for h in hs:
            h.remove()
    return ev, out


def _read_calls(m, b):
    """one forward of head m on batch b -> (how often the modules of the wiring read (`lnw`, `wq`, `wk`, `dest`) and the pair bias `wbias` ran, the output)"""
    calls = {n: 0 for n in ("lnw", "wq", "wk", "dest", "wbias")}
    hs = [getattr(m, n).register_forward_hook(lambda mod, i, o, n=n: calls.__setitem__(n, calls[n] + 1)) for n in calls]
    out = m(b)
    for h in hs:
        h.remove()
    return calls, out


def _check_early_reads(out, b, R):
    """the R early reads of `out` have the shape, the masks and the normalisation of the final one: softmaxes over the gate + none, a padding source has p(none) 1"""
    K = b["pos"].shape[1]
    assert len(out["logp_rounds"]) == R and all(lp.shape == out["logp"].shape and lp.dtype == torch.float32 for lp in out["logp_rounds"])
    nv = ~b["valid"]
    for lp in out["logp_rounds"] + [out["logp"]]:
        assert float((lp.exp().sum(-1) - 1).abs().max()) < 1e-5 and float(lp[..., :K].exp()[~out["cand"]].max()) == 0.0, "an early read is not a softmax over the gate + none"
        assert float((lp.exp()[..., -1][nv[:, None, :].expand(-1, 2, -1)] - 1).abs().max()) < 1e-6, "a padding source has p(none) = 1"


def test_rounds_head():
    """E392 `JTokenHead(rounds=R)` (R = 1, 2; perturbed rounds): the order inside the head by the identity of the tensors (layer 1 -> early read of its output -> round 1 on it -> layer 2 on the round's output -> [early read -> round 2] ->
    layer 3 -> the final read of its output); the early reads and the final read call the SAME modules (`lnw`, `wq`, `wk`, `dest` R + 1 times, `wbias` once) and the head has no parameter but the champion's and the rounds' (20736 each); the early reads
    have the shape, the masks and the normalisation of the final one (a padding source has p(none) 1); they run in eval mode too, and the rounds act (a head without them differs); the head stays a function of the SET (token order, padding)"""
    b = _batch(3)
    K = b["pos"].shape[1]
    champ = {n for n, _ in JK.JTokenHead().named_parameters()}
    for R in (1, 2):
        m = _rounds_head(R, step=JK.SWITCH + 5, perturb=True)
        extra = {n: p for n, p in m.named_parameters() if n not in champ}
        assert all(n.startswith("rounds.") for n in extra) and sum(p.numel() for p in extra.values()) == 20736 * R and {n.split(".")[1] for n in extra} == {str(i) for i in range(R)}
        ev, out = _trace(m, b)
        names = [e[0] for e in ev]
        l1, l2, l3 = (next(e for e in ev if e[0] == f"layer{i}") for i in (1, 2, 3))
        if R == 1:
            assert names == ["layer1", "read", "round1", "layer2", "layer3", "read"], names
            ok = ev[1][1] is l1[2] and ev[2][1] is l1[2] and l2[1] is ev[2][2] and l3[1] is l2[2] and ev[5][1] is l3[2]
        else:
            assert names == ["layer1", "read", "round1", "layer2", "read", "round2", "layer3", "read"], names
            ok = ev[1][1] is l1[2] and ev[2][1] is l1[2] and l2[1] is ev[2][2] and ev[4][1] is l2[2] and ev[5][1] is l2[2] and l3[1] is ev[5][2] and ev[7][1] is l3[2]
        assert ok, "the rounds are not between the layers as the card says (the next layer runs on the updated x, each read reads the x before its round)"
        calls, out = _read_calls(m, b)
        assert calls == dict(lnw=R + 1, wq=R + 1, wk=R + 1, dest=R + 1, wbias=1), calls
        _check_early_reads(out, b, R)
        m.eval()
        with torch.no_grad():
            o_eval = m(b)
            m_off = copy.deepcopy(m)
            m_off.n_early = 0
            o_off = m_off(b)
        assert len(o_eval["logp_rounds"]) == R and all(torch.equal(x, y) for x, y in zip(o_eval["logp_rounds"], out["logp_rounds"])) and torch.equal(o_eval["logp"], out["logp"]), "eval mode differs from train mode"
        assert "logp_rounds" not in o_off and not torch.equal(o_off["logp"], o_eval["logp"]), "the rounds do not act"
        m_u = copy.deepcopy(m)                                                                                # the `msg` option acts: uniform messages (the content null u) give another head than the real ones
        m_u.msg = "uniform"
        with torch.no_grad():
            o_u = m_u(b)
        assert torch.equal(o_u["logp_rounds"][0], o_eval["logp_rounds"][0]) and not torch.equal(o_u["logp"], o_eval["logp"]), "`msg` does not change the exchange (the null arm u would be the real arm)"
        n_rt, read_types = [], m.read_types                                                                   # w1 is read once after layer 1 and is stale at round 2 (the card): read_types runs exactly twice
        m.read_types = lambda x: (n_rt.append(1), read_types(x))[1]
        with torch.no_grad():
            m(b)
        del m.read_types
        assert len(n_rt) == 2, f"the key weight is read {len(n_rt)} times (after layer 1 and after layer 3 only)"
    m = _rounds_head(2, step=JK.SWITCH + 5, perturb=True).eval()                                           # a function of the set: the order of the tokens, padding
    torch.manual_seed(1)
    perm = torch.randperm(K)
    inp = {k: v for k, v in b.items() if k in JK.INPUTS}
    with torch.no_grad():
        o1 = m(inp)
        o2 = m({k: (v[:, perm] if v.dim() >= 2 and v.shape[1] == K else v) for k, v in inp.items()})
        junk = {k: (torch.where(b["valid"].view(*b["valid"].shape, *[1] * (v.dim() - 2)), v, torch.randn_like(v) * 5) if v.is_floating_point() else v) for k, v in inp.items()}
        o3 = m(junk)
        vv = b["valid"]
        for lp1, lp2, lp3 in zip(o1["logp_rounds"] + [o1["logp"]], o2["logp_rounds"] + [o2["logp"]], o3["logp_rounds"] + [o3["logp"]]):
            assert float((lp2[..., :K] - lp1[..., :K][:, :, perm][:, :, :, perm]).abs().max()) < 1e-4, "the token order reaches an early read"
            rows = vv[:, None, :, None].expand(-1, 2, -1, K)
            assert float((lp3[..., :K][rows] - lp1[..., :K][rows]).abs().max()) < 1e-4, "a padding token changes an early read of the valid tokens"
    print("rounds_head: R = 1, 2: the rounds sit between the layers (layer 1, early read, round 1, layer 2, [read, round 2], layer 3, final read: by tensor identity), the early reads call the final read's modules (R + 1 times, wbias once), "
          "the only extra parameters are the rounds' (20736 each), the early reads are softmaxes over the gate + none with p(none) = 1 for padding, run in eval mode, the rounds act; token order and padding leave the reads alone: ok")


def _early_grad(mod, R, step, msg="real"):
    """the gradient of the early term alone (the final read's outputs cut from the graph, wire_w 1) of a seeded rounds head of `mod` with live rounds (msg 'none': no rounds) -> {parameter: sum |grad|}"""
    m = _rounds_head(R, msg, step=step, perturb=True, mod=mod)
    b = mod.synthetic_batch(2, 48, seed=1)
    out = m(b)
    cut = dict(out, logp=out["logp"].detach(), types=out["types"].detach(), types1=out["types1"].detach(), scale=out["scale"].detach())
    loss = mod.jtoken_loss(cut, b, type_w=0.0, scale_w=0.0, none_weight=0.25, wire_w=1.0)[0]
    gs = torch.autograd.grad(loss, list(m.parameters()), allow_unused=True)
    return {n: 0.0 if g is None else float(g.abs().sum()) for (n, _), g in zip(m.named_parameters(), gs)}


def _check_early_grad(mod, msg="real"):
    """the early loss reaches the layers below its read, the rounds below it and the shared wiring read, and nothing else: R 1: layer 1 (and, after the switch, the type reading whose p_none is the key weight) + the wiring read; not layers 2 - 3, not
    round 1 (its read comes before it), not the scale branch; R 2: also round 1 and layer 2; not layer 3, round 2, the scale branch. msg 'none': the same without the rounds (a head that has none)"""
    low = ("ln0.", "proj.", "mlp.", "layers.0.")
    for R, step, live, dead in ((1, 0, low + RS, ("layers.1.", "layers.2.", "rounds.", "sc_", "tnorm.", "types.")),
                                (1, JK.SWITCH + 1, low + RS + ("tnorm.", "types."), ("layers.1.", "layers.2.", "rounds.", "sc_")),
                                (2, 0, low + RS + ("layers.1.",) + ("rounds.0.",) * (msg != "none"), ("layers.2.", "rounds.1.", "sc_", "tnorm.", "types."))):
        g = _early_grad(mod, R, step, msg)
        for pre in live:
            assert sum(v for n, v in g.items() if n.startswith(pre)) > 0, (R, step, pre, "the early loss does not reach it")
        for pre in dead:
            assert all(v == 0.0 for n, v in g.items() if n.startswith(pre)), (R, step, pre, "the early loss reaches what it must not")
        assert all(any(n.startswith(p) for p in live + dead) for n in g), [n for n in g if not any(n.startswith(p) for p in live + dead)]
    return True


def _logp_k_grad(mod, R, msg):
    """the final loss only (wire_w 0) of a seeded rounds head with live rounds -> the sum |grad| on the log-probabilities of every early read"""
    m = _rounds_head(R, msg, step=JK.SWITCH + 1, perturb=True, mod=mod)
    b = mod.synthetic_batch(2, 48, seed=1)
    out = m(b)
    for lp in out["logp_rounds"]:
        lp.retain_grad()
    mod.jtoken_loss(out, b, none_weight=0.25)[0].backward()
    return [0.0 if lp.grad is None else float(lp.grad.abs().sum()) for lp in out["logp_rounds"]]


def test_rounds_grad():
    """E392 gradients: (1) the early loss alone reaches layer 1, the shared wiring read (`lnw`, `wq`, `wk`, `wbias`, `dest`, `none_k`, `none_b`: the early read shares the final read's weights) and only what lies below its read (`_check_early_grad`);
    mutant: an early read on a COPY of the weights fails. (2) The final loss (wire_w 0) reaches every early read's log-probabilities through the messages, for both message modes (the uniform one through the degrees); mutant: p_k detached fails.
    (3) At the zero start the messages are 0: the gradient on logp_k is exactly 0 and the last layer of the round gets one; after ONE optimiser step the gradient reaches logp_k"""
    assert _check_early_grad(JK)
    must_fail(_check_early_grad, _mutant(("logp, _ = self.read_wiring(x, wb, logw, cand)", "logp, _ = __import__(\"copy\").deepcopy(self).read_wiring(x, wb, logw, cand)")))
    g = {(R, msg): _logp_k_grad(JK, R, msg) for R in (1, 2) for msg in JK.EXCHANGE}
    assert all(len(v) == R and all(x > 0 for x in v) for (R, _), v in g.items()), g
    detached = _mutant(("message_weights(link_prob(logp, cand), cand, self.msg)", "message_weights(link_prob(logp.detach(), cand), cand, self.msg)"))
    assert all(x == 0.0 for x in _logp_k_grad(detached, 1, "real")), "the detached mutant still has a gradient on logp_k"
    m = _rounds_head(1)
    b = _batch(2)
    out = m(b)
    out["logp_rounds"][0].retain_grad()
    LJK.jtoken_loss(out, b, none_weight=0.25)[0].backward()
    last = m.rounds[0].mlp[2]
    assert float(out["logp_rounds"][0].grad.abs().sum()) == 0.0 and float(last.weight.grad.abs().sum()) > 0 and float(m.rounds[0].wv.weight.grad.abs().sum()) == 0.0, "the zero start: no gradient on logp_k yet, the last layer learns first"
    torch.optim.SGD(m.parameters(), lr=0.1).step()
    m.zero_grad()
    out = m(b)
    out["logp_rounds"][0].retain_grad()
    LJK.jtoken_loss(out, b, none_weight=0.25)[0].backward()
    assert float(out["logp_rounds"][0].grad.abs().sum()) > 0 and float(m.rounds[0].wv.weight.grad.abs().sum()) > 0, "after one optimiser step the gradient does not reach logp_k"
    print(f"rounds_grad: the early loss alone reaches layer 1, the shared wiring read and what lies below its read, not layers above, the rounds below nor the scale branch (R 1 before / after the switch, R 2; the copied-weights mutant fails); the final "
          f"loss reaches every early read through the messages (sum |grad| on logp_k {', '.join(f'R{R} {msg} {min(v):.1e}' for (R, msg), v in g.items())}; the p_k-detached mutant gets 0); at the zero start the gradient on logp_k is 0 and the last "
          f"layer learns first, after one optimiser step it reaches logp_k: ok")


def _msg_case():
    """4 tokens, short gate 0 -> 1, 0 -> 2, 1 -> 2, 3 -> 2, long gate 0 -> 3, with the early probabilities p"""
    cand = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
    p = torch.zeros(1, 4, 4)
    for c, i, j, v in ((0, 0, 1, .5), (0, 0, 2, .3), (0, 1, 2, .6), (0, 3, 2, .2), (1, 0, 3, .7)):
        cand[0, c, i, j], p[0, i, j] = True, v
    return p, cand


def _check_msg(mod):
    """`mod.message_weights` by hand on `_msg_case`: 'real' = p for both directions; 'uniform' = the gate normalised per column (w_fwd) / per row (w_bwd), times the soft in-degree / out-degree of p; an empty gate gives 0, no NaN"""
    p, cand = _msg_case()
    assert mod.message_weights(p, cand)[0] is p and mod.message_weights(p, cand, "real")[1] is p
    wf, wb = mod.message_weights(p, cand, "uniform")
    third = 1.1 / 3                                                           # token 2: three allowed predecessors, soft in-degree .3 + .6 + .2
    want_f, want_b = torch.zeros(1, 4, 4), torch.zeros(1, 4, 4)
    for i, j, v in ((0, 1, .5), (0, 2, third), (1, 2, third), (3, 2, third), (0, 3, .7)):
        want_f[0, i, j] = v
    for i, j, v in ((0, 1, .5), (0, 2, .5), (0, 3, .5), (1, 2, .6), (3, 2, .2)):                  # token 0: three allowed successors, soft out-degree .5 + .3 + .7
        want_b[0, i, j] = v
    assert torch.allclose(wf, want_f, atol=1e-6) and torch.allclose(wb, want_b, atol=1e-6), (wf, wb)
    assert torch.allclose(wf.sum(1), p.sum(1), atol=1e-6) and torch.allclose(wb.sum(2), p.sum(2), atol=1e-6), "the degrees are not kept"
    e = mod.message_weights(torch.zeros(1, 4, 4), torch.zeros(1, 2, 4, 4, dtype=torch.bool), "uniform")
    assert all(bool(torch.isfinite(x).all()) and float(x.abs().sum()) == 0.0 for x in e), "an empty gate"
    return True


def _check_round(mod):
    """`mod.Round` by hand (explicit sums, asymmetric weights, a live last layer): mf_j = sum_i w_fwd[i, j] W_v x_i, mb_i = sum_j w_bwd[i, j] W_u x_j, x + MLP([mf ; mb]) (no LayerNorm); at the zero start the identity"""
    torch.manual_seed(3)
    r = mod.Round(8)
    x, wf, wb = torch.randn(1, 4, 8), torch.rand(1, 4, 4), torch.rand(1, 4, 4)
    assert torch.equal(r(x, wf, wb), x), "a new round is not the identity"
    with torch.no_grad():
        for p in r.mlp[2].parameters():
            p.copy_(torch.randn(p.shape) * 0.5)
    v, u = r.wv(x)[0], r.wu(x)[0]
    mf = torch.stack([sum(wf[0, i, j] * v[i] for i in range(4)) for j in range(4)])
    mb = torch.stack([sum(wb[0, i, j] * u[j] for j in range(4)) for i in range(4)])
    assert torch.allclose(r(x, wf, wb)[0], x[0] + r.mlp(torch.cat([mf, mb], -1)), atol=1e-5)
    return True


def test_message_weights():
    """E392 `message_weights` and `Round` by hand (`_check_msg`, `_check_round`); textual mutants fail: the degree scale left out, the in-degree used for the backward weights, the column normalisation used for the backward weights, no empty-sum guard, the
    forward message summed over the wrong index, the backward over the wrong one, a LayerNorm-free residual left out"""
    assert _check_msg(JK) and _check_round(JK)
    muts = {"no degree scale": [("clamp_min(1) * p.sum(1, keepdim=True), gate", "clamp_min(1), gate")],
            "backward scaled by the in-degree": [("gate.sum(2, keepdim=True).clamp_min(1) * p.sum(2, keepdim=True)", "gate.sum(2, keepdim=True).clamp_min(1) * p.sum(1, keepdim=True)")],
            "backward normalised per column": [("gate / gate.sum(2, keepdim=True)", "gate / gate.sum(1, keepdim=True)")],
            "no empty-sum guard": [("gate.sum(1, keepdim=True).clamp_min(1)", "gate.sum(1, keepdim=True)"), ]}
    for edits in muts.values():
        must_fail(_check_msg, _mutant(*edits))
    rmuts = {"forward message summed over the successors": [("w_fwd.transpose(1, 2).to(v.dtype) @ v", "w_fwd.to(v.dtype) @ v")],
             "backward message summed over the predecessors": [("w_bwd.to(u.dtype) @ u", "w_bwd.transpose(1, 2).to(u.dtype) @ u")],
             "no residual": [("return x + self.mlp(", "return 0.0 * x + self.mlp(")],
             "the last layer not at zero": [("nn.init.zeros_(self.mlp[2].weight)", "pass")]}
    for edits in rmuts.values():
        must_fail(_check_round, _mutant(*edits))
    print(f"message_weights: 'real' = p, 'uniform' = the column / row normalised gate x the soft in- / out-degree on a 4-token case by hand (degrees kept, an empty gate gives 0), `Round` by hand (the forward message from the predecessors, the "
          f"backward from the successors, a residual, no LayerNorm, the identity at the zero start); mutants ({', '.join(list(muts) + list(rmuts))}) fail: ok")


def _wire_out(frames=(F1, F2)):
    """the hand frames with a final read and two early reads on leaf logits -> (out, batch, frames, [(z, zn) of each early read])"""
    out, b, frames = _hand(frames)
    out = _with_logp(out)
    g = torch.Generator().manual_seed(5)
    K, B = frames[0]["K"], len(frames)
    zs = [(torch.randn(B, 2, K, K, generator=g).requires_grad_(), torch.randn(B, 2, K, generator=g).requires_grad_()) for _ in range(2)]
    out["logp_rounds"] = [JK.wiring_logp(z, zn, out["cand"]) for z, zn in zs]
    return out, b, frames, zs


def _check_wire(mod, pos="sum", w=0.5):
    """`mod.jtoken_loss(wire_w=w, none_weight=0.25, pos_loss=pos)` on `_wire_out` against the hand likelihood of each early read (per frame, the sources of the class, then the frames that have one: `_hand_loss`): info nll<k>_s / nll<k>_l and the loss
    = the loss without the early term + w x the sum over the reads"""
    out, b, frames, zs = _wire_out()
    base, _ = mod.jtoken_loss(out, b, none_weight=0.25, pos_loss=pos)
    loss, info = mod.jtoken_loss(out, b, none_weight=0.25, pos_loss=pos, wire_w=w)
    t1, t3 = out["types1"].detach().tolist(), out["types"].detach().tolist()
    tot = 0.0
    for k, (z, zn) in enumerate(zs, 1):
        per = [_hand_loss(f, z.detach().tolist()[n], zn.detach().tolist()[n], t1[n], t3[n], 0.25, pos) for n, f in enumerate(frames)]
        s, l = (sum(p[c] for p in per if p[c] is not None) / sum(p[c] is not None for p in per) for c in range(2))
        assert f"nll{k}_s" in info and f"nll{k}_l" in info, sorted(info)
        assert abs(info[f"nll{k}_s"] - s) < 1e-5 and abs(info[f"nll{k}_l"] - l) < 1e-5, (k, info, s, l)
        tot += s + l
    assert abs(float(loss) - float(base) - w * tot) < 1e-5, (float(loss), float(base), w * tot)
    return tot


def test_wire_loss():
    """E392 `jtoken_loss(wire_w=)`: the early term by hand on the two hand frames and two early reads (none weight 0.25, both `pos_loss` modes, W 0.5): info nll1_s / nll1_l / nll2_s / nll2_l and loss = the loss without it + W x the sum; wire_w 0 (the default)
    adds nothing, reads no `logp_rounds` (the same loss without the key, no early info key); wire_w > 0 on an output without early reads is refused; textual mutants fail: no none weight on the early reads, a global mean over the sources, the final read in
    place of the early one, the weight W ignored, only the first read"""
    tot = {pos: _check_wire(JK, pos) for pos in LJK.POS_LOSS}
    assert tot["max"] > tot["sum"] + 1e-3, tot
    out, b, frames, zs = _wire_out()
    plain = {k: v for k, v in out.items() if k != "logp_rounds"}
    l0, i0 = LJK.jtoken_loss(out, b, none_weight=0.25)
    assert torch.equal(l0, LJK.jtoken_loss(plain, b, none_weight=0.25)[0]) and not any(k.startswith("nll1") or k.startswith("nll2") for k in i0), "wire_w 0 must add nothing and read nothing"
    must_fail(LJK.jtoken_loss, plain, b, none_weight=0.25, wire_w=0.5)
    muts = {"no none weight on the early reads": [("nk = _source_nll(lpk, G, none_weight, pos_loss)", "nk = _source_nll(lpk, G, 1.0, pos_loss)")],
            "a global mean": [("s_k, l_k = (_frame_mean(nk[:, c], rows[:, c]) for c in range(2))", "s_k, l_k = ((nk[:, c] * rows[:, c]).sum() / rows[:, c].sum() for c in range(2))")],
            "the final read instead of the early one": [("nk = _source_nll(lpk, G, none_weight, pos_loss)", 'nk = _source_nll(out["logp"], G, none_weight, pos_loss)')],
            "W ignored": [("loss = loss + wire_w * (s_k + l_k)", "loss = loss + (s_k + l_k)")],
            "only the first read": [('enumerate(out["logp_rounds"], 1)', 'enumerate(out["logp_rounds"][:1], 1)')]}
    for edits in muts.values():
        must_fail(_check_wire, _mutant(*edits))
    print(f"wire_loss: the early term by hand on two early reads of the two hand frames (W 0.5, none weight 0.25; sum {tot['sum']:.5f}, max {tot['max']:.5f}); wire_w 0 adds nothing and reads nothing, W > 0 without early reads is refused; "
          f"mutants ({', '.join(muts)}) fail: ok")


RARMS = (("x", ["--rounds", "1"]), ("r1", ["--rounds", "1", "--wire-weight", "0.5"]), ("r2u", ["--rounds", "2", "--wire-weight", "0.5", "--msg", "uniform"]), ("n1", ["--rounds", "1", "--wire-weight", "0.5", "--msg", "none"]),
         ("s2", ["--rounds", "2", "--wire-weight", "0.5", "--msg", "self"]), ("m2", ["--rounds", "2", "--msg", "self"]))


def _rounds_runs(mod, runs):
    """the smoke runs (20 steps, the champion recipe on synthetic batches) of six E392 arms through `mod` (jtoken_train or a textual mutant): x (--rounds 1), r1 (+ --wire-weight 0.5), r2u (--rounds 2, --msg uniform, --wire-weight 0.5), n1 (--rounds 1, --msg none,
    --wire-weight 0.5), s2 / m2 (--rounds 2, --msg self, with / without --wire-weight 0.5: the capacity null): every step's loss is finite and called with the arm's wire_w; the early info keys exist exactly for W > 0 (nll1_*, nll2_* with two rounds); x starts as the champion (its first loss is CHAMP[0]) and r1's first loss is x's + 0.5 x its early term; n1 has
    the champion's head (no round module, the champion's state keys) and its first loss is CHAMP[0] + 0.5 x its early term (x untouched); the rounds move off zero; model.pt's args hold the three options and `load_model` rebuilds the head (rounds, msg, strict state);
    the recipe refuses another rounds / wire weight / msg at a resume"""
    real, got, now = LJK.jtoken_loss, {}, [None]

    def recording(out, batch, **k):
        r = real(out, batch, **k)
        got[now[0]].append((float(r[0]), dict(r[1]), k.get("wire_w", 0.0)))
        return r
    for name, extra in RARMS:
        now[0], got[name] = name, []
        patched(LJK, "jtoken_loss", recording, mod.train, mod.parse_args(["--name", name] + BASE + extra), runs, 500, 48)
    for name, extra in RARMS:
        R, W, msg = int(extra[1]), (float(extra[extra.index("--wire-weight") + 1]) if "--wire-weight" in extra else 0.0), (extra[extra.index("--msg") + 1] if "--msg" in extra else "real")
        steps = got[name]
        assert len(steps) == 20 and all(math.isfinite(s[0]) and s[2] == W for s in steps), (name, [s[0] for s in steps][:3], {s[2] for s in steps})
        assert all(("nll1_s" in s[1]) == (W > 0) and ("nll2_s" in s[1]) == (W > 0 and R == 2) for s in steps), (name, sorted(steps[0][1]))
        ck = torch.load(os.path.join(runs, name, "model.pt"), map_location="cpu", weights_only=False)
        assert (ck["args"]["rounds"], ck["args"]["wire_weight"], ck["args"]["msg"]) == (R, W, msg), ck["args"]
        m, a = mod.load_model(name, "cpu", runs)
        assert len(m.rounds) == (0 if msg == "none" else R) and m.n_early == R and m.msg == msg and all(torch.equal(v, m.state_dict()[k]) for k, v in ck["state"].items()), "load_model does not rebuild the head of the run"
        assert all(float(r.mlp[2].weight.abs().sum()) > 0 for r in m.rounds), "the rounds did not move off zero"
        assert (msg != "none") or (list(ck["state"]) == list(JK.JTokenHead().state_dict()) and JK.count_params(m) == 127131), "a msg none run does not hold the champion's parameters"
        rows = [json.loads(x) for x in open(os.path.join(runs, name, "log.jsonl"))]
        assert ("nll1_s" in rows[-1]) == (W > 0), rows[-1]
        rd = os.path.join(runs, name)
        for k, v in (("rounds", (R + 1) % 3), ("wire_weight", W + 0.25), ("msg", "uniform" if msg == "real" else "real")):
            try:
                mod.check_start(dict(ck["args"], **{k: v}), rd)
            except AssertionError:
                continue
            raise AssertionError(f"a resume of {name} with another {k} accepted")
        mod.check_start(ck["args"], rd)
    assert abs(got["x"][0][0] - CHAMP[0]) < 1e-9, (got["x"][0][0], CHAMP[0])
    assert abs(got["m2"][0][0] - CHAMP[0]) < 1e-9, ("m2 (msg self, no early loss) does not start as the champion", got["m2"][0][0], CHAMP[0])
    i3 = got["s2"][0][1]
    assert abs(got["s2"][0][0] - CHAMP[0] - 0.5 * sum(i3[k] for k in ("nll1_s", "nll1_l", "nll2_s", "nll2_l"))) < 1e-5, "step 1 of s2 is not the champion's loss + W x the two early terms"
    i1 = got["r1"][0][1]
    assert abs(got["r1"][0][0] - got["x"][0][0] - 0.5 * (i1["nll1_s"] + i1["nll1_l"])) < 1e-5, "step 1 of r1 is not x's + W x the early term"
    assert got["r1"][-1][0] != got["x"][-1][0]
    i2 = got["n1"][0][1]
    assert abs(got["n1"][0][0] - CHAMP[0] - 0.5 * (i2["nll1_s"] + i2["nll1_l"])) < 1e-5 and abs(i2["nll1_s"] - i1["nll1_s"]) < 1e-6, "step 1 of n1 is not the champion's loss + W x the early term (x untouched: the same early read as r1's at the zero start)"
    return got


def test_rounds_trainer():
    """E392 in the trainer (`_rounds_runs`): the flags, ARCH_DEFAULTS, RECIPE, `build_head` / `load_model`, `make_loss`; the defaults (0, 0.0, real) = the champion command line; refused: --rounds 3, --wire-weight or --msg uniform without --rounds, a negative weight; a model.pt of
    before (no rounds in its args) loads as rounds 0, one of a rounds head needs its args (strict load); textual mutants of jtoken_train.py fail: the weight not passed to the loss, rounds / msg not built, the recipe without wire_weight"""
    d = vars(JKT.parse_args(["--name", "x"]))
    assert (d["rounds"], d["wire_weight"], d["msg"]) == (0, 0.0, "real") and all(JKT.ARCH_DEFAULTS[k] == d[k] for k in ("rounds", "wire_weight", "msg")) and {"rounds", "wire_weight", "msg"} <= set(JKT.RECIPE)
    for bad in (["--wire-weight", "0.5"], ["--msg", "uniform"], ["--rounds", "1", "--wire-weight", "-0.5"]):
        must_fail(JKT.parse_args, ["--name", "x"] + bad)
    try:
        JKT.parse_args(["--name", "x", "--rounds", "3"])
    except SystemExit:
        pass
    else:
        raise AssertionError("--rounds 3 accepted")
    h = JKT.build_head(vars(JKT.parse_args(["--name", "x", "--rounds", "2", "--msg", "uniform", "--wire-weight", "0.5"])))
    assert len(h.rounds) == 2 and h.msg == "uniform" and len(JKT.build_head({}).rounds) == 0 and JKT.build_head({}).msg == "real"
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        for name, args, state in (("before", dict(seed=0), JK.JTokenHead().state_dict()), ("rr", dict(seed=0, rounds=2, msg="uniform", wire_weight=0.5), h.state_dict()), ("rr_noargs", dict(seed=0), h.state_dict())):
            os.makedirs(os.path.join(runs, name))
            torch.save(dict(state=state, args=args, steps=0), os.path.join(runs, name, "model.pt"))
        assert len(JKT.load_model("before", "cpu", runs)[0].rounds) == 0 and (len(JKT.load_model("rr", "cpu", runs)[0].rounds), JKT.load_model("rr", "cpu", runs)[0].msg) == (2, "uniform"), "load_model"
        try:
            JKT.load_model("rr_noargs", "cpu", runs)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a rounds state loaded into a champion head")
    for R, W in ((2, 0.5), (1, 0.0)):                                                                      # the clock line (call 50) names the early reads exactly when W > 0
        lines, hd, bt = [], JKT.build_head(dict(rounds=R)), JK.synthetic_batch(2, 48, seed=0)
        fn = JKT.make_loss(types.SimpleNamespace(log=lines.append), 0.25, wire_w=W)
        for _ in range(50):
            fn(hd, bt)
        clock = [x for x in lines if x.startswith("clock:")]
        assert len(clock) == 1 and ("early reads (E392) nll1_s" in clock[0]) == (W > 0) and ("nll2_s" in clock[0]) == (W > 0 and R == 2), clock
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        got = _rounds_runs(JKT, runs)
    muts = {"the wire weight is not passed to the loss": [("hs, a.wire_weight, a.curve_w, a.node_w), run,", "hs, 0.0, a.curve_w, a.node_w), run,")],
            "the rounds are not built": [('rounds=int(args.get("rounds", ARCH_DEFAULTS["rounds"])), msg=', "rounds=0, msg=")],
            "the messages are not built": [('msg=args.get("msg", ARCH_DEFAULTS["msg"]), pos_head=', 'msg="real", pos_head=')],
            "the recipe lacks wire_weight": [('"noise_e", "rounds", "wire_weight", "msg", "pos_head"', '"noise_e", "rounds", "msg", "pos_head"')]}
    for edits in muts.values():
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            must_fail(_rounds_runs, _mutant(*edits, of=JKT), tmp)
    print(f"rounds_trainer: defaults (0, 0.0, real) = the champion command line, refusals (--rounds 3, a weight or uniform messages without rounds, a negative weight), a model.pt of before loads as rounds 0; 20 smoke steps of x / r1 / r2u / n1 (msg none) / s2 / m2 (msg self): finite, "
          f"the early info keys exactly for W > 0, x starts as the champion (first loss {got['x'][0][0]:.6f}), r1's first loss = x's + 0.5 x the early term ({got['r1'][0][0]:.6f}), n1's = the champion's + 0.5 x its early term ({got['n1'][0][0]:.6f}; the champion's "
          f"parameters, no round module), rounds move off zero, the three options in model.pt and rebuilt by load_model, "
          f"a resume with another rounds / weight / msg refused; textual mutants ({', '.join(muts)}) fail: ok")


# ------------------------------------------------------------------------------------------------------------------------------ E392: msg 'none', deep supervision without exchange
def _none_run(mod, R, step, live, train):
    """a seeded head of `mod` (R None: the champion; else R early reads with msg 'none'), live_inputs `live`, in train / eval mode, head step `step`, on a synthetic batch with the float INPUTS as leaves: forward, the champion's loss (none weight 0.25,
    wire_w 0), backward -> (out, loss, {parameter: grad}, {input: grad})"""
    torch.manual_seed(0)
    m = mod.JTokenHead(live_inputs=live) if R is None else mod.JTokenHead(live_inputs=live, rounds=R, msg="none")
    m.set_step(step)
    m.train(train)
    b = _batch(2)
    x = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in b.items()}
    out = m(x)
    loss = mod.jtoken_loss(out, x, none_weight=0.25)[0]
    loss.backward()
    return out, loss, {n: p.grad for n, p in m.named_parameters()}, {k: x[k].grad for k in JK.INPUTS if x[k].is_floating_point()}


def _same(a, b, exact=True):
    """two optional tensors equal: both None, or bit for bit (exact) / to a relative 1e-5"""
    return (a is None and b is None) or (a is not None and b is not None and (torch.equal(a, b) if exact else torch.allclose(a, b, rtol=1e-5, atol=1e-8)))


def _check_none_trace(mod, R, train):
    """the order inside a msg 'none' head by the identity of the tensors (`_trace`): train mode layer 1 -> early read of its output -> layer 2 on THE SAME tensor [-> early read of layer 2's output] -> layer 3 on the same tensor -> the final read; eval mode: no
    early read, three layers and the final read; the reads call the modules of the final read R + 1 (train) / once (eval) times, the pair bias once"""
    m = _rounds_head(R, "none", step=JK.SWITCH + 5, mod=mod).train(train)
    b = _batch(2)
    ev, _ = _trace(m, b)
    L = [e for e in ev if e[0].startswith("layer")]
    want = ["layer1", "read", "layer2"] + ["read"] * (R == 2) + ["layer3", "read"] if train else ["layer1", "layer2", "layer3", "read"]
    assert [e[0] for e in ev] == want, ([e[0] for e in ev], want)
    reads = [e[1] for e in ev if e[0] == "read"]
    outs = ([L[0][2]] + [L[1][2]] * (R == 2) if train else []) + [L[2][2]]
    assert L[1][1] is L[0][2] and L[2][1] is L[1][2] and len(reads) == len(outs) and all(r is o for r, o in zip(reads, outs)), "x does not pass unchanged between the layers, or an early read reads another tensor"
    calls, _ = _read_calls(m, b)
    n = R + 1 if train else 1
    assert calls == dict(lnw=n, wq=n, wk=n, dest=n, wbias=1), calls


def _check_none_head(mod):
    """`mod.JTokenHead(rounds=R, msg='none')` (jtoken or a textual mutant), R = 1, 2, at head step 0 / after the switch, live_inputs both ways, the champion's loss: no round module, the champion's 127131 parameters, state keys, order, shapes and seeded values;
    EVAL mode = the champion's forward bit for bit (every output, no `logp_rounds`, the loss, every parameter gradient, every input gradient); TRAIN mode = the champion's outputs, loss and parameter gradients bit for bit (the loss does not read the early
    reads at wire_w 0; the input gradients to 1e-5: the backward order into g may differ) plus R early reads of the final read's shape and masks; the order and tensor identity of `_check_none_trace`"""
    torch.manual_seed(0)
    ch = mod.JTokenHead()
    for R in (1, 2):
        torch.manual_seed(0)
        h = mod.JTokenHead(rounds=R, msg="none")
        assert len(h.rounds) == 0 and h.n_early == R and mod.count_params(h) == mod.count_params(ch) == 127131, "msg none must have no round module and the champion's parameters"
        assert [(k, v.shape) for k, v in h.state_dict().items()] == [(k, v.shape) for k, v in ch.state_dict().items()] and all(torch.equal(v, ch.state_dict()[k]) for k, v in h.state_dict().items()), "the state of a msg none head is not the champion's"
        for train in (False, True):
            _check_none_trace(mod, R, train)
    b = _batch(2)
    for step in (0, JK.SWITCH + 5):
        for live in (False, True):
            for train in (False, True):
                ref = _none_run(mod, None, step, live, train)
                for R in (1, 2):
                    got = _none_run(mod, R, step, live, train)
                    assert set(got[0]) == set(ref[0]) | ({"logp_rounds"} if train else set()), (train, sorted(got[0]))
                    assert all(torch.equal(ref[0][k], got[0][k]) for k in ref[0]) and torch.equal(ref[1], got[1]), "the outputs or the loss differ from the champion's"
                    assert all(_same(g, got[2][n]) for n, g in ref[2].items()), "a parameter gradient differs from the champion's"
                    assert all(_same(g, got[3][k], exact=not train) for k, g in ref[3].items()), "an input gradient differs from the champion's"
                    if train:
                        _check_early_reads(got[0], b, R)
    return True


def _refused(fn, *a):
    try:
        fn(*a)
    except AssertionError:
        return True
    return False


def _check_none_refusals(mod):
    """`mod.parse_args` (jtoken_train or a mutant): --msg none is refused without --rounds and without --wire-weight > 0, accepted with both"""
    for bad in (["--msg", "none"], ["--msg", "none", "--wire-weight", "0.5"], ["--rounds", "1", "--msg", "none"], ["--rounds", "2", "--msg", "none", "--wire-weight", "0"]):
        assert _refused(mod.parse_args, ["--name", "x"] + bad), bad
    assert all(mod.parse_args(["--name", "x", "--rounds", R, "--msg", "none", "--wire-weight", "0.5"]).msg == "none" for R in ("1", "2"))
    return True


def test_msg_none():
    """E392 `msg='none'` (deep supervision without exchange): `_check_none_head` (the champion's parameters and state, eval mode = the champion's forward bit for bit incl. the gradients with live_inputs both ways, train mode = the champion's outputs and loss
    plus the early reads, x passes unchanged); textual mutants fail: x changed by the early read, the early reads in eval mode, round modules built, one early read only. The early loss alone reaches layer 1 (and layer 2), the shared wiring read and nothing else
    (`_check_early_grad`, mutant: an early read on a copy of the weights), `jtoken_loss(wire_w)` = the champion's loss + W x the early terms and refuses an eval-mode output (no early read). The trainer refuses --msg none without --rounds or without --wire-weight > 0
    (`_check_none_refusals`, mutant: the weight check dropped)"""
    assert _check_none_head(JK)
    muts = {"x changed by the early read": [("            return x, logp\n", "            return x + 0.01, logp\n")],
            "early reads in eval mode": [('n_early = self.n_early if self.msg not in SUPERVISE or self.training else 0', "n_early = self.n_early")],
            "round modules built": [('range(0 if msg == "none" else rounds)', "range(rounds)")],
            "one early read only": [("            if n <= n_early:", "            if n <= min(n_early, 1):")]}
    for edits in muts.values():
        must_fail(_check_none_head, _mutant(*edits))
    assert _check_early_grad(JK, "none")
    must_fail(_check_early_grad, _mutant(("logp, _ = self.read_wiring(x, wb, logw, cand)", "logp, _ = __import__(\"copy\").deepcopy(self).read_wiring(x, wb, logw, cand)")), "none")
    b = _batch(2)
    for R in (1, 2):
        m = _rounds_head(R, "none", step=JK.SWITCH + 5)
        out = m(b)
        l0, _ = LJK.jtoken_loss(out, b, none_weight=0.25)
        lw, info = LJK.jtoken_loss(out, b, none_weight=0.25, wire_w=0.5)
        assert abs(float(lw) - float(l0) - 0.5 * sum(info[f"nll{k}_{c}"] for k in range(1, R + 1) for c in "sl")) < 1e-5 and (f"nll{R}_s" in info) and ("nll3_s" not in info), info
        with torch.no_grad():
            must_fail(LJK.jtoken_loss, m.eval()(b), b, none_weight=0.25, wire_w=0.5)
    assert _check_none_refusals(JKT)
    must_fail(_check_none_refusals, _mutant(('args.get("msg", "real") != "none" or args.get("wire_weight", 0.0) > 0.0', "True"), of=JKT))
    print(f"msg_none: rounds 1 / 2 with msg none have no round module, the champion's {JK.count_params(JK.JTokenHead(rounds=1, msg='none'))} parameters, state keys and seeded values; eval mode = the champion's forward bit for bit (outputs, loss, parameter and input "
          f"gradients, live_inputs both ways, both key-weight modes), train mode = the champion's outputs, loss and parameter gradients plus the early reads (x passes unchanged: tensor identity, one wiring read in eval, R + 1 in train); mutants ({', '.join(muts)}) fail; "
          f"the early loss alone reaches layer 1 (2) and the shared read only; W x the early terms in the loss, an eval-mode output refused; the trainer refuses --msg none without --rounds or --wire-weight: ok")


# ------------------------------------------------------------------------------------------------------------------------------ E392: msg 'self', the capacity null of the exchange
def _check_self_head(mod):
    """`mod.JTokenHead(rounds=R, msg='self')` (jtoken or a textual mutant), R = 1, 2: the parameters of the real rounds head of the same seed (R x 20736 more than the champion's 127131, the same state keys and seeded values, the rounds built last);
    a perturbed `Round` called with no weights = called with the identity weights = x + MLP([W_v x ; W_u x]) by hand; EVAL mode: no early read, no `logp_rounds`, one wiring read, and the rounds ACT (the output differs from the head with its rounds
    off); TRAIN mode: R early reads in the shapes of the final read, and the final loss at wire_w 0 has no gradient on any of them (p_k does not enter x)"""
    b = _batch(2)
    for R in (1, 2):
        torch.manual_seed(0)
        h = mod.JTokenHead(rounds=R, msg="self")
        torch.manual_seed(0)
        real = mod.JTokenHead(rounds=R, msg="real")
        assert len(h.rounds) == R and mod.count_params(h) == 127131 + R * 20736 == mod.count_params(real), "a msg self head must have the real rounds' parameters"
        assert list(h.state_dict()) == list(real.state_dict()) and all(torch.equal(v, real.state_dict()[k]) for k, v in h.state_dict().items()), "the state of a msg self head is not the real rounds head's"
        m = _rounds_head(R, "self", step=JK.SWITCH + 5, perturb=True, mod=mod)
        r = m.rounds[0]
        x = torch.randn(2, 48, 64, generator=torch.Generator().manual_seed(5))
        eye = torch.eye(48)[None].expand(2, 48, 48)
        by_hand = x + r.mlp(torch.cat([r.wv(x), r.wu(x)], -1))
        assert torch.equal(r(x, None, None), by_hand) and torch.allclose(r(x, eye, eye), by_hand, atol=1e-6) and not torch.equal(by_hand, x), "the self round is not x + MLP([W_v x ; W_u x])"
        out = m.eval()(b)
        assert "logp_rounds" not in out, "an eval-mode msg self head holds an early read"
        calls, _ = _read_calls(m, b)
        assert calls == dict(lnw=1, wq=1, wk=1, dest=1, wbias=1), calls
        off = _rounds_head(R, "self", step=JK.SWITCH + 5, perturb=True, mod=mod).eval()
        with torch.no_grad():
            for q in off.rounds:
                for p in q.mlp[2].parameters():
                    p.zero_()
            assert float((off(b)["logp"] - out["logp"]).abs().max()) > 1e-6, "the rounds of an eval-mode msg self head do not act"
        tr = _rounds_head(R, "self", step=JK.SWITCH + 5, perturb=True, mod=mod).train()(b)
        assert len(tr["logp_rounds"]) == R and all(lp.shape == tr["logp"].shape for lp in tr["logp_rounds"]), "a msg self head in train mode has no early reads"
        assert all(torch.equal(tr[k], out[k]) for k in ("logp", "types", "types1", "scale")), "the train-mode and the eval-mode forward of a msg self head differ (a round applied twice, skipped or with another index in one mode)"
        assert _logp_k_grad(mod, R, "self") == [0.0] * R, "the final loss reaches an early read of a msg self head: p_k enters x"
    return True


def _check_self_refusals(mod):
    """`mod.parse_args` (jtoken_train or a mutant): --msg self is refused without --rounds, accepted with --rounds 1 / 2 at any non-negative --wire-weight"""
    assert _refused(mod.parse_args, ["--name", "x", "--msg", "self"]) and _refused(mod.parse_args, ["--name", "x", "--msg", "self", "--wire-weight", "0.5"])
    assert all(mod.parse_args(["--name", "x", "--rounds", R, "--msg", "self"] + w).msg == "self" for R in ("1", "2") for w in ([], ["--wire-weight", "0.5"]))
    assert all(_refused(mod.parse_args, ["--name", "x", "--rounds", "2", "--msg", m, "--wire-weight", "0.5", "--noise-pos", "0.1"]) for m in ("self", "none")), "the noise options with an early-read msg and a wire weight crash at the first step: refuse them"
    return True


def test_msg_self():
    """E392 `msg='self'` (the capacity null: the real rounds' parameters, each token's message is its own features, the early reads only supervise): `_check_self_head`; textual mutants fail: the messages through the real weights, no round in eval mode, early
    reads in eval mode, round modules not built; the early loss alone reaches the same parameters as the real exchange's (`_check_early_grad`); the trainer accepts it at wire weight 0 (the block alone) and 0.5 and refuses it without --rounds"""
    assert _check_self_head(JK)
    muts = {"messages through the real weights": [("            return self.rounds[k](x, None, None), logp\n", "            return self.rounds[k](x, *message_weights(link_prob(logp, cand), cand, \"real\")), logp\n")],
            "no round in eval mode": [('elif self.msg == "self" and n <= self.n_early:', "elif False:")],
            "early reads in eval mode": [('n_early = self.n_early if self.msg not in SUPERVISE or self.training else 0', "n_early = self.n_early")],
            "round modules not built": [('range(0 if msg == "none" else rounds)', 'range(0 if msg in ("none", "self") else rounds)')]}
    for edits in muts.values():
        must_fail(_check_self_head, _mutant(*edits))
    assert _check_early_grad(JK, "self")
    assert _check_self_refusals(JKT)
    must_fail(_check_self_refusals, _mutant(('assert args.get("rounds", 0) > 0 or', "assert True or"), of=JKT))
    print(f"msg_self: rounds 1 / 2 with msg self hold the real rounds' parameters ({JK.count_params(JK.JTokenHead(rounds=1, msg='self'))} / {JK.count_params(JK.JTokenHead(rounds=2, msg='self'))}), the round = x + MLP([W_v x ; W_u x]) = the identity-weight round; eval mode: no early read, "
          f"the rounds act; train mode: R early reads without a gradient from the final loss; mutants ({', '.join(muts)}) fail; the early loss reaches what the real exchange's does; the trainer accepts it at wire weight 0 and 0.5, refuses it without --rounds: ok")


# ------------------------------------------------------------------------------------------------------------------------------ E395: the position head and its curve loss
CR_M = 0.5 * np.array([[0, 2, 0, 0], [-1, 0, 1, 0], [2, -5, 4, -1], [-1, 3, -3, 1]], float)       # the Catmull-Rom basis matrix: [1 t t^2 t^3] @ CR_M @ [P0 P1 P2 P3]: an independent form of `jtoken.cr_points_t`
TS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _np_rho(x, beta):
    return 0.5 * x * x / beta if abs(x) < beta else abs(x) - 0.5 * beta


def _np_losses(refined, gpos, gdir, btok, gl, beta=0.5, ts=TS, chord_max=8.0, ratio_max=2.0):
    """the independent numpy form of `curve_losses` (loops over frames, links and t; the Catmull-Rom curve by the basis matrix): refined (B, K, 2), gpos / gdir (B, Ng, 2), btok (B, Ng), gl (B, Lg, 4) -> (curve, node, n_curve, n_links, n_excl, n_gl)"""
    rot = lambda v: np.array([-v[1], v[0]])  # noqa: E731
    curves, nodes, n_cur, n_lk, n_ex, n_gl = [], [], 0, 0, 0, 0
    for b in range(len(refined)):
        Q = {g: refined[b][btok[b][g]] for g in range(len(btok[b])) if btok[b][g] >= 0}
        ns = [_np_rho(float((Q[g] - gpos[b][g]) @ (rot(gdir[b][g]) / max(np.linalg.norm(gdir[b][g]), 1e-9))), beta) for g in Q]
        if ns:
            nodes.append(np.mean(ns))
        es = []
        for u, v, pre, suc in gl[b]:
            n_gl += u >= 0
            if u < 0 or u not in Q or v not in Q:
                continue
            n_lk += 1
            ap, asu = pre >= 0 and pre in Q, suc >= 0 and suc in Q

            def four(P):
                return np.stack([P[pre] if ap else 2 * P[u] - P[v], P[u], P[v], P[suc] if asu else 2 * P[v] - P[u]])
            Pq, Gt = four(Q), four({g: gpos[b][g] for g in range(len(gpos[b]))})
            ch = [np.linalg.norm(Gt[i + 1] - Gt[i]) for i in range(3)]
            if not (ch[1] <= chord_max and max(ch) <= ratio_max * min(ch)):
                n_ex += 1
                continue
            for t in ts:
                tan = np.array([0, 1, 2 * t, 3 * t * t]) @ CR_M @ Gt
                if np.linalg.norm(tan) < 1e-6:
                    continue
                dc = np.array([1, t, t * t, t ** 3]) @ CR_M @ (Pq - Gt)
                es.append(_np_rho(float(dc @ rot(tan) / np.linalg.norm(tan)), beta))
        n_cur += len(es)
        if es:
            curves.append(np.mean(es))
    return (np.mean(curves) if curves else 0.0), (np.mean(nodes) if nodes else 0.0), n_cur, n_lk, n_ex, n_gl


def _cl(mod, pos, dirv, dpos, gpos, gdir, btok, gl, **k):
    """`mod.curve_losses` on nested python lists (frames x ...) in float64 -> (curve, node, info, the dpos leaf)"""
    t = lambda x, dt=torch.float64: torch.tensor(x, dtype=dt)  # noqa: E731
    dp = t(dpos).requires_grad_()
    g = dict(gt_pos=t(gpos), gt_dir=t(gdir), gt_btok=t(btok, torch.long), gl=t(gl, torch.long))
    return (*mod.curve_losses(dp, t(pos), t(dirv), g, **k), dp)


NODE_TOY = dict(pos=[[[0, .3], [2, .3], [4, .3], [0, 0]], [[0, 1.0], [2, 5.0], [0, 0], [0, 0]]], dirv=[[[1, 0]] * 4] * 2, gpos=[[[0, 0], [2, 0], [4, 0]], [[0, 0], [2, 0], [0, 0]]], gdir=[[[1, 0]] * 3] * 2,
                btok=[[0, 1, 2], [0, -1, -1]], gl=[[[-1] * 4]] * 2)
# two frames, tokens 4 and 4 (the last of each is padding): frame 1 has three GT nodes with tokens 0.3 m to the left of their GT node (the lane heads +x, the left normal is +y); frame 2 one GT node with a token 1.0 m to the left
# and a second GT node without a token (its far-away token 1, 5 m off, must not count)


def _check_node_hand(mod):
    """the node loss by hand: rho(0.3) = 0.09 at beta 0.5 (frame 1), rho(1.0) = 1.0 - 0.25 = 0.75 (the knee; frame 2), per frame the mean over the present nodes, then over the frames: 0.42; the gradient 0.6 x 1/3 x 1/2 on the three tokens of
    frame 1, 1 x 1 x 1/2 on the token of frame 2, 0 on the token without a GT node and on the padding; dpos = -0.3 along the token normal puts frame 1 on its GT line (0.09 -> 0): 0.375; no chord rule on the node loss"""
    c, n, info, dp = _cl(mod, dpos=[[0] * 4] * 2, **NODE_TOY)
    assert abs(float(n) - 0.42) < 1e-12 and info["node"] == float(n) and float(c) == 0.0 and info["n_curve"] == 0.0, (float(n), info)
    n.backward()
    assert np.allclose(dp.grad.numpy(), [[.1, .1, .1, 0], [.5, 0, 0, 0]], atol=1e-12), dp.grad
    c, n, info, dp = _cl(mod, dpos=[[-.3, -.3, -.3, 0], [0] * 4], **NODE_TOY)
    assert abs(float(n) - 0.375) < 1e-12, float(n)


def _check_curve_hand(mod):
    """the curve loss by hand on a straight GT lane (nodes 2 m apart along +x, the GT curve is the line y = 0 with the normal +y) with the tokens of the middle two 0.2 m to the left, link (1, 2, pre 0, suc 3): the error of the Catmull-Rom curve is
    .2 + .1 t - .1 t^2 (so rho = e^2 below the knee): mean over t of its square; a missing ghost (pre -1, or its token missing) is reflected: .2 + .1 t^2 - .1 t^3; both ghosts missing: .2; a token of u or v missing: no link; dpos = -0.2 on the middle
    tokens cancels it; a GT tangent of length 0 at t = 0 (P0 = P2) drops that sample; the chord rule: chords 2 / 2 / 9 and 2 / 2 / 5 (ratio 2.5) and 12 / 2 / 2 left out, 2 / 2 / 2 and 2 / 2 / 4 (ratio 2) in, counted in n_links / n_excl"""
    base = dict(pos=[[[0, 0], [2, .2], [4, .2], [6, 0]]], dirv=[[[1, 0]] * 4], dpos=[[0] * 4], gpos=[[[0, 0], [2, 0], [4, 0], [6, 0]]], gdir=[[[1, 0]] * 4], btok=[[0, 1, 2, 3]], gl=[[[1, 2, 0, 3]]])
    ms = lambda f: float(np.mean(np.square([f(t) for t in TS])))  # noqa: E731
    both, one = ms(lambda t: .2 + .1 * t - .1 * t * t), ms(lambda t: .2 + .1 * t * t - .1 * t ** 3)
    for kw, want, n_cur in ((dict(), both, 5), (dict(gl=[[[1, 2, -1, 3]]]), one, 5), (dict(btok=[[-1, 1, 2, 3]]), one, 5), (dict(gl=[[[1, 2, -1, -1]]]), ms(lambda t: .2), 5),
                            (dict(dpos=[[0, -.2, -.2, 0]]), 0.0, 5)):
        c, n, info, _ = _cl(mod, **dict(base, **kw))
        assert abs(float(c) - want) < 1e-12 and info["n_curve"] == n_cur and info["n_gl"] == 1 and info["n_links"] == 1 and info["n_excl"] == 0, (kw, float(c), want, info)
    for kw in (dict(btok=[[0, -1, 2, 3]]), dict(btok=[[0, 1, -1, 3]]), dict(gl=[[[-1, 2, 0, 3]]]), dict(gl=[[[-1] * 4]])):          # u or v without a token, a padded link: no sample, a finite zero WITH a graph
        c, n, info, dp = _cl(mod, **dict(base, **kw))
        assert float(c) == 0.0 and info["n_curve"] == 0 and info["n_links"] == 0 and info["n_gl"] == (0 if "gl" in kw else 1) and c.requires_grad and math.isfinite(float(n)), (kw, info)
        c.backward()
        assert float(dp.grad.abs().sum()) == 0.0
    z = dict(base, gpos=[[[4, 0], [2, 0], [4, 0], [6, 0]]], pos=[[[4, 0], [2, 0], [4, 0], [6, 0]]])                                    # P0 = P2: the GT tangent (P2 - P0) / 2 is 0 at t = 0
    c, n, info, _ = _cl(mod, **z)
    assert info["n_curve"] == 4 and float(c) == 0.0, info
    for last, used in ((2, True), (4, True), (5, False), (9, False)):                                                                  # the last chord
        gp = [[[0, 0], [2, 0], [4, 0], [4 + last, 0]]]
        c, n, info, _ = _cl(mod, **dict(base, pos=gp, gpos=gp))
        assert (info["n_curve"], info["n_links"], info["n_excl"]) == ((5, 1, 0) if used else (0, 1, 1)), (last, info)
    gp = [[[0, 0], [2, 0], [14, 0], [16, 0]]]                                                                                          # chord_max: the link itself 12 m, the ratios 6
    assert _cl(mod, **dict(base, pos=gp, gpos=gp))[2]["n_excl"] == 1 and _cl(mod, **dict(base, pos=gp, gpos=gp), chord_max=20.0, ratio_max=7.0)[2]["n_excl"] == 0
    gp = [[[0, 0], [10, 0], [20, 0], [30, 0]]]                                                                                         # equal chords of 10 m: out by chord_max alone
    assert _cl(mod, **dict(base, pos=gp, gpos=gp))[2]["n_excl"] == 1
    # a frame whose links are all left out adds nothing: the mean over the frames is the other frame's, the batch of such frames is a finite zero with a graph
    two = dict(pos=base["pos"] * 2, dirv=base["dirv"] * 2, dpos=[[0] * 4] * 2, gpos=[base["gpos"][0], [[0, 0], [2, 0], [4, 0], [13, 0]]], gdir=base["gdir"] * 2, btok=base["btok"] * 2, gl=base["gl"] * 2)
    c, n, info, _ = _cl(mod, **two)
    assert abs(float(c) - both) < 1e-12 and info["n_curve"] == 5 and info["n_gl"] == 2 and info["n_links"] == 2 and info["n_excl"] == 1, (float(c), info)
    c, n, info, dp = _cl(mod, **dict(two, gpos=[two["gpos"][1]] * 2))
    assert float(c) == 0.0 and info["n_curve"] == 0 and info["n_excl"] == 2 and c.requires_grad and math.isfinite(float(c)), info
    c.backward()


def _ref_case(seed, B=3, K=12, Ng=9):
    """a random batch for `curve_losses`: per frame a bent GT lane (steps of 1 - 4 m, so that the chord rule leaves some links out), a random subset of its nodes with a token near them (the others missing), random other tokens, direction vectors
    of random length, links along the lane + one fork link, ghosts dropped at random, a padded link row; tokens with offsets (dpos)"""
    rng = np.random.RandomState(seed)
    out = dict(pos=[], dirv=[], gpos=[], gdir=[], btok=[], gl=[])
    for _ in range(B):
        th = rng.uniform(0, 6.28) + 0.35 * np.arange(Ng)
        gd = np.stack([np.cos(th), np.sin(th)], 1)
        gp = rng.uniform(-20, 20, 2) + np.concatenate([[[0.0, 0.0]], np.cumsum(rng.uniform(1.0, 4.0, (Ng - 1, 1)) * gd[:-1], 0)])
        perm = rng.permutation(K)[:Ng]
        tp, td = rng.uniform(-30, 30, (K, 2)), gd[rng.randint(0, Ng, K)].copy()
        tp[perm], td[perm] = gp + rng.normal(0, 0.4, (Ng, 2)), gd + rng.normal(0, 0.2, (Ng, 2))
        td = td / np.linalg.norm(td, axis=1, keepdims=True) * rng.uniform(0.5, 2.0, (K, 1))
        rows = [(g, g + 1, g - 1 if g > 0 else -1, g + 2 if g + 2 < Ng else -1) for g in range(Ng - 1)] + [(2, 6, 1, 7)]
        rows = [(u, v, p if rng.rand() > 0.25 else -1, s if rng.rand() > 0.25 else -1) for u, v, p, s in rows] + [(-1, -1, -1, -1)]
        for k, v in zip(out, (tp, td, gp, gd * rng.uniform(0.5, 2.0, (Ng, 1)), np.where(rng.rand(Ng) < 0.25, -1, perm), np.array(rows))):
            out[k].append(v.tolist())
    out["dpos"] = rng.normal(0, 0.3, (B, K)).tolist()
    return out


def _check_curve_ref(mod):
    """`curve_losses` against the independent numpy form on random batches (float64, 1e-9; beta 0.5 / 0.3, t in five / three values, the chord rule at its defaults and tightened), the counts, and the gradient against finite differences"""
    seen = []
    for seed, kw in ((0, {}), (1, {}), (2, dict(beta=0.3, ts=(0.0, 0.5, 1.0))), (3, dict(chord_max=2.5, ratio_max=1.5))):
        case = _ref_case(seed)
        c, n, info, dp = _cl(mod, **case, **kw)
        pos, dirv = np.array(case["pos"]), np.array(case["dirv"])
        nrm = np.stack([-dirv[..., 1], dirv[..., 0]], -1) / np.linalg.norm(dirv, axis=-1, keepdims=True)
        ref = _np_losses(pos + np.array(case["dpos"])[..., None] * nrm, np.array(case["gpos"]), np.array(case["gdir"]), np.array(case["btok"]), np.array(case["gl"]), **kw)
        assert abs(float(c) - ref[0]) < 1e-9 and abs(float(n) - ref[1]) < 1e-9 and (info["n_curve"], info["n_links"], info["n_excl"], info["n_gl"]) == ref[2:], (seed, float(c), float(n), info, ref)
        assert info["n_curve"] > 10 and info["n_links"] > 5, (seed, info)
        seen.append((info["n_excl"], info["n_links"]))
        g = {k: torch.tensor(np.array(v), dtype=torch.float64 if k in ("gt_pos", "gt_dir") else torch.long) for k, v in (("gt_pos", case["gpos"]), ("gt_dir", case["gdir"]), ("gt_btok", case["btok"]), ("gl", case["gl"]))}
        d0 = torch.tensor(case["dpos"], dtype=torch.float64).requires_grad_()
        assert torch.autograd.gradcheck(lambda d: mod.curve_losses(d, torch.tensor(pos), torch.tensor(dirv), g, **kw)[0], (d0,), eps=1e-6, atol=1e-6), seed
        assert torch.autograd.gradcheck(lambda d: mod.curve_losses(d, torch.tensor(pos), torch.tensor(dirv), g, **kw)[1], (d0,), eps=1e-6, atol=1e-6), seed
    assert all(0 < e < n for e, n in seen), f"the chord rule must leave some links out and keep some in every case: {seen}"


def _check_cut(mod):
    """no gradient reaches the tokens' pos / dirv through `refined_pos` or `curve_losses` (they are the head's INPUTS); dpos gets one"""
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    pos, dirv, dpos = t([[[0., .3], [2., .3]]]).requires_grad_(), t([[[1., 0.], [0.6, 0.8]]]).requires_grad_(), t([[.1, -.2]]).requires_grad_()
    mod.refined_pos(pos, dirv, dpos).sum().backward()
    assert pos.grad is None and dirv.grad is None and dpos.grad is not None
    g = dict(gt_pos=t([[[0., 0.], [2., 0.]]]), gt_dir=t([[[1., 0.], [1., 0.]]]), gt_btok=torch.tensor([[0, 1]]), gl=torch.tensor([[[0, 1, -1, -1]]]))
    c, n, _ = mod.curve_losses(dpos, pos, dirv, g)
    (c + n).backward()
    assert pos.grad is None and dirv.grad is None


def _check_pos_init(mod):
    """a position head starts at zero: dpos exactly 0 for every token in train and eval mode, whatever the inputs; the zero last Linear (weight and bias)"""
    torch.manual_seed(0)
    m = mod.JTokenHead(pos_head=True)
    assert all(float(p.abs().sum()) == 0.0 for p in m.posh[2].parameters()) and float(m.posh[0].weight.abs().sum()) > 0
    for train in (True, False):
        out = m.train(train)(_batch(2))
        assert out["dpos"].shape == out["valid"].shape and bool((out["dpos"] == 0).all())


def _check_cap(mod):
    """the cap of the offset: dpos = 3 m x tanh(posh / 3 m). With the weights of posh's last Linear at 0 its output is its bias for every token (padding too): bias 10 gives 3 tanh(10 / 3) = 2.99237, bias -1 gives -3 tanh(1 / 3) = -0.96454, bias 0 gives 0; |dpos| < 3"""
    assert abs(3.0 * math.tanh(10.0 / 3.0) - 2.99237) < 1e-5 and abs(-3.0 * math.tanh(1.0 / 3.0) + 0.964538) < 1e-6 and mod.POS_CAP == 3.0
    for bias in (10.0, -1.0, 0.0):
        torch.manual_seed(0)
        m = mod.JTokenHead(pos_head=True).eval()
        with torch.no_grad():
            m.posh[2].weight.zero_()
            m.posh[2].bias.fill_(bias)
            d = m(_batch(2))["dpos"]
        assert float((d - 3.0 * math.tanh(bias / 3.0)).abs().max()) < 1e-6 and float(d.abs().max()) < 3.0, (bias, float(d.min()), float(d.max()))


def _pos_head(rounds=0, step=0, perturb=False, mod=JK):
    """a seeded head of `mod` with the position head; perturb: its last layer moved off zero (at the zero start the first Linear and the trunk get no gradient from dpos)"""
    torch.manual_seed(0)
    m = mod.JTokenHead(rounds=rounds, pos_head=True)
    m.set_step(step)
    if perturb:
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for p in m.posh[2].parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return m


class _NoGeom(dict):
    def __getitem__(self, k):
        if k in JK.GT_KEYS or k in LJK.GEOM_KEYS or k.startswith("gt_"):
            raise KeyError(f"the forward read the GT key {k}")
        return super().__getitem__(k)


def test_pos_head():
    """E395 `JTokenHead(pos_head=)`: off = the champion bit for bit (the recorded numbers of git HEAD before E392 (`CHAMP_HEAD`), no `posh`, no `dpos`); on: the extra parameters are posh's alone (4225), the champion's (also with rounds) initialise as without it for the same
    seed, the state keys of posh come last; dpos is exactly 0 at the start (train and eval); the forward needs no GT key and ignores a poisoned one; a gradient from the curve loss reaches posh and the trunk (at the zero start only the last Linear) and never
    the inputs pos / dirv; the cap (dpos = 3 tanh(posh / 3): bias 10 gives 2.99237, -1 gives -3 tanh(1 / 3)); textual mutants (no zero init, a gradient into pos or dirv, no cap factor, the cap outside only, no tanh) fail"""
    for step in (0, JK.SWITCH + 5):
        torch.manual_seed(0)
        h = JK.JTokenHead(pos_head=False)
        h.set_step(step)
        got = _champ_numbers(JK, h)
        assert all(abs(g - w) <= 1e-9 * abs(w) for g, w in zip(got, CHAMP_HEAD[step])) and not any(k.startswith("posh") for k in h.state_dict()) and JK.count_params(h) == 127131, (step, got)
        assert "dpos" not in h(_batch(2))
    for R in (0, 2):
        torch.manual_seed(0)
        ch = JK.JTokenHead(rounds=R)
        ph = _pos_head(R)
        sc, sp = ch.state_dict(), ph.state_dict()
        assert list(sp)[:len(sc)] == list(sc) and all(k.startswith("posh.") for k in list(sp)[len(sc):]) and all(torch.equal(v, sp[k]) for k, v in sc.items() if k != "step"), "posh changed the init of an old parameter"
        assert JK.count_params(ph) - JK.count_params(ch) == 64 * 64 + 64 + 64 + 1 and not any(float(p.abs().sum()) for p in ph.posh[2].parameters())
        b = _batch(2)
        assert torch.equal(ph(b)["logp"], ch(b)["logp"]), "a position head at the zero start changes the wiring"
    _check_pos_init(JK)
    clean = {k: v for k, v in _batch(2).items() if k in JK.INPUTS}
    m = _pos_head(step=JK.SWITCH + 5, perturb=True).eval()
    bg = JK.synthetic_batch(2, 48, seed=0, geom=True)
    with torch.no_grad():
        o = m(_NoGeom(clean))
        assert float(o["dpos"].abs().max()) > 0.1 and torch.equal(m(dict(bg, gt_pos=bg["gt_pos"] * 7 + 3, gt_btok=bg["gt_btok"].flip(-1), gl=-bg["gl"]))["dpos"], o["dpos"]), "the forward depends on a GT key"
        assert float(o["dpos"].abs().max()) <= JK.POS_CAP
    x = {k: (v.clone().requires_grad_(True) if k in JK.INPUTS and v.is_floating_point() else v) for k, v in bg.items()}
    out = m.train()(x)
    LJK.curve_losses(out["dpos"], x["pos"], x["dirv"], {k: x[k] for k in LJK.GEOM_KEYS}, valid=out["valid"])[0].backward()
    got = {n: float(p.grad.abs().sum()) for n, p in m.named_parameters() if p.grad is not None}
    assert all(got.get(n, 0.0) > 0 for n in ("posh.0.weight", "posh.2.weight", "posh.2.bias", "lnw.weight", "proj.weight", "layers.0.qkv.weight", "layers.2.ffn.0.weight")), got
    assert all(x[k].grad is None for k in JK.INPUTS if x[k].is_floating_point()), "a gradient reached a cached input"
    m0 = _pos_head()
    out = m0(bg)
    LJK.curve_losses(out["dpos"], bg["pos"], bg["dirv"], {k: bg[k] for k in LJK.GEOM_KEYS})[0].backward()
    assert float(m0.posh[2].weight.grad.abs().sum()) > 0 and float(m0.posh[0].weight.grad.abs().sum()) == 0.0 == float(m0.layers[0].qkv.weight.grad.abs().sum()), "at the zero start only the last Linear gets a gradient"
    mut = {"no zero init": ([("nn.init.zeros_(self.posh[2].weight)", "pass")], _check_pos_init), "a gradient into dirv": ([("pos, dirv = _const(pos), _const(dirv)", "pos = _const(pos)")], _check_cut),
           "a gradient into pos": ([("pos, dirv = _const(pos), _const(dirv)", "dirv = _const(dirv)")], _check_cut)}
    cap = 'out["dpos"] = self.pos_cap * torch.tanh(self.posh(xn)[..., 0].float() / self.pos_cap)'
    mut.update({"tanh(posh) without the cap factor": ([(cap, 'out["dpos"] = torch.tanh(self.posh(xn)[..., 0].float())')], _check_cap),
                "the cap outside the tanh only": ([(cap, 'out["dpos"] = self.pos_cap * torch.tanh(self.posh(xn)[..., 0].float())')], _check_cap),
                "raw posh without tanh": ([(cap, 'out["dpos"] = self.posh(xn)[..., 0].float()')], _check_cap)})
    _check_cut(JK)
    _check_cap(JK)
    for edits, check in mut.values():
        must_fail(check, _mutant(*edits))
    print(f"pos_head: off = the champion of git HEAD (recorded numbers, 127131 parameters, no posh, no dpos); on: +4225 parameters (the others as without it, also with rounds 2, posh's keys last, the wiring unchanged at the start), dpos exactly 0 at "
          f"the start, the forward reads no GT key, a gradient from the curve loss reaches posh and the trunk (the zero start: the last Linear only) and no input; mutants ({', '.join(mut)}) fail: ok")


def test_curve_loss():
    """E395 `curve_losses` / `cr_points_t`: the Catmull-Rom segment by hand (passes through P1 and P2, the line for collinear equal steps, the tangent at P1 = (P2 - P0) / 2 and its derivative form); the node loss by hand (rho(0.3) = 0.09, the knee 1.0 -> 0.75, per frame, a node without
    a token left out, the gradient), dpos = -0.3 puts the node on its GT line; the curve loss by hand on a straight lane (ghosts available, reflected, a token missing, both missing, a missing end, no link = a finite zero with a graph, the zero-tangent sample, the chord rule: 2 / 2 / 9
    and 2 / 2 / 5 m out, 2 / 2 / 2 and 2 / 2 / 4 in, a frame of left-out links adds nothing); against an independent numpy implementation (basis matrix) on random curved batches with missing ghosts, missing tokens and a fork; textual mutants fail"""
    t = torch.tensor
    P0, P1, P2, P3 = (t([[x, y]], dtype=torch.float64) for x, y in ((-1.0, 0.5), (0.0, 0.0), (2.0, 1.0), (5.0, -1.0)))
    ts = t(TS, dtype=torch.float64)
    c = LJK.cr_points_t(P0, P1, P2, P3, ts)
    assert c.shape == (1, 5, 2) and torch.equal(c[0, 0], P1[0]) and torch.allclose(c[0, 4], P2[0], atol=1e-15), "the curve passes through P1 and P2"
    P = np.concatenate([x.numpy() for x in (P0, P1, P2, P3)])
    assert np.allclose(c[0].numpy(), np.stack([np.array([1, x, x * x, x ** 3]) @ CR_M @ P for x in TS]), atol=1e-14), "cr_points_t is the Catmull-Rom basis matrix"
    tan = LJK.cr_tangent_t(P0, P1, P2, P3, ts)[0]
    assert np.allclose(tan[0].numpy(), 0.5 * (P2 - P0)[0].numpy()) and np.allclose(tan[4].numpy(), 0.5 * (P3 - P1)[0].numpy()), "the tangent at P1 is (P2 - P0) / 2, at P2 (P3 - P1) / 2"
    h = 1e-6
    fd = (LJK.cr_points_t(P0, P1, P2, P3, ts + h) - LJK.cr_points_t(P0, P1, P2, P3, ts - h))[0] / (2 * h)
    assert float((fd - tan).abs().max()) < 1e-7, "cr_tangent_t is the derivative of cr_points_t"
    L = [t([[x * 2.0, 0.0]], dtype=torch.float64) for x in range(4)]
    assert float(LJK.cr_points_t(*L, ts)[0, :, 1].abs().max()) == 0.0 and np.allclose(LJK.cr_points_t(*L, ts)[0, :, 0].numpy(), 2.0 + 2.0 * np.array(TS)), "collinear equal steps: the straight line, uniform in t"
    for check in (_check_node_hand, _check_curve_hand, _check_curve_ref):
        check(JK)
    mut = {"the sign of n": ([("return pos + dpos[..., None] * _unit(_rot90(dirv), 1e-9)", "return pos - dpos[..., None] * _unit(_rot90(dirv), 1e-9)")], _check_node_hand),
           "ghost availability not shared between the curves": ([("G4 = four(gp, a_pre, a_suc)", "G4 = four(gp, pre >= 0, suc >= 0)")], _check_curve_ref),
           "a missing ghost not reflected": ([("2 * P1 - P2), P1, P2", "P1), P1, P2")], _check_curve_hand),
           "the knee on the squared error (node)": ([("_rho(e_node, beta)", "_rho(e_node * e_node, beta)")], _check_node_hand),
           "the knee on the squared error (curve)": ([("_rho(e_cur, beta)", "_rho(e_cur * e_cur, beta)")], _check_curve_hand),
           "no chord rule": ([("use = ok & (ch[..., 1] <= chord_max) & (ch.amax(-1) <= ratio_max * ch.amin(-1))", "use = ok")], _check_curve_hand),
           "the ratio rule only": ([("use = ok & (ch[..., 1] <= chord_max) & (ch.amax(-1)", "use = ok & (ch.amax(-1)")], _check_curve_hand),
           "the chord rule on c1 only": ([("& (ch.amax(-1) <= ratio_max * ch.amin(-1))", "")], _check_curve_hand),
           "a zero tangent kept": ([("(tan.norm(dim=-1) >= 1e-6)", "(tan.norm(dim=-1) >= -1.0)")], _check_curve_hand),
           "a link without both tokens used": ([("ok, a_pre, a_suc = has(u) & has(v), has(pre), has(suc)", "ok, a_pre, a_suc = (u >= 0) & (v >= 0), has(pre), has(suc)")], _check_curve_hand),
           "n_gl counts the links with both nodes": ([("n_gl=float((u >= 0).sum())", "n_gl=float(ok.sum())")], _check_curve_hand)}
    for edits, check in mut.values():
        must_fail(check, _mutant(*edits))
    print(f"curve_loss: Catmull-Rom (through P1 / P2, the tangent, the line, the basis matrix), the node loss by hand (0.42, 0.375, the gradient), the curve loss by hand (ghosts, reflection, missing tokens, empty = a finite zero with a graph, the chord rule, a frame of left-out links adds nothing), "
          f"against numpy on random batches (1e-9, gradcheck), mutants ({', '.join(mut)}) fail: ok")


def test_pos_loss_weights():
    """E395 `jtoken_loss(curve_w=, node_w=)`: 0 (the default) = the call without them, the value bit for bit and the info keys the same, also with a position head and the geometry keys in the batch; on the synthetic geometry toy known counts (3 frames x 28 links, node 5 without a
    token: 26 with both ends, the long link 9 -> 10 and its two neighbours left out by the chord rule: 23 x 5 samples), the loss = the base + W x curve (+ V x node), the info keys, dpos_abs / dpos_cap by hand; refused without dpos or without the geometry keys; `synthetic_batch(geom=True)`
    leaves every other key as geom False"""
    b0, bg = JK.synthetic_batch(3, 48, seed=2), JK.synthetic_batch(3, 48, seed=2, geom=True)
    nn_ = lambda x: torch.nan_to_num(x.double(), nan=-777.0)  # noqa: E731   (yscale holds NaN, and torch.equal says NaN != NaN)
    assert set(bg) == set(b0) | set(LJK.GEOM_KEYS) and all(torch.equal(nn_(bg[k]), nn_(b0[k])) for k in b0), "geom=True changes the other keys"
    assert [tuple(bg[k].shape) for k in LJK.GEOM_KEYS] == [(3, 32, 2), (3, 32, 2), (3, 32), (3, 30, 4)] and bg["gt_btok"].dtype == torch.int64 and (bg["gt_btok"][:, 5] == -1).all() and (bg["gl"][:, 28:] == -1).all()
    m = _pos_head(step=JK.SWITCH + 5, perturb=True)
    out = m(bg)
    base = LJK.jtoken_loss(out, bg, none_weight=0.25)
    for kw in (dict(), dict(curve_w=0.0, node_w=0.0), dict(curve_w=0, node_w=0)):
        l, i = LJK.jtoken_loss(out, bg, none_weight=0.25, **kw)
        assert torch.equal(l, base[0]) and i == base[1] and "curve" not in i, kw
    ch = JK.JTokenHead()
    ch.set_step(JK.SWITCH + 5)
    assert torch.equal(LJK.jtoken_loss(ch(b0), b0)[0], LJK.jtoken_loss(ch(b0), b0, curve_w=0.0, node_w=0.0)[0]) and "dpos" not in ch(b0)
    lc, ic = LJK.jtoken_loss(out, bg, none_weight=0.25, curve_w=0.5)
    ln, inn = LJK.jtoken_loss(out, bg, none_weight=0.25, node_w=0.25)
    lb, ib = LJK.jtoken_loss(out, bg, none_weight=0.25, curve_w=0.5, node_w=0.25)
    new = {"curve", "node", "n_curve", "n_gl", "n_links", "n_excl", "dpos_abs", "dpos_cap"}
    assert set(ic) == set(base[1]) | new == set(inn) == set(ib), set(ic) ^ set(base[1])
    assert abs(float(lc) - float(base[0]) - 0.5 * ic["curve"]) < 1e-5 and abs(float(ln) - float(base[0]) - 0.25 * inn["node"]) < 1e-5 and abs(float(lb) - float(base[0]) - 0.5 * ib["curve"] - 0.25 * ib["node"]) < 1e-5
    assert ic["curve"] > 0 and ic["node"] > 0 and (ic["n_gl"], ic["n_links"], ic["n_excl"], ic["n_curve"]) == (84.0, 78.0, 9.0, 345.0), ic
    dp, vd = out["dpos"].detach().float(), out["valid"]
    assert abs(ic["dpos_abs"] - float(dp.abs()[vd].mean())) < 1e-6 and abs(ic["dpos_cap"] - float((dp.abs()[vd] >= 0.99 * JK.POS_CAP).float().mean())) < 1e-9 and 0 < ic["dpos_abs"] < JK.POS_CAP
    must_fail(LJK.jtoken_loss, {k: v for k, v in out.items() if k != "dpos"}, bg, curve_w=0.5)                                  # a head without a position head
    must_fail(LJK.jtoken_loss, out, {k: v for k, v in bg.items() if k != "gl"}, node_w=0.5)                                   # a batch without the geometry keys
    print(f"pos_loss_weights: curve_w = node_w = 0 is the call without them (value and info keys, with a position head and with the champion), geom=True leaves the other keys alone; the synthetic toy has {ic['n_links']:.0f} links with both ends, {ic['n_excl']:.0f} left out by the chord rule, "
          f"{ic['n_curve']:.0f} samples; loss = base + W x curve (+ V x node), the info keys, dpos_abs / dpos_cap by hand; refused without dpos or the geometry keys: ok")


def test_dump_dpos():
    """E395 `jtoken_eval.batch_dump`: a head with a position head adds the key `dpos` ((K,) float32 = the head's offsets) to every frame and changes no other value; the dump of a head without one has the keys of before, exactly"""
    import jtoken_eval as JE
    bg = JK.synthetic_batch(3, 48, seed=1, geom=True)
    bb = dict(token=[f"t{i}" for i in range(3)], row=torch.arange(3), tokens_np=[dict(cell=np.arange(48)) for _ in range(3)])
    with torch.no_grad():
        out = _pos_head(step=JK.SWITCH + 5, perturb=True).eval()(bg)
        fr, fo = JE.batch_dump(out, bb), JE.batch_dump({k: v for k, v in out.items() if k != "dpos"}, bb)
    keys = ["argmax", "cand", "cell", "keep", "none_cls", "pnone", "ptype", "rec", "row", "token"]
    same = lambda a, b: all(same(a[k], b[k]) for k in a) if isinstance(a, dict) else (np.array_equal(a, b) and np.asarray(a).dtype == np.asarray(b).dtype)  # noqa: E731
    assert all(sorted(f) == keys for f in fo) and all(sorted(f) == sorted(keys + ["dpos"]) for f in fr), (sorted(fo[0]), sorted(fr[0]))
    for i in range(3):
        assert fr[i]["dpos"].dtype == np.float32 and fr[i]["dpos"].shape == (48,) and np.array_equal(fr[i]["dpos"], out["dpos"][i].numpy()) and float(np.abs(fr[i]["dpos"]).max()) > 0.1
        assert same({k: v for k, v in fr[i].items() if k != "dpos"}, fo[i]), "the dump of a head with a position head differs in another key"
    print("dump_dpos: batch_dump adds dpos (K,) float32 per frame for a head with a position head, nothing else changes; without one the keys are those of before: ok")


def _check_pos_trainer(mod):
    """`mod.train` (jtoken_train or a textual mutant of it) with --pos-head --curve-w 0.5: the smoke run on the synthetic geometry builds the position head and puts the curve term into the loss (its info in the last log row), the weights are in the recipe (another
    curve weight, or the node loss, in the folder is refused), and a non-smoke run asks the loader for geom=True exactly when a loss weight is given (four cases); the fake loader returns synthetic batches"""
    argv = ["--name", "p", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--pos-head", "--curve-w", "0.5"]
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        mod.train(mod.parse_args(argv), runs, 500, 48)
        row = [json.loads(r) for r in open(os.path.join(runs, "p", "log.jsonl")) if "total" in r][-1]
        state = torch.load(os.path.join(runs, "p", "model.pt"), map_location="cpu", weights_only=False)["state"]
        assert any(k.startswith("posh") for k in state) and "curve" in row and math.isfinite(row["total"]), row
        recipe = lambda extra: dict(vars(mod.parse_args(argv[:9] + extra)), steps=20, distill_sha256=None)  # noqa: E731
        mod.check_start(recipe(["--pos-head", "--curve-w", "0.5"]), os.path.join(runs, "p"))                                       # the same recipe may resume
        for other in (["--pos-head", "--curve-w", "0.25"], ["--pos-head", "--node-w", "0.5"], ["--pos-head"], []):
            assert _refused(mod.check_start, recipe(other), os.path.join(runs, "p")), f"another recipe in the same folder was accepted: {other}"
        os.makedirs(os.path.join(runs, "teacher"))
        open(os.path.join(runs, "teacher", "model.pt"), "wb").close()                                                              # a teacher folder that passes the existence check of --distill-from
        dargs = ["--name", "d", "--distill-from", os.path.join(runs, "teacher")]
        mod.parse_args(dargs)
        mod.parse_args(dargs + ["--pos-head"])                                                                                      # no position loss: the teachers are fine
        for w in ("--curve-w", "--node-w"):
            try:
                mod.parse_args(dargs + ["--pos-head", w, "0.5"])
            except AssertionError as e:
                assert "--distill-from" in str(e), str(e)
            else:
                raise AssertionError(f"--distill-from with {w} accepted")
        seen = {}

        def fake_loader(split, seed, bs, workers, **k):
            seen.update(k)
            return None, [JK.synthetic_batch(bs, 48, seed=s, geom=bool(k.get("geom"))) for s in range(3)]
        for name, extra, want in (("real_c", ["--pos-head", "--curve-w", "0.5"], True), ("real_n", ["--pos-head", "--node-w", "0.5"], True), ("real_0", ["--pos-head"], False), ("real_champ", [], False)):
            sys.modules["lane_ptok_data"] = types.SimpleNamespace(loader=fake_loader)
            seen.clear()
            try:
                mod.train(mod.parse_args(["--name", name, "--steps", "4", "--bs", "2", "--workers", "0"] + extra), runs=runs, ckpt_every=3)
            finally:
                del sys.modules["lane_ptok_data"]
            assert seen == dict(ident="e292_a", K=None, **({"geom": True} if want else {})), (name, seen)


def test_pos_head_trainer():
    """E395 `--pos-head` / `--curve-w` / `--node-w` of `jtoken_train`: defaults (absent in an old run's args) = off; refused without --pos-head, together, negative; the smoke run with --pos-head --curve-w 0.5 / --node-w 0.5 (synthetic geometry) trains finite
    and moves the position head (and a head without a loss keeps dpos at 0), the args / recipe / log hold the options, `load_model` rebuilds the head with posh, another recipe in the folder is refused; the real loader is asked for geom=True exactly then; the stage-2
    trainer (`jlive_train.resolve`) refuses a head run with a position head"""
    import jlive_train as JL
    assert JKT.ARCH_DEFAULTS["pos_head"] is False and JKT.ARCH_DEFAULTS["curve_w"] == JKT.ARCH_DEFAULTS["node_w"] == 0.0 and {"pos_head", "curve_w", "node_w"} <= set(JKT.RECIPE)
    assert JKT.build_head({}).pos_head is False and JKT.build_head(dict(pos_head=True)).pos_head is True
    for bad in (["--curve-w", "0.5"], ["--node-w", "0.5"], ["--pos-head", "--curve-w", "0.5", "--node-w", "0.5"], ["--pos-head", "--curve-w", "-1"], ["--pos-head", "--node-w", "-0.1"]):
        must_fail(JKT.parse_args, ["--name", "x"] + bad)
    JKT.parse_args(["--name", "x", "--pos-head"])
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        base = ["--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "0.25"]
        infos, real, now = {}, LJK.jtoken_loss, [None]

        def recording(out, batch, **k):                                                                                            # the loss info of every step of every run
            r = real(out, batch, **k)
            infos[now[0]].append((float(r[0]), r[1]))
            return r
        for name, extra in (("champ", []), ("curve", ["--pos-head", "--curve-w", "0.5"]), ("node", ["--pos-head", "--node-w", "0.5"]), ("noloss", ["--pos-head"])):
            now[0], infos[name] = name, []
            patched(LJK, "jtoken_loss", recording, JKT.train, JKT.parse_args(["--name", name] + base + extra), runs, 500, 48)
        rows = {n: [json.loads(r) for r in open(os.path.join(runs, n, "log.jsonl")) if "total" in r] for n in infos}
        assert all(len(v) == 20 and all(math.isfinite(l) for l, _ in v) for v in infos.values()) and "curve" not in infos["champ"][0][1] and "curve" not in infos["noloss"][0][1]
        assert "curve" in rows["curve"][-1] and "node" in rows["node"][-1] and "curve" not in rows["champ"][-1] and all(k in infos["curve"][0][1] for k in ("curve", "node", "n_curve", "n_gl", "n_links", "n_excl", "dpos_abs", "dpos_cap"))
        assert (infos["curve"][0][1]["n_curve"], infos["curve"][0][1]["n_gl"], infos["curve"][0][1]["n_links"], infos["curve"][0][1]["n_excl"]) == (2 * 23 * 5, 2 * 28, 2 * 26, 2 * 3) and infos["curve"][0][1]["dpos_abs"] == 0.0 == infos["node"][0][1]["dpos_abs"], "the first step starts at dpos 0 on the toy geometry"
        assert infos["curve"][-1][1]["dpos_abs"] > 0 and infos["node"][-1][1]["dpos_abs"] > 0 and infos["curve"][-1][0] != infos["node"][-1][0], "the position head did not move"
        ck = {n: torch.load(os.path.join(runs, n, "model.pt"), map_location="cpu", weights_only=False) for n in infos}
        assert not any(k.startswith("posh") for k in ck["champ"]["state"]) and all(ck[n]["args"]["pos_head"] and any(k.startswith("posh") for k in ck[n]["state"]) for n in ("curve", "node", "noloss"))
        assert (ck["curve"]["args"]["curve_w"], ck["curve"]["args"]["node_w"], ck["node"]["args"]["node_w"], ck["champ"]["args"]["pos_head"]) == (0.5, 0.0, 0.5, False)
        assert all(float(v.abs().sum()) == 0.0 for k, v in ck["noloss"]["state"].items() if k.startswith("posh.2")), "a position head without a loss moved"
        assert any(float(v.abs().sum()) > 0.0 for k, v in ck["curve"]["state"].items() if k.startswith("posh.2")) and rows["curve"][-1]["total"] != rows["champ"][-1]["total"]
        m, args = JKT.load_model("curve", "cpu", runs)
        assert m.pos_head and all(torch.equal(v, ck["curve"]["state"][k]) for k, v in m.state_dict().items()) and "dpos" in m(JK.synthetic_batch(2, 48, seed=1))
        log = open(os.path.join(runs, "curve", "train.log"), encoding="utf-8").read()
        assert "position head (E395) with curve loss weight 0.5, node loss weight 0.0" in log and "position head" not in open(os.path.join(runs, "champ", "train.log"), encoding="utf-8").read()
        sys.modules["lane_ptok_data"] = types.SimpleNamespace(loader=lambda split, seed, bs, workers, **k: (None, [JK.synthetic_batch(bs, 48, seed=s, geom=True) for s in range(3)]))
        try:
            JKT.train(JKT.parse_args(["--name", "clock", "--steps", "50", "--bs", "2", "--workers", "0", "--pos-head", "--curve-w", "0.5"]), runs=runs, ckpt_every=25)
        finally:
            del sys.modules["lane_ptok_data"]
        clock = open(os.path.join(runs, "clock", "train.log"), encoding="utf-8").read()                                            # the clock line of step 50 holds the position head's numbers
        assert "position head (E395) curve" in clock and "left out by the chord rule" in clock and "mean |dpos|" in clock, clock[-800:]
        _check_pos_trainer(JKT)
        # the stage-2 trainer takes the head run's options from its model.pt: a position head is refused (it neither builds nor trains it)
        for name in ("champ", "curve"):
            refused = None
            try:
                JL.resolve(JL.parse_args(["--name", "x", "--seed", "0", "--head-run", name, "--train-nodes", "1"]), runs)
            except AssertionError as e:
                refused = str(e)
            except FileNotFoundError:                                                                                              # passed the head run's checks: it wants B''s model.pt (not in this toy)
                pass
            assert (refused is not None and "position head" in refused) == (name == "curve") and (refused is None or name == "curve"), (name, refused)
    muts = {"the head is built without the position head": [('pos_head=bool(args.get("pos_head", ARCH_DEFAULTS["pos_head"])))', "pos_head=False)")],
            "the loss weights are not passed to the loss": [("wire_w=w, curve_w=curve_w, node_w=node_w)", "wire_w=w)")],
            "the curve weight is not in the recipe": [('"msg", "pos_head", "curve_w", "node_w")', '"msg", "pos_head", "node_w")')],
            "the loader is not asked for the geometry": [('**({"geom": True} if geom else {}))', "**{})")],
            "the loader is always asked for the geometry": [('**({"geom": True} if geom else {}))', '**{"geom": True})')],
            "the smoke batches have no geometry": [("seed=hs * 100003 + step, geom=geom)", "seed=hs * 100003 + step)")],
            "--distill-from is allowed with a position loss": [('assert args.get("distill_from") is None or not (cw or nw),', "assert True,")]}
    for edits in muts.values():
        must_fail(_check_pos_trainer, _mutant(*edits, of=JKT))
    print(f"pos_head_trainer: the options default to off, are refused without --pos-head / together / negative; the smoke runs (curve, node, no loss) train finite, the head moves (a head without a loss keeps dpos at 0), the options are in the args, the log "
          f"and model.pt (load_model rebuilds posh), another recipe is refused, the same one resumes; the loader gets geom=True exactly for a loss weight; the stage-2 trainer refuses a head run with a position head; mutants ({', '.join(muts)}) fail: ok")


def _check_pos_mean(mod):
    """`mod.make_loss` (jtoken_train or a textual mutant of it) on 130 calls of a head with a position head on synthetic geometry batches: the info of a call holds pos_mean = the mean of the TRAINED position loss (`curve` with curve_w, `node` with node_w) over the calls so far, at most
    the last LOG_WINDOW (100), and pos_n = the calls in that mean; a run without a position loss has neither key; the loss value is the loss of the jtoken loss alone (the window is information only)"""
    bs = [JK.synthetic_batch(2, 48, seed=s, geom=True) for s in range(5)]
    quiet = types.SimpleNamespace(log=lambda *a, **k: None)
    for kw, key, other in ((dict(curve_w=0.5), "curve", "node"), (dict(node_w=0.5), "node", "curve")):
        m, fn, infos, losses = _pos_head(step=JK.SWITCH + 5, perturb=True), mod.make_loss(quiet, 0.25, **kw), [], []
        with torch.no_grad():
            for i in range(130):
                lo, inf = fn(m, bs[i % 5])
                losses.append(float(lo))
                infos.append(inf)
            refs = [float(LJK.jtoken_loss(m(b_), b_, none_weight=0.25, **kw)[0]) for b_ in bs]                                          # the jtoken loss alone, the head unchanged after 130 calls (past the key-weight switch, no dropout)
        mean = lambda a, b, k: float(np.mean([infos[j][k] for j in range(a, b)]))  # noqa: E731
        assert (infos[0]["pos_n"], infos[49]["pos_n"], infos[99]["pos_n"], infos[129]["pos_n"]) == (1, 50, 100, 100), [i["pos_n"] for i in infos[::43]]
        for upto, lo_ in ((1, 0), (50, 0), (100, 0), (130, 30)):
            assert abs(infos[upto - 1]["pos_mean"] - mean(lo_, upto, key)) < 1e-12, (key, upto)
        assert abs(mean(30, 130, key) - mean(30, 130, other)) > 1e-4 and abs(infos[129]["pos_mean"] - mean(30, 130, other)) > 1e-4, "the two losses read the same: the check does not tell them apart"
        assert all(abs(losses[i] - refs[i % 5]) < 1e-6 for i in range(130)) and all(math.isfinite(x) for x in refs), "the window changed the loss"
    for head in (_pos_head(step=JK.SWITCH + 5, perturb=True), JK.JTokenHead()):                                                   # no position loss: no window keys
        head.set_step(JK.SWITCH + 5)
        with torch.no_grad():
            inf = mod.make_loss(quiet, 0.25)(head, bs[0])[1]
        assert "pos_mean" not in inf and "pos_n" not in inf and "curve" not in inf, sorted(inf)


def test_pos_mean():
    """E395 addendum A1: the log row of a position-loss run holds `pos_mean` (the mean of the trained loss over the last 100 calls) and `pos_n`; the numbers by hand on 130 calls (windows of 1, 50, 100 and 100 calls), the curve run reads `curve`, the node run `node`, a run without a
    position loss has neither key, the loss is unchanged; mutants (an unbounded window, the wrong loss, pos_n the cap, the last call instead of the mean) fail; the two smoke runs hold the keys in their log rows"""
    import jcr_eval as JC
    assert JC.WINDOW == JKT.LOG_WINDOW == 100, "the kill reader's window and the trainer's are the same number"
    _check_pos_mean(JKT)
    muts = {"the window is not bounded": [("pwin = collections.deque(maxlen=LOG_WINDOW)", "pwin = collections.deque()")],
            "a curve run reads the node loss": [('float(info["curve" if curve_w else "node"])', 'float(info["node"])')],
            "pos_n is the cap, not the calls": [("pos_n=len(pwin)", "pos_n=LOG_WINDOW")],
            "the row holds the last call, not the mean": [("pos_mean=float(np.mean(pwin))", "pos_mean=pwin[-1]")]}
    for edits in muts.values():
        must_fail(_check_pos_mean, _mutant(*edits, of=JKT))
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        real, seen = LJK.jtoken_loss, []

        def recording(out, batch, **k):                                                                                            # the loss info of every call of the smoke run
            r = real(out, batch, **k)
            seen.append(r[1])
            return r
        for name, extra in (("curve", ["--curve-w", "0.5"]), ("node", ["--node-w", "0.5"])):
            seen.clear()
            patched(LJK, "jtoken_loss", recording, JKT.train, JKT.parse_args(["--name", name, "--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "0.25", "--pos-head"] + extra), runs, 500, 48)
            last = [json.loads(r) for r in open(os.path.join(runs, name, "log.jsonl")) if "total" in r][-1]
            assert len(seen) == 20 and last["pos_n"] == 20 and abs(last["pos_mean"] - float(np.mean([i[name] for i in seen]))) < 1e-9, (name, last)
    print(f"pos_mean: pos_mean / pos_n = the mean and the count of the trained position loss over the last 100 calls (windows of 1, 50, 100, 100 on 130 calls by hand; curve run -> curve, node run -> node; none without a position loss; the loss unchanged), "
          f"in the log rows of the smoke runs; mutants ({', '.join(muts)}) fail: ok")


TESTS = [test_likelihood, test_none_weight, test_pos_loss, test_wiring_logp, test_decode, test_head, test_key_bias, test_writer_free, test_overfit, test_trainer, test_arch_options, test_pos_loss_trainer,
         test_live_inputs, test_head_seed, test_distill, test_distill_trainer, test_noise,
         test_rounds_zero, test_rounds_head, test_rounds_grad, test_message_weights, test_wire_loss, test_rounds_trainer, test_msg_none, test_msg_self,
         test_pos_head, test_curve_loss, test_pos_loss_weights, test_dump_dpos, test_pos_head_trainer, test_pos_mean]
if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtoken ok")
