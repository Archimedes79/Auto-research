"""E365 tests of the token head on predicted nodes (`jtoken.py`, `train/jtoken_train.py`); CPU only (the GPU is hidden), 2 threads, SYNTHETIC batches that follow the batch dict of the card (2a): no cache needed.

    wsl bash tools/env/_py14.sh training/grid/test_jtoken.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

likelihood (the loss by hand on a 5-token frame: a source with two GT successors, a source with none, a source whose GT link lies outside the gate, sources without a candidate masked; mutants: no none slot, a swapped
link direction, no gate; the none slot trains the empty sources), wiring_logp (a gate and padding mask, sums to 1), decode (the thresholds, the none rule, the kept ends, argmax; mutants: other thresholds / ends),
head (shapes, every parameter gets gradient, the gate of the head, order and padding invariance with a padding mutant), key_bias (the head's own p_none gets gradient from layer 2 on after the switch and not before;
no gradient reaches the cached inputs), writer_free (the forward reads no GT key), overfit (8 fixed frames, <= 300 steps, loss below 0.5 x its start, the decode of record reproduces the GT links),
trainer (the smoke run on synthetic batches, kill, resume, another recipe refused, no loader import).
"""
import json
import math
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                      # this test never touches the GPU
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


def _hand():
    """the 5-token frame of the likelihood test -> (out with leaf logits z / zn, batch). short candidates 0->1 0->2 0->3 1->2 3->4, long candidates 0->4 1->4; GT 0->1 0->2 (short), 0->4 (long), 3->0 (no gate)"""
    K = 5
    cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
    chord = torch.full((1, K, K), 3.0)
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (3, 4)):
        cand[0, 0, i, j] = True
    for i, j in ((0, 4), (1, 4)):
        cand[0, 1, i, j], chord[0, i, j] = True, 12.0
    g = torch.Generator().manual_seed(1)
    z, zn = torch.randn(1, 2, K, K, generator=g), torch.randn(1, 2, K, generator=g)
    types = torch.randn(1, K, 7, generator=g)
    y, ylong, ys = torch.zeros(1, K, K), torch.zeros(1, K, K, dtype=torch.bool), torch.full((1, K, K, 2), float("nan"))
    for i, j in ((0, 1), (0, 2), (0, 4), (3, 0)):
        y[0, i, j] = 1
    ylong[0, 0, 4], ys[0, 0, 4] = True, torch.tensor([0.3, 0.5])
    batch = dict(y=y, ylong=ylong, yscale=ys, ytype=torch.tensor([[1, 0, 1, 3, 0]]))
    out = dict(z=z.requires_grad_(), zn=zn.requires_grad_(), cand=cand, chord=chord, valid=torch.ones(1, K, dtype=torch.bool), scale=torch.full((1, K, K, 2), 0.4), types=types, types1=types.flip(-1))
    return out, batch


def _with_logp(out, cand=None):
    c = out["cand"] if cand is None else cand
    return dict(out, cand=c, logp=JK.wiring_logp(out["z"], out["zn"], c))


def _check_hand(out, batch):
    """the loss of the frame by hand (module docstring): asserts the numbers of `jtoken_loss`"""
    z, zn = out["z"][0].detach().tolist(), out["zn"][0].detach().tolist()
    nll_s = (-(lse(z[0][0][1], z[0][0][2]) - lse(z[0][0][1], z[0][0][2], z[0][0][3], zn[0][0]))                      # source 0: GT {1, 2} of {1, 2, 3}
             - (zn[0][1] - lse(z[0][1][2], zn[0][1]))                                                           # source 1: none
             - (zn[0][3] - lse(z[0][3][4], zn[0][3]))) / 3                                                      # source 3: its GT link 3 -> 0 is outside the gate: none
    nll_l = (-(z[1][0][4] - lse(z[1][0][4], zn[1][0])) - (zn[1][1] - lse(z[1][1][4], zn[1][1]))) / 2                  # long: 0: GT {4}; 1: none
    ce = lambda t: sum(-(t[0][n][k] - lse(*t[0][n])) for n, k in enumerate([1, 0, 1, 3, 0])) / 5  # noqa: E731
    ce1, ce3 = ce(out["types1"].detach().tolist()), ce(out["types"].detach().tolist())
    loss, info = JK.jtoken_loss(out, batch)
    want = nll_s + nll_l + 0.5 * (ce1 + ce3) + 0.5 * 0.1
    assert abs(info["nll_s"] - nll_s) < 1e-5 and abs(info["nll_l"] - nll_l) < 1e-5 and abs(info["ce1"] - ce1) < 1e-5 and abs(info["ce3"] - ce3) < 1e-5 and abs(info["scale_l1"] - 0.1) < 1e-6, (info, nll_s, nll_l)
    assert abs(float(loss) - want) < 1e-5, (float(loss), want)
    assert (info["gt_links"], info["pos_pairs"], info["gt_outside"], info["gt_class"]) == (4.0, 3.0, 1.0, 0.0), info
    return loss


def test_likelihood():
    out, b = _hand()
    out = _with_logp(out)
    loss = _check_hand(out, b)
    # mutants: no none slot (its logit -1e4: the softmax runs over the candidates only), a swapped link direction, the gate removed
    must_fail(_check_hand, _with_logp(dict(out, zn=torch.full_like(out["zn"], -1e4))), b)
    must_fail(_check_hand, out, dict(b, y=b["y"].transpose(1, 2), ylong=b["ylong"].transpose(1, 2)))
    must_fail(_check_hand, _with_logp(out, torch.ones_like(out["cand"])), b)
    # sources without a candidate (2, 4 short; 2, 3, 4 long) are masked: their none logit changes nothing; the empty sources are trained by the none slot (and push their candidates down)
    out2 = dict(out, zn=out["zn"].detach().clone())
    out2["zn"][0, 0, (2, 4)] = out2["zn"][0, 1, (2, 3, 4)] = 9.0
    assert abs(float(JK.jtoken_loss(_with_logp(out2), b)[0]) - float(loss)) < 1e-6, "a source without a candidate reaches the loss"
    loss.backward()
    zn_g, z_g = out["zn"].grad[0], out["z"].grad[0]
    assert float(zn_g[0, 1].abs()) > 0 and float(zn_g[0, 3].abs()) > 0 and float(zn_g[1, 1].abs()) > 0, "the none slot of a source with an empty G(i) gets no gradient"
    assert float(z_g[0, 1, 2]) > 0 and float(z_g[0, 3, 4]) > 0, "the candidate of an empty source must be pushed down (positive gradient on its logit)"
    assert float(zn_g[0, 2].abs()) == 0.0 and float(z_g[0, 0, 3]) > 0 and float(z_g[0, 0, 1]) < 0, "masked source / the non-GT candidate of a source with G(i): up for GT, down for the rest"
    assert float(z_g.abs().sum() - z_g[out["cand"][0]].abs().sum()) == 0.0, "a logit outside the gate gets gradient"
    print("likelihood: the loss by hand (two GT successors, empty G -> none, GT link outside the gate -> none, sources without a candidate masked, type CE on both readings, scale L1), mutants (no none slot, swapped "
          "direction, no gate) fail, the none slot trains the empty sources and pushes their candidates down: ok")


def test_wiring_logp():
    out, _ = _hand()
    lp = JK.wiring_logp(out["z"].detach(), out["zn"].detach(), out["cand"])
    p = lp.exp()
    assert float((p.sum(-1) - 1).abs().max()) < 1e-6, "a source must sum to 1 over the candidates + none"
    assert float(p[..., :5][~out["cand"]].max()) == 0.0, "probability outside the gate"
    assert float(p[0, 0, 2, 5]) == 1.0 and float(p[0, 1, 4, 5]) == 1.0, "a source without a candidate has p(none) = 1"
    assert abs(float(p[0, 0, 1, 2]) - math.exp(float(out["z"][0, 0, 1, 2]) - lse(float(out["z"][0, 0, 1, 2]), float(out["zn"][0, 0, 1])))) < 1e-6
    bad = JK.wiring_logp(out["z"].detach(), out["zn"].detach(), torch.ones_like(out["cand"])).exp()
    must_fail(holds, float(bad[..., :5][~out["cand"]].max()) == 0.0)                                         # the mutant (no gate) puts probability outside it
    print("wiring_logp: sums to 1 over {candidates} + none, exactly 0 outside the gate, p(none) = 1 without a candidate; the ungated mutant leaks: ok")


def _decode_out(p_none4):
    """5 tokens, short candidates 0->1 0->2 1->2 1->3 1->4 2->3 3->4, long 0->4: the per-source probabilities of `test_decode`; token 4 has p_none = p_none4"""
    K = 5
    cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
    P = torch.zeros(1, 2, K, K + 1)
    P[..., K] = 1.0
    for c, i, row in ((0, 0, {1: .36, 2: .34}), (0, 1, {2: .27, 3: .26, 4: .25}), (0, 2, {3: .31}), (0, 3, {4: .55}), (1, 0, {4: .6})):
        for j, v in row.items():
            cand[0, c, i, j], P[0, c, i, j] = True, v
        P[0, c, i, K] = 1.0 - sum(row.values())
    chord = torch.where(cand[0, 1].any(-1)[:, None] & cand[0, 1], 12.0, 3.0)[None]
    pt = torch.full((1, K, 7), 0.0)
    pn = torch.tensor([0.1, 0.1, 0.1, 0.2, p_none4])
    pt[0, :, 0], pt[0, :, 1] = pn, 1 - pn
    return dict(logp=P.clamp_min(1e-9).log(), cand=cand, chord=chord, valid=torch.ones(1, K, dtype=torch.bool), types=pt.clamp_min(1e-9).log(), types1=pt.clamp_min(1e-9).log())


def _pairs(d, key="link"):
    return sorted((int(i), int(j)) for i, j in d[key][0].nonzero().tolist())


def test_decode():
    a, bk = _decode_out(0.49), _decode_out(0.5)
    rec = lambda o, **k: _pairs(JK.decode(o, **k))  # noqa: E731
    # 0 -> 1 (.36) and 0 -> 2 (.34): both kept (a fork source), >= 0.3 and above p_none .30; source 1: .27 / .26 / .25 against p_none .22: all below 0.3; 2 -> 3 (.31): >= 0.3 but p_none .69 wins; 3 -> 4 (.55);
    # long 0 -> 4 (.6)
    exp = [(0, 1), (0, 2), (0, 4), (3, 4)]
    assert rec(a) == exp and _pairs(JK.decode(a), "long") == [(0, 4)]
    assert rec(bk) == [(0, 1), (0, 2)], "a link into a node with p_none >= 0.5 must be dropped (p_none 0.49 keeps it)"
    assert rec(a, thr=0.35) == [(0, 1), (0, 4), (3, 4)] and rec(a, thr=0.2) == sorted(exp + [(1, 2), (1, 3), (1, 4)]), "the threshold (the 0.2 mutant also keeps source 1; 2 -> 3 stays out: p_none wins)"
    assert _pairs(JK.decode_argmax(a)) == [(0, 1), (0, 4), (1, 2), (3, 4)], "argmax over {none} + keys: one link per source and class (source 1: .27 beats none .22), none wins for 2"
    assert _pairs(JK.decode_argmax(bk)) == [(0, 1), (1, 2)]
    for mut in (rec(a, thr=0.2), rec(bk), _pairs(JK.decode_argmax(a))):
        must_fail(holds, mut == exp)                                                                        # each mutant (threshold, kept ends, argmax) differs from the record
    pr = JK.probs(a)
    assert abs(float(pr["p"][0, 0, 2]) - 0.34) < 1e-6 and abs(float(pr["pnone"][0, 0, 2]) - 0.30) < 1e-6 and bool(pr["keep"][0, 4]) and not bool(JK.probs(bk)["keep"][0, 4])
    print("decode: p >= 0.3 and p >= p(none), both ends kept (p_none < 0.5, 0.5 itself dropped), forks keep both branches, argmax = one link per source; mutants (thr 0.2, no end rule, argmax) differ: ok")


def _batch(B=3, K=48, seed=0):
    return JK.synthetic_batch(B, K, seed)


def test_head():
    torch.manual_seed(0)
    b = _batch()
    m = JK.JTokenHead()
    m.step.fill_(JK.SWITCH + 5)                                              # the learned key weights: every parameter is on the path
    out = m(b)
    B, K = b["pos"].shape[:2]
    assert out["logp"].shape == (B, 2, K, K + 1) and out["scale"].shape == (B, K, K, 2) and out["types"].shape == (B, K, 7) and out["types1"].shape == (B, K, 7)
    pr = JK.probs(out)
    assert float((out["logp"].exp().sum(-1) - 1).abs().max()) < 1e-5
    _, al = JT.pair_geometry(b["pos"], b["dirv"], b["valid"], e=b["e"].sigmoid())
    assert torch.equal(out["cand"].any(1), al) and float((pr["p"] * ~al).abs().max()) == 0.0, "the candidates are the allowed gate of `jtok.pair_geometry`"
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
    print(f"head: shapes, probabilities sum to 1, the candidates are the allowed gate (padding neither key nor source), gradient reaches every parameter, order and padding leave the valid outputs alone "
          f"(padding mutant fails): ok; parameters {JK.count_params(m)} ({JK.count_params(m) / 1e6:.3f} M)")


def test_key_bias():
    """what it proves: (1) after the switch the head's own p_none read after layer 1 AND after the last layer receives gradient from a loss on the links only (type CE off), i.e. through the key bias of layers 2, 3 and the
    wiring layer; before the switch it receives none (the keys use sigmoid(e)). (2) no gradient reaches any cached input (e, e_raw, flags, tok, ...): B' is frozen and the hard top-K is cut"""
    torch.manual_seed(0)
    b = _batch(2)

    def grads(step):
        m = JK.JTokenHead()
        m.step.fill_(step)
        reads = []

        def tap(mod, inp, o):                                                # keeps the gradient of each reading of the type Linear (layer 1, last layer)
            o.retain_grad()
            reads.append(o)
        m.types.register_forward_hook(tap)
        out = m(b)
        JK.jtoken_loss(out, b, type_w=0.0, scale_w=0.0)[0].backward()
        return [float(r.grad.abs().sum()) if r.grad is not None else 0.0 for r in reads], m

    (g1, g3), m = grads(JK.SWITCH + 1)
    assert g1 > 0 and g3 > 0, (g1, g3)
    assert float(m.types.weight.grad.abs().sum()) > 0
    (z1, z3), m0 = grads(0)
    assert z1 == 0.0 and z3 == 0.0 and (m0.types.weight.grad is None or float(m0.types.weight.grad.abs().sum()) == 0.0), "before the switch the links must not reach the type head"
    must_fail(holds, grads(0)[0][0] > 0)                                                                    # mutant: a head that ignores the switch / never uses its own p_none
    live = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in b.items()}
    m1 = JK.JTokenHead()
    m1.step.fill_(JK.SWITCH + 1)
    JK.jtoken_loss(m1(live), live)[0].backward()
    assert all(live[k].grad is None for k in JK.INPUTS if live[k].is_floating_point()), "a gradient reached a cached input"
    keep, JK._const = JK._const, (lambda t: t)                               # mutant: the graph is not cut at the inputs
    try:
        live2 = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in b.items()}
        JK.jtoken_loss(m1(live2), live2)[0].backward()
        must_fail(holds, all(live2[k].grad is None for k in JK.INPUTS if live2[k].is_floating_point()))
    finally:
        JK._const = keep
    print(f"key_bias: after the switch a links-only loss moves the head's own type reading (layer 1 grad {g1:.2e}, last layer {g3:.2e}), before it not; no gradient into the cached inputs (cut mutant fails): ok")


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
        m.step.fill_(step0)
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
                JKT.main(argv, runs=runs, ckpt_every=5, **kw)
            except KeyboardInterrupt:
                pass
            else:
                raise RuntimeError("the kill did not happen")
        finally:
            JK.jtoken_loss = real
        rd = os.path.join(runs, "t_s0")
        assert os.path.exists(os.path.join(rd, "ckpt.pt")) and not os.path.exists(os.path.join(rd, ".done"))
        JKT.main(argv, runs=runs, ckpt_every=5, **kw)
        log = open(os.path.join(rd, "train.log"), encoding="utf-8").read()
        assert "resumed from" in log and "at step 10" in log and all(os.path.exists(os.path.join(rd, f)) for f in (".done", "model.pt", "ckpt.pt", "config.json", "log.jsonl", "train.log")), log[-600:]
        m, args = JKT.load_model("t_s0", "cpu", runs)
        assert (args["seed"], args["steps"], args["bs"]) == (0, 20, 2) and int(m.step) == 20, "the step buffer of the head must follow the run through the kill and the resume"
        rows = [json.loads(r) for r in open(os.path.join(rd, "log.jsonl"))]
        assert all(k in rows[-1] for k in ("nll_s", "nll_l", "ce1", "ce3", "gpu_gb")), rows[-1]
        must_fail(JKT.main, argv + ["--lr", "5e-4"], runs=runs, ckpt_every=5, **kw)                          # another recipe in the same folder
        assert "lane_ptok_data" not in sys.modules, "--smoke must not import the loader"
    print(f"trainer: smoke on synthetic batches (20 steps, kill at 12, resume from the checkpoint of step 10, the step buffer = 20), run-directory files, loss parts logged, another recipe refused, "
          f"no loader import: ok")


TESTS = [test_likelihood, test_wiring_logp, test_decode, test_head, test_key_bias, test_writer_free, test_overfit, test_trainer]

if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtoken ok")
