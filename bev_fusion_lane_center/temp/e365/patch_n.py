import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep('''def _hand_loss(f, z, zn, t1, t3):
    """the per-frame numbers by hand: (nll_s, nll_l or None, ce1, ce3). G(i)''', '''def _hand_loss(f, z, zn, t1, t3, w=1.0):
    """the per-frame numbers by hand: (nll_s, nll_l or None, ce1, ce3); w = the none weight (empty-G sources x w, the none class of the CE x w, normalised by the sum of the weights). G(i)''')
rep('''-(zn[c][i] - den))
        res.append''', '''-w * (zn[c][i] - den))
        res.append''')
rep('''    ce = lambda t: sum(-(t[n][f["ytype"][n]] - lse(*t[n])) for n in range(f["K"]) if f["valid"][n]) / sum(f["valid"])  # noqa: E731''', '''    wt = lambda n: w if f["ytype"][n] == 0 else 1.0  # noqa: E731
    ce = lambda t: sum(wt(n) * -(t[n][f["ytype"][n]] - lse(*t[n])) for n in range(f["K"]) if f["valid"][n]) / sum(wt(n) for n in range(f["K"]) if f["valid"][n])  # noqa: E731''')
rep('''def _check_hand(out, batch, frames):''', '''def _check_hand(out, batch, frames, w=1.0):''')
rep('''    per = [_hand_loss(f, z[b], zn[b], t1[b], t3[b]) for b, f in enumerate(frames)]''', '''    per = [_hand_loss(f, z[b], zn[b], t1[b], t3[b], w) for b, f in enumerate(frames)]''')
rep('''    loss, info = JK.jtoken_loss(out, batch)
    assert abs(info["nll_s"]''', '''    loss, info = JK.jtoken_loss(out, batch, none_weight=w)
    assert abs(info["nll_s"]''')
rep('''def test_wiring_logp():''', '''def test_none_weight():
    out, b, frames = _hand()
    out = _with_logp(out)
    l1 = _check_hand(out, b, frames, 1.0)
    l5 = _check_hand(out, b, frames, 0.5)                                                       # W = 0.5 by hand: the empty-G sources x 0.5 in both classes, the none class of the CE x 0.5, normalised by the weights
    assert abs(float(l1) - float(l5)) > 1e-3 and float(JK.jtoken_loss(out, b)[0]) == float(JK.jtoken_loss(out, b, none_weight=1.0)[0]), "W changes the loss, W = 1 is the loss of before"
    assert any(not [j for a, j in f["gt"] if a == i and j in js] for f in frames for i, js in f["short"].items()) and any(f["ytype"][n] == 0 and f["valid"][n] for f in frames for n in range(f["K"])), "the frames have empty-G sources and none tokens"
    orig = JK._source_nll
    must_fail(patched, JK, "_source_nll", lambda pos, none, has_g, w: w * torch.where(has_g, pos, none), _check_hand, out, b, frames, 0.5)        # the weight on the sources WITH a GT successor too
    must_fail(patched, JK, "_source_nll", lambda pos, none, has_g, w: orig(pos, none, has_g, 1.0), _check_hand, out, b, frames, 0.5)                # no weight on the none sources
    orig_fm = JK._frame_mean
    must_fail(patched, JK, "_frame_mean", lambda x, m, w=None: orig_fm(x, m), _check_hand, out, b, frames, 0.5)                                     # weighted terms divided by the token count instead of the sum of the weights
    # the part of the loss from the sources with a GT successor does not move with W: the gradient of a logit of a G-source is the same up to the shared softmax normalisation only through its own row
    for w_ in (0.5, 2.0):
        o2 = _with_logp(dict(out, z=out["z"].detach().clone().requires_grad_(), zn=out["zn"].detach().clone().requires_grad_()))
        JK.jtoken_loss(o2, b, none_weight=w_)[0].backward()
        o1 = _with_logp(dict(out, z=out["z"].detach().clone().requires_grad_(), zn=out["zn"].detach().clone().requires_grad_()))
        JK.jtoken_loss(o1, b)[0].backward()
        assert torch.equal(o2["z"].grad[0, 0, 0], o1["z"].grad[0, 0, 0]) and not torch.equal(o2["zn"].grad[0, 0, 1], o1["zn"].grad[0, 0, 1]), "frame 1 source 0 has a GT successor (short): untouched; source 1 (long class: empty) moves"
    print("none_weight: the loss by hand with W = 0.5 on the two hand frames (empty-G sources x W in both classes, the none class of the CE x W, normalised by the sum of the weights), W = 1 = the loss of before, "
          "mutants (the weight on the non-empty sources too, no weight on the none sources, the CE divided by the token count) fail, a source with a GT successor is untouched: ok")


def test_wiring_logp():''')
rep("test_likelihood, test_wiring_logp,", "test_likelihood, test_none_weight, test_wiring_logp,") if "test_likelihood, test_wiring_logp," in s else None
open(p, "w", encoding="utf-8").write(s)
