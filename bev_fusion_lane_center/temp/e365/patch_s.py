import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
rep('''def test_dump_roundtrip():''', '''def _links_set(l):
    return set(zip(l["src"].tolist(), l["dst"].tolist()))


def test_decode_variant():
    K = 6
    succ = {0: {1: 0.5, 2: 0.4}, 1: {2: 0.2}, 2: {3: 0.35}, 3: {4: 0.1}}                                  # none = the rest: src 0 two branches, src 1 weak (0.2), src 2 below its none (0.35 < 0.65), src 3 very weak
    out, tok = stub_out(K, succ, [0.1, 0.1, 0.55, 0.1, 0.1, 0.1])                                          # token 2 has p_none 0.55: not a node at 0.5
    (f,) = E.batch_dump(out, batch_of(tok))
    valid = np.ones(K, bool)
    rec = E.decode_variant(f, valid)                                                                       # the defaults are the decode of record, exactly
    assert all(np.array_equal(rec["rec"][k], f["rec"][k]) for k in f["rec"]) and np.array_equal(rec["keep"], f["keep"]), (rec["rec"], f["rec"])
    got = lambda **k: _links_set(E.decode_variant(f, valid, **k)["rec"])  # noqa: E731
    assert got() == {(0, 1)} and got(link_thr=0.2) == {(0, 1)}, "token 2 is no node at p_none 0.55: no link to or from it"
    assert got(keep_thr=0.6) == {(0, 1), (0, 2)} and got(link_thr=0.2, keep_thr=0.6) == {(0, 1), (0, 2)}, "a higher keep threshold adds the node"
    assert got(keep_thr=0.6, none_gate=False) == {(0, 1), (0, 2), (2, 3)}, "the none gate off adds 2 -> 3 (0.35 below p(none) 0.65)"
    assert got(link_thr=0.2, keep_thr=0.6, none_gate=False) == {(0, 1), (0, 2), (1, 2), (2, 3)} and got(link_thr=0.09, keep_thr=0.6, none_gate=False) == {(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)}
    for kt in (0.5, 0.6):
        for gate in (True, False):
            chain = [got(link_thr=t, keep_thr=kt, none_gate=gate) for t in (0.3, 0.2, 0.09, 0.05)]
            assert all(x <= y for x, y in zip(chain, chain[1:])), "a lower link threshold only adds links"
        assert got(link_thr=0.2, keep_thr=kt) <= got(link_thr=0.2, keep_thr=kt, none_gate=False), "no none gate only adds links"
    must_fail(E.decode_variant, f, valid, 0.04)                                                            # below the candidates the dump holds
    v = E.decode_variant(f, np.array([True, True, True, True, True, False]), keep_thr=0.6)                 # the valid mask: an invalid token is never a node, whatever its p_none
    assert not v["keep"][5] and v["keep"][2]
    pred = E.frame_prediction(tok, E.decode_variant(f, valid, keep_thr=0.6), "rec", 0.6)                   # a node kept at 0.6 still passes the scorer's e >= 0.5, e = 1 - p_none + 0.1
    import qgraph_eval as QE
    assert QE.kept(pred, 0.5, 0.5)[0].all() and abs(pred["nodes"]["e"][2] - 0.55) < 1e-6 and abs(pred["nodes"]["e"][0] - 1.0) < 1e-6 and len(pred["nodes"]["pos"]) == 6, pred["nodes"]["e"]
    # a dump of before the variants (no scales in the candidates, none probabilities in float16): the record's links keep their scales
    old = dict(f, none_cls=f["none_cls"].astype(np.float16), cand={k: v for k, v in f["cand"].items() if k not in ("a", "b")}, rec=dict(f["rec"], a=np.full(len(f["rec"]["src"]), 0.25, np.float32)))
    r_old = E.decode_variant(old, valid)["rec"]
    assert _links_set(r_old) == _links_set(f["rec"]) and np.allclose(r_old["a"], 0.25), "the record's links keep their scales"
    ex = E.decode_variant(old, valid, link_thr=0.09, keep_thr=0.6, none_gate=False)["rec"]
    added = [k for k, l in enumerate(zip(ex["src"].tolist(), ex["dst"].tolist())) if l not in _links_set(f["rec"])]
    assert len(added) == 4 and np.allclose(ex["a"][added], 1 / 3) and np.allclose(ex["a"][[k for k in range(len(ex["a"])) if k not in added]], 0.25), "links that are not the record's get 1/3"
    # a real head (random weights): the default variant is the decode of record on every frame, exactly
    torch.manual_seed(0)
    m = JK.JTokenHead().eval()
    m.set_step(JK.SWITCH)
    sb = JK.synthetic_batch(3, 48, seed=2)
    toks = [dict(pos=sb["pos"][b].double().numpy(), cell=np.arange(48)) for b in range(3)]
    frames = E.batch_dump(E.run_head(m, sb, "cpu"), dict(token=["a", "b", "c"], row=torch.arange(3), tokens_np=toks))
    n_rec = 0
    for fr_, val in zip(frames, sb["valid"].numpy()):
        d = E.decode_variant(fr_, val)
        assert all(np.array_equal(d["rec"][k], fr_["rec"][k]) for k in fr_["rec"]) and np.array_equal(d["keep"], fr_["keep"])
        n_rec += len(fr_["rec"]["src"])
    assert n_rec > 0
    print(f"decode_variant: the defaults equal the record decode exactly (hand frame and {n_rec} links of a real head on 3 frames), a lower link threshold only adds links, the none gate off adds the links of sources whose none mass is above them, a higher keep "
          "threshold adds nodes (e shifted so that the scorer keeps them), thresholds below the candidates refused, a dump of before the variants keeps the record's scales: ok")


def test_dump_roundtrip():''')
open(p, "w", encoding="utf-8").write(s)
