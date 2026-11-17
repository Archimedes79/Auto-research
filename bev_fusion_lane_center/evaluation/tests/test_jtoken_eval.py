"""Tests of `jtoken_eval.py` (E365 S1 read-out, controls and checks); CPU only (the GPU is hidden), 2 threads. Hand frames need no data; the tests marked (cache) read the first VAL frames of the predicted-node cache of B' seed 0 under
runs/e365_cache (built by the lane chat) and are skipped with a message when it is not there.

    wsl bash tools/env/_py14.sh test_jtoken_eval.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

decode_variant (the defaults = the record decode, thresholds and the none gate by hand, monotone), sweep (24 cells, counts monotone), keep_rule (a node is kept at p_none 0.49 and not at 0.50, e = 1 - p_none, the links of a dropped node go), decode_fork (rec keeps both branches of a fork, argmax one; the none slot wins when the branches are weak),
dump_roundtrip (a real head forward: the dump holds exactly the decodes of `jtoken`, the candidate list, the keep mask), writer_no_gt (the prediction is made from the tokens only: mutants that put a GT section, a GT flag, a GT
position or an extra GT field into the writer are caught by `check_writer`; the dump frame and `frame_prediction` have no GT input), oracle_hi (cache: an oracle head whose links are the c-hi links delivers exactly c-hi, and scores the same; the y pairs
between assigned tokens are more than c-hi), drop (cache: reproducible, the seed matters, the share, equal to the loader's own `token_drop` at the same seed), controls (cache: c-rule, c-floor and a trained-style head run through the writer
and the standard scorer), readout_dump (B's stored dump frame -> prediction: sec0 zeroes the sections, the close-node step drops the weaker of two close nodes), numbers (the layer-1 comparison at its tolerance, g1, the sample, the parity
comparison inside / outside its tolerance, the matching of two token sets by cell), parity_timing (cache: the check-parity and the timing code paths on a tiny head, CPU: stamped files, K from the cache), score_max (cache: `ctl --max` / `dump --max` / `score --max` end to end, g1 by hand).
"""
import contextlib
import json
import os
import sys
import tempfile
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                      # this test never touches the GPU
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in ("..", "../../training", "../../training/grid", "../../training/data", "../../training/data/prep", "../../training/train"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
import jtok as JT  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_eval as E  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
from test_jtok import must_fail  # noqa: E402

torch.set_num_threads(2)
CACHE_OK = any(os.path.exists(os.path.join(E.ROOT, "runs", "e365_cache", d, "val", "manifest.json")) for d in (os.listdir(os.path.join(E.ROOT, "runs", "e365_cache")) if os.path.isdir(os.path.join(E.ROOT, "runs", "e365_cache")) else []))
SMOKE = os.path.join(E.ROOT, "runs", "e365_smoke_build", "model.pt")          # a head of the smoke run (random-ish weights): only for plumbing


@contextlib.contextmanager
def patched(*pairs):
    """(object, name, value) triples set for the block and put back after it"""
    old = [(o, n, getattr(o, n)) for o, n, _ in pairs]
    for o, n, v in pairs:
        setattr(o, n, v)
    try:
        yield
    finally:
        for o, n, v in old:
            setattr(o, n, v)


# ------------------------------------------------------------------------------------------------------------------------------ hand frames
def hand_tokens(K):
    """K tokens on a line along +y, 2 m apart, heading +y; the fields of a `tokens_np` entry"""
    pos = np.stack([np.zeros(K), 2.0 * np.arange(K)], 1)
    return dict(pos=pos, dirv=np.tile([0.0, 1.0], (K, 1)), e_p=np.full(K, 0.9), flags_p=np.full((K, 8), 0.2), s_p=np.full(K, 0.3), cell=np.arange(K) + 100, valid=np.ones(K, bool))


def stub_out(K, succ, pnone, scale=None):
    """a head output of one frame (B = 1) for hand_tokens(K): succ = {source: {destination: probability}} of the short class (every pair of the line is a candidate; the none probability is the rest), pnone (K,) the type
    probability of none per token (the other types: lane point); scale (K, K, 2)"""
    tok = hand_tokens(K)
    chord = torch.cdist(torch.tensor(tok["pos"]), torch.tensor(tok["pos"])).float()[None]
    cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
    lp = torch.zeros(1, 2, K, K + 1)
    lp[..., :K] = JK.NEG
    for c in range(2):
        for i in range(K):
            for j in range(K):
                if i != j and j > i and ((chord[0, i, j] <= 8) == (c == 0)):
                    cand[0, c, i, j] = True
    for i, d in succ.items():
        for j, p in d.items():
            assert cand[0, 0, i, j], (i, j)
            lp[0, 0, i, j] = float(np.log(p))
        lp[0, 0, i, K] = float(np.log(1.0 - sum(d.values())))
    for c in range(2):                                                  # a source without an entry: p(none) = 1 (log 0), its candidates stay at NEG
        for i in range(K):
            if not (c == 0 and i in succ):
                lp[0, c, i, K] = 0.0
    pn = torch.tensor(pnone, dtype=torch.float32)
    types = torch.full((1, K, JK.NTYPE), -1e4)
    types[0, :, 0], types[0, :, 1] = pn.log(), (1 - pn).log()
    return dict(logp=lp, cand=cand, scale=torch.full((1, K, K, 2), THIRD) if scale is None else scale, types=types, types1=types, valid=torch.ones(1, K, dtype=torch.bool), chord=chord), tok


THIRD = 1 / 3


def batch_of(tok, n=1):
    return dict(token=[f"t{i}" for i in range(n)], row=torch.arange(n), tokens_np=[tok] * n)


def pred_of(out, tok, decode="rec"):
    (f,) = E.batch_dump(out, batch_of(tok))
    return E.frame_prediction(tok, f, decode), f


# ------------------------------------------------------------------------------------------------------------------------------ tests
def test_keep_rule():
    K = 4
    out, tok = stub_out(K, {0: {2: 0.8}, 1: {2: 0.8}, 2: {3: 0.8}}, [0.49, 0.50, 0.10, 0.90])
    pred, f = pred_of(out, tok)
    assert f["keep"].tolist() == [True, False, True, False], f["keep"]                               # p_none 0.49 is kept, exactly 0.50 and 0.90 are not
    nd = pred["nodes"]
    assert np.allclose(nd["e"], [0.51, 0.9]) and nd["pos"].tolist() == [[0.0, 0.0], [0.0, 4.0]], nd
    assert pred["links"]["src"].tolist() == [0] and pred["links"]["dst"].tolist() == [1], "only the link between two kept nodes survives, renumbered"
    import qgraph_eval as QE
    assert QE.kept(pred, 0.5, 0.5)[0].all(), "the scorer keeps every node the writer wrote (e > 0.5)"
    out, tok = stub_out(K, {0: {2: 0.8}}, [0.5, 0.5, 0.5, 0.5])
    assert len(pred_of(out, tok)[0]["nodes"]["pos"]) == 0, "p_none = 0.50 exactly: no node is kept (the card's p_none < 0.5)"
    print("keep_rule: p_none 0.49 kept (e 0.51), 0.50 and 0.90 not; a link to a dropped node is dropped; the writer never hands the scorer a node at e <= 0.5: ok")


def test_decode_fork():
    K = 4
    out, tok = stub_out(K, {0: {1: 0.5, 2: 0.4}}, [0.1] * K)                                          # a fork: 0.5 / 0.4, none 0.1
    _, f = pred_of(out, tok)
    assert sorted(zip(f["rec"]["src"].tolist(), f["rec"]["dst"].tolist())) == [(0, 1), (0, 2)], f["rec"]
    assert list(zip(f["argmax"]["src"].tolist(), f["argmax"]["dst"].tolist())) == [(0, 1)], f["argmax"]
    out, tok = stub_out(K, {0: {1: 0.25, 2: 0.2}}, [0.1] * K)                                         # weak branches, none 0.55: the none slot wins
    _, f = pred_of(out, tok)
    assert len(f["rec"]["src"]) == 0 and len(f["argmax"]["src"]) == 0, (f["rec"], f["argmax"])
    out, tok = stub_out(K, {0: {1: 0.3, 2: 0.3}}, [0.1] * K)                                          # exactly the threshold 0.3 (float32): p >= 0.3 and p >= p(none) 0.4? no: none 0.4 wins
    _, f = pred_of(out, tok)
    assert len(f["rec"]["src"]) == 0, "p(none) = 0.4 is above both branches: no link at the record decode"
    out, tok = stub_out(K, {0: {1: 0.45, 2: 0.4}}, [0.1] * K)                                         # both above the threshold and above none (0.15)
    _, f = pred_of(out, tok)
    assert len(f["rec"]["src"]) == 2 and len(f["argmax"]["src"]) == 1
    out, tok = stub_out(K, {0: {1: 0.25, 2: 0.25, 3: 0.25}}, [0.1] * K)                              # three weak branches and none, all 0.25: nothing reaches 0.3
    _, f = pred_of(out, tok)
    assert len(f["rec"]["src"]) == 0, f["rec"]
    assert list(zip(f["argmax"]["src"].tolist(), f["argmax"]["dst"].tolist())) == [(0, 1)], "a tie goes to the first key (torch argmax), one link: the argmax cap"
    print("decode_fork: record keeps both branches, argmax the better one, the none slot beats weak branches: ok")


def _links_set(l):
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
    assert got(link_thr=0.19, keep_thr=0.6, none_gate=False) == {(0, 1), (0, 2), (1, 2), (2, 3)} and got(link_thr=0.09, keep_thr=0.6, none_gate=False) == {(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)}
    for kt in (0.5, 0.6):
        for gate in (True, False):
            chain = [got(link_thr=t, keep_thr=kt, none_gate=gate) for t in (0.3, 0.19, 0.09, 0.05)]
            assert all(x <= y for x, y in zip(chain, chain[1:])), "a lower link threshold only adds links"
        assert got(link_thr=0.2, keep_thr=kt) <= got(link_thr=0.2, keep_thr=kt, none_gate=False), "no none gate only adds links"
    must_fail(E.decode_variant, f, valid, 0.04)                                                            # below the candidates the dump holds
    v = E.decode_variant(f, np.array([True, True, True, True, True, False]), keep_thr=0.6)                 # the valid mask: an invalid token is never a node, whatever its p_none
    assert not v["keep"][5] and v["keep"][2]
    pred = E.frame_prediction(tok, E.decode_variant(f, valid, keep_thr=0.6), "rec", 0.6)                   # a node kept at 0.6 still passes the scorer's e >= 0.5, e = 1 - p_none + 0.1
    import qgraph_eval as QE
    assert QE.kept(pred, 0.5, 0.5)[0].all() and abs(pred["nodes"]["e"][2] - 0.55) < 1e-6 and abs(pred["nodes"]["e"][0] - 1.0) < 1e-6 and len(pred["nodes"]["pos"]) == 6, pred["nodes"]["e"]
    old = dict(f, cand={k: v for k, v in f["cand"].items() if k not in ("a", "b")})                         # a dump of before the variants (no scales in the candidates): refused, re-dump
    must_fail(E.decode_variant, old, valid)
    E.check_record([f], [valid])
    must_fail(E.check_record, [dict(f, rec=dict(f["rec"], a=f["rec"]["a"] + 0.01))], [valid])                # a record whose scales are not the candidates' is caught before a variant is scored
    # a hand frame exactly on the thresholds: p = float32(0.3) at link 0.3, p == p(none), p_none == float32(0.6) at keep 0.6; each mutant of a comparison delivers another graph
    f32 = np.float32
    th = dict(token="t", pnone=np.array([0.1, 0.1, 0.6, 0.1, 0.1], f32), none_cls=np.array([[0.2, 0.4, 0.1, 0.1, 0.1], [1.0] * 5], f32),
              cand=dict(src=np.array([0, 1, 3, 3], np.int16), dst=np.array([1, 3, 2, 4], np.int16), p=np.array([0.3, 0.4, 0.5, 0.5], f32), long=np.zeros(4, bool), a=np.full(4, THIRD, f32), b=np.full(4, THIRD, f32)))
    v5 = np.ones(5, bool)
    got_th = E.decode_variant(th, v5, link_thr=0.3, keep_thr=0.6)
    want = {(0, 1), (1, 3), (3, 4)}
    assert _links_set(got_th["rec"]) == want and got_th["keep"].tolist() == [True, True, False, True, True], (got_th["rec"], got_th["keep"])

    def mutant(le_keep=False, gt_link=False, gt_none=False):                                               # the rule of `decode_variant` with one comparison changed
        c, lt, kt = th["cand"], f32(0.3), f32(0.6)
        keep = (th["pnone"] <= kt) if le_keep else (th["pnone"] < kt)
        s, d = c["src"].astype(int), c["dst"].astype(int)
        m = ((c["p"] > lt) if gt_link else (c["p"] >= lt)) & keep[s] & keep[d]
        pn = th["none_cls"][0, s]
        m &= (c["p"] > pn) if gt_none else (c["p"] >= pn)
        return set(zip(s[m].tolist(), d[m].tolist()))
    assert mutant() == want and all(mutant(**{k: True}) != want for k in ("le_keep", "gt_link", "gt_none")), [mutant(**{k: True}) for k in ("le_keep", "gt_link", "gt_none")]
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
          "threshold adds nodes (e shifted so that the scorer keeps them), thresholds below the candidates and a dump of before the variants refused, check_record catches a record that is not the default variant; a hand frame on "
          "the thresholds (p = 0.3, p = p(none), p_none = 0.6) kills the mutants p > thr, p > p(none), p_none <= keep: ok")


def test_dump_roundtrip():
    torch.manual_seed(0)
    m = JK.JTokenHead().eval()
    m.set_step(JK.SWITCH)
    sb = JK.synthetic_batch(3, 48, seed=1)
    out = E.run_head(m, sb, "cpu")
    toks = []
    for b in range(3):
        toks.append(dict(pos=sb["pos"][b].double().numpy(), dirv=sb["dirv"][b].double().numpy(), e_p=sb["e"][b].sigmoid().double().numpy(), flags_p=sb["flags"][b].sigmoid().double().numpy(),
                         s_p=sb["s"][b].sigmoid().double().numpy(), cell=np.arange(48), valid=sb["valid"][b].numpy()))
    frames = E.batch_dump(out, dict(token=["a", "b", "c"], row=torch.arange(3), tokens_np=toks))
    rec, alt, pr = JK.decode(out), JK.decode_argmax(out), JK.probs(out)
    for b, f in enumerate(frames):
        assert sorted(f) == ["argmax", "cand", "cell", "keep", "none_cls", "pnone", "ptype", "rec", "row", "token"], sorted(f)           # no field of a GT table
        assert f["keep"].tolist() == rec["keep"][b].tolist() and np.allclose(f["pnone"], pr["pt"][b, :, 0])
        for name, d in (("rec", rec), ("argmax", alt)):
            want = sorted(zip(*(x.tolist() for x in torch.nonzero(d["link"][b], as_tuple=True))))
            assert sorted(zip(f[name]["src"].tolist(), f[name]["dst"].tolist())) == want, name
        assert all(f["rec"]["p"] >= 0.3 - 1e-6) and (len(f["rec"]["src"]) == 0 or np.isin(f["rec"]["src"].astype(int) * 1000 + f["rec"]["dst"], f["cand"]["src"].astype(int) * 1000 + f["cand"]["dst"]).all()), "the candidate list holds every link"
        assert (f["cand"]["p"] >= E.KEEP_P - 1e-6).all() and f["none_cls"].shape == (2, 48) and (f["none_cls"] >= 0).all() and (f["none_cls"] <= 1.0001).all()
        lg = f["rec"]["long"]
        assert np.allclose(f["rec"]["a"][~lg], 1 / 3) and ((f["rec"]["a"][lg] >= 0.05) & (f["rec"]["a"][lg] <= 1.25)).all(), "scales: 1/3 for the short links, the head's for the long ones"
    print(f"dump_roundtrip: the dump holds the keep mask, both decodes, {sum(len(f['rec']['src']) for f in frames)} record links, the candidate list and the none probabilities of 3 frames of a real head: ok")


def test_writer_no_gt():
    import inspect
    K = 4
    out, tok = stub_out(K, {0: {1: 0.8}, 1: {2: 0.8}}, [0.1] * K)
    pred, f = pred_of(out, tok)
    assert list(inspect.signature(E.frame_prediction).parameters) == ["tokens", "f", "decode", "keep_thr"], "the writer's inputs are the tokens and the dump frame"
    gt_leaks = {"section": lambda nd: nd.update(section=np.arange(len(nd["pos"]))), "flags": lambda nd: nd.update(flags=np.ones_like(nd["flags"])), "pos": lambda nd: nd.update(pos=nd["pos"] + 0.7),
                "extra field": lambda nd: nd.update(piece=np.zeros(len(nd["pos"]), int)), "s": lambda nd: nd.update(s=np.ones_like(nd["s"]))}
    real = LP.to_pred
    for name, leak in gt_leaks.items():
        def mutant(frame, keep, e_node, links, flags=None, leak=leak):
            p = real(frame, keep, e_node, links, flags)
            leak(p["nodes"])
            return p
        with patched((LP, "to_pred", mutant)):
            must_fail(E.frame_prediction, tok, f, "rec")
    gt_links = dict(src=np.array([1]), dst=np.array([0]), p=np.ones(1), long=np.zeros(1, bool), a=np.full(1, 1 / 3), b=np.full(1, 1 / 3), curves=None)
    for name, leak in (("links replaced by GT links", lambda p_: p_.update(links=gt_links)), ("e set to 1", lambda p_: p_["nodes"].update(e=np.ones_like(p_["nodes"]["e"]))),
                       ("cell changed", lambda p_: p_["nodes"].update(cell=p_["nodes"]["cell"] + 1))):
        def mutant2(frame, keep, e_node, links, flags=None, leak=leak):
            p_ = real(frame, keep, e_node, links, flags)
            leak(p_)
            return p_
        with patched((LP, "to_pred", mutant2)):
            must_fail(E.frame_prediction, tok, f, "rec")
    def with_assign(frame, keep, e_node, links, flags=None):
        p = real(frame, keep, e_node, links, flags)
        p["assign"] = dict(P=np.zeros((1, 1)))
        return p
    with patched((LP, "to_pred", with_assign)):
        must_fail(E.frame_prediction, tok, f, "rec")
    tok2 = dict(tok, piece=np.arange(K), gt_id=np.arange(K), gt_flags=np.ones((K, 8)))                  # GT fields lying around in the token dict change nothing
    p2 = E.frame_prediction(tok2, f, "rec")
    assert all(np.array_equal(pred["nodes"][k], p2["nodes"][k]) for k in pred["nodes"]) and np.array_equal(pred["links"]["src"], p2["links"]["src"])
    print("writer_no_gt: a GT section / flag / position / s / extra field / assignment in the writer is caught by check_writer (6 mutants), links replaced by GT links / e set to 1 / a changed cell by frame_prediction (3 more); GT fields in the token dict change nothing: ok")


def _cache_frames(n=3):
    ds = E.open_cache("val", 0)
    b = next(iter(E.batches(ds, range(n), bs=n)))
    return ds, b


def test_oracle_hi():
    if not CACHE_OK:
        print("oracle_hi: SKIPPED (no predicted-node cache under runs/e365_cache)")
        return
    import qgraph_eval as QE
    import lane_tok_data as TD
    ds, b = _cache_frames(3)
    td = TD.TokData("val", E.LATTICE, feats=False)
    K = ds.K
    hi, orc, n_pairs, n_y = [], [], 0, 0
    for i, tok in enumerate(b["tokens_np"]):
        gt_id, y, ys = b["label"]["gt_id"][i].numpy(), b["y"][i].numpy(), b["yscale"][i].numpy()
        keep, e, lk = E.hi_frame(tok, gt_id, td.table(i))
        n_pairs += len(lk["src"])
        n_y += len(E.hi_y_frame(tok, gt_id, y, ys)[2]["src"])
        hi.append(E.control_prediction(tok, keep, e, lk))
        # an oracle head: p = 0.9 on exactly those links (class by chord), none elsewhere; assigned tokens p_none 1e-6, the others 0.999
        pos = torch.tensor(tok["pos"])
        chord = torch.cdist(pos, pos).float()[None]
        lp = torch.zeros(1, 2, K, K + 1)
        lp[..., :K] = JK.NEG
        cand = torch.zeros(1, 2, K, K, dtype=torch.bool)
        for s_, d_ in zip(lk["src"], lk["dst"]):
            c = int(chord[0, s_, d_] > JT.SHORT)
            cand[0, c, s_, d_], lp[0, c, s_, d_] = True, float(np.log(0.9))
        for c in range(2):
            has = cand[0, c].any(1)
            lp[0, c, :, K] = torch.where(has, torch.tensor(float(np.log(0.1))), torch.tensor(0.0))
        sc = np.full((K, K, 2), 1 / 3, np.float32)
        sc[lk["src"], lk["dst"]] = np.stack([lk["a"], lk["b"]], 1)
        pn = torch.where(torch.tensor(keep), torch.tensor(1e-6), torch.tensor(0.999))
        types = torch.full((1, K, JK.NTYPE), -1e4)
        types[0, :, 0], types[0, :, 1] = pn.log(), (1 - pn).log()
        out = dict(logp=lp, cand=cand, scale=torch.tensor(sc)[None], types=types, types1=types, valid=torch.tensor(tok["valid"])[None], chord=chord)
        (f,) = E.batch_dump(out, dict(token=[b["token"][i]], row=b["row"][i:i + 1], tokens_np=[tok]))
        orc.append(E.frame_prediction(tok, f, "rec"))
    for p, q in zip(hi, orc):
        for k in ("pos", "dirv", "flags", "s", "cell", "section"):
            assert np.array_equal(p["nodes"][k], q["nodes"][k]), k
        assert np.allclose(p["nodes"]["e"], q["nodes"]["e"], atol=1e-5)
        order = lambda lk: np.lexsort((lk["dst"], lk["src"]))  # noqa: E731
        for k in ("src", "dst", "long"):
            assert np.array_equal(p["links"][k][order(p["links"])], q["links"][k][order(q["links"])]), k
        assert np.allclose(p["links"]["a"][order(p["links"])], q["links"]["a"][order(q["links"])]) and np.allclose(p["links"]["b"][order(p["links"])], q["links"]["b"][order(q["links"])])
    frames = QE.val_frames()[:3]
    r_hi, r_orc = E.score_preds(hi, frames, log=lambda *a, **k: None), E.score_preds(orc, frames, log=lambda *a, **k: None)
    assert all(abs(r_hi["graph"][k] - r_orc["graph"][k]) <= 1e-6 for k in E.KEYS), {k: (r_hi["graph"][k], r_orc["graph"][k]) for k in E.KEYS}
    assert n_pairs > 0 and r_hi["graph"]["topo_F1"] > 0.3, r_hi["graph"]
    assert n_y > 1.2 * n_pairs, (n_y, n_pairs)                                  # the y pairs between assigned tokens include links to the successor's neighbours (why c-hi is made from the table)
    print(f"oracle_hi: an oracle head on the c-hi links = c-hi on 3 VAL frames ({n_pairs} links, the y pairs between assigned tokens are {n_y}): same nodes, links, scales, graph numbers equal to 1e-6 (topo F1 {r_hi['graph']['topo_F1']:.4f}): ok")


def test_drop():
    if not CACHE_OK:
        print("drop: SKIPPED (no predicted-node cache under runs/e365_cache)")
        return
    ds = E.open_cache("val", 0)
    idx = range(8)
    mk = lambda drop, seed: [E.apply_drop(b, drop, seed) for b in E.batches(ds, idx, bs=8)][0]  # noqa: E731
    a, a2, c = mk(0.17, 0), mk(0.17, 0), mk(0.17, 1)
    assert torch.equal(a["valid"], a2["valid"]) and not torch.equal(a["valid"], c["valid"]), "reproducible for one seed, another seed another removal"
    base = mk(0.0, 0)
    cut = float((base["valid"] & ~a["valid"]).sum() / base["valid"].sum())
    assert 0.10 < cut < 0.25 and not (a["valid"] & ~base["valid"]).any(), cut
    assert all(np.array_equal(t["valid"], a["valid"][i].numpy()) for i, t in enumerate(a["tokens_np"])), "tokens_np follows the removal"
    ds_d = LP.PTokData("val", 0, "e292_a", None, 0.17)                      # the loader's own token_drop is seeded with the cache seed (0)
    assert all(np.array_equal(ds_d[i]["valid"], a["valid"][i].numpy()) for i in idx), "the same removal as the loader's token_drop at drop seed = cache seed"
    print(f"drop: reproducible for one seed, another seed differs, {100 * cut:.1f} % of the valid tokens removed, identical to the loader's token_drop at the same seed: ok")


def test_controls():
    if not CACHE_OK:
        print("controls: SKIPPED (no predicted-node cache under runs/e365_cache)")
        return
    import qgraph_eval as QE
    ds, b = _cache_frames(3)
    frames = QE.val_frames()[:3]
    rule = [E.control_prediction(t, *E.rule_frame(t)) for t in b["tokens_np"]]
    r = E.score_preds(rule, frames, log=lambda *a, **k: None)["graph"]
    assert all(np.isfinite(r[k]) for k in E.KEYS) and r["topo_F1"] > 0.1, r
    fl = E.batch_dump(E.run_head(E.floor_model(), b, "cpu"), b)
    pf = [E.frame_prediction(t, f, "rec") for t, f in zip(b["tokens_np"], fl)]
    rf = E.score_preds(pf, frames, log=lambda *a, **k: None)["graph"]
    assert all(np.isfinite(rf[k]) for k in E.KEYS)
    if os.path.exists(SMOKE):
        m, _ = E.JKT.load_model("e365_smoke_build", "cpu")
        fs = E.batch_dump(E.run_head(m, b, "cpu"), b)
        assert all(len(f["keep"]) == ds.K for f in fs)
    print(f"controls: c-rule (topo F1 {r['topo_F1']:.4f}) and c-floor (topo F1 {rf['topo_F1']:.4f}) of 3 VAL frames through the writer and the standard scorer: ok")


def test_readout_dump():
    nd = lambda e: dict(row=7, e=np.array(e), pos=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 6.0]]), dirv=np.tile([0.0, 1.0], (3, 1)), flags=np.zeros((3, 8)), s=np.zeros(3), section=np.array([0, 3, 3]),  # noqa: E731
                       link_src=np.array([0, 1]), link_dst=np.array([1, 2]), link_p=np.array([0.9, 0.9]), link_long=np.zeros(2, bool), link_a=np.full(2, 0.3), link_b=np.full(2, 0.3))
    d = nd([0.9, 0.8, 0.95])                                                           # nodes 0 and 1 are 1.0 m apart: the close-node step keeps the stronger (0)
    own, sec0 = E.pred_of_dump(d, E.PLAIN, False), E.pred_of_dump(d, E.PLAIN, True)
    assert own["nodes"]["section"].tolist() == [0, 3, 3] and sec0["nodes"]["section"].tolist() == [0, 0, 0] and own["nodes"]["e"].tolist() == [0.9, 0.8, 0.95], "sections: B's own / zero; plain keeps every e"
    assert d["section"].tolist() == [0, 3, 3], "the stored dump is not modified"
    close = E.pred_of_dump(d, E.CLOSE, True)
    assert close["nodes"]["e"].tolist() == [0.9, 0.0, 0.95], close["nodes"]["e"]
    print("readout_dump: sec0 zeroes the sections without touching the dump, the close-node step drops the weaker of two close nodes, plain keeps all: ok")


def test_numbers():
    ref = {k: 0.5 for k in E.CHECK}
    ok, diff = E.compare_numbers({k: 0.5 + 9e-7 for k in E.CHECK}, ref)
    assert ok and abs(diff["topo_F1"] - 9e-7) < 1e-12
    ok, diff = E.compare_numbers(dict({k: 0.5 for k in E.CHECK}, dup_frac=0.5 + 2e-6), ref)
    assert not ok and abs(diff["dup_frac"] - 2e-6) < 1e-12
    must_fail(E.compare_numbers, {k: 0.5 for k in E.CHECK[:-1]}, ref)                      # a missing number is an error, not a pass
    assert abs(E.g1(0.65, 0.8, 0.5) - 0.5) < 1e-12 and E.g1(0.5, 0.5, 0.5) != E.g1(0.5, 0.5, 0.5), "g1 = (S1 - rule) / (hi - rule), NaN when the controls coincide"
    assert E.sample(873, "val").tolist() == list(range(873))
    s = E.sample(8414, "train")
    assert len(s) == 873 and s[0] == 0 and s[-1] == 8413 and (np.diff(s) > 0).all()
    out, tok = stub_out(4, {0: {1: 0.8}}, [0.499, 0.1, 0.1, 0.1])
    out2, _ = stub_out(4, {0: {1: 0.8}}, [0.501, 0.1, 0.1, 0.1])
    ok, num = E.compare_outputs(out, out2, 0.01)
    assert ok and num["kept_nodes_differ"] == 1 and num["max_dpnone"] < 0.0021, num                 # a node flips only where p_none is within the tolerance of 0.5
    ok, num = E.compare_outputs(out, stub_out(4, {0: {1: 0.8}}, [0.52, 0.1, 0.1, 0.1])[0], 0.01)
    assert not ok, num
    ok, num = E.compare_outputs(out, stub_out(4, {0: {1: 0.7}}, [0.499, 0.1, 0.1, 0.1])[0], 0.01)
    assert not ok and num["max_dp"] > 0.09, num
    cells = np.array([[5, 7, 9, 11]])
    pm = E.align(cells, np.array([[11, 5, 9, 99]]))                                              # b holds 11, 5, 9 and a token that a lacks; a's 7 is not in b
    assert pm.tolist() == [[1, -1, 2, 0]], pm
    bt = {k: torch.arange(4.0)[None, :, None].expand(1, 4, 2).clone() for k in JK.INPUTS if k != "valid"}
    bt["valid"] = torch.ones(1, 4, dtype=torch.bool)
    ro = E.reorder(bt, pm)
    assert ro["pos"][0, :, 0].tolist() == [1.0, 0.0, 2.0, 0.0] and ro["valid"].tolist() == [[True, False, True, True]], ro
    # parity: the valid tokens only one side has
    torch.manual_seed(0)
    m = JK.JTokenHead().eval()
    m.set_step(JK.SWITCH)
    sb = JK.synthetic_batch(2, 48, seed=3)
    cells = np.tile(np.arange(48), (2, 1))
    live = {k: sb[k].clone() for k in JK.INPUTS}
    live["cell"] = torch.from_numpy(cells.copy())
    ok, num = E.compare_sets(m, sb, cells, live, 10.0)
    assert ok and num["unmatched_share"] == 0.0 and num["max_dp"] == 0.0, num
    live["tok"] = live["tok"] + 1e-3                                                                 # a small difference in the inputs: probabilities agree within the tolerance
    assert E.compare_sets(m, sb, cells, live, 0.05)[0]
    live["cell"][:, :5] = 1000 + torch.arange(5)                                                      # 5 tokens per frame are other cells: they are valid on one side only
    ok, num = E.compare_sets(m, sb, cells, live, 10.0)
    assert not ok and num["unmatched_share"] > 10 * E.UNMATCHED_LIMIT and num["valid_cache_not_live"] > 0 and num["valid_live_not_cache"] > 0, num
    # gpu_free: this process and its parents never count as another run.py job
    fake = lambda out: types.SimpleNamespace(stdout=out)  # noqa: E731
    mine = "\n".join(str(x) for x in (os.getpid(), os.getppid()))
    with patched((E.subprocess, "run", lambda cmd, **k: fake("500\n") if cmd[0] == "nvidia-smi" else fake(mine + "\n"))):
        assert E.gpu_free()[0], E.gpu_free()
    with patched((E.subprocess, "run", lambda cmd, **k: fake("500\n") if cmd[0] == "nvidia-smi" else fake(mine + "\n999999\n"))):
        ok, why = E.gpu_free()
        assert not ok and "999999" in why and str(os.getpid()) not in why, why
    with patched((E.subprocess, "run", lambda cmd, **k: fake("2500\n") if cmd[0] == "nvidia-smi" else fake(""))):
        assert not E.gpu_free()[0]
    # stamped result files, the dump identity, the control card names
    with tempfile.TemporaryDirectory() as d:
        a1, a2 = E._write_stamped(d, "x", dict(v=np.float32(1.5))), E._write_stamped(d, "x", dict(v=2))
        assert a1 != a2 and json.load(open(a1)) == dict(v=1.5) and len(os.listdir(d)) == 2, "a result file is never skipped or overwritten"
        open(os.path.join(d, "f.bin"), "wb").write(b"abc")
        fid = E._file_id(os.path.join(d, "f.bin"))
        assert fid["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" and fid["mtime"].endswith("UTC"), fid
        assert E.ctl_name("hi", "e292_a", 0, 384) == "hi_e292_a_s0_K384.json" and E.ctl_name("hi", "e292_a", 0, 384, 3) == "hi_e292_a_s0_K384_n3.json"
        card = lambda K: dict(metrics=dict(topo_F1=0.8), info=dict(K=K))  # noqa: E731
        json.dump(card(384), open(os.path.join(d, "hi_e292_a_s0_K384.json"), "w"))
        json.dump(card(384), open(os.path.join(d, "rule_e292_a_s0.json"), "w"))                      # an earlier card without K in its name counts only when its K is this one
        assert E.read_ctl("hi", "e292_a", 0, 384, d) == 0.8 and E.read_ctl("hi", "e292_a", 0, 512, d) is None and E.read_ctl("rule", "e292_a", 0, 384, d) == 0.8 and E.read_ctl("rule", "e292_a", 0, 512, d) is None
    # layer 1 stores plain B's records only when it passes; score refuses without them
    with tempfile.TemporaryDirectory() as d:
        base = {k: 0.5 for k in E.KEYS}
        json.dump(dict(variants=dict(primary=dict(numbers=dict(base=base)))), open(os.path.join(d, "card_s{seed}.json".format(seed=9)), "w"))
        res = lambda off: dict(graph={k: v + off for k, v in base.items()}, dump=dict(path="x", sha256="0", mtime="t"))  # noqa: E731
        with patched((E, "BASE_CARD", os.path.join(d, "card_s{seed}.json")), (E, "SCRATCH", d), (E, "plain_b", lambda seed, dump=None: res(0.0))):
            assert E.check_layer1(types.SimpleNamespace(seed=9, dump=None)) == 0 and os.path.exists(E.planb_path(9)) and E.stored_plain_b(9)["dump"]["sha256"] == "0"
            os.remove(E.planb_path(9))
            with patched((E, "plain_b", lambda seed, dump=None: res(1e-3))):
                assert E.check_layer1(types.SimpleNamespace(seed=9, dump=None)) == 1 and not os.path.exists(E.planb_path(9)), "a FAIL stores nothing"
            must_fail(E.stored_plain_b, 9)
    print("numbers: layer-1 comparison at 1e-6, g1, the samples, parity inside / outside its tolerance, the matching of two token sets by cell, the unmatched-token limit (a perturbed live set FAILs), gpu_free without this process and its parents, "
          "stamped result files, the dump identity, control card names with K, layer 1 stores plain B only on a PASS: ok")


def test_parity_timing():
    if not CACHE_OK or not os.path.exists(SMOKE):
        print("parity_timing: SKIPPED (no cache or no smoke model)")
        return
    b_run = os.path.join(E.ROOT, "runs", "e292_a_s0")
    if not os.path.exists(os.path.join(b_run, "model.pt")):
        print("parity_timing: SKIPPED (no B' run)")
        return
    import glob
    with tempfile.TemporaryDirectory() as runs:
        os.makedirs(os.path.join(runs, "h"))
        torch.save(torch.load(SMOKE, map_location="cpu", weights_only=False), os.path.join(runs, "h", "model.pt"))
        rc = E.check_parity(E.parse_args(["check-parity", "--run", "h", "--seed", "0", "--n", "2", "--tol", "0.05"]), runs)
        (pf,) = glob.glob(os.path.join(runs, "h", "token_parity_*.json"))
        par = json.load(open(pf))
        assert rc in (0, 1) and par["frames"] == 2 and par["ok"] == (rc == 0) and "unmatched_share" in par and par["input_max_abs_diff_matched_valid_tokens"]["tok"] < 0.5, par
        K = E.open_cache("val", 0).K
        t_args = ["timing", "--run", "h", "--cpu", "--frames", "2", "--repeats", "1"]
        E.timing(E.parse_args(t_args + ["--b-run", os.path.relpath(b_run, E.ROOT)]), runs)
        E.timing(E.parse_args(t_args), runs)
        files = sorted(glob.glob(os.path.join(runs, "h", "token_timing_cpu_*.json")))
        assert len(files) == 2, files                                          # a second call never skips because a file exists
        ts = [json.load(open(f)) for f in files]
        t = next(x for x in ts if "whole" in x["median_ms"])
        assert all(x["K"] == K and x["quiet"] in (True, False) and x["usable_for_gcost"] is False for x in ts), ts            # K = the cache's K by default; a CPU result is never evidence for G-cost
        assert set(t["median_ms"]) == {"head", "bprime", "whole"} and t["params"] == JK.count_params(JK.JTokenHead()) and 0.05 < t["gmacs_head"] < 20 and t["device"] == "cpu" and t["params_whole_path"] == t["params"] + t["params_bprime"] < 2.74e6, t
        must_fail(E.timing, E.parse_args(t_args + ["--K", str(K + 1)]), runs)                                         # K above the cache's K
        print(f"parity_timing: parity on 2 VAL frames (rc {rc}: dp_none {par['max_dpnone']:.2e}, dp {par['max_dp']:.2e}, unmatched {par['unmatched_share']:.3f}, tok diff {par['input_max_abs_diff_matched_valid_tokens']['tok']:.1e}); timing code path on the CPU, "
              f"K {K}: head {t['median_ms']['head']:.0f} ms, whole {t['median_ms']['whole']:.0f} ms, {t['params']} parameters, {t['gmacs_head']:.2f} GMACs, two stamped files: ok")


def test_sweep():
    """the sweep on 3 cache frames with the untrained head (it keeps nodes and links): 24 cells, one is the record, the numbers move monotonically in the counts, the table is never overwritten"""
    import shutil
    if not CACHE_OK or not os.path.exists(SMOKE):
        print("sweep: SKIPPED (no cache or no smoke model)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        runs, out = os.path.join(tmp, "runs"), os.path.join(tmp, "out")
        os.makedirs(os.path.join(runs, "h"))
        os.makedirs(out)
        ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
        ck["state"] = E.floor_model().state_dict()
        torch.save(ck, os.path.join(runs, "h", "model.pt"))
        E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
        import pickle
        dp = os.path.join(runs, "h", "token_val_n3.pkl")
        d = pickle.load(open(dp, "rb"))
        planted = 0
        for f, t in zip(d["frames"], [E.open_cache("val", 0)[i]["tokens_np"] for i in range(3)]):        # p_none in [0.5, 0.7) on 12 valid tokens per frame: the keep threshold changes the kept nodes; keep and record stay consistent
            v = np.flatnonzero(t["valid"])[:12]
            f["pnone"][v] = np.where(np.arange(len(v)) % 2, np.float32(0.65), np.float32(0.55))
            f["keep"] = t["valid"] & (f["pnone"] < np.float32(0.5))
            f["rec"] = E.decode_variant(f, t["valid"])["rec"]
            planted += len(v)
        pickle.dump(d, open(dp, "wb"))
        try:                                                                                                # --split is required (argparse exits)
            E.parse_args(["sweep", "--run", "h", "--seed", "0", "--max", "3"])
        except SystemExit:
            pass
        else:
            raise RuntimeError("a sweep without --split was accepted")
        path = E.sweep(E.parse_args(["sweep", "--run", "h", "--seed", "0", "--split", "val", "--max", "3"]), runs, out)
        tab = json.load(open(path))
        rows = tab["rows"]
        assert len(rows) == 24 and sum(r["record"] for r in rows) == 1 and all(r["frames"] == 3 for r in rows), rows
        assert (tab["every"], tab["max"], tab["drop_seed"]) == (1, 3, 0) and tab["dump"]["sha256"] and "steps" in tab, tab.keys()
        cell = lambda lt, kt, g: next(r for r in rows if (r["link_thr"], r["keep_thr"], r["none_gate"]) == (lt, kt, g))  # noqa: E731
        for kt in E.SWEEP_KEEP:
            for g in (True, False):
                lp = [cell(lt, kt, g)["links_per_frame"] for lt in E.SWEEP_LINK]
                assert all(x <= y for x, y in zip(lp, lp[1:])), "a lower link threshold only adds links"
            assert all(cell(lt, kt, True)["links_per_frame"] <= cell(lt, kt, False)["links_per_frame"] for lt in E.SWEEP_LINK), "no none gate only adds links"
        for lt in E.SWEEP_LINK:
            kn = [cell(lt, kt, True)["kept_nodes_per_frame"] for kt in E.SWEEP_KEEP]
            assert all(x < y for x, y in zip(kn, kn[1:])), "the planted p_none 0.55 / 0.65: each higher keep threshold adds nodes"
        assert len({r["topo_F1"] for r in rows}) > 1, "the variants differ in the score"
        must_fail(E.sweep, E.parse_args(["sweep", "--run", "h", "--seed", "0", "--split", "val", "--max", "3"]), runs, out)
        print(f"sweep: 24 cells on 3 frames of an untrained head with {planted} planted p_none in [0.5, 0.7) (links per frame {min(r['links_per_frame'] for r in rows):.0f} - {max(r['links_per_frame'] for r in rows):.0f}), counts monotone in both thresholds and the gate, "
              "kept nodes rise strictly with the keep threshold, the record cell marked, --split required, the file names every / max / drop seed / steps / the dump's sha256, never overwritten: ok")


def test_every():
    """--every N: the sample, the file names, the meta of the dump, and score / sweep reading the subset back (frames 0, 3, 6 of VAL with --max 3; the TRAIN sample every third = 291 frames)"""
    import pickle
    import shutil
    s = E.sample(8414, "train")
    assert len(E.sample(8414, "train", 3)) == 291 and E.sample(8414, "train", 3).tolist() == s[::3].tolist() and E.sample(873, "val", 3).tolist()[:3] == [0, 3, 6] and E.sample(873, "val", 1).tolist() == list(range(873))
    if not CACHE_OK or not os.path.exists(SMOKE) or not os.path.exists(E.planb_path(0)):
        print("every: the sample ok, the dump / score / sweep part SKIPPED (no cache, smoke model or stored plain B)")
        return
    import qgraph_eval as QE
    ds = E.open_cache("val", 0)
    with tempfile.TemporaryDirectory() as tmp:
        runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
        out = os.path.join(scratch, "out")
        os.makedirs(os.path.join(runs, "h"))
        os.makedirs(out)
        ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
        ck["state"] = E.floor_model().state_dict()                                                       # an untrained head that keeps nodes and links
        torch.save(ck, os.path.join(runs, "h", "model.pt"))
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        with patched((E, "SCRATCH", scratch)):
            E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--every", "3", "--max", "3"]), runs)
            path = os.path.join(runs, "h", "token_val_every3_n3.pkl")
            d = pickle.load(open(path, "rb"))
            assert d["meta"]["every"] == 3 and d["meta"]["idx"] == [0, 3, 6] and d["meta"]["tokens"] == [ds.tokens[i] for i in (0, 3, 6)] and [f["token"] for f in d["frames"]] == d["meta"]["tokens"], d["meta"]
            must_fail(E.dump, E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--every", "3", "--max", "3"]), runs)       # never overwritten
            cards = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--every", "3", "--max", "3"]), runs, out)
            assert sorted(os.listdir(out)) == ["h_token_card_argmax_every3_n3.json", "h_token_card_rec_every3_n3.json"], os.listdir(out)
            fr = QE.val_frames()
            want = E.score_preds([E.frame_prediction(ds[i]["tokens_np"], f, "rec") for i, f in zip((0, 3, 6), d["frames"])], [fr[i] for i in (0, 3, 6)], log=lambda *a, **k: None)["graph"]
            c = cards["rec"]
            assert c["frames"] == 3 and all(abs(c["metrics"][k] - want[k]) < 1e-12 for k in E.KEYS) and "topo_F1" in c["info"]["paired_vs_plain_b"] and c["info"]["g1_topo_F1"] is None, "the card scores frames 0, 3, 6"
            pb = QE.graph_numbers([E.stored_plain_b(0)["rec"]["graph"][i] for i in (0, 3, 6)])
            assert all(c["info"]["plain_b"][k] == pb[k] for k in E.KEYS) and "paired_vs_bsec0_close" not in c["info"], "plain B on frames 0, 3, 6; no B' records of the first frames indexed by the subsample"
            tab = json.load(open(E.sweep(E.parse_args(["sweep", "--run", "h", "--seed", "0", "--split", "val", "--every", "3", "--max", "3"]), runs, out)))
            assert os.path.exists(os.path.join(out, "h_token_sweep_every3_n3.json")) and tab["frames"] == 3 and len(tab["rows"]) == 24
            rec = next(r for r in tab["rows"] if r["record"])
            assert abs(rec["topo_F1"] - want["topo_F1"]) < 1e-12, "the record cell of the sweep is the record card on the same frames"
            v = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--every", "3", "--max", "3", "--link-thr", "0.1", "--keep-thr", "0.6", "--no-none-gate"]), runs, out)["rec"]
            row = next(r for r in tab["rows"] if (r["link_thr"], r["keep_thr"], r["none_gate"]) == (0.1, 0.6, False))
            assert all(abs(v["metrics"][k] - row[k]) < 1e-12 for k in E.SWEEP_KEYS), "a variant card equals its sweep cell"
            assert all(abs(v["info"]["record"][k] - want[k]) < 1e-12 for k in E.KEYS) and "topo_F1" in v["info"]["paired_vs_record"] and v["info"]["record_vs_stored_card"]["topo_F1"] == 0.0, "the record beside the variant"
    print("every: the TRAIN sample every third = 291 frames; dump --every 3 --max 3 takes VAL frames 0, 3, 6 (file token_val_every3_n3.pkl, meta every / idx / tokens, never overwritten), score and sweep read it back (names ..._every3_n3), the card and the sweep's record cell equal a direct scoring of frames 0, 3, 6, "
          "plain B is read at frames 0, 3, 6, a variant card equals its sweep cell and carries the record of the same dump (paired, against the stored record card): ok")


def test_score_max():
    """dump then score of 3 frames, end to end on a smoke head: the controls of `ctl --max 3` (bsec0_close: the pickle for the paired interval) and planted hi / rule cards, g1 against the hand formula"""
    import pickle
    import shutil
    if not CACHE_OK or not os.path.exists(SMOKE) or not os.path.exists(E.planb_path(0)):
        print("score_max: SKIPPED (no cache, smoke model or stored plain B)")
        return
    K = E.open_cache("val", 0).K
    with tempfile.TemporaryDirectory() as tmp:
        runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
        out = os.path.join(scratch, "out")
        os.makedirs(os.path.join(runs, "h"))
        os.makedirs(out)
        torch.save(torch.load(SMOKE, map_location="cpu", weights_only=False), os.path.join(runs, "h", "model.pt"))
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        with patched((E, "SCRATCH", scratch)):
            E.ctl(E.parse_args(["ctl", "--seed", "0", "--ident", "e292_a", "--arms", "bsec0_close", "--max", "3"]))
            assert os.path.exists(os.path.join(out, f"bsec0_close_e292_a_s0_K{K}_n3.json")) and os.path.exists(E.bsec0_close_path("e292_a", 0, 3))
            for arm, v in (("hi", 0.8), ("rule", 0.5)):                                                     # planted controls
                json.dump(dict(metrics=dict(topo_F1=v), info=dict(K=K)), open(os.path.join(out, E.ctl_name(arm, "e292_a", 0, K, 3)), "w"))
            E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
            cards = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
            for dec, c in cards.items():
                assert c["frames"] == 3 and os.path.exists(os.path.join(out, f"h_token_card_{dec}_n3.json")) and not os.path.exists(os.path.join(runs, "h", f"token_card_{dec}.json"))
                want = (c["metrics"]["topo_F1"] - 0.5) / (0.8 - 0.5)                                         # g1 = (S1 - c-rule) / (c-hi - c-rule), by hand
                assert abs(c["info"]["g1_topo_F1"] - want) < 1e-12 and "exceed 1" in c["info"]["g1_note"], c["info"]["g1_topo_F1"]
                assert "topo_F1" in c["info"]["paired_vs_plain_b"] and "topo_F1" in c["info"]["paired_vs_bsec0_close"] and "plain_b_dump" in c["info"]
                assert c["protocol"]["reference"].startswith("val")
            v = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3", "--link-thr", "0.1", "--keep-thr", "0.6", "--no-none-gate"]), runs, out)
            assert list(v) == ["rec"] and os.path.exists(os.path.join(out, "h_token_card_rec_lt0.1_kt0.6_nng_n3.json")), (list(v), os.listdir(out))
            dv = v["rec"]["info"]["decode_variant"]
            assert (dv["link_thr"], dv["keep_thr"], dv["none_gate"]) == (0.1, 0.6, False) and "g1_topo_F1" in v["rec"]["info"], dv
            for f in os.listdir(out):
                if f.startswith("h_token_card"):
                    os.remove(os.path.join(out, f))
            d = pickle.load(open(os.path.join(runs, "h", "token_val_n3.pkl"), "rb"))                       # a dump of another run is refused
            d["meta"]["run"] = "other"
            pickle.dump(d, open(os.path.join(runs, "h", "token_val_n3.pkl"), "wb"))
            try:
                E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
            except AssertionError as e:
                assert "not the dump of this read-out" in str(e), e
            else:
                raise RuntimeError("a dump of another run was scored")
            os.remove(os.path.join(scratch, "planb_s0.pkl"))                                                # without the layer-1 record score refuses
            must_fail(E.score, E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
    print("score_max: ctl --max 3 (bsec0_close pickle, K in the file name), dump --max 3, score --max 3 with planted controls: g1 equals the hand formula, the paired intervals vs plain B and vs bsec0_close, the cards in the out folder; a dump of another run and a missing plain B record are refused: ok")


TESTS = [test_keep_rule, test_decode_fork, test_decode_variant, test_dump_roundtrip, test_writer_no_gt, test_oracle_hi, test_drop, test_controls, test_readout_dump, test_numbers, test_parity_timing, test_sweep, test_every, test_score_max]

if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtoken_eval ok")
