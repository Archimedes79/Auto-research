"""Tests of `jtoken_eval.py` (E365 S1 read-out and checks); CPU only (the GPU is hidden), 2 threads. Hand frames need no data; the tests marked (cache) read the first VAL frames of the predicted-node cache of B' seed 0 under
runs/e365_cache (built by the lane chat) and are skipped with a message when it is not there.

    wsl bash tools/env/_py14.sh test_jtoken_eval.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

decode_variant (the defaults = the record decode, thresholds and the none gate by hand, monotone), decode_perclass (E375 link_thr_long: None = one threshold, each class threshold moves only its class, the class = the token chord; textual
mutants), sweep (the per-class grid, the only one: 36 cells, counts monotone per class, one row = a direct scoring), keep_rule (a node is kept at p_none 0.49 and not at 0.50, e = 1 - p_none, the links of a dropped node go), decode_fork (rec keeps both branches of a fork, argmax one; the none slot wins when the branches are weak),
dump_roundtrip (a real head forward: the dump holds exactly the decodes of `jtoken`, the candidate list, the keep mask), writer_no_gt (the prediction is made from the tokens only: mutants that put a GT section, a GT flag, a GT
position or an extra GT field into the writer are caught by `check_writer`; the dump frame and `frame_prediction` have no GT input), drop (cache: reproducible, the seed matters, the share, equal to the loader's own `token_drop` at the same seed),
readout_dump (B's stored dump frame -> prediction: B's own sections, the close-node step drops the weaker of two close nodes), numbers (the layer-1 comparison at its tolerance, the sample, the paired junction F1 on a hand case with the draws of
`qgraph_eval.paired`, --ref-cell), every (cache: --every 3 through dump / score / sweep; a per-class card with its reference cell (another value in every field) and the paired topo / junction F1), timing (cache: the timing code path on a tiny
head, CPU: stamped files, K from the cache), score_max (cache: `dump --max` / `score --max` end to end), graphs (pass1 schema, systems added,
a duplicate refused, the per-frame table; frame-order and GT-field mutants; a kill between the two renames completed by a rerun), snapshot (cache: --model model_step<N>.pt through dump / score / sweep / graphs, _step<N> in
every file name, a dump of the other weights refused), ref_run (E376 score --ref-run: other frames / cells / valid tokens refused; cache: the run itself as reference = diff 0, interval [0, 0] at a variant cell; --ref-run-cell at
the card's own cell = the default, at another cell = --ref-cell of that cell; an empty reference = the card's own numbers; the _vs_<run>[_rc<cell>] names beside the card without them; refused: another cache seed, another cache
of the same seed (`same_tokens` end to end), --model with --ref-run), ensemble (E379: the means of a hand pair of dump frames by hand, 0 for a pair one head did not write, the scales over the heads that wrote it, the decode
by hand, the floor bias by hand, the card names, 5 mutants; cache: h + h-without-candidates at half the thresholds = h, refusals incl. another opened cache), choose_cell (E375's rule of E377 / E378 on hand tables, 7 mutants,
the command's names and its never-overwritten rule), anatomy (E376: a hand frame with every category, the medians, the AP with ties, the recall ceiling and the existence AP by hand; the mass concentration by hand
and on a stub head with a hand y; textual mutants: dup / wrong swapped, false_end on both ends, ties broken optimistically, the existence label from p_none, the max over all candidates, G without the class gate; cache: the
command on 3 frames, delivered = the decode's long links, the near_none section's none tokens = the cache's, its degree >= 3 tokens and kept none tokens = rebuilt at the command's thresholds (keep 0.6 and 0.2), never
overwritten), near_none (E377, the anatomy's section: a hand frame with a twin at 1.0 m, one at exactly 2.0 m, one at 2.01 m, one at 1.0 m with heading cosine 0.4, a near and a far duplicate, a token near a none token only,
padding and a link below its source's p(none); every count, share and median by hand; textual mutants of `jtoken.near_none` read through the anatomy (radius <, heading test dropped, assigned test dropped), the duplicates
from the row of y, the none gate on), argmax_keep (E376 score --argmax --keep-thr: the rebuilt argmax = the dump's on a real head and a hand frame with ties and
p_none at the threshold, a winner below the floor refused above the record keep, mutants; cache: keep 0.5 = the stored argmax card, keep 0.6 = a direct scoring, self-pairs with --ref-run at the rec / argmax / argmax kt0.6 decodes),
live (E372 dump / score / sweep / anatomy --live, --ref-cache, live-diff, `check_b`: a stub source with the cache's tokens gives the cache's records and read-outs; a joint run's own tokens, frame pairing, B' identity refusals; see its docstring).
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


def floor_model():
    """the untrained head (random init, seed 0) in the mode of a trained one (the key weight of its own p_none): a stand-in head that keeps nodes and links"""
    torch.manual_seed(0)
    m = JK.JTokenHead()
    m.set_step(JK.SWITCH)
    return m.eval()


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


def _mutant(fn, old, new):
    """the function `fn` (of jtoken_eval or another module) with one text replaced (found exactly once), compiled in its module's namespace: a textual mutant"""
    import inspect
    src = inspect.getsource(fn)
    assert src.count(old) == 1, old
    mod = sys.modules[fn.__module__]
    ns = dict(vars(mod))
    exec(compile(src.replace(old, new), mod.__file__, "exec"), ns)
    return ns[fn.__name__]


def _by_class(d):
    """the links of a decoded frame -> (short set, long set), by the dump's class"""
    s, t, lg = d["rec"]["src"].tolist(), d["rec"]["dst"].tolist(), d["rec"]["long"].tolist()
    return {(a, b) for a, b, x in zip(s, t, lg) if not x}, {(a, b) for a, b, x in zip(s, t, lg) if x}


def test_decode_perclass():
    """E375: link_thr_long. A hand frame with known answers (None = link_thr = the old one-threshold decode on all fields; a long threshold of 0.05 only adds long links; a short threshold only moves short links; below 0.05 refused); a real
    head (random weights, none gate off): the same on every frame, and the dump's class is the token chord > 8 m; textual mutants (the long threshold on the short links, the class read from a GT-chord field, None read as the record's
    threshold) are caught"""
    f32 = np.float32
    short, long_ = [(0, 1, 0.35), (1, 2, 0.25), (2, 3, 0.12)], [(0, 4, 0.35), (1, 5, 0.25), (2, 5, 0.08)]
    c = short + long_
    th = dict(token="t", pnone=np.full(6, 0.1, f32), none_cls=np.zeros((2, 6), f32),
              cand=dict(src=np.array([x[0] for x in c], np.int16), dst=np.array([x[1] for x in c], np.int16), p=np.array([x[2] for x in c], f32), long=np.array([0, 0, 0, 1, 1, 1], bool),
                        a=np.full(6, THIRD, f32), b=np.full(6, THIRD, f32), gt_long=np.array([0, 1, 0, 1, 1, 0], bool)))       # gt_long: a GT-chord class lying around (1 -> 2 long by GT, 2 -> 5 short by GT)
    v6 = np.ones(6, bool)
    one = E.decode_variant(th, v6, 0.3)
    pairs = ((E.decode_variant(th, v6, 0.3, link_thr_long=0.3), one), (E.decode_variant(th, v6), one), (E.decode_variant(th, v6, 0.1), E.decode_variant(th, v6, 0.1, link_thr_long=0.1)))
    for i, (same, ref) in enumerate(pairs):                                                                  # None = link_thr = one threshold on all fields, at the record's 0.3 and at 0.1 (not the record's)
        assert np.array_equal(same["keep"], ref["keep"]) and all(np.array_equal(same["rec"][k], ref["rec"][k]) for k in E.LINK_FIELDS), i
    got = lambda fn=E.decode_variant, **k: _by_class(fn(th, v6, **k))  # noqa: E731
    assert got(link_thr=0.3) == ({(0, 1)}, {(0, 4)}), got(link_thr=0.3)
    assert got(link_thr=0.1, link_thr_long=0.3) == ({(0, 1), (1, 2), (2, 3)}, {(0, 4)}), "a short threshold of 0.1 adds the short links only"
    must_fail(E.decode_variant, th, v6, 0.3, link_thr_long=0.04)
    cases = (dict(link_thr=0.3, link_thr_long=0.05), dict(link_thr=0.1))                                     # a long threshold of 0.05 adds the long links only; one threshold 0.1 (None) on both classes
    answers = [({(0, 1)}, {(0, 4), (1, 5), (2, 5)}), ({(0, 1), (1, 2), (2, 3)}, {(0, 4), (1, 5)})]
    assert [got(**k) for k in cases] == answers, [got(**k) for k in cases]
    mutants = {"the long threshold on the short links": ("np.array([link_thr, thr_long], np.float32)", "np.array([thr_long, thr_long], np.float32)"),
               "the class by the GT chord": ('[c["long"].astype(int)]                 #', '[c["gt_long"].astype(int)]                 #'),
               "None = the record's threshold": ("thr_long = link_thr if", "thr_long = LINK_THR if")}
    for old, new in mutants.values():
        assert [got(_mutant(E.decode_variant, old, new), **k) for k in cases] != answers, new
    # a real head (random weights), none gate off: per frame, the long threshold moves only long links, the short threshold only short ones; the dump's class is the token chord
    torch.manual_seed(0)
    m = JK.JTokenHead().eval()
    m.set_step(JK.SWITCH)
    sb = JK.synthetic_batch(3, 48, seed=2)
    toks = [dict(pos=sb["pos"][b].double().numpy(), cell=np.arange(48)) for b in range(3)]
    frames = E.batch_dump(E.run_head(m, sb, "cpu"), dict(token=["a", "b", "c"], row=torch.arange(3), tokens_np=toks))
    added = moved = 0
    for fr_, val, t in zip(frames, sb["valid"].numpy(), toks):
        cd = fr_["cand"]
        chord = np.linalg.norm(t["pos"][cd["dst"].astype(int)] - t["pos"][cd["src"].astype(int)], axis=1)
        assert np.array_equal(cd["long"][np.abs(chord - JT.SHORT) > 1e-3], (chord > JT.SHORT)[np.abs(chord - JT.SHORT) > 1e-3]), "the dump's class is the token chord > 8 m"
        dec = lambda lt, ltl: _by_class(E.decode_variant(fr_, val, lt, 0.5, False, ltl))  # noqa: E731
        (s3, l3), (s3w, l3w), (s1, l1) = dec(0.3, None), dec(0.3, 0.05), dec(0.1, 0.3)
        assert s3w == s3 and l3w >= l3 and l1 == l3 and s1 >= s3, "a class threshold moves the links of its class only"
        added, moved = added + len(l3w - l3), moved + len(s1 - s3)
    assert added > 0 and moved > 0, (added, moved)
    print(f"decode_perclass: a hand frame with known answers (None = link_thr = the one-threshold decode on all fields, at 0.3 and 0.1; long 0.05 adds the long links 1 -> 5, 2 -> 5 only; short 0.1 adds 1 -> 2, 2 -> 3 only; 0.04 refused); a real head on 3 frames "
          f"(long 0.05 adds {added} long links and no short one, short 0.1 adds {moved} short links and no long one; the class = the token chord); {len(mutants)} textual mutants caught ({', '.join(mutants)}): ok")


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


def test_readout_dump():
    nd = lambda e: dict(row=7, e=np.array(e), pos=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 6.0]]), dirv=np.tile([0.0, 1.0], (3, 1)), flags=np.zeros((3, 8)), s=np.zeros(3), section=np.array([0, 3, 3]),  # noqa: E731
                       link_src=np.array([0, 1]), link_dst=np.array([1, 2]), link_p=np.array([0.9, 0.9]), link_long=np.zeros(2, bool), link_a=np.full(2, 0.3), link_b=np.full(2, 0.3))
    d = nd([0.9, 0.8, 0.95])                                                           # nodes 0 and 1 are 1.0 m apart: the close-node step keeps the stronger (0)
    close = E.pred_of_dump(d, E.CLOSE)
    assert close["nodes"]["section"].tolist() == [0, 3, 3] and close["nodes"]["e"].tolist() == [0.9, 0.0, 0.95], close["nodes"]
    assert d["section"].tolist() == [0, 3, 3] and d["e"].tolist() == [0.9, 0.8, 0.95], "the stored dump is not modified"
    print("readout_dump: B's own sections, the close-node step drops the weaker of two close nodes without touching the dump: ok")


def test_numbers():
    ref = {k: 0.5 for k in E.CHECK}
    ok, diff = E.compare_numbers({k: 0.5 + 9e-7 for k in E.CHECK}, ref)
    assert ok and abs(diff["topo_F1"] - 9e-7) < 1e-12
    ok, diff = E.compare_numbers(dict({k: 0.5 for k in E.CHECK}, dup_frac=0.5 + 2e-6), ref)
    assert not ok and abs(diff["dup_frac"] - 2e-6) < 1e-12
    must_fail(E.compare_numbers, {k: 0.5 for k in E.CHECK[:-1]}, ref)                      # a missing number is an error, not a pass
    assert E.sample(873, "val").tolist() == list(range(873))
    s = E.sample(8414, "train")
    assert len(s) == 873 and s[0] == 0 and s[-1] == 8413 and (np.diff(s) > 0).all()
    # paired junction F1 (E375): counts pooled per draw, the scene draws of qgraph_eval.paired; two scenes of one frame each: a = [pred, GT, matched] [1, 1, 1] + [0, 3, 0] (pooled F1 0.4, the frame mean would be 0.5), b = [1, 1, 0] + [0, 3, 0]
    import qgraph_eval as QE
    jr = lambda *cs: dict(graph=[dict(lane=dict(junc=np.array(x, float)), rec=None, topo=None) for x in cs])  # noqa: E731
    fr2 = [types.SimpleNamespace(scene=s) for s in (0, 1)]
    draws, real_draws = [], E.ST.scene_draws

    def rec_draws(scenes, n=1000, seed=0):                                                             # records (scenes, B, seed) of every scene bootstrap
        draws.append((np.asarray(scenes).tolist(), n, seed))
        return real_draws(scenes, n, seed)
    with patched((E.ST, "scene_draws", rec_draws)):
        pj = E.paired_junction(jr([1, 1, 1], [0, 3, 0]), jr([1, 1, 0], [0, 3, 0]), fr2)["junction_F1"]
        with patched((E.ST, "paired_draws", lambda fa, fb, d, keys: {})):                              # qgraph_eval.paired on the same frames: only its draws are read
            QE.paired(jr([1, 1, 1], [0, 3, 0]), jr([1, 1, 0], [0, 3, 0]), fr2)
    assert len(draws) == 3 and draws[0] == draws[1] == draws[2], f"paired_junction and qgraph_eval.paired (bootstrap + topo) draw the same scenes, B and seed: {draws}"
    assert abs(pj["a"] - 0.4) < 1e-12 and pj["b"] == 0.0 and abs(pj["diff"] - 0.4) < 1e-12 and (pj["lo"], pj["hi"]) == (0.0, 1.0) and 0.7 < pj["p_a_better"] < 0.8, pj    # draws: {0,0} +1, {1,1} 0, mixed +0.4
    assert all(v["diff"] == v["lo"] == v["hi"] == 0.0 for v in E.paired_junction(jr([1, 1, 1], [0, 3, 0]), jr([1, 1, 1], [0, 3, 0]), fr2).values())
    assert E.parse_cell("0.2,0.05,0.6,off") == (0.2, 0.05, 0.6, False) and E.parse_args(["score", "--run", "h", "--seed", "0"]).ref_cell is None
    must_fail(E.parse_args, ["score", "--run", "h", "--seed", "0", "--link-thr", "0.2", "--ref-cell", "0.2,0.2,0.6,maybe"])
    must_fail(E.score, E.parse_args(["score", "--run", "h", "--seed", "0", "--ref-cell", "0.2,0.2,0.6,off"]))                   # a reference cell needs a variant (refused before anything is read)
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
    # stamped result files, the dump identity
    with tempfile.TemporaryDirectory() as d:
        a1, a2 = E._write_stamped(d, "x", dict(v=np.float32(1.5))), E._write_stamped(d, "x", dict(v=2))
        assert a1 != a2 and json.load(open(a1)) == dict(v=1.5) and len(os.listdir(d)) == 2, "a result file is never skipped or overwritten"
        open(os.path.join(d, "f.bin"), "wb").write(b"abc")
        fid = E._file_id(os.path.join(d, "f.bin"))
        assert fid["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" and fid["mtime"].endswith("UTC"), fid
    # layer 1 stores plain B's records only when it passes; score refuses without them
    with tempfile.TemporaryDirectory() as d:
        base = {k: 0.5 for k in E.KEYS}
        json.dump(dict(variants=dict(primary=dict(numbers=dict(base=base)))), open(os.path.join(d, "card_s{seed}.json".format(seed=9)), "w"))
        res = lambda off, sha="0": dict(graph={k: v + off for k, v in base.items()}, dump=dict(path="x", sha256=sha, mtime="t"))  # noqa: E731
        ns = lambda over=False: types.SimpleNamespace(seed=9, dump=None, overwrite=over)  # noqa: E731
        with patched((E, "BASE_CARD", os.path.join(d, "card_s{seed}.json")), (E, "PLANB", d), (E, "plain_b", lambda seed, dump=None: res(0.0))):
            assert E.check_layer1(ns()) == 0 and os.path.exists(E.planb_path(9)) and E.stored_plain_b(9)["dump"]["sha256"] == "0"
            with patched((E, "plain_b", lambda seed, dump=None: res(0.0, "1"))):
                assert E.check_layer1(ns()) == 0 and E.stored_plain_b(9)["dump"]["sha256"] == "0", "a stored record is kept unless --overwrite"
                assert E.check_layer1(ns(True)) == 0 and E.stored_plain_b(9)["dump"]["sha256"] == "1", "--overwrite replaces it"
            os.remove(E.planb_path(9))
            with patched((E, "plain_b", lambda seed, dump=None: res(1e-3))):
                assert E.check_layer1(ns(True)) == 1 and not os.path.exists(E.planb_path(9)), "a FAIL stores nothing"
            must_fail(E.stored_plain_b, 9)
    print("numbers: layer-1 comparison at 1e-6, the samples, the paired junction F1 on a hand case "
          "(pooled counts 0.4, interval 0 .. 1; a = b gives 0; the draws (scenes, B, seed) of qgraph_eval.paired), --ref-cell parsed and refused without a variant, gpu_free without this process and its parents, stamped result files, the dump identity, layer 1 stores "
          "plain B only on a PASS and keeps a stored record unless --overwrite: ok")


def test_timing():
    if not CACHE_OK or not os.path.exists(SMOKE):
        print("timing: SKIPPED (no cache or no smoke model)")
        return
    b_run = os.path.join(E.ROOT, "runs", "e292_a_s0")
    if not os.path.exists(os.path.join(b_run, "model.pt")):
        print("timing: SKIPPED (no B' run)")
        return
    import glob
    with tempfile.TemporaryDirectory() as runs:
        os.makedirs(os.path.join(runs, "h"))
        torch.save(torch.load(SMOKE, map_location="cpu", weights_only=False), os.path.join(runs, "h", "model.pt"))
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
        print(f"timing: the code path on the CPU, "
              f"K {K}: head {t['median_ms']['head']:.0f} ms, whole {t['median_ms']['whole']:.0f} ms, {t['params']} parameters, {t['gmacs_head']:.2f} GMACs, two stamped files: ok")


def _plant_pnone(path, toks):
    """p_none 0.55 / 0.65 on the first 12 valid tokens of every frame of the dump at `path` (toks: the cache tokens of its frames), keep and record made consistent, written back: the keep thresholds 0.5 / 0.6 / 0.7 keep other
    nodes (the untrained head keeps every valid token) -> (dump, planted tokens)"""
    import pickle
    d = pickle.load(open(path, "rb"))
    planted = 0
    for f, t in zip(d["frames"], toks):
        v = np.flatnonzero(t["valid"])[:12]
        f["pnone"][v] = np.where(np.arange(len(v)) % 2, np.float32(0.65), np.float32(0.55))
        f["keep"] = t["valid"] & (f["pnone"] < np.float32(0.5))
        f["rec"] = E.decode_variant(f, t["valid"])["rec"]
        planted += len(v)
    pickle.dump(d, open(path, "wb"))
    return d, planted


def test_sweep():
    """the sweep on 3 cache frames with the untrained head (it keeps nodes and links): the 36 cells of the per-class grid, the numbers move monotonically in the counts of each class, the table is never overwritten"""
    import shutil
    if not CACHE_OK or not os.path.exists(SMOKE):
        print("sweep: SKIPPED (no cache or no smoke model)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        runs, out = os.path.join(tmp, "runs"), os.path.join(tmp, "out")
        os.makedirs(os.path.join(runs, "h"))
        os.makedirs(out)
        ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
        ck["state"] = floor_model().state_dict()
        torch.save(ck, os.path.join(runs, "h", "model.pt"))
        E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
        toks = [E.open_cache("val", 0)[i]["tokens_np"] for i in range(3)]
        d, planted = _plant_pnone(os.path.join(runs, "h", "token_val_n3.pkl"), toks)
        sw = ["sweep", "--run", "h", "--seed", "0"]
        for bad in (sw + ["--max", "3"], sw + ["--split", "val", "--grid", "default"], ["ctl", "--seed", "0", "--ident", "e292_a"], ["fixture", "--seed", "0", "--split", "val"], ["check-parity", "--run", "h", "--seed", "0"]):
            try:                                                                                            # --split is required; the 24-cell grid and the commands ctl / fixture / check-parity are gone (D117; argparse exits)
                E.parse_args(bad)
            except SystemExit:
                continue
            raise RuntimeError(f"{bad} was accepted")
        grid = E.SWEEP_GRIDS["perclass"]                                                                    # E375: 36 cells, gate off, the champion's cell among them
        assert list(E.SWEEP_GRIDS) == ["perclass"] and len(grid) == 36 and set(grid) == {(lt, ltl, kt, False) for lt in (0.3, 0.2, 0.1) for ltl in (0.3, 0.2, 0.1, 0.05) for kt in (0.5, 0.6, 0.7)} and (0.3, 0.2, 0.6, False) in grid, grid
        pp = E.sweep(E.parse_args(["sweep", "--run", "h", "--seed", "0", "--split", "val", "--max", "3"]), runs, out)        # no --grid: the per-class grid
        pc = json.load(open(pp))
        prow = pc["rows"]
        pcell = {(r["link_thr"], r["link_thr_long"], r["keep_thr"]): r for r in prow}
        assert os.path.basename(pp) == "h_token_sweep_perclass_n3.json" and pc["grid"] == "perclass" and list(pcell) == [x[:3] for x in grid] and not any(r["none_gate"] or r["record"] for r in prow), list(pcell)
        assert (pc["every"], pc["max"], pc["drop_seed"]) == (1, 3, 0) and pc["dump"]["sha256"] and "steps" in pc and all(r["frames"] == 3 for r in prow), pc.keys()
        assert all(list(r) == ["link_thr", "link_thr_long", "keep_thr", "none_gate", "record", "frames", "kept_nodes_per_frame", "links_per_frame", *E.SWEEP_KEYS, *E.JUNCTION_KEYS, "junction_points_per_frame", "long_links_per_frame",
                               "node_link_long_P", "node_link_long_R"] for r in prow), "the per-class table keeps its rows and keys"
        assert all(np.isfinite(r[k]) for r in prow for k in E.JUNCTION_KEYS + ("junction_points_per_frame", "long_links_per_frame")), "the junction numbers and the long links of every row"
        import qgraph_eval as QE
        r0 = pcell[(0.1, 0.05, 0.6)]                                                                        # one row against a direct scoring of its decode
        d0 = E.score_preds([E.frame_prediction(t, E.decode_variant(f, t["valid"], 0.1, 0.6, False, 0.05), "rec", 0.6) for f, t in zip(d["frames"], toks)], QE.val_frames()[:3], log=lambda *a, **k: None)
        jn = np.sum([c["lane"]["junc"] for c in d0["rec"]["graph"]], 0)                                    # predicted, GT, matched junction points of the 3 frames (three different counts: a mix-up shows)
        assert len(set(jn.tolist())) == 3 and abs(r0["junction_points_per_frame"] * r0["frames"] - jn[0]) < 1e-9 and all(r0[k] == d0["graph"][k] for k in E.JUNCTION_KEYS), (jn, r0)
        assert all(r[f"node_link_long_{k}"] is not None for r in prow for k in "PR") and (r0["node_link_long_P"], r0["node_link_long_R"]) == (d0["nodes"]["link_long_P"], d0["nodes"]["link_long_R"]), r0
        n_short = lambda r: round((r["links_per_frame"] - r["long_links_per_frame"]) * r["frames"])  # noqa: E731
        n_long = lambda r: round(r["long_links_per_frame"] * r["frames"])  # noqa: E731
        S, L = E.PERCLASS_SHORT, E.PERCLASS_LONG
        for kt in E.SWEEP_KEEP:
            for ltl in L:
                ch = [pcell[(lt, ltl, kt)] for lt in S]
                assert all(n_short(x) <= n_short(y) for x, y in zip(ch, ch[1:])) and len({n_long(x) for x in ch}) == 1, "a lower short threshold only adds short links"
            for lt in S:
                ch = [pcell[(lt, ltl, kt)] for ltl in L]
                assert all(n_long(x) <= n_long(y) for x, y in zip(ch, ch[1:])) and len({n_short(x) for x in ch}) == 1, "a lower long threshold only adds long links"
        for lt in S:
            for ltl in L:
                kn = [pcell[(lt, ltl, kt)]["kept_nodes_per_frame"] for kt in E.SWEEP_KEEP]
                assert all(x < y for x, y in zip(kn, kn[1:])), "the planted p_none 0.55 / 0.65: each higher keep threshold adds nodes"
        added_s, added_l = (n_short(pcell[(0.1, 0.3, 0.6)]) - n_short(pcell[(0.3, 0.3, 0.6)]), n_long(pcell[(0.3, 0.05, 0.6)]) - n_long(pcell[(0.3, 0.3, 0.6)]))
        assert added_s > 0 and added_l > 0 and len({r["topo_F1"] for r in prow}) > 1, (added_s, added_l)
        must_fail(E.sweep, E.parse_args(["sweep", "--run", "h", "--seed", "0", "--split", "val", "--max", "3", "--grid", "perclass"]), runs, out)
        print(f"sweep: the per-class grid (36 cells, the only one: no --grid gives it, 'default' refused; the removed commands ctl / fixture / check-parity refused) on 3 frames of an untrained head with {planted} planted p_none in [0.5, 0.7), the rows and keys of before, a lower short / long threshold "
              f"adds links of its class only (keep 0.6: short 0.1 adds {added_s}, long 0.05 adds {added_l}), kept nodes rise strictly with the keep threshold, --split required, the file names every / max / drop seed / steps / the dump's sha256, "
              f"never overwritten, junction numbers per row (one row = a direct scoring: {jn[0]:.0f} predicted junction points, long-link P / R): ok")


def test_every():
    """--every N: the sample, the file names, the meta of the dump, and score / sweep reading the subset back (frames 0, 3, 6 of VAL with --max 3; the TRAIN sample every third = 291 frames)"""
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
        ck["state"] = floor_model().state_dict()                                                       # an untrained head that keeps nodes and links
        torch.save(ck, os.path.join(runs, "h", "model.pt"))
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        with patched((E, "PLANB", scratch)):
            E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--every", "3", "--max", "3"]), runs)
            path = os.path.join(runs, "h", "token_val_every3_n3.pkl")
            toks = [ds[i]["tokens_np"] for i in (0, 3, 6)]
            d, _ = _plant_pnone(path, toks)                                                                 # the keep threshold of a reference cell matters
            assert d["meta"]["every"] == 3 and d["meta"]["idx"] == [0, 3, 6] and d["meta"]["tokens"] == [ds.tokens[i] for i in (0, 3, 6)] and [f["token"] for f in d["frames"]] == d["meta"]["tokens"], d["meta"]
            must_fail(E.dump, E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--every", "3", "--max", "3"]), runs)       # never overwritten
            cards = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--every", "3", "--max", "3"]), runs, out)
            assert sorted(os.listdir(out)) == ["h_token_card_argmax_every3_n3.json", "h_token_card_rec_every3_n3.json"], os.listdir(out)
            fr = QE.val_frames()
            want = E.score_preds([E.frame_prediction(t, f, "rec") for t, f in zip(toks, d["frames"])], [fr[i] for i in (0, 3, 6)], log=lambda *a, **k: None)["graph"]
            c = cards["rec"]
            assert c["frames"] == 3 and all(abs(c["metrics"][k] - want[k]) < 1e-12 for k in E.KEYS) and "topo_F1" in c["info"]["paired_vs_plain_b"], "the card scores frames 0, 3, 6"
            pb = QE.graph_numbers([E.stored_plain_b(0)["rec"]["graph"][i] for i in (0, 3, 6)])
            assert all(c["info"]["plain_b"][k] == pb[k] for k in E.KEYS), "plain B on frames 0, 3, 6"
            with patched((E, "SWEEP_GRIDS", dict(perclass=[(0.3, 0.3, 0.5, True), (0.1, 0.1, 0.6, False)]))):          # the record cell and the cell of the variant card below
                tab = json.load(open(E.sweep(E.parse_args(["sweep", "--run", "h", "--seed", "0", "--split", "val", "--every", "3", "--max", "3"]), runs, out)))
            assert os.path.exists(os.path.join(out, "h_token_sweep_perclass_every3_n3.json")) and tab["frames"] == 3 and len(tab["rows"]) == 2
            rec = next(r for r in tab["rows"] if r["record"])
            assert abs(rec["topo_F1"] - want["topo_F1"]) < 1e-12, "the record cell of the sweep is the record card on the same frames"
            v = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--every", "3", "--max", "3", "--link-thr", "0.1", "--keep-thr", "0.6", "--no-none-gate"]), runs, out)["rec"]
            row = next(r for r in tab["rows"] if not r["record"])
            assert all(abs(v["metrics"][k] - row[k]) < 1e-12 for k in E.SWEEP_KEYS), "a variant card equals its sweep cell"
            assert all(abs(v["info"]["record"][k] - want[k]) < 1e-12 for k in E.KEYS) and "topo_F1" in v["info"]["paired_vs_record"] and v["info"]["record_vs_stored_card"]["topo_F1"] == 0.0, "the record beside the variant"
            assert "decode_variant" in v["info"] and "link_thr_long" not in v["info"]["decode_variant"] and "ref_cell" not in v["info"], "one threshold: the card of before"
            # E375: a per-class variant against a reference cell that differs from it in every field (and short != long: a swap shows)
            pv = ["score", "--run", "h", "--seed", "0", "--every", "3", "--max", "3", "--link-thr", "0.3", "--link-thr-long", "0.05", "--keep-thr", "0.6", "--no-none-gate"]
            w = E.score(E.parse_args(pv + ["--ref-cell", "0.1,0.2,0.5,on"]), runs, out)["rec"]
            assert os.path.exists(os.path.join(out, "h_token_card_rec_lt0.3_ltl0.05_kt0.6_nng_every3_n3.json")) and (w["info"]["decode_variant"]["link_thr"], w["info"]["decode_variant"]["link_thr_long"]) == (0.3, 0.05), w["info"]["decode_variant"]
            direct = lambda lt, ltl, kt, gate: E.score_preds([E.frame_prediction(t, E.decode_variant(f, t["valid"], lt, kt, gate, ltl), "rec", kt) for t, f in zip(toks, d["frames"])], [fr[i] for i in (0, 3, 6)], log=lambda *a, **k: None)  # noqa: E731
            dw, d1, d0 = direct(0.3, 0.05, 0.6, False), direct(0.3, None, 0.6, False), direct(0.1, 0.2, 0.5, True)
            assert all(abs(w["metrics"][k] - dw["graph"][k]) < 1e-12 for k in E.KEYS) and any(dw["graph"][k] != d1["graph"][k] for k in E.KEYS), "the card is the per-class decode (and differs from one threshold 0.3)"
            assert E.junction_numbers(dw["rec"]["graph"]) == {k: dw["graph"][k] for k in E.JUNCTION_KEYS}, "junction_numbers = the scorer's junction numbers"
            rc, pvr = w["info"]["ref_cell"], w["info"]["paired_vs_ref_cell"]
            assert (rc["link_thr"], rc["link_thr_long"], rc["keep_thr"], rc["none_gate"]) == (0.1, 0.2, 0.5, True) and all(rc[k] == d0["graph"][k] for k in E.KEYS + ("junction_P", "junction_R")), "the reference cell = a direct scoring of its decode"
            assert all(abs(pvr[k]["diff"] - (w["metrics"][k] - d0["graph"][k])) < 1e-12 and np.isfinite(pvr[k]["lo"]) and np.isfinite(pvr[k]["hi"]) for k in ("topo_F1", "junction_F1")), pvr
            must_fail(E.score, E.parse_args(pv[:9] + ["--link-thr", "0.1", "--link-thr-long", "0.1", "--keep-thr", "0.6", "--no-none-gate"]), runs, out)   # long = short: the name of the card v, never overwritten
    print("every: the TRAIN sample every third = 291 frames; dump --every 3 --max 3 takes VAL frames 0, 3, 6 (file token_val_every3_n3.pkl, meta every / idx / tokens, never overwritten), score and sweep read it back (names ..._every3_n3), the card and the sweep's record cell equal a direct scoring of frames 0, 3, 6, "
          "plain B is read at frames 0, 3, 6, a variant card equals its sweep cell and carries the record of the same dump (paired, against the stored record card); a per-class card (_ltl0.05) equals a direct scoring of the per-class decode and "
          "carries its reference cell (planted p_none; a cell that differs in every field = a direct scoring of its decode) with the paired topo / junction F1 (diff = card - reference); a long threshold equal to the short one is the card of before: ok")


def test_score_max():
    """dump then score of 3 frames, end to end on a smoke head: the cards in the out folder with the paired interval against plain B; a dump of another run and a missing plain B record refused"""
    import pickle
    import shutil
    if not CACHE_OK or not os.path.exists(SMOKE) or not os.path.exists(E.planb_path(0)):
        print("score_max: SKIPPED (no cache, smoke model or stored plain B)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
        out = os.path.join(scratch, "out")
        os.makedirs(os.path.join(runs, "h"))
        os.makedirs(out)
        torch.save(torch.load(SMOKE, map_location="cpu", weights_only=False), os.path.join(runs, "h", "model.pt"))
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        with patched((E, "PLANB", scratch)):
            E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
            cards = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
            for dec, c in cards.items():
                assert c["frames"] == 3 and os.path.exists(os.path.join(out, f"h_token_card_{dec}_n3.json")) and not os.path.exists(os.path.join(runs, "h", f"token_card_{dec}.json"))
                assert "topo_F1" in c["info"]["paired_vs_plain_b"] and "plain_b_dump" in c["info"]
                assert c["protocol"]["reference"].startswith("val")
            v = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3", "--link-thr", "0.1", "--keep-thr", "0.6", "--no-none-gate"]), runs, out)
            assert list(v) == ["rec"] and os.path.exists(os.path.join(out, "h_token_card_rec_lt0.1_kt0.6_nng_n3.json")), (list(v), os.listdir(out))
            dv = v["rec"]["info"]["decode_variant"]
            assert (dv["link_thr"], dv["keep_thr"], dv["none_gate"]) == (0.1, 0.6, False), dv
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
    print("score_max: dump --max 3, score --max 3: the paired interval vs plain B, the cards in the out folder; a dump of another run and a missing plain B record are refused: ok")


def test_graphs():
    """graphs: the delivered graphs of a dump in the schema of pass1.pkl (3 VAL frames of the untrained head), a second system appended, a duplicate name refused, the per-frame table pooled = the scorer's topo F1; mutants:
    a frame order swap and a GT field in a system record fail `check_graphs`"""
    import pickle
    if not CACHE_OK or not os.path.exists(SMOKE):
        print("graphs: SKIPPED (no cache or no smoke model)")
        return
    import qgraph_eval as QE
    with tempfile.TemporaryDirectory() as tmp:
        runs = os.path.join(tmp, "runs")
        os.makedirs(os.path.join(runs, "h"))
        ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
        ck["state"] = floor_model().state_dict()
        torch.save(ck, os.path.join(runs, "h", "model.pt"))
        E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
        out = os.path.join(tmp, "g", "graphs.pkl")
        os.makedirs(os.path.dirname(out))
        g_args = ["graphs", "--run", "h", "--seed", "0", "--max", "3", "--out", out]
        E.graphs(E.parse_args(g_args + ["--name", "S1"]), runs)
        E.graphs(E.parse_args(g_args + ["--name", "S1_v", "--link-thr", "0.1", "--link-thr-long", "0.05", "--keep-thr", "0.6", "--no-none-gate"]), runs)
        must_fail(E.graphs, E.parse_args(g_args + ["--name", "S1"]), runs)                                    # a name the file holds
        doc, tab = pickle.load(open(out, "rb")), json.load(open(os.path.join(tmp, "g", "perframe_table.json")))
        frames = QE.val_frames()[:3]
        E.check_graphs(doc, frames)
        r0 = doc["frames"][0]
        assert sorted(r0) == ["gt", "k", "man_pts", "n_man", "row", "scene", "sys", "token"] and sorted(r0["sys"]) == ["S1", "S1_v"] and sorted(r0["gt"][0]) == ["kinds", "pts", "tier", "vis"] and r0["gt"][0]["tier"].dtype == np.int8
        ds = E.open_cache("val", 0)
        d = pickle.load(open(os.path.join(runs, "h", "token_val_n3.pkl"), "rb"))
        want = E.score_preds([E.frame_prediction(ds[i]["tokens_np"], f, "rec") for i, f in enumerate(d["frames"])], frames, log=lambda *x, **k: None)["graph"]["topo_F1"]
        want_v = E.score_preds([E.frame_prediction(ds[i]["tokens_np"], E.decode_variant(f, ds[i]["tokens_np"]["valid"], 0.1, 0.6, False, 0.05), "rec", 0.6) for i, f in enumerate(d["frames"])], frames, log=lambda *x, **k: None)["graph"]["topo_F1"]
        assert abs(tab["sys"]["S1"]["pooled_topo_F1"] - want) < 1e-12 and len(tab["sys"]["S1"]["topo_F1"]) == 3 and tab["sys"]["S1_v"]["decode"] == "rec_lt0.1_ltl0.05_kt0.6_nng", tab["sys"]["S1"]
        assert abs(tab["sys"]["S1_v"]["pooled_topo_F1"] - want_v) < 1e-12, "the variant's graphs are the per-class decode"
        swapped = dict(doc, frames=[doc["frames"][1], doc["frames"][0], doc["frames"][2]])
        must_fail(E.check_graphs, swapped, frames)
        leak = pickle.loads(pickle.dumps(doc))
        leak["frames"][1]["sys"]["S1"]["tier"] = np.zeros(3, np.int8)
        must_fail(E.check_graphs, leak, frames)
        real_replace = os.replace

        def killed_at_pickle(src, dst):                                                                     # a kill between the two renames
            if dst == out:
                raise OSError("killed")
            real_replace(src, dst)
        with patched((E.os, "replace", killed_at_pickle)):
            try:
                E.graphs(E.parse_args(g_args + ["--name", "S2"]), runs)
            except OSError:
                pass
        held = lambda: ("S2" in json.load(open(os.path.join(tmp, "g", "perframe_table.json")))["sys"], "S2" in pickle.load(open(out, "rb"))["frames"][0]["sys"])  # noqa: E731
        assert held() == (True, False), "the table is renamed first, the pickle last"
        E.graphs(E.parse_args(g_args + ["--name", "S2"]), runs)                                              # a name in the table only: the rest of an interrupted write, written again
        assert held() == (True, True)
        must_fail(E.graphs, E.parse_args(g_args + ["--name", "S2"]), runs)
    print(f"graphs: 3 VAL frames in the schema of pass1.pkl (k, token, row, scene, gt pieces with tier / vis / kinds, maneuvers, sys lines + maneuver counts), a second system appended, a duplicate refused, the per-frame table "
          f"pooled = the scorer's topo F1 ({want:.4f}); mutants (frame order swap, a GT field in a system record) fail; a kill between the renames leaves the name in the table only, the rerun completes it: ok")


def test_snapshot():
    """--model model_step<N>.pt: a snapshot (step 10 of a 20-step smoke run) dumped, scored, swept and turned into graphs on 3 VAL cache frames; every file name carries _step10, the outputs record the model and its 10
    steps, the snapshot's dump is not model.pt's; refused: a dump of the other weights, a name other than model.pt / model_step<N>.pt"""
    import pickle
    import shutil
    assert (E.model_sfx("model.pt"), E.model_sfx("model_step500.pt"), E.model_step("model_step500.pt")) == ("", "_step500", 500)
    for bad in ("ckpt.pt", "model_step.pt", "model_step5.pt.tmp"):
        must_fail(E.parse_args, ["dump", "--run", "h", "--seed", "0", "--model", bad])
    if not CACHE_OK or not os.path.exists(E.planb_path(0)):
        print("snapshot: the names ok, the cache part SKIPPED (no cache or stored plain B)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
        out = os.path.join(scratch, "out")
        os.makedirs(out)
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        E.JKT.train(E.JKT.parse_args(["--name", "h", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--snapshots", "10"]), runs, 500, 48)
        snap, rd = ["--model", "model_step10.pt"], os.path.join(runs, "h")
        with patched((E, "PLANB", scratch)):
            for m in ([], snap):
                E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"] + m), runs)
            d20, d10 = (pickle.load(open(os.path.join(rd, f"token{s}_val_n3.pkl"), "rb")) for s in ("", "_step10"))
            assert (d10["meta"]["model"], d10["meta"]["steps"], d20["meta"]["model"], d20["meta"]["steps"]) == ("model_step10.pt", 10, "model.pt", 20), (d10["meta"], d20["meta"]["steps"])
            assert any(not np.array_equal(x["pnone"], y["pnone"]) for x, y in zip(d10["frames"], d20["frames"])), "the snapshot's dump is the snapshot's head"
            cards = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"] + snap), runs, out)
            tab = json.load(open(E.sweep(E.parse_args(["sweep", "--run", "h", "--seed", "0", "--split", "val", "--max", "3"] + snap), runs, out)))
            g = os.path.join(tmp, "g", "graphs.pkl")
            os.makedirs(os.path.dirname(g))
            E.graphs(E.parse_args(["graphs", "--run", "h", "--seed", "0", "--max", "3", "--out", g, "--name", "S10"] + snap), runs)
            names = sorted(os.listdir(out))
            assert names == ["h_token_card_argmax_step10_n3.json", "h_token_card_rec_step10_n3.json", "h_token_sweep_perclass_step10_n3.json"], names
            assert all(c["info"]["model"] == "model_step10.pt" and c["info"]["steps"] == 10 for c in cards.values()) and (tab["model"], tab["steps"]) == ("model_step10.pt", 10), tab["steps"]
            gt = json.load(open(os.path.join(tmp, "g", "perframe_table.json")))["sys"]["S10"]
            assert gt["model"] == "model_step10.pt" and gt["dump"]["path"].endswith("token_step10_val_n3.pkl"), gt["dump"]
            pickle.dump(d20, open(os.path.join(rd, "token_step10_val_n3.pkl"), "wb"))                         # model.pt's dump under the snapshot's name
            try:
                E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3", "--link-thr", "0.1"] + snap), runs, out)
            except AssertionError as e:
                assert "not the dump of this read-out" in str(e), e
            else:
                raise RuntimeError("a dump of model.pt was scored as the snapshot's")
    print(f"snapshot: --model model_step10.pt dumped / scored / swept / graphed on 3 VAL frames, every name carries _step10 ({', '.join(names)}, token_step10_val_n3.pkl), "
          "model and steps 10 recorded, the dump differs from model.pt's; refused: other names, model.pt's dump read as the snapshot's: ok")


def _floor_run(runs, name="h", seed=0, **args):
    """a run folder `runs/<name>` with the smoke checkpoint carrying the untrained head (it keeps nodes and links; args seed = `seed`, + `args`)"""
    os.makedirs(os.path.join(runs, name))
    ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
    ck["state"], ck["args"] = floor_model().state_dict(), dict(ck["args"], seed=seed, **args)
    torch.save(ck, os.path.join(runs, name, "model.pt"))


def test_ref_run():
    """score --ref-run (E376): `same_tokens` refuses other frames, cells or valid tokens; on 3 cache frames of the untrained head with planted p_none: the run itself as reference at a variant cell (which differs from the record)
    gives its own numbers, diff 0 and the interval [0, 0]; a reference with no node delivers nothing, so the diff is the card's own number (record decode: both cards); the card _vs_<run> sits beside the card without it, neither
    is overwritten; refused: a reference of another cache seed, a reference run on another cache of the same seed (end to end through `same_tokens`), --model with --ref-run"""
    import pickle
    import shutil
    r = dict(idx=np.arange(2), fr=[dict(token="a", cell=np.arange(4)), dict(token="b", cell=np.arange(4) + 9)], valid=[np.ones(4, bool), np.ones(4, bool)])
    E.same_tokens(r, pickle.loads(pickle.dumps(r)))
    muts = dict(frames=lambda x: x["fr"][1].update(token="c"), idx=lambda x: x.update(idx=np.arange(1, 3)), cells=lambda x: x["fr"][0]["cell"].__setitem__(2, 99), valid=lambda x: x["valid"][1].__setitem__(0, False))
    for mut in muts.values():
        x = pickle.loads(pickle.dumps(r))
        mut(x)
        must_fail(E.same_tokens, r, x)
    if not CACHE_OK or not os.path.exists(SMOKE) or not os.path.exists(E.planb_path(0)):
        print("ref_run: same_tokens ok, the cache part SKIPPED (no cache, smoke model or stored plain B)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
        out = os.path.join(scratch, "out")
        os.makedirs(out)
        for name, seed in (("h", 0), ("g", 0), ("g1", 1)):
            _floor_run(runs, name, seed)
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        with patched((E, "PLANB", scratch)):
            E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
            toks = [E.open_cache("val", 0)[i]["tokens_np"] for i in range(3)]
            d, _ = _plant_pnone(os.path.join(runs, "h", "token_val_n3.pkl"), toks)
            g = pickle.loads(pickle.dumps(d))                                                                  # g: the same frames and tokens, no node kept at any threshold
            g["meta"]["run"] = "g"
            for f, t in zip(g["frames"], toks):
                f["pnone"][:] = 1.0
                f["keep"][:] = False
                f["rec"], f["argmax"] = E.decode_variant(f, t["valid"])["rec"], {k: v[:0] for k, v in f["argmax"].items()}
            pickle.dump(g, open(os.path.join(runs, "g", "token_val_n3.pkl"), "wb"))
            shutil.copy(os.path.join(runs, "h", "token_val_n3.pkl"), os.path.join(runs, "g1", "token_val_n3.pkl"))
            base, var = ["score", "--run", "h", "--seed", "0", "--max", "3"], ["--link-thr", "0.3", "--link-thr-long", "0.05", "--keep-thr", "0.6", "--no-none-gate"]
            plain = E.score(E.parse_args(base + var), runs, out)["rec"]
            own = E.score(E.parse_args(base + var + ["--ref-run", "h"]), runs, out)["rec"]
            stem = "h_token_card_rec_lt0.3_ltl0.05_kt0.6_nng_n3"
            assert os.path.exists(os.path.join(out, stem + ".json")) and os.path.exists(os.path.join(out, stem + "_vs_h.json")) and "ref_run" not in plain["info"], os.listdir(out)
            must_fail(E.score, E.parse_args(base + var + ["--ref-run", "h"]), runs, out)                       # never overwritten
            rf, pr = own["info"]["ref_run"], own["info"]["paired_vs_ref_run"]
            assert any(own["info"]["record"][k] != own["metrics"][k] for k in E.KEYS), "the variant cell differs from the record on these frames"
            assert all(own["metrics"][k] == plain["metrics"][k] == rf[k] for k in E.KEYS + ("junction_P", "junction_R")) and rf["run"] == "h" and rf["dump"]["sha256"], rf
            assert all((pr[k]["diff"], pr[k]["lo"], pr[k]["hi"]) == (0.0, 0.0, 0.0) for k in ("topo_F1", "junction_F1", "junction_P")) and rf["cell"] == "the decode of this card", {k: pr[k] for k in ("topo_F1", "junction_F1")}
            must_fail(E.score, E.parse_args(base + var + ["--ref-run", "h", "--ref-run-cell", "0.3,0.05,0.6,off"]), runs, out)     # the card's own cell: the default's name (no _rc), never overwritten
            # E376 v2: the reference read at ANOTHER cell = the --ref-cell of the same cell on the same dump (a self-pair across two cells: a hand-known pair)
            cross = E.score(E.parse_args(base + var + ["--ref-cell", "0.1,0.2,0.5,on", "--ref-run", "h", "--ref-run-cell", "0.1,0.2,0.5,on"]), runs, out)["rec"]
            ci = cross["info"]
            assert os.path.exists(os.path.join(out, stem + "_vs_h_rc0.1_0.2_0.5_on.json")) and ci["ref_run"]["cell"] == dict(link_thr=0.1, link_thr_long=0.2, keep_thr=0.5, none_gate=True), ci["ref_run"]["cell"]
            assert all(ci["ref_run"][k] == ci["ref_cell"][k] for k in E.KEYS + ("junction_P", "junction_R")) and ci["paired_vs_ref_run"] == ci["paired_vs_ref_cell"] and ci["paired_vs_ref_run"]["topo_F1"]["diff"] != 0, ci["paired_vs_ref_run"]["topo_F1"]
            emp = E.score(E.parse_args(base + var + ["--ref-run", "g"]), runs, out)["rec"]
            rf, pr = emp["info"]["ref_run"], emp["info"]["paired_vs_ref_run"]
            assert (rf["topo_F1"], rf["junction_F1"]) == (0.0, 0.0) and emp["metrics"]["topo_F1"] > 0, rf
            assert all(abs(pr[k]["diff"] - emp["metrics"][k]) < 1e-12 for k in ("topo_F1", "junction_F1")), "this run minus the reference: the card's own number"
            cards = E.score(E.parse_args(base + ["--ref-run", "g"]), runs, out)                                  # the record decode: both cards, each against the reference's same decode
            assert sorted(cards) == ["argmax", "rec"] and all(os.path.exists(os.path.join(out, f"h_token_card_{x}_n3_vs_g.json")) for x in cards), os.listdir(out)
            assert all(c["info"]["ref_run"]["topo_F1"] == 0.0 and abs(c["info"]["paired_vs_ref_run"]["topo_F1"]["diff"] - c["metrics"]["topo_F1"]) < 1e-12 for c in cards.values())
            must_fail(E.score, E.parse_args(base + var + ["--ref-run", "g1"]), runs, out)                       # a run of another cache seed
            must_fail(E.score, E.parse_args(base + var + ["--ref-run-cell", "0.1,0.2,0.5,on"]), runs, out)      # a reference cell without a reference run
            must_fail(E.score, E.parse_args(base + var + ["--ref-run", "h", "--model", "model_step10.pt"]), runs, out)     # --ref-run compares model.pt heads
            _floor_run(runs, "p", 0, ident="e273_p")                                                           # p: seed 0 on plain B's cache: the same frames, other tokens (its own dump passes read_dump)
            E.dump(E.parse_args(["dump", "--run", "p", "--seed", "0", "--cpu", "--max", "3"]), runs)
            try:
                E.score(E.parse_args(base + var + ["--ref-run", "p"]), runs, out)
            except AssertionError as e:
                assert "other tokens" in str(e), e
            else:
                raise RuntimeError("a reference run on another cache was scored")
    print(f"ref_run: same_tokens refuses other frames / idx / cells / valid tokens ({len(muts)} mutants); on 3 cache frames the run itself as reference at a variant cell (not the record's numbers) = its own numbers, diff 0, interval [0, 0]; "
          f"--ref-run-cell at the card's own cell = the default (same name); at another cell (_rc0.1_0.2_0.5_on) = --ref-cell of that cell exactly (numbers and interval); an empty reference = 0, so the diff is the card's own number "
          f"(topo F1 {emp['metrics']['topo_F1']:.4f}; record decode: rec and argmax); _vs_<run> beside the card without it, never overwritten; refused: another cache seed, a cell without a run, --model with --ref-run, and a seed-0 "
          "run on plain B's cache (same frames, other tokens: `same_tokens`): ok")


def _ens_toy():
    """two heads' dump frames of one frame of 5 tokens (the fields `mean_frame` reads; a decode of their own lying around). Candidates: 0 -> 1 long in both (p 0.6 / 0.2), 1 -> 2 short in A only (0.5), 1 -> 4 long in A only
    (0.45, scales 0.9 / 0.8), 2 -> 3 short in B only (0.42), 3 -> 4 short in both (0.3 / 0.7)"""
    f32 = np.float32
    cand = lambda s, d, p, lg, a, b: dict(src=np.array(s, np.int16), dst=np.array(d, np.int16), p=np.array(p, f32), long=np.array(lg, bool), a=np.array(a, f32), b=np.array(b, f32))  # noqa: E731
    ptype = lambda pn: np.stack([pn, 1 - pn] + [np.zeros(5)] * (JK.NTYPE - 2), 1).astype(np.float16)  # noqa: E731
    frame = lambda pn, none_cls, c: dict(token="t", row=7, cell=np.arange(5, dtype=np.int16) + 100, pnone=np.array(pn, f32), ptype=ptype(np.array(pn)), none_cls=np.array(none_cls, f32), cand=c,  # noqa: E731
                                        keep=np.ones(5, bool), rec=c, argmax=c)
    fa = frame([0.1, 0.3, 0.7, 0.2, 0.9], [[0.2, 0.4, 0.3, 0.6, 0.5], [0.3, 0.2, 0.2, 1.0, 1.0]], cand([0, 1, 1, 3], [1, 2, 4, 4], [0.6, 0.5, 0.45, 0.3], [1, 0, 1, 0], [0.5, THIRD, 0.9, THIRD], [0.7, THIRD, 0.8, THIRD]))
    fb = frame([0.3, 0.5, 0.1, 0.4, 0.1], [[0.6, 0.2, 0.05, 0.2, 0.3], [0.3, 0.2, 0.2, 1.0, 1.0]], cand([0, 2, 3], [1, 3, 4], [0.2, 0.42, 0.7], [1, 0, 0], [0.3, THIRD, THIRD], [0.5, THIRD, THIRD]))
    return fa, fb


def test_ensemble():
    """E379 `ensemble` (`mean_frame`): a hand pair of dump frames, every mean by hand (p_none, ptype, p(none|i), the union of the candidates, 0 for a missing pair, the scales over the heads that wrote the pair), the decode at the D115
    cell and with the none gate on by hand; the self-ensemble is the frame; refused: other cells, a pair of another class; textual mutants (the mean over the present heads only, one head only, the none probabilities of one head)
    deliver another graph. Cache: 3 frames of the untrained head h and a run g with h's dump without candidates: the ensemble h + g at (0.15, 0.1, 0.6, off) = h at (0.3, 0.2, 0.6, off) exactly (p / 2: --ref-run h at that cell, diff
    and interval 0), and not h at its own cell; the card's name and info, never overwritten; refused: another cache seed, another cache of the same seed, one run"""
    import pickle
    import shutil
    f32 = np.float32
    fa, fb = _ens_toy()
    e = E.mean_frame(fa, fb)
    c = e["cand"]
    assert sorted(e) == ["cand", "cell", "none_cls", "pnone", "ptype", "row", "token"], "no decode of either head is carried"
    assert list(zip(c["src"].tolist(), c["dst"].tolist())) == [(0, 1), (1, 2), (1, 4), (2, 3), (3, 4)] and c["long"].tolist() == [True, False, True, False, False], c
    close = lambda x, want, tol=1e-7: np.allclose(x, want, atol=tol, rtol=0)  # noqa: E731
    assert [x.dtype for x in (c["src"], c["p"], c["a"], e["pnone"], e["none_cls"], e["ptype"])] == [np.int16, f32, f32, f32, f32, np.float16], "the dump's dtypes"
    assert close(c["p"], [0.4, 0.25, 0.225, 0.21, 0.5]), c["p"]                                                                       # a pair one head did not write counts 0 for it
    assert close(c["a"], [0.4, THIRD, 0.9, THIRD, THIRD]) and close(c["b"], [0.6, THIRD, 0.8, THIRD, THIRD]), (c["a"], c["b"])        # the scales: the mean over the heads that wrote the pair
    assert close(e["pnone"], [0.2, 0.4, 0.4, 0.3, 0.5]) and close(e["none_cls"], [[0.4, 0.3, 0.175, 0.4, 0.4], [0.3, 0.2, 0.2, 1.0, 1.0]]), (e["pnone"], e["none_cls"])
    assert close(e["ptype"][:, :2], [[0.2, 0.8], [0.4, 0.6], [0.4, 0.6], [0.3, 0.7], [0.5, 0.5]], 1e-3), e["ptype"]                   # float16
    v5 = np.ones(5, bool)
    d115 = lambda f: _links_set(E.decode_variant(f, v5, 0.3, 0.6, False, 0.2)["rec"])  # noqa: E731
    gate = lambda f: _links_set(E.decode_variant(f, v5, 0.2, 0.6, True)["rec"])  # noqa: E731
    want, want_gate = {(0, 1), (1, 4), (3, 4)}, {(0, 1), (1, 4), (2, 3), (3, 4)}          # D115: 1 -> 2 (0.25) and 2 -> 3 (0.21) under the short 0.3; gate on at 0.2: 2 -> 3 over the mean p(none) 0.175 (A's 0.3), 1 -> 2 under 0.3
    assert (d115(e), gate(e)) == (want, want_gate), (d115(e), gate(e))
    self_ = E.mean_frame(fa, fa)
    assert all(np.array_equal(self_[k], fa[k]) for k in ("pnone", "ptype", "none_cls")) and all(np.array_equal(self_["cand"][k], fa["cand"][k]) for k in E.LINK_FIELDS), "the ensemble of a head with itself is the head"
    assert c["heads"].tolist() == [2, 1, 1, 1, 2], c["heads"]
    must_fail(E.mean_frame, fa, dict(fb, cell=fb["cell"] + 1))                                                                         # other tokens
    must_fail(E.mean_frame, fa, dict(fb, cand=dict(fb["cand"], long=~fb["cand"]["long"])))                                              # 0 -> 1 short in B, long in A
    must_fail(E.mean_frame, fa, dict(fb, cand={k: np.concatenate([v, v[:1]]) for k, v in fb["cand"].items()}))                       # a pair twice in one head's list
    # the floor bias (`floor_numbers`): one-sided pairs not delivered, both ends kept, within 0.025 of delivery
    fl = lambda f, cell: (lambda r: (r["delivered_links"], r["floor_links"]))(E.floor_numbers([f], [v5], cell))  # noqa: E731
    assert fl(e, (0.3, 0.2, 0.6, False)) == (3, 0), "D115: 1 -> 2 (0.25) and 2 -> 3 (0.21) are more than 0.025 under 0.3"
    assert fl(e, (0.27, 0.2, 0.6, False)) == (3, 1) and fl(e, (0.27, 0.2, 0.6, True)) == (3, 0), "1 -> 2 (0.25, A only) within 0.025 of 0.27; with the gate its p(none) 0.3 is more than 0.025 above it"
    assert fl(e, (0.22, 0.2, 0.6, True)) == (3, 1), "gate on at 0.22: 2 -> 3 (0.21, B only) within reach (p(none) 0.175); 1 -> 2 above 0.22 but 0.05 under its p(none)"
    eg = dict(e, none_cls=e["none_cls"].copy())
    eg["none_cls"][0, 1] = 0.26                                                                                                        # 1 -> 2 above the threshold, only the gate holds it back, by 0.01
    assert fl(eg, (0.22, 0.2, 0.6, True)) == (3, 2) and fl(e, (0.52, 0.2, 0.6, False)) == (2, 0), "a gate-only miss counts; 3 -> 4 (0.5) written by both heads never counts"
    floor_mut = {"the threshold miss only (the gate-only miss dropped)": ("ends & ~ok & ", "ends & ~ok & (c[\"p\"] < thr) & "), "both-head pairs counted": ('(c["heads"] == 1)', '(c["heads"] >= 1)')}
    for name, (old, new) in floor_mut.items():
        fm = _mutant(E.floor_numbers, old, new)
        assert [fm([x], [v5], cell)["floor_links"] for x, cell in ((eg, (0.22, 0.2, 0.6, True)), (e, (0.52, 0.2, 0.6, False)))] != [2, 0], name
    mutants = {"the mean over the present heads only": ('p=(acc["p"] / 2)', 'p=(acc["p"] / acc["n"])'), "one head only": ("def mean_frame(fa, fb):", "def mean_frame(fa, fb):\n    fb = fa"),
               "the none probabilities of one head": ('none_cls=mean("none_cls", np.float32)', 'none_cls=fa["none_cls"]')}
    for name, (old, new) in mutants.items():
        m = _mutant(E.mean_frame, old, new)(fa, fb)
        assert (d115(m), gate(m)) != (want, want_gate), name
    # the run names and the card's destination (no data)
    assert E.run_pair(" e1 , e2 ") == ("e1", "e2")
    for s in ("e1", "e1, e1", "e1,e2,e3", "e1,"):
        must_fail(E.run_pair, s)
    ens = lambda *x: E.ens_card_path(E.parse_args(["ensemble", "--runs", " e1 , e2 ", "--seed", "0", *x]), "R")  # noqa: E731
    with patched((E, "SCRATCH", "S")):
        assert [ens(), ens("--split", "train", "--every", "3"), ens("--link-thr", "0.3", "--link-thr-long", "0.2", "--keep-thr", "0.6", "--no-none-gate", "--ref-run", "c", "--ref-run-cell", "0.3,0.3,0.5,on"),
                ens("--max", "3"), ens("--out-dir", "D", "--ref-run", "c")] == [
            (os.path.join("R", "e1", "token_card_ens_e2_rec.json"), "token_card_ens_e2_rec"), (os.path.join("R", "e1", "token_card_ens_e2_rec_train_every3.json"), "token_card_ens_e2_rec_train_every3"),
            (os.path.join("R", "e1", "token_card_ens_e2_rec_lt0.3_ltl0.2_kt0.6_nng_vs_c_rc0.3_0.3_0.5_on.json"), "token_card_ens_e2_rec_lt0.3_ltl0.2_kt0.6_nng_vs_c_rc0.3_0.3_0.5_on"),
            (os.path.join("S", "out", "e1_token_card_ens_e2_rec_n3.json"), "token_card_ens_e2_rec_n3"), (os.path.join("D", "e1_token_card_ens_e2_rec_vs_c.json"), "token_card_ens_e2_rec_vs_c")], "the card names"
    if not CACHE_OK or not os.path.exists(SMOKE):
        print(f"ensemble: the hand frames ok ({len(mutants)} mutants), the cache part SKIPPED (no cache or smoke model)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        runs, out = os.path.join(tmp, "runs"), os.path.join(tmp, "out")
        for name, seed in (("h", 0), ("g", 0), ("g1", 1)):
            _floor_run(runs, name, seed)
        E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
        toks = [E.open_cache("val", 0)[i]["tokens_np"] for i in range(3)]
        d, _ = _plant_pnone(os.path.join(runs, "h", "token_val_n3.pkl"), toks)
        g = pickle.loads(pickle.dumps(d))                                                                    # g: h's p_none, no candidate (every pair below its floor), no link
        g["meta"]["run"] = "g"
        for f in g["frames"]:
            f["cand"], f["none_cls"] = {k: v[:0] for k, v in f["cand"].items()}, np.ones_like(f["none_cls"])
            f["rec"], f["argmax"] = {k: v[:0] for k, v in f["rec"].items()}, {k: v[:0] for k, v in f["argmax"].items()}
        pickle.dump(g, open(os.path.join(runs, "g", "token_val_n3.pkl"), "wb"))
        shutil.copy(os.path.join(runs, "h", "token_val_n3.pkl"), os.path.join(runs, "g1", "token_val_n3.pkl"))
        args = ["ensemble", "--runs", "h,g", "--seed", "0", "--max", "3", "--out-dir", out, "--link-thr", "0.15", "--link-thr-long", "0.1", "--keep-thr", "0.6", "--no-none-gate"]
        card = E.ensemble(E.parse_args(args + ["--ref-run", "h", "--ref-run-cell", "0.3,0.2,0.6,off"]), runs)
        name = "h_token_card_ens_g_rec_lt0.15_ltl0.1_kt0.6_nng_n3_vs_h_rc0.3_0.2_0.6_off.json"
        assert os.listdir(out) == [name] and json.load(open(os.path.join(out, name)))["metrics"] == card["metrics"], os.listdir(out)
        info, pr = card["info"], card["info"]["paired_vs_ref_run"]
        assert info["runs"] == ["h", "g"] and set(info["dumps"]) == {"h", "g"} and info["dumps"]["g"]["path"].endswith("token_val_n3.pkl") and card["frames"] == 3 and info["decode_variant"]["link_thr_long"] == 0.1, info
        fl = info["floor"]
        assert fl["delivered_links"] > 0 and fl["floor_links"] > 0 and E.FLOOR_NOTE in fl["note"], fl                                  # g wrote no pair: every pair is one-sided, those within 0.025 under their threshold count
        assert all(card["metrics"][k] == info["ref_run"][k] for k in E.KEYS) and all((pr[k]["diff"], pr[k]["lo"], pr[k]["hi"]) == (0.0, 0.0, 0.0) for k in ("topo_F1", "junction_F1")), pr["topo_F1"]
        rh = E.read_dump(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, 3)
        own = E.score_cell(rh, (0.15, 0.1, 0.6, False), rh["frames"], log=lambda *x, **k: None)["graph"]
        assert any(own[k] != card["metrics"][k] for k in E.KEYS), "h at the ensemble's own cell is another graph (a mean over the present heads only would give it)"
        must_fail(E.ensemble, E.parse_args(args + ["--ref-run", "h", "--ref-run-cell", "0.3,0.2,0.6,off"]), runs)          # never overwritten
        must_fail(E.ensemble, E.parse_args(args[:2] + ["h,g1"] + args[3:]), runs)                                         # another cache seed
        must_fail(E.ensemble, E.parse_args(args[:2] + ["h"] + args[3:]), runs)                                            # one run
        must_fail(E.ensemble, E.parse_args(args + ["--ref-run-cell", "0.3,0.2,0.6,off"]), runs)                          # a reference cell without a reference run
        _floor_run(runs, "p", 0, ident="e273_p")                                                                         # p: seed 0 on plain B's cache: the same frames, another opened cache
        E.dump(E.parse_args(["dump", "--run", "p", "--seed", "0", "--cpu", "--max", "3"]), runs)
        try:
            E.ensemble(E.parse_args(args[:2] + ["h,p"] + args[3:]), runs)
        except AssertionError as x:
            assert "other caches" in str(x), x
        else:
            raise RuntimeError("an ensemble with a run on another cache was scored")
        assert os.listdir(out) == [name], os.listdir(out)
    print(f"ensemble: a hand pair (the union of the candidates, 0 for a pair one head did not write, the scales over the heads that wrote it, the means of p_none / ptype / p(none|i), the D115 decode and the gate-on decode by hand, "
          f"the self-ensemble = the head, other cells / another class / a pair twice refused; {len(mutants)} mutants caught: {', '.join(mutants)}); the floor bias by hand (a threshold miss, a gate-only miss, both-head pairs never; "
          f"{len(floor_mut)} mutants); run names stripped, A = B refused, the card names (default, split / every, variant + reference cell, --max, --out-dir); 3 cache frames: h + g (h without candidates) at (0.15, 0.1, 0.6, off) = "
          f"h at (0.3, 0.2, 0.6, off) (topo F1 {card['metrics']['topo_F1']:.4f}, paired diff and interval 0; h at the ensemble's own cell differs; floor {fl['floor_links']} of {fl['delivered_links']} delivered), {name}, never overwritten; "
          "refused: another cache seed, another cache of the same seed (the opened cache), one run, a cell without a run: ok")


def _cell_rows(cells):
    """rows of a per-class sweep table from {(short, long, keep): (topo F1, junction F1, lanes found, dup_frac)}, none gate off"""
    return [dict(link_thr=s, link_thr_long=l_, keep_thr=k, none_gate=False, topo_F1=t, junction_F1=j, lanes_found=n, dup_frac=d) for (s, l_, k), (t, j, n, d) in cells.items()]


def test_choose_cell():
    """`pick_cell` (E375's rule as pinned in E377 / E378) on hand tables: dup_frac 0.05 is eligible, 0.06 not; lanes found 0.010 under the reference cell's not; the window T - 0.005; the highest junction F1 in it; a tie in junction F1
    to the cell nearest the reference (short first); equal distances to the higher threshold; no eligible cell: the reference cell; two tables: the mean row by row; 6 textual mutants. `choose-cell`: the table of a run folder,
    the json name (one run, two runs, another reference), never overwritten (a stored choice is recomputed and printed; another cell refused), a default-grid table refused"""
    base = {(0.3, 0.2, 0.6): (0.660, 0.25, 0.70, 0.04), (0.2, 0.2, 0.6): (0.668, 0.26, 0.70, 0.05), (0.1, 0.05, 0.7): (0.700, 0.30, 0.70, 0.06), (0.3, 0.05, 0.5): (0.690, 0.30, 0.68, 0.01),
            (0.3, 0.1, 0.6): (0.6645, 0.26, 0.70, 0.03), (0.1, 0.3, 0.6): (0.662, 0.40, 0.70, 0.01)}
    a_ = _cell_rows(base)                                                                     # T 0.668 (0.2, 0.2, 0.6: dup 0.05 eligible); (0.1, 0.05, 0.7) dup 0.06, (0.3, 0.05, 0.5) lanes 0.68 < 0.69 not; (0.1, 0.3, 0.6) under the window
    b_ = _cell_rows({**base, (0.2, 0.3, 0.6): (0.666, 0.27, 0.70, 0.02)})                    # + a window cell with a higher junction F1
    c_ = _cell_rows({(0.3, 0.2, 0.5): (0.60, 0.3, 0.70, 0.0), (0.3, 0.2, 0.25): (0.70, 0.3, 0.70, 0.0), (0.3, 0.2, 0.75): (0.70, 0.3, 0.70, 0.0)})   # equal keep distances 0.25 to the reference keep 0.5
    d_ = _cell_rows({**base, (0.3, 0.1, 0.6): (0.650, 0.26, 0.70, 0.03)})                    # the second seed: (0.3, 0.1, 0.6) lower, the mean 0.65725 under the window
    cases = [(([a_],), (0.3, 0.1, 0.6, False)), (([b_],), (0.2, 0.3, 0.6, False)), (([c_], (0.3, 0.2, 0.5, False)), (0.3, 0.2, 0.75, False)), (([a_, d_],), (0.2, 0.2, 0.6, False)),
             (([_cell_rows({k: v[:3] + (0.06,) for k, v in base.items()})],), E.D115)]
    got = [E.pick_cell(*x)["cell"] for x, _ in cases]
    assert got == [w for _, w in cases], got
    r = E.pick_cell([a_])
    assert (r["T"], sorted(r["window"])) == (0.668, [(0.2, 0.2, 0.6, False), (0.3, 0.1, 0.6, False)]) and len(r["eligible"]) == 4, r
    assert E.pick_cell([_cell_rows({k: v[:3] + (0.06,) for k, v in base.items()})])["eligible"] == []
    must_fail(E.pick_cell, [a_, b_])                                                         # other cells
    must_fail(E.pick_cell, [a_], (0.3, 0.3, 0.6, False))                                     # the reference cell is not in the table
    mut = {"dup_frac < 0.05": ('r["dup_frac"] <= CHOOSE_DUP', 'r["dup_frac"] < CHOOSE_DUP'), "no window": ("T - CHOOSE_WINDOW)", "T)"),
           "ties long before short": ("abs(c[0] - ref[0]), abs(c[1] - ref[1])", "abs(c[1] - ref[1]), abs(c[0] - ref[0])"), "no lanes bound": (' and r["lanes_found"] >= rows[ref]["lanes_found"] - CHOOSE_LANES', ""),
           "no junction key": ('-rows[c]["junction_F1"], ', ""), "the lower thresholds on a tie": ("-c[0], -c[1], -c[2]", "c[0], c[1], c[2]"), "the first table only": ("sum(t[c][k] for t in cells) / len(cells)", "cells[0][c][k]")}
    for name, (old, new) in mut.items():
        fn = _mutant(E.pick_cell, old, new)
        assert [fn(*x)["cell"] for x, _ in cases] != [w for _, w in cases], name
    with tempfile.TemporaryDirectory() as tmp:
        runs, out = os.path.join(tmp, "runs"), os.path.join(tmp, "out")
        for name, rows in (("h", a_), ("g", d_)):
            E.JV.write_json(os.path.join(runs, name, "token_sweep_perclass_train_every3.json"), dict(grid="perclass", split="train", every=3, rows=rows))
        E.JV.write_json(os.path.join(runs, "x", "token_sweep_perclass_train_every3.json"), dict(split="train", every=3, rows=a_))          # a table without the per-class grid
        cc = lambda *x: E.choose_cell(E.parse_args(["choose-cell", "--runs", *x]), runs)  # noqa: E731
        assert [cc("h")["cell"], cc("h,g")["cell"], cc("h", "--ref-cell", "0.3,0.05,0.5,off")["cell"]] == ["0.3,0.1,0.6,off", "0.2,0.2,0.6,off", "0.3,0.05,0.5,off"]      # the reference (0.3, 0.05, 0.5): lanes bound 0.67, its own topo F1 0.690 = T
        names = sorted(os.listdir(os.path.join(runs, "h")))
        assert names == ["token_cell_perclass_train_every3.json", "token_cell_perclass_train_every3_mean_g.json", "token_cell_perclass_train_every3_ref0.3_0.05_0.5_off.json", "token_sweep_perclass_train_every3.json"], names
        stored = os.path.join(runs, "h", names[0])
        before = open(stored).read()
        assert cc("h")["cell"] == "0.3,0.1,0.6,off" and open(stored).read() == before, "a stored choice is recomputed and printed, not written again"
        E.JV.write_json(os.path.join(runs, "h", "token_sweep_perclass_train_every3.json"), dict(grid="perclass", split="train", every=3, rows=b_))
        must_fail(cc, "h")                                                                   # the table now gives another cell than the stored one
        must_fail(cc, "x")
        must_fail(cc, "h,h")
        assert cc("h", "--out-dir", out)["cell"] == "0.2,0.3,0.6,off" and os.listdir(out) == ["h_token_cell_perclass_train_every3.json"]
    print(f"choose_cell: hand tables (dup_frac 0.05 eligible / 0.06 not, lanes 0.010 under the reference not, the window, the junction F1 first, the short distance on a tie, the higher threshold on equal distances, the fall-back, the mean of "
          f"two tables), other cells / a missing reference refused, {len(mut)} mutants caught ({', '.join(mut)}); the command: names (one run, _mean_<R2>, _ref<cell>, --out-dir), a stored choice recomputed and not rewritten, "
          "another cell or a default-grid table or one run twice refused: ok")


def _anatomy_toy():
    """a hand frame of 8 tokens (heading +y) and 5 GT nodes: GT long links G0 -> G1, G2 -> G3, G4 -> G3 (no token near G4), a short GT link G1 -> G3; T0 = G0, T1 = G1, T2 a duplicate of T1 (unassigned), T3 = G3, T4 near no GT
    node (p_none 0.5), T5 = G2 (p_none 0.55: kept at 0.6, not at 0.5), T6 invalid, T7 a duplicate of T3 (p_none 0.7: not kept). Long candidates: 0 -> 1 tp 0.5, 0 -> 2 dup 0.5, 0 -> 3 wrong 0.5, 0 -> 4 false 0.3, 5 -> 1 wrong 0.4, 5 -> 3 tp 0.1,
    5 -> 7 0.9 (an end not kept); 1 -> 2 0.9 short. -> (dump frame, valid, GT side)"""
    f32 = np.float32
    pos = np.array([[0, 0], [0, 20], [0.5, 20], [10, 20], [30, 30], [10, 0], [40, 0], [10.3, 20]], float)
    tok = dict(pos=pos, dirv=np.tile([0.0, 1.0], (8, 1)))
    valid = np.array([1, 1, 1, 1, 1, 1, 0, 1], bool)
    table = dict(pos=np.array([[0, 0], [0, 20], [10, 0], [10, 20], [20, 0]], float), dirv=np.tile([0.0, 1.0], (5, 1)), src=np.array([0, 2, 4, 1]), dst=np.array([1, 3, 3, 3]), long=np.array([1, 1, 1, 0], bool))
    gt_id = np.array([0, 1, -1, 3, -1, 2, -1, -1])
    pt = np.zeros((8, JK.NTYPE), np.float16)
    pt[[0, 5], E.ENTRY] = [0.5, 0.25]
    pt[[1, 2, 3, 4], E.EXIT] = [0.5, 0.125, 0.25, 0.0625]
    cand = [(0, 1, 0.5, 1), (0, 2, 0.5, 1), (0, 3, 0.5, 1), (0, 4, 0.3, 1), (5, 1, 0.4, 1), (5, 3, 0.1, 1), (5, 7, 0.9, 1), (1, 2, 0.9, 0)]
    c = dict(src=np.array([x[0] for x in cand], np.int16), dst=np.array([x[1] for x in cand], np.int16), p=np.array([x[2] for x in cand], f32), long=np.array([x[3] for x in cand], bool), a=np.full(8, THIRD, f32), b=np.full(8, THIRD, f32))
    f = dict(token="t", pnone=np.array([0.1, 0.2, 0.3, 0.1, 0.5, 0.55, 0.05, 0.7], f32), ptype=pt, none_cls=np.zeros((2, 8), f32), cand=c)
    uout = np.zeros(8)
    uout[0] = np.log(3.0)                                                                                       # sigmoid 0.75 on T0, 0.5 elsewhere
    return f, valid, E.gt_labels(tok, valid, table, gt_id, uout, np.zeros(8))


def test_anatomy():
    """the anatomy (E376) on a hand frame (`_anatomy_toy`) with every number by hand; average_precision on a hand list with a tie (pessimistic 5/12, optimistic 7/12, a tie scored as one step 1/2); textual mutants; cache: the
    command on 3 frames of the untrained head"""
    import pickle
    assert abs(E.average_precision([0.8, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 5 / 12) < 1e-12 and E.average_precision([0.3, 0.2], [0, 0]) is None
    f, valid, lab = _anatomy_toy()
    run = lambda kt=0.6: E.anatomy_numbers([E.anatomy_frame(f, valid, lab, kt)], len(lab["gt_long"]), 0.2)  # noqa: E731
    n = run()
    cats = n["categories"]
    assert (n["gt_long_links"], n["delivered"], [cats[k]["n"] for k in E.LONG_CATS]) == (3, 5, [1, 1, 2, 1]), (n["delivered"], cats)
    assert [cats[k]["share"] for k in E.LONG_CATS] == [0.2, 0.2, 0.4, 0.2] and abs(n["tp_over_gt_long"] - 1 / 3) < 1e-12
    assert [cats[k]["median_p"] for k in E.LONG_CATS] == [0.5, 0.5, float(np.median(np.float32([0.5, 0.4]))), float(np.float32(0.3))], [cats[k]["median_p"] for k in E.LONG_CATS]
    assert [cats[k]["median_typed"] for k in E.LONG_CATS] == [0.25, 0.0625, 0.125, 0.03125], [cats[k]["median_typed"] for k in E.LONG_CATS]
    assert np.allclose([cats[k]["median_b_long"] for k in E.LONG_CATS], [0.375, 0.375, 0.3125, 0.375], atol=1e-12)
    assert abs(n["ap_long"] - 1 / 3) < 1e-12 and (n["long_candidates"], n["long_candidates_tp"]) == (6, 2) and abs(n["recall_ceiling"] - 2 / 3) < 1e-12, n      # pessimistic: the tp of the 0.5 tie ranks third
    assert abs(n["ap_existence"] - 11 / 12) < 1e-12 and (n["valid_tokens"], n["assigned_tokens"]) == (7, 4), n                                                      # 0.9 0.9 0.8 + . . 0.45 + . (T6 invalid)
    deliv = E.decode_variant(f, valid, 0.3, 0.6, False, 0.2)["rec"]
    assert {(s, d) for s, d, lg in zip(deliv["src"].tolist(), deliv["dst"].tolist(), deliv["long"]) if lg} == {(0, 1), (0, 2), (0, 3), (0, 4), (5, 1)}, "delivered = the long links of the D115 decode"
    n5 = run(0.5)
    assert (n5["delivered"], n5["long_candidates"], n5["categories"]["wrong_pair"]["n"]) == (3, 3, 1), "keep 0.5 drops T4 (p_none exactly 0.5) and T5 with their links"
    mutants = {"dup / wrong swapped": (E.link_category, "return 1 if any((int(x), int(y)) in gt_long for x in ns for y in nd) else 2", "return 2 if any((int(x), int(y)) in gt_long for x in ns for y in nd) else 1"),
               "false_end only with both ends far": (E.link_category, "len(ns) == 0 or len(nd) == 0", "len(ns) == 0 and len(nd) == 0"),
               "ties broken optimistically": (E.average_precision, "np.lexsort((label, -np.asarray", "np.lexsort((~label, -np.asarray"),
               "existence label from p_none": (E.anatomy_frame, 'ex_label=lab["gt_id"][v] >= 0', 'ex_label=np.asarray(f["pnone"])[v] < 0.5')}
    for what, (fn, old, new) in mutants.items():
        with patched((E, fn.__name__, _mutant(fn, old, new))):
            m = run()
        assert ([m["categories"][k]["n"] for k in E.LONG_CATS], m["ap_long"], m["ap_existence"]) != ([1, 1, 2, 1], n["ap_long"], n["ap_existence"]), what
    with patched((E, "average_precision", _mutant(E.average_precision, *mutants["ties broken optimistically"][1:]))):
        assert abs(E.average_precision([0.8, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 7 / 12) < 1e-12
    # the mass concentration on a hand class: source 0: G = {1 (0.3), 2 (0.1)} and a non-member 3 at 0.5; source 1: G = {4 (0.6)}, a non-member at 0.35; source 2: no GT successor (0.9 to 3); source 3: G = {4 (0.2)}
    p, G, cand = np.zeros((5, 5)), np.zeros((5, 5), bool), np.zeros((5, 5), bool)
    for i, j, pv, g in ((0, 1, 0.3, 1), (0, 2, 0.1, 1), (0, 3, 0.5, 0), (1, 2, 0.35, 0), (1, 4, 0.6, 1), (2, 3, 0.9, 0), (3, 4, 0.2, 1)):
        p[i, j], cand[i, j], G[i, j] = pv, True, bool(g)
    share, above, zero = E.group_concentration(p, G, cand)
    assert np.allclose(share, [0.75, 1.0, 1.0]) and above.tolist() == [2, 2, 0] and zero == 0, (share, above, zero)
    cn = E.concentration_numbers(share, above)
    assert cn["sources"] == 3 and cn["share_median"] == 1.0 and abs(cn["share_mean"] - 2.75 / 3) < 1e-12 and abs(cn["above_mean"] - 4 / 3) < 1e-12 and abs(cn["above_2plus"] - 2 / 3) < 1e-12 and cn["above_mean_given_one"] == 2.0, cn
    over_all = _mutant(E.group_concentration, "top, mass = np.where(G, p, 0.0).max(1)", "top, mass = np.where(cand, p, 0.0).max(1)")             # the max over all candidates instead of the group
    assert not np.allclose(over_all(p, G, cand)[0], share), "the largest member is taken within G(i)"
    # a stub head (7 tokens 2 m apart; short class: chord <= 8 m) with a hand y: short G(0) = {1 (0.5), 2 (0.35)}, G(3) = {4 (0.6)}, G(2) = {3} without mass (2 emits nothing); long G(1) = {6} (10 m) without mass
    out, _ = stub_out(7, {0: {1: 0.5, 2: 0.35}, 3: {4: 0.6}}, [0.1] * 7)
    y = torch.zeros(1, 7, 7, dtype=torch.uint8)
    for i, j in ((0, 1), (0, 2), (3, 4), (2, 3), (1, 6)):
        y[0, i, j] = 1
    hc = E.head_concentration(out, y)
    (ss, sa, sz), (ls, la, lz) = hc["short"], hc["long"]
    assert np.allclose(ss, [0.5 / 0.85, 1.0], atol=1e-6) and sa.tolist() == [2, 0, 1] and sz == 1 and len(ls) == 0 and la.tolist() == [0] and lz == 1, hc
    no_gate = _mutant(E.head_concentration, "G, cand = ((y > 0)[:, None] & out[\"cand\"])[0]", "G, cand = ((y > 0)[:, None] | (out[\"cand\"] & False))[0]")     # G without the class gate
    assert no_gate(out, y)["long"][2] != lz, "G(i) is the training label: y AND the class gate"
    if not CACHE_OK or not os.path.exists(SMOKE):
        print("anatomy: the hand frame and mutants ok, the cache part SKIPPED (no cache or no smoke model)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        runs, out = os.path.join(tmp, "runs"), os.path.join(tmp, "out")
        _floor_run(runs)
        E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
        args = ["anatomy", "--run", "h", "--seed", "0", "--max", "3", "--link-thr-long", "0.05"]
        doc = json.load(open(E.anatomy(E.parse_args(args), runs, out)))
        assert os.listdir(out) == ["h_token_anatomy_long_ltl0.05_kt0.6_n3.json"], os.listdir(out)
        d = pickle.load(open(os.path.join(runs, "h", "token_val_n3.pkl"), "rb"))
        ds = E.open_cache("val", 0)
        want = sum(int(E.decode_variant(f_, ds[i]["tokens_np"]["valid"], 0.3, 0.6, False, 0.05)["rec"]["long"].sum()) for i, f_ in enumerate(d["frames"]))
        assert doc["delivered"] == want > 0 and sum(x["n"] for x in doc["categories"].values()) == want and doc["frames"] == 3 and doc["gt_long_links"] > 0, (doc["delivered"], want)
        assert doc["valid_tokens"] == sum(int(ds[i]["tokens_np"]["valid"].sum()) for i in range(3)) and doc["dump"]["sha256"], doc["valid_tokens"]
        assert sorted(doc["concentration"]) == ["long", "short"] and all(x["sources"] > 0 and 0 < x["share_median"] <= 1 for x in doc["concentration"].values()), doc["concentration"]
        nc = doc["near_none"]["counts"]
        assert nc["near"] + nc["far"] == nc["none"] == sum(int((ds[i]["tokens_np"]["valid"] & (ds[i]["ytype"] == 0)).sum()) for i in range(3)) > 0 and doc["near_none"]["frames"] == 3, nc
        kept = {}
        for kt, dk in ((0.6, doc), (0.2, json.load(open(E.anatomy(E.parse_args(args + ["--keep-thr", "0.2"]), runs, out))))):      # the untrained head keeps every token at 0.6 (p_none <= 0.33); 0.2 splits them
            c_ = dk["near_none"]["counts"]
            recs = [E.decode_variant(f_, ds[i]["tokens_np"]["valid"], 0.3, kt, False, 0.05)["rec"] for i, f_ in enumerate(d["frames"])]
            deg3 = sum(int((np.bincount(r_["src"].astype(int), minlength=ds.K) + np.bincount(r_["dst"].astype(int), minlength=ds.K) >= 3).sum()) for r_ in recs)
            kept[kt] = sum(int((ds[i]["tokens_np"]["valid"] & (ds[i]["ytype"] == 0) & (f_["pnone"] < np.float32(kt))).sum()) for i, f_ in enumerate(d["frames"]))
            assert c_["deg3"] == deg3 > 0 and c_["kept_near"] + c_["kept_far"] == kept[kt], (kt, c_, deg3, kept[kt])
        assert kept[0.2] < kept[0.6], kept
        must_fail(E.anatomy, E.parse_args(args), runs, out)
    print(f"anatomy: a hand frame (delivered 5 of 3 GT long links: tp / dup / wrong / false 1 / 1 / 2 / 1, medians of p / typed / B' evidence, long AP 1/3 with a pessimistic tie, recall ceiling 2/3, existence AP 11/12, keep 0.5 drops two "
          f"nodes), AP 5/12 on a hand list ({len(mutants)} textual mutants caught: {', '.join(mutants)}); the mass concentration by hand (shares 0.75 / 1 / 1, candidates >= 0.3: 2 / 2 / 0; the max over all candidates caught) and on a stub head with a hand y (short shares 0.588 / 1, "
          f"a zero-mass group per class counted, not a NaN; G without the class gate caught); "
          f"the command on 3 cache frames: {doc['delivered']} delivered = the decode's long links, the concentration of both classes (a CPU forward that agrees with the dump), near + far = the cache's {nc['none']} valid none tokens, "
          f"the near_none degree >= 3 tokens and kept none tokens = rebuilt from the decode at the command's thresholds (keep 0.6: {kept[0.6]}, keep 0.2: {kept[0.2]} kept), never overwritten: ok")


def _near_toy():
    """E377 hand frame of 12 tokens (metres; heading +y unless said): assigned T0 (0, 0), T1 (20, 0), T11 (40, 0); none tokens: T2 a twin 1.0 m beside T0 and T3 exactly 2.0 m ahead of T0 (near), T4 2.01 m behind T0 (far), T5 1.0 m
    beside T0 with heading cosine 0.4 (far), T6 0.5 m beside T1 and in G(0) (near, a duplicate), T7 in G(1) and near nothing (a far duplicate, heading +x), T10 1.0 m beside T7 (near a NONE token only, heading +x); T8 (0.5 m beside
    T0) and T9 padding. Kept (p_none < 0.6): T0 T1 T2 T4 T6 T10 T11 (T3 0.7, T5 0.9, T7 0.65 not). Delivered (short >= 0.3, long >= 0.2, both ends kept): 0->1 L, 0->2 S, 4->0 S, 6->1 S, 1->11 L, 4->10 L, 10->4 L, 11->10 L; not:
    2->3 and 5->0 (an end dropped), 4->2 S 0.25, 0->6 L 0.15. Degree >= 3: T0, T1, T4, T10. T4's short p(none) 0.4 is above 4->0 (0.35): the none gate (off in the anatomy) would drop that link, and T0 and T4 with it
    -> (dump frame, valid, cache item)"""
    f32 = np.float32
    pos = np.array([[0, 0], [20, 0], [1, 0], [0, 2], [0, -2.01], [-1, 0], [20.5, 0], [-15, 30], [0.5, 0], [0, 0], [-15, 31], [40, 0]], float)
    dirv = np.tile([0.0, 1.0], (12, 1))
    dirv[5] = [np.sqrt(1 - 0.4 ** 2), 0.4]
    dirv[[7, 10]] = [1.0, 0.0]
    valid = np.ones(12, bool)
    valid[[8, 9]] = False
    ytype = np.zeros(12, np.int64)
    ytype[[0, 1, 11]] = [1, 2, 3]
    y = np.zeros((12, 12), np.uint8)
    y[[0, 0, 1, 1], [1, 6, 11, 7]] = 1
    cand = [(0, 1, 0.5, 1), (0, 2, 0.4, 0), (0, 6, 0.15, 1), (1, 11, 0.3, 1), (2, 3, 0.9, 0), (4, 0, 0.35, 0), (4, 2, 0.25, 0), (4, 10, 0.3, 1), (5, 0, 0.8, 0), (6, 1, 0.6, 0), (10, 4, 0.3, 1), (11, 10, 0.4, 1)]
    c = dict(src=np.array([x[0] for x in cand], np.int16), dst=np.array([x[1] for x in cand], np.int16), p=np.array([x[2] for x in cand], f32), long=np.array([x[3] for x in cand], bool),
             a=np.full(len(cand), THIRD, f32), b=np.full(len(cand), THIRD, f32))
    none_cls = np.zeros((2, 12), f32)
    none_cls[0, 4] = 0.4
    f = dict(token="t", pnone=np.array([0.1, 0.1, 0.5, 0.7, 0.3, 0.9, 0.55, 0.65, 0.0, 0.0, 0.2, 0.1], f32), none_cls=none_cls, cand=c)
    return f, valid, dict(tokens_np=dict(pos=pos, dirv=dirv), ytype=ytype, y=y)


def test_near_none():
    """the anatomy's section near_none (E377) on the hand frame `_near_toy`: every count, share and median by hand, a second frame of padding only (per frame = count / 2, an empty pool = shares 0 and medians None), the long threshold
    reaches the delivered graph; textual mutants of the label `jtoken.near_none` read through the anatomy, of the duplicates and of the none gate fail"""
    f, valid, item = _near_toy()
    part = E.near_frame(f, valid, item, 0.6, 0.2)
    want = dict(none=7, near=3, far=4, dup=2, dup_near=1, kept_near=2, kept_far=2, kept_dup=1, deg3=4, deg3_touch_near=2, deg3_touch_far=3, deg3_touch_dup=1)
    assert part["counts"] == want, part["counts"]
    empty = E.near_frame(f, np.zeros(12, bool), item, 0.6, 0.2)
    n = E.near_numbers([part, empty])
    assert n["frames"] == 2 and n["per_frame"] == {k: x / 2 for k, x in want.items()}, n["per_frame"]
    assert n["shares"] == dict(kept_near=2 / 3, kept_far=0.5, kept_dup=0.5, dup_near=0.5, deg3_touch_near=0.5, deg3_touch_far=0.75, deg3_touch_dup=0.25), n["shares"]
    f32 = lambda x: float(np.float32(x))  # noqa: E731
    assert n["median_pnone"]["near"] == f32(0.55) and abs(n["median_pnone"]["far"] - (f32(0.3) + f32(0.65)) / 2) < 1e-12, n["median_pnone"]       # near: 0.5 / 0.7 / 0.55; far: 0.3 / 0.9 / 0.65 / 0.2
    e = E.near_numbers([empty])
    assert set(e["shares"].values()) == {0.0} and e["median_pnone"] == dict(near=None, far=None), e
    assert E.near_frame(f, valid, item, 0.6, 0.1)["counts"]["deg3_touch_dup"] == 2, "long 0.1 delivers 0 -> 6: T0 touches a duplicate"
    label = {"< instead of <= at 2.0 m": ("d2 <= NEAR_R ** 2", "d2 < NEAR_R ** 2"), "the heading test dropped": (" & (cos >= NEAR_COS)", ""),
             "the assigned test dropped (near any other valid token)": ("asg[:, None, :]", "(valid[:, None, :] & ~torch.eye(valid.shape[-1], dtype=torch.bool))")}
    for what, (old, new) in label.items():
        with patched((JK, "near_none", _mutant(JK.near_none, old, new))):
            assert E.near_frame(f, valid, item, 0.6, 0.2)["counts"] != want, what
    own = {"the duplicates from the row of y": ('(np.asarray(item["y"]) > 0).any(0)', '(np.asarray(item["y"]) > 0).any(1)'),
           "the none gate on": ("decode_variant(f, v, LINK_THR, keep_thr, False, thr_long)", "decode_variant(f, v, LINK_THR, keep_thr, True, thr_long)")}
    for what, (old, new) in own.items():
        assert _mutant(E.near_frame, old, new)(f, valid, item, 0.6, 0.2)["counts"] != want, what
    print(f"near_none: the hand frame (near 3 / far 4 none tokens: a twin at 1.0 m, exactly 2.0 m near; 2.01 m, cosine 0.4, a token near a none token only far; padding not counted; duplicates 2, 1 near; kept near / far / dup "
          f"2 / 2 / 1; degree >= 3 tokens 4, touching a near / far / duplicate link 2 / 3 / 1; median p_none near 0.55, far 0.475), pooled with a padding frame (per frame / 2), an empty pool, the long threshold wired; "
          f"{len(label) + len(own)} textual mutants caught ({', '.join(list(label) + list(own))}): ok")


def test_argmax_keep():
    """score --argmax --keep-thr K (E376): `decode_argmax_variant` at the record keep = the dump's argmax decode of a real head (3 synthetic frames); a hand frame: a tie with the none slot goes to the candidate, a tie of two candidates
    to the lower destination, keep 0.6 adds the links of a token at p_none 0.55, a stored argmax link that the candidates cannot give is refused (`check_argmax`); textual mutants; cache: on a toy dump the argmax card at keep 0.5 = its
    stored argmax card, at keep 0.6 = a direct scoring; --argmax with a link threshold refused"""
    import pickle
    import shutil
    torch.manual_seed(0)
    m = JK.JTokenHead().eval()
    m.set_step(JK.SWITCH)
    sb = JK.synthetic_batch(3, 48, seed=2)
    toks = [dict(pos=sb["pos"][b].double().numpy(), cell=np.arange(48)) for b in range(3)]
    frames = E.batch_dump(E.run_head(m, sb, "cpu"), dict(token=["a", "b", "c"], row=torch.arange(3), tokens_np=toks))
    E.check_argmax(frames, sb["valid"].numpy())
    n_real = sum(len(f["argmax"]["src"]) for f in frames)
    f32 = np.float32
    pairs = [(0, 1, 0.4, 0), (0, 2, 0.4, 0), (0, 5, 0.35, 1), (1, 3, 0.3, 0), (1, 5, 0.2, 1), (2, 4, 0.5, 0), (4, 5, 0.6, 0), (6, 5, 0.6, 0)]          # in the dump's order (source, destination)
    c = dict(src=np.array([x[0] for x in pairs], np.int16), dst=np.array([x[1] for x in pairs], np.int16), p=np.array([x[2] for x in pairs], f32), long=np.array([x[3] for x in pairs], bool), a=np.full(8, THIRD, f32), b=np.full(8, THIRD, f32))
    pn, v7 = np.array([0.1, 0.1, 0.1, 0.1, 0.55, 0.1, 0.6], f32), np.ones(7, bool)                                    # token 6 exactly at the keep threshold 0.6: not kept there
    none_cls = np.array([[0.2, 0.3, 0.1, 1.0, 0.1, 1.0, 0.1], [0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0]], f32)             # 1 -> 3 ties its none (0.3), 0 -> 1 / 0 -> 2 tie (0.4), 1 -> 5 loses to none 0.7
    f = dict(token="t", pnone=pn, keep=pn < f32(0.5), none_cls=none_cls, cand=c, argmax={k: c[k][[0, 2, 3]] for k in E.LINK_FIELDS})
    got = lambda fn=E.decode_argmax_variant, kt=0.5: _links_set(fn(f, v7, kt)["argmax"])  # noqa: E731
    assert got() == {(0, 1), (0, 5), (1, 3)} and got(kt=0.6) == {(0, 1), (0, 5), (1, 3), (2, 4), (4, 5)}, (got(), got(kt=0.6))
    E.check_argmax([f], [v7])
    E.check_argmax([f], [v7], 0.6)
    must_fail(E.check_argmax, [dict(f, argmax={k: c[k][[0, 2, 3, 4]] for k in E.LINK_FIELDS})], [v7])                   # a stored link (1 -> 5) the candidates do not give
    below = dict(f, none_cls=np.array([none_cls[0], [0.3, 0.7, 1.0, 1.0, 0.01, 1.0, 1.0]], f32))                      # token 4 (kept at 0.6 only): p(none) 0.01 in the long class, no long candidate: its winner is below the floor
    E.check_argmax([below], [v7])
    must_fail(E.check_argmax, [below], [v7], 0.6)
    muts = {"none wins a tie": ("(c[\"p\"][w] >= np.asarray", "(c[\"p\"][w] > np.asarray"), "the higher destination wins a tie": ("np.lexsort((dst, -c", "np.lexsort((-dst, -c")}
    for old, new in muts.values():
        assert got(_mutant(E.decode_argmax_variant, old, new)) != got(), new
    le_keep = _mutant(E.decode_argmax_variant, "(np.asarray(f[\"pnone\"]) < np.float32(keep_thr))", "(np.asarray(f[\"pnone\"]) <= np.float32(keep_thr))")
    assert got(le_keep, 0.6) == got(kt=0.6) | {(6, 5)}, "p_none exactly at the keep threshold is not kept"
    if not CACHE_OK or not os.path.exists(SMOKE) or not os.path.exists(E.planb_path(0)):
        print(f"argmax_keep: a real head ({n_real} argmax links) and the hand frame ok, the cache part SKIPPED (no cache, smoke model or stored plain B)")
        return
    import qgraph_eval as QE
    with tempfile.TemporaryDirectory() as tmp:
        runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
        out = os.path.join(scratch, "out")
        os.makedirs(out)
        _floor_run(runs)
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        with patched((E, "PLANB", scratch)):
            E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
            path = os.path.join(runs, "h", "token_val_n3.pkl")
            toks = [E.open_cache("val", 0)[i]["tokens_np"] for i in range(3)]
            d, _ = _plant_pnone(path, toks)
            for f_, t in zip(d["frames"], toks):                                                          # the toy dump: the argmax among its candidates (an untrained head's winner may lie below the floor), none 1 where a class has no candidate
                has = np.zeros_like(f_["none_cls"], bool)
                has[f_["cand"]["long"].astype(int), f_["cand"]["src"].astype(int)] = True
                f_["none_cls"] = np.where(has, f_["none_cls"], np.float32(1.0))
                f_["argmax"] = E.decode_argmax_variant(f_, t["valid"])["argmax"]
            pickle.dump(d, open(path, "wb"))
            base = ["score", "--run", "h", "--seed", "0", "--max", "3"]
            stored = E.score(E.parse_args(base), runs, out)["argmax"]
            k5, k6 = (E.score(E.parse_args(base + ["--argmax", "--keep-thr", kt]), runs, out)["argmax"] for kt in ("0.5", "0.6"))
            assert all(os.path.exists(os.path.join(out, f"h_token_card_argmax_kt{kt}_n3.json")) for kt in ("0.5", "0.6")) and k6["info"]["decode_variant"] == dict(k6["info"]["decode_variant"], argmax=True, keep_thr=0.6)
            assert all(k5["metrics"][k] == stored["metrics"][k] for k in E.KEYS), "the argmax card at the record keep = the stored argmax card"
            direct = E.score_preds([E.frame_prediction(t, E.decode_argmax_variant(f_, t["valid"], 0.6), "argmax", 0.6) for f_, t in zip(d["frames"], toks)], QE.val_frames()[:3], log=lambda *x, **k: None)["graph"]
            assert all(k6["metrics"][k] == direct[k] for k in E.KEYS) and any(k6["metrics"][k] != k5["metrics"][k] for k in E.KEYS), "keep 0.6 = a direct scoring, and it adds the planted nodes"
            # the reference run is read at the decode of each card: self-pairs at the record decode (rec and argmax) and at --argmax 0.6
            own = E.score(E.parse_args(base + ["--ref-run", "h"]), runs, out)
            own["argmax_kt0.6"] = E.score(E.parse_args(base + ["--argmax", "--keep-thr", "0.6", "--ref-run", "h"]), runs, out)["argmax"]
            for name, c_ in own.items():
                rf, pr = c_["info"]["ref_run"], c_["info"]["paired_vs_ref_run"]
                assert all(rf[k] == c_["metrics"][k] for k in E.KEYS) and all((pr[k]["diff"], pr[k]["lo"], pr[k]["hi"]) == (0.0, 0.0, 0.0) for k in ("topo_F1", "junction_F1")), name
                assert name == "rec" or any(rf[k] != own["rec"]["metrics"][k] for k in E.KEYS), f"{name}: the reference is read at the argmax decode, not the record's"
            assert all(os.path.exists(os.path.join(out, f"h_token_card_{x}_n3_vs_h.json")) for x in ("rec", "argmax", "argmax_kt0.6")), os.listdir(out)
            must_fail(E.score, E.parse_args(base + ["--argmax", "--keep-thr", "0.7", "--link-thr", "0.2"]), runs, out)
    print(f"argmax_keep: the rebuilt argmax = the dump's on a real head ({n_real} links on 3 frames) and on a hand frame (tie with none -> the candidate, tie of two -> the lower destination, keep 0.6 adds two links, p_none exactly 0.6 "
          f"not kept, an impossible stored link refused, a winner below the floor of a source kept at 0.6 only refused at 0.6; {len(muts) + 1} textual mutants caught: {', '.join(muts)}, p_none <= keep); on a toy dump the card at keep 0.5 = the stored argmax card, keep 0.6 = "
          "a direct scoring (token_card_argmax_kt<K>); self-pairs with --ref-run at rec, argmax and argmax kt0.6 = their own numbers (the argmax ones not the rec card's), diff 0, interval [0, 0]; a link threshold refused: ok")


@contextlib.contextmanager
def mutated(fn, old, new):
    """`fn` replaced in its own module by a textual mutant (one text replaced, found exactly once) compiled in the module's OWN globals, so that the patches of the block reach it; put back after the block"""
    import inspect
    src = inspect.getsource(fn)
    assert src.count(old) == 1, old
    mod = sys.modules[fn.__module__]
    with patched((mod, fn.__name__, fn)):
        exec(compile(src.replace(old, new), mod.__file__, "exec"), vars(mod))
        yield


def refused_with(fn, why, *a):
    """fn(*a) must fail an assertion whose message holds `why`"""
    try:
        fn(*a)
    except AssertionError as e:
        assert why in str(e), (why, str(e))
        return
    raise RuntimeError(f"accepted although it must be refused ({why})")


def test_live():
    """E372 live read-out. Hand: `same_tokens` pairs a live read-out by frame only (two live ones of one B' batch and device), `pair_matrix`, `state_sha`, `same_tokens_as_cache` field by field, the refusals before anything is read.
    Cache + B' (runs/e292_a_s0): `live_source` (a cache-trained run reads runs/e292_a_s0, a joint run its b_state, train_nodes without b_state refused); a stub source that hands the CACHE tokens to the live path (the real label step,
    the real B' identity) and records its chunks: the live dump's records = the cache dump's (3 frames in one chunk; 10 frames: two head batches, chunks 4 4 2), tokens_exact, tokens_np / labels = the cache's, score / anatomy /
    sweep --live = the cache read-outs with the live meta, the parity read (--ref-cache) with diff 0 and the interval [0, 0]; the frame-order guard (a mutant without it is accepted); a joint run with shifted, reversed tokens (B' at
    bs 2, chunks 2 1): tokens_exact on no frame, scored on its own tokens (a read_dump that takes the cache's is caught), anatomy and sweep --live = a rebuild from its records, paired by frame with a run of the same B' batch (another
    batch refused), live-diff by hand; a z run (train_nodes 0) with a b_state equal to the base accepted; refused: a joint run on the cache path (dump, score, graphs), a fall-back to runs/e292_a_s0 (a mutant of live_source), a
    node head that did not move, a frozen run with a trained head, a moved pyramid, another b_sha256, the cache dump under the live name"""
    import pickle
    import shutil
    r = dict(idx=np.arange(2), fr=[dict(token="a", cell=np.arange(4)), dict(token="b", cell=np.arange(4) + 9)], valid=[np.ones(4, bool), np.ones(4, bool)], live=True, meta=dict(bs_b=4, device="cuda"))
    other = pickle.loads(pickle.dumps(dict(r, live=False)))
    other["fr"][0]["cell"][2], other["valid"][1][0] = 99, False
    E.same_tokens(r, other)                                                                                  # a live side: other cells and valid tokens of the same frames are paired
    E.same_tokens(r, dict(other, live=True))                                                                 # two live ones of one B' batch and device
    for meta in (dict(bs_b=8, device="cuda"), dict(bs_b=4, device="cpu")):
        refused_with(E.same_tokens, "another B' batch or device", r, dict(other, live=True, meta=meta))
    refused_with(E.same_tokens, "other tokens", dict(r, live=False), other)                                 # two cache read-outs: the cells
    x = pickle.loads(pickle.dumps(r))
    x["fr"][1]["token"] = "c"
    refused_with(E.same_tokens, "other frames", r, x)                                                        # other frames: refused, live or not
    sd = {"nodes.w": torch.arange(4.0), "pyr.b": torch.ones(2, dtype=torch.int64)}
    assert E.state_sha(sd) == E.state_sha(dict(reversed(list(sd.items())))) and E.state_sha(sd, "nodes.") != E.state_sha(sd), "the name order does not count, the prefix does"
    moved = dict(sd, **{"pyr.b": torch.zeros(2, dtype=torch.int64)})
    assert E.state_sha(moved, "nodes.") == E.state_sha(sd, "nodes.") and E.state_sha(moved) != E.state_sha(sd) and E.state_sha({"nodes.w": torch.arange(4.0).view(2, 2)}) != E.state_sha({"nodes.w": torch.arange(4.0)})
    assert E.b_shas(moved)["sha_frozen"] != E.b_shas(sd)["sha_frozen"] and E.b_shas(moved)["sha_nodes"] == E.b_shas(sd)["sha_nodes"]
    assert E.pair_matrix(np.array([[0, 2], [1, 0]], np.int16), 3).tolist() == [[0, 0, 1], [1, 0, 0], [0, 0, 0]] and not E.pair_matrix(np.zeros((0, 2), np.int16), 3).any()
    # tokens_exact: every token field (the head's inputs, the cells and the writer's e_p / flags_p / s_p) bit-equal to the cache's; float16 tok against its float32 copy is equal, one flipped value in any field is not
    assert E.TOKEN_KEYS == ("cell", "tok", "pos", "dirv", "e", "e_raw", "s", "uout", "uin", "flags", "valid", "e_p", "flags_p", "s_p")
    rng = np.random.default_rng(0)
    shapes = dict(cell=(), tok=(6,), pos=(2,), dirv=(2,), e=(), e_raw=(), s=(), uout=(), uin=(), flags=(8,), valid=(), e_p=(), flags_p=(8,), s_p=())
    cache = {k: (np.arange(5, dtype=np.int16) if k == "cell" else np.ones(5, bool) if k == "valid" else rng.random((5,) + s).astype(np.float16 if k == "tok" else np.float32)) for k, s in shapes.items()}
    fake = types.SimpleNamespace(c=types.SimpleNamespace(f={k: v[None] for k, v in cache.items()}))
    live_tk = {k: torch.from_numpy(v.astype(np.float32) if k == "tok" else v.astype(np.int64) if k == "cell" else v)[None] for k, v in cache.items()}
    assert E.same_tokens_as_cache(live_tk, fake, np.array([0])) == [True]
    for k in E.TOKEN_KEYS:
        t = {q: v.clone() for q, v in live_tk.items()}
        t[k].view(-1)[1] = ~t[k].view(-1)[1] if k == "valid" else t[k].view(-1)[1] + 1
        assert E.same_tokens_as_cache(t, fake, np.array([0])) == [False], k
    for bad, why in ((["score", "--run", "h", "--seed", "0", "--ref-run", "h", "--ref-cache"], "give --live and --ref-run"), (["score", "--run", "h", "--seed", "0", "--live", "--ref-cache"], "give --live and --ref-run")):
        refused_with(E.score, why, E.parse_args(bad))                                                        # refused before anything is read
    for bad, why in ((["--bs", "4"], "give --live"), (["--live", "--bs", "0"], "a batch of at least 1"), (["--live", "--drop", "0.17"], "no --drop")):
        refused_with(E.dump, why, E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu"] + bad))
    base_pt = os.path.join(E.ROOT, "runs", "e292_a_s0", "model.pt")
    if not CACHE_OK or not os.path.exists(SMOKE) or not os.path.exists(E.planb_path(0)) or not os.path.exists(base_pt):
        print("live: the hand part ok, the cache part SKIPPED (no cache, smoke model, stored plain B or runs/e292_a_s0)")
        return
    import lane_live_tok as LT
    import lane_tok_data as TD
    import qgraph_eval as QE
    ds = E.open_cache("val", 0)
    td = TD.TokData("val", E.LATTICE, feats=False)
    base = torch.load(base_pt, map_location="cpu")["state"]
    base_sha = E._file_id(base_pt)["sha256"]

    class Frames(torch.utils.data.Dataset):                                                                  # the stand-in of LiveData: x names the cache frame
        tokens, rows = ds.tokens, ds.rows

        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return dict(x=torch.full((1, 1, 1), float(i)), token=ds.tokens[i], row=ds.rows[i], table=td.table(i))

    class CacheB:                                                                                            # the stand-in of LiveSource: the CACHE tokens of the frames named by x (shift > 0: pos + shift m and the token order reversed); chunks = the B' batches
        def __init__(self, shift):
            self.shift, self.chunks = shift, []

        def to(self, dev):
            return self

        def tokens(self, x):
            self.chunks.append(int(x.shape[0]))
            tk = {k: torch.from_numpy(np.stack([np.array(ds.c.f[k][int(i)]) for i in x.flatten(1)[:, 0]])) for k in E.TOKEN_KEYS}
            tk["tok"] = tk["tok"].float()
            if self.shift:
                tk = {k: v.flip(1) for k, v in tk.items()}
                tk["pos"] = tk["pos"] + self.shift
            return None, tk

        def labelled(self, tk, tables, token, row, token_drop=0.0, seed=0, gate=True):
            return LT.live_batch(tk, tables, token, row, ds.K, token_drop, seed, gate)

    def live(shift, frames=Frames):                                                                          # the stub's tokens, the B' identity of the `live_source` in place now (a mutant inside `mutated`)
        stub, real = CacheB(shift), E.live_source
        return stub, ((E, "live_source", lambda path, ck, kw, K: (stub, real(path, ck, kw, K)[1])), (E, "live_data", lambda split, seed, ident: frames()))

    def joint(runs, name, b_state, train_nodes=1, b_sha256=base_sha):                                         # a run of jlive_train's format: args train_nodes, live_inputs, b_sha256; b_state
        _floor_run(runs, name, 0, train_nodes=train_nodes, live_inputs=True, b_sha256=b_sha256)
        p = os.path.join(runs, name, "model.pt")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if b_state is not None:
            ck["b_state"] = b_state
        torch.save(ck, p)
        return p, ck

    def dump_live(runs, name, shift=0.0, extra=(), frames=Frames):
        stub, pt = live(shift, frames)
        with patched(*pt):
            E.dump(E.parse_args(["dump", "--live", "--run", name, "--cpu"] + base_args + list(extra)), runs)
        return pickle.load(open(os.path.join(runs, name, f"token_val_live_n{base_args[-1]}.pkl"), "rb")), stub.chunks

    eq = lambda x, y: x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x) if isinstance(x, dict) else np.array_equal(x, y)  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
        out = os.path.join(scratch, "out")
        os.makedirs(out)
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        for name in ("h", "h2", "k"):
            _floor_run(runs, name)
        trained = {k: v.clone() for k, v in base.items()}
        key = next(k for k in sorted(trained) if k.startswith("nodes.") and trained[k].is_floating_point())
        trained[key] += 0.01                                                                                 # a node head that moved
        jp, jck = joint(runs, "j", trained)
        kw = dict(seed=0, ident="e292_a")
        src, bs_ = E.live_source(jp, jck, kw, ds.K)
        assert bs_["kind"] == "b_state" and bs_["sha_nodes"] == E.state_sha(trained, "nodes.") != E.state_sha(base, "nodes.") and torch.equal(src.model.state_dict()[key], trained[key]) and not src.model.nodes.training
        hck = torch.load(os.path.join(runs, "h", "model.pt"), map_location="cpu", weights_only=False)
        assert E.live_source("h", hck, kw, ds.K)[1]["sha_state"] == E.state_sha(base), "a cache-trained run reads runs/e292_a_s0"
        refused_with(E.live_source, "no fall-back", jp, {k: v for k, v in jck.items() if k != "b_state"}, kw, ds.K)      # train_nodes without b_state
        with patched((E, "PLANB", scratch)):
            base_args = ["--seed", "0", "--max", "3"]
            E.dump(E.parse_args(["dump", "--run", "h", "--cpu"] + base_args), runs)
            dl, chunks = dump_live(runs, "h")
            dc = pickle.load(open(os.path.join(runs, "h", "token_val_n3.pkl"), "rb"))
            m = dl["meta"]
            assert chunks == [3] and m["live"] and (m["bs_b"], m["bs_head"], m["device"], m["tokens_exact"], m["tokens_exact_frames"]) == (4, 8, "cpu", True, 3) and m["env"]["torch"] == torch.__version__ and m["env"]["gpu"] is None
            assert m["b_source"]["kind"] == "base" and {k: m["b_source"][k] for k in E.B_SHAS} == E.b_shas(base) and "live" not in dc["meta"], m["b_source"]
            for i, (f, g) in enumerate(zip(dc["frames"], dl["frames"])):                                     # the live path on the cache's tokens = the cache path: records, tokens, labels
                item = ds[i]
                assert all(eq(f[k], g[k]) for k in f), i
                assert all(np.array_equal(g["tokens_np"][k], item["tokens_np"][k]) for k in item["tokens_np"]), i
                lab = g["lab"]
                assert np.array_equal(lab["gt_id"], item["label"]["gt_id"]) and np.array_equal(lab["ytype"], item["ytype"]) and np.array_equal(E.pair_matrix(lab["ypair"], ds.K), item["y"]), i
                assert all(np.array_equal(lab[k], ds.c.f[k][i]) for k in ("uout", "uin")) and lab["stats"] == {k: int(ds.c.f[k][i]) for k in LT.STATS}, i
            # 10 frames: two head batches (8 + 2) with B' in chunks of 4 4 2: every record and the tokens still the cache's, frame by frame
            E.dump(E.parse_args(["dump", "--run", "k", "--cpu", "--seed", "0", "--max", "10"]), runs)
            base_args = ["--seed", "0", "--max", "10"]
            dk, chunks = dump_live(runs, "k")
            dkc = pickle.load(open(os.path.join(runs, "k", "token_val_n10.pkl"), "rb"))
            assert chunks == [4, 4, 2] and (dk["meta"]["tokens_exact"], dk["meta"]["tokens_exact_frames"]) == (True, 10) and all(eq(f[k], g[k]) for f, g in zip(dkc["frames"], dk["frames"]) for k in f)
            assert all(np.array_equal(g["tokens_np"]["pos"], ds[i]["tokens_np"]["pos"]) for i, g in enumerate(dk["frames"]))

            class Shuffled(Frames):                                                                          # live frames in another order than the cache's
                tokens = list(reversed(ds.tokens))
            base_args = ["--seed", "0", "--max", "3"]
            refused_with(dump_live, "not the frames of the cache", runs, "h2", 0.0, (), Shuffled)
            with mutated(E.live_dump, "assert list(data.tokens) == list(ds.tokens)", "assert True or list(data.tokens) == list(ds.tokens)"):
                dump_live(runs, "h2", 0.0, (), Shuffled)                                                     # without the guard the shuffled frames are accepted: the refusal above is the test of it
            os.remove(os.path.join(runs, "h2", "token_val_live_n3.pkl"))
            sc = ["score", "--run", "h"] + base_args
            c0, c1 = E.score(E.parse_args(sc), runs, out), E.score(E.parse_args(sc + ["--live"]), runs, out)
            assert all(c0[d]["metrics"] == c1[d]["metrics"] for d in ("rec", "argmax")) and all(c1["rec"]["info"][k] == m[k] for k in E.LIVE_META) and "live" not in c0["rec"]["info"]
            assert os.path.exists(os.path.join(out, "h_token_card_argmax_live_n3.json"))
            d115 = ["--link-thr", "0.3", "--link-thr-long", "0.2", "--keep-thr", "0.6", "--no-none-gate"]
            par = E.score(E.parse_args(sc + ["--live", "--ref-run", "h", "--ref-cache"] + d115), runs, out)["rec"]
            rf, pr = par["info"]["ref_run"], par["info"]["paired_vs_ref_run"]
            assert os.path.exists(os.path.join(out, "h_token_card_rec_lt0.3_ltl0.2_kt0.6_nng_live_n3_vs_h_cache.json")) and rf["dump"]["path"].endswith("token_val_n3.pkl") and "live" not in rf, rf
            assert par["info"]["tokens_exact"] and all((pr[k]["diff"], pr[k]["lo"], pr[k]["hi"]) == (0.0, 0.0, 0.0) for k in ("topo_F1", "junction_F1")), (par["info"]["tokens_exact"], pr["topo_F1"])
            an = ["anatomy", "--run", "h"] + base_args + ["--link-thr-long", "0.05"]
            a0, a1 = (json.load(open(E.anatomy(E.parse_args(an + x), runs, out))) for x in ([], ["--live"]))
            assert os.path.exists(os.path.join(out, "h_token_anatomy_long_ltl0.05_kt0.6_live_n3.json")) and a0["delivered"] > 0 and all(a1[k] == m[k] for k in E.LIVE_META) and "live" not in a0
            assert all(a0[k] == a1[k] for k in a0 if k not in ("dump", "note")) and "the forward of the live dump itself" in a1["note"] and "a CPU forward of the head" in a0["note"]       # the dump's labels and concentration = the cache's
            grid1 = dict(E.SWEEP_GRIDS, perclass=[(0.3, 0.2, 0.6, False)])
            with patched((E, "SWEEP_GRIDS", grid1)):
                sw = json.load(open(E.sweep(E.parse_args(["sweep", "--run", "h", "--split", "val", "--grid", "perclass", "--live"] + base_args), runs, out)))
            assert os.path.basename(sw["dump"]["path"]) == "token_val_live_n3.pkl" and sw["rows"][0]["topo_F1"] == par["metrics"]["topo_F1"] and all(sw[k] == m[k] for k in E.LIVE_META)
            # a joint run whose B' made other tokens (pos + 1 m, the order reversed; B' at bs 2): its own tokens, labels and concentration everywhere
            dj, chunks = dump_live(runs, "j", 1.0, ("--bs", "2"))
            mj = dj["meta"]
            assert chunks == [2, 1] and (mj["bs_b"], mj["bs_head"], mj["tokens_exact"], mj["tokens_exact_frames"]) == (2, 8, False, 0) and mj["b_source"]["kind"] == "b_state" and mj["b_source"]["sha_nodes"] == E.state_sha(trained, "nodes.")
            frames3 = QE.val_frames()[:3]
            direct = lambda fr, kt=0.5, lt=None: E.score_preds([E.frame_prediction(f_["tokens_np"], f_ if lt is None else E.decode_variant(f_, f_["tokens_np"]["valid"], lt[0], kt, False, lt[1]), "rec", kt) for f_ in fr], frames3, log=lambda *x, **k: None)["graph"]  # noqa: E731
            own = direct(dj["frames"])
            cj = E.score(E.parse_args(["score", "--run", "j", "--live"] + base_args), runs, out)["rec"]
            assert all(cj["metrics"][k] == own[k] for k in E.KEYS) and cj["metrics"] != c1["rec"]["metrics"], "the joint run is scored on its own tokens"
            with mutated(E.read_dump, 'toks = [f["tokens_np"] for f in fr] if live else', 'toks = [ds[int(i)]["tokens_np"] for i in idx] if live else'):     # a read_dump that takes the cache's tokens: caught
                os.makedirs(os.path.join(tmp, "out_m"))
                try:
                    cm = E.score(E.parse_args(["score", "--run", "j", "--live"] + base_args), runs, os.path.join(tmp, "out_m"))["rec"]
                    assert any(cm["metrics"][k] != own[k] for k in E.KEYS), "mutant survived: the cache's tokens give the same numbers"
                except AssertionError as e:
                    assert "mutant survived" not in str(e), e
            with patched((E, "SWEEP_GRIDS", grid1)):
                swj = json.load(open(E.sweep(E.parse_args(["sweep", "--run", "j", "--split", "val", "--grid", "perclass", "--live"] + base_args), runs, out)))
            assert swj["rows"][0]["topo_F1"] == direct(dj["frames"], 0.6, (0.3, 0.2))["topo_F1"] != sw["rows"][0]["topo_F1"], "the sweep reads the joint run's own tokens"
            aj = json.load(open(E.anatomy(E.parse_args(["anatomy", "--run", "j", "--live"] + base_args + ["--link-thr-long", "0.05"]), runs, out)))
            parts, nparts, ngt = [], [], 0                                                                   # the anatomy rebuilt from j's records: its tokens, its live labels, its concentration
            for i, f in enumerate(dj["frames"]):
                t, lb = f["tokens_np"], f["lab"]
                g = E.gt_labels(t, t["valid"], td.table(i), lb["gt_id"], lb["uout"], lb["uin"])
                parts.append(E.anatomy_frame(f, t["valid"], g, 0.6))
                nparts.append(E.near_frame(f, t["valid"], dict(tokens_np=t, ytype=lb["ytype"].astype(np.int64), y=E.pair_matrix(lb["ypair"], ds.K)), 0.6, 0.05))
                ngt += len(g["gt_long"])
            want = dict(E.anatomy_numbers(parts, ngt, 0.05), concentration=E.pool_concentration([f["conc"] for f in dj["frames"]]), near_none=E.near_numbers(nparts))
            want = json.loads(json.dumps(E._jsonable(want)))
            assert all(aj[k] == want[k] for k in want) and any(aj[k] != a1[k] for k in want), [k for k in want if aj[k] != want[k]]
            # pairing: j (B' at bs 2) against h (bs 4) refused; against h2 (bs 2, the cache's tokens) paired by frame, with h2's live meta in the ref block; live-diff by hand
            refused_with(E.score, "another B' batch or device", E.parse_args(["score", "--run", "j", "--live", "--ref-run", "h"] + base_args), runs, out)
            dump_live(runs, "h2", 0.0, ("--bs", "2"))
            ch2 = E.score(E.parse_args(["score", "--run", "h2", "--live"] + base_args), runs, out)["rec"]
            cj2 = E.score(E.parse_args(["score", "--run", "j", "--live", "--ref-run", "h2"] + base_args), runs, out)["rec"]
            rf, pr = cj2["info"]["ref_run"], cj2["info"]["paired_vs_ref_run"]
            assert rf["dump"]["path"].endswith("token_val_live_n3.pkl") and rf["bs_b"] == 2 and rf["b_source"]["kind"] == "base" and abs(pr["topo_F1"]["diff"] - (cj["metrics"]["topo_F1"] - ch2["metrics"]["topo_F1"])) < 1e-12, rf
            ld = json.load(open(E.live_diff(E.parse_args(["live-diff", "--run", "j", "--against", "h2"] + base_args), runs, out)))
            nv = sum(int(np.asarray(ds[i]["tokens_np"]["valid"]).sum()) for i in range(3))
            assert os.path.exists(os.path.join(out, "j_token_live_diff_vs_h2_n3.json")) and ld["frames"] == 3 and ld["live"]["j"]["bs_b"] == 2
            assert ld["cells"] == dict(order_changed_share=1.0, set_changed_share=0.0) and ld["pos"]["common_valid_cells"] == nv and abs(ld["pos"]["median_abs_delta_m"] - np.sqrt(2)) < 1e-5 and ld["n_valid"]["delta_mean"] == 0.0, ld
            lh = ld["labels"]["h2"]
            want = {k: sum(int(ds.c.f[k][i]) for i in range(3)) for k in ("n_pairs", "n_links", "n_gt", "n_gt_assigned", "n_valid")}
            assert (lh["pos_pairs"], lh["gt_links"], lh["gt_coverage"], lh["n_valid_per_frame"]) == (want["n_pairs"], want["n_links"], want["n_gt_assigned"] / want["n_gt"], want["n_valid"] / 3), lh
            refused_with(E.live_diff, "never overwritten", E.parse_args(["live-diff", "--run", "j", "--against", "h2"] + base_args), runs, out)
            # a z run (train_nodes 0) that carries a b_state equal to the base (jlive_train always writes one): accepted live, refused on the cache path
            joint(runs, "z0", {k: v.clone() for k, v in base.items()}, train_nodes=0)
            dump_live(runs, "z0")
            assert E.score(E.parse_args(["score", "--run", "z0", "--live"] + base_args), runs, out)["rec"]["info"]["b_source"]["kind"] == "b_state"
            # the cache path refuses a joint run: dump, score (a cache dump under its name), graphs
            for name in ("j", "z0"):
                refused_with(E.dump, "read only with --live", E.parse_args(["dump", "--run", name, "--cpu"] + base_args), runs)
            d = pickle.load(open(os.path.join(runs, "h", "token_val_n3.pkl"), "rb"))
            d["meta"]["run"] = "j"
            pickle.dump(d, open(os.path.join(runs, "j", "token_val_n3.pkl"), "wb"))
            refused_with(E.score, "read only with --live", E.parse_args(["score", "--run", "j"] + base_args), runs, out)
            refused_with(E.graphs, "read only with --live", E.parse_args(["graphs", "--run", "j", "--seed", "0", "--max", "3", "--name", "J", "--out", os.path.join(tmp, "g.pkl")]), runs)
            # check_b: what ran must be what the run demands
            def refused(name, why):                                                                          # into an empty folder: no stored card of the run answers first
                refused_with(E.score, why, E.parse_args(["score", "--run", name, "--live"] + base_args), runs, os.path.join(tmp, "out_r"))
            joint(runs, "j2", trained)                                                                       # the joint run read with runs/e292_a_s0: a fall-back of live_source
            with mutated(E.live_source, '    if own:\n        mb.load_state_dict(ck["b_state"])', '    if False:\n        mb.load_state_dict(ck["b_state"])'):
                dump_live(runs, "j2", 1.0)
            refused("j2", "did not run the run's own B'")
            joint(runs, "j3", {k: v.clone() for k, v in base.items()})                                       # a joint run whose node head did not move
            dump_live(runs, "j3", 1.0)
            refused("j3", "ran the node head of runs/e292_a_s0")
            pyr = next(k for k in sorted(trained) if not k.startswith("nodes.") and trained[k].is_floating_point())
            joint(runs, "j4", dict(trained, **{pyr: trained[pyr] + 0.01}))                                   # a joint run whose B' moved outside its node head
            dump_live(runs, "j4", 1.0)
            refused("j4", "outside its node head")
            joint(runs, "j5", trained, b_sha256="0" * 64)                                                    # a joint run that started from another B' file
            dump_live(runs, "j5", 1.0)
            refused("j5", "not the B' the run started from")
            joint(runs, "z", None, train_nodes=0)                                                            # a frozen-B' run read with a trained node head
            d = pickle.load(open(os.path.join(runs, "j", "token_val_live_n3.pkl"), "rb"))
            d["meta"]["run"] = "z"
            pickle.dump(d, open(os.path.join(runs, "z", "token_val_live_n3.pkl"), "wb"))
            refused("z", "did not run the node head of runs/e292_a_s0")
            shutil.copy(os.path.join(runs, "h", "token_val_n3.pkl"), os.path.join(runs, "j", "token_val_live_n3.pkl"))
            refused("j", "not the dump of this read-out")                                                    # the cache dump under the live name: no fall-back to the cache tokens
    print(f"live: same_tokens pairs a live read-out by frame only (two live ones of one B' batch and device), pair_matrix, state_sha and b_shas, tokens_exact field by field over {len(E.TOKEN_KEYS)} fields, the refusals of --ref-cache, --bs "
          f"(without --live, 0) and --drop; live_source (the joint run's b_state, runs/e292_a_s0 for a cache-trained run, train_nodes without b_state refused); the stub source with the cache's tokens: 3 frames (chunk 3) and 10 frames "
          f"(two head batches, chunks 4 4 2) = the cache dump's records, tokens_exact, tokens_np and labels = the cache's; the frame-order guard (its mutant accepts shuffled frames); score / anatomy / sweep --live = the cache read-outs "
          f"with the live meta (topo F1 {c1['rec']['metrics']['topo_F1']:.4f}), the parity read with diff 0; the joint run (pos + 1 m, reversed, chunks 2 1): tokens_exact 0, scored on its own tokens (topo F1 {cj['metrics']['topo_F1']:.4f}; "
          f"the cache-token mutant caught), sweep and anatomy = a rebuild from its records, refused against a run of another B' batch, paired with one of the same, live-diff by hand (|delta pos| sqrt 2 over {nv} cells); a z run with "
          "a base b_state accepted; refused: a joint run on the cache path (dump, score, graphs), a fall-back to runs/e292_a_s0 (live_source mutant), a node head that did not move, a moved pyramid, another b_sha256, a frozen "
          "run with a trained head, the cache dump under the live name: ok")


TESTS = [test_keep_rule, test_decode_fork, test_decode_variant, test_decode_perclass, test_dump_roundtrip, test_writer_no_gt, test_drop, test_readout_dump, test_numbers, test_timing, test_sweep, test_every, test_score_max,
         test_graphs, test_snapshot, test_ref_run, test_ensemble, test_choose_cell, test_anatomy, test_near_none, test_argmax_keep, test_live]

if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtoken_eval ok")
