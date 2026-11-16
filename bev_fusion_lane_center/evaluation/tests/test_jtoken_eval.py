"""Tests of `jtoken_eval.py` (E365 S1 read-out, controls and checks); CPU only (the GPU is hidden), 2 threads. Hand frames need no data; the tests marked (cache) read the first VAL frames of the predicted-node cache of B' seed 0 under
runs/e365_cache (built by the lane chat) and are skipped with a message when it is not there.

    wsl bash tools/env/_py14.sh test_jtoken_eval.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

keep_rule (a node is kept at p_none 0.49 and not at 0.50, e = 1 - p_none, the links of a dropped node go), decode_fork (rec keeps both branches of a fork, argmax one; the none slot wins when the branches are weak),
dump_roundtrip (a real head forward: the dump holds exactly the decodes of `jtoken`, the candidate list, the keep mask), writer_no_gt (the prediction is made from the tokens only: mutants that put a GT section, a GT flag, a GT
position or an extra GT field into the writer are caught by `check_writer`; the dump frame and `frame_prediction` have no GT input), oracle_hi (cache: an oracle head whose links are the c-hi links delivers exactly c-hi, and scores the same; the y pairs
between assigned tokens are more than c-hi), drop (cache: reproducible, the seed matters, the share, equal to the loader's own `token_drop` at the same seed), controls (cache: c-rule, c-floor and a trained-style head run through the writer
and the standard scorer), readout_dump (B's stored dump frame -> prediction: sec0 zeroes the sections, the close-node step drops the weaker of two close nodes), numbers (the layer-1 comparison at its tolerance, g1, the sample, the parity
comparison inside / outside its tolerance, the matching of two token sets by cell), parity_timing (cache: the check-parity and the timing code paths on a tiny head, CPU).
"""
import contextlib
import json
import os
import sys
import tempfile

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
    print("decode_fork: record keeps both branches, argmax the better one, the none slot beats weak branches: ok")


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
    assert list(inspect.signature(E.frame_prediction).parameters) == ["tokens", "f", "decode"], "the writer's inputs are the tokens and the dump frame"
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
    def with_assign(frame, keep, e_node, links, flags=None):
        p = real(frame, keep, e_node, links, flags)
        p["assign"] = dict(P=np.zeros((1, 1)))
        return p
    with patched((LP, "to_pred", with_assign)):
        must_fail(E.frame_prediction, tok, f, "rec")
    tok2 = dict(tok, piece=np.arange(K), gt_id=np.arange(K), gt_flags=np.ones((K, 8)))                  # GT fields lying around in the token dict change nothing
    p2 = E.frame_prediction(tok2, f, "rec")
    assert all(np.array_equal(pred["nodes"][k], p2["nodes"][k]) for k in pred["nodes"]) and np.array_equal(pred["links"]["src"], p2["links"]["src"])
    print("writer_no_gt: a GT section / flag / position / s / extra field / assignment in the writer is caught by check_writer (6 mutants); GT fields in the token dict change nothing: ok")


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
    print("numbers: layer-1 comparison at 1e-6 (9e-7 passes, 2e-6 fails, a missing key fails), g1, the samples, the parity comparison inside / outside its tolerance: ok")


def test_parity_timing():
    if not CACHE_OK or not os.path.exists(SMOKE):
        print("parity_timing: SKIPPED (no cache or no smoke model)")
        return
    b_run = os.path.join(E.ROOT, "runs", "e292_a_s0")
    if not os.path.exists(os.path.join(b_run, "model.pt")):
        print("parity_timing: SKIPPED (no B' run)")
        return
    with tempfile.TemporaryDirectory() as runs:
        os.makedirs(os.path.join(runs, "h"))
        torch.save(torch.load(SMOKE, map_location="cpu", weights_only=False), os.path.join(runs, "h", "model.pt"))
        rc = E.check_parity(E.parse_args(["check-parity", "--run", "h", "--seed", "0", "--n", "2", "--tol", "0.05"]), runs)
        par = json.load(open(os.path.join(runs, "h", "token_parity.json")))
        assert rc in (0, 1) and par["frames"] == 2 and par["valid_cache_not_live"] < 10 and par["input_max_abs_diff_matched_valid_tokens"]["tok"] < 0.5, par
        E.timing(E.parse_args(["timing", "--run", "h", "--cpu", "--frames", "2", "--repeats", "1", "--b-run", os.path.relpath(b_run, E.ROOT)]), runs)
        t = json.load(open(os.path.join(runs, "h", "token_timing_cpu.json")))
        assert set(t["median_ms"]) == {"head", "bprime", "whole"} and t["params"] == JK.count_params(JK.JTokenHead()) and 0.05 < t["gmacs_head"] < 20 and t["device"] == "cpu" and t["params_whole_path"] == t["params"] + t["params_bprime"] < 2.74e6, t
        print(f"parity_timing: parity on 2 VAL frames (rc {rc}: dp_none {par['max_dpnone']:.2e}, dp {par['max_dp']:.2e}, tok diff {par['input_max_abs_diff_matched_valid_tokens']['tok']:.1e}); timing code path on the CPU: head {t['median_ms']['head']:.0f} ms, "
              f"whole {t['median_ms']['whole']:.0f} ms, {t['params']} parameters, {t['gmacs_head']:.2f} GMACs: ok")


TESTS = [test_keep_rule, test_decode_fork, test_dump_roundtrip, test_writer_no_gt, test_oracle_hi, test_drop, test_controls, test_readout_dump, test_numbers, test_parity_timing]

if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtoken_eval ok")
