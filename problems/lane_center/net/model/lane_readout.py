"""The read-out of the lane network, a named OUTPUT step: the token head's output of a batch -> one record per frame (`batch_dump`: the kept nodes, the candidate
links, the decode of record) -> the links of a decode (`decode_variant`: the four thresholds of `Decode`) -> the delivered graph of the frame in the contract of
`qformat` (`frame_prediction`, written by `lane_ptok_data.to_pred` from the frame's own tokens; `check_writer`: nothing else enters).

    recs = batch_dump(out, batch)                       # out = the head's output; batch: token, row, tokens_np (per frame: pos, dirv, e_p, flags_p, s_p, cell, valid)
    pred = graph(tokens_np, rec, tokens_np["valid"], Decode(link_thr=0.3, link_thr_long=0.2, keep_thr=0.6, none_gate=False))

Decode of record (the defaults of `Decode`): a node is kept when p_none < 0.5 (written e = 1 - p_none); a link i -> j of class c when p^c(j|i) >= 0.3, p^c(j|i) >=
p^c(none|i) and both ends are kept (a fork keeps both branches). The thresholds are the only non-differentiable steps here; a record keeps every candidate with
p >= KEEP_P, so another decode is read from the same record. Handle scales: the head's for the long links, 1/3 for the short ones; p = 1 is written for every kept
link (the decode is the selection). The predict step stores the records (evaluation/lane_predict.py), the evaluation decodes them (evaluation/adapter.py); this
file is in the evaluator hash. Tests: evaluation/tests/test_jtoken_eval.py, net/tests/test_lane_net.py (`LaneNet.graph`)."""
import typing

import numpy as np

import jtoken as JK
import lane_ptok_data as LP
import lane_tokcache as LC

KEEP_P = 0.05                                                               # candidate links below this probability are not written into a dump
NODE_KEYS = ["cell", "dirv", "e", "flags", "pos", "s", "section"]
THIRD = 1 / 3
LINK_THR, KEEP_THR = 0.3, 0.5                                               # the decode of record


class Decode(typing.NamedTuple):
    """the decode of a read-out, the arguments of `decode_variant`: the link threshold of the short class, of the long class (None: the short one's), the keep
    threshold of a node (p_none below it), the none gate (a link also needs p >= p(none|i)). The defaults are the decode of record; a run's is its config `decode`"""
    link_thr: float = LINK_THR
    link_thr_long: typing.Optional[float] = None
    keep_thr: float = KEEP_THR
    none_gate: bool = True


D115 = Decode(link_thr=0.3, link_thr_long=0.2, keep_thr=0.6, none_gate=False)       # the delivered decode: the champion's four thresholds (R7, D115)


def links_of(link, p, longm, scale):
    """the links of one frame from the boolean (K, K) link matrix of a decode, the probabilities, the chord class (> 8 m) and the head's handle scales (K, K, 2): the head's scales for the long links, 1/3 for the short ones"""
    return cand_of(*np.nonzero(link), p, longm, scale)


def cand_of(cs, cd, p, longm, scale):
    """the candidate links of a frame (p >= KEEP_P, in the gate): src, dst, p, long and the handle scales a, b (the head's for the long ones, 1/3 for the short ones, as `links_of`)"""
    lg = longm[cs, cd]
    ab = np.where(lg[:, None], scale[cs, cd], THIRD)
    return dict(src=cs.astype(np.int16), dst=cd.astype(np.int16), p=p[cs, cd].astype(np.float32), long=lg, a=ab[:, 0].astype(np.float32), b=ab[:, 1].astype(np.float32))


def batch_dump(out, b):
    """the head's output of a batch -> one dict per frame: token, row, cell (the tokens' cells: the score checks them against the cache), pnone / ptype (K,), keep (K,), none_cls (2, K) = p^c(none|i), cand = the candidate links with p >= KEEP_P
    (src, dst, p, long, a, b; the dumps of before the decode variants lack a, b and hold none_cls in float16), rec = the links of the decode of record (src, dst, p, long, a, b)"""
    K = out["cand"].shape[-1]
    pr, rec = JK.probs(out), JK.decode(out)
    npy = lambda t: t.detach().cpu().numpy()  # noqa: E731
    pt, p, cand, longm, scale, none_cls = (npy(x) for x in (pr["pt"], pr["p"], pr["cand"], pr["longm"], out["scale"].float(), out["logp"][..., K].float().exp()))
    keep, lrec = npy(rec["keep"]), npy(rec["link"])
    pos = npy(out["pos"].float()) if "pos" in out else None                   # the head's refined node positions (`jtoken.refine_positions`)
    frames = []
    for i in range(len(b["token"])):
        cs, cd = np.nonzero((p[i] >= KEEP_P) & cand[i])
        frames.append(dict(token=b["token"][i], row=int(b["row"][i]), cell=np.asarray(b["tokens_np"][i]["cell"], np.int16), pnone=pt[i, :, 0].astype(np.float32), ptype=pt[i].astype(np.float16), keep=keep[i],
                           none_cls=none_cls[i].astype(np.float32), cand=cand_of(cs, cd, p[i], longm[i], scale[i]), rec=links_of(lrec[i], p[i], longm[i], scale[i]),
                           **({} if pos is None else dict(pos=pos[i].astype(np.float32)))))
    return frames


def check_writer(pred, tokens, keep):
    """the prediction of a frame holds the kept TOKENS and nothing else: the node fields are the tokens' own (pos, dirv, B''s flags and s) at the kept indices, one section, no assignment, no maps, e >= 0.5 (a kept node), the links name nodes of it"""
    nd, lk, idx = pred["nodes"], pred["links"], np.flatnonzero(keep)
    assert sorted(nd) == NODE_KEYS and pred["assign"] is None and pred["maps"] is None and lk["curves"] is None, (sorted(nd), pred["assign"] is None, pred["maps"] is None)
    for k, v in (("pos", tokens["pos"]), ("dirv", tokens["dirv"]), ("flags", tokens["flags_p"]), ("s", tokens["s_p"])):
        assert np.array_equal(nd[k], np.asarray(v, float)[idx]), f"node field {k} is not the tokens' own"
    assert not np.any(nd["section"]) and len(nd["section"]) == len(idx) and bool((nd["e"] >= 0.5).all()), "section / e of the written nodes"
    c = np.asarray(tokens["cell"], int)[idx]
    assert np.array_equal(nd["cell"], np.stack([c // LC.N, c % LC.N], 1)), "node cells are not the tokens' cells"
    assert all(len(lk[k]) == len(lk["src"]) for k in ("dst", "p", "long", "a", "b")) and (len(lk["src"]) == 0 or max(lk["src"].max(), lk["dst"].max()) < len(idx)), "links outside the nodes"
    return pred


def node_e(pnone, keep_thr=KEEP_THR):
    """the existence written for a kept node: 1 - p_none, shifted by keep_thr - 0.5 and capped at 1, so that the scorer's `e >= 0.5` is `p_none < keep_thr`; at the record's 0.5 exactly 1 - p_none"""
    return np.minimum(1.0 - np.asarray(pnone, np.float32).astype(float) + (keep_thr - 0.5), 1.0)


def frame_prediction(tokens, f, decode, keep_thr=KEEP_THR):
    """the qformat prediction of one frame from its tokens (`tokens_np`) and its dump frame (a record with `pos`: the head's refined positions stand for the tokens'): kept = f['keep'] (the head's p_none < keep_thr), e = `node_e` (1 - p_none at the record's keep_thr 0.5), the links of the decode (p = 1).
    Asserted on every frame: the nodes are the kept tokens' own (`check_writer`), e = `node_e` of the dump's p_none, the links are exactly the dump's links of the decode between kept nodes (renumbered). The real no-GT guarantee is
    upstream: the head reads only `jtoken.INPUTS` (test_jtoken.py)"""
    lk = dict(f[decode], p=np.ones(len(f[decode]["src"])))
    if "pos" in f:
        tokens = dict(tokens, pos=f["pos"])
    pred = check_writer(LP.to_pred(tokens, f["keep"], node_e(f["pnone"], keep_thr), lk), tokens, f["keep"])
    keep, idx = np.asarray(f["keep"], bool), np.flatnonzero(f["keep"])
    assert np.array_equal(pred["nodes"]["e"], node_e(f["pnone"], keep_thr)[idx]), f"e is not min(1 - p_none + {keep_thr} - 0.5, 1) of the dump"
    new = np.cumsum(keep) - 1
    s_, d_ = np.asarray(f[decode]["src"], int), np.asarray(f[decode]["dst"], int)
    ok = keep[s_] & keep[d_] & (s_ != d_)
    want = sorted(zip(new[s_[ok]].tolist(), new[d_[ok]].tolist()))
    assert sorted(zip(pred["links"]["src"].tolist(), pred["links"]["dst"].tolist())) == want, "the links are not the dump's links of the decode"
    return pred


LINK_FIELDS = ("src", "dst", "p", "long", "a", "b")


def decode_variant(f, valid, link_thr=LINK_THR, keep_thr=KEEP_THR, none_gate=True, link_thr_long=None):
    """a decode of the dump frame `f` other than the record: nodes kept at p_none < keep_thr (valid tokens), links = every candidate with p >= the threshold of its class (link_thr for the short class, link_thr_long for the long
    class = the dump's `long`, the token chord > 8 m of the head's candidates; None: link_thr for both; both >= KEEP_P, the dump holds the candidates above it), and p >= p(none|i) of the class of the pair unless `none_gate` is off, both
    ends kept -> the frame with `keep` and `rec` replaced; the handle scales are the dump's (`cand` a, b). A dump of before the variants (no scales in its candidates) is refused: re-dump. The default arguments give exactly the decode
    of record (`check_record`)"""
    thr_long = link_thr if link_thr_long is None else link_thr_long
    assert min(link_thr, thr_long) >= KEEP_P, f"the dump holds the candidates with p >= {KEEP_P}: --link-thr / --link-thr-long below it cannot be re-scored"
    c = f["cand"]
    assert "a" in c, f"frame {f['token']}: a dump of before the decode variants (no handle scales in its candidates): re-dump"
    keep = np.asarray(valid, bool) & (np.asarray(f["pnone"]) < np.float32(keep_thr))
    src, dst = c["src"].astype(int), c["dst"].astype(int)
    thr = np.array([link_thr, thr_long], np.float32)[c["long"].astype(int)]                 # the threshold of the pair's class (short, long)
    m = (c["p"] >= thr) & keep[src] & keep[dst]
    if none_gate:
        m &= c["p"] >= np.asarray(f["none_cls"], np.float32)[c["long"].astype(int), src]
    return dict(f, keep=keep, rec={k: c[k][m] for k in LINK_FIELDS})


def check_record(frames, valid):
    """before a variant is scored: on every frame `decode_variant` with the defaults equals the dump's decode of record on all six link fields and the kept nodes"""
    for f, v in zip(frames, valid):
        d = decode_variant(f, v)
        assert np.array_equal(d["keep"], f["keep"]) and all(np.array_equal(d["rec"][k], f["rec"][k]) for k in LINK_FIELDS), f"frame {f['token']}: the default decode variant is not the dump's decode of record"


def graph(tokens, f, valid, d=Decode()):
    """the delivered graph of one frame: its record `f` (`batch_dump`) at the decode `d`, written from its tokens (`tokens_np`; valid = the tokens the head saw)"""
    return frame_prediction(tokens, decode_variant(f, valid, d.link_thr, d.keep_thr, d.none_gate, d.link_thr_long), "rec", d.keep_thr)
