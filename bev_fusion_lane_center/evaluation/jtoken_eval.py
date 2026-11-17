"""Read-out, scoring, controls and checks of E365 S1 (the token head on the predicted nodes of the frozen lane head B', `training/grid/jtoken.py`); the cache and the loader are `training/data/lane_tokcache.py` / `lane_ptok_data.py`.

    wsl bash tools/env/_py14.sh jtoken_eval.py dump  --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0] [--every N] [--cpu] [--max N]   # GPU: the head on the cache tokens -> runs/R/token_<split>[_drop..].pkl
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0] [--every N] [--max N]                     # CPU -> runs/R/token_card_<decode>[_train][_drop..].json (rec, argmax; never overwritten)
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S --link-thr T --keep-thr K [--no-none-gate]                                      # CPU: another decode of the same dump -> token_card_rec_lt<T>_kt<K>[_nng][_train]..json
    wsl bash tools/env/_py14.sh jtoken_eval.py sweep --run R --seed S --split train [--every N] [--max N]                                                       # CPU: the grid of decode variants on the TRAIN sample -> runs/R/token_sweep_train.json
    wsl bash tools/env/_py14.sh jtoken_eval.py ctl   --seed S --ident e292_a|e273_p [--arms hi,hi_y,rule,floor,bsec0_plain,bsec0_close] [--max N]   # CPU: the controls -> runs/eval/e365_ctl/<arm>_<ident>_s<S>_K<K>.json
    wsl bash tools/env/_py14.sh jtoken_eval.py check-layer1 --seed S [--dump PATH]                                                       # CPU: this scorer fed B's stored dump reproduces runs/eval/e324s_val/card_s<S>.json (PASS / FAIL)
    wsl bash tools/env/_py14.sh jtoken_eval.py check-parity --run R --seed S [--b-run runs/e292_a_s<S>] [--n 20]                         # CPU: the head from the cache tokens (fp16 tok) against tokens recomputed live from B' (fp32)
    wsl bash tools/env/_py14.sh jtoken_eval.py timing --run R [--b-run runs/e292_a_s<S>] [--K 384] [--cpu]                               # parameters, GMACs, ms (GPU only when idle: < 2 GB used and no run.py job)

Read-out of record (named before the run, on the output only): a node is kept when p_none < 0.5 (e = 1 - p_none); decode `rec`: a link i -> j of class c when p^c(j|i) >= 0.3, p^c(j|i) >= p^c(none|i) and both ends are kept (a fork keeps both
branches); decode `argmax`: per source and class the argmax over {none} + the candidates (one link per source and class). The prediction is written by `lane_ptok_data.to_pred` from the cached TOKENS only (pos, dirv, B''s flags and s; every node in
section 0): no field of the GT table enters (the real guarantee: the head reads only `jtoken.INPUTS`, test_jtoken.py; `check_writer` / `frame_prediction` assert on every frame that the nodes are the kept tokens' own, e = 1 - p_none and the links are the dump's) and the scorer is the standard one (`qgraph_eval.readout` -> `graph_counts` -> `graph_numbers` through `score_frames`, inside the `jtok.lattice(2.0)` block because
`LaneRef` builds a node table at score time; the headline graph numbers use the old graph GT and do not depend on it: `check-layer1` proves it). Handle scales: the head's for the long links, 1/3 for the short ones. p = 1 is written for every kept link (the
decode is the selection); the probability stays in the dump. A TRAIN read-out is 873 frames spaced evenly over the cache (`sample`), the degraded read-out (`--drop`) removes the share of the tokens that E357's `keep_nodes` removes (seeded by `--drop-seed` and
the frame token) before the head reads them.
Controls (no learning, the same tokens, the same writer and scorer): hi = the GT links of the frame's table between the ASSIGNED tokens (nodes = the assigned tokens), contracted over the GT nodes without an assigned token; hi_y = the cache's y
pairs between assigned tokens (diagnostic: y also holds tokens that are assigned to the successor's neighbours); rule = E357's `lo_near` on the nodes with
B''s e >= 0.5; floor = the untrained head (random init seed 0, the key weights of a trained head), decode of record; bsec0_plain / bsec0_close = B's own stored nodes and links with every section 0, read out as `plain@0.5` / `close:1.5@0.5`
(`--ident e292_a`: B' with the Matrix-NMS layer, `e273_p`: plain B). Tests: evaluation/tests/test_jtoken_eval.py
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtok as JT  # noqa: E402
import jtok_eval as JV  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402

RUNS = JV.RUNS
CTL = os.path.join(JV.CARDS, "e365_ctl")
SCRATCH = os.path.join(ROOT, "temp", "e365")
DECODES = ("rec", "argmax")
ARMS = ("hi", "hi_y", "rule", "floor", "bsec0_plain", "bsec0_close")
LATTICE = 2.0
NSAMPLE = 873                                                               # frames of a TRAIN read-out
KEEP_P = 0.05                                                               # candidate links below this probability are not written into a dump
KEYS = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "Q", "junction_F1", "dup_frac", "vert100_dp")
CHECK = ("topo_F1", "F1@1.0", "lanes_found", "dup_frac", "vert100_dp")      # the layer-1 numbers (to 1e-6)
BASE_CARD = os.path.join(JV.CARDS, "e324s_val", "card_s{seed}.json")        # variants.primary.numbers.base = plain B with the close-node step, B's own sections
B_DUMPS = dict(e273_p="temp/e343/val/predfull_e273_p_s{seed}.pkl", e292_a="temp/e292/predfull_e292_a_s{seed}.pkl")      # the fresh VAL dump of plain B made the E324 s card; B' = the dump of E292
CLOSE, PLAIN = "close:1.5@0.5", "plain@0.5"
NODE_KEYS = ["cell", "dirv", "e", "flags", "pos", "s", "section"]
THIRD = 1 / 3
UNMATCHED_LIMIT = 0.005                                                      # parity: the valid tokens that only one side has, at most this share of the cache's valid tokens
MACS_NOTE = "matmul, linear and einsum only; softmax, norms, GELU, the pair-geometry features and the sort are not counted"


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x.item() if isinstance(x, (np.generic,)) else (x.tolist() if isinstance(x, np.ndarray) else x)


def _file_id(path):
    """sha256 and mtime of a file (the identity of a stored dump)"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return dict(path=os.path.relpath(path, ROOT) if path.startswith(ROOT) else path, sha256=h.hexdigest(), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(path))) + " UTC")


def _write_stamped(folder, stem, obj):
    """a result file that is never skipped and never overwritten: <stem>_<UTC>.json (a counter when two land in the same second)"""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path, n = os.path.join(folder, f"{stem}_{stamp}.json"), 1
    while os.path.exists(path):
        path, n = os.path.join(folder, f"{stem}_{stamp}_{n}.json"), n + 1
    JV.write_json(path, _jsonable(obj))
    return path


# ------------------------------------------------------------------------------------------------------------------------------ tokens and the head
def cache_kw(args):
    """the loader arguments of a run's recipe (`jtoken_train` writes ident / K / cache into the run's args; an older smoke run has none: B', K found, the default root)"""
    return dict(seed=args["seed"], ident=args.get("ident") or "e292_a", K=args.get("K"), **({"cache": args["cache"]} if args.get("cache") else {}))


def open_cache(split, seed, ident="e292_a", K=None, cache=None):
    return LP.PTokData(split, seed, ident, K, 0.0, **({"cache": cache} if cache else {}))


def sample(n, split, every=1):
    """cache indices of the frames that are read out: all of VAL, NSAMPLE evenly spaced frames of TRAIN; every > 1 takes every `every`-th frame of that sample (the first, the (every+1)-th, ...)"""
    idx = np.arange(n) if split == "val" else np.unique(np.linspace(0, n - 1, NSAMPLE).round().astype(int))
    return idx[::every]


def every_sfx(a):
    """'' or _every<N> in the file names of a read-out that takes every N-th frame of the sample"""
    return f"_every{a.every}" if a.every > 1 else ""


def batches(ds, idx, bs=8, workers=0):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [int(i) for i in idx]), batch_size=bs, shuffle=False, num_workers=workers, collate_fn=LP.collate)


def apply_drop(b, drop, drop_seed):
    """the degraded read-out: every token of a frame is removed with probability `drop` (`jtok_eval.keep_nodes`, seeded by (drop_seed, the frame token)) before the head reads it; the removed token is not valid (the loader's `token_drop` does the same with
    the cache seed as drop seed)"""
    if drop > 0:
        keep = np.stack([JV.keep_nodes(t, b["valid"].shape[1], drop, drop_seed) for t in b["token"]])
        b["valid"] = b["valid"] & torch.from_numpy(keep)
        for t, k in zip(b["tokens_np"], keep):
            t["valid"] = t["valid"] & k
    return b


def run_head(m, b, dev):
    with torch.no_grad():
        return m(JKT.to_device(b, dev))


def links_of(link, p, longm, scale):
    """the links of one frame from the boolean (K, K) link matrix of a decode, the probabilities, the chord class (> 8 m) and the head's handle scales (K, K, 2): the head's scales for the long links, 1/3 for the short ones"""
    src, dst = np.nonzero(link)
    lg = longm[src, dst]
    ab = np.where(lg[:, None], scale[src, dst], THIRD)
    return dict(src=src.astype(np.int16), dst=dst.astype(np.int16), p=p[src, dst].astype(np.float32), long=lg, a=ab[:, 0].astype(np.float32), b=ab[:, 1].astype(np.float32))


def cand_of(cs, cd, p, longm, scale):
    """the candidate links of a frame (p >= KEEP_P, in the gate): src, dst, p, long and the handle scales a, b (the head's for the long ones, 1/3 for the short ones, as `links_of`)"""
    lg = longm[cs, cd]
    ab = np.where(lg[:, None], scale[cs, cd], THIRD)
    return dict(src=cs.astype(np.int16), dst=cd.astype(np.int16), p=p[cs, cd].astype(np.float32), long=lg, a=ab[:, 0].astype(np.float32), b=ab[:, 1].astype(np.float32))


def batch_dump(out, b):
    """the head's output of a batch -> one dict per frame: token, row, cell (the tokens' cells: the score checks them against the cache), pnone / ptype (K,), keep (K,), none_cls (2, K) = p^c(none|i), cand = the candidate links with p >= KEEP_P
    (src, dst, p, long, a, b; the dumps of before the decode variants lack a, b and hold none_cls in float16), rec / argmax = the links of the two decodes (src, dst, p, long, a, b)"""
    K = out["cand"].shape[-1]
    pr, rec, alt = JK.probs(out), JK.decode(out), JK.decode_argmax(out)
    npy = lambda t: t.detach().cpu().numpy()  # noqa: E731
    pt, p, cand, longm, scale, none_cls = (npy(x) for x in (pr["pt"], pr["p"], pr["cand"], pr["longm"], out["scale"].float(), out["logp"][..., K].float().exp()))
    keep, lrec, lalt = npy(rec["keep"]), npy(rec["link"]), npy(alt["link"])
    frames = []
    for i in range(len(b["token"])):
        cs, cd = np.nonzero((p[i] >= KEEP_P) & cand[i])
        frames.append(dict(token=b["token"][i], row=int(b["row"][i]), cell=np.asarray(b["tokens_np"][i]["cell"], np.int16), pnone=pt[i, :, 0].astype(np.float32), ptype=pt[i].astype(np.float16), keep=keep[i],
                           none_cls=none_cls[i].astype(np.float32), cand=cand_of(cs, cd, p[i], longm[i], scale[i]),
                           rec=links_of(lrec[i], p[i], longm[i], scale[i]), argmax=links_of(lalt[i], p[i], longm[i], scale[i])))
    return frames


# ------------------------------------------------------------------------------------------------------------------------------ the writer
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


LINK_THR, KEEP_THR = 0.3, 0.5                                               # the decode of record


def node_e(pnone, keep_thr=KEEP_THR):
    """the existence written for a kept node: 1 - p_none, shifted by keep_thr - 0.5 and capped at 1, so that the scorer's `e >= 0.5` is `p_none < keep_thr`; at the record's 0.5 exactly 1 - p_none"""
    return np.minimum(1.0 - np.asarray(pnone, np.float32).astype(float) + (keep_thr - 0.5), 1.0)


def frame_prediction(tokens, f, decode, keep_thr=KEEP_THR):
    """the qformat prediction of one frame from its tokens (`tokens_np`) and its dump frame: kept = the head's p_none < 0.5, e = 1 - p_none, the links of the decode (p = 1). Asserted on every frame: the nodes are the kept tokens' own
    (`check_writer`), e = 1 - p_none of the dump, the links are exactly the dump's links of the decode between kept nodes (renumbered). The real no-GT guarantee is upstream: the head reads only `jtoken.INPUTS` (test_jtoken.py)"""
    lk = dict(f[decode], p=np.ones(len(f[decode]["src"])))
    pred = check_writer(LP.to_pred(tokens, f["keep"], node_e(f["pnone"], keep_thr), lk), tokens, f["keep"])
    keep, idx = np.asarray(f["keep"], bool), np.flatnonzero(f["keep"])
    assert np.array_equal(pred["nodes"]["e"], node_e(f["pnone"], keep_thr)[idx]), "e is not 1 - p_none of the dump"
    new = np.cumsum(keep) - 1
    s_, d_ = np.asarray(f[decode]["src"], int), np.asarray(f[decode]["dst"], int)
    ok = keep[s_] & keep[d_] & (s_ != d_)
    want = sorted(zip(new[s_[ok]].tolist(), new[d_[ok]].tolist()))
    assert sorted(zip(pred["links"]["src"].tolist(), pred["links"]["dst"].tolist())) == want, "the links are not the dump's links of the decode"
    return pred


def decode_variant(f, valid, link_thr=LINK_THR, keep_thr=KEEP_THR, none_gate=True):
    """a decode of the dump frame `f` other than the record: nodes kept at p_none < keep_thr (valid tokens), links = every candidate with p >= link_thr (>= KEEP_P, the dump holds the candidates above it), and p >= p(none|i) of the class of the pair
    unless `none_gate` is off, both ends kept -> the frame with `keep` and `rec` replaced. The handle scales are the dump's (`cand` a, b); a dump of before them gives the scales of the record's links where the link is one and 1/3 else. The default
    arguments give exactly the decode of record"""
    assert link_thr >= KEEP_P, f"the dump holds the candidates with p >= {KEEP_P}: --link-thr below it cannot be re-scored"
    keep = np.asarray(valid, bool) & (np.asarray(f["pnone"]) < np.float32(keep_thr))
    c = f["cand"]
    src, dst = c["src"].astype(int), c["dst"].astype(int)
    m = (c["p"] >= np.float32(link_thr)) & keep[src] & keep[dst]
    if none_gate:
        m &= c["p"] >= np.asarray(f["none_cls"], np.float32)[c["long"].astype(int), src]
    if "a" in c:
        a, b = c["a"][m], c["b"][m]
    else:
        known = {(int(s_), int(d_)): (a_, b_) for dec in DECODES for s_, d_, a_, b_ in zip(f[dec]["src"], f[dec]["dst"], f[dec]["a"], f[dec]["b"])}
        ab = np.array([known.get((int(s_), int(d_)), (THIRD, THIRD)) for s_, d_ in zip(src[m], dst[m])], np.float32).reshape(-1, 2)
        a, b = ab[:, 0], ab[:, 1]
    return dict(f, keep=keep, rec=dict(src=c["src"][m], dst=c["dst"][m], p=c["p"][m], long=c["long"][m], a=a, b=b))


# ------------------------------------------------------------------------------------------------------------------------------ the controls
def hi_frame(tokens, gt_id, table):
    """c-hi, the ceiling of the token set: the GT links of the frame's table between the ASSIGNED tokens (`gt_id` >= 0: the nodes = the assigned tokens, e = 1), contracted over the GT nodes that have no assigned token (`jtok_eval.sub_table`);
    fitted handle scales for the long links that have one, 1/3 else -> (keep, e, links)"""
    a = np.asarray(gt_id) >= 0
    has = np.zeros(len(table["pos"]), bool)
    has[np.asarray(gt_id)[a]] = True
    tok_of = -np.ones(len(has), int)
    tok_of[np.asarray(gt_id)[a]] = np.flatnonzero(a)
    sub = JV.sub_table(table, has)
    nodes = np.flatnonzero(has)
    src, dst = tok_of[nodes[sub["src"]]], tok_of[nodes[sub["dst"]]]
    return a, np.ones(len(a)), _links(tokens, src, dst, np.asarray(sub["scale"], float).reshape(-1, 2))


def hi_y_frame(tokens, gt_id, y, yscale):
    """c-hi-y, a diagnostic: the cache's y pairs between assigned tokens (what a perfect head of the cache's targets would decode: y holds every token within 1.5 m of the GT successor, so also tokens that are assigned to the successor's
    neighbours) -> (keep, e, links)"""
    a = np.asarray(gt_id) >= 0
    ii, jj = np.nonzero(np.asarray(y).astype(bool) & a[:, None] & a[None, :])
    return a, np.ones(len(a)), _links(tokens, ii, jj, np.asarray(yscale)[ii, jj])


def _links(tokens, src, dst, scale):
    """the links of a control: p = 1, the long class by the tokens' chord, the fitted scales (NaN: none) for the long links, 1/3 else"""
    lg = np.linalg.norm(tokens["pos"][src] - tokens["pos"][dst], axis=1) > JT.SHORT
    ab = np.where(lg[:, None] & np.isfinite(scale), scale, THIRD)
    return dict(src=src, dst=dst, p=np.ones(len(src)), long=lg, a=ab[:, 0], b=ab[:, 1])


def rule_frame(tokens):
    """c-rule: E357's `lo_near` links on the nodes with B''s e >= 0.5 (valid tokens) -> (keep, e, links)"""
    keep = np.asarray(tokens["valid"], bool) & (np.asarray(tokens["e_p"]) >= 0.5)
    idx = np.flatnonzero(keep)
    pos, dirv = tokens["pos"][idx], tokens["dirv"][idx]
    src, dst = JV.rule_links(pos, dirv)
    s2, d2 = JV.open_links(pos, dirv, src, dst, every=False)
    src, dst = idx[np.concatenate([src, s2]).astype(int)], idx[np.concatenate([dst, d2]).astype(int)]
    return keep, np.asarray(tokens["e_p"], float), _links(tokens, src, dst, np.full((len(src), 2), np.nan))


def control_prediction(tokens, keep, e, links):
    return check_writer(LP.to_pred(tokens, keep, e, links), tokens, keep)


def floor_model():
    """c-floor: the untrained head, random init (seed 0), in the mode of a trained one (the key weight of its own p_none)"""
    torch.manual_seed(0)
    m = JK.JTokenHead()
    m.set_step(JK.SWITCH)
    return m.eval()


def pred_of_dump(d, spec, sec0):
    """a stored full dump frame of B (`qgraph_readout.dump_full`) -> the prediction as the read-out `spec` sees it (as `jtest_eval.score_dumps`); sec0: every node in section 0"""
    nodes = dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=np.zeros_like(d["section"]) if sec0 else d["section"], s=d.get("s"))
    return QR.apply(dict(nodes=nodes, links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), QR.parse(spec))


# ------------------------------------------------------------------------------------------------------------------------------ scoring
def frames_of(split, ds):
    """the scorer's Frame of every cache frame, in the cache order (VAL = `qgraph_eval.val_frames()`; TRAIN from the train files of the same layout)"""
    if split == "val":
        fr = QE.val_frames()
    else:
        tokens = json.load(open(f"{QE.LS.B123}/feats/train_tokens.json"))
        scene = np.load(f"{QE.LS.B123}/poses_train.npz")["scene"]
        gt = pickle.load(open(f"{QE.LS.B123}/graph_gt_train.pkl", "rb"))
        fr = [QE.Frame(k, int(r), tokens[r], int(scene[r]), gt[r]) for k, r in enumerate(ds.rows)]
    assert [f.token for f in fr] == list(ds.tokens), "the scorer's frames are not the frames of the cache, in order"
    return fr


def score_preds(preds, frames, log=print):
    """the standard scorer on a list of predictions (qformat dicts) inside the lattice block -> dict(graph, nodes: the flat numbers, rec: the per-frame records for the paired bootstrap, lane_missing)"""
    rec, miss = dict(graph=[], node=[], assign=[]), 0
    t0 = time.time()
    with JT.lattice(LATTICE):
        JT.check_lattice(LATTICE)
        for a in range(0, len(frames), 100):
            out, r = QE.score_frames(preds[a:a + 100], frames[a:a + 100], records=True)
            miss += out["lane_missing"]
            for k in rec:
                rec[k] += r[k]
            log(f"   {min(a + 100, len(frames))}/{len(frames)} frames, {time.time() - t0:.0f} s", flush=True)
    graph = {k: v for k, v in QE.graph_numbers(rec["graph"]).items() if k not in JV.SEC_VOID}
    return dict(graph=graph, nodes=QE.node_numbers(QE.pool_counts(rec["node"])) if rec["node"] else {}, rec=rec, lane_missing=miss)


def show(tag, m):
    print(f"{tag:<34}" + "  ".join(f"{k} {m.get(k, float('nan')):.4f}" for k in KEYS), flush=True)


def write_card(path, name, res, n_frames, split="val", **info):
    """a card of one scored arm: the graph numbers + the node numbers (`node_`), never overwritten"""
    assert not os.path.exists(path), f"a stored card is never overwritten: {path}"
    metrics = dict(res["graph"], **{f"node_{k}": v for k, v in res["nodes"].items()})
    card = JV.make_card(name, n_frames, metrics, "sec0", lane_missing=res["lane_missing"], **info)
    card["split"] = split
    card["protocol"]["scorer"] = "qgraph_eval.score_frames (readout -> graph_counts -> graph_numbers) inside the 2 m lattice block, against the old graph GT; nodes: node_numbers against the lane GT nodes (node_*)"
    card["protocol"]["reference"] = "val, the frames of qgraph_eval.val_frames" if split == "val" else f"{split}, the frames of the cache (rows of QData('{split}') with a stored lane GT), evenly spaced when more than {NSAMPLE}; old graph GT of graph_gt_{split}.pkl"
    JV.write_json(path, _jsonable(card))
    return card


def planb_path(seed):
    return os.path.join(SCRATCH, f"planb_s{seed}.pkl")


def plain_b(seed, dump=None):
    """plain B as the E324 s card delivers it (B's own sections, the close-node step, links at p >= 0.5, no connectors) scored by THIS scorer -> the result of `score_preds` + `dump` (path, sha256, mtime of the dump used)"""
    dump_path = _abs(dump or B_DUMPS["e273_p"].format(seed=seed))
    d = pickle.load(open(dump_path, "rb"))
    frames = QE.val_frames()
    assert [x["row"] for x in d] == [f.row for f in frames], f"{dump_path} is not in the order of the VAL frames"
    res = score_preds([pred_of_dump(x, CLOSE, False) for x in d], frames)
    res["dump"] = _file_id(dump_path)
    return res


def stored_plain_b(seed):
    """the scored plain B that `check-layer1` stored (only a PASS stores it): `score` needs it for the paired intervals and refuses without it"""
    path = planb_path(seed)
    assert os.path.exists(path), f"{path} is missing: run `check-layer1 --seed {seed}` first (it stores plain B's records only when it reproduces the E324 s card)"
    return pickle.load(open(path, "rb"))


def bsec0_close_path(ident, seed, n=0):
    return os.path.join(SCRATCH, f"bsec0_close_{ident}_s{seed}{f'_n{n}' if n else ''}.pkl")


def compare_numbers(got, ref, keys=CHECK, tol=1e-6):
    """-> (ok, {key: got - ref}) for the keys of a stored card"""
    assert all(k in got and k in ref for k in keys), f"missing numbers: {[k for k in keys if k not in got or k not in ref]}"
    diff = {k: float(got[k]) - float(ref[k]) for k in keys}
    return all(abs(v) <= tol for v in diff.values()), diff


def g1(card_s, ctl_hi, ctl_rule):
    """the gap share (S1 - c-rule) / (c-hi - c-rule) on topo F1 (NaN when the controls do not differ); it can exceed 1: the head may keep duplicate tokens that c-hi does not"""
    d = ctl_hi - ctl_rule
    return (card_s - ctl_rule) / d if abs(d) > 1e-12 else float("nan")


G1_NOTE = "g1 can exceed 1: the head may keep duplicate tokens that c-hi does not"


def ctl_name(arm, ident, seed, K, n=0):
    return f"{arm}_{ident}_s{seed}_K{K}{f'_n{n}' if n else ''}.json"


def read_ctl(arm, ident, seed, K, ctl_dir=CTL, n=0):
    """topo F1 of a stored control card (`ctl_name`; a card of an earlier version without the K in its name counts when its recorded K is this one), None when there is none"""
    for name in (ctl_name(arm, ident, seed, K, n), f"{arm}_{ident}_s{seed}{f'_n{n}' if n else ''}.json"):
        p = os.path.join(ctl_dir, name)
        if os.path.exists(p):
            c = json.load(open(p))
            if c["info"].get("K", K) == K:
                return c["metrics"]["topo_F1"]
    return None


# ------------------------------------------------------------------------------------------------------------------------------ commands
def dump(a, runs=RUNS):
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU; --cpu only for tests"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jtoken_eval dump {a.run}")
    m, args = JKT.load_model(a.run, dev, runs)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split, a.every)[: a.max or None]
    path = os.path.join(runs, a.run, f"token_{a.split}{every_sfx(a)}{JV.suffix(2.5, a.drop, a.drop_seed)}{f'_n{len(idx)}' if a.max else ''}.pkl")
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    frames = []
    for b in batches(ds, idx, workers=0 if a.cpu else 2):
        frames += batch_dump(run_head(m, apply_drop(b, a.drop, a.drop_seed), dev), b)
    meta = dict(run=a.run, seed=a.seed, split=a.split, drop=a.drop, drop_seed=a.drop_seed, idx=idx.tolist(), every=a.every, tokens=[ds.tokens[i] for i in idx], K=ds.K, ident=kw["ident"], cache=ds.dir, steps=args.get("steps"))
    with open(path, "wb") as fh:
        pickle.dump(dict(meta=meta, frames=frames), fh)
    kept = np.mean([f["keep"].sum() for f in frames])
    print(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames, kept nodes per frame {kept:.1f}, links per frame rec {np.mean([len(f['rec']['src']) for f in frames]):.1f} argmax {np.mean([len(f['argmax']['src']) for f in frames]):.1f}, device {dev}")
    return path


def variant_suffix(a):
    """'' for the decode of record, else _lt<link thr>_kt<keep thr>[_nng] (no none gate)"""
    return "" if (a.link_thr, a.keep_thr, a.no_none_gate) == (LINK_THR, KEEP_THR, False) else f"_lt{a.link_thr:g}_kt{a.keep_thr:g}" + ("_nng" if a.no_none_gate else "")


def read_dump(a, runs, n_max):
    """the dump of this read-out with everything the scorer needs, checked against the cache -> dict(ck, args, kw, ds, idx, meta, fr, toks, valid, frames)"""
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    nsfx = f"_n{n_max}" if n_max else ""
    d = pickle.load(open(os.path.join(runs, a.run, f"token_{a.split}{every_sfx(a)}{JV.suffix(2.5, a.drop, a.drop_seed)}{nsfx}.pkl"), "rb"))
    meta, fr = d["meta"], d["frames"]
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split, a.every)[: n_max or None]
    assert meta["run"] == a.run and meta.get("every", 1) == a.every and meta["idx"] == idx.tolist() and meta["drop"] == a.drop and meta["drop_seed"] == a.drop_seed and [f["token"] for f in fr] == [ds.tokens[i] for i in idx], "the dump is not the dump of this read-out (run, frames, drop)"
    toks = [ds[int(i)]["tokens_np"] for i in idx]
    assert all(np.array_equal(f["cell"], t["cell"]) for f, t in zip(fr, toks)), "the dump's tokens are not the cache's (a stale dump or cache)"
    valid = [np.asarray(t["valid"], bool) & (JV.keep_nodes(f["token"], ds.K, a.drop, a.drop_seed) if a.drop > 0 else True) for t, f in zip(toks, fr)]     # the tokens the head saw
    assert all(np.array_equal(v & (f["pnone"] < np.float32(KEEP_THR)), f["keep"]) for v, f in zip(valid, fr)), "the valid tokens rebuilt from the cache are not the ones the dump's kept nodes came from"
    every = frames_of(a.split, ds)
    return dict(ck=ck, args=args, kw=kw, ds=ds, idx=idx, meta=meta, fr=fr, toks=toks, valid=valid, frames=[every[int(i)] for i in idx])


def score(a, runs=RUNS, out_dir=None):
    """the cards of one dump (module docstring). --max N reads the dump of `dump --max N`, scores its N frames and writes the cards to `out_dir` (default temp/e365/out) as ..._n<N>.json, with the controls of `ctl --max N` there.
    --link-thr T --keep-thr K [--no-none-gate] re-score the dump with another decode (`decode_variant`; only the `rec` card, its name carries the variant); the defaults are the decode of record, both decodes"""
    n_max, vsfx = a.max, variant_suffix(a)
    out_dir = (out_dir or os.path.join(SCRATCH, "out")) if n_max else None
    sfx = ("" if a.split == "val" else f"_{a.split}") + every_sfx(a) + JV.suffix(2.5, a.drop, a.drop_seed)
    nsfx = f"_n{n_max}" if n_max else ""
    decs = ("rec",) if vsfx else DECODES
    paths = {d: os.path.join(out_dir, f"{a.run}_token_card_{d}{vsfx}{sfx}{nsfx}.json") if n_max else os.path.join(runs, a.run, f"token_card_{d}{vsfx}{sfx}.json") for d in decs}
    held = [p for p in paths.values() if os.path.exists(p)]
    assert not held, f"a stored card is never overwritten: {held}"
    r = read_dump(a, runs, n_max)
    ck, kw, ds, frames = r["ck"], r["kw"], r["ds"], r["frames"]
    ref = stored_plain_b(a.seed) if a.split == "val" and not a.drop else None
    pick = lambda rec: dict(graph=[rec["graph"][int(i)] for i in r["idx"]])  # noqa: E731   # the graph records of plain B / B' on the frames of this read-out (the cache order = the VAL order); the paired interval reads these only
    ref_rec = None if ref is None else pick(ref["rec"])
    b0 = bsec0_close_path(kw["ident"], a.seed, n_max) if ref is not None and kw["ident"] == "e292_a" else None
    b0 = pickle.load(open(b0, "rb")) if b0 and os.path.exists(b0) else None
    params = JK.count_params(JKT.build_head(r["args"]))
    out = {}
    for dec in decs:
        fr = [decode_variant(f, v, a.link_thr, a.keep_thr, not a.no_none_gate) for f, v in zip(r["fr"], r["valid"])] if vsfx else r["fr"]
        res = score_preds([frame_prediction(t, f, dec, a.keep_thr) for t, f in zip(r["toks"], fr)], frames)
        show(f"{a.run} {dec}{vsfx}{sfx}{nsfx}", res["graph"])
        extra = {}
        if vsfx:
            extra["decode_variant"] = dict(link_thr=a.link_thr, keep_thr=a.keep_thr, none_gate=not a.no_none_gate, scales="the head's (dump candidates a, b)" if "a" in r["fr"][0]["cand"] else "the record's links keep their scales, other links 1/3 (a dump of before the variants)")
        if ref is not None:
            extra["paired_vs_plain_b"] = QE.paired(res["rec"], ref_rec, frames)
            extra["plain_b"] = {k: QE.graph_numbers(ref_rec["graph"])[k] for k in KEYS}
            extra["plain_b_dump"] = ref.get("dump")
            if b0 is not None:
                extra["paired_vs_bsec0_close"] = QE.paired(res["rec"], pick(b0["rec"]), frames)
            ctl_hi, ctl_rule = (read_ctl(x, kw["ident"], a.seed, ds.K, out_dir or CTL, n_max) if a.every == 1 else None for x in ("hi", "rule"))        # the controls are scored on the first N / all frames: not on a subsample
            extra["g1_topo_F1"] = None if ctl_hi is None or ctl_rule is None else g1(res["graph"]["topo_F1"], ctl_hi, ctl_rule)
            if extra["g1_topo_F1"] is not None:
                extra["g1_note"] = G1_NOTE
        out[dec] = write_card(paths[dec], f"{a.run}_token_{dec}{vsfx}{sfx}{nsfx}", res, len(frames), a.split, run=a.run, seed=a.seed, decode=dec, drop=a.drop, drop_seed=a.drop_seed, steps=ck.get("steps"), head_params=params,
                              K=ds.K, ident=kw["ident"], decode_rule="rec: p >= 0.3 and p >= p(none), both ends kept (p_none < 0.5); argmax: one link per source and class" if not vsfx else "the variant of `decode_variant`", **extra)
    return out


SWEEP_LINK, SWEEP_KEEP = (0.3, 0.2, 0.1, 0.05), (0.5, 0.6, 0.7)
SWEEP_KEYS = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "dup_frac")


def sweep(a, runs=RUNS, out_dir=None):
    """the decode variants of one dump on its read-out frames: link threshold {0.3, 0.2, 0.1, 0.05} x keep threshold {0.5, 0.6, 0.7} x none gate {on, off} -> one table runs/R/token_sweep_<split>[_drop..].json (--max N: temp/e365/out/..._n<N>.json),
    never overwritten; the cell (0.3, 0.5, on) is the decode of record. For choosing a decode on TRAIN frames (the head was trained on them: the table says what the output could deliver, not what it does on VAL)"""
    n_max = a.max
    sfx = ("" if a.split == "val" else f"_{a.split}") + every_sfx(a) + JV.suffix(2.5, a.drop, a.drop_seed)
    path = os.path.join((out_dir or os.path.join(SCRATCH, "out")) if n_max else os.path.join(runs, a.run), f"{a.run + '_' if n_max else ''}token_sweep{sfx}{f'_n{n_max}' if n_max else ''}.json")
    assert not os.path.exists(path), f"a stored table is never overwritten: {path}"
    r = read_dump(a, runs, n_max)
    rows = []
    for lt in SWEEP_LINK:
        for kt in SWEEP_KEEP:
            for gate in (True, False):
                fr = [decode_variant(f, v, lt, kt, gate) for f, v in zip(r["fr"], r["valid"])]
                res = score_preds([frame_prediction(t, f, "rec", kt) for t, f in zip(r["toks"], fr)], r["frames"], log=lambda *x, **k: None)
                rows.append(dict(link_thr=lt, keep_thr=kt, none_gate=gate, record=(lt, kt, gate) == (LINK_THR, KEEP_THR, True), frames=len(fr), kept_nodes_per_frame=float(np.mean([f["keep"].sum() for f in fr])),
                                 links_per_frame=float(np.mean([len(f["rec"]["src"]) for f in fr])), **{k: res["graph"][k] for k in SWEEP_KEYS}))
                print(f"link {lt:<5g} keep {kt:<4g} gate {'on ' if gate else 'off'}  " + "  ".join(f"{k} {rows[-1][k]:.4f}" for k in SWEEP_KEYS) + f"  links/frame {rows[-1]['links_per_frame']:.1f}", flush=True)
    JV.write_json(path, _jsonable(dict(run=a.run, seed=a.seed, split=a.split, drop=a.drop, frames=len(r["frames"]), rows=rows, K=r["ds"].K, note="decode variants of one dump (`decode_variant`); scales: " + ("the head's" if "a" in r["fr"][0]["cand"] else "the record's links keep theirs, others 1/3"))))
    print(f"written {os.path.relpath(path, ROOT)}")
    return path


def ctl(a, runs=RUNS):
    ds = open_cache("val", a.seed, a.ident, a.K, a.cache)
    arms = a.arms.split(",") if a.arms else list(ARMS)
    assert all(x in ARMS for x in arms), arms
    n = a.max or len(ds)
    idx = np.arange(n)
    frames = QE.val_frames()[:n]
    assert [f.token for f in frames] == list(ds.tokens[:n])
    out_dir = os.path.join(SCRATCH, "out") if a.max else CTL
    paths = {x: os.path.join(out_dir, ctl_name(x, a.ident, a.seed, ds.K, a.max)) for x in arms}
    held = [p for p in paths.values() if os.path.exists(p) and not a.max]
    assert not held, f"a stored card is never overwritten: {held}"
    preds = {x: [] for x in arms if x in ("hi", "hi_y", "rule", "floor")}
    m = floor_model() if "floor" in arms else None
    if "hi" in preds:
        import lane_tok_data as TD
        td = TD.TokData("val", LATTICE, feats=False)
        assert list(td.tokens[:n]) == list(ds.tokens[:n]), "the GT tables are not the frames of the cache, in order"
    if preds:
        k = 0
        for b in batches(ds, idx):
            fl = batch_dump(run_head(m, b, "cpu"), b) if m is not None else None
            for i, t in enumerate(b["tokens_np"]):
                gid = b["label"]["gt_id"][i].numpy()
                if "hi" in preds:
                    preds["hi"].append(control_prediction(t, *hi_frame(t, gid, td.table(k + i))))
                if "hi_y" in preds:
                    preds["hi_y"].append(control_prediction(t, *hi_y_frame(t, gid, b["y"][i].numpy(), b["yscale"][i].numpy())))
                if "rule" in preds:
                    preds["rule"].append(control_prediction(t, *rule_frame(t)))
                if "floor" in preds:
                    preds["floor"].append(frame_prediction(t, fl[i], "rec"))
            k += len(b["tokens_np"])
    for x in arms:
        if x.startswith("bsec0"):
            dump_path = _abs(a.dump or B_DUMPS[a.ident].format(seed=a.seed))
            d = pickle.load(open(dump_path, "rb"))[:n]
            assert [y["row"] for y in d] == [f.row for f in frames], f"{dump_path} is not in the order of the VAL frames"
            preds[x] = [pred_of_dump(y, PLAIN if x == "bsec0_plain" else CLOSE, True) for y in d]
    for x in arms:
        res = score_preds(preds[x], frames)
        show(f"ctl {x} {a.ident} s{a.seed}", res["graph"])
        write_card(paths[x], f"e365_ctl_{x}_{a.ident}_s{a.seed}_K{ds.K}", res, len(frames), "val", arm=x, ident=a.ident, seed=a.seed, K=ds.K, frames_scored=n)
        if x == "bsec0_close" and a.ident == "e292_a":                                       # B' read out with the close-node step, sections 0: the records for the paired interval of `score`
            os.makedirs(SCRATCH, exist_ok=True)
            pickle.dump(dict(res, dump=_file_id(dump_path)), open(bsec0_close_path(a.ident, a.seed, a.max), "wb"))
    return paths


def check_layer1(a):
    """the scorer path fed B's stored predictions (the fresh VAL dump, B's own sections, the E275 close-node step, links at p >= 0.5, no connectors: what `jtest_eval.score_dumps` makes of the `plain` variant) against the stored base numbers of the
    E324 s card -> 0 (PASS: topo F1, F1@1.0, lanes found, dup_frac, vert100_dp within 1e-6) / 1 (FAIL); the other stored numbers are printed, and the sha256 and mtime of the dump used. Only a PASS stores plain B's records (temp/e365/planb_s<S>.pkl),
    which `score` needs"""
    ref = json.load(open(BASE_CARD.format(seed=a.seed)))["variants"]["primary"]["numbers"]["base"]
    res = plain_b(a.seed, a.dump)
    ok, diff = compare_numbers(res["graph"], ref)
    for k in ref:
        print(f"   {k}: {res['graph'][k]:.9f} stored {ref[k]:.9f} diff {res['graph'][k] - ref[k]:+.2e}" + ("" if k in CHECK else "   (not checked)"))
    print(f"dump {res['dump']['path']} sha256 {res['dump']['sha256']} mtime {res['dump']['mtime']}")
    print(f"{'PASS' if ok else 'FAIL'}: layer 1, seed {a.seed}: this scorer on B's stored predictions against {os.path.relpath(BASE_CARD.format(seed=a.seed), ROOT)}: " + ", ".join(f"{k} {v:+.2e}" for k, v in diff.items()))
    if ok:
        os.makedirs(SCRATCH, exist_ok=True)
        pickle.dump(res, open(planb_path(a.seed), "wb"))
    return int(not ok)


def live_tokens(mb, x, K, exact=False):
    """B' (`QP.load_model`) on the input features x (B, 39, 200, 200) -> the head's input batch exactly as `lane_tokcache.build` makes the cache (the same `select_tokens`); exact: tok in float32 (the cache stores float16)"""
    P = mb.pyr(x.float())
    no = mb.node_stage(P)
    t = LC.select_tokens(no, None, K)
    if exact:
        order = torch.argsort(no["e"].float().flatten(1), dim=1, descending=True, stable=True)[:, :K]
        t["tok"] = no["tok"].flatten(2).gather(2, order[:, None, :].expand(-1, no["tok"].shape[1], -1)).transpose(1, 2).float()
    return {k: t[k] for k in JK.INPUTS + ("cell",)}


def compare_outputs(o1, o2, tol):
    """two head outputs on the same frames -> (ok, numbers): the largest difference of p_none (valid tokens) and of the link probabilities (pairs of valid tokens), the kept nodes and the links of the decode of record that differ; PASS
    when both probability differences are within `tol` (a kept node or a link can only differ where its probability is within `tol` of its threshold)"""
    p1, p2, r1, r2 = JK.probs(o1), JK.probs(o2), JK.decode(o1), JK.decode(o2)
    v = o1["valid"] & o2["valid"]
    d_none = float((p1["pt"][..., 0] - p2["pt"][..., 0]).abs()[v].max())
    d_p = float((p1["p"] - p2["p"]).abs()[v[:, :, None] & v[:, None, :]].max())
    num = dict(max_dpnone=d_none, max_dp=d_p, kept_nodes=int(r1["keep"].sum()), kept_nodes_differ=int((r1["keep"] != r2["keep"]).sum()), links=int(r1["link"].sum()), links_differ=int((r1["link"] != r2["link"]).sum()), tol=tol)
    return d_none <= tol and d_p <= tol, num


def align(cells_a, cells_b):
    """(B, K) cell ids of two token sets -> perm (B, K) long: perm[b, j] = the index in set b of the token with the cell of set a's token j, -1 when the cell is not in set b (a near tie in B''s e can swap two tokens' ranks between a GPU and a
    CPU run: the head does not depend on the order of the tokens, so the two sets are compared by cell)"""
    perm = -np.ones(np.shape(cells_a), int)
    for b in range(len(perm)):
        where = {int(c): k for k, c in enumerate(np.asarray(cells_b[b]))}
        perm[b] = [where.get(int(c), -1) for c in np.asarray(cells_a[b])]
    return torch.from_numpy(perm)


def reorder(batch, perm):
    """the head's input batch of set b in the token order of set a (`align`); a token of a that set b lacks is not valid"""
    pm = perm.clamp(min=0)
    ar = torch.arange(len(pm))[:, None]
    out = {k: batch[k][ar, pm] for k in JK.INPUTS}
    out["valid"] = out["valid"] & (perm >= 0)
    return out


def compare_sets(m, cb, cells_c, live, tol, limit=UNMATCHED_LIMIT):
    """the head on the cache batch `cb` (cells_c (B, K): its cells) and on the live batch `live` (inputs + `cell`), the live tokens matched to the cache's by cell (`align`) -> (ok, numbers). PASS when p_none and every link probability of the
    valid tokens agree within `tol` AND the valid tokens that only one side has (a cache token without a valid live match, a live token the cache lacks) are at most `limit` x the cache's valid tokens"""
    perm = align(cells_c, live["cell"].long().numpy())
    live_c = reorder(live, perm)                                                   # the live tokens in the cache's order
    ok, num = compare_outputs(run_head(m, cb, "cpu"), run_head(m, live_c, "cpu"), tol)
    v = cb["valid"] & live_c["valid"]
    fields = {k: float((cb[k].float() - live_c[k].float()).abs()[v].max()) for k in JK.INPUTS if k != "valid"}
    n_c, n_l = int(cb["valid"].sum()), int(live["valid"].sum())
    only = int((cb["valid"] & ~live_c["valid"]).sum()) + n_l - int(live_c["valid"].sum())
    num.update(valid_cache=n_c, valid_live=n_l, valid_cache_not_live=int((cb["valid"] & ~live_c["valid"]).sum()), valid_live_not_cache=n_l - int(live_c["valid"].sum()), unmatched_share=only / max(n_c, 1), unmatched_limit=limit,
               input_max_abs_diff_matched_valid_tokens=fields)
    return bool(ok and only <= limit * n_c), num


def check_parity(a, runs=RUNS):
    """the head on the first `--n` VAL frames, once from the cache tokens (fp16 `tok`) and once from tokens recomputed live from B' on the CPU in float32 (`live_tokens`, compared by cell, `compare_sets`): PASS when p_none and every link
    probability of the valid tokens agree within --tol and the valid tokens that only one side has are at most UNMATCHED_LIMIT (0.5 %) of the cache's valid tokens; the input fields of the matched valid tokens are reported. Writes
    runs/R/token_parity_<UTC>.json (a new file every time) -> 0 / 1"""
    import lane_tok_data as TD
    import qpredict as QP
    m, args = JKT.load_model(a.run, "cpu", runs)
    kw = cache_kw(args)
    ds = open_cache("val", kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    n = a.n
    b_run = _abs(a.b_run or f"runs/{kw['ident']}_s{kw['seed']}")
    mb = QP.load_model(b_run, "cpu")
    td = TD.TokData("val", LATTICE, feats=True)
    assert list(td.tokens[:n]) == list(ds.tokens[:n])
    cb = next(iter(batches(ds, range(n), bs=n)))
    x = torch.stack([td[i]["x"] for i in range(n)])
    with torch.no_grad():
        live = live_tokens(mb, x, ds.K, exact=True)
    ok, num = compare_sets(m, cb, np.stack([t["cell"] for t in cb["tokens_np"]]), live, a.tol)
    num.update(frames=n, run=a.run, b_run=os.path.relpath(b_run, ROOT), ok=ok, note="cache: tok fp16, built on the GPU; live: CPU, tok fp32; tokens are matched by cell, padding tokens are not compared")
    print(json.dumps(num, indent=1))
    print(f"{'PASS' if ok else 'FAIL'}: parity of the head on {n} VAL frames, cache against live B' ({a.run}), tolerance {a.tol}, unmatched limit {UNMATCHED_LIMIT}; written {os.path.relpath(_write_stamped(os.path.join(runs, a.run), 'token_parity', num), ROOT)}")
    return int(not ok)


def head_macs(m, K=384):
    """GMACs of the head for one frame of K tokens (torch FlopCounterMode; MACS_NOTE says what it counts)"""
    from torch.utils.flop_counter import FlopCounterMode
    m = m.cpu().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(JK.synthetic_batch(1, K, seed=0))
    return fc.get_total_flops() / 2e9


def _ancestors():
    """the pids of this process and its parents (a timing started as a run.py job has run.py among them)"""
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def gpu_free(limit_mib=2000):
    """(ok, why): the timing rule of the card: nvidia-smi shows less than `limit_mib` used and no run.py job is active, not counting this process and its ancestors (the run.py job this timing may be)"""
    try:
        used = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().split("\n")[0])
        jobs = [j for j in subprocess.run(["pgrep", "-f", "autoresearch/tools/run.py"], capture_output=True, text=True, timeout=20).stdout.split() if int(j) not in _ancestors()]
    except Exception as e:  # noqa: BLE001
        return False, f"cannot read the GPU: {e}"
    if used >= limit_mib or jobs:
        return False, f"GPU memory used {used:.0f} MiB (limit {limit_mib}), run.py jobs {jobs}"
    return True, f"GPU memory used {used:.0f} MiB, no other run.py job"


def timing(a, runs=RUNS):
    """parameters, GMACs and ms per frame of the head (batch 1, bf16 autocast on the GPU, CUDA events, `--frames` cached VAL frames, `--repeats` passes after one untimed pass) and, with --b-run, of B' alone (pyramid + node head + top-K gather) and of
    the whole path (B' + head in one timed region). K = --K or the K of the cache the run was trained on (at most that). The GPU only when `gpu_free`; --cpu times the code path on the CPU. Writes runs/R/token_timing[_cpu]_<UTC>.json (a new
    file every time) with `quiet` (host load < 3 and an idle GPU before and after) and `usable_for_gcost` (true only for a quiet GPU result)"""
    import jrecur_time as JRT
    import jtiming as JTG
    m, args = JKT.load_model(a.run, "cpu", runs)
    ok, why = (False, "--cpu") if a.cpu else gpu_free()
    if not a.cpu and not ok:
        sys.exit(f"STOP: no GPU timing ({why}); --cpu times the code path on the CPU only")
    dev = "cpu" if a.cpu else "cuda"
    kw = cache_kw(args)
    ds = open_cache("val", kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    K = a.K or ds.K
    assert K <= ds.K, f"--K {K} is above the K of the cache ({ds.K})"
    gmacs = head_macs(m, K)                                                            # on the CPU, before the head moves
    m = m.to(dev)
    load0, gpu0 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    cut = lambda b: {k: v[:, :K] for k, v in b.items() if k in JK.INPUTS}  # noqa: E731
    items = [JKT.to_device(cut(LP.collate([ds[i]])), dev) for i in range(a.frames)]
    ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")  # noqa: E731
    mb = xs = None
    if a.b_run:
        import lane_tok_data as TD
        import qpredict as QP
        mb = QP.load_model(_abs(a.b_run), dev)
        td = TD.TokData("val", LATTICE, feats=True)
        xs = [td[i]["x"][None].to(dev) for i in range(a.frames)]
    ms = dict(head=[], bprime=[], whole=[])
    for rep in range(a.repeats + 1):
        row = dict(head=[], bprime=[], whole=[])
        for i, b in enumerate(items):
            with torch.no_grad(), ac():
                row["head"].append(JRT.timed(lambda: m(b), dev == "cuda")[1])
                if mb is not None:
                    row["bprime"].append(JRT.timed(lambda: live_tokens(mb, xs[i], K), dev == "cuda")[1])
                    row["whole"].append(JRT.timed(lambda: m(live_tokens(mb, xs[i], K)), dev == "cuda")[1])
        if rep:                                                                           # the first pass is untimed
            for k in ms:
                if row[k]:                                                                # B' alone and the whole path only with --b-run
                    ms[k].append(row[k])
    load1, gpu1 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    quiet, line = JTG.quiet_verdict(load0, load1, dict(util_max=0.0) if a.cpu else gpu0, dict(util_max=0.0) if a.cpu else gpu1)
    res = dict(run=a.run, device=dev, K=K, frames=a.frames, repeats=a.repeats, params=JK.count_params(m), gmacs_head=gmacs, host=line, quiet=bool(quiet), usable_for_gcost=bool(quiet and dev == "cuda"), gpu_rule=why, bf16=dev == "cuda",
               median_ms={k: float(np.median(v)) for k, v in ms.items() if v}, p90_ms={k: float(np.percentile(v, 90)) for k, v in ms.items() if v}, macs_note=MACS_NOTE)
    if mb is not None:
        res["params_bprime"] = JK.count_params(mb.pyr) + JK.count_params(mb.nodes)           # the path of the tokens: pyramid + node head (the link, map and assign heads of the checkpoint are not on it)
        res["params_whole_path"] = res["params_bprime"] + res["params"]
    print(json.dumps(res, indent=1))
    print(f"written {os.path.relpath(_write_stamped(os.path.join(runs, a.run), 'token_timing' + ('_cpu' if a.cpu else ''), res), ROOT)}")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ command line
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "score", "sweep"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--split", default="val", choices=("val", "train"))
        s.add_argument("--drop", type=float, default=0.0, help="the degraded read-out: the share of the tokens removed (0.17)")
        s.add_argument("--drop-seed", type=int, default=0)
        s.add_argument("--every", type=int, default=1, help="every N-th frame of the split's sample (TRAIN: 873 evenly spaced frames; every 3 = 291); the file names carry _every<N>; score and sweep read the dump with the same value")
        if name == "dump":
            s.add_argument("--cpu", action="store_true", help="tests only: dump on the CPU")
            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; `score --max N` reads it)")
        else:
            s.add_argument("--max", type=int, default=0, help="score the dump of `dump --max N` (cards to temp/e365/out/)")
        if name == "score":
            s.add_argument("--link-thr", type=float, default=LINK_THR, help="a decode variant: every candidate with p >= T is a link (>= 0.05; the decode of record: 0.3)")
            s.add_argument("--keep-thr", type=float, default=KEEP_THR, help="a decode variant: a node is kept at p_none < K (record: 0.5)")
            s.add_argument("--no-none-gate", action="store_true", help="a decode variant: no p >= p(none) condition on a link")
    s = sub.add_parser("ctl")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--ident", required=True, choices=("e292_a", "e273_p"))
    s.add_argument("--K", type=int, default=None, help="tokens per frame, only when the cache root holds more than one K for the seed")
    s.add_argument("--cache", default=None)
    s.add_argument("--arms", default=None, help="comma list of " + ",".join(ARMS) + " (default all)")
    s.add_argument("--dump", default=None, help="the stored B dump of the bsec0 arms (default per ident, see B_DUMPS)")
    s.add_argument("--max", type=int, default=0, help="the first N frames; cards go to temp/e365/out/")
    s = sub.add_parser("check-layer1")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--dump", default=None, help="the stored dump of plain B (default temp/e343/val/predfull_e273_p_s<seed>.pkl)")
    s = sub.add_parser("check-parity")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--b-run", default=None, help="the folder of B' (default runs/<ident>_s<seed>)")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--tol", type=float, default=0.01)
    s = sub.add_parser("timing")
    s.add_argument("--run", required=True)
    s.add_argument("--b-run", default=None, help="also time B' alone and the whole path")
    s.add_argument("--K", type=int, default=None, help="tokens of the head (default: the K of the cache of the run, 384 in the card)")
    s.add_argument("--frames", type=int, default=24)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(2)
    r = {"dump": dump, "score": score, "sweep": sweep, "ctl": ctl, "check-layer1": check_layer1, "check-parity": check_parity, "timing": timing}[a.cmd](a)
    sys.exit(r if isinstance(r, int) else 0)


if __name__ == "__main__":
    main()
