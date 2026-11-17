"""Read-out, scoring, controls and checks of E365 S1 (the token head on the predicted nodes of the frozen lane head B', `training/grid/jtoken.py`); the cache and the loader are `training/data/lane_tokcache.py` / `lane_ptok_data.py`.

    wsl bash tools/env/_py14.sh jtoken_eval.py dump  --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0] [--every N] [--cpu] [--max N]   # GPU: the head on the cache tokens -> runs/R/token_<split>[_drop..].pkl
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0] [--every N] [--max N]                     # CPU -> runs/R/token_card_<decode>[_train][_drop..].json (rec, argmax; never overwritten)
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S --link-thr T [--link-thr-long T_L] --keep-thr K [--no-none-gate] [--ref-cell S,L,K,on|off]   # CPU: another decode of the same dump -> token_card_rec_lt<T>[_ltl<T_L>]_kt<K>[_nng][_train]..json
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S [a decode variant] --ref-run R2 [--ref-run-cell S,L,K,on|off]   # CPU, E376: + R2's dump of the same read-out at the same decode or that cell, paired R - R2 -> token_card_.._vs_<R2>[_rc<cell>].json
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S --argmax --keep-thr K                                          # CPU, E376: the argmax decode at another keep threshold -> token_card_argmax_kt<K>..json
    wsl bash tools/env/_py14.sh jtoken_eval.py anatomy --run R --seed S [--split val] [--link-thr-long 0.2] [--keep-thr 0.6] [--every N]   # CPU, E376: the delivered long links by category + the long-link and existence AP -> runs/R/token_anatomy_long_ltl<T>_kt<K>..json
    wsl bash tools/env/_py14.sh jtoken_eval.py sweep --run R --seed S --split train [--every N] [--max N] [--grid perclass]                                    # CPU: the grid of decode variants on the TRAIN sample -> runs/R/token_sweep[_perclass]_train.json
    wsl bash tools/env/_py14.sh jtoken_eval.py ctl   --seed S --ident e292_a|e273_p [--arms hi,hi_y,rule,floor,bsec0_plain,bsec0_close] [--max N]   # CPU: the controls -> runs/eval/e365_ctl/<arm>_<ident>_s<S>_K<K>.json
    wsl bash tools/env/_py14.sh jtoken_eval.py ctl   --seed S --ident e292_a --split train --every 3                                    # CPU: c-rule on the TRAIN sample -> runs/eval/e365_ctl/rule_<ident>_s<S>_K<K>_train_every3.json
    wsl bash tools/env/_py14.sh jtoken_eval.py fixture --seed S --split train|val [--every 3] [--lam 12 --mu 10.4]                     # CPU, E369: the untrained --prior rule head against c-rule -> runs/eval/e369_fix/fix_s<S>_<split>.._<UTC>.json, PASS / FAIL
    wsl bash tools/env/_py14.sh jtoken_eval.py fixture --seed S --run R --split train --every 3                                        # CPU, E370 stage A: the run's head against c-rule (per class, in the gate) -> runs/R/token_fixture_.._<UTC>.json
    wsl bash tools/env/_py14.sh jtoken_eval.py graphs --run R --seed S --name <system> --out <file.pkl> [--link-thr T --link-thr-long T_L --keep-thr K --no-none-gate]   # CPU: the delivered VAL graphs in the schema of pass1.pkl + perframe_table.json
    wsl bash tools/env/_py14.sh jtoken_eval.py dump|score|sweep|graphs|fixture --run R --model model_step<N>.pt ...                       # a snapshot of `jtoken_train --snapshots` instead of model.pt: every file name carries _step<N>
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
Anatomy (E376, promoted from temp/_session_scratch_d113/anatomy_long.py): a long link is DELIVERED at p >= T_L with both ends kept (p_none < K; the long class of the D115 decode, no none gate); each is tp (the ends are assigned to the
GT nodes of a GT long link), dup_of_true, wrong_pair or false_end (`link_category`); the long-link AP ranks every long candidate of the dump with both ends kept, the existence AP every valid token (`average_precision`); the mass
concentration (`concentration`) reads the head's own probabilities on its training label G(i) in both classes.
"""
import argparse
import hashlib
import json
import os
import pickle
import re
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

ST = QE.TM.ST                                                               # the shared scene bootstrap (autoresearch/Evaluation/stats.py) that `qgraph_eval.paired` draws with
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


def model_step(name):
    """None for model.pt, N for a snapshot model_step<N>.pt of `jtoken_train --snapshots`; any other name is refused"""
    if name == "model.pt":
        return None
    m = re.fullmatch(r"model_step(\d+)\.pt", name)
    assert m, f"--model {name}: model.pt or a snapshot model_step<N>.pt"
    return int(m.group(1))


def model_sfx(name):
    """'' or _step<N>: the part of every read-out file name that says which weights were read"""
    n = model_step(name)
    return "" if n is None else f"_step{n}"


def dump_path(a, runs, n=0):
    """the dump of a read-out: runs/R/token[_step<N>]_<split>[_every<E>][_drop..][_n<n>].pkl (n: the frames of a `--max` dump)"""
    return os.path.join(runs, a.run, f"token{model_sfx(a.model)}_{a.split}{every_sfx(a)}{JV.suffix(2.5, a.drop, a.drop_seed)}{f'_n{n}' if n else ''}.pkl")


def card_sfx(a):
    """the part of a card or sweep name after its decode: [_step<N>][_<split> unless val][_every<E>][_drop..]"""
    return model_sfx(a.model) + ("" if a.split == "val" else f"_{a.split}") + every_sfx(a) + JV.suffix(2.5, a.drop, a.drop_seed)


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
    """the qformat prediction of one frame from its tokens (`tokens_np`) and its dump frame: kept = f['keep'] (the head's p_none < keep_thr), e = `node_e` (1 - p_none at the record's keep_thr 0.5), the links of the decode (p = 1).
    Asserted on every frame: the nodes are the kept tokens' own (`check_writer`), e = `node_e` of the dump's p_none, the links are exactly the dump's links of the decode between kept nodes (renumbered). The real no-GT guarantee is
    upstream: the head reads only `jtoken.INPUTS` (test_jtoken.py)"""
    lk = dict(f[decode], p=np.ones(len(f[decode]["src"])))
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


def decode_argmax_variant(f, valid, keep_thr=KEEP_THR):
    """the head's argmax decode (`jtoken.decode_argmax`: per source and class the argmax over {none} + its candidates, a link when a candidate wins and both ends are kept) rebuilt from the dump's candidates and none probabilities
    at another keep threshold (p_none < keep_thr, valid tokens) -> the frame with `keep` and `argmax` replaced. A tie goes to the candidate of the lower index, as torch's argmax (the none slot is the last). Exact only when every
    winner is a dump candidate (p >= KEEP_P): `check_argmax` asserts that at the record keep threshold before a read-out"""
    c = f["cand"]
    keep = np.asarray(valid, bool) & (np.asarray(f["pnone"]) < np.float32(keep_thr))
    src, dst, cls = c["src"].astype(int), c["dst"].astype(int), c["long"].astype(int)
    order = np.lexsort((dst, -c["p"], cls, src))                                                 # per source and class its best candidate first (a tie: the lower destination)
    first = np.ones(len(order), bool)
    first[1:] = (np.diff(src[order]) != 0) | (np.diff(cls[order]) != 0)
    w = order[first]
    w = np.sort(w[(c["p"][w] >= np.asarray(f["none_cls"], np.float32)[cls[w], src[w]]) & keep[src[w]] & keep[dst[w]]])       # the dump's order (source, destination)
    return dict(f, keep=keep, argmax={k: c[k][w] for k in LINK_FIELDS})


def check_argmax(frames, valid):
    """before an argmax read-out at another keep threshold: on every frame `decode_argmax_variant` at the record keep threshold equals the dump's argmax decode on all six link fields (a winner below the dump's floor is refused)"""
    for f, v in zip(frames, valid):
        d = decode_argmax_variant(f, v)
        assert np.array_equal(d["keep"], f["keep"]) and all(np.array_equal(d["argmax"][k], f["argmax"][k]) for k in LINK_FIELDS), f"frame {f['token']}: the argmax decode cannot be rebuilt from the dump's candidates"


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
    """c-rule: E357's `lo_near` links on the nodes with B''s e >= 0.5 (valid tokens; `jtoken.rule_links`, the one code path of the rule) -> (keep, e, links)"""
    keep = np.asarray(tokens["valid"], bool) & (np.asarray(tokens["e_p"]) >= JK.RULE_P)
    src, dst = JK.rule_links(tokens["pos"], tokens["dirv"], keep)
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


JUNCTION_KEYS = ("junction_F1", "junction_P", "junction_R")


def junction_numbers(graph_recs, idx=None):
    """junction F1 / P / R of the per-frame graph records (`score_preds` rec['graph']) of the frames idx (None: all; a frame drawn twice counts twice) from the pooled counts lane['junc'] (predicted, GT, matched within 3 m: the scorer's
    junction points of degree >= 3), as `qgraph_eval.graph_numbers`"""
    sel = graph_recs if idx is None else [graph_recs[i] for i in idx]
    return {k: float(v) for k, v in QE.MS.topo_numbers(dict(marb=np.zeros(4), junc=np.sum([c["lane"]["junc"] for c in sel], 0))).items() if k in JUNCTION_KEYS}


def paired_junction(rec_a, rec_b, frames, B=1000, seed=0):
    """a - b of junction F1 / P / R (`junction_numbers`, pooled per draw) with the 95 % interval over the scene draws of `qgraph_eval.paired` (the same stream, B and seed: the same draws as its topo F1) -> key -> a, b, diff, lo, hi, p_a_better"""
    res = ST.paired_draws(lambda idx: junction_numbers(rec_a["graph"], idx), lambda idx: junction_numbers(rec_b["graph"], idx), ST.scene_draws(np.array([f.scene for f in frames]), B, seed), JUNCTION_KEYS)
    return {k: dict(a=v["a"], b=v["b"], diff=v["diff"], lo=v["lo"], hi=v["hi"], p_a_better=v["p_gt0"]) for k, v in res.items()}


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


def ctl_name(arm, ident, seed, K, n=0, split="val", every=1):
    return f"{arm}_{ident}_s{seed}_K{K}{'' if split == 'val' else '_' + split}{f'_every{every}' if every > 1 else ''}{f'_n{n}' if n else ''}.json"


def read_ctl(arm, ident, seed, K, ctl_dir=CTL, n=0):
    """topo F1 of a stored VAL control card (`ctl_name`; a card of an earlier version without the K in its name counts when its recorded K is this one), None when there is none; a TRAIN card is never returned"""
    for name in (ctl_name(arm, ident, seed, K, n), f"{arm}_{ident}_s{seed}{f'_n{n}' if n else ''}.json"):
        p = os.path.join(ctl_dir, name)
        if os.path.exists(p):
            c = json.load(open(p))
            if c["info"].get("K", K) == K and c.get("split", "val") == "val":
                return c["metrics"]["topo_F1"]
    return None


# ------------------------------------------------------------------------------------------------------------------------------ commands
def dump(a, runs=RUNS):
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU; --cpu only for tests"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jtoken_eval dump {a.run}")
    m, args = JKT.load_model(a.run, dev, runs, name=a.model)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split, a.every)[: a.max or None]
    path = dump_path(a, runs, len(idx) if a.max else 0)
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    frames = []
    for b in batches(ds, idx, workers=0 if a.cpu else 2):
        frames += batch_dump(run_head(m, apply_drop(b, a.drop, a.drop_seed), dev), b)
    step = model_step(a.model)
    meta = dict(run=a.run, seed=a.seed, split=a.split, drop=a.drop, drop_seed=a.drop_seed, idx=idx.tolist(), every=a.every, tokens=[ds.tokens[i] for i in idx], K=ds.K, ident=kw["ident"], cache=ds.dir,
                steps=args.get("steps") if step is None else step, model=a.model)
    with open(path, "wb") as fh:
        pickle.dump(dict(meta=meta, frames=frames), fh)
    kept = np.mean([f["keep"].sum() for f in frames])
    print(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames, kept nodes per frame {kept:.1f}, links per frame rec {np.mean([len(f['rec']['src']) for f in frames]):.1f} argmax {np.mean([len(f['argmax']['src']) for f in frames]):.1f}, device {dev}")
    return path


def long_thr(a):
    """the long-class link threshold of a read-out: --link-thr-long, else --link-thr (one threshold for both classes)"""
    return a.link_thr if a.link_thr_long is None else a.link_thr_long


def variant_suffix(a):
    """'' for the decode of record, else _lt<link thr>[_ltl<long link thr>, only when it differs]_kt<keep thr>[_nng] (no none gate)"""
    ltl = long_thr(a)
    if (a.link_thr, ltl, a.keep_thr, a.no_none_gate) == (LINK_THR, LINK_THR, KEEP_THR, False):
        return ""
    return f"_lt{a.link_thr:g}" + ("" if ltl == a.link_thr else f"_ltl{ltl:g}") + f"_kt{a.keep_thr:g}" + ("_nng" if a.no_none_gate else "")


def parse_cell(s):
    """--ref-cell '<short>,<long>,<keep>,<on|off>' -> (link thr, long link thr, keep thr, none gate)"""
    lt, ltl, kt, gate = s.split(",")
    assert gate in ("on", "off"), f"--ref-cell {s}: the none gate is on or off"
    return float(lt), float(ltl), float(kt), gate == "on"


def read_dump(a, runs, n_max):
    """the dump of this read-out (of the weights `a.model`; a dump of before --model counts as model.pt's) with everything the scorer needs, checked against the cache -> dict(ck, args, kw, ds, idx, meta, fr, toks, valid, frames, path)"""
    ck = torch.load(os.path.join(runs, a.run, a.model), map_location="cpu", weights_only=False)
    args = ck["args"]
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    path = dump_path(a, runs, n_max)
    d = pickle.load(open(path, "rb"))
    meta, fr = d["meta"], d["frames"]
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split, a.every)[: n_max or None]
    assert meta["run"] == a.run and meta.get("model", "model.pt") == a.model and meta.get("every", 1) == a.every and meta["idx"] == idx.tolist() and meta["drop"] == a.drop and meta["drop_seed"] == a.drop_seed and [f["token"] for f in fr] == [ds.tokens[i] for i in idx], \
        "the dump is not the dump of this read-out (run, model, frames, drop)"
    toks = [ds[int(i)]["tokens_np"] for i in idx]
    assert all(np.array_equal(f["cell"], t["cell"]) for f, t in zip(fr, toks)), "the dump's tokens are not the cache's (a stale dump or cache)"
    valid = [np.asarray(t["valid"], bool) & (JV.keep_nodes(f["token"], ds.K, a.drop, a.drop_seed) if a.drop > 0 else True) for t, f in zip(toks, fr)]     # the tokens the head saw
    assert all(np.array_equal(v & (f["pnone"] < np.float32(KEEP_THR)), f["keep"]) for v, f in zip(valid, fr)), "the valid tokens rebuilt from the cache are not the ones the dump's kept nodes came from"
    all_frames = frames_of(a.split, ds)
    return dict(ck=ck, args=args, kw=kw, ds=ds, idx=idx, meta=meta, fr=fr, toks=toks, valid=valid, frames=[all_frames[int(i)] for i in idx], path=path)


def same_tokens(r, rr):
    """two read-outs (`read_dump`) hold the same frames in the same order and per frame the same tokens (cells) and the same valid tokens: one cache, so a paired comparison of their heads"""
    assert r["idx"].tolist() == rr["idx"].tolist() and [f["token"] for f in r["fr"]] == [f["token"] for f in rr["fr"]], "the reference run's dump holds other frames"
    assert all(np.array_equal(x["cell"], y["cell"]) and np.array_equal(u, w) for x, y, u, w in zip(r["fr"], rr["fr"], r["valid"], rr["valid"])), "the reference run's dump holds other tokens (another cache)"


def read_ref(a, r, runs, n_max):
    """--ref-run: the dump of the SAME read-out (split, every, drop, drop seed, --max) of another run, of its model.pt (`read_dump`), refused unless its frames and tokens are those of this read-out `r` (`same_tokens`)"""
    rr = read_dump(argparse.Namespace(**dict(vars(a), run=a.ref_run, model="model.pt")), runs, n_max)
    same_tokens(r, rr)
    return rr


def decode_frames(r, a):
    """the dump frames of `r` (`read_dump`) at the decode of the read-out `a`: the dump's own (the record), the variant of --link-thr / --link-thr-long / --keep-thr / --no-none-gate (`decode_variant`) or, with --argmax, the
    argmax decode at --keep-thr (`decode_argmax_variant`)"""
    if a.argmax:
        return [decode_argmax_variant(f, v, a.keep_thr) for f, v in zip(r["fr"], r["valid"])]
    if variant_suffix(a):
        return [decode_variant(f, v, a.link_thr, a.keep_thr, not a.no_none_gate, a.link_thr_long) for f, v in zip(r["fr"], r["valid"])]
    return r["fr"]


def check_decode(r, a):
    """before the dump `r` is read at the decode of the read-out `a`: `check_argmax` (--argmax), `check_record` (a variant); the record needs none"""
    if a.argmax:
        check_argmax(r["fr"], r["valid"])
    elif variant_suffix(a):
        check_record(r["fr"], r["valid"])


def score_dump(r, a, dec, frames, log=print):
    """the standard scorer on the dump `r` at the decode of the read-out `a` (`decode_frames`; `dec` = the links written: rec or argmax)"""
    return score_preds([frame_prediction(t, f, dec, a.keep_thr) for t, f in zip(r["toks"], decode_frames(r, a))], frames, log=log)


def score_cell(r, cell, frames, log=print):
    """the standard scorer on the dump `r` at a threshold cell (link short, link long, keep, none gate) of `decode_variant`"""
    lt, ltl, kt, gate = cell
    return score_preds([frame_prediction(t, decode_variant(f, v, lt, kt, gate, ltl), "rec", kt) for t, f, v in zip(r["toks"], r["fr"], r["valid"])], frames, log=log)


def ref_run_cell(a, dec):
    """the cell at which --ref-run is decoded for the card of `dec`: --ref-run-cell when it is given and is not this card's own cell (an argmax card has none), else None = this read-out's own decode"""
    own = None if a.argmax or dec == "argmax" else (a.link_thr, long_thr(a), a.keep_thr, not a.no_none_gate)
    return a.ref_run_cell if a.ref_run_cell is not None and tuple(a.ref_run_cell) != own else None


def cell_sfx(cell):
    """_rc<short>_<long>_<keep>_<on|off>: the part of a card name that says at which cell the reference run was decoded (only when not at the card's own)"""
    return f"_rc{cell[0]:g}_{cell[1]:g}_{cell[2]:g}_{'on' if cell[3] else 'off'}"


def score(a, runs=RUNS, out_dir=None):
    """the cards of one dump (module docstring). --max N reads the dump of `dump --max N`, scores its N frames and writes the cards to `out_dir` (default temp/e365/out) as ..._n<N>.json, with the controls of `ctl --max N` there.
    --link-thr T [--link-thr-long T_L] --keep-thr K [--no-none-gate] re-score the dump with another decode (`decode_variant`, after `check_record`; only the `rec` card, its name carries the variant), the decode of record scored beside it
    on the same dump (`paired_vs_record`; its numbers against the stored record card are reported, not asserted: a re-dump may differ); the defaults are the decode of record, both decodes. A variant with --ref-cell S,L,K,on|off
    also scores that cell on the same dump: its numbers (`ref_cell`) and the paired scene-bootstrap interval of the variant against it (`paired_vs_ref_cell`: the keys of `qgraph_eval.paired` + junction F1 / P / R, `paired_junction`).
    --ref-run R2 (E376) also scores R2's dump of the same read-out (`read_ref`: its model.pt, the same frames and tokens) at the same decode, or at --ref-run-cell S,L,K,on|off: its numbers and cell (`ref_run`) and the paired interval of
    this run minus R2 (`paired_vs_ref_run`, the keys of `paired_vs_ref_cell`, the same draws); the card's name ends in _vs_<R2>[_rc<cell>, only when R2 is read at another cell].
    --argmax --keep-thr K (E376): the head's argmax decode at another keep threshold (`decode_argmax_variant`, after `check_argmax`) -> the card token_card_argmax_kt<K>..json"""
    n_max = a.max
    assert not a.argmax or (a.link_thr, a.link_thr_long, a.no_none_gate, a.ref_cell) == (LINK_THR, None, False, None), "--argmax takes --keep-thr only"
    assert a.ref_run or a.ref_run_cell is None, "--ref-run-cell is the cell of --ref-run"
    vsfx = f"_kt{a.keep_thr:g}" if a.argmax else variant_suffix(a)
    assert vsfx or a.ref_cell is None, "--ref-cell is compared with a decode variant: give --link-thr / --link-thr-long / --keep-thr / --no-none-gate"
    out_dir = (out_dir or os.path.join(SCRATCH, "out")) if n_max else None
    sfx = card_sfx(a)
    nsfx = f"_n{n_max}" if n_max else ""
    decs = ("argmax",) if a.argmax else ("rec",) if vsfx else DECODES
    rcells = {d: ref_run_cell(a, d) for d in decs}
    rsfx = {d: f"_vs_{a.ref_run}" + (cell_sfx(rcells[d]) if rcells[d] else "") if a.ref_run else "" for d in decs}
    card_path = lambda d, v, x="": os.path.join(out_dir, f"{a.run}_token_card_{d}{v}{sfx}{nsfx}{x}.json") if n_max else os.path.join(runs, a.run, f"token_card_{d}{v}{sfx}{x}.json")  # noqa: E731
    paths = {d: card_path(d, vsfx, rsfx[d]) for d in decs}
    held = [p for p in paths.values() if os.path.exists(p)]
    assert not held, f"a stored card is never overwritten: {held}"
    r = read_dump(a, runs, n_max)
    rr = read_ref(a, r, runs, n_max) if a.ref_run else None
    ck, kw, ds, frames = r["ck"], r["kw"], r["ds"], r["frames"]
    ref = stored_plain_b(a.seed) if a.split == "val" and not a.drop else None

    def pick(rec):                                                                  # the graph records of plain B / B' on the frames of this read-out (the cache order = the VAL order); the paired interval reads these only
        n = len(rec["graph"])
        assert n == len(ds) if not n_max else n > int(r["idx"].max()), f"control records of {n} frames cannot be indexed by this read-out (frames {r['idx'][:5].tolist()} ..., {len(ds)} in the cache)"
        return dict(graph=[rec["graph"][int(i)] for i in r["idx"]])
    ref_rec = None if ref is None else pick(ref["rec"])
    b0 = bsec0_close_path(kw["ident"], a.seed, n_max) if ref is not None and kw["ident"] == "e292_a" and not (n_max and a.every > 1) else None      # `ctl --max N` holds the first N frames, not a subsample
    b0 = pickle.load(open(b0, "rb")) if b0 and os.path.exists(b0) else None
    params = JK.count_params(JKT.build_head(r["args"]))
    check_decode(r, a)
    if rr is not None:
        if not all(rcells.values()):                                                # a card that reads R2 at this read-out's decode
            check_decode(rr, a)
        if any(rcells.values()):
            check_record(rr["fr"], rr["valid"])
    quiet = lambda *x, **k: None  # noqa: E731
    out = {}
    for dec in decs:
        res = score_dump(r, a, dec, frames)
        show(f"{a.run} {dec}{vsfx}{sfx}{nsfx}", res["graph"])
        extra = {}
        if rr is not None:
            rc = rcells[dec]
            res_ref = score_cell(rr, rc, frames, log=quiet) if rc else score_dump(rr, a, dec, frames, log=quiet)
            show(f"  ref {a.ref_run}{cell_sfx(rc) if rc else ''}", res_ref["graph"])
            extra["ref_run"] = dict(run=a.ref_run, model="model.pt", dump=_file_id(rr["path"]), cell=dict(zip(("link_thr", "link_thr_long", "keep_thr", "none_gate"), rc)) if rc else "the decode of this card",
                                    **{k: res_ref["graph"][k] for k in KEYS + ("junction_P", "junction_R")})
            extra["paired_vs_ref_run"] = dict(QE.paired(res["rec"], res_ref["rec"], frames), **paired_junction(res["rec"], res_ref["rec"], frames))
        if a.argmax:
            extra["decode_variant"] = dict(argmax=True, keep_thr=a.keep_thr, e_written=f"min(1 - p_none + {a.keep_thr} - 0.5, 1): only the scorer's e >= 0.5 is meant; a later read-out must not rank nodes by it")
        elif vsfx:
            extra["decode_variant"] = dict(link_thr=a.link_thr, **({} if long_thr(a) == a.link_thr else dict(link_thr_long=long_thr(a))), keep_thr=a.keep_thr, none_gate=not a.no_none_gate, scales="the head's (dump candidates a, b)",
                                           e_written=f"min(1 - p_none + {a.keep_thr} - 0.5, 1): only the scorer's e >= 0.5 is meant; a later read-out must not rank nodes by it")
            rec0 = score_preds([frame_prediction(t, f, "rec") for t, f in zip(r["toks"], r["fr"])], frames, log=lambda *x, **k: None)
            extra["paired_vs_record"] = QE.paired(res["rec"], rec0["rec"], frames)
            extra["record"] = {k: rec0["graph"][k] for k in KEYS}
            stored = card_path("rec", "")
            extra["record_vs_stored_card"] = ({k: rec0["graph"][k] - json.load(open(stored))["metrics"][k] for k in KEYS} if os.path.exists(stored) else None)
            extra["record_card"] = os.path.relpath(stored, ROOT) if os.path.exists(stored) else f"no stored record card {os.path.relpath(stored, ROOT)}"
            if a.ref_cell is not None:
                lt0, ltl0, kt0, g0 = a.ref_cell
                res0 = score_cell(r, a.ref_cell, frames, log=quiet)
                extra["ref_cell"] = dict(link_thr=lt0, link_thr_long=ltl0, keep_thr=kt0, none_gate=g0, **{k: res0["graph"][k] for k in KEYS + ("junction_P", "junction_R")})
                extra["paired_vs_ref_cell"] = dict(QE.paired(res["rec"], res0["rec"], frames), **paired_junction(res["rec"], res0["rec"], frames))
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
        rule = "the argmax decode at another keep threshold (`decode_argmax_variant`)" if a.argmax else "the variant of `decode_variant`" if vsfx else "rec: p >= 0.3 and p >= p(none), both ends kept (p_none < 0.5); argmax: one link per source and class"
        out[dec] = write_card(paths[dec], f"{a.run}_token_{dec}{vsfx}{sfx}{nsfx}{rsfx[dec]}", res, len(frames), a.split, run=a.run, model=a.model, seed=a.seed, decode=dec, drop=a.drop, drop_seed=a.drop_seed, steps=ck.get("steps"), head_params=params,
                              K=ds.K, ident=kw["ident"], decode_rule=rule, **extra)
    return out


SWEEP_LINK, SWEEP_KEEP = (0.3, 0.2, 0.1, 0.05), (0.5, 0.6, 0.7)
SWEEP_KEYS = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "dup_frac")
PERCLASS_SHORT, PERCLASS_LONG = (0.3, 0.2, 0.1), (0.3, 0.2, 0.1, 0.05)                  # E375: the link thresholds of the short and of the long class
SWEEP_GRIDS = dict(default=[(lt, lt, kt, g) for lt in SWEEP_LINK for kt in SWEEP_KEEP for g in (True, False)],                          # cells (link thr, long link thr, keep thr, none gate)
                   perclass=[(lt, ltl, kt, False) for lt in PERCLASS_SHORT for ltl in PERCLASS_LONG for kt in SWEEP_KEEP])


def sweep(a, runs=RUNS, out_dir=None):
    """the decode variants of one dump on its read-out frames -> one table runs/R/token_sweep[_perclass][_step<N>]_<split>[_drop..].json (--max N: temp/e365/out/..._n<N>.json), never overwritten. --grid default: link threshold
    {0.3, 0.2, 0.1, 0.05} x keep threshold {0.5, 0.6, 0.7} x none gate {on, off}; the cell (0.3, 0.5, on) is the decode of record. --grid perclass (E375): link threshold short {0.3, 0.2, 0.1} x long {0.3, 0.2, 0.1, 0.05} x keep
    threshold {0.5, 0.6, 0.7}, none gate off; its rows also carry link_thr_long, junction F1 / P / R and the predicted junction points per frame (the scorer's, degree >= 3), the delivered long links per frame and the scorer's long-link
    precision / recall (node_link_long_P / _R: GT long links between matched nodes). For choosing a decode on TRAIN frames (the head was trained on them: the table says what the output could deliver, not what it does on VAL)"""
    n_max, pc = a.max, a.grid == "perclass"
    sfx = ("_perclass" if pc else "") + card_sfx(a)
    path = os.path.join((out_dir or os.path.join(SCRATCH, "out")) if n_max else os.path.join(runs, a.run), f"{a.run + '_' if n_max else ''}token_sweep{sfx}{f'_n{n_max}' if n_max else ''}.json")
    assert not os.path.exists(path), f"a stored table is never overwritten: {path}"
    r = read_dump(a, runs, n_max)
    check_record(r["fr"], r["valid"])
    rows = []
    for lt, ltl, kt, gate in SWEEP_GRIDS[a.grid]:
        fr = [decode_variant(f, v, lt, kt, gate, ltl) for f, v in zip(r["fr"], r["valid"])]
        res = score_preds([frame_prediction(t, f, "rec", kt) for t, f in zip(r["toks"], fr)], r["frames"], log=lambda *x, **k: None)
        row = dict(link_thr=lt, **(dict(link_thr_long=ltl) if pc else {}), keep_thr=kt, none_gate=gate, record=(lt, ltl, kt, gate) == (LINK_THR, LINK_THR, KEEP_THR, True), frames=len(fr),
                   kept_nodes_per_frame=float(np.mean([f["keep"].sum() for f in fr])), links_per_frame=float(np.mean([len(f["rec"]["src"]) for f in fr])), **{k: res["graph"][k] for k in SWEEP_KEYS})
        if pc:
            row.update({k: res["graph"][k] for k in JUNCTION_KEYS}, junction_points_per_frame=float(np.mean([c["lane"]["junc"][0] for c in res["rec"]["graph"]])), long_links_per_frame=float(np.mean([f["rec"]["long"].sum() for f in fr])),
                       **{f"node_link_long_{k}": res["nodes"].get(f"link_long_{k}") for k in ("P", "R")})
        rows.append(row)
        print(f"link {lt:<5g} " + (f"long {ltl:<5g} " if pc else "") + f"keep {kt:<4g} gate {'on ' if gate else 'off'}  " + "  ".join(f"{k} {row[k]:.4f}" for k in SWEEP_KEYS + (JUNCTION_KEYS if pc else ()))
              + f"  links/frame {row['links_per_frame']:.1f}", flush=True)
    JV.write_json(path, _jsonable(dict(run=a.run, model=a.model, seed=a.seed, split=a.split, every=a.every, max=a.max, drop=a.drop, drop_seed=a.drop_seed, steps=r["ck"].get("steps"), dump=_file_id(r["path"]), frames=len(r["frames"]), rows=rows, K=r["ds"].K,
                                       **(dict(grid="perclass") if pc else {}), note="decode variants of one dump (`decode_variant`, the head's scales); e written = min(1 - p_none + keep_thr - 0.5, 1): a later read-out must not rank nodes by it")))
    print(f"written {os.path.relpath(path, ROOT)}")
    return path


# ------------------------------------------------------------------------------------------------------------------------------ the anatomy of the delivered long links (E376)
LONG_CATS = ("tp", "dup_of_true", "wrong_pair", "false_end")
ENTRY, EXIT = LC.TYPES.index("entry"), LC.TYPES.index("exit")              # columns of the dump's ptype (the head's type softmax in the order of `lane_tokcache.TYPES`)


def link_category(s, d, gt_id, near, gt_long):
    """the category of the link s -> d between two tokens, an index of LONG_CATS: tp = the ends are assigned (`gt_id`) to the GT nodes a, b of a GT long link a -> b (`gt_long`: a set of pairs of table rows); else false_end = an end
    is near no GT node (`near` (K, M), the cache's label rule `lane_tokcache.near_matrix`: 1.5 m and the cosine of the headings >= 0.5); dup_of_true = a GT node near s and one near d form a GT long link (an end is a duplicate token
    of a true end); wrong_pair = both ends near GT nodes, no pair of them a GT long link"""
    a, b = int(gt_id[s]), int(gt_id[d])
    if a >= 0 and b >= 0 and (a, b) in gt_long:
        return 0
    ns, nd = np.flatnonzero(near[s]), np.flatnonzero(near[d])
    if len(ns) == 0 or len(nd) == 0:
        return 3
    return 1 if any((int(x), int(y)) in gt_long for x in ns for y in nd) else 2


def average_precision(score, label):
    """the area under the precision-recall step curve of the ranking by `score` (descending) = the mean over the positives of the precision at their rank (recall over the positives of the list); ties broken pessimistically (in a
    group of equal scores the negatives rank first); None without a positive"""
    label = np.asarray(label, bool)
    if not label.any():
        return None
    hit = label[np.lexsort((label, -np.asarray(score, float)))]
    return float((np.cumsum(hit) / np.arange(1, len(hit) + 1))[hit].mean())


def gt_labels(tokens, valid, table, gt_id, uout, uin):
    """the GT side of one frame: gt_id (K,) of the cache, near (K, M) = the valid tokens near the GT nodes of the frame's table (`lane_tokcache.near_matrix`), gt_long = the GT long links (pairs of table rows), b_out / b_in =
    B''s long-link evidence sigmoid(uout) / sigmoid(uin) of the tokens"""
    sig = lambda u: 1.0 / (1.0 + np.exp(-np.asarray(u, float)))  # noqa: E731
    return dict(gt_id=np.asarray(gt_id, int), near=LC.near_matrix(tokens["pos"], tokens["dirv"], valid, table["pos"], table["dirv"])[0], gt_long={(int(s), int(d)) for s, d, lg in zip(table["src"], table["dst"], table["long"]) if lg},
                b_out=sig(uout), b_in=sig(uin))


def anatomy_frame(f, valid, lab, keep_thr):
    """one dump frame and its GT side (`gt_labels`) -> the long candidates with both ends kept (`decode_variant` at the dump's floor KEEP_P, keep_thr, no none gate): p, cat (`link_category`), typed = p_entry(src) * p_exit(dst)
    of the head's type softmax, b_long = B''s sigmoid(uout[src]) * sigmoid(uin[dst]); and the valid tokens: ex_score = 1 - p_none, ex_label = assigned to a GT node (gt_id >= 0)"""
    c = decode_variant(f, valid, KEEP_P, keep_thr, False, KEEP_P)["rec"]
    lg = np.asarray(c["long"], bool)
    s, d = c["src"][lg].astype(int), c["dst"][lg].astype(int)
    pt, v = np.asarray(f["ptype"], float), np.asarray(valid, bool)
    return dict(p=c["p"][lg], cat=np.array([link_category(i, j, lab["gt_id"], lab["near"], lab["gt_long"]) for i, j in zip(s, d)], int), typed=pt[s, ENTRY] * pt[d, EXIT],
                b_long=lab["b_out"][s] * lab["b_in"][d], ex_score=1.0 - np.asarray(f["pnone"], float)[v], ex_label=lab["gt_id"][v] >= 0)


def anatomy_numbers(parts, n_gt_long, thr_long):
    """the frames' `anatomy_frame` parts pooled: the delivered long links (p >= thr_long) per category (n, share, median p / typed / b_long), tp over the GT long links; the long-link AP over every kept long candidate (label tp) with its
    recall ceiling (the share of the GT long links that are such a candidate: the assignment is one to one, so a tp candidate is one GT link) and the existence AP over the valid tokens"""
    cat = lambda k: np.concatenate([x[k] for x in parts])  # noqa: E731
    p, c, typed, bl = cat("p"), cat("cat"), cat("typed"), cat("b_long")
    dl = p >= np.float32(thr_long)
    n = int(dl.sum())
    med = lambda x: float(np.median(x)) if len(x) else None  # noqa: E731
    cats = {k: dict(n=int((dl & (c == i)).sum()), share=float((dl & (c == i)).sum() / max(n, 1)), median_p=med(p[dl & (c == i)]), median_typed=med(typed[dl & (c == i)]), median_b_long=med(bl[dl & (c == i)]))
            for i, k in enumerate(LONG_CATS)}
    n_tp = int((c == 0).sum())
    ex = cat("ex_label")
    return dict(gt_long_links=n_gt_long, delivered=n, categories=cats, tp_over_gt_long=cats["tp"]["n"] / max(n_gt_long, 1), ap_long=average_precision(p, c == 0), long_candidates=len(p), long_candidates_tp=n_tp,
                recall_ceiling=n_tp / max(n_gt_long, 1), ap_existence=average_precision(cat("ex_score"), ex), valid_tokens=len(ex), assigned_tokens=int(ex.sum()))


def group_concentration(p, G, cand, thr=LINK_THR):
    """one class of one frame: p (K, K) = p^c(j|i), G (K, K) bool = the in-class GT successor group G(i) of every source (the training label of `jtoken.jtoken_loss`: y and the class gate), cand (K, K) bool = the class gate ->
    for every source with a non-empty G(i): share = the largest member's share of the group's mass (max over G(i) of p / sum over G(i) of p), above = its in-class candidates with p >= thr"""
    rows = G.any(1)
    top = np.where(G, p, 0.0).max(1)[rows]
    return top / np.where(G, p, 0.0).sum(1)[rows], ((p >= thr) & cand).sum(1)[rows]


def concentration(m, ds, idx, fr, valid):
    """the mass concentration of the head `m` on the cache frames idx, per class (a CPU forward, batch 1: the dump does not hold the probabilities below KEEP_P; its p_none must agree with the dump's `fr` to 1e-3 on the `valid`
    tokens) -> class -> `concentration_numbers` over the sources with a non-empty G(i) (`group_concentration`)"""
    got = {c: ([], []) for c in ("short", "long")}
    for b, f, v in zip(batches(ds, idx, bs=1), fr, valid):
        out = run_head(m, b, "cpu")
        K = out["cand"].shape[-1]
        assert np.abs(JK.probs(out)["pt"][0, :, 0].numpy() - f["pnone"])[v].max(initial=0.0) < 1e-3, f"frame {f['token']}: the head's p_none is not the dump's"
        G, cand = ((b["y"] > 0)[:, None] & out["cand"])[0].numpy(), out["cand"][0].numpy()          # G: the training label of `jtoken.jtoken_loss` (y and the class gate)
        p = out["logp"][0, :, :, :K].double().exp().numpy()
        for c, (sh, ab) in enumerate(got.values()):
            s, n = group_concentration(p[c], G[c], cand[c])
            sh.append(s)
            ab.append(n)
    return {c: concentration_numbers(np.concatenate(sh), np.concatenate(ab)) for c, (sh, ab) in got.items()}


def concentration_numbers(s, n):
    """the pooled shares s and counts n of `group_concentration` -> sources, share median / mean, above_mean, above_2plus (the share of sources with 2 or more), above_mean_given_one (over the sources with one or more); None where empty"""
    v = lambda x, fn: float(fn(x)) if len(x) else None  # noqa: E731
    return dict(sources=len(s), share_median=v(s, np.median), share_mean=v(s, np.mean), above_mean=v(n, np.mean), above_2plus=v(n >= 2, np.mean), above_mean_given_one=v(n[n >= 1], np.mean))


def anatomy(a, runs=RUNS, out_dir=None):
    """E376: the anatomy of the delivered long links of the run's dump at --link-thr-long / --keep-thr (module docstring; `anatomy_numbers`; the GT side: the cache's gt_id / uout / uin, read once, and the frame's table of
    `lane_tok_data.TokData`) and the mass concentration of the head in both classes (`concentration`, a CPU forward of --model on the same frames) -> runs/R/token_anatomy_long_ltl<T>_kt<K>[_step<N>][_<split>][_every<N>].json,
    never overwritten (out_dir or --max: <out_dir or temp/e365/out>/R_token_anatomy_..[_n<N>].json), and a table"""
    import lane_tok_data as TD
    assert a.link_thr_long >= KEEP_P, f"the dump holds the candidates with p >= {KEEP_P}"
    a.drop, a.drop_seed = 0.0, 0
    name = f"token_anatomy_long_ltl{a.link_thr_long:g}_kt{a.keep_thr:g}{card_sfx(a)}{f'_n{a.max}' if a.max else ''}.json"
    path = os.path.join(out_dir or os.path.join(SCRATCH, "out"), f"{a.run}_{name}") if out_dir or a.max else os.path.join(runs, a.run, name)
    assert not os.path.exists(path), f"a stored anatomy is never overwritten: {path}"
    r = read_dump(a, runs, a.max)
    ds, idx = r["ds"], r["idx"]
    td = TD.TokData(a.split, LATTICE, feats=False)
    assert [td.tokens[int(i)] for i in idx] == [ds.tokens[int(i)] for i in idx], "the GT tables are not the frames of the cache"
    gid, uout, uin = (np.asarray(ds.c.f[k][idx]) for k in ("gt_id", "uout", "uin"))
    parts, n_gt = [], 0
    for k, (i, f, t, v) in enumerate(zip(idx, r["fr"], r["toks"], r["valid"])):
        lab = gt_labels(t, v, td.table(int(i)), gid[k], uout[k], uin[k])
        parts.append(anatomy_frame(f, v, lab, a.keep_thr))
        n_gt += len(lab["gt_long"])
    num = anatomy_numbers(parts, n_gt, a.link_thr_long)
    num["concentration"] = concentration(JKT.load_model(a.run, "cpu", runs, name=a.model)[0], ds, idx, r["fr"], r["valid"])
    JV.write_json(path, _jsonable(dict(run=a.run, model=a.model, seed=a.seed, split=a.split, every=a.every, max=a.max, frames=len(idx), K=ds.K, link_thr_long=a.link_thr_long, keep_thr=a.keep_thr, dump=_file_id(r["path"]),
                                       near=dict(r=LC.R_NEAR, cos=LC.COS_NEAR), **num, note="delivered = long candidate with p >= link_thr_long and both ends kept (p_none < keep_thr), no none gate; categories of `link_category`; "
                                       "AP = the mean precision at the positives, ties broken pessimistically, recall over the positives of the list (x recall_ceiling = recall over the GT long links); concentration: a CPU forward of the "
                                       f"head, per class the sources with a non-empty G(i) (y and the class gate, the training label), share = max / sum of p over G(i), above = the in-class candidates with p >= {LINK_THR}")))
    fmt = lambda x, s: "-" if x is None else format(x, s)  # noqa: E731
    print(f"{a.run} {a.split}, {len(idx)} frames: GT long links {n_gt}, delivered long links (p >= {a.link_thr_long:g}, both ends p_none < {a.keep_thr:g}) {num['delivered']}")
    for k, x in num["categories"].items():
        print(f"  {k:<12} {x['n']:7d}  share {x['share']:.3f}  median p {fmt(x['median_p'], '.2f')}  typed {fmt(x['median_typed'], '.3f')}  B' uout*uin {fmt(x['median_b_long'], '.2f')}")
    print(f"  tp / GT long links {num['tp_over_gt_long']:.3f}; AP long links {fmt(num['ap_long'], '.4f')} over {num['long_candidates']} kept long candidates (recall ceiling {num['recall_ceiling']:.3f}); "
          f"AP existence {fmt(num['ap_existence'], '.4f')} over {num['valid_tokens']} valid tokens ({num['assigned_tokens']} assigned)")
    for c, x in num["concentration"].items():
        print(f"  {c:<5} {x['sources']} sources with a GT successor: largest member's share of G(i)'s mass median {fmt(x['share_median'], '.3f')} mean {fmt(x['share_mean'], '.3f')}; candidates >= {LINK_THR} per source "
              f"{fmt(x['above_mean'], '.3f')}, 2 or more {fmt(x['above_2plus'], '.3f')}, per source with one or more {fmt(x['above_mean_given_one'], '.3f')}")
    print(f"written {os.path.relpath(path, ROOT)}")
    return path


def ctl(a, runs=RUNS):
    """the controls of the module docstring on VAL (the first --max frames); `--split train [--every N]`: the arm rule only, on the TRAIN sample (cards ..._train[_every<N>].json)"""
    ds = open_cache(a.split, a.seed, a.ident, a.K, a.cache)
    arms = a.arms.split(",") if a.arms else (list(ARMS) if a.split == "val" else ["rule"])
    assert all(x in ARMS for x in arms), arms
    assert (a.split == "val" and a.every == 1) or arms == ["rule"], "a TRAIN sample or a subsample: the arm rule only (the other arms read the first frames of VAL)"
    idx = sample(len(ds), a.split, a.every)[: a.max or None]
    n = len(idx)
    every_frame = frames_of(a.split, ds)
    frames = [every_frame[int(i)] for i in idx]
    out_dir = os.path.join(SCRATCH, "out") if a.max else CTL
    paths = {x: os.path.join(out_dir, ctl_name(x, a.ident, a.seed, ds.K, a.max, a.split, a.every)) for x in arms}
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
        show(f"ctl {x} {a.ident} s{a.seed} {a.split}", res["graph"])
        write_card(paths[x], f"e365_ctl_{x}_{a.ident}_s{a.seed}_K{ds.K}" + ("" if a.split == "val" else f"_{a.split}") + (f"_every{a.every}" if a.every > 1 else ""), res, len(frames), a.split, arm=x, ident=a.ident,
                   seed=a.seed, K=ds.K, frames_scored=n, every=a.every)
        if x == "bsec0_close" and a.ident == "e292_a":                                       # B' read out with the close-node step, sections 0: the records for the paired interval of `score`
            os.makedirs(SCRATCH, exist_ok=True)
            pickle.dump(dict(res, dump=_file_id(dump_path)), open(bsec0_close_path(a.ident, a.seed, a.max), "wb"))
    return paths


# ------------------------------------------------------------------------------------------------------------------------------ fixtures: a head's delivered graph against c-rule on the same frames
FIX_DIR = os.path.join(JV.CARDS, "e369_fix")
FIX_E369 = dict(present=0.999, extra=0.001, kept_diff=0, topo_val=0.001)    # E369: the untrained prior head = c-rule (links present / extra as shares of c-rule's links, kept sets identical, topo F1 on VAL)
FIX_E370 = dict(short=0.95, long=0.80, extra=0.05, kept_agree=0.99)        # E370 stage A: against c-rule restricted to the head's gate, per class; extra as a share of the restricted links; kept agreement over the valid tokens


def fixture_counts(dec_frames, gate, toks):
    """the link and node counts of a head's delivered frames (`batch_dump`) against c-rule (`rule_frame`) on the same tokens; gate (B, K, K) bool = the head's candidate gate -> (counts dict, c-rule's links, c-rule's in-gate links)"""
    c = {k: 0 for k in ("rule", "rule_short", "rule_long", "rule_in_short", "rule_in_long", "out_gate", "head", "present", "present_in_short", "present_in_long", "extra", "kept_diff", "valid")}
    rules, rules_in = [], []
    for f, g, t in zip(dec_frames, gate, toks):
        keep, e, lk = rule_frame(t)
        src, dst, lg = np.asarray(lk["src"], int), np.asarray(lk["dst"], int), np.asarray(lk["long"], bool)
        inside = g[src, dst]
        R, H = set(zip(src.tolist(), dst.tolist())), set(zip(f["rec"]["src"].astype(int).tolist(), f["rec"]["dst"].astype(int).tolist()))
        for cls, m in (("short", ~lg), ("long", lg)):
            Rc = set(zip(src[m & inside].tolist(), dst[m & inside].tolist()))
            c[f"rule_{cls}"] += int(m.sum())
            c[f"rule_in_{cls}"] += len(Rc)
            c[f"present_in_{cls}"] += len(H & Rc)
        c["rule"] += len(R)
        c["out_gate"] += int((~inside).sum())
        c["head"] += len(H)
        c["present"] += len(H & R)
        c["extra"] += len(H - R)
        c["kept_diff"] += int((np.asarray(f["keep"], bool) != keep).sum())
        c["valid"] += int(np.asarray(t["valid"], bool).sum())
        rules.append((keep, e, lk))
        rules_in.append((keep, e, {k: np.asarray(v)[inside] for k, v in lk.items()}))
    return c, rules, rules_in


def fixture_verdict(c, topo, split, mode):
    """-> (ok, the numbers of the rule): mode 'e369' (FIX_E369; topo F1 within 0.001 of c-rule on VAL only) or 'e370a' (FIX_E370)"""
    sh = lambda a, b: a / max(b, 1)  # noqa: E731
    if mode == "e369":
        num = dict(present=sh(c["present"], c["rule"]), extra=sh(c["extra"], c["rule"]), kept_diff=c["kept_diff"], topo_diff=topo["head"] - topo["rule"])
        ok = num["present"] >= FIX_E369["present"] and num["extra"] <= FIX_E369["extra"] and num["kept_diff"] <= FIX_E369["kept_diff"] and (split != "val" or abs(num["topo_diff"]) <= FIX_E369["topo_val"])
    else:
        num = dict(short=sh(c["present_in_short"], c["rule_in_short"]), long=sh(c["present_in_long"], c["rule_in_long"]), extra=sh(c["extra"], c["rule_in_short"] + c["rule_in_long"]), kept_agree=1.0 - sh(c["kept_diff"], c["valid"]),
                   topo_diff=topo["head"] - topo["rule"], topo_diff_restricted=topo["head"] - topo["rule_in"])
        ok = num["short"] >= FIX_E370["short"] and num["long"] >= FIX_E370["long"] and num["extra"] <= FIX_E370["extra"] and num["kept_agree"] >= FIX_E370["kept_agree"]
    return bool(ok), num


def fixture(a, runs=RUNS, out_dir=None):
    """E369 (no --run): the untrained `--prior rule` head (built after `trainer.seed_all(seed)` as the trainer builds it, step 0, lam / mu as given) or E370 stage A (--run R: the run's head) on the frames of the read-out
    (`sample`, --every, --max), against c-rule on the same frames: links present / extra, kept-node differences, the rule links outside the head's gate, the GT loss of the head on these frames (`jtoken_loss`, eval mode), topo F1 of the
    head, of c-rule and of c-rule restricted to the gate (the standard scorer) -> PASS / FAIL by FIX_E369 / FIX_E370; a stamped json in runs/eval/e369_fix/ (no --run) or runs/R/ (token_fixture[_step<N>]_..., --model), --max: temp/e365/out/. 0 / 1"""
    mode = "e370a" if a.run else "e369"
    assert a.run or a.model == "model.pt", "--model reads a snapshot of a run: give --run"
    if a.run:
        m, args = JKT.load_model(a.run, "cpu", runs, name=a.model)
        assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
        kw = cache_kw(args)
    else:
        JKT.T.seed_all(a.seed)
        m, args = JKT.build_head(dict(prior="rule")).eval(), dict(prior="rule")
        m.lam.fill_(a.lam)
        m.mu.fill_(a.mu)
        kw = dict(seed=a.seed, ident=a.ident, K=None)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split, a.every)[: a.max or None]
    all_frames = frames_of(a.split, ds)
    frames = [all_frames[int(i)] for i in idx]
    c, heads, rules, rules_in, toks, losses = None, [], [], [], [], []
    for b in batches(ds, idx):
        out = run_head(m, b, "cpu")
        losses.append((float(JK.jtoken_loss(out, b)[0]), len(b["token"])))
        fr = batch_dump(out, b)
        ci, r, ri = fixture_counts(fr, out["cand"].any(1).numpy(), b["tokens_np"])
        c = ci if c is None else {k: c[k] + v for k, v in ci.items()}
        heads += [frame_prediction(t, f, "rec") for t, f in zip(b["tokens_np"], fr)]
        rules += [control_prediction(t, *x) for t, x in zip(b["tokens_np"], r)]
        rules_in += [control_prediction(t, *x) for t, x in zip(b["tokens_np"], ri)]
        toks += b["token"]
    quiet = lambda *x, **k: None  # noqa: E731
    res = {name: score_preds(p, frames, log=quiet)["graph"] for name, p in (("head", heads), ("rule", rules), ("rule_in", rules_in))}
    topo = {k: v["topo_F1"] for k, v in res.items()}
    ok, num = fixture_verdict(c, topo, a.split, mode)
    stored = read_ctl("rule", kw["ident"], a.seed, ds.K) if a.split == "val" and a.every == 1 and not a.max else None
    doc = dict(mode=mode, ok=ok, rule=FIX_E369 if mode == "e369" else FIX_E370, numbers=num, counts=c, topo_F1=topo, graph={k: {q: v[q] for q in KEYS} for k, v in res.items()},
               gt_loss=sum(l_ * n_ for l_, n_ in losses) / max(len(toks), 1), split=a.split, every=a.every, max=a.max, frames=len(frames), K=ds.K, ident=kw["ident"], seed=a.seed, cache=ds.dir,
               run=a.run, model=a.model if a.run else None, steps=int(m.step), lam=float(m.lam) if hasattr(m, "lam") else None, mu=float(m.mu) if hasattr(m, "mu") else None, rule_card_topo_F1=stored, rule_vs_card=None if stored is None else topo["rule"] - stored,
               note="links of the decode of record against c-rule (`rule_frame`) per frame; gt_loss = jtoken_loss with the GT targets of the cache on these frames, eval mode, the mean over frames of the batch means")
    tag = f"{a.split}{f'_every{a.every}' if a.every > 1 else ''}{f'_n{a.max}' if a.max else ''}"
    folder = (out_dir or os.path.join(SCRATCH, "out")) if a.max else (os.path.join(runs, a.run) if a.run else FIX_DIR)
    os.makedirs(folder, exist_ok=True)
    path = _write_stamped(folder, (f"token_fixture{model_sfx(a.model)}_{tag}" if a.run else f"fix_s{a.seed}_{tag}"), doc)
    print(json.dumps(dict(numbers=num, counts=c, topo_F1=topo, gt_loss=doc["gt_loss"], rule_vs_card=doc["rule_vs_card"]), indent=1))
    print(f"{'PASS' if ok else 'FAIL'}: fixture {mode}, seed {a.seed}, {a.split} ({len(frames)} frames)" + (f", run {a.run}" if a.run else f", lam {doc['lam']:g} mu {doc['mu']:g}") + f"; written {os.path.relpath(path, ROOT)}")
    return int(not ok)


# ------------------------------------------------------------------------------------------------------------------------------ delivered graphs for the qualitative report (the schema of temp/qual_report_1115/pass1.pkl)
SYS_KEYS = ("lines", "man_ok", "man_n", "man_real", "junc_hit", "junc_near")


def frame_record(f):
    """the GT side of a frame as `temp/qual_report_1115/pass1_q1115.py` wrote it (re-implemented here, that script is scratch): k, token, row, scene, gt = the lane GT pieces with more than one point (pts float32, tier int8,
    vis bool, kinds), n_man / man_pts = the maneuvers of `jmaneuver.maneuvers(lane GT, qends.cluster)`, sys = {}"""
    import jmaneuver as JM
    import qends as QENDS
    g = f.lane()
    mans = JM.maneuvers(g, QENDS.cluster)
    return dict(k=f.k, token=f.token, row=f.row, scene=f.scene,
                gt=[dict(pts=np.asarray(p["pts"], np.float32), tier=np.asarray(p.get("tier", np.zeros(len(p["pts"]))), np.int8), vis=np.asarray(p.get("vis", np.ones(len(p["pts"]))), bool),
                         kinds=[(s["kind"], int(s["i0"]), int(s["i1"])) for s in p["segments"]]) for p in g["pieces"] if len(p["pts"]) > 1],
                n_man=len(mans), man_pts=[np.asarray(x[2], np.float32) for x in mans], sys={}), mans


def sys_record(raw, mans, f):
    """one system's delivered graph of a frame as pass1 wrote it: lines (float32 polylines), man_ok / man_n / man_real / junc_hit / junc_near of `jmaneuver.frame_counts(raw, maneuvers, lane GT tree)`"""
    import jmaneuver as JM
    n_ok, n, hit, near, real = JM.frame_counts(raw, mans, f.ref().tree) if mans else (0, 0, 0, 0, [])
    return dict(lines=[np.asarray(p, np.float32) for p in raw["pts"]], man_ok=n_ok, man_n=n, man_real=[bool(x) for x in real], junc_hit=hit, junc_near=near)


def check_graphs(doc, frames):
    """a graphs file holds the VAL frames in order (k, token, row, scene) and in every system record only the delivered fields (`SYS_KEYS`): no GT field"""
    assert [(r["k"], r["token"], r["row"], r["scene"]) for r in doc["frames"]] == [(f.k, f.token, f.row, f.scene) for f in frames], "the frames are not the VAL frames in order"
    for r in doc["frames"]:
        for name, s in r["sys"].items():
            assert tuple(sorted(s)) == tuple(sorted(SYS_KEYS)), f"frame {r['k']}, system {name}: fields {sorted(s)} are not {sorted(SYS_KEYS)}"
            assert all(isinstance(p, np.ndarray) and p.dtype == np.float32 and p.ndim == 2 for p in s["lines"]), f"frame {r['k']}, system {name}: lines"


def graphs(a, runs=RUNS):
    """the DELIVERED graphs of the run's VAL dump (of --model; the decode of record, or the variant of --link-thr / --link-thr-long / --keep-thr / --no-none-gate) in the schema of temp/qual_report_1115/pass1.pkl, as system `--name`: a new file, or one
    more system in an existing --out (its frames must be the VAL frames in order); per frame the topo F1 / P / R (`qgraph_eval.graph_counts` inside the 2 m lattice block, as the cards) into perframe_table.json next to --out (merged
    the same way). The pooled topo F1 is compared with the stored card of the same model and decode: a difference goes into `mismatch` as ('pooled_topo_F1', name, pooled, card), one pooled entry per system (pass1's entries are per
    frame). Writes only derived data: both files to temp names, then the table renamed first and the pickle last, so a name in the pickle is a finished write (or a system of another tool) and is refused, never overwritten;
    a name in the table only is the rest of an interrupted write and is written again"""
    TM = QE.TM
    out_path = _abs(a.out)
    table_path = os.path.join(os.path.dirname(out_path), "perframe_table.json")
    a.split, a.every, a.drop, a.drop_seed = "val", 1, 0.0, 0
    r = read_dump(a, runs, a.max)
    frames, vsfx = r["frames"], variant_suffix(a)
    doc = pickle.load(open(out_path, "rb")) if os.path.exists(out_path) else None
    tab = json.load(open(table_path)) if os.path.exists(table_path) else None
    if doc is not None:
        check_graphs(doc, frames)
        assert a.name not in doc["frames"][0]["sys"], f"{out_path} already holds the system {a.name}: never overwritten"
    assert tab is None or tab["tokens"] == [f.token for f in frames], f"{table_path}: other frames"
    fr = r["fr"]
    if vsfx:
        check_record(fr, r["valid"])
        fr = [decode_variant(f, v, a.link_thr, a.keep_thr, not a.no_none_gate, a.link_thr_long) for f, v in zip(fr, r["valid"])]
    recs, topo = [], []
    for k, (t, f, fm) in enumerate(zip(r["toks"], fr, frames)):
        pred = frame_prediction(t, f, "rec", a.keep_thr)
        graph, _ = QE.readout(pred, 0.5, 0.5)
        with JT.lattice(LATTICE):
            topo.append(QE.graph_counts(graph, fm)["topo"])
        fm._lane = fm._ref = None
        base, mans = frame_record(fm)
        recs.append((base, sys_record(QE.to_raw(graph), mans, fm)))
        fm._lane = fm._ref = None
        if (k + 1) % 100 == 0:
            print(f"   {k + 1}/{len(frames)} frames", flush=True)
    pooled = TM.pooled(topo)["topo_F1"]
    card = os.path.join(runs, a.run, f"token_card_rec{vsfx}{card_sfx(a)}.json")
    ref = json.load(open(card))["metrics"]["topo_F1"] if os.path.exists(card) and not a.max else None
    if doc is None:
        sys.path.append(os.path.join(ROOT, "tools", "viz"))
        import qgraph_view as QV
        try:
            scn = {s["name"]: s for s in json.load(open(os.path.join(os.path.expanduser("~/data/nuscenes_full/v1.0-trainval"), "scene.json")))}
            scenes = {sid: dict(name=nm, description=scn[nm]["description"], n=len(tk), tokens=tk) for sid, (nm, tk) in QV.val_scenes().items()}
        except FileNotFoundError:
            scenes = {}
        doc = dict(frames=[b for b, _ in recs], mismatch=[], scenes=scenes)
    for d, (_, s) in zip(doc["frames"], recs):
        d["sys"][a.name] = s
    if ref is not None and abs(pooled - ref) > 1e-9:
        doc["mismatch"].append(("pooled_topo_F1", a.name, pooled, ref))
    check_graphs(doc, frames)
    per = [TM.pooled([x]) for x in topo]
    tab = tab or dict(tokens=[f.token for f in frames], rows=[f.row for f in frames], scenes=[f.scene for f in frames], sys={})
    tab["sys"][a.name] = dict(run=a.run, model=a.model, seed=a.seed, decode="rec" + vsfx, dump=_file_id(r["path"]), pooled_topo_F1=pooled, card_topo_F1=ref,
                              topo_F1=[x["topo_F1"] for x in per], topo_P=[x["topo_P"] for x in per], topo_R=[x["topo_R"] for x in per])
    files = ((table_path, lambda fh: fh.write(json.dumps(_jsonable(tab)).encode())), (out_path, lambda fh: pickle.dump(doc, fh)))      # the table first, the pickle last
    for path, write in files:
        with open(path + ".tmp", "wb") as fh:
            write(fh)
    for path, _ in files:
        os.replace(path + ".tmp", path)
    print(f"{a.name}: {len(frames)} frames, pooled topo F1 {pooled:.4f} (stored card {ref}), maneuvers realised {sum(s['man_ok'] for _, s in recs)} of {sum(s['man_n'] for _, s in recs)}; written {out_path} and {table_path}")
    return 0


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
    model_arg = lambda s: s.add_argument("--model", default="model.pt", help="the weights of the run: model.pt or a snapshot model_step<N>.pt (`jtoken_train --snapshots`); the file names carry _step<N>")  # noqa: E731
    for name in ("dump", "score", "sweep"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        model_arg(s)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--split", choices=("val", "train"), **({"required": True} if name == "sweep" else {"default": "val"}))
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
            s.add_argument("--link-thr-long", type=float, default=None, help="a decode variant: the threshold of the long class (token chord > 8 m; >= 0.05; default: --link-thr); the names carry _ltl<T_L> when it differs")
            s.add_argument("--keep-thr", type=float, default=KEEP_THR, help="a decode variant: a node is kept at p_none < K (record: 0.5)")
            s.add_argument("--no-none-gate", action="store_true", help="a decode variant: no p >= p(none) condition on a link")
            s.add_argument("--ref-cell", type=parse_cell, default=None, help="with a variant: S,L,K,on|off = a reference cell (link short, link long, keep, none gate) scored on the same dump; the card gets its numbers and the paired interval")
            s.add_argument("--ref-run", default=None, help="another run's dump of the same read-out (its model.pt, the same cache) scored at the same decode: its numbers and the paired interval this run - it; the card's name ends in _vs_<run>")
            s.add_argument("--ref-run-cell", type=parse_cell, default=None, help="S,L,K,on|off: --ref-run read at this cell instead of this read-out's decode; the name gets _rc<cell> when it is not the card's own cell")
            s.add_argument("--argmax", action="store_true", help="the argmax decode (one link per source and class) at --keep-thr instead of the record's 0.5 -> token_card_argmax_kt<K>..json")
        if name == "sweep":
            s.add_argument("--grid", default="default", choices=tuple(SWEEP_GRIDS), help="default: the 24 cells (link x keep x none gate); perclass (E375): link short x link long x keep, none gate off (36 cells)")
    s = sub.add_parser("anatomy")
    s.add_argument("--run", required=True)
    model_arg(s)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--split", choices=("val", "train"), default="val")
    s.add_argument("--every", type=int, default=1)
    s.add_argument("--max", type=int, default=0, help="the dump of `dump --max N` (a test; the json goes to temp/e365/out/)")
    s.add_argument("--link-thr-long", type=float, default=0.2, help="a long link is delivered at p >= T_L (the D115 decode: 0.2)")
    s.add_argument("--keep-thr", type=float, default=0.6, help="a node is kept at p_none < K (the D115 decode: 0.6)")
    s = sub.add_parser("ctl")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--ident", required=True, choices=("e292_a", "e273_p"))
    s.add_argument("--K", type=int, default=None, help="tokens per frame, only when the cache root holds more than one K for the seed")
    s.add_argument("--cache", default=None)
    s.add_argument("--arms", default=None, help="comma list of " + ",".join(ARMS) + " (default all on VAL; on TRAIN only rule)")
    s.add_argument("--dump", default=None, help="the stored B dump of the bsec0 arms (default per ident, see B_DUMPS)")
    s.add_argument("--max", type=int, default=0, help="the first N frames; cards go to temp/e365/out/")
    s.add_argument("--split", default="val", choices=("val", "train"), help="train: the arm rule on the TRAIN sample (cards ..._train[_every<N>].json)")
    s.add_argument("--every", type=int, default=1, help="every N-th frame of the sample (the arm rule only)")
    s = sub.add_parser("fixture")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--run", default=None, help="E370 stage A: the run's head (else E369: the untrained --prior rule head of the seed)")
    model_arg(s)
    s.add_argument("--ident", default="e292_a", choices=("e292_a", "e273_p"), help="the cache of the untrained head (a run uses its own)")
    s.add_argument("--split", required=True, choices=("val", "train"))
    s.add_argument("--every", type=int, default=1)
    s.add_argument("--max", type=int, default=0, help="the first N frames of the sample (a test; the json goes to temp/e365/out/)")
    s.add_argument("--lam", type=float, default=JK.LAM)
    s.add_argument("--mu", type=float, default=JK.MU)
    s = sub.add_parser("graphs")
    s.add_argument("--run", required=True)
    model_arg(s)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--name", required=True, help="the system name in the file")
    s.add_argument("--out", required=True, help="the graphs pickle (pass1 schema; an existing one gets the system added); perframe_table.json goes next to it")
    s.add_argument("--max", type=int, default=0, help="the dump of `dump --max N` (a test)")
    s.add_argument("--link-thr", type=float, default=LINK_THR)
    s.add_argument("--link-thr-long", type=float, default=None)
    s.add_argument("--keep-thr", type=float, default=KEEP_THR)
    s.add_argument("--no-none-gate", action="store_true")
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
    a = ap.parse_args(argv)
    assert getattr(a, "every", 1) >= 1, f"--every {a.every}: at least 1"
    model_step(getattr(a, "model", "model.pt"))                                         # refuses another name before anything is loaded
    return a


def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(2)
    r = {"dump": dump, "score": score, "sweep": sweep, "anatomy": anatomy, "ctl": ctl, "fixture": fixture, "graphs": graphs, "check-layer1": check_layer1, "check-parity": check_parity, "timing": timing}[a.cmd](a)
    sys.exit(r if isinstance(r, int) else 0)


if __name__ == "__main__":
    main()
