"""Read-out, scoring and checks of E365 S1 (the token head on the predicted nodes of the frozen lane head B', `net/model/jtoken.py`); the cache and the loader are `net/prep/lane_tokcache.py` / `lane_ptok_data.py`.

    wsl bash tools/env/_py14.sh jtoken_eval.py dump  --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0] [--every N] [--cpu] [--max N]   # GPU: the head on the cache tokens -> runs/R/token_<split>[_drop..].pkl
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0] [--every N] [--max N]                     # CPU -> runs/R/token_card_<decode>[_train][_drop..].json (rec, argmax; never overwritten)
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S --link-thr T [--link-thr-long T_L] --keep-thr K [--no-none-gate] [--ref-cell S,L,K,on|off]   # CPU: another decode of the same dump -> token_card_rec_lt<T>[_ltl<T_L>]_kt<K>[_nng][_train]..json
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S [a decode variant] --ref-run R2 [--ref-run-cell S,L,K,on|off]   # CPU, E376: + R2's dump of the same read-out at the same decode or that cell, paired R - R2 -> token_card_.._vs_<R2>[_rc<cell>].json
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S --argmax --keep-thr K                                          # CPU, E376: the argmax decode at another keep threshold -> token_card_argmax_kt<K>..json
    wsl bash tools/env/_py14.sh jtoken_eval.py ensemble --runs A,B --seed S [--split val|train] [--every N] [a decode variant] [--ref-run R [--ref-run-cell S,L,K,on|off]] [--out-dir D]   # CPU, E379: the mean of two heads' dumps -> runs/A/token_card_ens_<B>_rec..[_vs_<R>..].json
    wsl bash tools/env/_py14.sh jtoken_eval.py choose-cell --runs R1[,R2] [--ref-cell 0.3,0.2,0.6,off] [--split train --every 3] [--out-dir D]   # CPU: E375's cell rule (E377 / E378) on the per-class table(s) -> prints S,L,K,on|off, runs/R1/token_cell_perclass..json
    wsl bash tools/env/_py14.sh jtoken_eval.py anatomy --run R --seed S [--split val] [--link-thr-long 0.2] [--keep-thr 0.6] [--every N]   # CPU, E376: the delivered long links by category + the long-link and existence AP -> runs/R/token_anatomy_long_ltl<T>_kt<K>..json
    wsl bash tools/env/_py14.sh jtoken_eval.py sweep --run R --seed S --split train [--every N] [--max N] [--grid perclass]                                    # CPU: the per-class grid of decode variants on the TRAIN sample -> runs/R/token_sweep_perclass_train.json
    wsl bash tools/env/_py14.sh jtoken_eval.py graphs --run R --seed S --name <system> --out <file.pkl> [--link-thr T --link-thr-long T_L --keep-thr K --no-none-gate]   # CPU: the delivered VAL graphs in the schema of pass1.pkl + perframe_table.json
    wsl bash tools/env/_py14.sh jtoken_eval.py dump --live --run R --seed S [--bs 4] [--split val|train] [--every N] [--cpu] [--max N]   # GPU, E372: B' + the head on the LIVE tokens -> runs/R/token_<split>_live.pkl
    wsl bash tools/env/_py14.sh jtoken_eval.py score|sweep|anatomy --live ...                                                            # CPU, E372: a live dump read with its own tokens (its B' checked: `check_b`); every file name carries _live
    wsl bash tools/env/_py14.sh jtoken_eval.py score --live --run R --seed S [a decode] --ref-run R2 [--ref-cache]                       # CPU, E372: paired BY FRAME with R2's live dump, or (--ref-cache) with R2's CACHE dump (the parity read) -> .._live_vs_<R2>[_cache]..json
    wsl bash tools/env/_py14.sh jtoken_eval.py live-diff --run R --seed S --against P [--split val] [--max N]                           # CPU, E372: did B' move (cells, positions, n_valid, the live labels) -> runs/R/token_live_diff_vs_<P>.json
    wsl bash tools/env/_py14.sh jtoken_eval.py check-layer1 --seed S [--dump PATH] [--overwrite]                                        # CPU: this scorer fed B's stored dump reproduces runs/eval/e324s_val/card_s<S>.json (PASS / FAIL)
    wsl bash tools/env/_py14.sh jtoken_eval.py timing --run R [--b-run runs/e292_a_s<S>] [--K 384] [--cpu]                               # parameters, GMACs, ms (GPU only when idle: < 2 GB used and no run.py job)

Read-out of record (named before the run, on the output only): a node is kept when p_none < 0.5 (e = 1 - p_none); decode `rec`: a link i -> j of class c when p^c(j|i) >= 0.3, p^c(j|i) >= p^c(none|i) and both ends are kept (a fork keeps both
branches); decode `argmax`: per source and class the argmax over {none} + the candidates (one link per source and class). The prediction is written by `lane_ptok_data.to_pred` from the cached TOKENS only (pos, dirv, B''s flags and s; every node in
section 0): no field of the GT table enters (the real guarantee: the head reads only `jtoken.INPUTS`, test_jtoken.py; `check_writer` / `frame_prediction` assert on every frame that the nodes are the kept tokens' own, e = 1 - p_none and the links are the dump's) and the scorer is the standard one (`qgraph_eval.readout` -> `graph_counts` -> `graph_numbers` through `score_frames`, inside the `jtok.lattice(2.0)` block because
`LaneRef` builds a node table at score time; the headline graph numbers use the old graph GT and do not depend on it: `check-layer1` proves it). Handle scales: the head's for the long links, 1/3 for the short ones. p = 1 is written for every kept link (the
decode is the selection); the probability stays in the dump. A TRAIN read-out is 873 frames spaced evenly over the cache (`sample`), the degraded read-out (`--drop`) removes the share of the tokens that E357's `keep_nodes` removes (seeded by `--drop-seed` and
the frame token) before the head reads them. Tests: evaluation/tests/test_jtoken_eval.py
Anatomy (E376, promoted from temp/_session_scratch_d113/anatomy_long.py): a long link is DELIVERED at p >= T_L with both ends kept (p_none < K; the long class of the D115 decode, no none gate); each is tp (the ends are assigned to the
GT nodes of a GT long link), dup_of_true, wrong_pair or false_end (`link_category`); the long-link AP ranks every long candidate of the dump with both ends kept, the existence AP every valid token (`average_precision`); the mass
concentration (`concentration`) reads the head's own probabilities on its training label G(i) in both classes.
Live read-out (E372 S4, `dump --live`): B' runs with the head (`lane_live_tok.LiveSource`, frozen, no_grad; `live_source`: the run's own trained B' `b_state` (a joint run), else runs/<ident>_s<S>; a run that trained the node head
without a b_state is refused) on the cache's frames in its order: B' in batches of --bs (default 4, the cache build's), the head in batches of 8 (the cache dump's); the labels as the cache makes them (gate on, no token drop). Each record
of the dump also holds the frame's own `tokens_np` (the read-out writes the nodes from them, not from the cache), the label side (`lab`: the anatomy and `live-diff` read it) and the head's mass concentration (`conc`: a CPU forward
cannot remake the tokens of a GPU dump); meta LIVE_META: live, bs_b, bs_head, device, env, b_source (the sha256 of the B' state that ran, of its node head and of the rest), tokens_exact (every frame's token fields bit-equal to the
cache's: the parity read of a cache-trained run) and tokens_exact_frames; its cards, sweeps and anatomies carry them. Every read of a live dump checks its B' (`check_b`: runs/<ident>_s<S> is the file the run started from (args
b_sha256) and everything outside the node head is its; a run with train_nodes ran its own b_state, whose node head differs from runs/<ident>_s<S>'s; any other run that node head). A joint run (b_state or train_nodes) is read only
with --live (dump and the reads of a dump refuse it). Two live dumps hold other tokens of the same frames: `--ref-run` pairs them by frame (both of one B' batch and device); `--ref-cache` pairs the live read-out with the
reference's cache dump (the parity read: the topo / junction F1 difference in `paired_vs_ref_run`).
Ensemble (E379 arm e2, promoted from temp/e379/ens_step0.py): one frame per frame from the cache dumps of two runs of the same read-out (`mean_frame`), decoded and scored as one head's dump.
"""
import argparse
import functools
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
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtok as JT  # noqa: E402
import jtok_eval as JV  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
from pmap import pmap  # noqa: E402

ST = QE.TM.ST                                                               # the shared scene bootstrap (autoresearch/Evaluation/stats.py) that `qgraph_eval.paired` draws with
RUNS = JV.RUNS
SCRATCH = os.path.join(ROOT, "temp", "e365")
PLANB = os.path.join(RUNS, "e365_planb")                                    # plain B's scored records of `check-layer1` (planb_s<S>.pkl), read by `score`
DECODES = ("rec", "argmax")
LATTICE = 2.0
NSAMPLE = 873                                                               # frames of a TRAIN read-out
KEEP_P = 0.05                                                               # candidate links below this probability are not written into a dump
KEYS = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "Q", "junction_F1", "dup_frac", "vert100_dp")
CHECK = ("topo_F1", "F1@1.0", "lanes_found", "dup_frac", "vert100_dp")      # the layer-1 numbers (to 1e-6)
BASE_CARD = os.path.join(JV.CARDS, "e324s_val", "card_s{seed}.json")        # variants.primary.numbers.base = plain B with the close-node step, B's own sections
B_DUMP = "runs/kept/e343/val/predfull_e273_p_s{seed}.pkl"                       # the fresh VAL dump of plain B made the E324 s card
CLOSE = "close:1.5@0.5"
NODE_KEYS = ["cell", "dirv", "e", "flags", "pos", "s", "section"]
THIRD = 1 / 3
LIVE_BS = 4                                                                  # the batch of B' in a live dump: the cache build's (`lane_live_tok`: GPU parity with the cache is claimed at bs 4 only)
HEAD_BS = 8                                                                  # the batch of the head in every dump (cache and live): the cache dump's
TOKEN_KEYS = ("cell", "tok", "pos", "dirv", "e", "e_raw", "s", "uout", "uin", "flags", "valid", "e_p", "flags_p", "s_p")      # the token fields of a live dump compared with the cache's (`tokens_exact`; e_p / flags_p / s_p: the writer's)
LIVE_META = ("live", "b_source", "bs_b", "bs_head", "device", "env", "tokens_exact", "tokens_exact_frames")                   # the meta of a live dump that its cards, sweeps and anatomies carry
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


def live_sfx(a):
    """'' or _live: a read-out of the LIVE tokens of B' (`dump --live`); a command without --live is a cache read-out"""
    return "_live" if getattr(a, "live", False) else ""


def dump_path(a, runs, n=0):
    """the dump of a read-out: runs/R/token_<split>[_every<E>][_drop..][_live][_n<n>].pkl (n: the frames of a `--max` dump)"""
    return os.path.join(runs, a.run, f"token_{a.split}{every_sfx(a)}{JV.suffix(2.5, a.drop, a.drop_seed)}{live_sfx(a)}{f'_n{n}' if n else ''}.pkl")


def card_sfx(a):
    """the part of a card or sweep name after its decode: [_<split> unless val][_every<E>][_drop..][_live]"""
    return ("" if a.split == "val" else f"_{a.split}") + every_sfx(a) + JV.suffix(2.5, a.drop, a.drop_seed) + live_sfx(a)


def batches(ds, idx, bs=HEAD_BS, workers=0, collate=LP.collate):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [int(i) for i in idx]), batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate)


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
    return cand_of(*np.nonzero(link), p, longm, scale)


def cand_of(cs, cd, p, longm, scale):
    """the candidate links of a frame (p >= KEEP_P, in the gate): src, dst, p, long and the handle scales a, b (the head's for the long ones, 1/3 for the short ones, as `links_of`)"""
    lg = longm[cs, cd]
    ab = np.where(lg[:, None], scale[cs, cd], THIRD)
    return dict(src=cs.astype(np.int16), dst=cd.astype(np.int16), p=p[cs, cd].astype(np.float32), long=lg, a=ab[:, 0].astype(np.float32), b=ab[:, 1].astype(np.float32))


def batch_dump(out, b):
    """the head's output of a batch -> one dict per frame: token, row, cell (the tokens' cells: the score checks them against the cache), pnone / ptype (K,), keep (K,), none_cls (2, K) = p^c(none|i), cand = the candidate links with p >= KEEP_P
    (src, dst, p, long, a, b; the dumps of before the decode variants lack a, b and hold none_cls in float16), rec / argmax = the links of the two decodes (src, dst, p, long, a, b); dpos (K,) float32 = the normal offsets of a head with a
    position head (E395; the key is absent for a head without one)"""
    K = out["cand"].shape[-1]
    pr, rec, alt = JK.probs(out), JK.decode(out), JK.decode_argmax(out)
    npy = lambda t: t.detach().cpu().numpy()  # noqa: E731
    pt, p, cand, longm, scale, none_cls = (npy(x) for x in (pr["pt"], pr["p"], pr["cand"], pr["longm"], out["scale"].float(), out["logp"][..., K].float().exp()))
    keep, lrec, lalt = npy(rec["keep"]), npy(rec["link"]), npy(alt["link"])
    dpos = npy(out["dpos"].float()) if "dpos" in out else None
    frames = []
    for i in range(len(b["token"])):
        cs, cd = np.nonzero((p[i] >= KEEP_P) & cand[i])
        frames.append(dict(token=b["token"][i], row=int(b["row"][i]), cell=np.asarray(b["tokens_np"][i]["cell"], np.int16), pnone=pt[i, :, 0].astype(np.float32), ptype=pt[i].astype(np.float16), keep=keep[i],
                           none_cls=none_cls[i].astype(np.float32), cand=cand_of(cs, cd, p[i], longm[i], scale[i]),
                           rec=links_of(lrec[i], p[i], longm[i], scale[i]), argmax=links_of(lalt[i], p[i], longm[i], scale[i])))
        if dpos is not None:
            frames[-1]["dpos"] = dpos[i].astype(np.float32)
    return frames


# ------------------------------------------------------------------------------------------------------------------------------ the live read-out (E372)
def state_sha(sd, prefix=""):
    """sha256 over the tensors of a state dict whose names start with `prefix`, in sorted name order (name, dtype, shape and bytes of each): the identity of a B' that ran"""
    h = hashlib.sha256()
    for k in sorted(k for k in sd if k.startswith(prefix)):
        t = sd[k].detach().cpu().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


B_SHAS = ("sha_state", "sha_nodes", "sha_frozen")


def b_shas(sd):
    """the identity of a B' state (B_SHAS): sha_state (every tensor), sha_nodes (the node head, `nodes.*`), sha_frozen (every other tensor)"""
    return dict(sha_state=state_sha(sd), sha_nodes=state_sha(sd, "nodes."), sha_frozen=state_sha({k: v for k, v in sd.items() if not k.startswith("nodes.")}))


def joint_run(ck):
    """a run whose model.pt carries a B' of its own (`b_state`) or says it trained one (args train_nodes): an E372 joint run, read only with --live (the cache's tokens are not the ones its head was trained on)"""
    return "b_state" in ck or bool(ck["args"].get("train_nodes"))


def base_b(kw):
    """runs/<ident>_s<seed>: the B' of the run's cache (relative to the project folder)"""
    return f"runs/{kw['ident']}_s{kw['seed']}"


def live_source(path, ck, kw, K):
    """B' of a live read-out (`dump --live`) on the CPU, frozen, eval -> (`lane_live_tok.LiveSource`, b_source). ck = the run's checkpoint (`path`): a joint run carries its trained B' as `b_state` (a complete `QGraph` state, loaded
    strictly into the architecture of `base_b`); a run that trained the node head (args train_nodes) without one is refused (no fall-back); every other run reads `base_b`. b_source = kind, the file read (path, sha256, mtime), the
    architecture's run and `b_shas` of the state that runs"""
    import lane_live_tok as LT
    import qpredict as QP
    own = "b_state" in ck
    assert own or not ck["args"].get("train_nodes"), f"{path}: a run that trained B''s node head (train_nodes) must carry it (b_state): no fall-back to {base_b(kw)}"
    mb = QP.load_model(_abs(base_b(kw)), "cpu")
    if own:
        mb.load_state_dict(ck["b_state"])
    src = dict(kind="b_state" if own else "base", file=_file_id(path if own else os.path.join(_abs(base_b(kw)), "model.pt")), arch=base_b(kw), **b_shas(mb.state_dict()))
    return LT.LiveSource(mb, K, train_nodes=False).eval(), src


def check_b(meta, ck, kw, runs=RUNS):
    """the B' a live dump ran (`b_shas` in its b_source) is the one its run demands: `base_b`'s model.pt is the file the run started from (args b_sha256, when the run records it) and every tensor outside the node head that ran is
    the START's (only `nodes.*` may train); a run that trained the node head (args train_nodes) ran its own `b_state`, whose node head differs from the start's; every other run (train_nodes 0, also with a b_state; a cache-trained head:
    the parity read) ran the node head of the start (and its `b_state` when it has one). The start = `base_b`'s state, or (E393: a run started with args b_state_from) the b_state of that run under `runs` (its model.pt the file
    of args b_state_sha256). E393 train_pyr: the pyramid (`pyr.*`) ran moved from the start, every tensor outside the node head and the pyramid as at the start"""
    got, path = meta["b_source"], os.path.join(_abs(base_b(kw)), "model.pt")
    assert "b_sha256" not in ck["args"] or ck["args"]["b_sha256"] == _file_id(path)["sha256"], f"{base_b(kw)}/model.pt is not the B' the run started from (args b_sha256)"
    frm = ck["args"].get("b_state_from")
    if frm:
        sp = os.path.join(runs, frm, "model.pt")
        assert _file_id(sp)["sha256"] == ck["args"].get("b_state_sha256"), f"{sp} is not the start run's file the run started from (args b_state_sha256)"
        start_sd, start_name = torch.load(sp, map_location="cpu", weights_only=False)["b_state"], f"the start run {frm}"
    else:
        start_sd, start_name = torch.load(path, map_location="cpu")["state"], base_b(kw)
    base = b_shas(start_sd)
    if "b_state" in ck:
        assert {k: got[k] for k in B_SHAS} == b_shas(ck["b_state"]), "the live dump did not run the run's own B' (b_state)"
    if ck["args"].get("train_pyr"):
        assert "b_state" in ck, "a run that trained B''s pyramid carries no b_state"
        rest = lambda sd: state_sha({k: v for k, v in sd.items() if not k.startswith(("nodes.", "pyr."))})  # noqa: E731
        assert rest(ck["b_state"]) == rest(start_sd), f"the tensors of B' outside its node head and pyramid are not those of {start_name}"
        assert state_sha(ck["b_state"], "pyr.") != state_sha(start_sd, "pyr."), f"a run that trained B''s pyramid ran the pyramid of {start_name}"
    else:
        assert got["sha_frozen"] == base["sha_frozen"], f"the tensors of B' outside its node head are not those of {start_name}"
    if ck["args"].get("train_nodes"):
        assert "b_state" in ck and got["sha_nodes"] != base["sha_nodes"], f"a run that trained B''s node head ran the node head of {start_name} (or carries no b_state)"
    else:
        assert got["sha_nodes"] == base["sha_nodes"], f"a run with B''s node head frozen did not run the node head of {start_name}"


def live_data(split, seed, ident):
    """the frames of a live read-out: `lane_live_tok.LiveData` (x, table, token, row; the frames of the cache of B' seed `seed`, its order asserted)"""
    import lane_live_tok as LT
    return LT.LiveData(split, seed, ident)


def live_records(out, b):
    """`batch_dump` of a live batch `b` (`LiveSource.labelled`), each record with the frame's own `tokens_np`, `lab` (the label side the anatomy and `live-diff` read: gt_id, B''s
    uout / uin of the tokens, the label counts `lane_live_tok.STATS`) and `conc` (`head_concentration` of the frame: the anatomy's mass concentration)"""
    import lane_live_tok as LT
    npy = lambda t: t.detach().cpu().numpy()  # noqa: E731
    frames = batch_dump(out, b)
    for j, f in enumerate(frames):
        f.update(tokens_np=b["tokens_np"][j], conc=head_concentration({k: out[k][j:j + 1].detach().cpu() for k in ("cand", "logp")}, b["y"][j:j + 1].cpu()),
                 lab=dict(gt_id=npy(b["label"]["gt_id"][j]), uout=npy(b["uout"][j]), uin=npy(b["uin"][j]),
                          stats={k: int(b["lab"][j][k]) for k in LT.STATS}))
    return frames


def same_tokens_as_cache(tk, ds, idx):
    """per frame of the live tokens `tk` (B, K, ...) of the cache frames idx: every field of TOKEN_KEYS bit-equal to the cache's (values compared in float64, which holds both exactly; tok is float16 in the cache)"""
    return [all(np.array_equal(tk[k][j].detach().cpu().numpy().astype(np.float64), np.asarray(ds.c.f[k][int(i)]).astype(np.float64)) for k in TOKEN_KEYS) for j, i in enumerate(idx)]


def live_dump(a, m, ck, kw, ds, idx, dev, runs):
    """the records of a live dump (`live_records`) of the cache frames idx: `live_data` (asserted: the frames of the cache `ds`, in its order) in batches of HEAD_BS (the head's, as the cache dump), B' (`live_source` of the run's
    checkpoint ck) on chunks of --bs of them (default LIVE_BS: the cache build's), under no_grad `tokens` -> `labelled` (the gate on, no token drop: the labels as the cache makes them) -> the head -> (frames, the meta entries
    LIVE_META: live, b_source, bs_b, bs_head, device, env (torch, CUDA, cuDNN, the GPU), tokens_exact = the token fields of every frame bit-equal to the cache's (`same_tokens_as_cache`; the parity read of a cache-trained run),
    tokens_exact_frames)"""
    import lane_live_tok as LT
    src, b_source = live_source(os.path.join(runs, a.run, "model.pt"), ck, kw, ds.K)
    src = src.to(dev)
    data = live_data(a.split, kw["seed"], kw["ident"])
    assert list(data.tokens) == list(ds.tokens) and [int(r) for r in data.rows] == [int(r) for r in ds.rows], "the live frames are not the frames of the cache, in its order"
    frames, exact, bs = [], [], LIVE_BS if a.bs is None else a.bs
    for n, b in enumerate(batches(data, idx, collate=LT.collate)):
        with torch.no_grad():
            tks = [src.tokens(b["x"][c:c + bs])[1] for c in range(0, len(b["x"]), bs)]
            tk = {k: torch.cat([t[k] for t in tks]) for k in tks[0]}
            exact += same_tokens_as_cache(tk, ds, idx[n * HEAD_BS:(n + 1) * HEAD_BS])
            lb = src.labelled(tk, b["table"], b["token"], b["row"], token_drop=0.0, gate=True)
            frames += live_records(run_head(m, lb, dev), lb)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None)       # as `lane_live_tok.fixture` records it
    return frames, dict(live=True, b_source=b_source, bs_b=bs, bs_head=HEAD_BS, device=dev, env=env, tokens_exact=all(exact), tokens_exact_frames=int(sum(exact)))


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
    winner is a dump candidate (p >= KEEP_P): `check_argmax` asserts that before a read-out"""
    c = f["cand"]
    keep = np.asarray(valid, bool) & (np.asarray(f["pnone"]) < np.float32(keep_thr))
    src, dst, cls = c["src"].astype(int), c["dst"].astype(int), c["long"].astype(int)
    order = np.lexsort((dst, -c["p"], cls, src))                                                 # per source and class its best candidate first (a tie: the lower destination)
    first = np.ones(len(order), bool)
    first[1:] = (np.diff(src[order]) != 0) | (np.diff(cls[order]) != 0)
    w = order[first]
    w = np.sort(w[(c["p"][w] >= np.asarray(f["none_cls"], np.float32)[cls[w], src[w]]) & keep[src[w]] & keep[dst[w]]])       # the dump's order (source, destination)
    return dict(f, keep=keep, argmax={k: c[k][w] for k in LINK_FIELDS})


def check_argmax(frames, valid, keep_thr=KEEP_THR):
    """before an argmax read-out at keep_thr: on every frame `decode_argmax_variant` at the record keep threshold equals the dump's argmax decode on all six link fields; above the record's keep threshold also no source kept at
    keep_thr has a class with p(none) < KEEP_P and no dump candidate (its winner is a candidate below the dump's floor: the rebuild would drop that link silently)"""
    for f, v in zip(frames, valid):
        d = decode_argmax_variant(f, v)
        assert np.array_equal(d["keep"], f["keep"]) and all(np.array_equal(d["argmax"][k], f["argmax"][k]) for k in LINK_FIELDS), f"frame {f['token']}: the argmax decode cannot be rebuilt from the dump's candidates"
        if keep_thr > KEEP_THR:
            c, keep = f["cand"], decode_argmax_variant(f, v, keep_thr)["keep"]
            has = np.zeros((2, len(keep)), bool)
            has[c["long"].astype(int), c["src"].astype(int)] = True
            lost = keep[None, :] & ~has & (np.asarray(f["none_cls"], np.float32) < np.float32(KEEP_P))
            assert not lost.any(), f"frame {f['token']}: {int(lost.sum())} sources kept at {keep_thr} have their argmax below the dump's floor {KEEP_P}"


def mean_frame(fa, fb):
    """E379 arm e2: the ensemble of two heads' dump frames of the same frame and tokens -> one dump frame: pnone, ptype and none_cls = the means of the two heads; the candidates = the union of the two lists, sorted by (source,
    destination) as a dump's, p = the mean with 0 for a head that wrote no candidate for the pair (below its floor KEEP_P), long = the pair's class (the token chord: equal in both), the handle scales a, b = the mean over the heads
    that wrote the pair, heads = how many heads wrote it (1: the mean is low by less than KEEP_P / 2, `floor_numbers`). No decode of its own (keep, rec, argmax are not carried: `decode_variant` makes one). Means in float64,
    stored in the dump's dtypes"""
    assert fa["token"] == fb["token"] and fa["row"] == fb["row"] and np.array_equal(fa["cell"], fb["cell"]), f"frame {fa['token']}: the two dumps are not the same frame and tokens"
    K = len(fa["pnone"])
    mean = lambda k, dt: ((np.asarray(fa[k], np.float64) + np.asarray(fb[k], np.float64)) / 2).astype(dt)  # noqa: E731
    ca, cb = fa["cand"], fb["cand"]
    ka, kb = (c["src"].astype(np.int64) * K + c["dst"].astype(np.int64) for c in (ca, cb))
    assert len(np.unique(ka)) == len(ka) and len(np.unique(kb)) == len(kb), f"frame {fa['token']}: a pair twice in one head's candidates"
    key = np.union1d(ka, kb)
    acc = {k: np.zeros(len(key)) for k in ("p", "a", "b", "n")}
    cls = np.full((2, len(key)), -1, np.int8)
    for h, (c, i) in enumerate(((ca, np.searchsorted(key, ka)), (cb, np.searchsorted(key, kb)))):
        for k in ("p", "a", "b"):
            acc[k][i] += c[k]
        acc["n"][i] += 1
        cls[h, i] = c["long"]
    assert not ((cls >= 0).all(0) & (cls[0] != cls[1])).any(), f"frame {fa['token']}: a pair of another class in the two dumps"
    cand = dict(src=(key // K).astype(np.int16), dst=(key % K).astype(np.int16), p=(acc["p"] / 2).astype(np.float32), long=cls.max(0) == 1,
                a=(acc["a"] / acc["n"]).astype(np.float32), b=(acc["b"] / acc["n"]).astype(np.float32), heads=acc["n"].astype(np.int8))
    return dict(token=fa["token"], row=fa["row"], cell=fa["cell"], pnone=mean("pnone", np.float32), ptype=mean("ptype", np.float16), none_cls=mean("none_cls", np.float32), cand=cand)


FLOOR_NOTE = "links are a subset of the full-probability ensemble's at this cell"


def floor_numbers(fr, valid, cell):
    """the floor bias of the ensemble frames `fr` (`mean_frame`) at a cell (link short, link long, keep, none gate): delivered = the links of `decode_variant`; floor = the pairs written by ONE head only (the other's p, below its
    floor KEEP_P, counts 0: the mean is low by less than KEEP_P / 2) that are not delivered, with both ends kept and within KEEP_P / 2 of delivery (p >= the class threshold - KEEP_P / 2, and with the none gate p + KEEP_P / 2 >=
    p(none|i), also when p is above the threshold). p(none|i) and p_none are full means, so the delivered links are a subset of the full-probability ensemble's (FLOOR_NOTE), which delivers at most `floor` more -> dict"""
    lt, ltl, kt, gate = cell
    half, deliv, floor = np.float32(KEEP_P / 2), 0, 0
    for f, v in zip(fr, valid):
        c = f["cand"]
        src, dst, cls = c["src"].astype(int), c["dst"].astype(int), c["long"].astype(int)
        keep = np.asarray(v, bool) & (np.asarray(f["pnone"]) < np.float32(kt))
        thr, none = np.array([lt, ltl], np.float32)[cls], np.asarray(f["none_cls"], np.float32)[cls, src]
        ends, ok = keep[src] & keep[dst], (c["p"] >= thr) & ((c["p"] >= none) | (not gate))
        deliv += int((ends & ok).sum())
        floor += int((ends & ~ok & (c["heads"] == 1) & (c["p"] >= thr - half) & ((c["p"] + half >= none) | (not gate))).sum())
    n = max(len(fr), 1)
    return dict(delivered_links=deliv, delivered_links_per_frame=deliv / n, floor_links=floor, floor_links_per_frame=floor / n, floor_share=floor / max(deliv, 1), note=FLOOR_NOTE + f"; it delivers at most floor_links more "
                "(pairs written by one head only, both ends kept, within KEEP_P / 2 of delivery)")


# ------------------------------------------------------------------------------------------------------------------------------ plain B (the paired reference of `score`)
def pred_of_dump(d, spec):
    """a stored full dump frame of B (`qgraph_readout.dump_full`) -> the prediction as the read-out `spec` sees it (as `jtest_eval.score_dumps`), B's own sections"""
    nodes = dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s"))
    return QR.apply(dict(nodes=nodes, links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), QR.parse(spec))


# ------------------------------------------------------------------------------------------------------------------------------ scoring
@functools.lru_cache(maxsize=1)
def _train_graph_gt():
    """the old graph GT of the 8414 TRAIN frames (a 97 MB pickle: loaded once per process, `read_dump` of a TRAIN dump is called two or three times by `ensemble` / `--ref-run`)"""
    return pickle.load(open(f"{QE.LS.B123}/graph_gt_train.pkl", "rb"))


def frames_of(split, ds):
    """the scorer's Frame of every cache frame, in the cache order (VAL = `qgraph_eval.val_frames()`; TRAIN from the train files of the same layout)"""
    if split == "val":
        fr = QE.val_frames()
    else:
        tokens = json.load(open(f"{QE.LS.B123}/feats/train_tokens.json"))
        scene = np.load(f"{QE.LS.B123}/poses_train.npz")["scene"]
        gt = _train_graph_gt()
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
            out, r = QE.score_frames(preds[a:a + 100], frames[a:a + 100], records=True, lean=True)       # lean: no tier / section numbers, no topo marbles (nothing below reads them)
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
    return os.path.join(PLANB, f"planb_s{seed}.pkl")


def plain_b(seed, dump=None):
    """plain B as the E324 s card delivers it (B's own sections, the close-node step, links at p >= 0.5, no connectors) scored by THIS scorer -> the result of `score_preds` + `dump` (path, sha256, mtime of the dump used)"""
    dump_path = _abs(dump or B_DUMP.format(seed=seed))
    d = pickle.load(open(dump_path, "rb"))
    frames = QE.val_frames()
    assert [x["row"] for x in d] == [f.row for f in frames], f"{dump_path} is not in the order of the VAL frames"
    res = score_preds([pred_of_dump(x, CLOSE) for x in d], frames)
    res["dump"] = _file_id(dump_path)
    return res


def stored_plain_b(seed):
    """the scored plain B that `check-layer1` stored (only a PASS stores it): `score` needs it for the paired intervals and refuses without it"""
    path = planb_path(seed)
    assert os.path.exists(path), f"{path} is missing: run `check-layer1 --seed {seed}` first (it stores plain B's records only when it reproduces the E324 s card)"
    return pickle.load(open(path, "rb"))


def compare_numbers(got, ref, keys=CHECK, tol=1e-6):
    """-> (ok, {key: got - ref}) for the keys of a stored card"""
    assert all(k in got and k in ref for k in keys), f"missing numbers: {[k for k in keys if k not in got or k not in ref]}"
    diff = {k: float(got[k]) - float(ref[k]) for k in keys}
    return all(abs(v) <= tol for v in diff.values()), diff


# ------------------------------------------------------------------------------------------------------------------------------ commands
def dump(a, runs=RUNS):
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU; --cpu only for tests"
    assert a.live or a.bs is None, "--bs is the batch of B' in a live read-out: give --live"
    assert a.bs is None or a.bs > 0, f"--bs {a.bs}: a batch of at least 1"
    assert not (a.live and a.drop > 0), "--live: no --drop (the live read-out reads the tokens as B' makes them)"
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu", weights_only=False)
    assert a.live or not joint_run(ck), "a joint run is read only with --live"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jtoken_eval dump {a.run}")
    m, args = JKT.load_model(a.run, dev, runs)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split, a.every)[: a.max or None]
    path = dump_path(a, runs, len(idx) if a.max else 0)
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    if a.live:
        frames, live = live_dump(a, m, ck, kw, ds, idx, dev, runs)
    else:
        frames, live = [], {}
        for b in batches(ds, idx, workers=0 if a.cpu else 2):
            frames += batch_dump(run_head(m, apply_drop(b, a.drop, a.drop_seed), dev), b)
    meta = dict(run=a.run, seed=a.seed, split=a.split, drop=a.drop, drop_seed=a.drop_seed, idx=idx.tolist(), every=a.every, tokens=[ds.tokens[i] for i in idx], K=ds.K, ident=kw["ident"], cache=ds.dir,
                steps=args.get("steps"), model="model.pt", **live)
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
    """the dump of this read-out (of the run's model.pt) with everything the scorer needs, checked against the cache -> dict(ck, args, kw, ds, idx, meta, fr, toks, valid, frames, path, live).
    A live dump (`a.live`, its meta live) brings its own tokens (`tokens_np` of every record); the cache gives only the frames"""
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    live = bool(live_sfx(a))
    path = dump_path(a, runs, n_max)
    d = pickle.load(open(path, "rb"))
    meta, fr = d["meta"], d["frames"]
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split, a.every)[: n_max or None]
    assert meta["run"] == a.run and meta.get("model", "model.pt") == "model.pt" and meta.get("every", 1) == a.every and meta["idx"] == idx.tolist() and meta["drop"] == a.drop and meta["drop_seed"] == a.drop_seed and [f["token"] for f in fr] == [ds.tokens[i] for i in idx] \
        and bool(meta.get("live")) == live, "the dump is not the dump of this read-out (run, model, frames, drop, live)"
    assert live or not joint_run(ck), "a joint run is read only with --live"
    if live:
        check_b(meta, ck, kw, runs)
    toks = [f["tokens_np"] for f in fr] if live else [ds[int(i)]["tokens_np"] for i in idx]          # live: the tokens this run's B' made for its head (the cache's are another B''s or another device's)
    assert all(np.array_equal(f["cell"], t["cell"]) for f, t in zip(fr, toks)), "the dump's cells are not its tokens_np's" if live else "the dump's tokens are not the cache's (a stale dump or cache)"
    valid = [np.asarray(t["valid"], bool) & (JV.keep_nodes(f["token"], ds.K, a.drop, a.drop_seed) if a.drop > 0 else True) for t, f in zip(toks, fr)]     # the tokens the head saw
    assert all(np.array_equal(v & (f["pnone"] < np.float32(KEEP_THR)), f["keep"]) for v, f in zip(valid, fr)), "the valid tokens rebuilt from the cache are not the ones the dump's kept nodes came from"
    all_frames = frames_of(a.split, ds)
    return dict(ck=ck, args=args, kw=kw, ds=ds, idx=idx, meta=meta, fr=fr, toks=toks, valid=valid, frames=[all_frames[int(i)] for i in idx], path=path, live=live)


def live_meta(r):
    """the LIVE_META entries of a live read-out (`read_dump`) for its cards, sweeps and anatomies; {} for a cache read-out (its outputs stay as they were)"""
    return {k: r["meta"][k] for k in LIVE_META} if r["live"] else {}


def same_tokens(r, rr):
    """two read-outs (`read_dump`) hold the same frames in the same order; two cache read-outs also per frame the same tokens (cells) and the same valid tokens (one cache: a paired comparison of their heads). A live read-out on
    either side (each B' makes its own tokens) is paired by frame only; two live read-outs must have run B' at the same batch on the same device (meta bs_b, device)"""
    assert r["idx"].tolist() == rr["idx"].tolist() and [f["token"] for f in r["fr"]] == [f["token"] for f in rr["fr"]], "the reference run's dump holds other frames"
    if not (r.get("live") or rr.get("live")):
        assert all(np.array_equal(x["cell"], y["cell"]) and np.array_equal(u, w) for x, y, u, w in zip(r["fr"], rr["fr"], r["valid"], rr["valid"])), "the reference run's dump holds other tokens (another cache)"
    elif r.get("live") and rr.get("live"):
        assert all(r["meta"][k] == rr["meta"][k] for k in ("bs_b", "device")), f"two live dumps of another B' batch or device: {[(r['meta'][k], rr['meta'][k]) for k in ('bs_b', 'device')]}"


def read_ref(a, r, runs, n_max):
    """--ref-run: the dump of the SAME read-out (split, every, drop, drop seed, --max, live) of another run, of its model.pt (`read_dump`), refused unless its frames (and, two cache read-outs, its tokens) are those of this read-out `r`
    (`same_tokens`); --ref-cache: the reference's CACHE dump against this live read-out (the parity read)"""
    rr = read_dump(argparse.Namespace(**dict(vars(a), run=a.ref_run, live=a.live and not a.ref_cache)), runs, n_max)
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
        check_argmax(r["fr"], r["valid"], a.keep_thr)
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
    """the cards of one dump (module docstring). --max N reads the dump of `dump --max N`, scores its N frames and writes the cards to `out_dir` (default temp/e365/out) as ..._n<N>.json.
    --link-thr T [--link-thr-long T_L] --keep-thr K [--no-none-gate] re-score the dump with another decode (`decode_variant`, after `check_record`; only the `rec` card, its name carries the variant), the decode of record scored beside it
    on the same dump (`paired_vs_record`; its numbers against the stored record card are reported, not asserted: a re-dump may differ); the defaults are the decode of record, both decodes. A variant with --ref-cell S,L,K,on|off
    also scores that cell on the same dump: its numbers (`ref_cell`) and the paired scene-bootstrap interval of the variant against it (`paired_vs_ref_cell`: the keys of `qgraph_eval.paired` + junction F1 / P / R, `paired_junction`).
    --ref-run R2 (E376) also scores R2's dump of the same read-out (`read_ref`: its model.pt, the same frames and tokens) at the same decode, or at --ref-run-cell S,L,K,on|off: its numbers and cell (`ref_run`) and the paired interval of
    this run minus R2 (`paired_vs_ref_run`, the keys of `paired_vs_ref_cell`, the same draws); the card's name ends in _vs_<R2>[_rc<cell>, only when R2 is read at another cell].
    --argmax --keep-thr K (E376): the head's argmax decode at another keep threshold (`decode_argmax_variant`, after `check_argmax`) -> the card token_card_argmax_kt<K>..json.
    --live (E372): the live dump (`dump --live`) with its own tokens (`check_b` first), the names carry _live, the card info the dump's LIVE_META; --ref-run then reads R2's live dump paired
    by frame, or with --ref-cache R2's CACHE dump (the parity read; the name ends in _vs_<R2>_cache[_rc<cell>])"""
    n_max = a.max
    assert not a.argmax or (a.link_thr, a.link_thr_long, a.no_none_gate, a.ref_cell) == (LINK_THR, None, False, None), "--argmax takes --keep-thr only"
    assert a.ref_run or a.ref_run_cell is None, "--ref-run-cell is the cell of --ref-run"
    assert not a.ref_cache or (a.live and a.ref_run), "--ref-cache reads --ref-run's cache dump against this live read-out: give --live and --ref-run"
    vsfx = f"_kt{a.keep_thr:g}" if a.argmax else variant_suffix(a)
    assert vsfx or a.ref_cell is None, "--ref-cell is compared with a decode variant: give --link-thr / --link-thr-long / --keep-thr / --no-none-gate"
    out_dir = (out_dir or os.path.join(SCRATCH, "out")) if n_max else None
    sfx = card_sfx(a)
    nsfx = f"_n{n_max}" if n_max else ""
    decs = ("argmax",) if a.argmax else ("rec",) if vsfx else DECODES
    rcells = {d: ref_run_cell(a, d) for d in decs}
    rsfx = {d: f"_vs_{a.ref_run}" + ("_cache" if a.ref_cache else "") + (cell_sfx(rcells[d]) if rcells[d] else "") if a.ref_run else "" for d in decs}
    card_path = lambda d, v, x="": os.path.join(out_dir, f"{a.run}_token_card_{d}{v}{sfx}{nsfx}{x}.json") if n_max else os.path.join(runs, a.run, f"token_card_{d}{v}{sfx}{x}.json")  # noqa: E731
    paths = {d: card_path(d, vsfx, rsfx[d]) for d in decs}
    held = [p for p in paths.values() if os.path.exists(p)]
    assert not held, f"a stored card is never overwritten: {held}"
    r = read_dump(a, runs, n_max)
    rr = read_ref(a, r, runs, n_max) if a.ref_run else None
    ck, kw, ds, frames = r["ck"], r["kw"], r["ds"], r["frames"]
    ref = stored_plain_b(a.seed) if a.split == "val" and not a.drop else None

    def pick(rec):                                                                  # the graph records of plain B on the frames of this read-out (the cache order = the VAL order); the paired interval reads these only
        n = len(rec["graph"])
        assert n == len(ds) if not n_max else n > int(r["idx"].max()), f"plain B's records of {n} frames cannot be indexed by this read-out (frames {r['idx'][:5].tolist()} ..., {len(ds)} in the cache)"
        return dict(graph=[rec["graph"][int(i)] for i in r["idx"]])
    ref_rec = None if ref is None else pick(ref["rec"])
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
                                    **{k: res_ref["graph"][k] for k in KEYS + ("junction_P", "junction_R")}, **live_meta(rr))
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
        rule = "the argmax decode at another keep threshold (`decode_argmax_variant`)" if a.argmax else "the variant of `decode_variant`" if vsfx else "rec: p >= 0.3 and p >= p(none), both ends kept (p_none < 0.5); argmax: one link per source and class"
        out[dec] = write_card(paths[dec], f"{a.run}_token_{dec}{vsfx}{sfx}{nsfx}{rsfx[dec]}", res, len(frames), a.split, run=a.run, model="model.pt", seed=a.seed, decode=dec, drop=a.drop, drop_seed=a.drop_seed, steps=ck.get("steps"), head_params=params,
                              K=ds.K, ident=kw["ident"], decode_rule=rule, **live_meta(r), **extra)
    return out


def run_pair(s, what="--runs"):
    """'A,B' -> (A, B): two run names, stripped, not the same run"""
    x = [r.strip() for r in s.split(",")]
    assert len(x) == 2 and all(x) and x[0] != x[1], f"{what} {s}: two different runs A,B"
    return tuple(x)


def ens_card_path(a, runs):
    """the card of `ensemble` -> (path, name): runs/A/<name>.json, name = token_card_ens_<B>_rec<variant>[_<split>][_every<N>][_n<N>][_vs_<R>[_rc<cell>]]; --out-dir D or --max N: <D or temp/e365/out>/A_<name>.json"""
    A, B = run_pair(a.runs)
    rc = ref_run_cell(a, "rec")
    rsfx = f"_vs_{a.ref_run}" + (cell_sfx(rc) if rc else "") if a.ref_run else ""
    name = f"token_card_ens_{B}_rec{variant_suffix(a)}{card_sfx(a)}{f'_n{a.max}' if a.max else ''}{rsfx}"
    out_dir = a.out_dir or (os.path.join(SCRATCH, "out") if a.max else None)
    return (os.path.join(out_dir, f"{A}_{name}.json") if out_dir else os.path.join(runs, A, f"{name}.json")), name


def ensemble(a, runs=RUNS):
    """E379 arm e2: the cache dumps of the same read-out (split, every, --max; model.pt, no drop) of the two runs of --runs A,B (`read_dump`: the seed of both runs is --seed; one cache, opened at the same folder; `same_tokens`:
    the same frames, tokens and valid tokens; `check_record` on both) -> one frame per frame (`mean_frame`), decoded at the decode of record or the variant of --link-thr / --link-thr-long / --keep-thr / --no-none-gate
    (`decode_variant`) and scored by the standard scorer; `floor` = `floor_numbers` at the cell; --ref-run R [--ref-run-cell S,L,K,on|off]: R's dump of the same read-out (`read_ref`) at the same decode or that cell, `ref_run` +
    `paired_vs_ref_run` (the ensemble minus R, as `score`). The card `ens_card_path`, never overwritten"""
    A, B = run_pair(a.runs)
    n_max = a.max
    assert a.ref_run or a.ref_run_cell is None, "--ref-run-cell is the cell of --ref-run"
    cell = (a.link_thr, long_thr(a), a.keep_thr, not a.no_none_gate)
    rc = ref_run_cell(a, "rec")
    path, name = ens_card_path(a, runs)
    assert not os.path.exists(path), f"a stored card is never overwritten: {path}"
    ra, rb = (read_dump(argparse.Namespace(**dict(vars(a), run=x)), runs, n_max) for x in (A, B))
    assert os.path.realpath(ra["ds"].dir) == os.path.realpath(rb["ds"].dir), f"{A} and {B} read other caches ({ra['ds'].dir}, {rb['ds'].dir}): B's probabilities would be paired with A's tokens"
    same_tokens(ra, rb)
    for r in (ra, rb):
        check_record(r["fr"], r["valid"])
    ens = dict(toks=ra["toks"], valid=ra["valid"], fr=[mean_frame(x, y) for x, y in zip(ra["fr"], rb["fr"])])
    frames = ra["frames"]
    res = score_cell(ens, cell, frames)
    show(f"ens {A}+{B}", res["graph"])
    extra = dict(floor=floor_numbers(ens["fr"], ens["valid"], cell))
    print(f"  delivered links per frame {extra['floor']['delivered_links_per_frame']:.2f}, of which the floor may hide at most {extra['floor']['floor_links_per_frame']:.3f} more ({FLOOR_NOTE})", flush=True)
    if a.ref_run:
        rr = read_ref(a, ra, runs, n_max)
        check_record(rr["fr"], rr["valid"])
        res_ref = score_cell(rr, rc or cell, frames, log=lambda *x, **k: None)
        show(f"  ref {a.ref_run}{cell_sfx(rc) if rc else ''}", res_ref["graph"])
        extra["ref_run"] = dict(run=a.ref_run, model="model.pt", dump=_file_id(rr["path"]), cell=dict(zip(("link_thr", "link_thr_long", "keep_thr", "none_gate"), rc)) if rc else "the decode of this card",
                                **{k: res_ref["graph"][k] for k in KEYS + ("junction_P", "junction_R")})
        extra["paired_vs_ref_run"] = dict(QE.paired(res["rec"], res_ref["rec"], frames), **paired_junction(res["rec"], res_ref["rec"], frames))
    return write_card(path, f"{A}_{name}", res, len(frames), a.split, runs=[A, B], dumps={x: _file_id(r["path"]) for x, r in ((A, ra), (B, rb))}, seed=a.seed, every=a.every, K=ra["ds"].K, ident=ra["kw"]["ident"],
                      decode="rec", decode_variant=dict(zip(("link_thr", "link_thr_long", "keep_thr", "none_gate"), cell)), steps={A: ra["ck"].get("steps"), B: rb["ck"].get("steps")},
                      ensemble="`mean_frame`: p_none, ptype and p(none|i) of each class = the means of the two heads; the candidate links = the union of the two dumps' lists, p = the mean with 0 for a head that wrote no candidate "
                               f"for the pair (below its floor {KEEP_P}), the handle scales = the mean over the heads that wrote the pair; then `decode_variant` (the scales of the ensemble's candidates) and the standard scorer",
                      e_written=f"min(1 - p_none + {a.keep_thr} - 0.5, 1): only the scorer's e >= 0.5 is meant; a later read-out must not rank nodes by it", **extra)


SWEEP_KEEP = (0.5, 0.6, 0.7)
SWEEP_KEYS = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "dup_frac")
PERCLASS_SHORT, PERCLASS_LONG = (0.3, 0.2, 0.1), (0.3, 0.2, 0.1, 0.05)                  # E375: the link thresholds of the short and of the long class
SWEEP_GRIDS = dict(perclass=[(lt, ltl, kt, False) for lt in PERCLASS_SHORT for ltl in PERCLASS_LONG for kt in SWEEP_KEEP])      # cells (link thr, long link thr, keep thr, none gate)


def sweep(a, runs=RUNS, out_dir=None):
    """the decode variants of one dump on its read-out frames -> one table runs/R/token_sweep_perclass_<split>[_drop..].json (--max N: temp/e365/out/..._n<N>.json), never overwritten. The grid (E375, `--grid perclass`,
    the only one): link threshold short {0.3, 0.2, 0.1} x long {0.3, 0.2, 0.1, 0.05} x keep threshold {0.5, 0.6, 0.7}, none gate off; the rows carry junction F1 / P / R and the predicted junction points per frame (the scorer's,
    degree >= 3), the delivered long links per frame and the scorer's long-link precision / recall (node_link_long_P / _R: GT long links between matched nodes). For choosing a decode on TRAIN frames (the head was trained on them: the
    table says what the output could deliver, not what it does on VAL)"""
    n_max = a.max
    sfx = "_perclass" + card_sfx(a)
    path = os.path.join((out_dir or os.path.join(SCRATCH, "out")) if n_max else os.path.join(runs, a.run), f"{a.run + '_' if n_max else ''}token_sweep{sfx}{f'_n{n_max}' if n_max else ''}.json")
    assert not os.path.exists(path), f"a stored table is never overwritten: {path}"
    r = read_dump(a, runs, n_max)
    check_record(r["fr"], r["valid"])

    def cell_row(cell):
        lt, ltl, kt, gate = cell
        fr = [decode_variant(f, v, lt, kt, gate, ltl) for f, v in zip(r["fr"], r["valid"])]
        res = score_preds([frame_prediction(t, f, "rec", kt) for t, f in zip(r["toks"], fr)], r["frames"], log=lambda *x, **k: None)
        row = dict(link_thr=lt, link_thr_long=ltl, keep_thr=kt, none_gate=gate, record=(lt, ltl, kt, gate) == (LINK_THR, LINK_THR, KEEP_THR, True), frames=len(fr),
                   kept_nodes_per_frame=float(np.mean([f["keep"].sum() for f in fr])), links_per_frame=float(np.mean([len(f["rec"]["src"]) for f in fr])), **{k: res["graph"][k] for k in SWEEP_KEYS})
        row.update({k: res["graph"][k] for k in JUNCTION_KEYS}, junction_points_per_frame=float(np.mean([c["lane"]["junc"][0] for c in res["rec"]["graph"]])), long_links_per_frame=float(np.mean([f["rec"]["long"].sum() for f in fr])),
                   **{f"node_link_long_{k}": res["nodes"].get(f"link_long_{k}") for k in ("P", "R")})
        print(f"link {lt:<5g} long {ltl:<5g} keep {kt:<4g} gate {'on ' if gate else 'off'}  " + "  ".join(f"{k} {row[k]:.4f}" for k in SWEEP_KEYS + JUNCTION_KEYS)
              + f"  links/frame {row['links_per_frame']:.1f}", flush=True)                   # the progress line of the cell (with --jobs N the lines of different cells interleave; the table is in grid order)
        return row

    rows = pmap(cell_row, SWEEP_GRIDS[a.grid], getattr(a, "jobs", 1))                          # --jobs N: the cells are independent, N forked processes, the same numbers in the same order
    JV.write_json(path, _jsonable(dict(run=a.run, model="model.pt", seed=a.seed, split=a.split, every=a.every, max=a.max, drop=a.drop, drop_seed=a.drop_seed, steps=r["ck"].get("steps"), dump=_file_id(r["path"]), frames=len(r["frames"]), rows=rows, K=r["ds"].K,
                                       grid="perclass", **live_meta(r), note="decode variants of one dump (`decode_variant`, the head's scales); e written = min(1 - p_none + keep_thr - 0.5, 1): a later read-out must not rank nodes by it")))
    print(f"written {os.path.relpath(path, ROOT)}")
    return path


D115 = (0.3, 0.2, 0.6, False)                                                     # the champion's cell (link short, link long, keep, none gate)
CHOOSE_DUP, CHOOSE_LANES, CHOOSE_WINDOW = 0.05, 0.010, 0.005                       # E375's rule as pinned in E377 / E378: dup_frac bound, lanes-found loss to the reference cell, topo F1 window
CHOOSE_KEYS = ("topo_F1", "junction_F1", "lanes_found", "dup_frac")


def table_cell(row):
    """the cell of a sweep row: (link thr, long link thr, keep thr, none gate)"""
    return row["link_thr"], row["link_thr_long"], row["keep_thr"], row["none_gate"]


def pick_cell(tables, ref=D115):
    """E375's cell rule as pinned in E377 / E378 (promoted from temp/e377/choose377.py and temp/e378/choose378.py): tables = the rows of one or two sweep tables of the same cells, averaged row by row; eligible = mean dup_frac <=
    CHOOSE_DUP and mean lanes found >= the mean at the reference cell - CHOOSE_LANES; T = the highest mean topo F1 of the eligible cells; the window = the eligible cells with topo F1 >= T - CHOOSE_WINDOW; the chosen cell = the highest
    mean junction F1 of the window, ties by |short - ref short|, |long - ref long|, |keep - ref keep|, then the higher thresholds; no eligible cell: the reference cell -> dict(cell, rows, eligible, T, window)"""
    cells = [{table_cell(r): r for r in t} for t in tables]
    assert all(c.keys() == cells[0].keys() and len(c) == len(t) for c, t in zip(cells, tables)) and ref in cells[0], "the tables hold other cells (or a cell twice, or not the reference cell)"
    rows = {c: {k: sum(t[c][k] for t in cells) / len(cells) for k in CHOOSE_KEYS} for c in cells[0]}
    ok = [c for c, r in rows.items() if r["dup_frac"] <= CHOOSE_DUP and r["lanes_found"] >= rows[ref]["lanes_found"] - CHOOSE_LANES]
    if not ok:
        return dict(cell=ref, rows=rows, eligible=[], T=None, window=[])
    T = max(rows[c]["topo_F1"] for c in ok)
    win = sorted((c for c in ok if rows[c]["topo_F1"] >= T - CHOOSE_WINDOW), key=lambda c: (-rows[c]["junction_F1"], abs(c[0] - ref[0]), abs(c[1] - ref[1]), abs(c[2] - ref[2]), -c[0], -c[1], -c[2]))
    return dict(cell=win[0], rows=rows, eligible=ok, T=T, window=win)


def cell_str(c):
    """'S,L,K,on|off' (the form of --ref-cell / --ref-run-cell)"""
    return f"{c[0]:g},{c[1]:g},{c[2]:g},{'on' if c[3] else 'off'}"


def choose_cell(a, runs=RUNS):
    """the decode cell of one run (its own per-class sweep table, `sweep --grid perclass`) or of a recipe of two runs (the mean of their tables row by row) by `pick_cell` against --ref-cell (default D115) -> prints the cell as
    S,L,K,on|off (the last line) and writes runs/R1/token_cell_perclass[_<split>][_every<N>][_mean_<R2>][_ref<cell>].json (the reference only when not D115; --out-dir D: <D>/R1_token_cell_..json). Never overwritten: a
    stored choice is recomputed, must be the same cell, and is printed"""
    rs = [r.strip() for r in a.runs.split(",")]
    assert 1 <= len(rs) <= 2 and all(rs) and len(set(rs)) == len(rs), f"--runs {a.runs}: one run or two different runs"
    ref = a.ref_cell or D115
    paths = [os.path.join(runs, r, f"token_sweep_perclass{card_sfx(a)}.json") for r in rs]
    tabs = [json.load(open(p)) for p in paths]
    assert all(t.get("grid") == "perclass" and t["split"] == a.split and t.get("every", 1) == a.every for t in tabs), "per-class sweep tables of this split and sample"
    got = pick_cell([t["rows"] for t in tabs], ref)
    name = f"token_cell_perclass{card_sfx(a)}{f'_mean_{rs[1]}' if len(rs) == 2 else ''}{'' if ref == D115 else '_ref' + cell_sfx(ref)[3:]}.json"
    path = os.path.join(a.out_dir, f"{rs[0]}_{name}") if a.out_dir else os.path.join(runs, rs[0], name)
    num = lambda c: {k: got["rows"][c][k] for k in CHOOSE_KEYS}  # noqa: E731
    doc = dict(runs=rs, tables={r: _file_id(p) for r, p in zip(rs, paths)}, ref_cell=cell_str(ref), cell=cell_str(got["cell"]), numbers=num(got["cell"]), ref_numbers=num(ref), eligible=len(got["eligible"]), cells=len(got["rows"]),
               T=got["T"], window=[dict(cell=cell_str(c), **num(c)) for c in got["window"]], rule=f"eligible: dup_frac <= {CHOOSE_DUP} and lanes found >= the reference cell's - {CHOOSE_LANES} (the mean of the tables row by row); "
               f"T = the highest topo F1; window topo F1 >= T - {CHOOSE_WINDOW}; the highest junction F1; ties by |short - ref|, |long - ref|, |keep - ref|, then the higher thresholds; no eligible cell: the reference cell")
    if os.path.exists(path):
        assert json.load(open(path))["cell"] == doc["cell"], f"{path} holds another cell than the tables give now: never overwritten"
        print(f"stored {os.path.relpath(path, ROOT)}")
    else:
        JV.write_json(path, _jsonable(doc))
        print(f"written {os.path.relpath(path, ROOT)}: {len(got['eligible'])} of {len(got['rows'])} cells eligible, T {got['T']}, window {len(got['window'])}")
    print(doc["cell"])
    return doc


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
    for every source with a non-empty G(i): above = its in-class candidates with p >= thr; share = the largest member's share of the group's mass (max over G(i) of p / sum over G(i) of p) for the sources whose group has mass,
    zero = the sources whose group has none (no share) -> (share, above, zero)"""
    rows = G.any(1)
    top, mass = np.where(G, p, 0.0).max(1), np.where(G, p, 0.0).sum(1)
    ok = rows & (mass > 0)
    return top[ok] / mass[ok], ((p >= thr) & cand).sum(1)[rows], int((rows & ~ok).sum())


def head_concentration(out, y):
    """a head output of one frame (B = 1) and its GT pairs y (1, K, K) -> class (short, long) -> `group_concentration` on the training label of `jtoken.jtoken_loss` (G = y and the class gate) with p = exp(logp) in float64"""
    K = out["cand"].shape[-1]
    G, cand = ((y > 0)[:, None] & out["cand"])[0].numpy(), out["cand"][0].numpy()
    p = out["logp"][0, :, :, :K].double().exp().numpy()
    return {name: group_concentration(p[c], G[c], cand[c]) for c, name in enumerate(("short", "long"))}


def concentration(m, ds, idx, fr, valid):
    """the mass concentration of the head `m` on the cache frames idx, per class (a CPU forward, batch 1: the dump does not hold the probabilities below KEEP_P; its p_none must agree with the dump's `fr` to 1e-3 on the `valid`
    tokens) -> class -> `concentration_numbers` over the sources with a non-empty G(i) (`head_concentration`, `pool_concentration`)"""
    parts = []
    for b, f, v in zip(batches(ds, idx, bs=1), fr, valid):
        out = run_head(m, b, "cpu")
        assert np.abs(JK.probs(out)["pt"][0, :, 0].numpy() - f["pnone"])[v].max(initial=0.0) < 1e-3, f"frame {f['token']}: the head's p_none is not the dump's"
        parts.append(head_concentration(out, b["y"]))
    return pool_concentration(parts)


def pool_concentration(parts):
    """the frames' `head_concentration` parts pooled -> class -> `concentration_numbers`"""
    return {c: concentration_numbers(np.concatenate([p[c][0] for p in parts]), np.concatenate([p[c][1] for p in parts]), sum(p[c][2] for p in parts)) for c in ("short", "long")}


def concentration_numbers(s, n, zero=0):
    """the pooled shares s, counts n and zero-mass sources of `group_concentration` -> sources, zero_mass_sources, share median / mean (the sources with group mass), above_mean, above_2plus (the share of sources with 2 or more),
    above_mean_given_one (over the sources with one or more); None where empty"""
    v = lambda x, fn: float(fn(x)) if len(x) else None  # noqa: E731
    return dict(sources=len(n), zero_mass_sources=zero, share_median=v(s, np.median), share_mean=v(s, np.mean), above_mean=v(n, np.mean), above_2plus=v(n >= 2, np.mean), above_mean_given_one=v(n[n >= 1], np.mean))


def anatomy(a, runs=RUNS, out_dir=None):
    """E376: the anatomy of the delivered long links of the run's dump at --link-thr-long / --keep-thr (module docstring; `anatomy_numbers`; the GT side: the cache's gt_id / uout / uin, read once, and the frame's table of
    `lane_tok_data.TokData`) and the mass concentration of the head in both classes (`concentration`, a CPU forward of the head on the same frames)
    -> runs/R/token_anatomy_long_ltl<T>_kt<K>[_<split>][_every<N>][_live].json, never overwritten (out_dir or --max: <out_dir or temp/e365/out>/R_token_anatomy_..[_n<N>].json), and a table. --live: the label side and the
    concentration of the live dump's records (`live_records`), not the cache's"""
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
    if r["live"]:
        labs = [f["lab"] for f in r["fr"]]
        gid, uout, uin = ([x[k] for x in labs] for k in ("gt_id", "uout", "uin"))
    else:
        gid, uout, uin = (np.asarray(ds.c.f[k][idx]) for k in ("gt_id", "uout", "uin"))
    parts, n_gt = [], 0
    for k, (i, f, t, v) in enumerate(zip(idx, r["fr"], r["toks"], r["valid"])):
        lab = gt_labels(t, v, td.table(int(i)), gid[k], uout[k], uin[k])
        parts.append(anatomy_frame(f, v, lab, a.keep_thr))
        n_gt += len(lab["gt_long"])
    num = anatomy_numbers(parts, n_gt, a.link_thr_long)
    num["concentration"] = pool_concentration([f["conc"] for f in r["fr"]]) if r["live"] else concentration(JKT.load_model(a.run, "cpu", runs)[0], ds, idx, r["fr"], r["valid"])
    JV.write_json(path, _jsonable(dict(run=a.run, model="model.pt", seed=a.seed, split=a.split, every=a.every, max=a.max, frames=len(idx), K=ds.K, link_thr_long=a.link_thr_long, keep_thr=a.keep_thr, dump=_file_id(r["path"]),
                                       near=dict(r=LC.R_NEAR, cos=LC.COS_NEAR), **num, **live_meta(r), note="delivered = long candidate with p >= link_thr_long and both ends kept (p_none < keep_thr), no none gate; categories of `link_category`; "
                                       "median_b_long = the median of the PRODUCT sigmoid(uout[src]) * sigmoid(uin[dst]), median_typed of the product p_entry(src) * p_exit(dst); AP = the mean precision at the positives, ties broken "
                                       "pessimistically, recall over the positives of the list (x recall_ceiling = recall over the GT long links); concentration: "
                                       + ("the forward of the live dump itself (`live_records`, on its device, with its live labels)" if r["live"] else "a CPU forward of the head") + ", per class the sources with a non-empty G(i) "
                                       f"(y and the class gate, the training label), share = max / sum of p over G(i) (sources with no mass on G(i): zero_mass_sources), above = the in-class candidates with p >= {LINK_THR}")))
    fmt = lambda x, s: "-" if x is None else format(x, s)  # noqa: E731
    print(f"{a.run} {a.split}, {len(idx)} frames: GT long links {n_gt}, delivered long links (p >= {a.link_thr_long:g}, both ends p_none < {a.keep_thr:g}) {num['delivered']}")
    for k, x in num["categories"].items():
        print(f"  {k:<12} {x['n']:7d}  share {x['share']:.3f}  median p {fmt(x['median_p'], '.2f')}  typed {fmt(x['median_typed'], '.3f')}  B' uout*uin {fmt(x['median_b_long'], '.2f')}")
    print(f"  tp / GT long links {num['tp_over_gt_long']:.3f}; AP long links {fmt(num['ap_long'], '.4f')} over {num['long_candidates']} kept long candidates (recall ceiling {num['recall_ceiling']:.3f}); "
          f"AP existence {fmt(num['ap_existence'], '.4f')} over {num['valid_tokens']} valid tokens ({num['assigned_tokens']} assigned)")
    for c, x in num["concentration"].items():
        print(f"  {c:<5} {x['sources']} sources with a GT successor ({x['zero_mass_sources']} with no mass on it): largest member's share of G(i)'s mass median {fmt(x['share_median'], '.3f')} mean {fmt(x['share_mean'], '.3f')}; candidates >= {LINK_THR} per source "
              f"{fmt(x['above_mean'], '.3f')}, 2 or more {fmt(x['above_2plus'], '.3f')}, per source with one or more {fmt(x['above_mean_given_one'], '.3f')}")
    print(f"written {os.path.relpath(path, ROOT)}")
    return path


def label_numbers(fr):
    """the live labels of a dump's records (`lab` stats) pooled: target pairs per GT link, valid tokens per frame, GT node coverage (assigned GT nodes / GT nodes)"""
    s = {k: sum(f["lab"]["stats"][k] for f in fr) for k in ("n_pairs", "n_links", "n_valid", "n_gt", "n_gt_assigned")}
    return dict(pos_pairs=s["n_pairs"], gt_links=s["n_links"], pos_pairs_per_gt_link=s["n_pairs"] / max(s["n_links"], 1), n_valid_per_frame=s["n_valid"] / max(len(fr), 1), gt_coverage=s["n_gt_assigned"] / max(s["n_gt"], 1))


def live_diff(a, runs=RUNS, out_dir=None):
    """E372: did B' move? The live dump of --run against the live dump of --against (P: the parity read of the same seed, the cache-trained champion with the untouched B') on the same frames (`same_tokens`; both through `read_dump`,
    so `check_b` holds for both): the share of frames whose top-K cell list changed (as a list, as a set), |delta pos| of the cells valid in both (median, p90), delta n_valid per frame, and the moving-label numbers of both
    (`label_numbers`) -> runs/R/token_live_diff_vs_<P>[_<split>][_every<N>].json, never overwritten (--max or out_dir: <out_dir or temp/e365/out>/R_token_live_diff_vs_<P>.._n<N>.json)"""
    a.live, a.drop, a.drop_seed = True, 0.0, 0
    name = f"token_live_diff_vs_{a.against}{'' if a.split == 'val' else '_' + a.split}{every_sfx(a)}{f'_n{a.max}' if a.max else ''}.json"
    path = os.path.join(out_dir or os.path.join(SCRATCH, "out"), f"{a.run}_{name}") if out_dir or a.max else os.path.join(runs, a.run, name)
    assert not os.path.exists(path), f"a stored table is never overwritten: {path}"
    r = read_dump(a, runs, a.max)
    p = read_dump(argparse.Namespace(**dict(vars(a), run=a.against)), runs, a.max)
    same_tokens(r, p)
    order, sets, dpos, dvalid = [], [], [], []
    for f, g, t, u, v, w in zip(r["fr"], p["fr"], r["toks"], p["toks"], r["valid"], p["valid"]):
        order.append(not np.array_equal(f["cell"], g["cell"]))
        sets.append(set(f["cell"].tolist()) != set(g["cell"].tolist()))
        where = {int(c): k for k, c in enumerate(g["cell"])}
        both = [(k, where[int(c)]) for k, c in enumerate(f["cell"]) if int(c) in where and v[k] and w[where[int(c)]]]
        i, j = (np.array([x[n] for x in both], int) for n in (0, 1))
        dpos.append(np.linalg.norm(np.asarray(t["pos"], float)[i] - np.asarray(u["pos"], float)[j], axis=1) if both else np.zeros(0))
        dvalid.append(int(v.sum()) - int(w.sum()))
    dp = np.concatenate(dpos)
    lr, lp = label_numbers(r["fr"]), label_numbers(p["fr"])
    doc = dict(run=a.run, against=a.against, seed=a.seed, split=a.split, every=a.every, max=a.max, frames=len(order), dumps={a.run: _file_id(r["path"]), a.against: _file_id(p["path"])},
               live={a.run: live_meta(r), a.against: live_meta(p)}, cells=dict(order_changed_share=float(np.mean(order)), set_changed_share=float(np.mean(sets))),
               pos=dict(common_valid_cells=len(dp), median_abs_delta_m=float(np.median(dp)) if len(dp) else None, p90_abs_delta_m=float(np.percentile(dp, 90)) if len(dp) else None),
               n_valid=dict(delta_mean=float(np.mean(dvalid)), delta_abs_mean=float(np.mean(np.abs(dvalid)))), labels={a.run: lr, a.against: lp, "delta": {k: lr[k] - lp[k] for k in lr}},
               note="cells: the top-K list of a frame differs (order) / as a set; pos: |delta| of the cells valid in both dumps; n_valid: run minus against per frame; labels: the live labels of each dump (n_pairs / n_links, n_valid, n_gt_assigned / n_gt)")
    JV.write_json(path, _jsonable(doc))
    print(f"{a.run} vs {a.against} ({len(order)} frames): cell list changed on {doc['cells']['order_changed_share']:.3f} of the frames (as a set {doc['cells']['set_changed_share']:.3f}), median |delta pos| {doc['pos']['median_abs_delta_m']} m over "
          f"{len(dp)} common valid cells, delta n_valid {doc['n_valid']['delta_mean']:+.2f} per frame; target pairs per GT link {lr['pos_pairs_per_gt_link']:.3f} vs {lp['pos_pairs_per_gt_link']:.3f}, GT node coverage {lr['gt_coverage']:.4f} vs {lp['gt_coverage']:.4f}; written {os.path.relpath(path, ROOT)}")
    return path


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
    """the DELIVERED graphs of the run's VAL dump (the decode of record, or the variant of --link-thr / --link-thr-long / --keep-thr / --no-none-gate) in the schema of temp/qual_report_1115/pass1.pkl, as system `--name`: a new file, or one
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
            topo.append(QE.graph_counts(graph, fm, lean=True)["topo"])                                # lean: loads no lane GT (frame_record below loads it once)
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
    tab["sys"][a.name] = dict(run=a.run, model="model.pt", seed=a.seed, decode="rec" + vsfx, dump=_file_id(r["path"]), pooled_topo_F1=pooled, card_topo_F1=ref,
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
    E324 s card -> 0 (PASS: topo F1, F1@1.0, lanes found, dup_frac, vert100_dp within 1e-6) / 1 (FAIL); the other stored numbers are printed, and the sha256 and mtime of the dump used. Only a PASS stores plain B's records (runs/e365_planb/planb_s<S>.pkl),
    which `score` needs; a stored file is kept unless --overwrite"""
    ref = json.load(open(BASE_CARD.format(seed=a.seed)))["variants"]["primary"]["numbers"]["base"]
    res = plain_b(a.seed, a.dump)
    ok, diff = compare_numbers(res["graph"], ref)
    for k in ref:
        print(f"   {k}: {res['graph'][k]:.9f} stored {ref[k]:.9f} diff {res['graph'][k] - ref[k]:+.2e}" + ("" if k in CHECK else "   (not checked)"))
    print(f"dump {res['dump']['path']} sha256 {res['dump']['sha256']} mtime {res['dump']['mtime']}")
    print(f"{'PASS' if ok else 'FAIL'}: layer 1, seed {a.seed}: this scorer on B's stored predictions against {os.path.relpath(BASE_CARD.format(seed=a.seed), ROOT)}: " + ", ".join(f"{k} {v:+.2e}" for k, v in diff.items()))
    path = planb_path(a.seed)
    if ok and os.path.exists(path) and not a.overwrite:
        print(f"not written: {os.path.relpath(path, ROOT)} exists (--overwrite replaces it)")
    elif ok:
        os.makedirs(PLANB, exist_ok=True)
        pickle.dump(res, open(path, "wb"))
    return int(not ok)


def live_tokens(mb, x, K):
    """B' (`QP.load_model`) on the input features x (B, 39, 200, 200) -> the head's input batch exactly as `lane_tokcache.build` makes the cache (the same `select_tokens`)"""
    P = mb.pyr(x.float())
    no = mb.node_stage(P)
    t = LC.select_tokens(no, None, K)
    return {k: t[k] for k in JK.INPUTS + ("cell",)}


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
                row["head"].append(JTG.timed(lambda: m(b), dev == "cuda")[1])
                if mb is not None:
                    row["bprime"].append(JTG.timed(lambda: live_tokens(mb, xs[i], K), dev == "cuda")[1])
                    row["whole"].append(JTG.timed(lambda: m(live_tokens(mb, xs[i], K)), dev == "cuda")[1])
        if rep:                                                                           # the first pass is untimed
            for k in ms:
                if row[k]:                                                                # B' alone and the whole path only with --b-run
                    ms[k].append(row[k])
    if not a.cpu:
        torch.cuda.synchronize()
        time.sleep(2.5)                                                                   # as jtiming's protocol: our own last kernels age out of nvidia-smi's averaging window before the after-sample
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
        s.add_argument("--split", choices=("val", "train"), **({"required": True} if name == "sweep" else {"default": "val"}))
        s.add_argument("--drop", type=float, default=0.0, help="the degraded read-out: the share of the tokens removed (0.17)")
        s.add_argument("--drop-seed", type=int, default=0)
        s.add_argument("--every", type=int, default=1, help="every N-th frame of the split's sample (TRAIN: 873 evenly spaced frames; every 3 = 291); the file names carry _every<N>; score and sweep read the dump with the same value")
        s.add_argument("--live", action="store_true", help="E372: the LIVE tokens of B' (dump: B' runs with the head; score / sweep: the live dump with its own tokens); the file names carry _live")
        if name == "dump":
            s.add_argument("--cpu", action="store_true", help="tests only: dump on the CPU")
            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; `score --max N` reads it)")
            s.add_argument("--bs", type=int, default=None, help=f"--live: the batch of B' (default {LIVE_BS}, the cache build's: the GPU parity holds at bs 4 only; the head runs at {HEAD_BS} as the cache dump); in the dump's meta")
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
            s.add_argument("--ref-cache", action="store_true", help="with --live: --ref-run's CACHE dump instead of its live dump (the parity read); the name ends in _vs_<run>_cache")
            s.add_argument("--argmax", action="store_true", help="the argmax decode (one link per source and class) at --keep-thr instead of the record's 0.5 -> token_card_argmax_kt<K>..json")
        if name == "sweep":
            s.add_argument("--grid", default="perclass", choices=tuple(SWEEP_GRIDS), help="perclass (E375, the only grid): link short x link long x keep, none gate off (36 cells)")
            s.add_argument("--jobs", type=int, default=1, help="N forked processes score the cells (the same numbers in the same order; about 2.5x at 3); default 1 = the loop of before")
    s = sub.add_parser("ensemble")
    s.add_argument("--runs", required=True, help="A,B: two runs whose cache dumps of this read-out (model.pt, the same cache seed, frames and tokens) are averaged; the card goes to runs/A/")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--split", choices=("val", "train"), default="val")
    s.add_argument("--every", type=int, default=1)
    s.add_argument("--max", type=int, default=0, help="the dumps of `dump --max N` (a test; the card goes to --out-dir or temp/e365/out/)")
    s.add_argument("--out-dir", default=None, help="the card goes to <D>/A_token_card_ens_..json instead of runs/A/ (a read-out that is not a run's record)")
    s.add_argument("--link-thr", type=float, default=LINK_THR)
    s.add_argument("--link-thr-long", type=float, default=None)
    s.add_argument("--keep-thr", type=float, default=KEEP_THR)
    s.add_argument("--no-none-gate", action="store_true")
    s.add_argument("--ref-run", default=None, help="another run's dump of the same read-out at the same decode (or --ref-run-cell): its numbers and the paired interval ensemble - it; the card's name ends in _vs_<run>")
    s.add_argument("--ref-run-cell", type=parse_cell, default=None)
    s.set_defaults(drop=0.0, drop_seed=0, live=False, argmax=False, ref_cache=False)          # the cache read-out of model.pt that `read_dump` / `read_ref` / `card_sfx` read
    s = sub.add_parser("choose-cell")
    s.add_argument("--runs", required=True, help="R1[,R2]: one run (its own per-class table) or the two runs of a recipe (the mean of the two tables); the json goes to runs/R1/")
    s.add_argument("--ref-cell", type=parse_cell, default=None, help="S,L,K,on|off: the reference cell (default D115 0.3,0.2,0.6,off): the lanes-found bound, the tie-break and the fall-back")
    s.add_argument("--split", choices=("val", "train"), default="train")
    s.add_argument("--every", type=int, default=3, help="the table of `sweep --grid perclass --every N` (default 3: the TRAIN every-3 table)")
    s.add_argument("--out-dir", default=None, help="the json goes to <D>/R1_token_cell_..json instead of runs/R1/")
    s.set_defaults(drop=0.0, drop_seed=0, live=False)                                       # the table name of `sweep` (`card_sfx`)
    s = sub.add_parser("anatomy")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--split", choices=("val", "train"), default="val")
    s.add_argument("--every", type=int, default=1)
    s.add_argument("--max", type=int, default=0, help="the dump of `dump --max N` (a test; the json goes to temp/e365/out/)")
    s.add_argument("--link-thr-long", type=float, default=0.2, help="a long link is delivered at p >= T_L (the D115 decode: 0.2)")
    s.add_argument("--keep-thr", type=float, default=0.6, help="a node is kept at p_none < K (the D115 decode: 0.6)")
    s.add_argument("--live", action="store_true", help="E372: the live dump (its own tokens, labels and concentration); the name carries _live")
    s = sub.add_parser("live-diff")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--against", required=True, help="the run whose live dump is the reference (the parity read: the cache-trained champion read live with the untouched B')")
    s.add_argument("--split", choices=("val", "train"), default="val")
    s.add_argument("--every", type=int, default=1)
    s.add_argument("--max", type=int, default=0, help="the live dumps of `dump --live --max N` (a test; the json goes to temp/e365/out/)")
    s = sub.add_parser("graphs")
    s.add_argument("--run", required=True)
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
    s.add_argument("--dump", default=None, help="the stored dump of plain B (default runs/kept/e343/val/predfull_e273_p_s<seed>.pkl)")
    s.add_argument("--overwrite", action="store_true", help="replace the stored plain B records runs/e365_planb/planb_s<seed>.pkl (kept otherwise)")
    s = sub.add_parser("timing")
    s.add_argument("--run", required=True)
    s.add_argument("--b-run", default=None, help="also time B' alone and the whole path")
    s.add_argument("--K", type=int, default=None, help="tokens of the head (default: the K of the cache of the run, 384 in the card)")
    s.add_argument("--frames", type=int, default=24)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    assert getattr(a, "every", 1) >= 1, f"--every {a.every}: at least 1"
    return a


def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(2)
    r = {"dump": dump, "score": score, "sweep": sweep, "ensemble": ensemble, "choose-cell": choose_cell, "anatomy": anatomy, "live-diff": live_diff, "graphs": graphs, "check-layer1": check_layer1, "timing": timing}[a.cmd](a)
    sys.exit(r if isinstance(r, int) else 0)


if __name__ == "__main__":
    main()
