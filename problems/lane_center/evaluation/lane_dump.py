"""The dump of a read-out and its checked reader: the READ side of the predict step, on the scored path (in the evaluator hash, `adapter.EVALUATOR_FILES`); it runs no network.

    r = read_dump(a, runs, n_max)        # a = the read-out (run, seed, split, every, live): dict(ck, args, kw, ds, idx, meta, fr, toks, valid, path, live)

A dump is <run>/token_<split>[_every<E>][_live][_n<N>].pkl: meta + one record per frame (`lane_readout.batch_dump`); a live dump's record also holds the frame's own `tokens_np`.
`read_dump` checks that the dump is the one of this read-out (run, model, frames in the order of the token cache: all of VAL, `NSAMPLE` evenly spaced frames of TRAIN), that a
live dump ran the B' its run demands (`check_b`: the sha256 of the state that ran against the run's files), that a joint run is read live only, and that the kept nodes of the
records follow from the tokens it hands on: a live dump's own tokens, a cache dump's from the cache. The writer of a dump is evaluation/lane_predict.py.
Tests: evaluation/tests/test_jtoken_eval.py, evaluation/tests/test_evaluator_hash.py"""
import hashlib
import os
import pickle
import time

import numpy as np
import torch

import bflc_paths as BP
import lane_ptok_data as LP
import lane_readout as RO

NSAMPLE = 873                                                               # frames of a TRAIN read-out


def file_id(path):
    """sha256 and mtime of a file (the identity of a stored dump)"""
    return dict(path=os.path.relpath(path, BP.ROOT) if path.startswith(BP.ROOT) else path, sha256=BP.file_sha256(path), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(path))) + " UTC")


def cache_kw(args):
    """the cache of a run's recipe: the seed and the frozen head whose tokens are cached (`jtoken_train` writes ident into the run's args; an older smoke run has none: B')"""
    return dict(seed=args["seed"], ident=args.get("ident") or "e292_a")


def open_cache(split, seed, ident="e292_a"):
    return LP.PTokData(split, seed, ident)


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
    """the dump of a read-out: runs/R/token_<split>[_every<E>][_live][_n<n>].pkl (n: the frames of a `--max` dump)"""
    return os.path.join(runs, a.run, f"token_{a.split}{every_sfx(a)}{live_sfx(a)}{f'_n{n}' if n else ''}.pkl")


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


def check_b(meta, ck, kw, runs=BP.RUNS):
    """the B' a live dump ran (`b_shas` in its b_source) is the one its run demands: `base_b`'s model.pt is the file the run started from (args b_sha256, when the run records it) and every tensor outside the node head that ran is
    the START's (only `nodes.*` may train); a run that trained the node head (args train_nodes) ran its own `b_state`, whose node head differs from the start's; every other run (train_nodes 0, also with a b_state; a cache-trained head:
    the parity read) ran the node head of the start (and its `b_state` when it has one). The start = `base_b`'s state, or (E393: a run started with args b_state_from) the b_state of that run under `runs` (its model.pt the file
    of args b_state_sha256). E393 train_pyr: the pyramid (`pyr.*`) ran moved from the start, every tensor outside the node head and the pyramid as at the start"""
    got, path = meta["b_source"], os.path.join(BP.resolve(base_b(kw)), "model.pt")
    assert "b_sha256" not in ck["args"] or ck["args"]["b_sha256"] == file_id(path)["sha256"], f"{base_b(kw)}/model.pt is not the B' the run started from (args b_sha256)"
    frm = ck["args"].get("b_state_from")
    if frm:
        sp = os.path.join(runs, frm, "model.pt")
        assert file_id(sp)["sha256"] == ck["args"].get("b_state_sha256"), f"{sp} is not the start run's file the run started from (args b_state_sha256)"
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


def read_dump(a, runs, n_max):
    """the dump of the read-out `a` (of the run's model.pt), checked against the run and the cache -> dict(ck, args, kw, ds, idx, meta, fr, toks, valid, path, live).
    A live dump (`a.live`, its meta live) brings its own tokens (`tokens_np` of every record); the cache gives only the frames"""
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    live = bool(live_sfx(a))
    path = dump_path(a, runs, n_max)
    d = pickle.load(open(path, "rb"))
    meta, fr = d["meta"], d["frames"]
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"])
    idx = sample(len(ds), a.split, a.every)[: n_max or None]
    assert not meta.get("drop"), f"{path} was made with --drop {meta.get('drop')}: the degraded read-out no longer exists (the tag v1 keeps it)"
    assert meta["run"] == a.run and meta.get("model", "model.pt") == "model.pt" and meta.get("every", 1) == a.every and meta["idx"] == idx.tolist() and [f["token"] for f in fr] == [ds.tokens[i] for i in idx] \
        and bool(meta.get("live")) == live, "the dump is not the dump of this read-out (run, model, frames, live)"
    assert live or not joint_run(ck), "a joint run is read only with --live"
    if live:
        check_b(meta, ck, kw, runs)
    toks = [f["tokens_np"] for f in fr] if live else [ds[int(i)]["tokens_np"] for i in idx]          # live: the tokens this run's B' made for its head (the cache's are another B''s or another device's)
    assert all(np.array_equal(f["cell"], t["cell"]) for f, t in zip(fr, toks)), "the dump's cells are not its tokens_np's" if live else "the dump's tokens are not the cache's (a stale dump or cache)"
    valid = [np.asarray(t["valid"], bool) for t in toks]                                             # the tokens the head saw
    assert all(np.array_equal(v & (f["pnone"] < np.float32(RO.KEEP_THR)), f["keep"]) for v, f in zip(valid, fr)), "the valid tokens rebuilt from the cache are not the ones the dump's kept nodes came from"
    return dict(ck=ck, args=args, kw=kw, ds=ds, idx=idx, meta=meta, fr=fr, toks=toks, valid=valid, path=path, live=live)
