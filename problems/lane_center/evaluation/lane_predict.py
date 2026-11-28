"""The prediction step of a lane run, and the dump it writes: the records of the read-out (`lane_readout.batch_dump`) of the run's network on the frames of a split.

    python -P -m lane_center.evaluation.lane_predict <the run's config.json>       (the worker starts it; RESEARCH_SPLIT names the split, default val)

On a GPU the step also measures compute time and memory of the whole path, one frame per forward (`measure_cost`: every frame's `ms`, the meta `cost`).

The step writes the LIVE dump <run>/token_<split>_live.pkl: the network (`lane_net.LaneNet`, frozen, no_grad) on the frames of the token cache in its order, B' in batches of
LIVE_BS 4 (the cache build's: its GPU parity holds at batch 4 only), the head in batches of HEAD_BS 8; no label is made and no ground truth is read. Each record holds
the frame's own `tokens_np` (the evaluation writes the nodes from them); meta LIVE_META: live, bs_b, bs_head, device, env, b_source (the sha256 of the B' state that ran, of its
node head and of the rest), tokens_exact (every frame's token fields bit-equal to the cache's: the parity read of a cache-trained run) and tokens_exact_frames. The evaluation step
(evaluation/adapter.py) reads the dump (evaluation/lane_dump.py `read_dump`: the run, the frames and the B' that ran are checked, `check_b`), decodes and scores it; nothing here computes a number.
B' of a run = its own trained `b_state` (a joint run), else runs/<ident>_s<S> (`base_b`); a run that trained the node head without a b_state is refused. A joint run is read
only live. A run that trains nothing (a measurement of an existing network) names it in the config: import_weights = "runs/<folder>" (may hold {seed}); its model.pt is copied
byte for byte into the run's folder first, since its checks (B''s sha256, the token cache) read the file. A stored dump is never overwritten: the step is done when it exists.
`dump` without `live` is the CACHE dump of the diagnostics (`jtoken_eval.py dump`: the head on the cached tokens -> token_<split>.pkl). Tests: evaluation/tests/test_jtoken_eval.py"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from lane_center.training import bflc_paths as BP
from lane_center.evaluation import jtiming as JTG
from lane_center.evaluation import lane_dump as DU
from lane_center.net.prep import lane_live_data as LLD
from lane_center.net.model import lane_net as LN
from lane_center.net.prep import lane_ptok_data as LP
from lane_center.net.model import lane_readout as RO

LIVE_BS = 4                                                                  # the batch of B' in a live dump: the cache build's (GPU parity with the cache is claimed at bs 4 only)
HEAD_BS = 8                                                                  # the batch of the head in every dump (cache and live): the cache dump's
TOKEN_KEYS = ("cell", "tok", "pos", "dirv", "e", "e_raw", "s", "uout", "uin", "flags", "valid", "e_p", "flags_p", "s_p")      # the token fields of a live dump compared with the cache's (`tokens_exact`; e_p / flags_p / s_p: the writer's)
LIVE_META = ("live", "b_source", "bs_b", "bs_head", "device", "env", "tokens_exact", "tokens_exact_frames")                   # the meta of a live dump that its cards and sweeps carry


# ------------------------------------------------------------------------------------------------------------------------------ the frames of a dump


def batches(ds, idx, bs=HEAD_BS, workers=0, collate=LP.collate):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [int(i) for i in idx]), batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate)


def run_head(m, b, dev):
    with torch.no_grad():
        return m(LN.to_device(b, dev))


# ------------------------------------------------------------------------------------------------------------------------------ the B' of a live dump


def live_source(path, ck, kw, K):
    """the network of a live read-out on the CPU, frozen, eval -> (`lane_net.LaneNet`, b_source). ck = the run's checkpoint (`path`): a joint run carries its trained B' as `b_state` (a complete `QGraph` state, loaded
    strictly into the architecture of `base_b`); a run that trained the node head (args train_nodes) without one is refused (no fall-back); every other run reads `base_b`. b_source = kind, the file read (path, sha256, mtime), the
    architecture's run and `b_shas` of the state that runs"""
    own = "b_state" in ck
    assert own or not ck["args"].get("train_nodes"), f"{path}: a run that trained B''s node head (train_nodes) must carry it (b_state): no fall-back to {DU.base_b(kw)}"
    net = LN.LaneNet.from_checkpoint(path, "cpu", K, b_run=BP.resolve(DU.base_b(kw)))[0]
    src = dict(kind="b_state" if own else "base", file=DU.file_id(path if own else os.path.join(BP.resolve(DU.base_b(kw)), "model.pt")), arch=DU.base_b(kw), **DU.b_shas(net.b.state_dict()))
    return net, src


def frames_collate(b):
    """x, token, row of the frames: the predict step reads no table and no target"""
    return dict(x=torch.stack([o["x"] for o in b]), token=[o["token"] for o in b], row=torch.tensor([o["row"] for o in b]))


def live_data(split, seed, ident):
    """the frames of a live read-out: `lane_live_data.LiveData` (x, table, token, row; the frames of the cache of B' seed `seed`, its order asserted)"""
    return LLD.LiveData(split, seed, ident)


def live_records(out, b):
    """`batch_dump` of a live batch `b` (token, row, tokens_np), each record with the frame's own `tokens_np`"""
    frames = RO.batch_dump(out, b)
    for j, f in enumerate(frames):
        f["tokens_np"] = b["tokens_np"][j]
    return frames


def same_tokens_as_cache(tk, ds, idx):
    """per frame of the live tokens `tk` (B, K, ...) of the cache frames idx: every field of TOKEN_KEYS bit-equal to the cache's (values compared in float64, which holds both exactly; tok is float16 in the cache)"""
    return [all(np.array_equal(tk[k][j].detach().cpu().numpy().astype(np.float64), np.asarray(ds.c.f[k][int(i)]).astype(np.float64)) for k in TOKEN_KEYS) for j, i in enumerate(idx)]


def live_dump(a, ck, kw, ds, idx, dev, runs):
    """the records of a live dump (`live_records`) of the cache frames idx: `live_data` (asserted: the frames of the cache `ds`, in its order) in batches of HEAD_BS (the head's, as the cache dump), B' (`live_source` of the run's
    checkpoint ck) on chunks of --bs of them (default LIVE_BS: the cache build's), under no_grad `tokens` -> the head on the tokens (no label is made: no ground truth is read) -> (frames, the meta entries
    LIVE_META: live, b_source, bs_b, bs_head, device, env (torch, CUDA, cuDNN, the GPU), tokens_exact = the token fields of every frame bit-equal to the cache's (`same_tokens_as_cache`; the parity read of a cache-trained run),
    tokens_exact_frames)"""
    src, b_source = live_source(os.path.join(runs, a.run, "model.pt"), ck, kw, ds.K)
    src = src.to(dev)
    data = live_data(a.split, kw["seed"], kw["ident"])
    assert list(data.tokens) == list(ds.tokens) and [int(r) for r in data.rows] == [int(r) for r in ds.rows], "the live frames are not the frames of the cache, in its order"
    frames, exact, bs = [], [], LIVE_BS if a.bs is None else a.bs
    for n, b in enumerate(batches(data, idx, collate=frames_collate)):
        with torch.no_grad():
            tks = [src.tokens(b["x"][c:c + bs])[1] for c in range(0, len(b["x"]), bs)]
            tk = {k: torch.cat([t[k] for t in tks]) for k in tks[0]}
            exact += same_tokens_as_cache(tk, ds, idx[n * HEAD_BS:(n + 1) * HEAD_BS])
            tn = LLD.host_tokens(tk)
            frames += live_records(run_head(src.head, LLD.head_inputs(tk), dev), dict(token=b["token"], row=b["row"], tokens_np=[LLD.tokens_np(tn, j) for j in range(len(b["token"]))]))
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None)       # as the token fixture records it
    meta = dict(live=True, b_source=b_source, bs_b=bs, bs_head=HEAD_BS, device=dev, env=env, tokens_exact=all(exact), tokens_exact_frames=int(sum(exact)))
    if dev == "cuda":
        ms, meta["cost"] = measure_cost(src, data, idx)
        for f, t in zip(frames, ms):
            f["ms"] = t
    return frames, meta


# ------------------------------------------------------------------------------------------------------------------------------ compute time and memory
WARMUP = 20                                                                  # untimed frames before the measurement (cuDNN picks its kernels on the first ones)


def _quiet_sample():
    """-> (the 1-minute load, the GPU state after our own kernels left nvidia-smi's window, the other compute processes on the GPU)"""
    time.sleep(2.5)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    return JTG.load1(), JTG.gpu_state(), JTG.gpu_others(apps, JTG.ancestors())


def measure_cost(net, data, idx):
    """the whole path (`lane_net.LaneNet.tokens` inside the bf16 autocast, then the head: the protocol of the recorded timing) on the frames `idx` of `data`, one frame per forward -> (ms per frame, dict(peak_MB, params_M,
    quiet, host)); the features are on the GPU before the clock starts (`lane_cost.py` says what the numbers mean)"""
    def forward(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return net.head(LLD.head_inputs(net.tokens(x, float32=False)[1]))

    xs = lambda i: data[int(i)]["x"][None].cuda()
    before = _quiet_sample()
    torch.cuda.empty_cache()
    for i in idx[:WARMUP]:
        forward(xs(i))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ms = [float(JTG.timed(lambda x=xs(i): forward(x), True)[1]) for i in idx]
    peak = torch.cuda.max_memory_allocated() / 2**20
    after = _quiet_sample()
    quiet, line = JTG.quiet_verdict(before[0], after[0], before[1], after[1])
    others = sorted(set(before[2]) | set(after[2]))
    params = sum(p.numel() for m in (net.b.pyr, net.b.nodes, net.head) for p in m.parameters())
    return ms, dict(peak_MB=float(peak), params_M=params / 1e6, quiet=bool(quiet and not others), host=line + (f"; other compute processes {others}" if others else ""))


# ------------------------------------------------------------------------------------------------------------------------------ write and read
def dump(a, runs=BP.RUNS):
    """the dump of the read-out `a` (run, seed, split, every, max, live, bs, cpu) -> its path: the cache dump (the head on the cached tokens) or, with `a.live`, the live dump (`live_dump`); never overwritten"""
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU; --cpu only for tests"
    assert a.live or a.bs is None, "--bs is the batch of B' in a live read-out: give --live"
    assert a.bs is None or a.bs > 0, f"--bs {a.bs}: a batch of at least 1"
    ck = LN.load_checkpoint(os.path.join(runs, a.run, "model.pt"))
    assert a.live or not DU.joint_run(ck), "a joint run is read only with --live"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        from lane_center.evaluation import gpu_gate
        gpu_gate.wait_for_room(who=f"lane_predict dump {a.run}")
    args = ck["args"]
    LN.check_removed(args, f"run {a.run}")
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    kw = DU.cache_kw(args)
    ds = DU.open_cache(a.split, kw["seed"], kw["ident"])
    idx = DU.sample(len(ds), a.split, a.every)[: a.max or None]
    path = DU.dump_path(a, runs, len(idx) if a.max else 0)
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    if a.live:
        frames, live = live_dump(a, ck, kw, ds, idx, dev, runs)
    else:
        m = LN.load_head(a.run, dev, runs)[0]
        frames, live = [], {}
        for b in batches(ds, idx, workers=0 if a.cpu else 2):
            frames += RO.batch_dump(run_head(m, b, dev), b)
    meta = dict(run=a.run, seed=a.seed, split=a.split, idx=idx.tolist(), every=a.every, tokens=[ds.tokens[i] for i in idx], K=ds.K, ident=kw["ident"], cache=ds.dir,
                steps=args.get("steps"), model="model.pt", **live)
    with open(path, "wb") as fh:
        pickle.dump(dict(meta=meta, frames=frames), fh)
    kept = np.mean([f["keep"].sum() for f in frames])
    print(f"written {os.path.relpath(path, BP.ROOT)}: {len(frames)} frames, kept nodes per frame {kept:.1f}, links per frame {np.mean([len(f['rec']['src']) for f in frames]):.1f}, device {dev}")
    return path


# ------------------------------------------------------------------------------------------------------------------------------ the step
def import_weights(cfg, run_dir):
    """the model.pt of `import_weights` into the run's folder, the same bytes"""
    src = os.path.join(BP.resolve(cfg["import_weights"].format(seed=cfg["seed"])), "model.pt")
    dst = os.path.join(run_dir, "model.pt")
    if not os.path.isfile(dst):
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
    if BP.file_sha256(src) != BP.file_sha256(dst):
        raise SystemExit(f"{dst} is not the model.pt of {src}: move it away")
    return src


def main(argv):
    cfg = json.load(open(argv[0], encoding="utf-8"))
    runs = os.path.normpath(cfg["out_root"])
    if runs != os.path.normpath(BP.RUNS):
        raise SystemExit(f"the run's out_root {runs} is not the problem's run root {BP.RUNS}")
    run_dir = os.path.join(runs, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    if cfg.get("import_weights"):
        print(f"weights of {import_weights(cfg, run_dir)}", flush=True)
    a = argparse.Namespace(run=cfg["name"], seed=cfg["seed"], split=os.environ.get("RESEARCH_SPLIT") or "val", live=True, bs=None, every=1, max=0, cpu=False)
    path = DU.dump_path(a, BP.RUNS, 0)
    if os.path.exists(path):
        print(f"stored: {path}")
        return 0
    dump(a, runs=BP.RUNS)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
