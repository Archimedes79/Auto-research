"""Seconds per training step of the token head: the plain path against the speed path (`jtoken_fast.py`, `lane_ptok_gpu.py`) on the real TRAIN cache of B' seed 0, and K 384 against K 512. A DIAGNOSTIC: it decides nothing and feeds no verdict.

    wsl bash tools/env/_py14.sh bench_token_step.py gate [--out DIR]     # K 512: plain (CPU loader, workers 2) / fast head + CPU loader / fast + GPU loader / fast + GPU loader + compile_pairs, each in turn (diagnostics);
                                                                       # then THE GATE `gate_abba`: plain against the fast path as the card's arm runs it (fast head + fast loss + GPU loader, eager), blocks ABBA of 60 steps in one process
    wsl bash tools/env/_py14.sh bench_token_step.py pair [--out DIR]     # the speed path at K 512 and K 384 (cache root runs/e383_cache, built by `data/build_tokcache_root.py --root runs/e383_cache --run runs/e292_a_s0 --split train --K 384 --no-f2`)
                                                                       # in ONE process, blocks ABBA of 60 steps, eager -> ms per training step of K 384 over K 512

The step is the trainer's (bf16 autocast, clip 3.0, AdamW, a `float()` of the loss and the gradient norm each step, in every variant alike). 2 CPU threads, one GPU job below 4 GB; time it on a quiet host (idle GPU, no other run.py job:
a load moves both blocks of `pair` alike, `gate` less so). The json `<out>/bench_<mode>.json` (default runs/eval/token_step_bench) says which code it ran: the git HEAD, and the dirty / untracked files among the speed-path files
(`code`), the host load and the GPU used before the start. Start it through run.py as a diagnostic: `python3 ../autoresearch/tools/run.py --id <id> --gpu --no-card --timeout 20 -- bash tools/env/_py14.sh bench_token_step.py pair`.
"""
import json
import os
import subprocess
import sys
import time

import torch

torch.set_num_threads(2)
R = os.environ["BFLC_ROOT"]
for sub_ in ("training", "training/grid", "training/data", "training/train", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub_))
import jtoken as JK  # noqa: E402
import jtoken_fast as JF  # noqa: E402
import jtoken_train as JT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
import lane_ptok_gpu as LG  # noqa: E402

FILES = ("training/grid/jtoken_fast.py", "training/data/lane_ptok_gpu.py", "training/data/lane_ptok_data.py", "training/grid/jtoken.py", "training/train/jtoken_train.py", "training/train/bench_token_step.py")
dev = "cuda"


def code_state():
    """which code this ran: the git HEAD and, per speed-path file, clean / modified / untracked (a bench of a dirty tree must say so)"""
    git = lambda *a: subprocess.run(["git", *a], cwd=R, capture_output=True, text=True).stdout.strip()  # noqa: E731
    st = {}
    for f in FILES:
        s = git("status", "--porcelain", "--", f)
        st[f] = "clean" if not s else ("untracked" if s.startswith("??") else "modified")
    return dict(git_head=git("rev-parse", "--short", "HEAD"), files=st)


class NoRun:
    def log(self, *a, **k):
        pass


def endless(loader):
    while True:
        for b in loader:
            yield b


def make_step(head, loss_fn):
    head.to(dev).train()
    head._step_py = 10 ** 6                                            # past the key-weight switch: the learned path, as in most of a run
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-6)

    def step(b):
        with torch.autocast("cuda", torch.bfloat16):
            loss, info = loss_fn(head, b)
        if not torch.isfinite(loss):
            raise SystemExit("non-finite")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        float(torch.nn.utils.clip_grad_norm_(head.parameters(), 3.0))
        opt.step()
        float(loss)
        return loss
    return step


def timed(step, it, n):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        loss = step(next(it))
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3, float(loss)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    assert mode in ("gate", "pair"), mode
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(R, "runs", "eval", "token_step_bench")
    assert torch.cuda.is_available(), "a step benchmark needs the GPU (and a quiet one)"
    res = dict(mode=mode, code=code_state(), host_load=os.getloadavg()[0], gpu_used_mb_before=round(torch.cuda.mem_get_info()[1] / 1e6 - torch.cuda.mem_get_info()[0] / 1e6), date=time.strftime("%Y-%m-%d %H:%M:%S"))

    def run(name, step, it, n=120, warm=20):
        timed(step, it, warm)
        torch.cuda.reset_peak_memory_stats()
        ms, loss = timed(step, it, n)
        res[name] = dict(ms_per_step=round(ms, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), loss_end=round(loss, 4))
        print(name, res[name], flush=True)

    keep = JK.jtoken_loss
    if mode == "gate":
        lf = JT.make_loss(NoRun(), 0.25, "sum", (), 1.0)
        ds, dl = LP.loader("train", 0, 8, 2, ident="e292_a")
        run("plain_cpu_loader_w2", make_step(JT.build_head(dict(JT.ARCH_DEFAULTS)), lambda m, b: lf(m, b)), endless(dl))
        del dl
        JK.jtoken_loss = JF.fast_loss
        lf2 = JT.make_loss(NoRun(), 0.25, "sum", (), 1.0)
        ds, dl = LP.loader("train", 0, 8, 2, ident="e292_a")
        run("fast_head_cpu_loader_w2", make_step(JF.FastJTokenHead(), lambda m, b: lf2(m, b)), endless(dl))
        del dl
        t = time.time()
        gds, gdl = LG.loader("train", 0, 8, 0, ident="e292_a", dev=dev)
        res.update(gpu_loader_load_s=round(time.time() - t, 1), gpu_loader_frames=len(gds), gpu_loader_pairs_P=gds.P, gpu_resident_GB=round(torch.cuda.memory_allocated() / 1e9, 2))
        run("fast_head_gpu_loader", make_step(JF.FastJTokenHead(), lambda m, b: lf2(m, b)), endless(gdl))
        run("fast_head_gpu_loader_compile", make_step(JF.FastJTokenHead().compile_pairs(), lambda m, b: lf2(m, b)), endless(gdl), n=120, warm=40)
        # THE GATE: the exact path the card's arm runs (fast head + fast loss + GPU loader, no compile) against the plain trainer's step (plain head + plain loss + CPU loader, 2 workers), ABBA in one process
        pl, pdl = LP.loader("train", 0, 8, 2, ident="e292_a")

        def with_loss(fn, lf_):                                             # `make_loss` resolves JK.jtoken_loss at call time: set it for each call (the plain step must not run the fast loss)
            def call(m, b):
                JK.jtoken_loss = fn
                return lf_(m, b)
            return call
        ab = {"plain": (make_step(JT.build_head(dict(JT.ARCH_DEFAULTS)), with_loss(keep, lf)), endless(pdl)), "fast": (make_step(JF.FastJTokenHead(), with_loss(JF.fast_loss, lf2)), endless(gdl))}
        for k in ab:
            timed(*ab[k], 30)
        torch.cuda.reset_peak_memory_stats()
        blk = {"plain": [], "fast": []}
        for k in ("plain", "fast", "fast", "plain"):
            blk[k].append(timed(*ab[k], 60)[0])
        mp, mf = sum(blk["plain"]) / 2, sum(blk["fast"]) / 2
        res["gate_abba"] = dict(plain_ms_blocks=[round(v, 1) for v in blk["plain"]], fast_ms_blocks=[round(v, 1) for v in blk["fast"]], plain_ms=round(mp, 1), fast_ms=round(mf, 1), ratio_plain_over_fast=round(mp / mf, 3),
                                peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        print("gate_abba", res["gate_abba"], flush=True)
        t = time.time()
        it = endless(gdl)
        for _ in range(100):
            next(it)
        torch.cuda.synchronize()
        res["gpu_loader_ms_per_batch"] = round((time.time() - t) / 100 * 1e3, 2)
    else:
        JK.jtoken_loss = JF.fast_loss
        lf2 = JT.make_loss(NoRun(), 0.25, "sum", (), 1.0)
        arms = {}
        for K, root in ((512, None), (384, "runs/e383_cache")):
            ds, dl = LG.loader("train", 0, 8, 0, ident="e292_a", K=K, dev=dev, **({"cache": root} if root else {}))
            arms[K] = (make_step(JF.FastJTokenHead(), lambda m, b: lf2(m, b)), endless(dl))
        for K in arms:
            timed(*arms[K], 30)
        blocks = {512: [], 384: []}
        for K in (512, 384, 384, 512):                                  # ABBA
            blocks[K].append(timed(*arms[K], 60)[0])
        res.update({f"K{K}_ms_blocks": [round(v, 1) for v in vs] for K, vs in blocks.items()})
        m512, m384 = sum(blocks[512]) / 2, sum(blocks[384]) / 2
        res.update(K512_ms=round(m512, 1), K384_ms=round(m384, 1), ratio_384_over_512=round(m384 / m512, 3), gpu_resident_GB=round(torch.cuda.memory_allocated() / 1e9, 2))
    JK.jtoken_loss = keep
    print(json.dumps(res, indent=1))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, f"bench_{mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
