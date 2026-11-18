"""scratch (report chat; named by the card E383, run through run.py): seconds per training step of the plain path against the speed path on the real TRAIN cache of B' seed 0.
    mode gate : K 512, plain (CPU loader, workers 2) / fast head + CPU loader / fast + GPU loader / fast + GPU loader + compile_pairs      -> temp/e383/bench_gate.json
    mode pair : the speed path at K 512 and at K 384 (the cache root runs/e383_cache) in ONE process, blocks ABBA of 60 steps, eager     -> temp/e383/bench_pair.json
usage: _py14.sh bench_fast.py gate|pair ; 2 CPU threads, one GPU job below 4 GB. The step is the trainer's (autocast bf16, clip 3.0, AdamW) with a float() of the loss and the gradient norm each step, in every variant alike."""
import json
import os
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

mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
dev = "cuda"
res = dict(mode=mode, host_load=os.getloadavg()[0])
OUT = os.path.join(R, "temp", "e383", f"bench_{mode}.json")


class NoRun:
    def log(self, *a, **k):
        pass


def endless(loader):
    while True:
        for b in loader:
            yield b


def make_step(head, loss_fn):
    head.to(dev).train()
    head._step_py = 10 ** 6
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-6)

    def step(b):
        with torch.autocast("cuda", torch.bfloat16):
            loss, info = loss_fn(head, b)
        if not torch.isfinite(loss):
            raise SystemExit("non-finite")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 3.0)
        float(gn)
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


def run(name, step, it, n=120, warm=20):
    timed(step, it, warm)
    torch.cuda.reset_peak_memory_stats()
    ms, loss = timed(step, it, n)
    res[name] = dict(ms_per_step=round(ms, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), loss_end=round(loss, 4))
    print(name, res[name], flush=True)


keep = JK.jtoken_loss
if mode == "gate":
    args = dict(JT.ARCH_DEFAULTS)
    lf = JT.make_loss(NoRun(), 0.0, 0, 0.25, "none", "sum", None, (), 1.0)
    ds, dl = LP.loader("train", 0, 8, 2, ident="e292_a")
    run("plain_cpu_loader_w2", make_step(JT.build_head(args), lambda m, b: lf(m, b)), endless(dl))
    del dl
    JK.jtoken_loss = JF.fast_loss
    lf2 = JT.make_loss(NoRun(), 0.0, 0, 0.25, "none", "sum", None, (), 1.0)
    ds, dl = LP.loader("train", 0, 8, 2, ident="e292_a")
    run("fast_head_cpu_loader_w2", make_step(JF.FastJTokenHead(), lambda m, b: lf2(m, b)), endless(dl))
    del dl
    t = time.time()
    gds, gdl = LG.loader("train", 0, 8, 0, ident="e292_a", dev=dev)
    res.update(gpu_loader_load_s=round(time.time() - t, 1), gpu_loader_frames=len(gds), gpu_loader_pairs_P=gds.P, gpu_resident_GB=round(torch.cuda.memory_allocated() / 1e9, 2))
    run("fast_head_gpu_loader", make_step(JF.FastJTokenHead(), lambda m, b: lf2(m, b)), endless(gdl))
    run("fast_head_gpu_loader_compile", make_step(JF.FastJTokenHead().compile_pairs(), lambda m, b: lf2(m, b)), endless(gdl), n=120, warm=40)
    t = time.time()
    it = endless(gdl)
    for _ in range(100):
        next(it)
    torch.cuda.synchronize()
    res["gpu_loader_ms_per_batch"] = round((time.time() - t) / 100 * 1e3, 2)
else:
    JK.jtoken_loss = JF.fast_loss
    lf2 = JT.make_loss(NoRun(), 0.0, 0, 0.25, "none", "sum", None, (), 1.0)
    arms = {}
    for K, root in ((512, None), (384, "runs/e383_cache")):
        kw = dict(cache=os.path.join(R, root)) if root else {}
        ds, dl = LG.loader("train", 0, 8, 0, ident="e292_a", K=K, dev=dev, **kw)
        arms[K] = (make_step(JF.FastJTokenHead(), lambda m, b: lf2(m, b)), endless(dl))
    for K in arms:
        timed(*arms[K], 30)                                                   # warm-up of both
    blocks = {512: [], 384: []}
    for K in (512, 384, 384, 512):                                            # ABBA
        blocks[K].append(timed(*arms[K], 60)[0])
    res.update({f"K{K}_ms_blocks": [round(v, 1) for v in vs] for K, vs in blocks.items()})
    m512, m384 = sum(blocks[512]) / 2, sum(blocks[384]) / 2
    res.update(K512_ms=round(m512, 1), K384_ms=round(m384, 1), ratio_384_over_512=round(m384 / m512, 3), gpu_resident_GB=round(torch.cuda.memory_allocated() / 1e9, 2))
JK.jtoken_loss = keep
print(json.dumps(res, indent=1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
