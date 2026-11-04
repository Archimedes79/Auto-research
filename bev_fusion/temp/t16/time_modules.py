"""scratch of the t16 bring-up, stage S3 (D44): milliseconds per frame of the cached chain at batch 1, split by module (forward pre / post hooks
with a CUDA sync around every module: the split is exact, the sum is a little above the unsynchronised total), and for training the forward /
backward / optimiser share of one micro-batch. Idle GPU only: the load average and the GPU's other processes are printed with the numbers.

    python bev_fusion/temp/t16/time_modules.py runs/t16_small_s0 [--n 30] [--frames 2]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
MODULES = ["lift", "cam_reduce", "cam_reduce2", "cam_merge", "cam_temporal", "radar", "rad_temporal", "fusion", "trunk", "head_reduce",
           "motion_block", "sem_head", "obj_head", "aux_cam", "aux_rad"]


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    from evaluation import bev_adapter as A
    A.setup_paths(run)
    src, subj, task = A.build(run, device=a.device, split="val", workers=0)
    ns, model, loss, cfg = subj.ns, subj.model, subj.loss, subj.cfg
    T, dev = ns.T, subj.device
    fr = src.frames(src.scenes()[0])
    batch = T.to_device(src.load(fr[len(fr) // 2]).inputs, dev)
    print("run", run, "step", subj.step, "device", dev, "load", open("/proc/loadavg").read().split()[:3], flush=True)
    try:
        print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or "no other GPU process")
    except Exception:                                                                  # noqa: BLE001
        pass

    acc = {n: [] for n in MODULES}
    t_start = {}

    def pre(mod, i, n):
        sync(dev)
        t_start[n] = time.perf_counter()

    def post(mod, i, o, n):
        sync(dev)
        acc[n].append((time.perf_counter() - t_start[n]) * 1e3)

    hooks = []
    for n in MODULES:
        m = getattr(model, n, None)
        if m is not None:
            hooks.append(m.register_forward_pre_hook(lambda mod, i, n=n: pre(mod, i, n)))
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=n: post(mod, i, o, n) and None))
    dts = (-0.5, 0.0, 0.5)
    orig_features = model.trunk.features                                       # the trunk is called through .features(), not forward: wrap it

    def timed_features(*args, **kw):
        sync(dev)
        t0 = time.perf_counter()
        out = orig_features(*args, **kw)
        sync(dev)
        acc["trunk"].append((time.perf_counter() - t0) * 1e3)
        return out
    model.trunk.features = timed_features
    with torch.no_grad():
        for _ in range(a.warmup):
            model(batch, dts=dts)
        for k in acc:
            acc[k].clear()
        for _ in range(a.n):
            model(batch, dts=dts)
    for h in hooks:
        h.remove()
    model.trunk.features = orig_features
    per = {k: statistics.median(v) for k, v in acc.items() if v}
    calls = {k: len(v) // a.n for k, v in acc.items() if v}
    # the unsynchronised total, as the evaluation's `time` measures it
    tot = []
    with torch.no_grad():
        for _ in range(a.n):
            sync(dev)
            t0 = time.perf_counter()
            model(batch, dts=dts)
            sync(dev)
            tot.append((time.perf_counter() - t0) * 1e3)
    total = statistics.median(tot)
    s = sum(per.values())
    print(f"\nINFERENCE, batch 1, dts {dts}, median of {a.n} (ms): total {total:.1f}, sum of the modules {s:.1f}")
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:8.2f} ms  ({calls[k]} call(s) per frame, {100 * v / s:4.1f} % of the module sum)")
    params = {k: sum(p.numel() for p in getattr(model, k).parameters()) for k in MODULES if getattr(model, k, None) is not None}
    # training micro-batch: forward / backward of the loss on this batch (batch 1, one clip frame) for a rough split
    model.train()
    fw, bw = [], []
    for i in range(a.warmup + 8):
        model.zero_grad(set_to_none=True)
        sync(dev)
        t0 = time.perf_counter()
        out, res, _ = T.forward_loss(model, loss, batch, 1.0, dev, cfg.amp, cfg, train_terms=True)
        sync(dev)
        t1 = time.perf_counter()
        out["total"].backward()
        sync(dev)
        t2 = time.perf_counter()
        if i >= a.warmup:
            fw.append((t1 - t0) * 1e3)
            bw.append((t2 - t1) * 1e3)
    print(f"\nTRAINING micro-batch of 1 sample (forward + loss / backward), median of 8 (ms): {statistics.median(fw):.1f} / {statistics.median(bw):.1f}")
    print("peak GPU memory (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.startswith("cuda") else None)
    res_json = {"run": os.path.basename(run), "step": subj.step, "inference_ms": {"total": total, "per_module": per, "calls": calls},
                "params": params, "train_microbatch_ms": {"forward_loss": statistics.median(fw), "backward": statistics.median(bw)},
                "load": open("/proc/loadavg").read().split()[:3]}
    out = a.out or os.path.join(run, "eval", "timing_modules.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res_json, open(out, "w"), indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
