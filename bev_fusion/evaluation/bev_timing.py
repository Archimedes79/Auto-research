"""Milliseconds per module: where does the time of one frame go? Batch 1, the cached chain, the dts (-0.5, 0, +0.5) the evaluation uses; and for training the
forward / backward share of one micro-batch.

    python evaluation/bev_timing.py RUN [--n 30] [--warmup 5] [--device cuda] [--out FILE] [--force]     (also: bev_suite.py RUN timing ...)
    python evaluation/bev_timing.py --selftest                                                             (toy modules with known sleeps, CPU)

How: a forward pre-hook and a forward hook on every module of `bev_modules.MODULES` with a CUDA synchronisation around each call, so the split is exact and the
sum is a little above the unsynchronised total (printed next to it). The trunk is called through `.features()`, which is wrapped. A module called several times
per frame (the heads: once per dt) is reported per call AND per frame (the sum of its calls; the module sum then adds up to about the total). Training: the
loss of a micro-batch (forward) and its backward, median of 8.

IDLE GPU ONLY. The load average, the GPU utilisation and the other GPU processes are printed with the numbers and stored (`idle`); on a busy GPU the run is
refused (it would disturb the other job and the numbers would be wrong) unless --force. Result: RUN/eval/timing_modules.json (default).
The existing `bev_suite.py RUN time` gives the headline latency of the whole chain (eval/latency.json); this is its split by module."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evaluation.bev_modules import MODULES  # noqa: E402


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def time_modules(model, forward, device="cpu", n=30, warmup=5, modules=MODULES):
    """-> {"total_ms": median of the unsynchronised forward, "per_call_ms": {module: median per call}, "per_frame_ms": {module: median of the sum of its calls
    in one frame}, "calls": {module: calls per frame}}. forward() runs one frame of the model (no_grad is the caller's)."""
    cur, frames, t0 = defaultdict(float), defaultdict(list), {}
    per_call, calls = defaultdict(list), defaultdict(int)
    hooks = []

    def pre(name):
        def f(mod, inputs):
            sync(device)
            t0[name] = time.perf_counter()
        return f

    def post(name):
        def f(mod, inputs, output):                                     # returns None: a hook that returns a value would REPLACE the output
            sync(device)
            dt = (time.perf_counter() - t0[name]) * 1e3
            per_call[name].append(dt)
            cur[name] += dt
            calls[name] += 1
        return f
    for name in modules:
        m = getattr(model, name, None)
        if m is not None and name != "trunk":
            hooks += [m.register_forward_pre_hook(pre(name)), m.register_forward_hook(post(name))]
    trunk = getattr(model, "trunk", None)
    plain = trunk.features if trunk is not None and hasattr(trunk, "features") else None
    if plain is not None:
        def timed(*args, **kw):                                         # the trunk is called through .features(), not forward
            sync(device)
            t = time.perf_counter()
            out = plain(*args, **kw)
            sync(device)
            dt = (time.perf_counter() - t) * 1e3
            per_call["trunk"].append(dt)
            cur["trunk"] += dt
            calls["trunk"] += 1
            return out
        trunk.features = timed
    try:
        for _ in range(warmup):
            forward()
        per_call.clear()
        calls.clear()
        for _ in range(n):
            cur.clear()
            forward()
            for k, v in cur.items():
                frames[k].append(v)
    finally:
        for h in hooks:
            h.remove()
        if plain is not None:
            trunk.features = plain
    tot = []
    for _ in range(n):
        sync(device)
        t = time.perf_counter()
        forward()
        sync(device)
        tot.append((time.perf_counter() - t) * 1e3)
    return {"total_ms": statistics.median(tot), "per_call_ms": {k: statistics.median(v) for k, v in per_call.items() if v},
            "per_frame_ms": {k: statistics.median(v) for k, v in frames.items() if v}, "calls": {k: calls[k] // n for k in calls if calls[k]}}


def train_split(loss_fn, zero_grad, device="cpu", n=8, warmup=5):
    """forward + loss and backward of one micro-batch, median of n (ms). loss_fn() -> the total loss tensor (grad enabled), zero_grad() clears the gradients"""
    fw, bw = [], []
    for i in range(warmup + n):
        zero_grad()
        sync(device)
        t0 = time.perf_counter()
        loss = loss_fn()
        sync(device)
        t1 = time.perf_counter()
        loss.backward()
        sync(device)
        t2 = time.perf_counter()
        if i >= warmup:
            fw.append((t1 - t0) * 1e3)
            bw.append((t2 - t1) * 1e3)
    return {"forward_loss": statistics.median(fw), "backward": statistics.median(bw)}


def gpu_state():
    """{"util_percent", "other_processes": [...], "load": loadavg} of this machine (nvidia-smi; None when there is no GPU tool)"""
    st = {"util_percent": None, "other_processes": [], "load": open("/proc/loadavg").read().split()[:3] if os.path.exists("/proc/loadavg") else None}
    try:
        u = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        st["util_percent"] = float(u.strip().splitlines()[0])
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        st["other_processes"] = [ln.strip() for ln in p.splitlines() if ln.strip()]
    except Exception:                                                   # noqa: BLE001  (no nvidia-smi: CPU timing)
        pass
    return st


def is_idle(state, device):
    """a CPU run is always 'idle'; a GPU run needs < 10 % utilisation and no other compute process"""
    return True if not str(device).startswith("cuda") else (state["util_percent"] is not None and state["util_percent"] < 10 and not state["other_processes"])


def report(res):
    t, s = res["inference_ms"]["total_ms"], sum(res["inference_ms"]["per_frame_ms"].values())
    print(f"\nINFERENCE, batch 1, dts (-0.5, 0, 0.5), median of {res['n']} frames (ms): total {t:.1f}, sum of the modules {s:.1f} (synchronised around every module)")
    for k, v in sorted(res["inference_ms"]["per_frame_ms"].items(), key=lambda kv: -kv[1]):
        c = res["inference_ms"]["calls"][k]
        print(f"  {k:14s} {v:8.2f} ms per frame  ({c} call(s), {res['inference_ms']['per_call_ms'][k]:.2f} ms per call, {100 * v / s:4.1f} % of the module sum, {res['params'].get(k, 0) / 1e6:.2f} M parameters)")
    if res.get("train_microbatch_ms"):
        print(f"\nTRAINING micro-batch of 1 sample (forward + loss / backward), median of 8 (ms): {res['train_microbatch_ms']['forward_loss']:.1f} / {res['train_microbatch_ms']['backward']:.1f}")
    print("load", res["state"]["load"], "GPU utilisation", res["state"]["util_percent"], "other GPU processes", res["state"]["other_processes"] or "none",
          "-> idle" if res["idle"] else "-> NOT IDLE: do not quote these numbers")


def run_timing(run, n=30, warmup=5, device="cuda", out=None, force=False):
    run = os.path.abspath(run)
    state = gpu_state()
    idle = is_idle(state, device)
    if not idle and not force:
        raise SystemExit(f"refused: the GPU is not idle (utilisation {state['util_percent']} %, processes {state['other_processes']}); timing it would disturb the other job and "
                         "the numbers would be wrong. Run it when nothing else uses the GPU, or --force (the result is then marked not idle)")
    from evaluation import bev_adapter as A
    A.setup_paths(run)
    src, subj, _ = A.build(run, device=device, split="val", workers=0)
    ns, model, loss, cfg = subj.ns, subj.model, subj.loss, subj.cfg
    T, dev = ns.T, subj.device
    fr = src.frames(src.scenes()[0])
    batch = T.to_device(src.load(fr[len(fr) // 2]).inputs, dev)
    print("run", run, "step", subj.step, "device", dev, "load", state["load"], flush=True)
    dts = (-0.5, 0.0, 0.5)
    model.eval()

    def forward():
        with torch.no_grad():
            model(batch, dts=dts)
    inf = time_modules(model, forward, dev, n, warmup)
    model.train()                                                       # (a training-mode forward updates the BatchNorm statistics in memory only; this process ends after the timing)
    try:
        train = train_split(lambda: T.forward_loss(model, loss, batch, 1.0, dev, cfg.amp, cfg, train_terms=True)[0]["total"], lambda: model.zero_grad(set_to_none=True), dev)
    finally:
        model.eval()
    params = {k: sum(p.numel() for p in getattr(model, k).parameters()) for k in MODULES if getattr(model, k, None) is not None}
    res = {"run": os.path.basename(run), "step": subj.step, "device": str(dev), "n": n, "idle": idle, "state": state, "inference_ms": inf, "params": params,
           "train_microbatch_ms": train, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if str(dev).startswith("cuda") else None}
    out = out or os.path.join(run, "eval", "timing_modules.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    report(res)
    print("written", out)
    return res


# ------------------------------------------------------------------------------------------------ self-test
def selftest():
    import torch.nn as nn

    class Sleep(nn.Module):
        def __init__(self, ms):
            super().__init__()
            self.ms = ms

        def forward(self, x):
            time.sleep(self.ms / 1e3)
            return x

    class Trunk(nn.Module):
        def features(self, x):
            time.sleep(0.015)
            return x

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift, self.radar, self.sem_head, self.trunk = Sleep(10), Sleep(20), Sleep(5), Trunk()

        def forward(self, x):
            x = self.trunk.features(self.radar(self.lift(x)))
            for _ in range(3):                                          # a head called once per dt
                self.sem_head(x)
            return x
    m = Toy()
    plain = m.trunk.features
    res = time_modules(m, lambda: m(torch.ones(1)), "cpu", n=8, warmup=2)
    pf, pc, calls = res["per_frame_ms"], res["per_call_ms"], res["calls"]
    near = lambda v, want: want <= v < want * 1.5 + 3        # noqa: E731  (sleep never returns early; the overhead is small)
    assert near(pc["lift"], 10) and near(pc["radar"], 20) and near(pc["trunk"], 15) and near(pc["sem_head"], 5), pc
    assert calls == {"lift": 1, "radar": 1, "trunk": 1, "sem_head": 3}, calls
    assert near(pf["sem_head"], 15) and abs(pf["sem_head"] - 3 * pc["sem_head"]) < 3, "per frame = the sum of the calls"
    assert pf["radar"] > pf["trunk"] > pf["lift"] and sum(pf.values()) <= res["total_ms"] * 1.15 and res["total_ms"] <= sum(pf.values()) * 1.25 + 5, (pf, res["total_ms"])
    assert m.trunk.features == plain and not any(mod._forward_hooks or mod._forward_pre_hooks for mod in (m.lift, m.radar, m.sem_head)), "hooks and wrapper removed"
    # the training split: a loss whose forward takes 6 ms and whose backward takes 9 ms

    class Slow(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            time.sleep(0.006)
            return x * 1.0

        @staticmethod
        def backward(ctx, g):
            time.sleep(0.009)
            return g
    w = torch.ones(1, requires_grad=True)
    ts = train_split(lambda: Slow.apply(w).sum(), lambda: setattr(w, "grad", None), "cpu", n=5, warmup=1)
    assert near(ts["forward_loss"], 6) and near(ts["backward"], 9), ts
    assert is_idle({"util_percent": 99, "other_processes": ["1, 5000 MiB"]}, "cpu") and not is_idle({"util_percent": 99, "other_processes": []}, "cuda") \
        and is_idle({"util_percent": 2, "other_processes": []}, "cuda") and not is_idle({"util_percent": 2, "other_processes": ["1, 5000 MiB"]}, "cuda") \
        and not is_idle({"util_percent": None, "other_processes": []}, "cuda"), "idle rule"
    print("bev_timing selftest ok: per-module and per-frame milliseconds of toy modules with known sleeps, calls per frame, hooks removed, training split, idle rule")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="time even when the GPU is not idle (the result says so and must not be quoted)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.run:
        ap.error("RUN is required (or --selftest)")
    run_timing(a.run, a.n, a.warmup, a.device, a.out, a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
