"""E351 throw-away probe (scratch, 2026-11-15; the critic's 'twenty lines'): does a CUDA graph of pyramid + maps head (+ B's node head) give the forward back, with bit-identical outputs?
usage: wsl bash tools/env/_py14.sh graph_probe.py [--runs e312_p_s0,e324_s_s0] [--rounds 10] [--iters 30]   (GPU; the host load and the GPU state are printed; bf16 autocast, batch 1, no_grad, cudnn.benchmark False)
The model forward is mirrored stage by stage (as `JEndModel.forward`): front = jmodel(x, joint=False) (+ node head), then the eager pair head `je`. Arms: eager (the model's own forward), g1 (the front as ONE CUDA graph, the pair head eager on its static outputs).
Reports: torch.equal of every tensor of the returned trees (eager against g1) on 24 frames, ms per frame of the front alone and of the whole forward (alternating blocks, paired difference with a bootstrap over the rounds), GPU busy ms per stage (sum of kernel durations from the profiler)."""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends_eval as JE  # noqa: E402
import lane_gt_data as LD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="e312_p_s0,e324_s_s0")
ap.add_argument("--rounds", type=int, default=10)
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()
torch.backends.cudnn.benchmark = False
dev = "cuda"
print("loadavg", open("/proc/loadavg").read().strip(), "| cudnn.benchmark", torch.backends.cudnn.benchmark)
LD.QData.DEFAULT_JUNCTION = True
ds = LD.QData("val_clean", map_mode="error", pixels=False)
xs = [LD.collate([ds[i]])["x"].to(dev).float() for i in np.linspace(0, len(ds) - 1, 24).astype(int)]


def flat(t, prefix=""):
    if torch.is_tensor(t):
        return {prefix: t}
    out = {}
    if isinstance(t, dict):
        for k, v in t.items():
            out.update(flat(v, f"{prefix}/{k}"))
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t):
            out.update(flat(v, f"{prefix}[{i}]"))
    return out


class Probe:
    def __init__(self, run):
        self.m, _ = JE.load_model(run, dev)
        self.xs = torch.zeros_like(xs[0])
        self.g = None

    def front(self, x):                                                              # JEndModel.forward without the pair head
        m = self.m
        P, out = m.jmodel(x, joint=False)
        tok = nd = None
        if m.nodes is not None:
            nd = m.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
            tok = nd["tok"]
        return P, out, tok, nd

    def eager(self, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m(x, gt=None)

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            for _ in range(3):
                self.front(self.xs)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), torch.cuda.graph(self.g):
            self.static = self.front(self.xs)

    def g1(self, x):
        self.xs.copy_(x)
        self.g.replay()
        P, out, tok, nd = self.static
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return P, out, self.m.je(out, P, gt=None, tok=tok, nodes=nd if self.m.je.attach is not None else None)


runs = a.runs.split(",")
pr = {r: Probe(r) for r in runs}
for r in runs:
    p = pr[r]
    for i in range(5):
        p.eager(xs[i])
    p.xs.copy_(xs[0])
    p.capture()
    p.g1(xs[0])
torch.cuda.synchronize()
ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
for r in runs:                                                                       # equality on 24 frames: every tensor of the returned trees (P, out, je out)
    p = pr[r]
    bad, n, ndiff = [], 0, 0.0
    for x in xs:
        e = flat(p.eager(x))
        e = {k: v.clone() for k, v in e.items()}
        g = flat(p.g1(x))
        for k in e:
            n += 1
            if k not in g or not torch.equal(e[k], g[k]):
                bad.append(k)
                if k in g and e[k].is_floating_point():
                    ndiff = max(ndiff, float((e[k].float() - g[k].float()).abs().max()))
    print(f"{r}: eager against g1 on {len(xs)} frames: {n} tensors compared, {len(bad)} not bit-identical; keys {sorted(set(bad))[:8]}; max abs diff of the float ones {ndiff:.3e}")
    null = []
    for x in xs[:6]:
        e1 = {k: v.clone() for k, v in flat(p.eager(x)).items()}
        e2 = flat(p.eager(x))
        null += [bool(torch.equal(e1[k], e2[k])) for k in e1]
    print(f"{r}: NULL control, eager against eager (same process): {sum(null)} of {len(null)} tensors identical")
blocks = {r: {"eager": [], "g1": [], "front_eager": [], "front_g1": []} for r in runs}
order = [(r, k) for r in runs for k in ("eager", "g1")]
for rd in range(a.rounds):
    seq = order if rd % 2 == 0 else list(reversed(order))
    for r, kind in seq:
        p = pr[r]
        ms = []
        for i in range(a.iters):
            x = xs[(rd * a.iters + i) % len(xs)]
            e0, e1 = ev(), ev()
            torch.cuda.synchronize()
            e0.record()
            (p.eager if kind == "eager" else p.g1)(x)
            e1.record()
            torch.cuda.synchronize()
            ms.append(e0.elapsed_time(e1))
        blocks[r][kind].append(float(np.median(ms)))
        fm = []                                                                        # the front alone
        for i in range(a.iters):
            x = xs[(rd * a.iters + i) % len(xs)]
            e0, e1 = ev(), ev()
            torch.cuda.synchronize()
            e0.record()
            if kind == "eager":
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p.front(x)
            else:
                p.xs.copy_(x)
                p.g.replay()
            e1.record()
            torch.cuda.synchronize()
            fm.append(e0.elapsed_time(e1))
        blocks[r]["front_" + kind].append(float(np.median(fm)))


def paired(x, y, n=10000):
    d = np.asarray(x) - np.asarray(y)
    rng = np.random.RandomState(0)
    m = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


print(f"\nms per frame (median of {a.rounds} block medians of {a.iters} forwards), loadavg after {open('/proc/loadavg').read().split()[0]}")
for r in runs:
    b = blocks[r]
    print(f"  {r}: whole forward eager {np.median(b['eager']):.2f}, g1 {np.median(b['g1']):.2f}; front (pyramid + maps (+ node head)) eager {np.median(b['front_eager']):.2f}, graph replay {np.median(b['front_g1']):.2f}")
    m, lo, hi = paired(b["g1"], b["eager"])
    print(f"      paired whole forward g1 - eager {m:+.2f} [{lo:+.2f}, {hi:+.2f}] = {100 * m / np.median(b['eager']):+.1f} %")
    m, lo, hi = paired(b["front_g1"], b["front_eager"])
    print(f"      paired front only   g1 - eager {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
from torch.profiler import ProfilerActivity, profile  # noqa: E402
print("\nGPU busy time (sum of kernel durations of one forward, profiler) and the wall time of that forward:")
for r in runs:
    p = pr[r]
    for kind in ("eager", "g1"):
        with torch.no_grad():
            (p.eager if kind == "eager" else p.g1)(xs[3])
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            (p.eager if kind == "eager" else p.g1)(xs[3])
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000
        busy = sum(e.device_time_total for e in prof.key_averages() if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA) / 1000.0
        print(f"  {r} {kind:6s}: kernel busy {busy:6.2f} ms, wall {wall:6.2f} ms (the profiler adds overhead to the wall time)")
