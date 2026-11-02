"""Milliseconds per module: where does the time of one frame go? Batch 1, the cached chain, the dts (-0.5, 0, +0.5) of `benchmark()`; and for training the
forward / backward share of one micro-batch.

    python evaluation/bev_timing.py RUN [--n 30] [--warmup 5] [--device cuda] [--out FILE] [--ckpt-dir DIR] [--force]     (also: bev_suite.py RUN timing ...)
    python evaluation/bev_timing.py --selftest                                                                           (toy modules with known sleeps, CPU)

How: a forward pre-hook and a forward hook on every direct child module of the model (`bev_modules.chain_modules`: lift AND lift2 of a dual camera, lane, ...; the
trunk through its `.features()`), with a CUDA synchronisation around each call. A module called inside another one (a head that calls a shared mover) is
attributed EXCLUSIVELY (its time is taken off the caller), a module called several times per frame (the heads: once per dt) is reported per call AND per frame
(the sum of its calls). The sum of the modules is printed next to the unsynchronised total and the difference is `unattributed_ms`: host-side work outside every
module (precompute, rasterising the radar, decoding) plus the synchronisation overhead (every sync stalls the queue: the sum of a small-kernel chain is a
little ABOVE its fused time on the GPU, while the host work makes the total larger): read the split as shares, not as an exact budget.
A job that is merely QUEUED (a run.py launcher waiting for its GPU slot) is not seen, and when a slot frees it starts under a running timing: time through a GPU
slot (as `temp/t16/eval_chain.sh` does) or look at the schedule first.
This is the fp32 path of `BEVSubject.benchmark()` with a cold memory (no state carried in) and without the ego correction; a run trained with cfg.amp != off or
a warm-memory deployment is not what is timed here (a warning says so). Training: the loss of a micro-batch (forward) and its backward, median of 8 with min / max.

IDLE GPU ONLY. The GPU utilisation (max of 3 samples), the memory in use, the other GPU processes (`nvidia-smi` shows none under WSL2, so `ps` is read too: a python
process running a trainer / evaluation / probe script on the GPU counts, a run.py launcher waiting for a slot does not), the load average are stored with the numbers (`idle`) and a busy GPU is refused - before the model is loaded AND again before
the timed loop - unless --force (the result is then marked not idle and must not be quoted). Result: RUN/eval/timing_modules.json (default).
`bev_suite.py RUN time` gives the headline latency of the whole chain (eval/latency.json); this is its split by module."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evaluation.bev_modules import chain_modules, wrap_method  # noqa: E402

GPU_BASELINE_MB = 2000                       # desktop apps hold ~1.3 GB with no job
GPU_SCRIPTS = {"train.py": "cuda", "train_motion_block.py": "cuda", "bev_suite.py": "cuda", "bev_runner.py": "cuda", "bev_timing.py": "cuda",
               "bev_prototype.py": "cuda", "cache_bevfusion_features.py": "cuda", "bev_modules.py": "cpu"}        # script -> the device it uses when --device is not given
SUITE_OPTIONS_WITH_VALUE = ("--device", "--workers", "--split", "--ckpt-dir", "--out", "--control", "--state")    # bev_suite.py's global options that take a value


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def time_modules(model, forward, device="cpu", n=30, warmup=5, modules=None):
    """-> {"total_ms": median of the unsynchronised forward, "per_call_ms": {module: median exclusive time per call}, "per_frame_ms": {module: median over frames of the sum
    of its calls}, "calls": {module: calls per frame}, "unattributed_ms": total - sum of the modules}. forward() runs one frame (no_grad is the caller's)."""
    cur, frames, stack = defaultdict(float), defaultdict(list), []
    per_call, calls = defaultdict(list), defaultdict(int)

    def begin(name):
        sync(device)
        stack.append([name, 0.0, time.perf_counter()])

    def end():
        sync(device)
        name, child_ms, t0 = stack.pop()
        incl = (time.perf_counter() - t0) * 1e3
        if stack:
            stack[-1][1] += incl                                        # the caller's exclusive time excludes this call
        per_call[name].append(incl - child_ms)
        cur[name] += incl - child_ms
        calls[name] += 1
    hooks = []
    for name in (chain_modules(model) if modules is None else modules):
        m = getattr(model, name, None)
        if m is not None and name != "trunk":
            hooks += [m.register_forward_pre_hook(lambda mod, inp, n=name: begin(n)),
                      m.register_forward_hook(lambda mod, inp, out: end() and None)]       # returns None: a hook that returns a value would REPLACE the output
    trunk = getattr(model, "trunk", None)
    restore = None
    if trunk is not None and hasattr(trunk, "features"):
        def wrapper(original):
            def timed(*args, **kw):                                     # the trunk is called through .features(), not forward
                begin("trunk")
                try:
                    return original(*args, **kw)
                finally:
                    end()
            return timed
        restore = wrap_method(trunk, "features", wrapper)
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
        if restore is not None:
            restore()
    tot = []
    for _ in range(n):
        sync(device)
        t = time.perf_counter()
        forward()
        sync(device)
        tot.append((time.perf_counter() - t) * 1e3)
    total = statistics.median(tot)
    per_frame = {k: statistics.median(v) for k, v in frames.items() if v}
    return {"total_ms": total, "per_call_ms": {k: statistics.median(v) for k, v in per_call.items() if v}, "per_frame_ms": per_frame,
            "calls": {k: calls[k] // n for k in calls if calls[k]}, "unattributed_ms": total - sum(per_frame.values())}


def train_split(loss_fn, zero_grad, device="cpu", n=8, warmup=5):
    """forward + loss and backward of one micro-batch (ms): median, min and max of n. loss_fn() -> the total loss tensor (grad enabled), zero_grad() clears every gradient"""
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
    return {"forward_loss": statistics.median(fw), "backward": statistics.median(bw), "forward_loss_min_max": [min(fw), max(fw)], "backward_min_max": [min(bw), max(bw)]}


def holds_gpu(args):
    """does this command line (`ps -eo args`) belong to a process that holds the GPU? Only a python interpreter RUNNING one of GPU_SCRIPTS does (the script is the first
    token ending in .py, or the module of `-m`; `-W ignore` / `-X dev` values are not mistaken for it), on the device it was given (--device, else the script's default;
    bev_suite's `modules` command defaults to the CPU). A waiting run.py launcher (queued for a GPU slot, its own arguments name train.py and run_ar.sh), a bash wrapper,
    `python -c` and a shell are not users of the GPU."""
    tok = args.split()
    while tok and (os.path.basename(tok[0]) == "env" or "=" in tok[0]):         # `env VAR=1 python ...`
        tok = tok[1:]
    if not tok or not os.path.basename(tok[0]).startswith("python"):
        return False
    opts = tok[1:]
    if "-c" in opts[:4]:
        return False
    if "-m" in opts[:4]:
        i = opts.index("-m")
        script, after = opts[i + 1].split(".")[-1] + ".py" if i + 1 < len(opts) else "", opts[i + 2:]
    else:
        i = next((k for k, t in enumerate(opts) if t.endswith(".py")), None)
        script, after = (os.path.basename(opts[i]), opts[i + 1:]) if i is not None else ("", [])
    if script not in GPU_SCRIPTS:
        return False
    device = GPU_SCRIPTS[script]
    m = re.search(r"--device(?:=|\s+)(\S+)", args)
    if m:
        device = m.group(1)
    elif script == "bev_suite.py":
        positional, k = [], 0
        while k < len(after) and len(positional) < 2:
            if after[k].startswith("--"):
                k += 2 if after[k] in SUITE_OPTIONS_WITH_VALUE else 1
            else:
                positional.append(after[k])
                k += 1
        if len(positional) == 2 and positional[1] == "modules":
            device = "cpu"
    return device.startswith("cuda")


def busy_processes(ps_text, own):
    """the processes of a `ps -eo pid,args` listing that hold the GPU (holds_gpu), except the pids in `own`"""
    out = []
    for ln in ps_text.splitlines()[1:]:
        parts = ln.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) not in own and holds_gpu(parts[1]):
            out.append(f"{parts[0]} {parts[1][:100]}")
    return out


def gpu_state(samples=3, gap=0.5):
    """{"util_percent", "mem_used_mb" (max of `samples`), "other_processes" (nvidia-smi), "busy_processes" (ps), "load"} of this machine; None where a tool is missing"""
    st = {"util_percent": None, "mem_used_mb": None, "other_processes": [], "busy_processes": [],
          "load": open("/proc/loadavg").read().split()[:3] if os.path.exists("/proc/loadavg") else None}
    try:
        utils, mems = [], []
        for i in range(samples):
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
            u, m = (float(x) for x in q.strip().splitlines()[0].split(","))
            utils.append(u)
            mems.append(m)
            if i + 1 < samples:
                time.sleep(gap)
        st["util_percent"], st["mem_used_mb"] = max(utils), max(mems)
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        st["other_processes"] = [ln.strip() for ln in p.splitlines() if ln.strip()]
    except Exception:                                                   # noqa: BLE001  (no nvidia-smi: unknown)
        pass
    try:
        st["busy_processes"] = busy_processes(subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=20).stdout, {os.getpid(), os.getppid()})
    except Exception:                                                   # noqa: BLE001
        pass
    return st


def is_idle(state, device, check_memory=True):
    """a CPU run is always idle; a GPU run needs a known utilisation < 10 %, memory in use below the desktop baseline (not after our own model is loaded:
    check_memory=False), no compute process and no trainer / evaluation process. Unknown (no nvidia-smi) is not idle."""
    if not str(device).startswith("cuda"):
        return True
    if state["util_percent"] is None or state["util_percent"] >= 10 or state["other_processes"] or state["busy_processes"]:
        return False
    return state["mem_used_mb"] is not None and (not check_memory or state["mem_used_mb"] < GPU_BASELINE_MB)


def why_busy(state):
    return (f"utilisation {state['util_percent']} %, memory in use {state['mem_used_mb']} MB (baseline < {GPU_BASELINE_MB}), GPU processes {state['other_processes'] or 'none listed'}, "
            f"trainer / evaluation processes {state['busy_processes'] or 'none'}")


def report(res):
    t = res["inference_ms"]["total_ms"]
    s = sum(res["inference_ms"]["per_frame_ms"].values())
    print(f"\nINFERENCE, batch 1, dts (-0.5, 0, 0.5), cold memory, median of {res['n']} frames (ms): total {t:.1f}, sum of the modules {s:.1f}, "
          f"unattributed {res['inference_ms']['unattributed_ms']:+.1f} (host work outside the modules, sync overhead)")
    for k, v in sorted(res["inference_ms"]["per_frame_ms"].items(), key=lambda kv: -kv[1]):
        c = res["inference_ms"]["calls"][k]
        print(f"  {k:14s} {v:8.2f} ms per frame  ({c} call(s), {res['inference_ms']['per_call_ms'][k]:.2f} ms per call, {100 * v / s:4.1f} % of the module sum, "
              f"{res['params'].get(k, 0) / 1e6:.2f} M parameters)")
    if res.get("train_microbatch_ms"):
        tm = res["train_microbatch_ms"]
        print(f"\nTRAINING micro-batch of 1 sample (forward + loss / backward), median of 8 (ms): {tm['forward_loss']:.1f} / {tm['backward']:.1f}  "
              f"(min-max {tm['forward_loss_min_max'][0]:.1f}-{tm['forward_loss_min_max'][1]:.1f} / {tm['backward_min_max'][0]:.1f}-{tm['backward_min_max'][1]:.1f})")
    print("load", res["state"]["load"], "|", why_busy(res["state"]), "->", "idle" if res["idle"] else "NOT IDLE: do not quote these numbers")


def run_timing(run, n=30, warmup=5, device="cuda", out=None, force=False, ckpt_dir=None):
    run = os.path.abspath(run)
    state = gpu_state()
    idle = is_idle(state, device)
    if not idle and not force:
        raise SystemExit(f"refused: the GPU is not idle ({why_busy(state)}); timing it would disturb the other job and the numbers would be wrong. "
                         "Run it when nothing else uses the GPU, or --force (the result is then marked not idle)")
    from evaluation import bev_adapter as A
    A.setup_paths(run)
    src, subj, _ = A.build(run, device=device, split="val", workers=0, ckpt_dir=ckpt_dir)
    ns, model, loss, cfg = subj.ns, subj.model, subj.loss, subj.cfg
    T, dev = ns.T, subj.device
    if str(device).startswith("cuda") and not str(dev).startswith("cuda"):
        print(f"WARNING: cuda was requested but the model runs on {dev}: these are CPU timings", flush=True)
        idle = False
    if getattr(cfg, "amp", "off") != "off":
        print(f"WARNING: the run was trained with amp={cfg.amp}; this times the fp32 path of benchmark()", flush=True)
    state2 = gpu_state(samples=2)                                      # the first check was minutes ago (the model load): look again before the timed loop
    idle2 = is_idle(state2, device, check_memory=False)
    if not idle2 and not force:
        raise SystemExit(f"refused before the timed loop: the GPU got busy while the model was loading ({why_busy(state2)})")
    idle = idle and idle2
    fr = src.frames(src.scenes()[0])
    batch = T.to_device(src.load(fr[len(fr) // 2]).inputs, dev)
    print("run", run, "step", subj.step, "device", dev, "load", state["load"], flush=True)
    dts = (-0.5, 0.0, 0.5)
    model.eval()

    def forward():
        with torch.no_grad():
            model(batch, dts=dts)
    inf = time_modules(model, forward, dev, n, warmup)
    buffers = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches_tracked" in k}
    model.train()                                                       # a training-mode forward updates the BatchNorm statistics: put them back afterwards
    try:
        def clear():
            model.zero_grad(set_to_none=True)
            loss.zero_grad(set_to_none=True)
        train = train_split(lambda: T.forward_loss(model, loss, batch, 1.0, dev, cfg.amp, cfg, train_terms=True)[0]["total"], clear, dev)
    finally:
        model.eval()
        model.load_state_dict({**model.state_dict(), **buffers})
    params = {k: sum(p.numel() for p in getattr(model, k).parameters()) for k in chain_modules(model) if hasattr(getattr(model, k), "parameters")}
    res = {"run": os.path.basename(run), "step": subj.step, "device": str(dev), "n": n, "idle": idle, "state": state, "state_before_loop": state2, "inference_ms": inf,
           "params": params, "train_microbatch_ms": train, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if str(dev).startswith("cuda") else None}
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
            return x + 1

    class Trunk(nn.Module):
        def features(self, x):
            time.sleep(0.015)
            return x + 1

    class Outer(nn.Module):                                             # a module that calls another probed module (like a head calling a shared mover)
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            time.sleep(0.005)
            return self.inner(x)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift, self.radar, self.sem_head, self.trunk = Sleep(10), Sleep(20), Sleep(5), Trunk()
            self.mover = Sleep(20)
            self.wrapper = Outer(self.mover)                            # `mover` is a child of the model too: 20 ms inside a 5 ms caller

        def forward(self, x):
            x = self.trunk.features(self.radar(self.lift(x)))
            for _ in range(3):                                          # a head called once per dt
                self.sem_head(x)
            return self.wrapper(x)
    m = Toy()
    plain = m.trunk.features
    reference = m(torch.zeros(1))
    res = time_modules(m, lambda: m(torch.zeros(1)), "cpu", n=8, warmup=2)
    pf, pc, calls = res["per_frame_ms"], res["per_call_ms"], res["calls"]
    near = lambda v, want: want <= v < want * 1.6 + 4        # noqa: E731  (a sleep never returns early; the overhead is small but a loaded machine adds some)
    assert near(pc["lift"], 10) and near(pc["radar"], 20) and near(pc["trunk"], 15) and near(pc["sem_head"], 5), pc
    assert calls["sem_head"] == 3 and calls["lift"] == 1 and calls["trunk"] == 1, calls
    assert near(pf["sem_head"], 15) and abs(pf["sem_head"] - 3 * pc["sem_head"]) < 4, "per frame = the sum of the calls"
    assert near(pc["wrapper"], 5) and pc["wrapper"] < 12 and near(pc["mover"], 20) and pc["mover"] < 30, ("exclusive: the caller keeps only its own 5 ms", pc)
    assert pf["radar"] > pf["trunk"] > pf["lift"] and abs(res["unattributed_ms"]) < 0.25 * res["total_ms"] + 5, (pf, res)
    assert abs(res["total_ms"] - sum(pf.values()) - res["unattributed_ms"]) < 1e-6
    assert m.trunk.features == plain and "features" not in m.trunk.__dict__ and not any(mod._forward_hooks or mod._forward_pre_hooks for mod in m.children()), "hooks and wrapper removed"
    assert torch.equal(m(torch.zeros(1)), reference), "the timing hooks do not change the output"
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
    assert near(ts["forward_loss"], 6) and near(ts["backward"], 9) and ts["forward_loss_min_max"][0] <= ts["forward_loss"] <= ts["forward_loss_min_max"][1], ts
    # the idle rule: GPU utilisation, memory, processes (nvidia-smi lists none under WSL2: ps is read), unknown is busy; a CPU run is idle
    ok = {"util_percent": 2.0, "mem_used_mb": 1300.0, "other_processes": [], "busy_processes": []}
    assert is_idle(ok, "cuda") and is_idle({**ok, "util_percent": 99}, "cpu")
    for bad in ({"util_percent": 99.0}, {"util_percent": None}, {"mem_used_mb": 11800.0}, {"mem_used_mb": None}, {"other_processes": ["1, 5000 MiB"]}, {"busy_processes": ["7 python train.py"]}):
        assert not is_idle({**ok, **bad}, "cuda"), bad
    assert is_idle({**ok, "mem_used_mb": 11800.0}, "cuda", check_memory=False), "after our own model is loaded the memory says nothing"
    ps = "  PID COMMAND\n  101 /usr/bin/python -u runs/t16/code/train.py cfg\n  102 python evaluation/bev_suite.py runs/x timing\n  103 bash -c sleep 5\n  104 python other.py\n"
    assert [x.split()[0] for x in busy_processes(ps, {102})] == ["101"] and busy_processes(ps, {101, 102}) == [], "ps: trainers / evaluations, not ourselves"
    # queued jobs are not GPU users (the bug: a run.py launcher waiting for a slot, whose arguments name train.py / run_ar.sh / bev_suite, refused the timing on an idle GPU)
    waiting = ("  PID COMMAND\n"
               "  201 python3 /mnt/c/Auto-research/autoresearch/tools/run.py --id t17.data --gpu --split train --hyp more scenes -- bash temp/train/run_ar.sh temp/compose/t17_data.json\n"
               "  202 python3 ../autoresearch/tools/run.py --id t17.data.eval --gpu --split val --no-card --after t17.data -- bash temp/t16/eval_chain.sh runs/t17_data\n"
               "  203 bash temp/train/run_ar.sh temp/compose/t17_data.json\n"
               "  204 python3 ../autoresearch/tools/run.py --id x --gpu -- python evaluation/bev_suite.py runs/x eval --n 10\n")
    assert busy_processes(waiting, set()) == [], busy_processes(waiting, set())
    running = ("  PID COMMAND\n  301 /home/ml/miniconda3/envs/gcar/bin/python -u runs/t17_data/code/train.py temp/compose/t17_data.json\n"
               "  302 python -u evaluation/bev_suite.py runs/t17_data eval --n 1000\n  303 python -u evaluation/bev_suite.py runs/t17_data --device cpu eval --n 6\n"
               "  304 python evaluation/bev_suite.py runs/t17_data modules --frames 3\n  305 python evaluation/bev_suite.py runs/t17_data --device cuda modules\n  306 python -c import train.py\n")
    assert [x.split()[0] for x in busy_processes(running, set())] == ["301", "302", "305"], busy_processes(running, set())
    for line, want in (("python3.12 -u runs/x/code/train.py cfg.json", True), ("python -W ignore runs/x/code/train.py cfg.json", True), ("python -X dev -u train.py cfg", True),
                       ("env CUDA_VISIBLE_DEVICES=0 python train.py cfg", True), ("python -m evaluation.bev_suite runs/x eval --n 5", True),
                       ("python -m evaluation.bev_suite runs/x --device=cpu eval --n 5", False), ("python evaluation/bev_suite.py runs/x eval --n 5 --out modules", True),
                       ("python evaluation/bev_suite.py runs/x --workers 0 modules", False), ("python evaluation/bev_suite.py runs/modules eval", True),
                       ("python evaluation/bev_suite.py runs/x --curves --fusion eval", True), ("python evaluation/bev_suite.py runs/x --device=cuda modules", True),
                       ("python evaluation/bev_suite.py runs/x --device cuda modules", True), ("python runner/bev_runner.py full runs/x", True),
                       ("python runner/bev_runner.py full runs/x --device cpu", False), ("python evaluation/bev_timing.py runs/x", True), ("python evaluation/bev_modules.py runs/x", False),
                       ("python evaluation/bev_modules.py runs/x --device cuda", True), ("python evaluation/bev_prototype.py runs/x", True),
                       ("python -c train.py", False), ("python -m json.tool train.py", False), ("python3 tools/run.py --id x --gpu -- python train.py", False),
                       ("bash temp/train/run_ar.sh cfg train.py", False), ("python other.py train.py", False)):
        assert holds_gpu(line) == want, (line, want)
    print("bev_timing selftest ok: per call / per frame / exclusive milliseconds of toy modules with known sleeps, hooks removed and output unchanged, training split, idle rule")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="the folder of a moved checkpoint")
    ap.add_argument("--force", action="store_true", help="time even when the GPU is not idle (the result says so and must not be quoted)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.run:
        ap.error("RUN is required (or --selftest)")
    run_timing(a.run, a.n, a.warmup, a.device, a.out, a.force, a.ckpt_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
