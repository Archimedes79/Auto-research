"""Milliseconds per module: where does the time of one frame go? Batch 1, the cached chain, the dts (-0.5, 0, +0.5) of `benchmark()`; and for training the
forward / backward share of one micro-batch.

    python -P -m bev_fusion.evaluation.bev_timing RUN [--n 30] [--warmup 5] [--device cuda] [--out FILE] [--ckpt-dir DIR] [--force]
                                                                                                                 (also: `Evaluation.time(RUN)`)

How: a forward pre-hook and a forward hook on every direct child module of the model (`bev_modules.chain_modules`: lift AND lift2 of a dual camera, lane, ...;
the trunk through its `.features()`), with a CUDA synchronisation around each call. A module called inside another one (a head that calls a shared mover) is
attributed EXCLUSIVELY (its time is taken off the caller), a module called several times per frame (the heads: once per dt) is reported per call AND per frame
(the sum of its calls). The sum of the modules is printed next to the unsynchronised total and the difference is `unattributed_ms`: host-side work outside every
module (precompute, rasterising the radar, decoding) plus the synchronisation overhead (every sync stalls the queue: the sum of a small-kernel chain is a
little ABOVE its fused time on the GPU, while the host work makes the total larger): read the split as shares, not as an exact budget.
This is the fp32 path of `BEVSubject.benchmark()` with a cold memory (no state carried in) and without the ego correction; a run trained with cfg.amp != off or
a warm-memory deployment is not what is timed here (a warning says so). Training: the loss of a micro-batch (forward) and its backward, median of 8 with min /
max.

IDLE GPU ONLY. The GPU utilisation (max of 3 samples), the memory in use, the other GPU processes (`nvidia-smi` lists the holders of a CUDA context under WSL2,
this process not counted, and `ps` is read too: a python
process running a GPU step of a research run, a run's worker (also one that waits for its GPU slot: its step would start under the timing) or a GPU tool of this
problem counts, see `holds_gpu`), the load average are stored with the numbers (`idle`) and a busy GPU is refused - before the model is loaded AND again before
the timed loop - unless --force (the result is then marked not idle and must not be quoted). Result: RUN/eval/timing_modules.json (default).
`Evaluation.get_cost` gives the headline latency of the whole chain (ms per frame as the network ran); this is its split by module."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bev_fusion.net import chain
from bev_fusion.net.prep.batches import to_device
from bev_fusion.evaluation import bev_adapter as A
from bev_fusion.evaluation.bev_modules import chain_modules, wrap_method

GPU_BASELINE_MB = 2000                       # desktop apps hold ~1.3 GB with no job
PROBLEMS = os.path.join(ROOT, "..")          # problems/: the steps of their problem.json say which scripts take a GPU
# this problem's GPU tools that are no step -> the device without --device
OWN_GPU_SCRIPTS = {"bev_timing.py": "cuda", "cache_bevfusion_features.py": "cuda", "bev_modules.py": "cpu"}
RESEARCH_MODULES = ("research.evaluate",)    # the core's evaluation step; `-m research worker` (a run's worker) counts too


def gpu_scripts(problems_dir=PROBLEMS):
    """script name -> the device it uses without --device: this problem's GPU tools and every script of a problem step with "gpus" >= 1, started by its
    file or as a module (`-m <package>.<module>`: the module's last name + .py, the form holds_gpu compares); problems/*/problem.json, read only; a
    problem.json that cannot be read is an error, not a script less"""
    out = dict(OWN_GPU_SCRIPTS)
    for path in sorted(glob.glob(os.path.join(problems_dir, "*", "problem.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                steps = json.load(f).get("steps", {})
            for step in steps.values():
                run = step.get("run")
                if isinstance(run, list) and step.get("gpus", 0) >= 1:
                    out.update((os.path.basename(t), "cuda") for t in run if isinstance(t, str) and t.endswith(".py"))
                    out.update((str(module).split(".")[-1] + ".py", "cuda") for flag, module in zip(run, run[1:]) if flag == "-m")
        except (OSError, ValueError, AttributeError, TypeError) as e:
            raise ValueError(f"{path}: the GPU steps cannot be read ({type(e).__name__}: {e})") from None
    return out


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def time_modules(model, forward, device="cpu", n=30, warmup=5, modules=None):
    """-> {"total_ms": median of the unsynchronised forward, "per_call_ms": {module: median exclusive time per call}, "per_frame_ms": {module: median over
    frames of the sum of its calls}, "calls": {module: calls per frame}, "unattributed_ms": total - sum of the modules}. forward() runs one frame (no_grad is
    the caller's)."""
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
    """forward + loss and backward of one micro-batch (ms): median, min and max of n. loss_fn() -> the total loss tensor (grad enabled), zero_grad() clears
    every gradient"""
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


def holds_gpu(args, scripts):
    """does this command line (`ps -eo args`) belong to a process that holds the GPU? A python interpreter running a run's worker (`-m research worker`: it runs
    a step or waits for its GPU slot, and a step that starts under a timing spoils it), the core's evaluation step (`-m research.evaluate`) or one of `scripts`
    (gpu_scripts(); the script is the first token ending in .py, or the last part of the module of `-m`; `-W ignore` / `-X dev` values are not mistaken for it)
    on the device it was given
    (--device, else the script's default). A bash wrapper, `python -c` and a shell are not users of the GPU."""
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
        module, after = (opts[i + 1], opts[i + 2:]) if i + 1 < len(opts) else ("", [])
        if module in RESEARCH_MODULES or (module == "research" and after[:1] == ["worker"]):
            return True
        script = module.split(".")[-1] + ".py" if module else ""
    else:
        script = next((os.path.basename(t) for t in opts if t.endswith(".py")), "")
    if script not in scripts:
        return False
    m = re.search(r"--device(?:=|\s+)(\S+)", args)
    return (m.group(1) if m else scripts[script]).startswith("cuda")


def busy_processes(ps_text, own, scripts=None):
    """the processes of a `ps -eo pid,args` listing that hold the GPU (holds_gpu), except the pids in `own`; scripts: default gpu_scripts()"""
    scripts = gpu_scripts() if scripts is None else scripts
    out = []
    for ln in ps_text.splitlines()[1:]:
        parts = ln.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) not in own and holds_gpu(parts[1], scripts):
            out.append(f"{parts[0]} {parts[1][:100]}")
    return out


def gpu_state(samples=3, gap=0.5):
    """{"util_percent", "mem_used_mb" (max of `samples`), "other_processes" (nvidia-smi), "busy_processes" (ps), "load"} of this machine; None where a tool is
    missing"""
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
        # nvidia-smi lists this process too once it holds a CUDA context
        st["other_processes"] = [ln.strip() for ln in p.splitlines() if ln.strip() and ln.split(",")[0].strip() != str(os.getpid())]
    except Exception:                                                   # noqa: BLE001  (no nvidia-smi: unknown)
        pass
    # outside the try: a problem.json that cannot be read stops the timing, never a quiet miss
    scripts = gpu_scripts()
    try:
        st["busy_processes"] = busy_processes(subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=20).stdout, {os.getpid(), os.getppid()}, scripts)
    except Exception:                                                   # noqa: BLE001  (no ps: unknown)
        pass
    return st


def settled_state(device, samples=2):
    """gpu_state after our own work has left the GPU: nvidia-smi's utilisation covers the last second, so the tail of the model load would read as a busy GPU"""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        time.sleep(1.5)
    return gpu_state(samples)


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
    src, subj, _ = A.build(run, device=device, split="val", workers=0, ckpt_dir=ckpt_dir)
    model, loss, cfg, dev = subj.model, subj.loss, subj.cfg, subj.device
    if str(device).startswith("cuda") and not str(dev).startswith("cuda"):
        print(f"WARNING: cuda was requested but the model runs on {dev}: these are CPU timings", flush=True)
        idle = False
    if cfg.amp != "off":
        print(f"WARNING: the run was trained with amp={cfg.amp}; this times the fp32 path of benchmark()", flush=True)
    state2 = settled_state(device)                                     # the first check was minutes ago (the model load): look again before the timed loop
    idle2 = is_idle(state2, device, check_memory=False)
    if not idle2 and not force:
        raise SystemExit(f"refused before the timed loop: the GPU got busy while the model was loading ({why_busy(state2)})")
    idle = idle and idle2
    fr = src.frames(src.scenes()[0])
    batch = to_device(src.load(fr[len(fr) // 2]).inputs, dev)
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
        train = train_split(lambda: chain.forward_loss(model, loss, batch, 1.0, dev, cfg, train_terms=True)[0]["total"], clear, dev)
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="the folder of a moved checkpoint")
    ap.add_argument("--force", action="store_true", help="time even when the GPU is not idle (the result says so and must not be quoted)")
    a = ap.parse_args(argv)
    run_timing(a.run, a.n, a.warmup, a.device, a.out, a.force, a.ckpt_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
