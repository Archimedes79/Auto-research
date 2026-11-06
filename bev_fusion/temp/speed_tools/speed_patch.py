"""throw-away: the speed changes for train.py (argv[1]) and the profiler for probe.py (argv[2]); every anchor must occur exactly once"""
import sys

p, q = sys.argv[1], sys.argv[2]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


# 1 imports
rep("    from .probe import ModuleProbe, StepTimer, batch_tokens, dump_incident\n", "    from .probe import ModuleProbe, StepProfiler, StepTimer, batch_tokens, dump_incident\n")
rep("    from probe import ModuleProbe, StepTimer, batch_tokens, dump_incident\n", "    from probe import ModuleProbe, StepProfiler, StepTimer, batch_tokens, dump_incident\n")

# 2 the config: opt-in switches, all off
rep("    probe: bool = False ", '''    pin_memory: bool = False                     # DataLoader with pinned host memory and non-blocking copies to the GPU (cuda only; same numbers, faster transfer)
    prefetch_factor: int = 0                     # batches each loader worker prepares ahead (0 = the torch default, 2); only with workers > 0
    cudnn_benchmark: bool = False                # torch.backends.cudnn.benchmark: the fastest convolution algorithm per input shape (shapes are static here); the last bits may differ
    tf32: bool = False                           # TF32 tensor cores for fp32 matmul and convolutions (Ampere): faster, products with a 10-bit mantissa; changes the numbers slightly
    profile_steps: int = 0                       # > 0: torch.profiler over that many optimiser steps after 10 warm-up steps; writes profile.txt into the run folder (probe.StepProfiler)
    probe: bool = False ''')
rep('"grad_diag_every", "probe")', '"grad_diag_every", "probe", "pin_memory", "prefetch_factor", "profile_steps")')

# 3 the loader and the copies
rep('''    return torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=n, collate_fn=fn, drop_last=train,
                                       persistent_workers=train and n > 0, generator=generator)''',
    '''    extra = {}
    if cfg.pin_memory and torch.cuda.is_available():
        extra["pin_memory"] = True
    if cfg.prefetch_factor > 0 and n > 0:
        extra["prefetch_factor"] = cfg.prefetch_factor
    return torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=n, collate_fn=fn, drop_last=train,
                                       persistent_workers=train and n > 0, generator=generator, **extra)''')
rep('''def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif k == "radar":
            out[k] = {a: b.to(device) for a, b in v.items()}
        elif k == "steps":
            out[k] = [[tuple(t.to(device) for t in s) for s in fr] for fr in v]''',
    '''def to_device(batch, device, non_blocking=False):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        elif k == "radar":
            out[k] = {a: b.to(device, non_blocking=non_blocking) for a, b in v.items()}
        elif k == "steps":
            out[k] = [[tuple(t.to(device, non_blocking=non_blocking) for t in s) for s in fr] for fr in v]''')
rep("    batch = to_device(batch, device)\n", "    batch = to_device(batch, device, non_blocking=bool(cfg is not None and cfg.pin_memory))\n")

# 4 the switches at the start of train()
rep("def train(cfg, eval_fn=None, datasets=None):\n", '''def apply_speed_flags(cfg):
    """the opt-in speed switches of the config; all off by default and then nothing is touched. -> {name: value} of what was set"""
    done = {}
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        done["cudnn.benchmark"] = True
    if cfg.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        done["tf32"] = "high"
    return done


def train(cfg, eval_fn=None, datasets=None):
''')
rep("    device = cfg.device if (cfg.device != \"cuda\" or torch.cuda.is_available()) else \"cpu\"\n    out_dir = os.path.join(cfg.out_root, cfg.name)\n",
    "    device = cfg.device if (cfg.device != \"cuda\" or torch.cuda.is_available()) else \"cpu\"\n    speed = apply_speed_flags(cfg)\n    if speed:\n        print(\"speed flags:\", speed, flush=True)\n    out_dir = os.path.join(cfg.out_root, cfg.name)\n")

# 5 the profiler in the loop
rep("    probe, timer = (ModuleProbe(model, every=cfg.log_every), StepTimer()) if cfg.probe else (None, None)\n",
    "    probe, timer = (ModuleProbe(model, every=cfg.log_every), StepTimer()) if cfg.probe else (None, None)\n"
    "    prof = StepProfiler(out_dir, cfg.profile_steps, device=device) if cfg.profile_steps > 0 else None\n")
rep("        trow = timer.row() if timer is not None else {}\n", "        trow = timer.row() if timer is not None else {}\n        if prof is not None:\n            prof.tick()\n")
rep("    return model, loss, ema, last\n", "    if prof is not None:\n        prof.close()                                            # a run shorter than warm-up + window writes what it has\n    return model, loss, ema, last\n")

# 6 the scalar terms are read back with ONE transfer per clip
rep('''def _f(v):
    return float(v.detach()) if torch.is_tensor(v) else float(v)
''', '''def _f(v):
    return float(v.detach()) if torch.is_tensor(v) else float(v)


def _pack(out, selected):
    """snapshot of the logged scalar terms of ONE keyframe: (order, packed). The tensors are stacked NOW into one device tensor (a term that is a view of a parameter, like
    the Kendall weights, which a later keyframe changes in place, must be read as it is at this point), numbers are kept. Nothing is read back here: the old code called
    float() per term, and every call waited for the GPU, which was still busy with the backward of this keyframe, so the host could not queue the next keyframe meanwhile"""
    order, tens = [], []
    for k, v in out.items():
        if selected(k):
            if torch.is_tensor(v):
                order.append((k, True, len(tens)))
                tens.append(v.detach().float().reshape(()))
            else:
                order.append((k, False, float(v)))
    return order, (torch.stack(tens) if tens else None)


def _unpack(kf, L):
    """-> (acc, totals) of the packed keyframes with ONE device-to-host transfer. Bit-identical to the per-term floats: the same float32 values widened to Python floats, summed in
    the same order (keyframe by keyframe, term by term)"""
    packs = [p for _, p in kf if p is not None]
    flat = torch.cat(packs).cpu().tolist() if packs else []
    acc, totals, pos = {}, [], 0
    for order, p in kf:
        base = pos
        pos += 0 if p is None else p.numel()
        for k, is_t, x in order:
            val = flat[base + x] if is_t else x
            acc[k] = acc.get(k, 0.0) + val / L
            if k == "total":
                totals.append(val)
    return acc, totals
''')
rep('''    L, state, prev, acc, totals, dg = len(clip), None, None, {}, [], None
    bptt, accum''', '''    L, state, prev, kf, dg = len(clip), None, None, [], None
    selected = lambda k: k == "total" or k == "consistency" or k.startswith(("group.", "s.")) or k in LOGGED_TERMS          # noqa: E731
    bptt, accum''')
rep('''        if not torch.isfinite(out["total"]):
            return acc, False, dg, totals
        totals.append(_f(out["total"]))
        if backward:''', '''        if not torch.isfinite(out["total"]):
            acc, totals = _unpack(kf, L)
            return acc, False, dg, totals
        kf.append(_pack(out, selected))                           # before the backward, which changes the Kendall weights in place (clamp_)
        if backward:''')
rep('''        for k, v in out.items():
            if k == "total" or k == "consistency" or k.startswith(("group.", "s.")) or k in LOGGED_TERMS:
                acc[k] = acc.get(k, 0.0) + _f(v) / L
        if "diag" in out:''', '''        if "diag" in out:''')
rep('''        prev = batch
    return acc, True, dg, totals
''', '''        prev = batch
    acc, totals = _unpack(kf, L)                                  # the only synchronisation of the clip besides the finite check of each keyframe
    return acc, True, dg, totals
''')
open(p, "w", encoding="utf-8").write(s)

# the profiler
t = open(q, encoding="utf-8").read()
assert "class StepProfiler" not in t
t += '''

class StepProfiler:
    """torch.profiler over `steps` optimiser steps after `warmup` steps (TrainConfig.profile_steps; off by default). Writes <run folder>/profile.txt: the wall time of the window,
    how much of it the GPU was busy (the sum of the kernel times over the wall time: a low share means the GPU waits for the host), how often the host synchronised with
    the GPU (aten::item / cudaStreamSynchronize / cudaMemcpy calls per step) and the top operators by GPU time and by CPU time. Call tick() once at the end of every optimiser
    step and close() when the loop ends (a run shorter than warm-up + window writes what it has)"""

    def __init__(self, out_dir, steps, warmup=10, device="cuda"):
        self.out_dir, self.steps, self.warmup, self.cuda = out_dir, int(steps), warmup, str(device).startswith("cuda")
        self.n, self.prof, self.t0, self.done = 0, None, None, False

    def tick(self):
        if self.done or self.steps <= 0:
            return
        self.n += 1
        if self.prof is None and self.n == self.warmup:
            acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if self.cuda else [])
            self.prof = torch.profiler.profile(activities=acts)
            self.prof.__enter__()
            self.t0, self.n0 = time.perf_counter(), self.n
        elif self.prof is not None and self.n - self.n0 >= self.steps:
            self.close()

    def close(self):
        if self.prof is None or self.done:
            return
        self.done = True
        if self.cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - self.t0
        self.prof.__exit__(None, None, None)
        steps = max(1, self.n - self.n0)
        ka = self.prof.key_averages()
        dev = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0))       # noqa: E731 - the attribute was renamed in torch 2.x
        gpu_s = sum(dev(e) for e in ka if e.device_type == torch.autograd.DeviceType.CPU) / 1e6        # the operator rows only, as torch's own "Self CUDA time total": the kernel rows repeat the same time
        syncs = {e.key: e.count for e in ka if e.key in ("aten::item", "aten::_local_scalar_dense", "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync", "cudaMemcpy")}
        lines = [f"window: {steps} optimiser steps, {wall:.2f} s wall = {wall / steps:.3f} s per step",
                 (f"GPU kernels busy: {gpu_s:.2f} s of {wall:.2f} s = {100 * gpu_s / wall:.0f} % (the rest is the GPU waiting for the host or the data)" if self.cuda
                  else "no CUDA in this run: CPU operators only"),
                 "host-device synchronisations per step: " + (", ".join(f"{k} {v / steps:.1f}" for k, v in sorted(syncs.items())) or "none seen"), ""]
        key = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total" if ka else None
        if self.cuda and key:
            lines += ["top operators by GPU time:", ka.table(sort_by=key, row_limit=25, max_name_column_width=60), ""]
        lines += ["top operators by CPU time:", ka.table(sort_by="self_cpu_time_total", row_limit=20, max_name_column_width=60)]
        path = os.path.join(self.out_dir, "profile.txt")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write("\\n".join(lines) + "\\n")
        os.replace(path + ".tmp", path)
        self.prof = None                                        # the events of the window are millions of objects (about 12 GB for 30 steps of this model): free them now
        del ka
'''
open(q, "w", encoding="utf-8").write(t)
print("speed patch applied")
