"""ms per frame of the junction system on an IDLE GPU, and where the pair head spends them (E342 part B, E341 / E348 saving): batch 1, bf16 autocast like training, the cached input features of 12 VAL frames, CUDA events.

    wsl bash tools/env/_py14.sh jtiming.py run --runs e312_p_s0,e341_p_s0,e324_s_s0,e348_w_s0 [--rounds 10] [--iters 25] [--sections 40] [--tag NAME]    # GPU, writes runs/eval/jtiming/<tag>.json and prints the tables
    wsl bash tools/env/_py14.sh jtiming.py kernels --runs e312_p_s0,e324_s_s0 [--tag NAME]                                                              # GPU: CUDA launches, memcopies and syncs per stage and per section of one forward (E351 step 1)
    wsl bash tools/env/_py14.sh jtiming.py graph --runs e312_p_s0,e324_s_s0 [--rounds 10] [--iters 30] [--tag NAME]                                      # GPU: the eager forward (c) against the CUDA-graph forward of `jgraph.py` (g1), the speed half of E351
    wsl bash tools/env/_py14.sh jtiming.py selftest                                                                                                      # CPU: the statistics, the schedule, the section markers on a toy JEnds, the E351 rule helpers

Protocol (the A/B timer of the BEV system, in short): all models are loaded into the same process; a ROUND times every model for `iters` forwards in a block, the order of the blocks alternates (A B C D, then D C B A), so a slow drift of the GPU
(clocks, temperature) cancels in the paired differences; the number of a model is the median of its block medians, the difference of two models the mean of the per-round differences with a 95 % interval of a bootstrap over the rounds.
Plain blocks measure the forward with two CUDA events only. The SECTION pass times the same forward with markers: forward hooks around the pyramid, the maps head, B's node head (when the model has one) and the pair head, and markers INSIDE
`JEnds.forward`, which are inserted into a COPY of its source by line number (the frozen file is not edited; the line numbers are checked against the text, a change of `jends.py` stops the run with a message). Per section two numbers: the time on the
GPU timeline between two markers (CUDA events: includes the gaps in which the GPU waits for the host) and the host time between them (`perf_counter`: kernel launches and syncs); a section whose host time is its GPU time is host bound.
A short `torch.profiler` pass counts the CUDA kernels of one forward. The GPU is checked before and after (utilisation, memory): timing only on an idle GPU, otherwise the run refuses (`--force` records the fact in the file instead).
`graph` (E351, the speed half of the card's rule): per system two arms in the same process and the same alternating blocks (A B C D | D C B A over all arms), 10 rounds x 30 forwards: c = the model's own eager forward without markers or hooks
(`jgraph.eager_forward`), g1 = `jgraph.GraphedModel.forward` (copy into the static input, replay, eager pair head, clone of every returned tensor: all inside the timed region); `cudnn.benchmark` False and printed. Per system the paired difference
g1 - c with the bootstrap over the rounds, the minimum effect (15 % of the c median), the label of the rule (`speed_half`: ADOPT / TIE / INCONCLUSIVE) and the host: the 1-minute load average before and after AND the GPU state before and after
(QUIET when both loads are below 3 on the 12 cores and the GPU showed at most 5 % utilisation, else NOT QUIET and the speed half is INCONCLUSIVE: the eager forward is host bound and a loaded host or a busy GPU inflates the control, also in a `--force` run).
Before any timing `precheck` compares the graph forward with the eager forward on the cached frames (`jgraph.Compare`) and stops the run unless they are identical; the result is in the file (`equality_precheck`). The equality half of the rule is `jgraph.py check`.
"""
import argparse
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
import types

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends as JEN  # noqa: E402

# (line of training/grid/jends.py in front of which a marker is inserted, name of the boundary, the text the line starts with: the check)
FIRST_LINE = 155                                                                                                  # the line of `def forward(self, jout, P, ...)` in jends.py
BOUNDARIES = (
    (156, "b0_start", "j1, f1 = jout"),
    (180, "b1_after_peaks", "pos, D, pr, feat = [], [], [], []"),
    (198, "b2_after_softargmax", "K_ = self.k"),
    (203, "b3_after_kind", "t = self.kp("),
    (206, "b4_after_tokens", "ta_t, tb_t = t[:, :K_]"),
    (222, "b5_after_geometry", "if self.evidence:"),
    (231, "b6_after_evidence", "ga = ra = gb = rb = None"),
    (235, "b7_after_rank", "ls = None"),
    (242, "b8_after_pair_mlp", "out = dict("),
    (252, "b9_before_return", "return out"),
)
SECTION_NAMES = ("peaks_topk", "softargmax_gather", "kind_h2d", "tokens_setblocks", "pair_geometry", "evidence", "rank_matrix_host", "pair_mlp", "outputs_scales_attach")   # between consecutive boundaries


class Marks:
    """a list of (name, host time, CUDA event): `mark(name)` appends when `on`"""

    def __init__(self, cuda):
        self.on, self.items, self.cuda = False, [], cuda

    def __call__(self, name):
        if not self.on:
            return
        ev = None
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
        self.items.append((name, time.perf_counter(), ev))

    def reset(self):
        self.items = []

    def span(self, a, b):
        """(GPU timeline ms or None, host ms) between the first marks a and b"""
        names = [n for n, _, _ in self.items]
        if a not in names or b not in names:
            return None
        ia, ib = names.index(a), names.index(b)
        ev_a, ev_b = self.items[ia][2], self.items[ib][2]
        gpu = ev_a.elapsed_time(ev_b) if ev_a is not None and ev_b is not None else None
        return gpu, (self.items[ib][1] - self.items[ia][1]) * 1000.0


def instrument(je, mark):
    """replaces `je.forward` by a copy of the source of `JEnds.forward` with `mark("<boundary>")` in front of the lines of BOUNDARIES (same code otherwise: same outputs, tested)"""
    lines = inspect.getsource(JEN.JEnds.forward).split("\n")
    assert lines[0].lstrip().startswith("def forward(self, jout, P"), "jends.py changed: the forward does not start where jtiming.py expects it"
    for n, name, start in BOUNDARIES:
        text = lines[n - FIRST_LINE].strip() if 0 <= n - FIRST_LINE < len(lines) else ""
        assert text.startswith(start), f"jends.py changed: line {n} is {text!r}, jtiming.py expects it to start with {start!r}: update BOUNDARIES"
    for n, name, start in sorted(BOUNDARIES, reverse=True):
        lines.insert(n - FIRST_LINE, " " * 8 + f'_mark("{name}")')
    ns = dict(vars(JEN))
    ns["_mark"] = mark
    exec(textwrap.dedent("\n".join(lines)), ns)                                                               # noqa: S102 (a copy of the project's own source)
    je.forward = types.MethodType(ns["forward"], je)


def hook(module, tag, mark):
    module.register_forward_pre_hook(lambda m, a: mark(tag + ":in"))
    module.register_forward_hook(lambda m, a, o: mark(tag + ":out"))


def schedule(models, r):
    """the order of the blocks of round r: A B C D, then D C B A, ..."""
    return list(models) if r % 2 == 0 else list(reversed(models))


def paired(a, b, n=10000, seed=0):
    """mean of a - b over the rounds and the 95 % interval of a bootstrap over the rounds: (mean, lo, hi)"""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.RandomState(seed)
    means = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


MIN_EFFECT = 0.15                                                                                          # E351: the minimum effect of the speed half, as a fraction of the c arm's median of the same session
QUIET_LOAD = 3.0                                                                                           # E351: the 1-minute load average (12 cores) below which the host is quiet


def load1():
    """the 1-minute load average of the host (/proc/loadavg), None when it cannot be read"""
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def gpu_idle(state):
    """the GPU counts as idle when `gpu_state()` could read it and no sample showed more than 5 % utilisation"""
    return state is not None and "error" not in state and state["util_max"] <= 5.0


def quiet_verdict(load_before, load_after, gpu_before, gpu_after, limit=QUIET_LOAD):
    """-> (quiet, line): QUIET when both load averages are below `limit` AND the GPU was idle before and after (`gpu_idle`), else NOT QUIET (an unreadable value is not quiet; a run forced onto a busy GPU is never quiet): the speed half of E351
    is then INCONCLUSIVE"""
    quiet = load_before is not None and load_after is not None and load_before < limit and load_after < limit and gpu_idle(gpu_before) and gpu_idle(gpu_after)
    txt = lambda s: "unreadable" if s is None or "error" in s else f"util max {s['util_max']:.0f} %"  # noqa: E731
    return quiet, f"host load (1 min) before {load_before} after {load_after}, limit {limit}; GPU before {txt(gpu_before)}, after {txt(gpu_after)} (idle: at most 5 %): {'QUIET' if quiet else 'NOT QUIET'}"


def speed_half(d, c_median, quiet, frac=MIN_EFFECT):
    """the speed half of the rule of E351 for ONE system: d = paired (mean, lo, hi) of g1 - c over the rounds (ms), c_median the median of the c blocks (ms) -> (label, reason). INCONCLUSIVE: the host was not quiet or the interval contains 0;
    ADOPT: the drop is at least frac * c_median and the interval excludes 0; TIE: the interval excludes 0 but the drop is smaller (or g1 is slower): the eager forward stays"""
    mean, lo, hi = d
    need = frac * c_median
    if not quiet:
        return "INCONCLUSIVE", "the host was not quiet"
    if lo <= 0.0 <= hi:
        return "INCONCLUSIVE", f"the interval [{lo:+.2f}, {hi:+.2f}] contains 0"
    if mean <= -need:
        return "ADOPT", f"the drop {-mean:.2f} ms is at least {need:.2f} ms ({100 * frac:.0f} % of {c_median:.2f}) and the interval excludes 0"
    return "TIE", f"the interval excludes 0 but the change {mean:+.2f} ms is not a drop of the minimum effect {need:.2f} ms"


def gpu_state():
    """utilisation (%) and memory used (MiB) of the GPU by nvidia-smi, 5 samples one second apart; None when nvidia-smi is not there"""
    try:
        out = []
        for _ in range(5):
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            u, m, t, c = [float(x) for x in r.stdout.strip().split("\n")[0].split(",")]
            out.append((u, m, t, c))
            time.sleep(1.0)
        a = np.array(out)
        return dict(util_max=float(a[:, 0].max()), mem_mib=float(a[:, 1].mean()), temp_c=float(a[:, 2].mean()), sm_clock_mhz=float(a[:, 3].mean()))
    except Exception as e:                                                                                    # noqa: BLE001
        return dict(error=str(e))


def profile_forward(m, x):
    try:
        from torch.profiler import ProfilerActivity, profile
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m(x, gt=None)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                m(x, gt=None)
                torch.cuda.synchronize()
        ka = prof.key_averages()
        kernels = sum(e.count for e in ka if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA)
        top = sorted(ka, key=lambda e: -e.self_cpu_time_total)[:10]
        return dict(cuda_kernels=int(kernels), top_self_cpu=[(e.key[:50], int(e.count), round(e.self_cpu_time_total / 1000.0, 3)) for e in top])
    except Exception as e:                                                                                    # noqa: BLE001
        return dict(error=str(e))


LAUNCH_NAMES = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaMemcpyAsync", "cudaMemcpy", "cudaMemsetAsync", "cudaStreamSynchronize", "cudaDeviceSynchronize")


class ProfMarks(Marks):
    """markers that are zero-length `record_function` ranges, so that they stand in the profiler's own timeline"""

    def __init__(self):
        super().__init__(cuda=False)
        self.on = True

    def __call__(self, name):
        from torch.profiler import record_function
        with record_function("mk:" + name):
            pass


def count_by_section(events):
    """events: (name, start us, is_marker) of one profiled forward -> {segment: {runtime call: count}}; a segment is the stretch between two consecutive markers"""
    marks = sorted((t, n[3:]) for n, t, is_m in events if is_m)
    out = {}
    for n, t, is_m in events:
        if is_m or n not in LAUNCH_NAMES:
            continue
        k = max((i for i, (mt, _) in enumerate(marks) if mt <= t), default=None)
        if k is None or k + 1 >= len(marks):
            continue
        seg = marks[k][1] + " > " + marks[k + 1][1]
        out.setdefault(seg, {})
        out[seg][n] = out[seg].get(n, 0) + 1
    return out


def kernels(a):
    import jends_eval as JE
    import lane_gt_data as LD
    from torch.profiler import ProfilerActivity, profile
    assert torch.cuda.is_available(), "no GPU"
    runs = a.runs.split(",")
    before = gpu_state()
    if ("error" in before or before["util_max"] > 5.0) and not a.force:
        sys.exit(f"STOP: the GPU is not idle ({before})")
    LD.QData.DEFAULT_JUNCTION = True
    ds = LD.QData("val_clean", map_mode="error", pixels=False)
    x = LD.collate([ds[len(ds) // 2]])["x"].to("cuda").float()
    res = {}
    for r_ in runs:
        m, _ = JE.load_model(r_, "cuda")
        mk = ProfMarks()
        mk.on = False
        hook(m.jmodel.pyr, "pyr", mk)
        hook(m.jmodel.jm, "jm", mk)
        if m.nodes is not None:
            hook(m.nodes, "nodes", mk)
        hook(m.je, "je", mk)
        instrument(m.je, mk)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                m(x, gt=None)
            torch.cuda.synchronize()
            mk.on = True
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                mk("fwd:in")
                m(x, gt=None)
                mk("fwd:out")
                torch.cuda.synchronize()
        evs = [(e.name, float(e.time_range.start), e.name.startswith("mk:")) for e in prof.events()]
        res[r_] = count_by_section(evs)
    names = [BOUNDARIES[k][1] + " > " + BOUNDARIES[k + 1][1] for k in range(len(SECTION_NAMES))]
    stages = ["pyr:in > pyr:out", "jm:in > jm:out", "nodes:in > nodes:out"]
    print(f"CUDA runtime calls per stage / section of ONE forward (launches / memcopies / syncs), after {before}")
    print("  " + "segment".ljust(40) + "".join(r.rjust(22) for r in runs))
    allsegs = []
    for r_ in runs:
        for k in res[r_]:
            if k not in allsegs:
                allsegs.append(k)
    for k in allsegs:
        row = "  " + k.ljust(40)
        for r_ in runs:
            c = res[r_].get(k, {})
            row += (f"{c.get('cudaLaunchKernel', 0) + c.get('cudaLaunchKernelExC', 0)} / {c.get('cudaMemcpyAsync', 0) + c.get('cudaMemcpy', 0)} / {c.get('cudaStreamSynchronize', 0) + c.get('cudaDeviceSynchronize', 0)}" if c else "-").rjust(22)
        print(row)
    tot = {r_: {n: sum(c.get(n, 0) for c in res[r_].values()) for n in LAUNCH_NAMES} for r_ in runs}
    print("  " + "TOTAL".ljust(40) + "".join((f"{t['cudaLaunchKernel'] + t['cudaLaunchKernelExC']} / {t['cudaMemcpyAsync'] + t['cudaMemcpy']} / {t['cudaStreamSynchronize'] + t['cudaDeviceSynchronize']}").rjust(22) for t in tot.values()))
    out = a.out or os.path.join(ROOT, "runs", "eval", "jtiming", f"{a.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    assert not os.path.exists(out), f"{out} exists"
    json.dump(dict(runs=runs, gpu_before=before, segments=res, totals=tot, frame="the middle VAL frame, bf16, batch 1"), open(out, "w"), indent=1)
    print("written", out)


def precheck(model, gm, xs):
    """E351: before any timing the graph forward must equal the eager forward on every cached frame (torch.equal and the same bits, `jgraph.Compare`); a fast and wrong g1 is never timed -> the report of the comparison; stops the run otherwise"""
    import jgraph as JG
    cmp = JG.Compare()
    for x in xs:
        cmp.add(JG.eager_forward(model, x), gm.forward(x))
    if cmp.bad:
        sys.exit("STOP: the graph forward is not equal to the eager forward on the cached frames; nothing is timed\n" + "\n".join(cmp.lines("graph against eager")))
    return cmp.report()


def graph(a):
    import jends_eval as JE
    import jgraph as JG
    import lane_gt_data as LD
    assert torch.cuda.is_available(), "no GPU: this measurement is for an idle GPU"
    torch.backends.cudnn.benchmark = False
    runs = a.runs.split(",")
    out = a.out or os.path.join(ROOT, "runs", "eval", "jtiming", f"{a.tag}.json")
    assert not os.path.exists(out), f"{out} exists: another tag (a measurement is never overwritten)"
    load_before = load1()
    before = gpu_state()
    idle = gpu_idle(before)
    if not idle and not a.force:
        sys.exit(f"STOP: the GPU is not idle ({before}); timing only on an idle GPU (--force writes the state into the file instead: the speed half is then INCONCLUSIVE)")
    dev = "cuda"
    print(f"load average (1 min) {load_before}, cudnn.benchmark {torch.backends.cudnn.benchmark}")
    LD.QData.DEFAULT_JUNCTION = True
    ds = LD.QData("val_clean", map_mode="error", pixels=False)
    xs = [LD.collate([ds[i]])["x"].to(dev).float() for i in np.linspace(0, len(ds) - 1, a.frames).astype(int)]
    fns, equality = {}, {}
    for r_ in runs:
        m, _ = JE.load_model(r_, dev)
        for i in range(a.warm):
            JG.eager_forward(m, xs[i % len(xs)])
        gm = JG.GraphedModel(m).capture(xs[0])
        for i in range(a.warm):
            gm.forward(xs[i % len(xs)])
        equality[r_] = precheck(m, gm, xs)
        fns[(r_, "c")], fns[(r_, "g1")] = (lambda x, m=m: JG.eager_forward(m, x)), gm.forward
    torch.cuda.synchronize()
    arms = [(r_, arm) for r_ in runs for arm in ("c", "g1")]
    blocks = {k: [] for k in arms}
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    for r in range(a.rounds):                                                                                 # the order of the blocks alternates: A B C D, then D C B A
        for key in schedule(arms, r):
            ms = []
            for i in range(a.iters):
                x = xs[(r * a.iters + i) % len(xs)]
                e0, e1 = ev(), ev()
                torch.cuda.synchronize()
                e0.record()
                fns[key](x)
                e1.record()
                torch.cuda.synchronize()
                ms.append(e0.elapsed_time(e1))
            blocks[key].append(float(np.median(ms)))
    torch.cuda.synchronize()
    time.sleep(2.5)                                                                 # the last timed kernels (this process's own work) must age out of nvidia-smi's averaging window, else the "after" sample reads our own tail as a busy GPU (2026-11-15, E351.g1)
    load_after, after = load1(), gpu_state()
    quiet, qline = quiet_verdict(load_before, load_after, before, after)
    res = dict(runs=runs, rounds=a.rounds, iters=a.iters, frames=a.frames, warm=a.warm, precision="bf16 autocast, batch 1, no_grad", cudnn_benchmark=torch.backends.cudnn.benchmark, device=torch.cuda.get_device_name(0),
               timed_region="c: the model's eager forward; g1: copy into the static input + graph replay + eager pair head + clone of every returned tensor", gpu_before=before, gpu_after=after, gpu_idle_at_start=bool(idle),
               load_before=load_before, load_after=load_after, quiet=quiet, quiet_line=qline, equality_precheck=equality, block_medians_ms={f"{r_}/{arm}": v for (r_, arm), v in blocks.items()}, systems={})
    print(f"GPU {res['device']}, {res['precision']}; GPU before {before}, after {after}\n{qline}\n\nms per frame (median of the {a.rounds} block medians of {a.iters} forwards), paired g1 - c over the rounds (95 % bootstrap)")
    for r_ in runs:
        c_med, g_med = float(np.median(blocks[(r_, "c")])), float(np.median(blocks[(r_, "g1")]))
        d = paired(blocks[(r_, "g1")], blocks[(r_, "c")])
        label, why = speed_half(d, c_med, quiet)
        res["systems"][r_] = dict(c_median_ms=c_med, g1_median_ms=g_med, paired_g1_minus_c=d, percent=100 * d[0] / c_med, min_effect_ms=MIN_EFFECT * c_med, speed_half=label, reason=why)
        print(f"  {r_:14s} c {c_med:7.3f}  g1 {g_med:7.3f}  g1 - c {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] = {100 * d[0] / c_med:+.1f} %  | speed half: {label} ({why})")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("written", out)


def run(a):
    import jends_eval as JE
    import lane_gt_data as LD
    assert torch.cuda.is_available(), "no GPU: this measurement is for an idle GPU"
    runs = a.runs.split(",")
    before = gpu_state()
    idle = gpu_idle(before)
    if not idle and not a.force:
        sys.exit(f"STOP: the GPU is not idle ({before}); timing only on an idle GPU (--force writes the state into the file instead)")
    dev = "cuda"
    LD.QData.DEFAULT_JUNCTION = True
    ds = LD.QData("val_clean", map_mode="error", pixels=False)
    xs = [LD.collate([ds[i]])["x"].to(dev).float() for i in np.linspace(0, len(ds) - 1, a.frames).astype(int)]
    models, marks = {}, {}
    for r_ in runs:
        m, _ = JE.load_model(r_, dev)
        mk = Marks(cuda=True)
        hook(m.jmodel.pyr, "pyr", mk)
        hook(m.jmodel.jm, "jm", mk)
        if m.nodes is not None:
            hook(m.nodes, "nodes", mk)
        hook(m.je, "je", mk)
        instrument(m.je, mk)
        models[r_], marks[r_] = m, mk
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731

    def forward(r_, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return models[r_](x, gt=None)

    for r_ in runs:                                                                                           # warm-up
        for i in range(a.warm):
            forward(r_, xs[i % len(xs)])
    torch.cuda.synchronize()
    blocks = {r_: [] for r_ in runs}
    raw = {r_: [] for r_ in runs}
    for r in range(a.rounds):                                                                                 # plain blocks: two events around the forward, alternating order
        for r_ in schedule(runs, r):
            ms = []
            for i in range(a.iters):
                x = xs[(r * a.iters + i) % len(xs)]
                e0, e1 = ev(), ev()
                torch.cuda.synchronize()
                e0.record()
                forward(r_, x)
                e1.record()
                torch.cuda.synchronize()
                ms.append(e0.elapsed_time(e1))
            blocks[r_].append(float(np.median(ms)))
            raw[r_] += ms
    secs = {}
    for r_ in runs:                                                                                           # the section pass
        mk = marks[r_]
        acc = {}
        for i in range(a.sections):
            x = xs[i % len(xs)]
            mk.reset()
            mk.on = True
            mk("fwd:in")
            forward(r_, x)
            mk("fwd:out")
            torch.cuda.synchronize()
            mk.on = False
            spans = [("fwd_total", "fwd:in", "fwd:out"), ("pyramid", "pyr:in", "pyr:out"), ("maps_head", "jm:in", "jm:out"), ("node_head", "nodes:in", "nodes:out"), ("pair_head", "je:in", "je:out")]
            spans += [(nm, BOUNDARIES[k][1], BOUNDARIES[k + 1][1]) for k, nm in enumerate(SECTION_NAMES)]
            for nm, p, q in spans:
                s = mk.span(p, q)
                if s is not None:
                    acc.setdefault(nm, []).append(s)
        secs[r_] = {nm: dict(gpu_ms=float(np.median([g for g, _ in v])), host_ms=float(np.median([h for _, h in v]))) for nm, v in acc.items()}
    prof = {r_: profile_forward(models[r_], xs[0]) for r_ in runs}
    after = gpu_state()
    res = dict(runs=runs, rounds=a.rounds, iters=a.iters, frames=a.frames, precision="bf16 autocast, batch 1, no_grad", gpu_before=before, gpu_after=after, gpu_idle_at_start=bool(idle), device=torch.cuda.get_device_name(0),
               plain_block_medians_ms=blocks, plain_median_ms={k: float(np.median(v)) for k, v in raw.items()}, sections=secs, profiler=prof)
    pairs = [x.split("-") for x in a.pairs.split(",")] if a.pairs else []
    res["paired"] = {}
    for x, y in pairs:
        if x in blocks and y in blocks:
            res["paired"][f"{x} - {y}"] = paired(blocks[x], blocks[y])
    out = a.out or os.path.join(ROOT, "runs", "eval", "jtiming", f"{a.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    assert not os.path.exists(out), f"{out} exists: another tag (a measurement is never overwritten)"
    json.dump(res, open(out, "w"), indent=1, default=float)
    print_tables(res)
    print("written", out)


def print_tables(res):
    runs = res["runs"]
    print(f"GPU {res['device']}, {res['precision']}; idle at start: {res['gpu_idle_at_start']} (before {res['gpu_before']}, after {res['gpu_after']})")
    print(f"\nms per frame (median over {res['rounds']} x {res['iters']} forwards; the median of the {res['rounds']} block medians in brackets, min - max)")
    for r in runs:
        b = res["plain_block_medians_ms"][r]
        print(f"  {r:18s} {res['plain_median_ms'][r]:7.3f}   [{np.median(b):.3f}, {min(b):.3f} - {max(b):.3f}]   CUDA kernels per forward {res['profiler'][r].get('cuda_kernels', 'n/a')}")
    if res["paired"]:
        print("\npaired differences (mean of the per-round block medians, 95 % bootstrap over the rounds), ms:")
        for k, (m, lo, hi) in res["paired"].items():
            print(f"  {k:32s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    names = ["fwd_total", "pyramid", "maps_head", "node_head", "pair_head"] + list(SECTION_NAMES)
    print("\nsections, GPU timeline ms / host ms (median of the section pass):")
    print("  " + "section".ljust(24) + "".join(r.rjust(24) for r in runs))
    for nm in names:
        row = "  " + nm.ljust(24)
        for r in runs:
            v = res["sections"][r].get(nm)
            row += (f"{v['gpu_ms']:.3f} / {v['host_ms']:.3f}" if v else "-").rjust(24)
        print(row)


def selftest():
    # statistics
    m, lo, hi = paired([2.0, 2.2, 1.9, 2.1], [1.0, 1.1, 0.9, 1.0])
    assert abs(m - 1.05) < 1e-9 and lo <= m <= hi and lo > 0.8, (m, lo, hi)
    m0, lo0, hi0 = paired([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert m0 == 0 and lo0 == 0 and hi0 == 0
    assert schedule(["a", "b", "c"], 0) == ["a", "b", "c"] and schedule(["a", "b", "c"], 1) == ["c", "b", "a"]
    # the markers: a toy JEnds on the CPU, the instrumented forward gives the same outputs, every boundary is marked once, in order
    torch.manual_seed(0)
    je = JEN.JEnds(order=True, evidence=True, scales=True)
    B = 1
    jout = {"j1": torch.randn(B, 8, 100, 100) * 2, "f1": torch.randn(B, 64, 100, 100), "j05": torch.randn(B, 2, 100, 100)}
    P = {"centre1": torch.randn(B, 3, 100, 100)}
    with torch.no_grad():
        ref = je(jout, P)
        mk = Marks(cuda=False)
        mk.on = True
        instrument(je, mk)
        out = je(jout, P)
    assert set(ref) == set(out) and all(torch.equal(ref[k], out[k]) for k in ref), "the instrumented forward changes an output"
    assert [n for n, _, _ in mk.items] == [b[1] for b in BOUNDARIES], [n for n, _, _ in mk.items]
    sp = mk.span("b0_start", "b9_before_return")
    assert sp is not None and sp[0] is None and sp[1] > 0
    total = mk.span("b0_start", "b9_before_return")[1]
    parts = sum(mk.span(BOUNDARIES[k][1], BOUNDARIES[k + 1][1])[1] for k in range(len(SECTION_NAMES)))
    assert abs(total - parts) < 1e-6, (total, parts)
    # a changed source line is caught (mutation): the check names the line
    saved = BOUNDARIES
    try:
        globals()["BOUNDARIES"] = ((156, "b0_start", "this line does not exist"),) + saved[1:]
        try:
            instrument(JEN.JEnds(), Marks(cuda=False))
            raise SystemExit("a wrong boundary text was accepted")
        except AssertionError as e:
            assert "line 156" in str(e)
    finally:
        globals()["BOUNDARIES"] = saved
    ev = [("mk:a", 0.0, True), ("cudaLaunchKernel", 1.0, False), ("cudaLaunchKernel", 2.0, False), ("mk:b", 3.0, True), ("cudaMemcpyAsync", 4.0, False), ("cudaStreamSynchronize", 5.0, False), ("mk:c", 6.0, True), ("cudaLaunchKernel", 7.0, False)]
    c = count_by_section(ev)
    assert c == {"a > b": {"cudaLaunchKernel": 2}, "b > c": {"cudaMemcpyAsync": 1, "cudaStreamSynchronize": 1}}, c     # a call after the last marker belongs to no segment
    # E351 rule helpers, known answers (c median 17.3 ms: the minimum effect is 2.595 ms; the host is quiet below a load of 3 before AND after)
    idle, busy, broken = dict(util_max=0.0, mem_mib=1000.0), dict(util_max=60.0, mem_mib=9000.0), dict(error="no nvidia-smi")
    assert quiet_verdict(2.9, 2.9, idle, idle)[0] and not quiet_verdict(2.9, 3.0, idle, idle)[0] and not quiet_verdict(3.5, 1.0, idle, idle)[0] and not quiet_verdict(None, 1.0, idle, idle)[0]
    assert quiet_verdict(1.0, 1.0, idle, idle)[1].endswith(": QUIET") and quiet_verdict(3.0, 1.0, idle, idle)[1].endswith(": NOT QUIET")
    for g in ((busy, idle), (idle, busy), (idle, broken), (None, idle)):                                      # the GPU counts as much as the load: a busy, unreadable or missing state is not quiet
        assert not quiet_verdict(1.0, 1.0, *g)[0], g
    assert gpu_idle(dict(util_max=5.0)) and not gpu_idle(dict(util_max=5.1)) and not gpu_idle(broken) and not gpu_idle(None)
    assert "util max 60 %" in quiet_verdict(1.0, 1.0, busy, idle)[1]
    forced = quiet_verdict(1.0, 1.0, busy, busy)[0]                                                           # a --force run on a busy GPU with a quiet host: whatever the drop, never ADOPT
    assert speed_half((-9.0, -10.0, -8.0), 17.3, forced)[0] == "INCONCLUSIVE"
    cases = [((-4.0, -5.0, -3.0), True, "ADOPT"), ((-2.6, -3.0, -2.0), True, "ADOPT"), ((-2.0, -3.0, -1.0), True, "TIE"), ((-4.0, -9.0, 1.0), True, "INCONCLUSIVE"), ((-4.0, -5.0, -3.0), False, "INCONCLUSIVE"), ((1.0, 0.5, 1.5), True, "TIE")]
    for d, quiet, want in cases:
        assert speed_half(d, 17.3, quiet)[0] == want, (d, quiet, want, speed_half(d, 17.3, quiet))
    arms = [("a", "c"), ("a", "g1"), ("b", "c"), ("b", "g1")]
    assert schedule(arms, 1) == arms[::-1] and schedule(arms, 2) == arms
    # E351 precheck: a graph forward that equals the eager one passes and is reported; a fast and wrong one (here: one logit moved) stops the run
    import jgraph as JG
    m = JG._toy()
    xs = [torch.randn(1, 39, 200, 200) for _ in range(2)]
    gm = JG.GraphedModel(m, runner=JG.PoolRunner()).capture(xs[0])
    rep = precheck(m, gm, xs)
    assert rep["identical"] and rep["frames"] == 2 and rep["tensors"] > 40, rep

    class Wrong:
        def forward(self, x):
            P, out, oe = gm.forward(x)
            oe["logit"] = oe["logit"] + 1.0
            return P, out, oe
    msg = None
    try:
        precheck(m, Wrong(), xs)
    except SystemExit as e:
        msg = str(e)
    assert msg and "not equal to the eager forward" in msg and "[2]/logit" in msg, msg
    print("jtiming selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", default="e312_p_s0,e341_p_s0,e324_s_s0,e348_w_s0")
    r.add_argument("--rounds", type=int, default=10)
    r.add_argument("--iters", type=int, default=25)
    r.add_argument("--warm", type=int, default=20)
    r.add_argument("--sections", type=int, default=40)
    r.add_argument("--frames", type=int, default=12)
    r.add_argument("--pairs", default="e341_p_s0-e312_p_s0,e348_w_s0-e324_s_s0,e324_s_s0-e312_p_s0,e348_w_s0-e312_p_s0")
    r.add_argument("--tag", default="e342b")
    r.add_argument("--out", default=None)
    r.add_argument("--force", action="store_true")
    k = sub.add_parser("kernels")
    k.add_argument("--runs", default="e312_p_s0,e341_p_s0,e324_s_s0,e348_w_s0")
    k.add_argument("--tag", default="e351_kernels")
    k.add_argument("--out", default=None)
    k.add_argument("--force", action="store_true")
    g = sub.add_parser("graph")
    g.add_argument("--runs", default="e312_p_s0,e324_s_s0")
    g.add_argument("--rounds", type=int, default=10)
    g.add_argument("--iters", type=int, default=30)
    g.add_argument("--warm", type=int, default=20)
    g.add_argument("--frames", type=int, default=24)
    g.add_argument("--tag", default="e351_graph")
    g.add_argument("--out", default=None)
    g.add_argument("--force", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "graph":
        return graph(a)
    if a.cmd == "kernels":
        return kernels(a)
    run(a)


if __name__ == "__main__":
    main()
