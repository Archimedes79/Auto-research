"""ms per frame of the junction system on an IDLE GPU, and where the pair head spends them (E342 part B, E341 / E348 saving): batch 1, bf16 autocast like training, the cached input features of 12 VAL frames, CUDA events.

    wsl bash tools/env/_py14.sh jtiming.py run --runs e312_p_s0,e341_p_s0,e324_s_s0,e348_w_s0 [--rounds 10] [--iters 25] [--sections 40] [--tag NAME]    # GPU, writes runs/eval/jtiming/<tag>.json and prints the tables
    wsl bash tools/env/_py14.sh jtiming.py selftest                                                                                                      # CPU: the statistics, the schedule, the section markers on a toy JEnds

Protocol (the A/B timer of the BEV system, in short): all models are loaded into the same process; a ROUND times every model for `iters` forwards in a block, the order of the blocks alternates (A B C D, then D C B A), so a slow drift of the GPU
(clocks, temperature) cancels in the paired differences; the number of a model is the median of its block medians, the difference of two models the mean of the per-round differences with a 95 % interval of a bootstrap over the rounds.
Plain blocks measure the forward with two CUDA events only. The SECTION pass times the same forward with markers: forward hooks around the pyramid, the maps head, B's node head (when the model has one) and the pair head, and markers INSIDE
`JEnds.forward`, which are inserted into a COPY of its source by line number (the frozen file is not edited; the line numbers are checked against the text, a change of `jends.py` stops the run with a message). Per section two numbers: the time on the
GPU timeline between two markers (CUDA events: includes the gaps in which the GPU waits for the host) and the host time between them (`perf_counter`: kernel launches and syncs); a section whose host time is its GPU time is host bound.
A short `torch.profiler` pass counts the CUDA kernels of one forward. The GPU is checked before and after (utilisation, memory): timing only on an idle GPU, otherwise the run refuses (`--force` records the fact in the file instead).
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


def run(a):
    import jends_eval as JE
    import lane_gt_data as LD
    assert torch.cuda.is_available(), "no GPU: this measurement is for an idle GPU"
    runs = a.runs.split(",")
    before = gpu_state()
    idle = "error" not in before and before["util_max"] <= 5.0
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
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    run(a)


if __name__ == "__main__":
    main()
