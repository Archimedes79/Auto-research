"""The evaluation path of a narrowed junction checkpoint (E353): the stored tools of `evaluation/` with a surgery-aware model loader that is installed in THIS process only (no digested file is edited).

    wsl bash tools/env/_py14.sh jnarrow_eval.py dump  --run e353_n1_s0 [--thr 0.5] [--max N]               # `jends_eval.dump`: runs/<run>/ends_val.pkl + ends_head.json, the format of the stored runs; REFUSES a run that `jnarrow_train.py` did not write (no `narrow` / `init_run` in its args: a stored run's dump would be overwritten), `--unnarrowed-ok` lifts that
    wsl bash tools/env/_py14.sh jnarrow_eval.py score --run e353_n1_s0 --seed 0 [flags of jends_eval.py score]  # `jends_eval.py score` as it is (it reads the state, it never builds the model; the pyramid line reads 'False' for a changed pyramid)
    wsl bash tools/env/_py14.sh jnarrow_eval.py time paired --runs e324_s_s0,e353_n1_s0 --tag e353_time_s0     # the speed half of E353 (below): control first, narrowed second, the file runs/eval/jtiming/e353_time_s0.json
    wsl bash tools/env/_py14.sh jnarrow_eval.py time  run --runs e324_s_s0,e353_n1_s0 [flags of jtiming.py]   # `jtiming.py run | graph | kernels` with the narrowed run next to the stored one, one process (no load average, no quiet label)
`jnarrow_train.py dump | score | time ...` is the same call. A checkpoint without `args["narrow"]` (the stored runs, the cp control) loads through the stored loader unchanged, so the dump of an unnarrowed run (`--unnarrowed-ok`) equals `jends_eval.py dump` frame for frame.

time paired: ONE process, both models through the surgery-aware loader (`JN.load_model`), the model's own eager forward (`jgraph.eager_forward`: no_grad + bf16 autocast, batch 1), `frames` (24) cached VAL frames, CUDA events, `cudnn.benchmark` False; a ROUND times each model for `iters` (30)
forwards in a block, the blocks alternate A B | B A (`jtiming.schedule`), `rounds` (10) rounds; the number of a model is the median of its block medians, the difference narrowed - control the mean of the per-round differences with a 95 % bootstrap over the rounds (`jtiming.paired`). The host is read as in `jtiming.py graph`: the 1-minute load average and
the GPU state (`gpu_state`, 5 samples) before, and again after the last forward plus a 2.5 s pause (the after-sample must not read our own tail, E351.g1); `quiet_verdict`: QUIET when both loads are below 3 and the GPU showed at most 5 % utilisation, else NOT QUIET (the number is then not a speed result). A busy GPU is refused (`--force` times it and writes the
fact into the file; the label is then NOT QUIET). The result file is never overwritten (another `--tag`).
"""
import argparse
import json
import os
import runpy
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jends_eval as JEV  # noqa: E402
import jgraph as JG  # noqa: E402
import jnarrow as JN  # noqa: E402
import jtiming as JT  # noqa: E402

SETTLE_S = 2.5                                                                              # the pause after the last timed kernel, before the GPU after-sample


def install():
    """`jends_eval.load_model` (called by `dump` and by `jtiming.py`) becomes the surgery-aware loader, in this process"""
    pass


def written_by_jnarrow(run):
    """a run of `jnarrow_train.py` carries `narrow` (the resolved spec, empty for the cp control) and `init_run` in its args; a stored run does not"""
    args = torch.load(os.path.join(JEV.ROOT, "runs", run, "model.pt"), map_location="cpu")["args"]
    return "narrow" in args and "init_run" in args


def dump(argv):
    ap = argparse.ArgumentParser(prog="dump")
    ap.add_argument("--run", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; the score needs all 873)")
    ap.add_argument("--unnarrowed-ok", action="store_true", help="dump a run that jnarrow_train.py did not write (the equality test of the unnarrowed dump): it overwrites that run's ends_val.pkl")
    a = ap.parse_args(argv)
    assert a.unnarrowed_ok or written_by_jnarrow(a.run), f"{a.run} was not written by jnarrow_train.py (no narrow / init_run in its args): `dump` would overwrite the ends_val.pkl of a stored run; --unnarrowed-ok if that is meant"
    install()
    JEV.dump(SimpleNamespace(run=a.run, thr=a.thr, max=a.max))


def score(argv):
    """the stored `jends_eval.py score` with its own argument parser (it builds no model: the narrowed state is read as it is)"""
    old, sys.argv = sys.argv, ["jends_eval.py", "score", *argv]
    try:
        runpy.run_path(JEV.__file__, run_name="__main__")
    finally:
        sys.argv = old


# ---------------------------------------------------------------- time paired
def one_ms(fn, x, cuda):
    """ms of one forward: two CUDA events around it (the host waits before and after), perf_counter on the CPU (the test)"""
    if not cuda:
        t0 = time.perf_counter()
        fn(x)
        return (time.perf_counter() - t0) * 1000.0
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    fn(x)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1)


def paired_blocks(fns, xs, rounds, iters, cuda):
    """{name: [median ms of the block of `iters` forwards, per round]}; the blocks of a round alternate (A B in the even rounds, B A in the odd ones), the frames cycle as in `jtiming.graph`"""
    blocks = {k: [] for k in fns}
    for r in range(rounds):
        for k in JT.schedule(list(fns), r):
            blocks[k].append(float(np.median([one_ms(fns[k], xs[(r * iters + i) % len(xs)], cuda) for i in range(iters)])))
    return blocks


def paired_result(runs, blocks, host):
    """the result file: the medians of the two runs, the paired difference narrowed - control over the rounds (mean, 95 % bootstrap lo, hi; ms) and the verdict on the host; runs = (control, narrowed), host = load_before, load_after, gpu_before, gpu_after, forced"""
    c, n = runs
    quiet, line = JT.quiet_verdict(host["load_before"], host["load_after"], host["gpu_before"], host["gpu_after"])
    c_med, n_med = float(np.median(blocks[c])), float(np.median(blocks[n]))
    d = JT.paired(blocks[n], blocks[c])
    return dict(runs=[c, n], control=c, narrowed=n, control_median_ms=c_med, narrowed_median_ms=n_med, paired_narrowed_minus_control=d, percent=100 * d[0] / c_med, label="QUIET" if quiet else "NOT QUIET", quiet=quiet, quiet_line=line,
                forced=bool(host["forced"]), load_before=host["load_before"], load_after=host["load_after"], gpu_before=host["gpu_before"], gpu_after=host["gpu_after"], block_medians_ms=blocks)


def paired_run(a, dev="cuda"):
    """the measurement of the module docstring (`dev` is "cuda"; the CPU test passes "cpu": no events, no bf16) -> the result dict, written to a.out or runs/eval/jtiming/<a.tag>.json"""
    import lane_gt_data as LD
    cuda = dev == "cuda"
    assert not cuda or torch.cuda.is_available(), "no GPU: this measurement is for an idle GPU"
    out = a.out or os.path.join(ROOT, "runs", "eval", "jtiming", f"{a.tag}.json")
    assert not os.path.exists(out), f"{out} exists: another --tag (a measurement is never overwritten)"
    torch.backends.cudnn.benchmark = False
    load_before = JT.load1()
    before = JT.gpu_state()
    if not JT.gpu_idle(before) and not a.force:
        sys.exit(f"STOP: the GPU is not idle ({before}); timing only on an idle GPU (--force times it anyway and writes the state into the file: the label is then NOT QUIET)")
    print(f"load average (1 min) {load_before}, cudnn.benchmark {torch.backends.cudnn.benchmark}")
    LD.QData.DEFAULT_JUNCTION = True
    ds = LD.QData("val_clean", map_mode="error", pixels=False)
    xs = [LD.collate([ds[i]])["x"].to(dev).float() for i in np.linspace(0, len(ds) - 1, a.frames).astype(int)]
    dtype = torch.bfloat16 if cuda else None
    fns, narrowed = {}, {}
    for r_ in a.runs:
        m, args = JN.load_model(r_, dev)
        narrowed[r_] = bool(args.get("narrow", {}).get("taps"))
        fns[r_] = lambda x, m=m: JG.eager_forward(m, x, dtype=dtype)
        for i in range(a.warm):
            fns[r_](xs[i % len(xs)])
    blocks = paired_blocks(fns, xs, a.rounds, a.iters, cuda)
    if cuda:
        torch.cuda.synchronize()
    time.sleep(SETTLE_S)                                                                    # the last timed kernels must age out of nvidia-smi's averaging window (E351.g1)
    host = dict(load_before=load_before, gpu_before=before, load_after=JT.load1(), gpu_after=JT.gpu_state(), forced=a.force)
    res = dict(paired_result(a.runs, blocks, host), rounds=a.rounds, iters=a.iters, frames=a.frames, warm=a.warm, narrowed_runs=narrowed, cudnn_benchmark=torch.backends.cudnn.benchmark, device=torch.cuda.get_device_name(0) if cuda else "cpu",
               precision="bf16 autocast, batch 1, no_grad" if cuda else "fp32, batch 1, no_grad (CPU test)", timed_region="the model's eager forward (jgraph.eager_forward), CUDA events around it")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x") as fh:                                                              # "x": never over an existing file
        json.dump(res, fh, indent=1, default=float)
    d = res["paired_narrowed_minus_control"]
    print(f"{res['device']}, {res['precision']}; {res['quiet_line']}\nms per frame (median of the {a.rounds} block medians of {a.iters} forwards): control {a.runs[0]} {res['control_median_ms']:.3f}, narrowed {a.runs[1]} {res['narrowed_median_ms']:.3f}\n"
          f"narrowed - control {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] ms = {res['percent']:+.1f} %  ({res['label']})\nwritten {out}")
    return res


def paired_cmd(argv):
    ap = argparse.ArgumentParser(prog="time paired")
    ap.add_argument("--runs", required=True, help="control,narrowed: two run names (the difference is narrowed - control)")
    ap.add_argument("--tag", required=True, help="the result file runs/eval/jtiming/<tag>.json (never overwritten)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="time on a busy GPU anyway: the state goes into the file and the label is NOT QUIET")
    a = ap.parse_args(argv)
    a.runs = a.runs.split(",")
    assert len(a.runs) == 2 and a.runs[0] != a.runs[1], "--runs needs two different run names: control,narrowed"
    paired_run(a)


def time_cmd(argv):
    """`time paired ...` (above) or the stored `jtiming.py` commands with the narrowed loader installed"""
    if argv[:1] == ["paired"]:
        return paired_cmd(argv[1:])
    install()
    old, sys.argv = sys.argv, ["jtiming.py", *argv]
    try:
        JT.main()
    finally:
        sys.argv = old


COMMANDS = {"dump": dump, "score": score, "time": time_cmd}


def main(argv):
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    if sys.argv[1:2] and sys.argv[1] in COMMANDS:
        main(sys.argv[1:])
    else:
        sys.exit(__doc__)
