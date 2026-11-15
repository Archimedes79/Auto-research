"""E354: ms per frame of the junction system WITH the recursive feature state carried, against E341 and the equal-budget twin z (the cost line of the card, D86): batch 1, the cached input features of the VAL frames of ONE scene in stream order, the state of the
previous keyframe passed in and the new state taken out of every forward.

    wsl bash tools/env/_py14.sh jrecur_time.py time --runs e341_p_s0,e354_z_s0,e354_r_s0 --scene-frames 40 --repeats 4 --tag <tag>    # GPU, quiet host: runs/eval/jtiming/<tag>/<run>.json + pairs.json, the tables on stdout
    wsl bash tools/env/_py14.sh jrecur_time.py selftest                                                                              # CPU, about 1 minute

Protocol = `jtiming.py run` / `graph` (their helpers are imported, not copied): all runs are loaded into the same process, bf16 autocast, `no_grad`, `cudnn.benchmark` False, two CUDA events around ONE forward with a sync in front, the blocks alternate (A B C, then C B A:
`jtiming.schedule`), a 2.5 s pause before the GPU is read at the end, the host load (1 min) and the GPU before and after (`jtiming.quiet_verdict`: QUIET only below a load of 3 AND a GPU with at most 5 % utilisation). A host that is not quiet REFUSES the run;
`--force` runs it and writes LOADED into every file and table. A block = the stream of the scene from its first keyframe (state empty) to its last, one forward per keyframe; `--repeats` blocks per run, one untimed stream first (warm-up).
Timed region = what `jtiming.py run` times for E312 / E341: ONE call of the whole model on a cached input = the frozen pyramid + the junction maps head + the pair head (`model(x, gt=None)`; the input features are on the GPU, no data loading). With the state it is the same call with
`state=` and `rel=`: the warp (`jrecur.warp_state`, two `grid_sample`), the fusion convolution and the new state taken out (`jout["state"]`) are INSIDE it; the carried state stays on the GPU (no host copy). The model of E341 is loaded as `jtiming.py` loads it
(`jends_eval.load_model`: no fusion module); `--pair-via jr` loads it through `jrecur.load_model` instead (zero gate, state off: the fusion convolution still runs on an empty state). z (state module present, never a state) and r (state carried) load through
`jrecur.load_model`. Frames: the first keyframe of a stream has no state, so the statistics (median, p90) are over the keyframes 1 .. n-1 of the repeats; the first keyframe is reported on its own. The paired differences x - y (later run minus earlier run of `--runs`) are the means
of the per-keyframe medians over the repeats (keyframes 1 .. n-1), 95 % bootstrap over the keyframes (`jtiming.paired`). Per run: parameters, GMACs from `jmodel_cost.cost` (CPU copy, batch 1; `grid_sample` is not counted by the FLOP counter),
the peak of the GPU memory allocated in a stream (above the memory at its start and in total; every run's weights are resident in the process).
"""
import argparse
import collections
import copy
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jends_eval as JEV  # noqa: E402
import jmodel_cost as JC  # noqa: E402
import jrecur as JR  # noqa: E402
import jrecur_eval as JRE  # noqa: E402
import jtiming as JTG  # noqa: E402
import lane_gt_data as LD  # noqa: E402
import lane_seq_data as LS  # noqa: E402

PAIR_VIA = ("frozen", "jr")


class Arm:
    """one run: the model, how it was loaded and whether the state is carried. kind: `plain` (a pair run from before E354 through the frozen loader: no fusion module), `module, no state` (the fusion module present, never a state: z, or E341 via `--pair-via jr`),
    `state carried` (args['state'] == 'on')"""

    def __init__(self, run, dev, pair_via="frozen"):
        assert pair_via in PAIR_VIA, pair_via
        args = torch.load(os.path.join(ROOT, "runs", run, "model.pt"), map_location="cpu")["args"]
        self.run, self.dev, self.cuda = run, dev, dev == "cuda"
        if "state" not in args and pair_via == "frozen":
            self.model, self.args = JEV.load_model(run, dev)[0], args
            self.module, self.carries, self.kind = False, False, "plain"
        else:
            self.model, self.args = JR.load_model(run, dev)
            self.module, self.carries = True, args.get("state") == "on"
            self.kind = "state carried" if self.carries else "module, no state"
        self.model.eval()

    def step(self, x, state=None, rel=None):
        """ONE forward of the whole model on the cached input x -> the new state (None for a model without the module)"""
        with torch.no_grad(), torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.cuda):
            if not self.module:
                self.model(x, gt=None)
                return None
            return self.model(x, gt=None, state=state, rel=rel)[1]["state"]


def timed(fn, cuda):
    """-> (fn(), ms): two CUDA events with a sync in front and behind (as `jtiming.py`), or `perf_counter` on the CPU"""
    if not cuda:
        t = time.perf_counter()
        r = fn()
        return r, (time.perf_counter() - t) * 1000.0
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    r = fn()
    e1.record()
    torch.cuda.synchronize()
    return r, e0.elapsed_time(e1)


def stream_ms(arm, frames):
    """one pass over the frames in stream order -> the ms of every forward. The first keyframe has no state; an arm that carries the state gets the state of the keyframe before and the pose `rel` of the previous lane frame, the others neither;
    the new state is taken out inside the timed call"""
    state, ms = None, []
    for i, (x, rel) in enumerate(frames):
        s, r = (state, rel) if arm.carries and i > 0 else (None, None)
        state, t = timed(lambda: arm.step(x, s, r), arm.cuda)
        ms.append(t)
    return ms


def summarise(ms):
    """ms (repeats, n) -> the numbers over the keyframes 1 .. n-1 (state present for an arm that carries it) and the first keyframe on its own"""
    a = np.asarray(ms, float)
    c = a[:, 1:]
    return dict(median_ms=float(np.median(c)), p90_ms=float(np.percentile(c, 90)), mean_ms=float(c.mean()), first_frame_median_ms=float(np.median(a[:, 0])), n_timed=int(c.size), per_frame_ms=c.mean(0).tolist())


def host_label(load_before, load_after, gpu_before, gpu_after, cpu=False):
    """-> (label, line) QUIET / LOADED by `jtiming.quiet_verdict`; on the CPU (selftest) the GPU does not count"""
    idle = dict(util_max=0.0)
    quiet, line = JTG.quiet_verdict(load_before, load_after, idle if cpu else gpu_before, idle if cpu else gpu_after)
    return ("QUIET" if quiet else "LOADED"), line


def gate(load, gpu, cpu, force):
    """the refusal of `jtiming.py`: a host that is not quiet at the start stops the run (SystemExit) unless `force`; -> (label, line) of the start"""
    label, line = host_label(load, load, gpu, gpu, cpu)
    if label != "QUIET" and not force:
        sys.exit(f"STOP: the host is not quiet ({line}); timing only on a quiet host (--force runs anyway and writes LOADED into the files)")
    return label, line


def scene_frames(n, scene, dev):
    """the cached VAL input features of the first n keyframes of ONE scene in stream order -> ([(x (1, 39, 200, 200) on dev, rel or None)], {scene, rows}); scene None = the first scene with n keyframes (else the longest).
    rel = the pose of the previous lane frame in the current one, as `jrecur_eval.run_stream` builds it"""
    ds, pose, order = JRE.val_stream()
    starts = [i for i, (_, r) in enumerate(order) if r] + [len(order)]
    segs = list(zip(starts[:-1], starts[1:]))
    k = scene if scene is not None else next((j for j, (a, b) in enumerate(segs) if b - a >= n), max(range(len(segs)), key=lambda j: segs[j][1] - segs[j][0]))
    a, b = segs[k]
    rows = [r for r, _ in order[a:min(b, a + n)]]
    assert len(rows) >= 3, f"scene {k} has {len(rows)} keyframes: at least 3 are needed"
    i_of_row = {int(ds.bd.rows[kk]): i for i, kk in enumerate(ds.ks)}
    frames = []
    for j, row in enumerate(rows):
        x = LD.collate([ds[i_of_row[row]]])["x"].to(dev).float()
        rel = None
        if j > 0:
            prow = int(pose["prev"][row])
            assert prow == rows[j - 1], (row, prow, rows[j - 1])
            R, t = LS.rel_pose(pose["R"][row], pose["t"][row], pose["R"][prow], pose["t"][prow])
            rel = (torch.from_numpy(R)[None].to(dev), torch.from_numpy(t)[None].to(dev))
        frames.append((x, rel))
    return frames, dict(scene=int(k), rows=rows)


def model_cost(arm):
    """parameters and GMACs of the model (`jmodel_cost.cost` on a CPU copy, batch 1, empty state: the fusion convolution is counted, `grid_sample` is not) -> dict"""
    c = JC.cost(copy.deepcopy(arm.model).cpu().float())
    res = dict(params_m=c["total"]["params_m"], gmacs_total=c["total"]["macs_g"], gmacs_maps_head=c["junction maps head"]["macs_g"], gmacs_pair_head=c["pair head"]["macs_g"], gmacs_pyramid=c["pyramid (B's, frozen)"]["macs_g"])
    if arm.module:
        res["params_fuse_m"] = JC.npar(arm.model.jmodel.jm.fuse) / 1e6
    return res


def measure(arms, frames, repeats):
    """the blocks of all arms in alternating order -> {run: dict(ms (repeats, n), peak MiB)}; one untimed stream per arm first"""
    cuda = arms[0].cuda
    for arm in arms:
        stream_ms(arm, frames)
    ms = {a.run: [] for a in arms}
    peak = {a.run: dict(extra=0.0, total=0.0) for a in arms}
    for r in range(repeats):
        for arm in JTG.schedule(arms, r):
            if cuda:
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            ms[arm.run].append(stream_ms(arm, frames))
            if cuda:
                p = torch.cuda.max_memory_allocated()
                peak[arm.run] = dict(extra=max(peak[arm.run]["extra"], (p - base) / 2**20), total=max(peak[arm.run]["total"], p / 2**20))
    return {a.run: dict(ms=ms[a.run], gpu_peak_extra_mib=peak[a.run]["extra"] if cuda else None, gpu_peak_total_mib=peak[a.run]["total"] if cuda else None) for a in arms}


def pair_table(res, runs):
    """every (later run - earlier run) of `runs`: {"x - y": (mean, lo, hi)} ms over the keyframes 1 .. n-1 (per-keyframe mean over the repeats, bootstrap over the keyframes)"""
    pf = {r: np.asarray(summarise(res[r]["ms"])["per_frame_ms"]) for r in runs}
    return {f"{runs[j]} - {runs[i]}": JTG.paired(pf[runs[j]], pf[runs[i]]) for i in range(len(runs)) for j in range(i + 1, len(runs))}


def write_results(out_dir, tag, arms, res, pairs, host):
    """one json per run + pairs.json in out_dir; never overwrites"""
    assert not os.path.exists(out_dir), f"{out_dir} exists: another tag (a measurement is never overwritten)"
    os.makedirs(out_dir)
    for a in arms:
        r = res[a.run]
        d = dict(run=a.run, kind=a.kind, tag=tag, label=host["label"], **summarise(r["ms"]), ms_raw=r["ms"], **{k: v for k, v in r.items() if k != "ms"}, **host["common"])
        json.dump(d, open(os.path.join(out_dir, a.run + ".json"), "w"), indent=1, default=float)
    json.dump(dict(tag=tag, label=host["label"], pairs=pairs, pair_unit="ms, mean of the per-keyframe medians over the repeats, 95 % bootstrap over the keyframes 1 .. n-1", **host["common"]), open(os.path.join(out_dir, "pairs.json"), "w"), indent=1, default=float)


def print_tables(arms, res, pairs, host):
    c = host["common"]
    print(f"{host['label']}: {host['line']}\n{c['precision']}; scene {c['scene']}, {c['n_frames']} keyframes (stream order), {c['repeats']} repeats; cudnn.benchmark {c['cudnn_benchmark']}; device {c['device']}")
    print(f"\nms per frame, keyframes 1 .. {c['n_frames'] - 1} of the repeats (the first keyframe has no state)")
    print("  " + "run".ljust(14) + "kind".ljust(18) + "median".rjust(9) + "p90".rjust(9) + "first".rjust(9) + "params M".rjust(10) + "GMACs".rjust(8) + "maps".rjust(7) + "peak MiB".rjust(10))
    for a in arms:
        r, s = res[a.run], summarise(res[a.run]["ms"])
        pk = "-" if r["gpu_peak_extra_mib"] is None else f"{r['gpu_peak_extra_mib']:.0f}"
        print("  " + a.run.ljust(14) + a.kind.ljust(18) + f"{s['median_ms']:9.3f}{s['p90_ms']:9.3f}{s['first_frame_median_ms']:9.3f}{r['params_m']:10.3f}{r['gmacs_total']:8.3f}{r['gmacs_maps_head']:7.3f}{pk:>10}")
    print("\npaired differences (later - earlier), ms [95 % bootstrap over the keyframes]")
    for k, (m, lo, hi) in pairs.items():
        print(f"  {k:32s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


def time_cmd(a):
    cpu = a.cpu
    if cpu or a.threads:
        torch.set_num_threads(a.threads or 2)
    assert cpu or torch.cuda.is_available(), "no GPU: this measurement is for a quiet host with an idle GPU (--cpu only for tests)"
    dev = "cpu" if cpu else "cuda"
    runs = a.runs.split(",")
    out_dir = a.out_dir or os.path.join(ROOT, "runs", "eval", "jtiming", a.tag)
    assert not os.path.exists(out_dir), f"{out_dir} exists: another tag (a measurement is never overwritten)"
    torch.backends.cudnn.benchmark = False
    load_before, gpu_before = JTG.load1(), (None if cpu else JTG.gpu_state())
    gate(load_before, gpu_before, cpu, a.force)
    frames, scene = scene_frames(a.scene_frames, a.scene, dev)
    arms = [Arm(r, dev, a.pair_via) for r in runs]
    res = measure(arms, frames, a.repeats)
    if not cpu:
        torch.cuda.synchronize()
        time.sleep(2.5)                                                                                           # this process's own kernels must age out of nvidia-smi's window (E351)
    load_after, gpu_after = JTG.load1(), (None if cpu else JTG.gpu_state())
    label, line = host_label(load_before, load_after, gpu_before, gpu_after, cpu)
    for arm in arms:                                                                                              # after the host readings: the cost counter loads the CPU for a few seconds
        res[arm.run].update(model_cost(arm))
    host = dict(label=label, line=line, common=dict(load_before=load_before, load_after=load_after, gpu_before=gpu_before, gpu_after=gpu_after, forced=bool(a.force), repeats=a.repeats, n_frames=len(frames), scene=scene["scene"], rows=scene["rows"],
                                                    precision="fp32 (CPU)" if cpu else "bf16 autocast, batch 1, no_grad", cudnn_benchmark=torch.backends.cudnn.benchmark, pair_via=a.pair_via, runs=runs,
                                                    device="cpu" if cpu else torch.cuda.get_device_name(0), timed_region="one call of the whole model on the cached input (pyramid + maps head [+ state warp + fusion] + pair head); the new state is taken out inside it"))
    pairs = pair_table(res, runs)
    write_results(out_dir, a.tag, arms, res, pairs, host)
    print_tables(arms, res, pairs, host)
    print("written", out_dir)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def selftest():
    import tempfile
    from torch.utils._python_dispatch import TorchDispatchMode
    torch.set_num_threads(2)
    # statistics, known answers: keyframes 1 .. n-1 only, p90 of 1 .. 11 is 10, the pairs are later - earlier
    s = summarise([[9.0, 1, 2, 3, 4], [7.0, 3, 2, 1, 4]])
    assert s["median_ms"] == 2.5 and s["first_frame_median_ms"] == 8.0 and s["n_timed"] == 8 and s["per_frame_ms"] == [2.0, 2.0, 2.0, 4.0], s
    assert abs(summarise([[0.0] + list(range(1, 12))])["p90_ms"] - 10.0) < 1e-9
    pt = pair_table({"a": dict(ms=[[5.0, 1.0, 1.0, 1.0]]), "b": dict(ms=[[5.0, 2.0, 2.0, 2.0]]), "c": dict(ms=[[5.0, 4.0, 3.0, 5.0]])}, ["a", "b", "c"])
    assert list(pt) == ["b - a", "c - a", "c - b"] and pt["b - a"] == (1.0, 1.0, 1.0) and abs(pt["c - a"][0] - 3.0) < 1e-9, pt
    # the refusal: loaded or unreadable host stops the run, --force lets it through and labels it LOADED, the GPU counts on the GPU only
    idle, busy = dict(util_max=0.0), dict(util_max=60.0)
    assert gate(2.9, idle, False, False)[0] == "QUIET" and gate(2.9, busy, True, False)[0] == "QUIET"                # on the CPU the GPU does not count
    for load, gpu in ((3.0, idle), (11.0, idle), (None, idle), (1.0, busy), (1.0, None)):
        try:
            gate(load, gpu, False, False)
            raise AssertionError(f"a host with load {load} and GPU {gpu} was not refused")
        except SystemExit as e:
            assert "not quiet" in str(e)
    assert gate(11.0, idle, False, True)[0] == "LOADED" and host_label(1.0, 3.5, idle, idle)[0] == "LOADED" and host_label(1.0, 1.0, idle, busy)[0] == "LOADED" and host_label(1.0, 1.0, idle, busy, cpu=True)[0] == "QUIET"
    # the models and the first keyframes of a scene (real checkpoints, CPU)
    frames = scene_frames(3, None, "cpu")[0]
    assert len(frames) == 3 and frames[0][1] is None and frames[1][1][0].shape == (1, 2, 2) and frames[1][1][1].shape == (1, 2) and frames[0][0].shape == (1, 39, 200, 200), (len(frames), frames[0][0].shape)
    plain, zed, rec = Arm("e341_p_s0", "cpu"), Arm("e354_z_s0", "cpu"), Arm("e354_r_s0", "cpu")
    via = Arm("e341_p_s0", "cpu", "jr")
    assert (plain.kind, zed.kind, rec.kind, via.kind) == ("plain", "module, no state", "state carried", "module, no state") and not plain.module and rec.carries and not zed.carries and not via.carries
    # z = E341 when the gate is zero: the stored pair run through the frozen loader and through the state builder, empty state AND a carried state, every output tensor equal
    x0, (x1, rel1) = frames[0][0], frames[1]
    same = lambda a, b: set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))  # noqa: E731
    with torch.no_grad():
        ref0, ref1 = plain.model(x0, gt=None), plain.model(x1, gt=None)
        n0 = via.model(x0, gt=None)
        assert same(ref0[2], n0[2]), "zero gate, no state: the outputs differ from the stored model's"
        st = n0[1]["state"]
        assert float(via.model.gate().max()) == 0.0
        n1 = via.model(x1, gt=None, state=st, rel=rel1)
        assert same(ref1[2], n1[2]), "zero gate, a carried state: the outputs differ from the stored model's"
        # r: the state is used (the output differs from the empty-state output of the same weights), z's weights have a state module that was never trained with a state
        r0 = rec.model(x0, gt=None)
        r1s = rec.model(x1, gt=None, state=r0[1]["state"], rel=rel1)
        r1e = rec.model(x1, gt=None)
        assert float(rec.model.gate().max()) > 0 and not same(r1s[2], r1e[2]), "the carried state does not change the output of r"

    # the cost structure: the state step adds the warp (2 grid_sample) and drops the two zero tensors of the empty branch, nothing else; per stream n fusion calls, n - 1 warps
    class Ops(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.c = collections.Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.c[str(func)] += 1
            return func(*args, **(kwargs or {}))

    def ops(fn):
        with Ops() as o:
            fn()
        return o.c
    c0 = torch.zeros(1, 64, 50, 50)
    w_ops = ops(lambda: JR.warp_state(c0, *rel1))
    z_ops = ops(lambda: (torch.zeros_like(c0), torch.zeros_like(c0[:, :1])))
    assert w_ops["aten.grid_sampler_2d.default"] == 2, w_ops
    e_ops, s_ops = ops(lambda: rec.step(x1, None, None)), ops(lambda: rec.step(x1, r0[1]["state"], rel1))
    assert s_ops == e_ops - z_ops + w_ops, f"state step - empty step is not the warp minus the zero tensors: {dict(s_ops - e_ops)} / {dict(e_ops - s_ops)}"
    def count_stream(arm):
        """one stream with the calls of the warp and of three blocks counted and the state passed in at every keyframe recorded -> (Counter, ids)"""
        calls, ids, real, orig = collections.Counter(), [], JR.warp_state, arm.step
        JR.warp_state = lambda *a_, **k_: (calls.update(["warp"]), real(*a_, **k_))[1]
        hooks = [m.register_forward_hook(lambda m, i, o, k=k: calls.update([k])) for k, m in (("fuse", arm.model.jmodel.jm.fuse), ("pair", arm.model.je), ("pyr", arm.model.jmodel.pyr))]
        arm.step = lambda x, s, r: ids.append(None if s is None else id(s)) or orig(x, s, r)
        try:
            stream_ms(arm, frames)
        finally:
            JR.warp_state, arm.step = real, orig
            for h in hooks:
                h.remove()
        return calls, ids
    calls, ids = count_stream(rec)
    assert calls == dict(warp=2, fuse=3, pair=3, pyr=3), calls                                                         # n = 3: two warps, every other block three times
    assert ids[0] is None and ids[1] is not None and ids[2] is not None and ids[1] != ids[2], ids                      # the state passed in is the one taken out of the keyframe before
    assert count_stream(zed)[0] == dict(fuse=3, pair=3, pyr=3), "the twin: the same blocks, no warp"
    cost_r, cost_p = model_cost(rec), model_cost(plain)                                                              # known answer: the fusion convolution (2d+1) d 3 3 on 50 x 50 cells = 0.186 GMACs, 74368 parameters with the gate
    assert abs(cost_r["gmacs_maps_head"] - cost_p["gmacs_maps_head"] - 129 * 64 * 9 * 2500 / 1e9) < 1e-3 and abs(cost_r["params_fuse_m"] - 0.074368) < 1e-9 and cost_r["gmacs_pair_head"] == cost_p["gmacs_pair_head"], (cost_r, cost_p)
    # the command on the CPU (3 keyframes, 2 repeats, --force as the host here may be busy): the files, the label, the pairs; a second call with the same tag does not overwrite
    with tempfile.TemporaryDirectory() as td:
        a = argparse.Namespace(runs="e341_p_s0,e354_z_s0,e354_r_s0", scene_frames=3, repeats=2, scene=None, tag="t", pair_via="frozen", threads=2, force=True, cpu=True, out_dir=os.path.join(td, "t"))
        time_cmd(a)
        d = json.load(open(os.path.join(a.out_dir, "e354_r_s0.json")))
        pj = json.load(open(os.path.join(a.out_dir, "pairs.json")))
        assert d["label"] in ("QUIET", "LOADED") and d["forced"] and d["kind"] == "state carried" and d["n_timed"] == 4 and d["p90_ms"] >= d["median_ms"] and d["gmacs_total"] > 6 and d["params_fuse_m"] == cost_r["params_fuse_m"], d
        assert list(pj["pairs"]) == ["e354_z_s0 - e341_p_s0", "e354_r_s0 - e341_p_s0", "e354_r_s0 - e354_z_s0"] and pj["label"] == d["label"]
        try:
            time_cmd(a)
            raise AssertionError("a measurement was overwritten")
        except AssertionError as e:
            assert "exists" in str(e)
    print("jrecur_time selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time")
    t.add_argument("--runs", default="e341_p_s0,e354_z_s0,e354_r_s0", help="the order matters: the paired differences are later - earlier")
    t.add_argument("--scene-frames", type=int, default=40)
    t.add_argument("--repeats", type=int, default=4)
    t.add_argument("--scene", type=int, default=None, help="index of the VAL scene (default: the first with enough keyframes)")
    t.add_argument("--tag", required=True)
    t.add_argument("--pair-via", default="frozen", choices=PAIR_VIA, help="how a pair run from before E354 is loaded: frozen = as jtiming.py (no fusion module), jr = through jrecur.load_model (zero gate)")
    t.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch's own on the GPU, 2 with --cpu)")
    t.add_argument("--force", action="store_true", help="run on a host that is not quiet; the files say LOADED")
    t.add_argument("--cpu", action="store_true", help="tests only")
    t.add_argument("--out-dir", default=None)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest" or a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                                                                      # before CUDA is first touched: the selftest never uses the GPU
    selftest() if a.cmd == "selftest" else time_cmd(a)


if __name__ == "__main__":
    main()
