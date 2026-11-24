"""Read-out of E368 (stage S2 of the token champion): the head with (arm r) or without (arm z) the cross-attention block to the past, run frame by frame in the E365 cache order with the state carried (`training/train/jpast_train.py`, `training/grid/jpast.py`).
It only DUMPS; the scorer, the writer and the decode are those of `jtoken_eval.py` (the dump has its format, the prediction is written from the cached tokens by `lane_ptok_data.to_pred`: no GT field).

    wsl bash tools/env/_py14.sh jpast_eval.py dump --run R --seed S [--split val|train] [--mode stream|truncated|off|shuffled|still] [--window 3] [--cpu] [--max N]   # GPU: -> runs/<R or R_<mode>>/token_<split>[_n<N>].pkl
    wsl bash tools/env/_py14.sh jpast_eval.py use  --run R [--split val] [--max N]                                                                                # CPU: u and the carried state's RMS per stream position, what the past changed (stream against off) -> runs/R/past_use_<split>.json
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R_off --seed S   |   sweep --run R_off --seed S --split train ...                                       # a mode dump is read by the scorer like any run: its folder is the sibling R_<mode>

Modes (one frame per forward, batch 1, the cache order = the stream order (scene, time); the state is empty at every frame without a previous keyframe: the 22 VAL scene starts):
  stream     the state of the previous frame is carried through the whole scene (the verdict mode; its folder is the run's own: `runs/R/token_<split>.pkl`, the file `jtoken_eval.py score` / `sweep` read)
  truncated  the state is also reset every `--window` frames of a scene (3 = the training distribution: a third of the frames are cold)
  off        every frame cold (the checkpoint's own control: no past)          shuffled  the state of ANOTHER scene at the same step index (donor = the next scene, cyclically, that has the frame; VAL only), warped by this frame's pose
  still      the identity pose instead of the ego pose (the state is not moved)   (arm z: no block, every mode is the same head on the same frames)
Every other mode goes to the SIBLING run folder `runs/R_<mode>/` (stream: `runs/R/`): `token_<split>[_n<N>].pkl` with meta['run'] = `R_<mode>`, and `model.pt` = a copy of R's (an existing one must be the same bytes), so that `jtoken_eval.py score --run R_<mode>`
and `sweep` read it as they read a run (they need model.pt for its args and the dump named as above; their cards land in the sibling folder; the parameter count they print is that of the E365 head without the block).
TRAIN (873 evenly spaced frames, as `jtoken_eval.sample`): the stream runs through the whole split, only the sampled frames are dumped. The key weight of the past is the head's own 1 - p_none (the switch of 1000 steps is over in a trained head).
The dump of `jtoken_eval.dump` (meta + frames of `batch_dump`) with meta['use'] = per dumped frame has_prev, position in the scene, ego shift (m), the sums of the kill measures (`jpast.SUMS`: d2_none / x2 / n against the none key only, d2_other / x2_other / n_other against the past of ANOTHER SCENE = the donor of `donor_plan`, VAL and a block only; 0 without a past) and state_ss / state_n = sum of squares and count of the CARRIED state (the head's final token states x of that frame, valid tokens).
`use` writes u = RMS(update with the past - update with another scene's past) / RMS(x) (the kill measure of the card; 0 when the block ignores the past content), u_none (against the none key only: any content-free attention satisfies it, reported next to u, no verdict) and the RMS of the carried state per stream position (0 = the cold start, 1, 2, ... frames since the last reset), the stream against off (what the past changed) and the frame indices of the first 5 frames of every scene against the later ones (the card's topo-F1 split).
Tests: training/grid/test_jpast_train.py (the modes by hand on the stub head, the donor plan, the use report)
"""
import argparse
import filecmp
import json
import math
import os
import pickle
import shutil
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jpast as JP  # noqa: E402
import jpast_train as JPT  # noqa: E402
import jtiming as JTG  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_eval as JE  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402

MODES = ("stream", "truncated", "off", "shuffled", "still")
FIRST = 5                                                                   # the card's split of the topo F1: the first 5 frames of every scene (positions 0 - 4) against the later ones
BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 10), (11, 10 ** 9))                  # stream positions of the u table: 1st, 2nd, 3rd, 4th - 10th, 11th and later frame of a scene (0 = the cold start)


def run_folder(run, mode):
    """the run folder a dump of `mode` is written to: the run itself for stream, the sibling `<run>_<mode>` for every other mode"""
    return run if mode == "stream" else f"{run}_{mode}"


def sibling(runs, run, mode):
    """the folder of `run_folder` with the run's model.pt in it (a copy; an existing one must be the same bytes: another checkpoint under the same name is refused) -> its name"""
    name = run_folder(run, mode)
    if name != run:
        os.makedirs(os.path.join(runs, name), exist_ok=True)
        src, dst = os.path.join(runs, run, "model.pt"), os.path.join(runs, name, "model.pt")
        if os.path.exists(dst):
            assert filecmp.cmp(src, dst, shallow=False), f"{dst} is not a copy of {src}"
        else:
            shutil.copyfile(src, dst)
    return name


def dump_file(runs, name, split, n=0):
    """runs/<name>/token_<split>[_n<n>].pkl (n: the frames of a `--max` dump), the name `jtoken_eval.dump_path` builds for model.pt, no drop, every 1"""
    return os.path.join(runs, name, f"token_{split}{f'_n{n}' if n else ''}.pkl")


def donor_plan(prev):
    """the frame whose state a frame borrows in mode `shuffled`: (F,) list, prev = the stored previous-frame index (-1 at a scene start) of the stream. The frame at step n >= 1 of scene s borrows the state of step n - 1 of the next scene (cyclically) that has such a frame
    (length >= n); None at a scene start, or when no other scene is long enough"""
    prev = np.asarray(prev)
    starts = np.flatnonzero(prev < 0).tolist()
    ends = starts[1:] + [len(prev)]
    scene = np.cumsum(prev < 0) - 1
    out = []
    for i in range(len(prev)):
        s, n = int(scene[i]), i - starts[int(scene[i])]
        donor = None
        if n >= 1:
            for d in range(1, len(starts)):
                t = (s + d) % len(starts)
                if ends[t] - starts[t] >= n:
                    donor = starts[t] + n - 1
                    break
        out.append(donor)
    return out


def is_cold(mode, has_prev, pos_in_scene, window):
    """no state at this frame: mode off, a frame without a previous keyframe, or (truncated) every `window`-th frame of the scene"""
    return mode == "off" or not has_prev or (mode == "truncated" and pos_in_scene % window == 0)


def iter_outputs(m, batches, mode="stream", window=3, dev="cpu", donor=None, states=None, frame_of=None):
    """the head on the frames of `batches` (batch size 1: `lane_past_data.loader(..., stream=True)`: dict(frames=[frame], rel_R (1, 1, 2, 2), rel_t (1, 1, 2), has_prev (1, 1), idx (1, 1))) in order, the state carried as `mode` says -> yields
    (cache index, frame batch on the CPU, the head's output, dict(has_prev, pos = position in the scene, shift = the ego shift in m, the sums of the kill measures `jpast.SUMS` (d2_none, x2, n: against the none key only; d2_other, x2_other, n_other: against
    the past of another scene), state_ss / state_n = the sum of squares and the count of the final token states x of the frame's valid tokens: the state that is carried on)). `donor` = `donor_plan`, `states` = cache index -> (x, w) of a stream pass (CPU tensors),
    `frame_of(j)` = the frame batch of cache index j: in mode shuffled the donor's state IS the past; in the other modes it is the OTHER scene's past that the block's update is compared with (u; a batch of one sample has no other sample in it; without
    `donor` only u_none is measured). A head without a block (arm z) runs every frame cold"""
    block = getattr(m, "past_block", None)
    meter = JP.UseMeter(block) if block is not None else None
    out_prev, fr_prev, pos_in_scene = None, None, 0
    try:
        for b in batches:
            fr, i = b["frames"][0], int(b["idx"][0, 0])
            hp = bool(b["has_prev"][0, 0])
            pos_in_scene = pos_in_scene + 1 if hp else 0
            R, t = b["rel_R"][:, 0].to(dev), b["rel_t"][:, 0].to(dev)
            if mode == "still":
                R, t = torch.eye(2, device=dev)[None].expand_as(R), torch.zeros_like(t)
            frd = JKT.to_device(fr, dev)
            past = {}
            if block is not None and not is_cold(mode, hp, pos_in_scene, window):
                learned = m._step_py >= JK.SWITCH
                other = None
                if donor is not None and donor[i] is not None:
                    j = donor[i]
                    other = JP.make_past(dict(x=states[j][0].to(dev), w=states[j][1].to(dev)), JKT.to_device(frame_of(j), dev), frd, R, t, learned)
                if mode == "shuffled":
                    past = other or {}
                else:
                    past = JP.make_past(out_prev, fr_prev, frd, R, t, learned)
                    meter.donor = other
            with torch.no_grad():
                out = m(frd, past) if block is not None else m(frd)
            sums = meter.take_sums() if meter is not None else dict.fromkeys(JP.SUMS, 0)
            if meter is not None:
                meter.donor = None
            v = out["valid"][0].bool()
            yield i, fr, out, dict(has_prev=hp, pos=pos_in_scene, shift=float(b["rel_t"][0, 0].norm()), state_ss=float((out["x"][0].float() ** 2)[v].sum()), state_n=int(v.sum()) * int(out["x"].shape[-1]), **sums)
            out_prev, fr_prev = out, frd
    finally:
        if meter is not None:
            meter.detach()


def dump_frames(m, batches, keep, mode="stream", window=3, dev="cpu", donor=None, states=None, frame_of=None, log=print):
    """the dump frames (`jtoken_eval.batch_dump`) of the cache indices in `keep` (a set) + meta['use'] arrays; the stream runs on through the frames that are not kept (the state is carried through them) and stops after the last kept one"""
    last, frames, use = max(keep), [], {k: [] for k in ("idx", "has_prev", "pos", "shift", "state_ss", "state_n") + JP.SUMS}
    for i, fr, out, info in iter_outputs(m, batches, mode, window, dev, donor, states, frame_of):
        if i in keep:
            frames += JE.batch_dump(out, fr)
            use["idx"].append(i)
            for k in ("has_prev", "pos", "shift", "state_ss", "state_n") + JP.SUMS:
                use[k].append(info[k])
            if len(frames) % 100 == 0:
                log(f"   {len(frames)}/{len(keep)} frames", flush=True)
        if i >= last:
            break
    return frames, use


def dump(a, runs=JE.RUNS):
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU; --cpu only for tests"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jpast_eval dump {a.run}")
    import lane_past_data as LPD
    m, args = JPT.load_model(a.run, dev, runs)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    assert a.mode != "shuffled" or a.split == "val", "mode shuffled keeps the states of a whole pass in memory: VAL only"
    kw = JE.cache_kw(args)
    ds, dl = LPD.loader(a.split, kw["seed"], 1, 0 if a.cpu else 2, ident=kw["ident"], K=kw["K"], shuffle=False, stream=True, **({"cache": kw["cache"]} if kw.get("cache") else {}))
    idx = JE.sample(len(ds), a.split, 1)[: a.max or None]
    name = run_folder(a.run, a.mode)
    path = dump_file(runs, name, a.split, len(idx) if a.max else 0)
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    sibling(runs, a.run, a.mode)
    donor = states = frame_of = None
    if a.split == "val" and a.mode != "off" and not args.get("no_past"):                  # the states of a whole stream pass: the past of mode shuffled, the other scene's past that u compares with in the other modes (TRAIN: u_none only)
        states = {i: (out["x"].cpu(), out["w"].cpu()) for i, _, out, _ in iter_outputs(m, dl, "stream", dev=dev)}
        donor, frame_of = donor_plan(ds.prev), (lambda j: LP.collate([ds.tok[j]]))
    frames, use = dump_frames(m, dl, set(int(i) for i in idx), a.mode, a.window, dev, donor, states, frame_of)
    meta = dict(run=name, seed=a.seed, split=a.split, drop=0.0, drop_seed=0, idx=idx.tolist(), every=1, tokens=[ds.tok.tokens[i] for i in idx], K=ds.tok.K, ident=kw["ident"], cache=ds.dir, steps=args.get("steps"), model="model.pt", of_run=a.run, mode=a.mode, window=a.window, use=use,
                past="none (arm z)" if args.get("no_past") else type(m.past_block).__name__)
    assert use["idx"] == idx.tolist() and [f["token"] for f in frames] == meta["tokens"], "the dump is not the frames asked for, in order"
    with open(path, "wb") as fh:
        pickle.dump(dict(meta=meta, frames=frames), fh)
    used, used_o = sum(1 for n in use["n"] if n), sum(1 for n in use["n_other"] if n)
    u_txt = (f"u_none {math.sqrt(sum(use['d2_none']) / max(sum(use['x2']), 1e-30)):.4f} over {used} frames with a past" + (f", u {math.sqrt(sum(use['d2_other']) / max(sum(use['x2_other']), 1e-30)):.4f} (against another scene's past) over {used_o}" if used_o else "")
             if used else "no past read (mode off or arm z)")
    print(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames, mode {a.mode}, kept nodes per frame {np.mean([f['keep'].sum() for f in frames]):.1f}, {u_txt}, device {dev}")
    return path


def use_report(stream, off=None):
    """the kill measures overall and by stream position, and (with the dump of mode off) what the past changed: the dumps are the dicts of `dump` (meta + frames) -> dict. u = RMS(update with the past - update with the past of ANOTHER scene) / RMS(x) (the frames
    that have an other-scene past: VAL, a block, a mode with a past; NaN else), u_none = the same against the none key only (any content-free attention satisfies it: no test of use)"""
    u = stream["meta"]["use"]
    d2n, x2, d2o, x2o, pos, hp, sq, sn = (np.asarray(u[k], float) for k in ("d2_none", "x2", "d2_other", "x2_other", "pos", "has_prev", "state_ss", "state_n"))
    ratio = lambda d2, xx, m: float(math.sqrt(d2[m].sum() / max(xx[m].sum(), 1e-30))) if m.any() and xx[m].sum() > 0 else float("nan")  # noqa: E731
    bucket = {f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": (pos >= lo) & (pos <= hi) & (hp > 0) for lo, hi in BUCKETS}
    has_o = np.asarray(u["n_other"], float) > 0
    rep = dict(run=stream["meta"]["run"], split=stream["meta"]["split"], mode=stream["meta"].get("mode", "stream"), frames=len(d2n), frames_with_past=int(hp.sum()), frames_with_other_past=int(has_o.sum()), u=ratio(d2o, x2o, has_o), u_none=ratio(d2n, x2, hp > 0),
               u_by_position={k: ratio(d2o, x2o, mk & has_o) for k, mk in bucket.items()}, u_none_by_position={k: ratio(d2n, x2, mk) for k, mk in bucket.items()},
               state_rms_by_position=[float(math.sqrt(sq[pos == p].sum() / sn[pos == p].sum())) if sn[pos == p].sum() > 0 else float("nan") for p in range(int(pos.max()) + 1)] if len(pos) else [],
               position_per_frame=pos.astype(int).tolist(), frame_idx_first5=[int(i) for i, p in zip(stream["meta"]["idx"], pos) if p < FIRST], frame_idx_later=[int(i) for i, p in zip(stream["meta"]["idx"], pos) if p >= FIRST])
    if off is not None:
        assert [f["token"] for f in off["frames"]] == [f["token"] for f in stream["frames"]], "the two dumps are not the same frames"
        gain = lose = flips = changed = frames_any = 0
        link_set = lambda f: set(zip(f["rec"]["src"].tolist(), f["rec"]["dst"].tolist()))  # noqa: E731
        for fs, fo in zip(stream["frames"], off["frames"]):
            ks, ko = np.asarray(fs["keep"], bool), np.asarray(fo["keep"], bool)
            g, l_ = int((ks & ~ko).sum()), int((~ks & ko).sum())
            gain, lose, flips = gain + g, lose + l_, flips + g + l_
            changed += len(link_set(fs) ^ link_set(fo))
            frames_any += int(g + l_ > 0)
        n = max(len(stream["frames"]), 1)
        rep["against_off"] = dict(kept_gained_per_frame=gain / n, kept_lost_per_frame=lose / n, kept_flips_per_frame=flips / n, frames_with_a_flip=frames_any / n, links_changed_per_frame=changed / n)
    return rep


def use(a, runs=JE.RUNS):
    paths = {m: dump_file(runs, run_folder(a.run, m), a.split, a.max) for m in ("stream", "off")}
    rep = use_report(pickle.load(open(paths["stream"], "rb")), pickle.load(open(paths["off"], "rb")) if os.path.exists(paths["off"]) else None)
    path = os.path.join(runs, a.run, f"past_use_{a.split}{f'_n{a.max}' if a.max else ''}.json")
    assert not os.path.exists(path), f"a stored result is never overwritten: {path}"
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))
    return rep


def time_head(m, batches, dev, repeats):
    """-> (warm_ms, cold_ms): the per-frame time of the HEAD ALONE (batch 1, bf16 autocast on the GPU, CUDA events with a sync in front and behind: `jtiming.timed`) over `repeats` passes of `batches` (after one untimed pass) in stream order: the past of frame t = the head's own states of
    frame t - 1 moved by the stored pose, exactly what the stream read-out does, without its kill-measure hooks. A frame with a past (or any frame of a head without a block) is warm; the first frame of a scene (a block, no past) is cold"""
    block = getattr(m, "past_block", None)
    cuda = dev == "cuda"
    warm, cold = [], []
    for rep in range(repeats + 1):
        out_prev = fr_prev = None
        row_warm, row_cold = [], []
        for b in batches:
            hp = bool(b["has_prev"][0, 0])
            frd = JKT.to_device(b["frames"][0], dev)
            R_, t_ = b["rel_R"][:, 0].to(dev), b["rel_t"][:, 0].to(dev)
            is_warm = block is None or (hp and out_prev is not None)
            if block is None:
                fn = lambda: m(frd)  # noqa: E731
            elif is_warm:
                fn = lambda: m(frd, JP.make_past(out_prev, fr_prev, frd, R_, t_, m._step_py >= JK.SWITCH))  # noqa: E731  (the state is built inside the timed region: a deployed stateful head pays for it)
            else:
                fn = lambda: m(frd, {})  # noqa: E731  (the first frame of a scene: the block runs on the none key only, as in training and in the stream dump)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=cuda):
                out, ms = JTG.timed(fn, cuda)
            (row_warm if is_warm else row_cold).append(ms)
            out_prev, fr_prev = out, frd
        if rep:
            warm += row_warm
            cold += row_cold
    return warm, cold


def gcost_usable(quiet, dev):
    """a timing feeds the G-cost only when the host was QUIET and the device was the GPU"""
    return bool(quiet and dev == "cuda")


def block_costs(arms):
    """arms = {run: {has_block, median_ms, ...}} -> {run: its median minus the median of the ONE arm without a block} for every arm with a block; {} unless exactly one arm has no block"""
    base = [v for v in arms.values() if not v["has_block"]]
    if len(base) != 1:
        return {}
    return {k: v["median_ms"] - base[0]["median_ms"] for k, v in arms.items() if v["has_block"]}


def time_cmd(a, runs=JE.RUNS):
    """E368 / E387 inference cost: the head alone of each run in `--runs` (a run WITH a block first, the run without one second: the block cost = the difference of the medians) on the first `--frames` VAL frames of the stream, `--repeats` passes; host load and GPU state before and after (the after-sample waits 2.5 s: `token_time_ens.after_state`) -> a json
    `runs/<first run>/past_timing_<UTC>.json` with the QUIET verdict (`usable_for_gcost`)"""
    import token_time_ens as TT
    import lane_past_data as LPD
    ok, why = (False, "--cpu") if a.cpu else JE.gpu_free()
    if not a.cpu and not ok:
        sys.exit(f"STOP: no GPU timing ({why})")
    dev = "cpu" if a.cpu else "cuda"
    res = dict(device=dev, frames=a.frames, repeats=a.repeats, gpu_rule=why, arms={})
    load0, gpu0 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    for run in a.runs.split(","):
        m, args = JPT.load_model(run, dev, runs)
        m.eval()
        kw = JE.cache_kw(args)
        _, dl = LPD.loader("val", kw["seed"], 1, 0, ident=kw["ident"], K=kw["K"], shuffle=False, stream=True, **({"cache": kw["cache"]} if kw.get("cache") else {}))
        batches = []
        for b in dl:
            batches.append(b)
            if len(batches) == a.frames:
                break
        warm, cold = time_head(m, batches, dev, a.repeats)
        block = getattr(m, "past_block", None)
        res["arms"][run] = dict(params=JK.count_params(m), params_block=JK.count_params(block) if block is not None else 0, has_block=block is not None, n_timed=len(warm), median_ms=float(np.median(warm)), p90_ms=float(np.percentile(warm, 90)),
                                cold_frames=len(cold), cold_median_ms=float(np.median(cold)) if cold else None)
    load1, gpu1 = TT.after_state(JTG, a.cpu)
    quiet, line = JTG.quiet_verdict(load0, load1, dict(util_max=0.0) if a.cpu else gpu0, dict(util_max=0.0) if a.cpu else gpu1)
    res.update(host=line, quiet=bool(quiet), usable_for_gcost=gcost_usable(quiet, dev), block_cost_ms=block_costs(res["arms"]))
    print(json.dumps(res, indent=1))
    first = a.runs.split(",")[0]
    print("written", os.path.relpath(JE._write_stamped(os.path.join(runs, first), "past_timing" + ("_cpu" if a.cpu else ""), res), JE.ROOT))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dump")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--mode", default="stream", choices=MODES)
    s.add_argument("--window", type=int, default=3)
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--max", type=int, default=0)
    s = sub.add_parser("use")
    s.add_argument("--run", required=True)
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--max", type=int, default=0, help="the dumps of `dump --max N` (their names carry _n<N>)")
    s = sub.add_parser("time")
    s.add_argument("--runs", required=True, help="the runs with a block and ONE run without one (the block cost of each = the difference of the medians)")
    s.add_argument("--frames", type=int, default=24)
    s.add_argument("--repeats", type=int, default=4)
    s.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    {"dump": dump, "use": use, "time": time_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
