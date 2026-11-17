"""Read-out of E368 (stage S2 of the token champion): the head with (arm r) or without (arm z) the cross-attention block to the past, run frame by frame in the E365 cache order with the state carried (`training/train/jpast_train.py`, `training/grid/jpast.py`).
It only DUMPS; the scorer, the writer and the decode are those of `jtoken_eval.py` (the dump has its format, the prediction is written from the cached tokens by `lane_ptok_data.to_pred`: no GT field).

    wsl bash tools/env/_py14.sh jpast_eval.py dump --run R --seed S [--split val|train] [--mode stream|truncated|off|shuffled|still] [--window 3] [--cpu] [--max N]   # GPU: -> runs/R/token_<split>[_<mode>][_n<N>].pkl
    wsl bash tools/env/_py14.sh jpast_eval.py use  --run R [--split val]                                                                                          # CPU: u per stream position and what the past changed (stream against off) -> runs/R/past_use_<split>.json

Modes (one frame per forward, batch 1, the cache order = the stream order (scene, time); the state is empty at every frame without a previous keyframe: the 22 VAL scene starts):
  stream     the state of the previous frame is carried through the whole scene (the verdict mode; its file is `token_<split>.pkl`, the one `jtoken_eval.py score` / `sweep` read)
  truncated  the state is also reset every `--window` frames of a scene (3 = the training distribution: a third of the frames are cold)
  off        every frame cold (the checkpoint's own control: no past)          shuffled  the state of ANOTHER scene at the same step index (donor = the next scene, cyclically, that has the frame; VAL only), warped by this frame's pose
  still      the identity pose instead of the ego pose (the state is not moved)   (arm z: no block, every mode is the same head on the same frames)
TRAIN (873 evenly spaced frames, as `jtoken_eval.sample`): the stream runs through the whole split, only the sampled frames are dumped. The key weight of the past is the head's own 1 - p_none (the switch of 1000 steps is over in a trained head).
The dump of `jtoken_eval.dump` (meta + frames of `batch_dump`) with meta['use'] = per dumped frame has_prev, position in the scene, ego shift (m), sums d2 / x2 / n of the kill measure u = RMS(block(x, past) - block(x, none key only)) / RMS(x) (NaN-free: 0 without a past).
Tests: training/grid/test_jpast_train.py (the modes by hand on the stub head, the donor plan, the use report)
"""
import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jpast as JP  # noqa: E402
import jpast_train as JPT  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_eval as JE  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402

MODES = ("stream", "truncated", "off", "shuffled", "still")
BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 10), (11, 10 ** 9))                  # stream positions of the u table: 1st, 2nd, 3rd, 4th - 10th, 11th and later frame of a scene (0 = the cold start)


def mode_sfx(mode):
    return "" if mode == "stream" else f"_{mode}"


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
    (cache index, frame batch on the CPU, the head's output, dict(has_prev, pos = position in the scene, shift = the ego shift in m, d2 / x2 / n = the sums of the kill measure)). shuffled: `donor` = `donor_plan`, `states` = cache index -> (x, w) of a
    stream pass (CPU tensors), `frame_of(j)` = the frame batch of cache index j. A head without a block (arm z) runs every frame cold"""
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
                if mode == "shuffled":
                    j = donor[i]
                    if j is not None:
                        x, w = states[j]
                        past = JP.make_past(dict(x=x.to(dev), w=w.to(dev)), JKT.to_device(frame_of(j), dev), frd, R, t, learned)
                else:
                    past = JP.make_past(out_prev, fr_prev, frd, R, t, learned)
            with torch.no_grad():
                out = m(frd, past) if block is not None else m(frd)
            d2, x2, n = meter.take_sums() if meter is not None else (0.0, 0.0, 0)
            yield i, fr, out, dict(has_prev=hp, pos=pos_in_scene, shift=float(b["rel_t"][0, 0].norm()), d2=d2, x2=x2, n=n)
            out_prev, fr_prev = out, frd
    finally:
        if meter is not None:
            meter.detach()


def dump_frames(m, batches, keep, mode="stream", window=3, dev="cpu", donor=None, states=None, frame_of=None, log=print):
    """the dump frames (`jtoken_eval.batch_dump`) of the cache indices in `keep` (a set) + meta['use'] arrays; the stream runs on through the frames that are not kept (the state is carried through them) and stops after the last kept one"""
    last, frames, use = max(keep), [], dict(idx=[], has_prev=[], pos=[], shift=[], d2=[], x2=[], n=[])
    for i, fr, out, info in iter_outputs(m, batches, mode, window, dev, donor, states, frame_of):
        if i in keep:
            frames += JE.batch_dump(out, fr)
            use["idx"].append(i)
            for k in ("has_prev", "pos", "shift", "d2", "x2", "n"):
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
    path = os.path.join(runs, a.run, f"token_{a.split}{mode_sfx(a.mode)}{f'_n{len(idx)}' if a.max else ''}.pkl")
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    donor = states = frame_of = None
    if a.mode == "shuffled":
        states = {i: (out["x"].cpu(), out["w"].cpu()) for i, _, out, _ in iter_outputs(m, dl, "stream", dev=dev)}
        donor, frame_of = donor_plan(ds.prev), (lambda j: LP.collate([ds.tok[j]]))
    frames, use = dump_frames(m, dl, set(int(i) for i in idx), a.mode, a.window, dev, donor, states, frame_of)
    meta = dict(run=a.run, seed=a.seed, split=a.split, drop=0.0, drop_seed=0, idx=idx.tolist(), every=1, tokens=[ds.tok.tokens[i] for i in idx], K=ds.tok.K, ident=kw["ident"], cache=ds.dir, steps=args.get("steps"), mode=a.mode, window=a.window, use=use,
                past="none (arm z)" if args.get("no_past") else "PastBlock")
    assert use["idx"] == idx.tolist() and [f["token"] for f in frames] == meta["tokens"], "the dump is not the frames asked for, in order"
    with open(path, "wb") as fh:
        pickle.dump(dict(meta=meta, frames=frames), fh)
    n = np.asarray(use["n"], float)
    u = math.sqrt(sum(use["d2"]) / max(sum(use["x2"]), 1e-30)) if n.sum() else float("nan")
    print(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames, mode {a.mode}, kept nodes per frame {np.mean([f['keep'].sum() for f in frames]):.1f}, u {u:.4f} over {int(sum(use['has_prev']))} frames with a past, device {dev}")
    return path


def use_report(stream, off=None):
    """u of the kill measure overall and by stream position, and (with the dump of mode off) what the past changed: the dumps are the dicts of `dump` (meta + frames) -> dict"""
    u = stream["meta"]["use"]
    d2, x2, pos, hp = (np.asarray(u[k], float) for k in ("d2", "x2", "pos", "has_prev"))
    ratio = lambda m: float(math.sqrt(d2[m].sum() / max(x2[m].sum(), 1e-30))) if m.any() and x2[m].sum() > 0 else float("nan")  # noqa: E731
    rep = dict(run=stream["meta"]["run"], split=stream["meta"]["split"], mode=stream["meta"].get("mode", "stream"), frames=len(d2), frames_with_past=int(hp.sum()), u=ratio(hp > 0),
               u_by_position={f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": ratio((pos >= lo) & (pos <= hi) & (hp > 0)) for lo, hi in BUCKETS})
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
    load = lambda sfx: pickle.load(open(os.path.join(runs, a.run, f"token_{a.split}{sfx}.pkl"), "rb"))  # noqa: E731
    off_path = os.path.join(runs, a.run, f"token_{a.split}{mode_sfx('off')}.pkl")
    rep = use_report(load(""), load(mode_sfx("off")) if os.path.exists(off_path) else None)
    path = os.path.join(runs, a.run, f"past_use_{a.split}.json")
    assert not os.path.exists(path), f"a stored result is never overwritten: {path}"
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))
    return rep


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
    a = ap.parse_args(argv)
    {"dump": dump, "use": use}[a.cmd](a)


if __name__ == "__main__":
    main()
