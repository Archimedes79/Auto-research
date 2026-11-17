"""Read-out of E374 (stage S3 of the token champion): the head with (arm m) or without (arm z) the cross-attention block to the SD-map tokens, frame by frame in the E365 cache order, with a map CONTROL (`training/train/jmap_train.py`, `training/grid/jmap.py`, `training/data/lane_map_data.py`).
It only DUMPS; the scorer, the writer and the decode are those of `jtoken_eval.py` (the dump has its format, the prediction is written from the cached tokens by `lane_ptok_data.to_pred`: no GT field).

    wsl bash tools/env/_py14.sh jmap_eval.py dump --run R --seed S [--split val|train] [--mode err|clean|gross|none|S|U|C2] [--cpu] [--max N]   # GPU: -> runs/<R or R_<mode>>/token_<split>[_n<N>].pkl
    wsl bash tools/env/_py14.sh jmap_eval.py use  --run R [--split val] [--max N]                                                                  # CPU: v and v_none by frame (and overall) of the record dump -> runs/R/map_use_<split>.json
    wsl bash tools/env/_py14.sh jmap_eval.py params --run R                                                                                        # CPU: the parameters of the head WITH the block (strict load), of the head without it and of the block -> printed
    wsl bash tools/env/_py14.sh jmap_eval.py time  --run R [--frames 20] [--repeats 5] [--K N] [--cpu]                                             # ms per frame (batch 1, bf16 autocast, CUDA events, the VAL frames with their `error` maps), GMACs with and without the block -> runs/R/map_timing[_cpu]_<UTC>.json
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R_none --seed S   |   sweep --run R --seed S --split train ...                          # a mode dump is read by the scorer like any run: its folder is the sibling R_<mode>

Modes (the map the block reads; one frame per forward, batch 1; the maps are those of `lane_map_data.MapData`):
  err    the `error` map: a FIXED map per frame (shift 1 - 2 m, yaw N(0, 1 degree), degrade_sd), the VAL record (its folder is the run's own: `runs/R/token_<split>.pkl`, the file `jtoken_eval.py score` / `sweep` read; the TRAIN sample is read with the same map)
  clean  the exact road map          gross  the `error` map shifted rigidly by 5 m          none  no map: the none key only
  S      the `error` map of ANOTHER SCENE (a fixed plan: frame (i + 97) mod N, moved on to another scene)       U  every lane count 'unknown' (0), the geometry kept       C2  every lane count 2 (a road-collapsed control)
Every other mode goes to the SIBLING run folder `runs/R_<mode>/` (err: `runs/R/`): `token_<split>[_n<N>].pkl` with meta['run'] = `R_<mode>`, and `model.pt` = a copy of R's (an existing one must be the same bytes: `jpast_eval.sibling`), so that `jtoken_eval.py score --run R_<mode>` and `sweep` read it as they read a run (their
cards land in the sibling folder; the parameter count they print is that of the E365 head without the block). Free read-outs of one checkpoint, never a verdict. Arm z (`--no-map` in the run's args): no block, no map is built, every mode is the same head on the same frames.
TRAIN (873 evenly spaced frames, as `jtoken_eval.sample`). The dump of `jtoken_eval.dump` (meta + frames of `batch_dump`) with meta['use'] = per dumped frame n_map (the samples of its map) and the sums of the kill measures (`jpast.SUMS`: d2_none / x2 / n against the none key only, d2_other / x2_other / n_other against the `S` map of the frame
= the map of ANOTHER SCENE, the donor of the read-out; in mode S the frame's own map IS that map: v = 0 by construction). `use` writes v = RMS(update with the map - update with another scene's map) / RMS(x) (the kill measure of the card; 0 when the block ignores the map content), v_none (against the none key only: any
content-free attention satisfies it, no verdict) and both per frame (NaN where undefined). `load_model` is the strict loader of an arm-m run (the head WITH the block; `jtoken_eval.py timing` builds the head without it and cannot load an arm-m checkpoint); `params` and `time` are `jtoken_eval.timing`'s protocol for it (its quiet-host rule, `gpu_free`, the CUDA-event timer
`jrecur_time.timed`; only the loaders and the call of the head differ). Tests: training/grid/test_jmap_train.py (the modes by hand on the stub head and the real head, the use report, the dump on real frames of VAL and of the TRAIN sample, `params`, `time`)
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
import jmap as JM  # noqa: E402
import jmap_train as JMT  # noqa: E402
import jpast_eval as JPE  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_eval as JE  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import lane_map_data as LMD  # noqa: E402

MODES = ("err", "clean", "gross", "none", "S", "U", "C2")
LOADER_MODE = dict(err="error", clean="exact", gross="gross", none="none", S="S", U="U", C2="C2")      # the read-out name -> the mode of `lane_map_data.MapData`
RECORD = "err"
USE_KEYS = ("n_map",) + JM.SUMS


def run_folder(run, mode):
    """the run folder a dump of `mode` is written to: the run itself for the record mode, the sibling `<run>_<mode>` for every other"""
    return run if mode == RECORD else f"{run}_{mode}"


def load_model(run, dev, runs=JE.RUNS, name="model.pt"):
    """-> (the head of runs/<run>/<name> in eval mode, its args): the head WITH the block of an arm-m run (strict state dict), the plain head of an arm-z run (`jmap_train.load_model`; the head must match the run's `no_map`)"""
    m, args = JMT.load_model(run, dev, runs, name)
    assert (getattr(m, "map_block", None) is None) == bool(args.get("no_map")), f"{run}: the head does not match its recipe (no_map {args.get('no_map')})"
    return m, args


def to_device(maps, dev):
    return {k: v.to(dev) for k, v in maps.items()}


def iter_outputs(m, batches, dev="cpu"):
    """the head on the frames of `batches` (batch size 1: `lane_map_data.collate` of a `MapData(with_donor=True)`) in order -> yields (the frame's batch on the CPU, the head's output, dict(n_map = the samples of the frame's map, the sums of the kill measures `jpast.SUMS` (d2_none, x2, n: against the none key
    only; d2_other, x2_other, n_other: against the donor = the `S` map of the frame))). A head without a block (arm z) gets no maps and reports zeros"""
    block = getattr(m, "map_block", None)
    meter = JM.UseMeter(block) if block is not None else None
    try:
        for b in batches:
            frd = JKT.to_device(b, dev)
            maps = to_device(b["maps"], dev) if block is not None else None
            if meter is not None:
                meter.donor = to_device(b["donor"], dev)
            with torch.no_grad():
                out = m(frd, maps=maps) if block is not None else m(frd)
            sums = meter.take_sums() if meter is not None else dict.fromkeys(JM.SUMS, 0)
            yield b, out, dict(n_map=int(b["maps"]["valid"].sum()) if block is not None else 0, **sums)
    finally:
        if meter is not None:
            meter.detach()


def dump_frames(m, batches, dev="cpu", log=print):
    """the dump frames (`jtoken_eval.batch_dump`) of the frames of `batches` in order + meta['use'] arrays (`USE_KEYS`, one entry per frame)"""
    frames, use = [], {k: [] for k in USE_KEYS}
    for b, out, info in iter_outputs(m, batches, dev):
        frames += JE.batch_dump(out, b)
        for k in USE_KEYS:
            use[k].append(info[k])
        if len(frames) % 100 == 0:
            log(f"   {len(frames)} frames", flush=True)
    return frames, use


def dump(a, runs=JE.RUNS):
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU; --cpu only for tests"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jmap_eval dump {a.run} {a.mode}")
    m, args = load_model(a.run, dev, runs)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    kw = JE.cache_kw(args)
    no_map = bool(args.get("no_map"))
    ds = LMD.MapData(a.split, kw["seed"], kw["ident"], kw["K"], LOADER_MODE[a.mode], maps=not no_map, with_donor=not no_map, **({"cache": kw["cache"]} if kw.get("cache") else {}))
    idx = JE.sample(len(ds), a.split, 1)[: a.max or None]
    name = run_folder(a.run, a.mode)
    path = JPE.dump_file(runs, name, a.split, len(idx) if a.max else 0)
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    if a.mode != RECORD:
        JPE.sibling(runs, a.run, a.mode)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [int(i) for i in idx]), batch_size=1, shuffle=False, num_workers=0 if a.cpu else 2, collate_fn=LMD.collate)
    frames, use = dump_frames(m, dl, dev)
    meta = dict(run=name, seed=a.seed, split=a.split, drop=0.0, drop_seed=0, idx=idx.tolist(), every=1, tokens=[ds.tokens[i] for i in idx], K=ds.tok.K, ident=kw["ident"], cache=ds.tok.dir, steps=args.get("steps"), model="model.pt", of_run=a.run, mode=a.mode,
                maps="none (arm z)" if no_map else "MapBlock", use=use)
    assert [f["token"] for f in frames] == meta["tokens"] and len(use["n_map"]) == len(frames), "the dump is not the frames asked for, in order"
    with open(path, "wb") as fh:
        pickle.dump(dict(meta=meta, frames=frames), fh)
    used = sum(1 for n in use["n_other"] if n)
    v_txt = (f"v_none {math.sqrt(sum(use['d2_none']) / max(sum(use['x2']), 1e-30)):.4f}, v {math.sqrt(sum(use['d2_other']) / max(sum(use['x2_other']), 1e-30)):.4f} (against another scene's map) over {used} frames" if used else "no map read (arm z)")
    print(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames, mode {a.mode}, kept nodes per frame {np.mean([f['keep'].sum() for f in frames]):.1f}, mean map samples {np.mean(use['n_map']):.1f}, {v_txt}, device {dev}")
    return path


def use_report(d):
    """the kill measures overall and by frame from a dump (the dict of `dump`: meta + frames) -> dict. v = RMS(update with the map - update with the map of another scene) / RMS(x) over the frames that have a donor (NaN else), v_none = the same against the none key only (any content-free
    attention satisfies it: no test of use)"""
    u = {k: np.asarray(v, float) for k, v in d["meta"]["use"].items()}
    ok_o, ok_n = u["n_other"] > 0, u["n"] > 0
    ratio = lambda d2, xx, m: float(math.sqrt(d2[m].sum() / max(xx[m].sum(), 1e-30))) if m.any() and xx[m].sum() > 0 else float("nan")  # noqa: E731
    per = lambda d2, xx, m: [float(math.sqrt(a / b)) if k and b > 0 else float("nan") for a, b, k in zip(d2, xx, m)]  # noqa: E731
    return dict(run=d["meta"]["run"], split=d["meta"]["split"], mode=d["meta"].get("mode", RECORD), frames=len(u["n"]), frames_with_map=int((u["n_map"] > 0).sum()), frames_with_donor=int(ok_o.sum()), v=ratio(u["d2_other"], u["x2_other"], ok_o), v_none=ratio(u["d2_none"], u["x2"], ok_n),
                v_frame=per(u["d2_other"], u["x2_other"], ok_o), v_none_frame=per(u["d2_none"], u["x2"], ok_n), n_map_frame=[int(n) for n in u["n_map"]])


def use(a, runs=JE.RUNS):
    d = pickle.load(open(JPE.dump_file(runs, a.run, a.split, a.max), "rb"))
    rep = use_report(d)
    path = os.path.join(runs, a.run, f"map_use_{a.split}{f'_n{a.max}' if a.max else ''}.json")
    assert not os.path.exists(path), f"a stored result is never overwritten: {path}"
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("_frame")}, indent=1))
    return rep


def params(a, runs=JE.RUNS):
    """the parameters of an arm-m / arm-z run: the whole head (the block included), the head without the block (the E365 head of the recipe) and the block"""
    m, args = load_model(a.run, "cpu", runs)
    res = dict(run=a.run, no_map=bool(args.get("no_map")), params=JK.count_params(m), params_head=JK.count_params(JKT.build_head(args)), params_block=JK.count_params(m.map_block) if getattr(m, "map_block", None) is not None else 0)
    assert res["params"] == res["params_head"] + res["params_block"], res
    print(json.dumps(res, indent=1))
    return res


def head_macs(m, b, maps=None):
    """GMACs of the head for one frame (torch FlopCounterMode on the CPU: `jtoken_eval.MACS_NOTE` says what it counts); maps None: the block is not called"""
    from torch.utils.flop_counter import FlopCounterMode
    m = m.cpu().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(b, maps=maps) if maps is not None else m(b)
    return fc.get_total_flops() / 2e9


def timing(a, runs=JE.RUNS):
    """parameters, GMACs and ms per frame of an arm-m / arm-z run: batch 1, bf16 autocast on the GPU, CUDA events, `--frames` cached VAL frames with their `error` maps (the block's cost depends on the map size: the maps are the real ones), `--repeats` passes after one untimed pass; K = --K or the K of
    the cache (at most that). The GPU only when `jtoken_eval.gpu_free` (an idle GPU, no run.py job); --cpu times the code path on the CPU. Writes runs/R/map_timing[_cpu]_<UTC>.json (a new file every time) with `quiet` (host load < 3 and an idle GPU before and after) and `usable_for_gcost` (true only for a
    quiet GPU result). gmacs_head = the head with the block on these maps, gmacs_no_map = the same checkpoint with maps None (the block not called)"""
    import jrecur_time as JRT
    import jtiming as JTG
    m, args = load_model(a.run, "cpu", runs)
    ok, why = (False, "--cpu") if a.cpu else JE.gpu_free()
    if not a.cpu and not ok:
        sys.exit(f"STOP: no GPU timing ({why}); --cpu times the code path on the CPU only")
    dev = "cpu" if a.cpu else "cuda"
    kw = JE.cache_kw(args)
    no_map = bool(args.get("no_map"))
    ds = LMD.MapData("val", kw["seed"], kw["ident"], kw["K"], "error", maps=not no_map, **({"cache": kw["cache"]} if kw.get("cache") else {}))
    K = a.K or ds.tok.K
    assert K <= ds.tok.K, f"--K {K} is above the K of the cache ({ds.tok.K})"
    cut = lambda b: dict({k: v[:, :K] for k, v in b.items() if k in JK.INPUTS}, **({"maps": b["maps"]} if "maps" in b else {}))  # noqa: E731
    items = [cut(LMD.collate([ds[i]])) for i in range(a.frames)]
    n_map = float(np.mean([int(b["maps"]["valid"].sum()) for b in items])) if not no_map else 0.0
    first = {k: v for k, v in items[0].items() if k != "maps"}
    gmacs = head_macs(m, first, items[0].get("maps"))
    gmacs_no_map = head_macs(m, first) if not no_map else gmacs
    m = m.to(dev)
    load0, gpu0 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    items = [(JKT.to_device({k: v for k, v in b.items() if k != "maps"}, dev), None if no_map else to_device(b["maps"], dev)) for b in items]
    ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")  # noqa: E731
    ms = []
    for rep in range(a.repeats + 1):
        row = []
        for b, mp in items:
            with torch.no_grad(), ac():
                row.append(JRT.timed(lambda: m(b, maps=mp) if mp is not None else m(b), dev == "cuda")[1])
        if rep:                                                                            # the first pass is untimed
            ms.append(row)
    load1, gpu1 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    quiet, line = JTG.quiet_verdict(load0, load1, dict(util_max=0.0) if a.cpu else gpu0, dict(util_max=0.0) if a.cpu else gpu1)
    flat = np.concatenate(ms)
    res = dict(run=a.run, device=dev, K=K, frames=a.frames, repeats=a.repeats, no_map=no_map, params=JK.count_params(m), params_block=JK.count_params(m.map_block) if not no_map else 0, gmacs_head=gmacs, gmacs_no_map=gmacs_no_map, host=line, quiet=bool(quiet),
               usable_for_gcost=bool(quiet and dev == "cuda"), gpu_rule=why, bf16=dev == "cuda", map_samples_mean=n_map, n_timings=int(flat.size), median_ms=float(np.median(flat)), p90_ms=float(np.percentile(flat, 90)), macs_note=JE.MACS_NOTE)
    print(json.dumps(res, indent=1))
    print(f"written {os.path.relpath(JE._write_stamped(os.path.join(runs, a.run), 'map_timing' + ('_cpu' if a.cpu else ''), res), ROOT)}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dump")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--mode", default=RECORD, choices=MODES)
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--max", type=int, default=0)
    s = sub.add_parser("use")
    s.add_argument("--run", required=True)
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--max", type=int, default=0, help="the dumps of `dump --max N` (their names carry _n<N>)")
    s = sub.add_parser("params")
    s.add_argument("--run", required=True)
    s = sub.add_parser("time")
    s.add_argument("--run", required=True)
    s.add_argument("--frames", type=int, default=20)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--K", type=int)
    s.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    {"dump": dump, "use": use, "params": params, "time": timing}[a.cmd](a)


if __name__ == "__main__":
    main()
