import os
R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "evaluation/jmap_eval.py")
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


sub("""    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R_none --seed S""", """    wsl bash tools/env/_py14.sh jmap_eval.py params --run R                                                                                        # CPU: the parameters of the head WITH the block (strict load), of the head without it and of the block -> printed
    wsl bash tools/env/_py14.sh jmap_eval.py time  --run R [--frames 20] [--repeats 5] [--K N] [--cpu]                                             # ms per frame (batch 1, bf16 autocast, CUDA events, the VAL frames with their `error` maps), GMACs with and without the block -> runs/R/map_timing[_cpu]_<UTC>.json
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R_none --seed S""")
sub('''Tests: training/grid/test_jmap_train.py (the modes by hand on the stub head and the real head, the use report, the dump on real frames)
"""''', '''`load_model` is the strict loader of an arm-m run (the head WITH the block; `jtoken_eval.py timing` builds the head without it and cannot load an arm-m checkpoint); `params` and `time` are `jtoken_eval.timing`'s protocol for it (its quiet-host rule, `gpu_free`, the CUDA-event timer
`jrecur_time.timed`; only the loaders and the call of the head differ). Tests: training/grid/test_jmap_train.py (the modes by hand on the stub head and the real head, the use report, the dump on real frames of VAL and of the TRAIN sample, `params`, `time`)
"""''')
sub('''def to_device(maps, dev):''', '''def load_model(run, dev, runs=JE.RUNS, name="model.pt"):
    """-> (the head of runs/<run>/<name> in eval mode, its args): the head WITH the block of an arm-m run (strict state dict), the plain head of an arm-z run (`jmap_train.load_model`; the head must match the run's `no_map`)"""
    m, args = JMT.load_model(run, dev, runs, name)
    assert (getattr(m, "map_block", None) is None) == bool(args.get("no_map")), f"{run}: the head does not match its recipe (no_map {args.get('no_map')})"
    return m, args


def to_device(maps, dev):''')
sub("""    m, args = JMT.load_model(a.run, dev, runs)
    assert args["seed"] == a.seed""", """    m, args = load_model(a.run, dev, runs)
    assert args["seed"] == a.seed""")

tail_new = '''def params(a, runs=JE.RUNS):
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
               usable_for_gcost=bool(quiet and dev == "cuda"), gpu_rule=why, bf16=dev == "cuda", map_samples_mean=n_map, median_ms=float(np.median(flat)), p90_ms=float(np.percentile(flat, 90)), macs_note=JE.MACS_NOTE)
    print(json.dumps(res, indent=1))
    print(f"written {os.path.relpath(JE._write_stamped(os.path.join(runs, a.run), 'map_timing' + ('_cpu' if a.cpu else ''), res), ROOT)}")
    return res


def main(argv=None):'''
sub("def main(argv=None):", tail_new)
sub("""    a = ap.parse_args(argv)
    {"dump": dump, "use": use}[a.cmd](a)""", """    s = sub.add_parser("params")
    s.add_argument("--run", required=True)
    s = sub.add_parser("time")
    s.add_argument("--run", required=True)
    s.add_argument("--frames", type=int, default=20)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--K", type=int)
    s.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    {"dump": dump, "use": use, "params": params, "time": timing}[a.cmd](a)""")
open(p, "w", encoding="utf-8").write(s)
print("patched eval")
