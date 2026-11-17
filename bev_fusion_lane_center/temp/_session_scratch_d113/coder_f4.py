p = "evaluation/jpast_eval.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
rep('''dict(idx=[], has_prev=[], pos=[], shift=[], d2=[], x2=[], n=[])''', '''dict(idx=[], has_prev=[], pos=[], shift=[], d2=[], x2=[], n=[], state_ss=[], state_n=[])''')
rep('''            for k in ("has_prev", "pos", "shift", "d2", "x2", "n"):''', '''            for k in ("has_prev", "pos", "shift", "d2", "x2", "n", "state_ss", "state_n"):''')
rep('''    path = os.path.join(runs, a.run, f"token_{a.split}{mode_sfx(a.mode)}{f'_n{len(idx)}' if a.max else ''}.pkl")
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
''', '''    name = run_folder(a.run, a.mode)
    path = dump_file(runs, name, a.split, len(idx) if a.max else 0)
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
''')
rep('''    meta = dict(run=a.run, seed=a.seed,''', '''    sibling(runs, a.run, a.mode)
    meta = dict(run=name, seed=a.seed,''')
rep('''K=ds.tok.K, ident=kw["ident"], cache=ds.dir, steps=args.get("steps"), mode=a.mode,''', '''K=ds.tok.K, ident=kw["ident"], cache=ds.dir, steps=args.get("steps"), model="model.pt", of_run=a.run, mode=a.mode,''')
rep('''    n = np.asarray(use["n"], float)
    u = math.sqrt(sum(use["d2"]) / max(sum(use["x2"]), 1e-30)) if n.sum() else float("nan")
    print(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames, mode {a.mode}, kept nodes per frame {np.mean([f['keep'].sum() for f in frames]):.1f}, u {u:.4f} over {int(sum(use['has_prev']))} frames with a past, device {dev}")
''', '''    used = sum(1 for n in use["n"] if n)
    u_txt = f"u {math.sqrt(sum(use['d2']) / max(sum(use['x2']), 1e-30)):.4f} over {used} frames with a past" if used else "no past read (mode off or arm z)"
    print(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames, mode {a.mode}, kept nodes per frame {np.mean([f['keep'].sum() for f in frames]):.1f}, {u_txt}, device {dev}")
''')
rep('''    u = stream["meta"]["use"]
    d2, x2, pos, hp = (np.asarray(u[k], float) for k in ("d2", "x2", "pos", "has_prev"))''', '''    u = stream["meta"]["use"]
    d2, x2, pos, hp, sq, sn = (np.asarray(u[k], float) for k in ("d2", "x2", "pos", "has_prev", "state_ss", "state_n"))''')
rep('''               u_by_position={f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": ratio((pos >= lo) & (pos <= hi) & (hp > 0)) for lo, hi in BUCKETS})''', '''               u_by_position={f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": ratio((pos >= lo) & (pos <= hi) & (hp > 0)) for lo, hi in BUCKETS},
               state_rms_by_position=[float(math.sqrt(sq[pos == p].sum() / sn[pos == p].sum())) if sn[pos == p].sum() > 0 else float("nan") for p in range(int(pos.max()) + 1)] if len(pos) else [],
               position_per_frame=pos.astype(int).tolist(), frame_idx_first5=[int(i) for i, p in zip(stream["meta"]["idx"], pos) if p < FIRST], frame_idx_later=[int(i) for i, p in zip(stream["meta"]["idx"], pos) if p >= FIRST])''')
rep('''BUCKETS = (''', '''FIRST = 5                                                                   # the card's split of the topo F1: the first 5 frames of every scene (positions 0 - 4) against the later ones
BUCKETS = (''')
rep('''def use(a, runs=JE.RUNS):
    load = lambda sfx: pickle.load(open(os.path.join(runs, a.run, f"token_{a.split}{sfx}.pkl"), "rb"))  # noqa: E731
    off_path = os.path.join(runs, a.run, f"token_{a.split}{mode_sfx('off')}.pkl")
    rep = use_report(load(""), load(mode_sfx("off")) if os.path.exists(off_path) else None)
    path = os.path.join(runs, a.run, f"past_use_{a.split}.json")''', '''def use(a, runs=JE.RUNS):
    paths = {m: dump_file(runs, run_folder(a.run, m), a.split, a.max) for m in ("stream", "off")}
    rep = use_report(pickle.load(open(paths["stream"], "rb")), pickle.load(open(paths["off"], "rb")) if os.path.exists(paths["off"]) else None)
    path = os.path.join(runs, a.run, f"past_use_{a.split}{f'_n{a.max}' if a.max else ''}.json")''')
rep('''    s = sub.add_parser("use")
    s.add_argument("--run", required=True)
    s.add_argument("--split", default="val", choices=("val", "train"))
''', '''    s = sub.add_parser("use")
    s.add_argument("--run", required=True)
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--max", type=int, default=0, help="the dumps of `dump --max N` (their names carry _n<N>)")
''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
