import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:80])
    s = s.replace(a, b)
rep('''def sample(n, split):
    """cache indices of the frames that are read out: all of VAL, NSAMPLE evenly spaced frames of TRAIN"""
    return np.arange(n) if split == "val" else np.unique(np.linspace(0, n - 1, NSAMPLE).round().astype(int))''',
'''def sample(n, split, every=1):
    """cache indices of the frames that are read out: all of VAL, NSAMPLE evenly spaced frames of TRAIN; every > 1 takes every `every`-th frame of that sample (the first, the (every+1)-th, ...)"""
    idx = np.arange(n) if split == "val" else np.unique(np.linspace(0, n - 1, NSAMPLE).round().astype(int))
    return idx[::every]


def every_sfx(a):
    """'' or _every<N> in the file names of a read-out that takes every N-th frame of the sample"""
    return f"_every{a.every}" if a.every > 1 else ""''')
rep('''    idx = sample(len(ds), a.split)[: a.max or None]
    path = os.path.join(runs, a.run, f"token_{a.split}{JV.suffix(2.5, a.drop, a.drop_seed)}{f'_n{len(idx)}' if a.max else ''}.pkl")''',
'''    idx = sample(len(ds), a.split, a.every)[: a.max or None]
    path = os.path.join(runs, a.run, f"token_{a.split}{every_sfx(a)}{JV.suffix(2.5, a.drop, a.drop_seed)}{f'_n{len(idx)}' if a.max else ''}.pkl")''')
rep('''idx=idx.tolist(), K=ds.K,''', '''idx=idx.tolist(), every=a.every, tokens=[ds.tokens[i] for i in idx], K=ds.K,''')
rep('''f"token_{a.split}{JV.suffix(2.5, a.drop, a.drop_seed)}{nsfx}.pkl"''', '''f"token_{a.split}{every_sfx(a)}{JV.suffix(2.5, a.drop, a.drop_seed)}{nsfx}.pkl"''')
rep('''    idx = sample(len(ds), a.split)[: n_max or None]
    assert meta["run"] == a.run and meta["idx"] == idx.tolist()''', '''    idx = sample(len(ds), a.split, a.every)[: n_max or None]
    assert meta["run"] == a.run and meta.get("every", 1) == a.every and meta["idx"] == idx.tolist()''')
rep('''    sfx = ("" if a.split == "val" else f"_{a.split}") + JV.suffix(2.5, a.drop, a.drop_seed)''', '''    sfx = ("" if a.split == "val" else f"_{a.split}") + every_sfx(a) + JV.suffix(2.5, a.drop, a.drop_seed)''', 2)
rep('''            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; `score --max N` reads it)")''', '''            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; `score --max N` reads it)")''')
# parser: --every on dump, score, sweep
rep('''        s.add_argument("--drop-seed", type=int, default=0)
''', '''        s.add_argument("--drop-seed", type=int, default=0)
        s.add_argument("--every", type=int, default=1, help="every N-th frame of the split's sample (TRAIN: 873 evenly spaced frames; every 3 = 291); the file names carry _every<N>; score and sweep read the dump with the same value")
''')
rep("[--drop 0.17 --drop-seed 0] [--cpu] [--max N]   #", "[--drop 0.17 --drop-seed 0] [--every N] [--cpu] [--max N]   #")
rep("[--drop 0.17 --drop-seed 0] [--max N]                     #", "[--drop 0.17 --drop-seed 0] [--every N] [--max N]                     #")
rep("sweep --run R --seed S --split train [--max N]", "sweep --run R --seed S --split train [--every N] [--max N]")
open(p, "w", encoding="utf-8").write(s)
