p = "training/train/jtoken_fast_train.py"
t = open(p, encoding="utf-8").read()
reps = [
('''def resolve_cache(cache):
    """--cache given relative to the problem folder -> its absolute path (the run's args carry it, so every later read-out opens the same root from any working directory: the scripts run in their own folder)"""
    return cache if not cache or os.path.isabs(cache) else os.path.join(ROOT, cache)


''', ''),
('''    a.cache = resolve_cache(a.cache)
''', ''),
('''`--cache <root>` is taken relative to the problem folder and stored absolute in the run's args (a second K lives in its own root: data/build_tokcache_root.py).
''','''`--cache <root>` may be relative to the problem folder (`lane_ptok_data.cache_dir` resolves it; a second K lives in its own root: data/build_tokcache_root.py).
'''),
('''    folder = os.path.join(T.RUNS, a.name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "fast_path.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(head="jtoken_fast.FastJTokenHead", loss="jtoken_fast.fast_loss", loader="lane_ptok_data (CPU)" if (own["--cpu-loader"] or a.smoke) else "lane_ptok_gpu (GPU-resident)", compile_pairs=own["--compile-pairs"]), fh)
    JT.train(a, T.RUNS)''','''    note = dict(head="jtoken_fast.FastJTokenHead", loss="jtoken_fast.fast_loss", loader="lane_ptok_data (CPU)" if (own["--cpu-loader"] or a.smoke) else "lane_ptok_gpu (GPU-resident)", compile_pairs=own["--compile-pairs"])
    check = JT.check_start

    def check_then_note(args, folder):                                      # the run folder is written only after the recipe check accepted it
        check(args, folder)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "fast_path.json"), "w", encoding="utf-8") as fh:
            json.dump(note, fh)

    JT.check_start = check_then_note
    try:
        JT.train(a, T.RUNS)
    finally:
        JT.check_start = check'''),
('''same `model.pt` / `ckpt.pt` (a run of this script resumes a plain run's checkpoint and the other way round)''','''same `model.pt` / `ckpt.pt` and recipe (a run of this script resumes a plain run's checkpoint with the same arguments, and the other way round)'''),
]
for a, b in reps:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
open(p, "w", encoding="utf-8").write(t)
p = "training/grid/test_jtoken_fast.py"
t = open(p, encoding="utf-8").read()
a = '''    assert FT.resolve_cache(None) is None and FT.resolve_cache("/abs/x") == "/abs/x" and FT.resolve_cache("runs/c") == os.path.join(FT.ROOT, "runs/c")
'''
assert t.count(a) == 1
t = t.replace(a, '')
open(p, "w", encoding="utf-8").write(t)
print("ok")
