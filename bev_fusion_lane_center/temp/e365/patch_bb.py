p = "training/grid/test_jtoken.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)

rep('''    # 20 synthetic steps with the defaults reproduce what the committed code did (recorded from the committed code before the changes, 2 threads: seed 0, bs 2, K 48)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        JKT.train(JKT.parse_args(["--name", "r", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]), runs, 500, 48)
        row = [json.loads(r) for r in open(os.path.join(runs, "r", "log.jsonl"))][-1]''', '''    # 20 synthetic steps with the defaults reproduce what the committed code did (recorded from the committed code before the changes, 2 threads: seed 0, bs 2, K 48). THESE CONSTANTS HOLD IN THIS TEST'S PROCESS ONLY (this file
    # sets 2 threads; a fresh process with another thread count / BLAS order gives 2.8011269 for the same run): the portable check is the same-process one below (defaults against the explicit default arguments)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        JKT.train(JKT.parse_args(["--name", "r", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]), runs, 500, 48)
        row = [json.loads(r) for r in open(os.path.join(runs, "r", "log.jsonl"))][-1]''')
rep('''        # the other sizes: parameter counts, one smoke run, the rebuild by load_model, a recipe that differs is refused, an old run folder resumes with the defaults
        counts = {}''', '''        JKT.train(JKT.parse_args(["--name", "r2", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--d", "64", "--layers", "3", "--heads", "4", "--ffn", "128", "--token-drop", "0.0", "--none-weight", "1.0"]), runs, 500, 48)
        rows2 = [json.loads(r) for r in open(os.path.join(runs, "r2", "log.jsonl"))]
        sd2 = torch.load(os.path.join(runs, "r2", "model.pt"), map_location="cpu", weights_only=False)["state"]
        assert [r["total"] for r in rows2] == [json.loads(r)["total"] for r in open(os.path.join(runs, "r", "log.jsonl"))] and list(sd2) == list(sd) and all(torch.equal(v, sd[k]) for k, v in sd2.items()), "explicit default arguments = the defaults, same process, bit for bit"
        # the other sizes: parameter counts, one smoke run, the rebuild by load_model, a recipe that differs is refused, an old run folder resumes with the defaults
        counts = {}''')
rep('''        must_fail(JKT.build_head, dict(d=70, heads=4))                                                     # d must be a multiple of the heads''', '''        must_fail(JKT.build_head, dict(d=70, heads=4))                                                     # d must be a multiple of the heads
        must_fail(JKT.build_head, dict(ffn=0))                                                             # 0 is not "the default"
        for bad in (["--d", "70"], ["--d", "96", "--heads", "5"], ["--ffn", "0"], ["--ffn", "-4"], ["--layers", "0"], ["--token-drop", "-0.1"], ["--token-drop", "1.0"]):
            must_fail(JKT.parse_args, ["--name", "x"] + bad)                                               # refused before anything is built or loaded
        assert JKT.parse_args(["--name", "x", "--token-drop", "0.5", "--ffn", "96", "--d", "96", "--heads", "6"]).token_drop == 0.5''')
rep('''        must_fail(JKT.check_start, dict(a_old, token_drop=0.1), os.path.join(runs, "old"))''', '''        must_fail(JKT.check_start, dict(a_old, token_drop=0.1), os.path.join(runs, "old"))
        os.makedirs(os.path.join(runs, "big"))                                                             # the reverse: a run of d 96 / layers 4 is not resumed by the default command line
        json.dump(dict(design="jtoken", args=dict(seed=0, cache=None, ident="e292_a", K=None, bs=2, steps=20, lr=1e-3, d=96, layers=4)), open(os.path.join(runs, "big", "config.json"), "w"))
        a_big = vars(JKT.parse_args(["--name", "big", "--bs", "2", "--steps", "20"]))
        must_fail(JKT.check_start, a_big, os.path.join(runs, "big"))
        JKT.check_start(dict(a_big, d=96, layers=4), os.path.join(runs, "big"))
        must_fail(JKT.check_start, dict(a_big, d=96, layers=4, ffn=192), os.path.join(runs, "big"))         # ffn None and the explicit 2 x d are the same head: refused all the same (conservative)''')
rep('''    try:
        import lane_ptok_data as LP
        from jtok_eval import keep_nodes
        base, dropped = LP.PTokData("val", 0, "e292_a", None, 0.0), LP.PTokData("val", 0, "e292_a", None, 0.17)
        b = LP.collate''', '''    import lane_ptok_data as LP                                                                              # an ImportError is a failure, not a skip
    from jtok_eval import keep_nodes
    try:
        base, dropped = LP.PTokData("val", 0, "e292_a", None, 0.0), LP.PTokData("val", 0, "e292_a", None, 0.17)
    except (AssertionError, FileNotFoundError) as e:                                                         # only the cache lookup may skip the comparison
        if isinstance(e, AssertionError) and "finished caches match" not in str(e):
            raise
        base = None
    if base is None:
        same = "loader comparison SKIPPED (no cache)"
    else:
        b = LP.collate''')
rep('''        same = "equal to the loader's token_drop on 2 cache frames"
    except (AssertionError, FileNotFoundError, ImportError) as e:
        if "match" not in str(e) and "cache" not in str(e) and not isinstance(e, (FileNotFoundError, ImportError)):
            raise
        same = "loader comparison SKIPPED (no cache)"
''', '''        same = "equal to the loader's token_drop on 2 cache frames"
''')
# frame mean clamp
rep('''    orig_fm = JK._frame_mean''', '''    x, mk, wk = torch.full((1, 2), 0.1), torch.ones(1, 2, dtype=torch.bool), torch.full((1, 2), 0.1)       # weights summing below 1: sum(w x) / sum(w) = 0.02 / 0.2, not / 1
    assert abs(float(JK._frame_mean(x, mk, wk)) - 0.5) < 1e-6 and abs(float(JK._frame_mean(x, mk)) - 0.1) < 1e-6 and float(JK._frame_mean(x, torch.zeros_like(mk), wk)) == 0.0
    orig_fm = JK._frame_mean''')
open(p, "w", encoding="utf-8").write(s)
