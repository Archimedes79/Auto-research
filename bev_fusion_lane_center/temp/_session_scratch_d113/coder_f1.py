p = "training/data/lane_past_data.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
rep('''    s = sub.add_parser("fixture")
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--n-pairs", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ident", default="e292_a")
''', '''    for name, n_default in (("fixture", 200), ("loader-fixture", 40)):
        s = sub.add_parser(name)
        s.add_argument("--split", default="val", choices=("val", "train"))
        s.add_argument("--n-pairs", type=int, default=n_default)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--ident", default="e292_a")
        s.add_argument("--out")
''')
rep('''    elif a.cmd == "fixture":
        assert fixture(a.split, a.n_pairs, a.seed, a.ident)["passed"], "the fixture FAILED"
''', '''    elif a.cmd in ("fixture", "loader-fixture"):
        res = (fixture if a.cmd == "fixture" else loader_fixture)(a.split, a.n_pairs, a.seed, a.ident)
        if a.out:
            _save_json(a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out), res)
        assert res["passed"], f"the {a.cmd} FAILED"
''')
rep('''    wsl bash tools/env/_py14.sh lane_past_data.py fixture --split val [--n-pairs 200] [--seed 0] [--ident e292_a]   # the pose against the GT lane points and the B' tokens: PASS / FAIL + numbers; the wrong conventions must FAIL
''', '''    wsl bash tools/env/_py14.sh lane_past_data.py fixture --split val [--n-pairs 200] [--seed 0] [--ident e292_a] [--out FILE]   # the pose against the GT lane points and the B' tokens: PASS / FAIL + numbers; the wrong conventions must FAIL
    wsl bash tools/env/_py14.sh lane_past_data.py loader-fixture --split val [--n-pairs 40] [--seed 0] [--out FILE]               # the TRAINING path: the loader's items through `jpast.make_past` against the delivered B' tokens; wrong pose / shifted entry must FAIL
''')
rep('''the warped delivered B' tokens (valid, p >= 0.5) of t-1 have a delivered token of t within p90 <= 1.5 m; the identity pose, the inverse pose, the transposed rotation and a previous frame shifted by one must FAIL the point test.''', '''the warped delivered B' tokens (valid, p >= 0.5) of t-1 have a delivered token of t within p90 <= 1.5 m; the identity pose, the inverse pose, the transposed rotation and a previous frame shifted by one must FAIL the point test.
The loader fixture repeats the token test through the TRAINING path (`PastData` items, `collate`, `jpast.make_past`); the identity pose, the pose entry of the previous item and a shifted previous frame must FAIL it.''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
