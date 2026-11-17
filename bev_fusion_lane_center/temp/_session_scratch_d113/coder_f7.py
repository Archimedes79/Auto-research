p = "training/grid/test_jpast_train.py"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep('''    m2 = dict(meta=dict(run="r", split="val", idx=[10, 11, 12, 13], use=dict(d2=[0.0, 1.0, 4.0, 1.0], x2=[4.0] * 4, pos=[0, 1, 5, 1], has_prev=[False, True, True, True], state_ss=[8.0, 18.0, 50.0, 18.0], state_n=[2, 2, 2, 2])), frames=[])''',
    '''    m2 = dict(meta=dict(run="r", split="val", idx=[10, 11, 12, 13, 14], use=dict(d2=[0.0, 1.0, 4.0, 1.0, 1.0], x2=[4.0] * 5, pos=[0, 1, 5, 4, 1], has_prev=[False, True, True, True, True], state_ss=[8.0, 18.0, 50.0, 18.0, 36.0], state_n=[2] * 5)), frames=[])''')
rep('''    assert len(sr) == 6 and sr[0] == 2.0 and sr[1] == 3.0 and sr[5] == 5.0 and all(np.isnan(sr[q]) for q in (2, 3, 4)), sr
    assert r2["position_per_frame"] == [0, 1, 5, 1] and r2["frame_idx_first5"] == [10, 11, 13] and r2["frame_idx_later"] == [12], r2''',
    '''    assert len(sr) == 6 and sr[0] == 2.0 and abs(sr[1] - math.sqrt(54 / 4)) < 1e-12 and sr[4] == 3.0 and sr[5] == 5.0 and all(np.isnan(sr[q]) for q in (2, 3)), sr
    assert r2["position_per_frame"] == [0, 1, 5, 4, 1] and r2["frame_idx_first5"] == [10, 11, 13, 14] and r2["frame_idx_later"] == [12], r2''')
rep("import inspect\nimport os", "import inspect\nimport math\nimport os")
rep('''        res = E.use(argparse.Namespace(run="x", split="val", max=12), runs=tmp)                              # the command: stream + the sibling x_off -> past_use_val_n12.json''',
    '''        res = E.use(argparse.Namespace(run="x", split="val", max=12), runs=tmp)                              # the command: stream + the sibling x_off -> past_use_val_n12.json
        assert res["against_off"] == rep["against_off"] and res["against_off"]["kept_flips_per_frame"] > 0, "use reads the off dump of the sibling folder"''')
old_start = s.index("def real_chain_check():")
old_end = s.index("# ------------------------------------------------------------------------------------------------------------------------------ mutants")
new = '''def real_chain_check(train_path=TRAIN):
    """one batch of the real TRAIN chains (the cache and its extension, 2 sequences of 3 keyframes) through the loop with the real head on the CPU, the sequence order of two arms with the same seeds, and `train()` with the real loader (3 steps, the stub head):
    the arguments it hands to `lane_past_data.loader` (the smoke path never calls it). Not part of the selftest: it needs the cache"""
    import json
    import torch
    import trainer as TR
    import lane_past_data as LPD
    T = load(train_path, "jpast_train_real")
    ds, dl = LPD.loader("train", 0, 2, 0)
    TR.seed_all(0)
    b = next(iter(dl))
    rows = torch.stack([f["row"] for f in b["frames"]], 1)
    assert tuple(rows.shape) == (2, 3) and (rows[:, 1:] - rows[:, :-1] == 1).all() and b["has_prev"][:, 1:].all() and tuple(b["rel_t"].shape) == (2, 3, 2) and b["frames"][0]["tok"].shape[1] == ds.tok.K, "real chains"
    torch.manual_seed(0)
    m = T.build_head(dict(no_past=False))
    for p in (m.past_block.o.weight, m.past_block.o.bias):
        torch.nn.init.normal_(p, std=0.3)
    m.train()
    m.set_step(0)
    loss, infos = T.sequence_loss(m, b, "cpu")
    loss.backward()
    assert torch.isfinite(loss) and len(infos) == 3 and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in m.past_block.parameters()), "the real chains through the loop"

    def first(seed):
        TR.seed_all(seed)
        return [bb["idx"].tolist() for _, bb in zip(range(3), LPD.loader("train", 0, 4, 0)[1])]
    assert first(0) == first(0) and first(0) != first(1), "the sequence order follows the seed"
    calls, orig = [], LPD.loader

    def recording(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)
    LPD.loader = recording
    try:
        with Patch(make_stub()), tempfile.TemporaryDirectory() as tmp:
            T.train(T.parse_args(["--name", "rl", "--seed", "1", "--init-seed", "0", "--bs", "2", "--seq", "2", "--steps", "3", "--workers", "0"]), runs=tmp, ckpt_every=500)
            ms, args = T.load_model("rl", "cpu", tmp)
            assert ms._step_py == 3 and args["seed"] == 1 and json.load(open(os.path.join(tmp, "rl", "config.json")))["seed"] == 0 and os.path.exists(os.path.join(tmp, "rl", ".done"))
    finally:
        LPD.loader = orig
    assert len(calls) == 1 and calls[0][0] == ("train", 1, 2, 0) and calls[0][1] == dict(K_seq=2, ident="e292_a", K=None), f"the loader call of train(): {calls}"
    print(f"real chains ok: loss {float(loss):.3f}, {len(ds)} chains, first batch rows {rows.tolist()}")


'''
s = s[:old_start] + new + s[old_end:]
rep('''def one(args):
    k, (mod, name, a, b), tmp = args''', '''def one(args):
    k, (mod, name, a, b, *rest), tmp = args
    cmd = rest[0] if rest else "one"                                                                        # the command a mutant runs: the selftest ("one") or the real-data check ("real")''')
rep('''    r = subprocess.run([sys.executable, os.path.abspath(__file__), "one", other[0], other[1]], capture_output=True, text=True, env=ENV)
    return name,''', '''    r = subprocess.run([sys.executable, os.path.abspath(__file__), cmd, other[0], other[1]], capture_output=True, text=True, env=ENV)
    return name,''')
rep('''    if len(sys.argv) > 1 and sys.argv[1] == "real":
        real_chain_check()
        sys.exit(0)''', '''    if len(sys.argv) > 1 and sys.argv[1] == "real":
        real_chain_check(sys.argv[2] if len(sys.argv) > 2 else TRAIN)
        sys.exit(0)''')
SEED_ASSERT_OLD = 'assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args[' + "'seed'" + ']})"'
rep('''    ("jpast_eval.py", "the stream carries the state over a scene start",''', '''    (T0, "the switch is judged one step late", "learned = s0 >= JK.SWITCH", "learned = s0 > JK.SWITCH"),
    (T0, "U_WINDOW is 10", "U_WINDOW = 100 ", "U_WINDOW = 10 "),
    (T0, "a change of seq is not refused at a resume", 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_past")'),
    (T0, "the loader gets K_seq 3", "K_seq=a.seq, ident=a.ident", "K_seq=3, ident=a.ident", "real"),
    (T0, "the loader gets the head seed", 'LPD.loader("train", a.seed, a.bs', 'LPD.loader("train", init, a.bs', "real"),
    (T0, "the loader gets another batch size", "a.seed, a.bs, a.workers", "a.seed, 4, a.workers", "real"),
    ("jpast_eval.py", "the read-out's switch is one step late", "learned = m._step_py >= JK.SWITCH", "learned = m._step_py > JK.SWITCH"),
    ("jpast_eval.py", "the last scene ends one frame early", "ends = starts[1:] + [len(prev)]", "ends = starts[1:] + [len(prev) - 1]"),
    ("jpast_eval.py", "dump does not check the seed", SEED_ASSERT_OLD, "pass"),
    ("jpast_eval.py", "the sibling folder has no model.pt", "shutil.copyfile(src, dst)", "pass"),
    ("jpast_eval.py", "a sibling with another model.pt is accepted", 'assert filecmp.cmp(src, dst, shallow=False), f"{dst} is not a copy of {src}"', "pass"),
    ("jpast_eval.py", "the meta names the run, not the sibling", "meta = dict(run=name,", "meta = dict(run=a.run,"),
    ("jpast_eval.py", "every mode goes to a sibling", 'return run if mode == "stream" else f"{run}_{mode}"', 'return f"{run}_{mode}"'),
    ("jpast_eval.py", "use reads the off dump of the run itself", "dump_file(runs, run_folder(a.run, m), a.split, a.max)", "dump_file(runs, a.run, a.split, a.max)"),
    ("jpast_eval.py", "the state count forgets the channels", 'state_n=int(v.sum()) * int(out["x"].shape[-1])', "state_n=int(v.sum())"),
    ("jpast_eval.py", "the carried state is the block input", 'state_ss=float((out["x"][0].float() ** 2)[v].sum())', "state_ss=x2"),
    ("jpast_eval.py", "the first 5 frames are 4", "FIRST = 5 ", "FIRST = 4 "),
    ("jpast_eval.py", "the state RMS stops one position early", "for p in range(int(pos.max()) + 1)]", "for p in range(int(pos.max()))]"),
    ("jpast_eval.py", "the stream carries the state over a scene start",''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
