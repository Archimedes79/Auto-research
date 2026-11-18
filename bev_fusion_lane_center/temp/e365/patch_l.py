import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = s.index("TESTS = [")
new = '''def test_arch_options():
    import numpy as np
    # defaults = the head of the first build, bit for bit
    dflt = JKT.parse_args(["--name", "x"])
    torch.manual_seed(0)
    new = JKT.build_head(vars(dflt))
    torch.manual_seed(0)
    old = JK.JTokenHead()
    assert list(new.state_dict()) == list(old.state_dict()) and all(torch.equal(v, old.state_dict()[k]) for k, v in new.state_dict().items()), "default options must build the old head"
    assert JK.count_params(new) == 127131 and JKT.build_head({}).state_dict().keys() == old.state_dict().keys(), "an old run's args (no options) mean the defaults"
    # 20 synthetic steps with the defaults reproduce what the committed code did (recorded before the change: seed 0, bs 2, K 48)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:
        JKT.train(JKT.parse_args(["--name", "r", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]), runs, 500, 48)
        row = [json.loads(r) for r in open(os.path.join(runs, "r", "log.jsonl"))][-1]
        sd = torch.load(os.path.join(runs, "r", "model.pt"), map_location="cpu", weights_only=False)["state"]
        tot = sum(float(v.double().abs().sum()) for v in sd.values() if v.dtype.is_floating_point)
        assert abs(row["total"] - 2.8015905618667603) < 1e-9 and abs(row["nll_s"] - 1.288527250289917) < 1e-9 and abs(tot / 8250.530359720806 - 1) < 1e-12, (row["total"], tot)
        # the other sizes: parameter counts, one smoke run, the rebuild by load_model, a recipe that differs is refused, an old run folder resumes with the defaults
        counts = {}
        for name, kw in (("d96l4", ["--d", "96", "--layers", "4"]), ("d96l4f", ["--d", "96", "--layers", "4", "--ffn", "512"]), ("d32l2h2", ["--d", "32", "--layers", "2", "--heads", "2"])):
            JKT.train(JKT.parse_args(["--name", name, "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"] + kw), runs, 500, 48)
            m, a = JKT.load_model(name, "cpu", runs)
            counts[name] = JK.count_params(m)
            assert a["d"] == int(kw[1]) and m.proj.out_features == a["d"] and len(m.layers) == a["layers"] and m.layers[0].ffn[0].out_features == (a["ffn"] or 2 * a["d"]) and counts[name] != 127131
        assert counts["d96l4f"] > counts["d96l4"]
        must_fail(JKT.build_head, dict(d=70, heads=4))                                                     # d must be a multiple of the heads
        os.makedirs(os.path.join(runs, "old"))
        json.dump(dict(design="jtoken", args=dict(seed=0, cache=None, ident="e292_a", K=None, bs=2, steps=20, lr=1e-3)), open(os.path.join(runs, "old", "config.json"), "w"))     # a config of before the options
        a_old = vars(JKT.parse_args(["--name", "old", "--bs", "2", "--steps", "20"]))
        JKT.check_start(a_old, os.path.join(runs, "old"))
        must_fail(JKT.check_start, dict(a_old, d=96), os.path.join(runs, "old"))
        must_fail(JKT.check_start, dict(a_old, token_drop=0.1), os.path.join(runs, "old"))
    # token drop: the effect on a batch, the draw per step
    sb = JK.synthetic_batch(2, 48, seed=1)
    keep = torch.ones(2, 48, dtype=torch.bool)
    keep[0, 3] = keep[1, 12] = False
    out = JKT.drop_tokens(sb, keep)
    assert not out["valid"][0, 3] and not out["valid"][1, 12] and out["valid"].sum() == sb["valid"].sum() - 2 and out["ytype"][0, 3] == 0 and out["ytype"][1, 12] == 0
    assert out["y"][0, :, 3].sum() == 0 and out["y"][0, 3, :].sum() == 0 and out["y"][0, 4, 5] == sb["y"][0, 4, 5] and out["y"][1, 12, :].sum() == 0
    assert out["ylong"][1, 11, 12] == 0 and torch.isnan(out["yscale"][0, 3]).all() and torch.equal(out["gt_out"] if "gt_out" in out else sb["ytype"], sb["gt_out"] if "gt_out" in sb else sb["ytype"])
    assert sb["y"][0, 2, 3] == 1 and out["y"][0, 2, 3] == 0, "a GT link into a removed token is lost, the link through it is not bridged"
    assert torch.equal(JKT.drop_tokens(sb, torch.ones(2, 48, dtype=torch.bool))["y"], sb["y"])
    toks = ["a", "b"]
    k1, k1b, k2, k3 = JKT.step_keep(toks, 48, 0.3, 0, 5), JKT.step_keep(toks, 48, 0.3, 0, 5), JKT.step_keep(toks, 48, 0.3, 0, 6), JKT.step_keep(toks, 48, 0.3, 1, 5)
    big = JKT.step_keep(["a"] * 1, 100000, 0.3, 0, 5)
    assert (k1 == k1b).all() and not (k1 == k2).all() and not (k1 == k3).all() and abs(big.mean() - 0.7) < 0.01, "reproducible per (seed, step, token), new draw per step, the share"
    assert JKT.step_keep(["a"], 48, 0.0, 0, 5).all()
    # the same effect as the loader's own token_drop on a cache frame (when the cache exists)
    try:
        import lane_ptok_data as LP
        from jtok_eval import keep_nodes
        base, dropped = LP.PTokData("val", 0, "e292_a", None, 0.0), LP.PTokData("val", 0, "e292_a", None, 0.17)
        b = LP.collate([base[i] for i in range(2)])
        kp = torch.from_numpy(np.stack([keep_nodes(t, base.K, 0.17, 0) for t in b["token"]]))
        mine = JKT.drop_tokens(b, kp)
        ref = LP.collate([dropped[i] for i in range(2)])
        assert all(torch.equal(mine[k], ref[k]) or (k == "yscale" and torch.equal(mine[k].nan_to_num(-7), ref[k].nan_to_num(-7))) for k in ("valid", "ytype", "y", "ylong", "yscale")), "drop_tokens is the loader's token_drop on the batch"
        same = "equal to the loader's token_drop on 2 cache frames"
    except (AssertionError, FileNotFoundError, ImportError) as e:
        if "match" not in str(e) and "cache" not in str(e) and not isinstance(e, (FileNotFoundError, ImportError)):
            raise
        same = "loader comparison SKIPPED (no cache)"
    with tempfile.TemporaryDirectory(dir=SCRATCH) as runs:                                                    # a training run with token drop: finite, differs from the run without
        rows = {}
        for name, extra in (("p0", []), ("p3", ["--token-drop", "0.3"])):
            JKT.train(JKT.parse_args(["--name", name, "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"] + extra), runs, 500, 48)
            rows[name] = [json.loads(r) for r in open(os.path.join(runs, name, "log.jsonl"))][-1]["total"]
        assert math.isfinite(rows["p3"]) and rows["p3"] != rows["p0"], rows
    print(f"arch_options: defaults = the old head (127131 parameters, 20 smoke steps reproduce the recorded loss 2.80159 and the weights), d96 / layers 4: {counts['d96l4']} parameters (ffn 512: {counts['d96l4f']}), d32 / l2 / h2: {counts['d32l2h2']}, "
          f"rebuilt by load_model, an old config resumes with the defaults and another recipe is refused; token drop: targets of removed tokens go, no re-contraction, a new draw per step, {same}: ok")


'''
s = s[:a] + new + s[a:]
s = s.replace("test_overfit, test_trainer]", "test_overfit, test_trainer, test_arch_options]")
open(p, "w", encoding="utf-8").write(s)
