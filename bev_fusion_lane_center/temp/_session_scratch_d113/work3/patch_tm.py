import os
R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "training/grid/test_jmap_train.py")
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---------------------------------------------------------------- trainer tests
sub("""lr=lambda v: v * 2, no_map=lambda v: not v)""", """lr=lambda v: v * 2, no_map=lambda v: not v, map_mode=lambda v: "other" if v == "train" else "train")""")
sub('''    # the options: arm m needs two frames per step (v compares with another sample of the batch), arm z does not
    refuses(lambda: T.parse_args(["--name", "x", "--bs", "1"]), "at least 2 frames")
    T.parse_args(["--name", "x", "--bs", "1", "--no-map"])
''', '''    # the options: arm m needs two frames per step (v compares with another sample of the batch), arm z does not; the capacity control (--map-mode other) needs the block
    refuses(lambda: T.parse_args(["--name", "x", "--bs", "1"]), "at least 2 frames")
    T.parse_args(["--name", "x", "--bs", "1", "--no-map"])
    refuses(lambda: T.parse_args(["--name", "x", "--no-map", "--map-mode", "other"]), "needs the block")
    assert T.parse_args(["--name", "x", "--map-mode", "other"]).map_mode == "other" and T.parse_args(["--name", "x"]).map_mode == "train"
    # the defaults of the loop that a run uses (the tests pass their own): a checkpoint every 500 steps, the smoke frames of the real K
    sg = inspect.signature(T.train).parameters
    assert sg["ckpt_every"].default == 500 and sg["smoke_k"].default == T.SMOKE_K == 512, f"{label}: the defaults of train(): {sg['ckpt_every'].default} {sg['smoke_k'].default}"
''')
sub('''    # the smoke run to its end: files, the counter of the head = the steps, then the same run after a kill = the straight run''', '''    # the capacity control: the smoke batch of mode other gives every frame the map of ANOTHER frame of the batch (rolled by one): the maps are no longer near the tokens of their frame; everything else is the same batch
    bo = T.synthetic_batch(B, K, 1, False, True)
    assert all(torch.equal(bo["maps"][k], bm["maps"][k].roll(1, 0)) for k in LMD.KEYS) and all(same_t(bo[k], bm[k]) for k in bm if k != "maps"), f"{label}: the smoke maps of mode other"
    d_o = torch.cdist(bo["maps"]["pos"], bo["pos"]).min(-1).values[bo["maps"]["valid"]]
    assert (d_o < 2.5).float().mean() < 0.5 and (d_.min(-1).values[bm["maps"]["valid"]] < 2.5).all(), f"{label}: in mode other the maps are those of other frames"
    # the maps are moved to the device of the head (a head whose first parameter lives on the `meta` device stands in for a GPU: the head sees where its maps went)
    class Stop(Exception):
        pass

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dev = torch.nn.Parameter(torch.empty(1, device="meta"))
            self.map_block = JM.MapBlock()
            self.seen = None

        def forward(self, batch, past=None, maps=None):
            self.seen = {k: v.device.type for k, v in maps.items()}
            raise Stop
    pr = Probe()
    try:
        T.make_loss(NoRun(), pr, False)(pr, bm)
        raise AssertionError("the probe head was not called")
    except Stop:
        pass
    assert pr.seen == dict(pos="meta", dir="meta", count="meta", valid="meta"), f"{label}: the maps are not moved to the device of the head: {pr.seen}"
    # the smoke run to its end: files, the counter of the head = the steps, then the same run after a kill = the straight run''')
sub('''        mz, _ = T.load_model("z", "cpu", tmp)
        assert mz.map_block is None and mz._step_py == 20''', '''        oa = T.parse_args(["--name", "o", "--smoke", "--bs", "2", "--seed", "0", "--map-mode", "other"])
        T.train(oa, runs=tmp, ckpt_every=10, smoke_k=K)
        mo, ao = T.load_model("o", "cpu", tmp)
        assert ao["map_mode"] == "other" and mo._step_py == 20 and isinstance(mo.map_block, JM.MapBlock), f"{label}: the capacity control run"
        refuses(lambda: T.train(a.__class__(**dict(vars(oa), name="m")), runs=tmp, ckpt_every=10, smoke_k=K), "of another run")                 # the folder of arm m is not the capacity control's
        mz, _ = T.load_model("z", "cpu", tmp)
        assert mz.map_block is None and mz._step_py == 20''')

# ---------------------------------------------------------------- the read-out on real frames
sub('''        ns = lambda mode, run="x", seed=0, mx=12: argparse.Namespace(run=run, seed=seed, split="val", mode=mode, cpu=True, max=mx)  # noqa: E731''', '''        ns = lambda mode, run="x", seed=0, mx=12, split="val": argparse.Namespace(run=run, seed=seed, split=split, mode=mode, cpu=True, max=mx)  # noqa: E731''')
sub('''        # arm z: no block, no map built, every mode the same plain head on the frames, v zero''', '''        # the TRAIN sample (873 evenly spaced frames of the 8414, not the first ones): the same dump on 6 of them, the error map of the arm's loader, the scorer's reader accepts it, the use report
        from jtoken_eval import sample as jsample
        ds_tr = LP.PTokData("train", 0, "e292_a")
        want_idx = jsample(len(ds_tr), "train", 1)[:6].tolist()
        assert want_idx != list(range(6)) and want_idx[1] > 5, want_idx
        ptr = E.dump(ns("err", split="train", mx=6), runs=tmp)
        assert os.path.relpath(ptr, tmp) == "x/token_train_n6.pkl"
        dtr = pickle.load(open(ptr, "rb"))
        assert dtr["meta"]["idx"] == want_idx and dtr["meta"]["split"] == "train" and dtr["meta"]["tokens"] == [ds_tr.tokens[i] for i in want_idx] and [f["token"] for f in dtr["frames"]] == dtr["meta"]["tokens"], "the TRAIN dump is not the TRAIN sample"
        mt = dtr["meta"]                                                                                  # what the scorer's reader checks (`jtoken_eval.read_dump`; it also loads the TRAIN GT of the whole split, not done here)
        assert mt["run"] == "x" and mt["model"] == "model.pt" and mt["every"] == 1 and mt["drop"] == 0.0 and mt["drop_seed"] == 0 and mt["mode"] == "err" and len(dtr["frames"]) == 6
        dtr_ds = LMD.MapData("train", 0, "e292_a", None, "error", with_donor=True)
        for jj in (1, 4):
            b = LMD.collate([dict(ds_tr[want_idx[jj]], maps=dtr_ds.table(want_idx[jj]))])
            with torch.no_grad():
                o = m(JKT.to_device(b, "cpu"), maps=b["maps"])
            assert np.array_equal(dtr["frames"][jj]["pnone"], E.JE.batch_dump(o, b)[0]["pnone"]), f"TRAIN frame {want_idx[jj]} is not the head on its error map"
        rep_tr = E.use_report(dtr)
        assert rep_tr["frames_with_donor"] == 6 and rep_tr["v"] > 0 and rep_tr["split"] == "train"
        # params and time (the strict loader of an arm-m run, `jtoken_eval.timing`'s protocol): the block's 20 264 parameters, the head without it, GMACs with and without the block, ms per frame on the CPU, a new file every time
        import jtoken as JK
        pm = E.params(argparse.Namespace(run="x"), runs=tmp)
        assert pm["params_block"] == 20264 == JK.count_params(m.map_block) and pm["params"] == JK.count_params(m) and pm["params_head"] == pm["params"] - 20264 == JK.count_params(JKT.build_head(args)) and not pm["no_map"], pm
        tm_ = E.timing(argparse.Namespace(run="x", cpu=True, frames=3, repeats=1, K=64), runs=tmp)
        assert tm_["K"] == 64 and tm_["frames"] == 3 and tm_["params"] == pm["params"] and tm_["params_block"] == 20264 and tm_["gmacs_head"] > tm_["gmacs_no_map"] > 0 and tm_["median_ms"] > 0 and tm_["p90_ms"] >= tm_["median_ms"] and tm_["map_samples_mean"] > 50 and not tm_["usable_for_gcost"] and tm_["device"] == "cpu", tm_
        assert [f for f in os.listdir(os.path.join(tmp, "x")) if f.startswith("map_timing_cpu_")], "timing wrote no file"
        refuses(lambda: E.timing(argparse.Namespace(run="x", cpu=True, frames=2, repeats=1, K=100000), runs=tmp), "is above the K of the cache")
        refuses(lambda: E.load_model("nonexistent", "cpu", tmp), "")
        try:                                                                                              # `jtoken_train.load_model` builds the head without the block and cannot read an arm-m checkpoint: the reason for `load_model` here
            JKT.load_model("x", "cpu", tmp)
            raise AssertionError("the plain loader read an arm-m checkpoint")
        except RuntimeError:
            pass
        assert isinstance(E.load_model("x", "cpu", tmp)[0].map_block, type(m.map_block)) and E.load_model("x", "cpu", tmp)[1]["seed"] == 0
        # arm z: no block, no map built, every mode the same plain head on the frames, v zero''')
sub('''        dz = {md: pickle.load(open(E.dump(ns(md, "z", mx=6), runs=tmp), "rb")) for md in ("err", "none", "S")}''', '''        dz = {md: pickle.load(open(E.dump(ns(md, "z", mx=6), runs=tmp), "rb")) for md in ("err", "none", "S")}
        pz = E.params(argparse.Namespace(run="z"), runs=tmp)
        tz = E.timing(argparse.Namespace(run="z", cpu=True, frames=2, repeats=1, K=64), runs=tmp)
        assert pz["params_block"] == 0 and pz["params"] == pz["params_head"] and pz["no_map"] and tz["params_block"] == 0 and tz["gmacs_head"] == tz["gmacs_no_map"] > 0 and tz["map_samples_mean"] == 0.0, (pz, tz)''')

# ---------------------------------------------------------------- the real loader check
sub('''    assert ds.mode == "error" and LMD.loader("train", 0, 2, 0)[0].mode == "train"''', '''    do_, dlo = LMD.loader("train", 0, 4, 0, init_seed=5, mode="other")
    bo_ = next(iter(dlo))
    assert do_.mode == "other" and int(do_.draws.sum()) == 4 and sorted(bo_["maps"]) == sorted(LMD.KEYS) and bo_["maps"]["valid"].shape[0] == 4, "the loader of the capacity control"
    assert ds.mode == "error" and LMD.loader("train", 0, 2, 0)[0].mode == "train"''')
sub('''            T.train(T.parse_args(["--name", "rz", "--seed", "1", "--bs", "3", "--steps", "2", "--workers", "0", "--no-map"]), runs=tmp, ckpt_every=500)''', '''            T.train(T.parse_args(["--name", "rz", "--seed", "1", "--bs", "3", "--steps", "2", "--workers", "0", "--no-map"]), runs=tmp, ckpt_every=500)
            T.train(T.parse_args(["--name", "ro", "--seed", "1", "--bs", "2", "--steps", "2", "--workers", "0", "--map-mode", "other"]), runs=tmp, ckpt_every=500)''')
old_assert = s[s.index('    assert len(calls) == 2 and calls[0][0] == ("train", 1, 2, 0)'): s.index('    print(f"real loader ok')]
new_assert = '''    want_calls = [(("train", 1, 2, 0), dict(init_seed=0, maps=True, mode="train", ident="e292_a", K=None)), (("train", 1, 3, 0), dict(init_seed=1, maps=False, mode="train", ident="e292_a", K=None)), (("train", 1, 2, 0), dict(init_seed=1, maps=True, mode="other", ident="e292_a", K=None))]
    assert calls == want_calls, f"the loader calls of train(): {calls}"
'''
s = s.replace(old_assert, new_assert)

# ---------------------------------------------------------------- mutants
sub('''    (T0, "the loader builds the maps of arm z", "maps=not a.no_map, ident=a.ident", "maps=True, ident=a.ident", "real"),''', '''    (T0, "the loader builds the maps of arm z", "maps=not a.no_map, mode=a.map_mode, ident=a.ident", "maps=True, mode=a.map_mode, ident=a.ident", "real"),
    (T0, "the capacity control loads the maps of its own frame", "maps=not a.no_map, mode=a.map_mode, ident=a.ident", 'maps=not a.no_map, mode="train", ident=a.ident', "real"),
    (T0, "map_mode is not in the recipe", '"lr", "no_map", "map_mode")', '"lr", "no_map")'),
    (T0, "the capacity control without the block is accepted", 'assert not args["no_map"] or args["map_mode"] == "train",', 'assert True or args["map_mode"] == "train",'),
    (T0, "the smoke maps of the capacity control are the frame's own", "maps={k: v.roll(1, 0) for k, v in maps.items()} if other else maps)", "maps=maps)"),
    (T0, "the smoke batches ignore the map mode", 'init * 100003 + step, a.no_map, a.map_mode == "other")', "init * 100003 + step, a.no_map, False)"),
    (T0, "the maps stay on the CPU", 'MapsView(m, {k: v.to(dev, non_blocking=True) for k, v in b["maps"].items()})', 'MapsView(m, {k: v for k, v in b["maps"].items()})'),
    (T0, "the checkpoint interval is 400 steps", "ckpt_every=500, smoke_k=SMOKE_K):", "ckpt_every=400, smoke_k=SMOKE_K):"),
    (T0, "the smoke frames have 384 tokens", "SMOKE_K = 512 ", "SMOKE_K = 384 "),''')
sub('''    (E0, "the sibling has no model.pt",''', '''    (E0, "the TRAIN read-out takes the VAL frames", "idx = JE.sample(len(ds), a.split, 1)[: a.max or None]", 'idx = JE.sample(len(ds), "val", 1)[: a.max or None]'),
    (E0, "params counts the block twice", "params_block=JK.count_params(m.map_block) if getattr(m, \\"map_block\\", None) is not None else 0)", "params_block=2 * JK.count_params(m.map_block) if getattr(m, \\"map_block\\", None) is not None else 0)"),
    (E0, "the GMACs without the block are those with it", "gmacs_no_map = head_macs(m, first) if not no_map else gmacs", "gmacs_no_map = gmacs"),
    (E0, "the timing ignores --K", "K = a.K or ds.tok.K", "K = ds.tok.K"),
    (E0, "the timing reads the first frames of a shorter list", "items = [cut(LMD.collate([ds[i]])) for i in range(a.frames)]", "items = [cut(LMD.collate([ds[i]])) for i in range(a.frames + 1)]"),
    (E0, "the timing measures nothing", "median_ms=float(np.median(flat))", "median_ms=0.0"),
    (E0, "the sibling has no model.pt",''')
open(p, "w", encoding="utf-8").write(s)
print("patched tests")
