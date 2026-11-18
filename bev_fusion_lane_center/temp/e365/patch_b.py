import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def cut(start_marker, end_marker, new):
    global s
    a = s.index(start_marker)
    b = s.index(end_marker)
    s = s[:a] + new + s[b:]


cut("def score(a, runs=RUNS):", "def ctl(a, runs=RUNS):", '''def score(a, runs=RUNS, out_dir=None):
    """the cards of one dump (module docstring). --max N reads the dump of `dump --max N`, scores its N frames and writes the cards to `out_dir` (default temp/e365/out) as ..._n<N>.json, with the controls of `ctl --max N` there"""
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    n_max = a.max
    out_dir = (out_dir or os.path.join(SCRATCH, "out")) if n_max else None
    sfx = ("" if a.split == "val" else f"_{a.split}") + JV.suffix(2.5, a.drop, a.drop_seed)
    nsfx = f"_n{n_max}" if n_max else ""
    paths = {d: os.path.join(out_dir, f"{a.run}_token_card_{d}{sfx}{nsfx}.json") if n_max else os.path.join(runs, a.run, f"token_card_{d}{sfx}.json") for d in DECODES}
    held = [p for p in paths.values() if os.path.exists(p)]
    assert not held, f"a stored card is never overwritten: {held}"
    d = pickle.load(open(os.path.join(runs, a.run, f"token_{a.split}{JV.suffix(2.5, a.drop, a.drop_seed)}{nsfx}.pkl"), "rb"))
    meta, fr = d["meta"], d["frames"]
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split)[: n_max or None]
    assert meta["run"] == a.run and meta["idx"] == idx.tolist() and meta["drop"] == a.drop and meta["drop_seed"] == a.drop_seed and [f["token"] for f in fr] == [ds.tokens[i] for i in idx], "the dump is not the dump of this read-out (run, frames, drop)"
    toks = [ds[int(i)]["tokens_np"] for i in idx]
    assert all(np.array_equal(f["cell"], t["cell"]) for f, t in zip(fr, toks)), "the dump's tokens are not the cache's (a stale dump or cache)"
    every = frames_of(a.split, ds)
    frames = [every[int(i)] for i in idx]
    ref = stored_plain_b(a.seed) if a.split == "val" and not a.drop else None
    ref_rec = None if ref is None else {k: v[:len(frames)] for k, v in ref["rec"].items()}
    b0 = bsec0_close_path(kw["ident"], a.seed, n_max) if ref is not None and kw["ident"] == "e292_a" else None
    b0 = pickle.load(open(b0, "rb")) if b0 and os.path.exists(b0) else None
    params = JK.count_params(JK.JTokenHead())
    out = {}
    for dec in DECODES:
        res = score_preds([frame_prediction(t, f, dec) for t, f in zip(toks, fr)], frames)
        show(f"{a.run} {dec}{sfx}{nsfx}", res["graph"])
        extra = {}
        if ref is not None:
            extra["paired_vs_plain_b"] = QE.paired(res["rec"], ref_rec, frames)
            extra["plain_b"] = {k: QE.graph_numbers(ref_rec["graph"])[k] for k in KEYS}
            extra["plain_b_dump"] = ref.get("dump")
            if b0 is not None:
                extra["paired_vs_bsec0_close"] = QE.paired(res["rec"], {k: v[:len(frames)] for k, v in b0["rec"].items()}, frames)
            ctl_hi, ctl_rule = (read_ctl(x, kw["ident"], a.seed, ds.K, out_dir or CTL, n_max) for x in ("hi", "rule"))
            extra["g1_topo_F1"] = None if ctl_hi is None or ctl_rule is None else g1(res["graph"]["topo_F1"], ctl_hi, ctl_rule)
            if extra["g1_topo_F1"] is not None:
                extra["g1_note"] = G1_NOTE
        out[dec] = write_card(paths[dec], f"{a.run}_token_{dec}{sfx}{nsfx}", res, len(frames), a.split, run=a.run, seed=a.seed, decode=dec, drop=a.drop, drop_seed=a.drop_seed, steps=ck.get("steps"), head_params=params,
                              K=ds.K, ident=kw["ident"], decode_rule="rec: p >= 0.3 and p >= p(none), both ends kept (p_none < 0.5); argmax: one link per source and class", **extra)
    return out


''')

cut("def ctl(a, runs=RUNS):", "def live_tokens(", '''def ctl(a, runs=RUNS):
    ds = open_cache("val", a.seed, a.ident, a.K, a.cache)
    arms = a.arms.split(",") if a.arms else list(ARMS)
    assert all(x in ARMS for x in arms), arms
    n = a.max or len(ds)
    idx = np.arange(n)
    frames = QE.val_frames()[:n]
    assert [f.token for f in frames] == list(ds.tokens[:n])
    out_dir = os.path.join(SCRATCH, "out") if a.max else CTL
    paths = {x: os.path.join(out_dir, ctl_name(x, a.ident, a.seed, ds.K, a.max)) for x in arms}
    held = [p for p in paths.values() if os.path.exists(p) and not a.max]
    assert not held, f"a stored card is never overwritten: {held}"
    preds = {x: [] for x in arms if x in ("hi", "hi_y", "rule", "floor")}
    m = floor_model() if "floor" in arms else None
    if "hi" in preds:
        import lane_tok_data as TD
        td = TD.TokData("val", LATTICE, feats=False)
        assert list(td.tokens[:n]) == list(ds.tokens[:n]), "the GT tables are not the frames of the cache, in order"
    if preds:
        k = 0
        for b in batches(ds, idx):
            fl = batch_dump(run_head(m, b, "cpu"), b) if m is not None else None
            for i, t in enumerate(b["tokens_np"]):
                gid = b["label"]["gt_id"][i].numpy()
                if "hi" in preds:
                    preds["hi"].append(control_prediction(t, *hi_frame(t, gid, td.table(k + i))))
                if "hi_y" in preds:
                    preds["hi_y"].append(control_prediction(t, *hi_y_frame(t, gid, b["y"][i].numpy(), b["yscale"][i].numpy())))
                if "rule" in preds:
                    preds["rule"].append(control_prediction(t, *rule_frame(t)))
                if "floor" in preds:
                    preds["floor"].append(frame_prediction(t, fl[i], "rec"))
            k += len(b["tokens_np"])
    for x in arms:
        if x.startswith("bsec0"):
            dump_path = _abs(a.dump or B_DUMPS[a.ident].format(seed=a.seed))
            d = pickle.load(open(dump_path, "rb"))[:n]
            assert [y["row"] for y in d] == [f.row for f in frames], f"{dump_path} is not in the order of the VAL frames"
            preds[x] = [pred_of_dump(y, PLAIN if x == "bsec0_plain" else CLOSE, True) for y in d]
    for x in arms:
        res = score_preds(preds[x], frames)
        show(f"ctl {x} {a.ident} s{a.seed}", res["graph"])
        write_card(paths[x], f"e365_ctl_{x}_{a.ident}_s{a.seed}_K{ds.K}", res, len(frames), "val", arm=x, ident=a.ident, seed=a.seed, K=ds.K, frames_scored=n)
        if x == "bsec0_close" and a.ident == "e292_a":                                       # B' read out with the close-node step, sections 0: the records for the paired interval of `score`
            os.makedirs(SCRATCH, exist_ok=True)
            pickle.dump(dict(res, dump=_file_id(dump_path)), open(bsec0_close_path(a.ident, a.seed, a.max), "wb"))
    return paths


def check_layer1(a):
    """the scorer path fed B's stored predictions (the fresh VAL dump, B's own sections, the E275 close-node step, links at p >= 0.5, no connectors: what `jtest_eval.score_dumps` makes of the `plain` variant) against the stored base numbers of the
    E324 s card -> 0 (PASS: topo F1, F1@1.0, lanes found, dup_frac, vert100_dp within 1e-6) / 1 (FAIL); the other stored numbers are printed, and the sha256 and mtime of the dump used. Only a PASS stores plain B's records (temp/e365/planb_s<S>.pkl),
    which `score` needs"""
    ref = json.load(open(BASE_CARD.format(seed=a.seed)))["variants"]["primary"]["numbers"]["base"]
    res = plain_b(a.seed, a.dump)
    ok, diff = compare_numbers(res["graph"], ref)
    for k in ref:
        print(f"   {k}: {res['graph'][k]:.9f} stored {ref[k]:.9f} diff {res['graph'][k] - ref[k]:+.2e}" + ("" if k in CHECK else "   (not checked)"))
    print(f"dump {res['dump']['path']} sha256 {res['dump']['sha256']} mtime {res['dump']['mtime']}")
    print(f"{'PASS' if ok else 'FAIL'}: layer 1, seed {a.seed}: this scorer on B's stored predictions against {os.path.relpath(BASE_CARD.format(seed=a.seed), ROOT)}: " + ", ".join(f"{k} {v:+.2e}" for k, v in diff.items()))
    if ok:
        os.makedirs(SCRATCH, exist_ok=True)
        pickle.dump(res, open(planb_path(a.seed), "wb"))
    return int(not ok)


''')

old = '''            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; the score needs all of them)")
'''
assert s.count(old) == 1
s = s.replace(old, '''            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; `score --max N` reads it)")
        else:
            s.add_argument("--max", type=int, default=0, help="score the dump of `dump --max N` (cards to temp/e365/out/)")
''')
open(p, "w", encoding="utf-8").write(s)
