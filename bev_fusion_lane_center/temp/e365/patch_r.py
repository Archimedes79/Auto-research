import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


# dump: candidate scales and float32 none probabilities (new dumps; older ones lack the scales)
rep('''    pt, p, cand, longm, scale, none_cls = (npy(x) for x in (pr["pt"], pr["p"], pr["cand"], pr["longm"], out["scale"].float(), out["logp"][..., K].exp()))''',
    '''    pt, p, cand, longm, scale, none_cls = (npy(x) for x in (pr["pt"], pr["p"], pr["cand"], pr["longm"], out["scale"].float(), out["logp"][..., K].float().exp()))''')
rep('''none_cls=none_cls[i].astype(np.float16), cand=dict(src=cs.astype(np.int16), dst=cd.astype(np.int16), p=p[i][cs, cd].astype(np.float32), long=longm[i][cs, cd]),''',
    '''none_cls=none_cls[i].astype(np.float32), cand=cand_of(cs, cd, p[i], longm[i], scale[i]),''')
rep('''def batch_dump(out, b):''', '''def cand_of(cs, cd, p, longm, scale):
    """the candidate links of a frame (p >= KEEP_P, in the gate): src, dst, p, long and the handle scales a, b (the head's for the long ones, 1/3 for the short ones, as `links_of`)"""
    lg = longm[cs, cd]
    ab = np.where(lg[:, None], scale[cs, cd], THIRD)
    return dict(src=cs.astype(np.int16), dst=cd.astype(np.int16), p=p[cs, cd].astype(np.float32), long=lg, a=ab[:, 0].astype(np.float32), b=ab[:, 1].astype(np.float32))


def batch_dump(out, b):''')
rep('''    (src, dst, p, long), rec / argmax''', '''    (src, dst, p, long, a, b; the dumps of before the decode variants lack a, b and hold none_cls in float16), rec / argmax''')

# frame_prediction with keep_thr
rep('''def frame_prediction(tokens, f, decode):''', '''LINK_THR, KEEP_THR = 0.3, 0.5                                               # the decode of record


def node_e(pnone, keep_thr=KEEP_THR):
    """the existence written for a kept node: 1 - p_none, shifted by keep_thr - 0.5 and capped at 1, so that the scorer's `e >= 0.5` is `p_none < keep_thr`; at the record's 0.5 exactly 1 - p_none"""
    return np.minimum(1.0 - np.asarray(pnone, np.float32).astype(float) + (keep_thr - 0.5), 1.0)


def frame_prediction(tokens, f, decode, keep_thr=KEEP_THR):''')
rep('''    pred = check_writer(LP.to_pred(tokens, f["keep"], 1.0 - f["pnone"].astype(float), lk), tokens, f["keep"])
    keep, idx = np.asarray(f["keep"], bool), np.flatnonzero(f["keep"])
    assert np.array_equal(pred["nodes"]["e"], 1.0 - np.asarray(f["pnone"], np.float32)[idx].astype(float)), "e is not 1 - p_none of the dump"''',
    '''    pred = check_writer(LP.to_pred(tokens, f["keep"], node_e(f["pnone"], keep_thr), lk), tokens, f["keep"])
    keep, idx = np.asarray(f["keep"], bool), np.flatnonzero(f["keep"])
    assert np.array_equal(pred["nodes"]["e"], node_e(f["pnone"], keep_thr)[idx]), "e is not 1 - p_none of the dump"''')

# decode variant
rep('''# ------------------------------------------------------------------------------------------------------------------------------ the controls''', '''def decode_variant(f, valid, link_thr=LINK_THR, keep_thr=KEEP_THR, none_gate=True):
    """a decode of the dump frame `f` other than the record: nodes kept at p_none < keep_thr (valid tokens), links = every candidate with p >= link_thr (>= KEEP_P, the dump holds the candidates above it), and p >= p(none|i) of the class of the pair
    unless `none_gate` is off, both ends kept -> the frame with `keep` and `rec` replaced. The handle scales are the dump's (`cand` a, b); a dump of before them gives the scales of the record's links where the link is one and 1/3 else. The default
    arguments give exactly the decode of record"""
    assert link_thr >= KEEP_P, f"the dump holds the candidates with p >= {KEEP_P}: --link-thr below it cannot be re-scored"
    keep = np.asarray(valid, bool) & (np.asarray(f["pnone"]) < np.float32(keep_thr))
    c = f["cand"]
    src, dst = c["src"].astype(int), c["dst"].astype(int)
    m = (c["p"] >= np.float32(link_thr)) & keep[src] & keep[dst]
    if none_gate:
        m &= c["p"] >= np.asarray(f["none_cls"], np.float32)[c["long"].astype(int), src]
    if "a" in c:
        a, b = c["a"][m], c["b"][m]
    else:
        known = {(int(s_), int(d_)): (a_, b_) for dec in DECODES for s_, d_, a_, b_ in zip(f[dec]["src"], f[dec]["dst"], f[dec]["a"], f[dec]["b"])}
        ab = np.array([known.get((int(s_), int(d_)), (THIRD, THIRD)) for s_, d_ in zip(src[m], dst[m])], np.float32).reshape(-1, 2)
        a, b = ab[:, 0], ab[:, 1]
    return dict(f, keep=keep, rec=dict(src=c["src"][m], dst=c["dst"][m], p=c["p"][m], long=c["long"][m], a=a, b=b))


# ------------------------------------------------------------------------------------------------------------------------------ the controls''')

# score + sweep
a = s.index("def score(a, runs=RUNS, out_dir=None):")
b = s.index("def ctl(a, runs=RUNS):")
new = '''def variant_suffix(a):
    """'' for the decode of record, else _lt<link thr>_kt<keep thr>[_nng] (no none gate)"""
    return "" if (a.link_thr, a.keep_thr, a.no_none_gate) == (LINK_THR, KEEP_THR, False) else f"_lt{a.link_thr:g}_kt{a.keep_thr:g}" + ("_nng" if a.no_none_gate else "")


def read_dump(a, runs, n_max):
    """the dump of this read-out with everything the scorer needs, checked against the cache -> dict(ck, args, kw, ds, idx, meta, fr, toks, valid, frames)"""
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    nsfx = f"_n{n_max}" if n_max else ""
    d = pickle.load(open(os.path.join(runs, a.run, f"token_{a.split}{JV.suffix(2.5, a.drop, a.drop_seed)}{nsfx}.pkl"), "rb"))
    meta, fr = d["meta"], d["frames"]
    kw = cache_kw(args)
    ds = open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = sample(len(ds), a.split)[: n_max or None]
    assert meta["run"] == a.run and meta["idx"] == idx.tolist() and meta["drop"] == a.drop and meta["drop_seed"] == a.drop_seed and [f["token"] for f in fr] == [ds.tokens[i] for i in idx], "the dump is not the dump of this read-out (run, frames, drop)"
    toks = [ds[int(i)]["tokens_np"] for i in idx]
    assert all(np.array_equal(f["cell"], t["cell"]) for f, t in zip(fr, toks)), "the dump's tokens are not the cache's (a stale dump or cache)"
    valid = [np.asarray(t["valid"], bool) & (JV.keep_nodes(f["token"], ds.K, a.drop, a.drop_seed) if a.drop > 0 else True) for t, f in zip(toks, fr)]     # the tokens the head saw
    assert all(np.array_equal(v & (f["pnone"] < np.float32(KEEP_THR)), f["keep"]) for v, f in zip(valid, fr)), "the valid tokens rebuilt from the cache are not the ones the dump's kept nodes came from"
    every = frames_of(a.split, ds)
    return dict(ck=ck, args=args, kw=kw, ds=ds, idx=idx, meta=meta, fr=fr, toks=toks, valid=valid, frames=[every[int(i)] for i in idx])


def score(a, runs=RUNS, out_dir=None):
    """the cards of one dump (module docstring). --max N reads the dump of `dump --max N`, scores its N frames and writes the cards to `out_dir` (default temp/e365/out) as ..._n<N>.json, with the controls of `ctl --max N` there.
    --link-thr T --keep-thr K [--no-none-gate] re-score the dump with another decode (`decode_variant`; only the `rec` card, its name carries the variant); the defaults are the decode of record, both decodes"""
    n_max, vsfx = a.max, variant_suffix(a)
    out_dir = (out_dir or os.path.join(SCRATCH, "out")) if n_max else None
    sfx = ("" if a.split == "val" else f"_{a.split}") + JV.suffix(2.5, a.drop, a.drop_seed)
    nsfx = f"_n{n_max}" if n_max else ""
    decs = ("rec",) if vsfx else DECODES
    paths = {d: os.path.join(out_dir, f"{a.run}_token_card_{d}{vsfx}{sfx}{nsfx}.json") if n_max else os.path.join(runs, a.run, f"token_card_{d}{vsfx}{sfx}.json") for d in decs}
    held = [p for p in paths.values() if os.path.exists(p)]
    assert not held, f"a stored card is never overwritten: {held}"
    r = read_dump(a, runs, n_max)
    ck, kw, ds, frames = r["ck"], r["kw"], r["ds"], r["frames"]
    ref = stored_plain_b(a.seed) if a.split == "val" and not a.drop else None
    ref_rec = None if ref is None else {k: v[:len(frames)] for k, v in ref["rec"].items()}
    b0 = bsec0_close_path(kw["ident"], a.seed, n_max) if ref is not None and kw["ident"] == "e292_a" else None
    b0 = pickle.load(open(b0, "rb")) if b0 and os.path.exists(b0) else None
    params = JK.count_params(JKT.build_head(r["args"]))
    out = {}
    for dec in decs:
        fr = [decode_variant(f, v, a.link_thr, a.keep_thr, not a.no_none_gate) for f, v in zip(r["fr"], r["valid"])] if vsfx else r["fr"]
        res = score_preds([frame_prediction(t, f, dec, a.keep_thr) for t, f in zip(r["toks"], fr)], frames)
        show(f"{a.run} {dec}{vsfx}{sfx}{nsfx}", res["graph"])
        extra = {}
        if vsfx:
            extra["decode_variant"] = dict(link_thr=a.link_thr, keep_thr=a.keep_thr, none_gate=not a.no_none_gate, scales="the head's (dump candidates a, b)" if "a" in r["fr"][0]["cand"] else "the record's links keep their scales, other links 1/3 (a dump of before the variants)")
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
        out[dec] = write_card(paths[dec], f"{a.run}_token_{dec}{vsfx}{sfx}{nsfx}", res, len(frames), a.split, run=a.run, seed=a.seed, decode=dec, drop=a.drop, drop_seed=a.drop_seed, steps=ck.get("steps"), head_params=params,
                              K=ds.K, ident=kw["ident"], decode_rule="rec: p >= 0.3 and p >= p(none), both ends kept (p_none < 0.5); argmax: one link per source and class" if not vsfx else "the variant of `decode_variant`", **extra)
    return out


SWEEP_LINK, SWEEP_KEEP = (0.3, 0.2, 0.1, 0.05), (0.5, 0.6, 0.7)
SWEEP_KEYS = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "dup_frac")


def sweep(a, runs=RUNS, out_dir=None):
    """the decode variants of one dump on its read-out frames: link threshold {0.3, 0.2, 0.1, 0.05} x keep threshold {0.5, 0.6, 0.7} x none gate {on, off} -> one table runs/R/token_sweep_<split>[_drop..].json (--max N: temp/e365/out/..._n<N>.json),
    never overwritten; the cell (0.3, 0.5, on) is the decode of record. For choosing a decode on TRAIN frames (the head was trained on them: the table says what the output could deliver, not what it does on VAL)"""
    n_max = a.max
    sfx = ("" if a.split == "val" else f"_{a.split}") + JV.suffix(2.5, a.drop, a.drop_seed)
    path = os.path.join((out_dir or os.path.join(SCRATCH, "out")) if n_max else os.path.join(runs, a.run), f"{a.run + '_' if n_max else ''}token_sweep{sfx}{f'_n{n_max}' if n_max else ''}.json")
    assert not os.path.exists(path), f"a stored table is never overwritten: {path}"
    r = read_dump(a, runs, n_max)
    rows = []
    for lt in SWEEP_LINK:
        for kt in SWEEP_KEEP:
            for gate in (True, False):
                fr = [decode_variant(f, v, lt, kt, gate) for f, v in zip(r["fr"], r["valid"])]
                res = score_preds([frame_prediction(t, f, "rec", kt) for t, f in zip(r["toks"], fr)], r["frames"], log=lambda *x, **k: None)
                rows.append(dict(link_thr=lt, keep_thr=kt, none_gate=gate, record=(lt, kt, gate) == (LINK_THR, KEEP_THR, True), frames=len(fr), kept_nodes_per_frame=float(np.mean([f["keep"].sum() for f in fr])),
                                 links_per_frame=float(np.mean([len(f["rec"]["src"]) for f in fr])), **{k: res["graph"][k] for k in SWEEP_KEYS}))
                print(f"link {lt:<5g} keep {kt:<4g} gate {'on ' if gate else 'off'}  " + "  ".join(f"{k} {rows[-1][k]:.4f}" for k in SWEEP_KEYS) + f"  links/frame {rows[-1]['links_per_frame']:.1f}", flush=True)
    JV.write_json(path, _jsonable(dict(run=a.run, seed=a.seed, split=a.split, drop=a.drop, frames=len(r["frames"]), rows=rows, K=r["ds"].K, note="decode variants of one dump (`decode_variant`); scales: " + ("the head's" if "a" in r["fr"][0]["cand"] else "the record's links keep theirs, others 1/3"))))
    print(f"written {os.path.relpath(path, ROOT)}")
    return path


'''
s = s[:a] + new + s[b:]

# parser
rep('''    for name in ("dump", "score"):
        s = sub.add_parser(name)''', '''    for name in ("dump", "score", "sweep"):
        s = sub.add_parser(name)''')
rep('''        else:
            s.add_argument("--max", type=int, default=0, help="score the dump of `dump --max N` (cards to temp/e365/out/)")
''', '''        else:
            s.add_argument("--max", type=int, default=0, help="score the dump of `dump --max N` (cards to temp/e365/out/)")
        if name == "score":
            s.add_argument("--link-thr", type=float, default=LINK_THR, help="a decode variant: every candidate with p >= T is a link (>= 0.05; the decode of record: 0.3)")
            s.add_argument("--keep-thr", type=float, default=KEEP_THR, help="a decode variant: a node is kept at p_none < K (record: 0.5)")
            s.add_argument("--no-none-gate", action="store_true", help="a decode variant: no p >= p(none) condition on a link")
''')
rep('''"score": score, "ctl": ctl,''', '''"score": score, "sweep": sweep, "ctl": ctl,''')
# docstring
rep('''    wsl bash tools/env/_py14.sh jtoken_eval.py ctl ''', '''    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R --seed S --link-thr T --keep-thr K [--no-none-gate]                                      # CPU: another decode of the same dump -> token_card_rec_lt<T>_kt<K>[_nng][_train]..json
    wsl bash tools/env/_py14.sh jtoken_eval.py sweep --run R --seed S --split train [--max N]                                                       # CPU: the grid of decode variants on the TRAIN sample -> runs/R/token_sweep_train.json
    wsl bash tools/env/_py14.sh jtoken_eval.py ctl ''')
open(p, "w", encoding="utf-8").write(s)
