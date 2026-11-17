def refuses(fn, text):
    """fn must fail with an AssertionError whose message holds `text`; another exception is reported as such, a call that is accepted raises a RuntimeError (which no test catches)"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    except Exception as e:
        raise AssertionError(f"raised {type(e).__name__} ({e}) instead of refusing ({text})") from e
    raise RuntimeError(f"accepted although it must be refused ({text})")


def _t_labels():
    B, K = 2, 16
    no = toy_no(B)
    tk = live_tokens(no, K)
    fl = np.zeros((3, 8), bool)
    fl[1, 0], fl[2, 2] = True, True                                                                             # node 1 is a stop, node 2 a split
    tabs = [toy_table([[-9.0, 5.3], [-7.0, 5.0], [-5.0, 4.6]], [(0, 1), (1, 2)], [False, False], fl), toy_table(np.zeros((0, 2)), [], [])]      # the GT nodes lie 0.3 / 0 / 0.4 m from the tokens
    out = live_batch(tk, tabs, ["f0", "f1"], [7, 8], K)
    assert out["y"].shape == (B, K, K) and out["y"].dtype == torch.uint8 and out["ylong"].shape == (B, K, K) and out["yscale"].shape == (B, K, K, 2) and out["ytype"].dtype == torch.int64 and out["row"].tolist() == [7, 8]
    lb = out["label"]
    assert lb["gt_id"][0].tolist() == [0, 1, 2] + [-1] * 13 and (lb["gt_id"][1] == -1).all(), "the three tokens take the three GT nodes in the order of their existence"
    assert np.allclose(lb["gd"][0][:3].numpy(), [0.3, 0.0, 0.4], atol=1e-6) and torch.isnan(lb["gd"][0][3:]).all() and torch.isnan(lb["gd"][1]).all(), "gd = the distance of a token to its GT node, NaN without one"
    assert lb["gt_flags"][0][1].tolist() == [True] + [False] * 7 and lb["gt_flags"][0][2].tolist() == [False, False, True] + [False] * 5 and not lb["gt_flags"][0][0].any() and not lb["gt_flags"][0][3:].any() and not lb["gt_flags"][1].any(), "the raw flags of the assigned node"
    assert lb["gt_aux"][0][0].tolist() == [True, False, False] and not lb["gt_aux"][0][1:].any() and not lb["gt_aux"][1].any(), "the section start of node 0; no long link"
    assert out["ytype"][0].tolist() == [1, 4, 2] + [0] * 13 and out["ytype"][1].sum() == 0, "lane point, stop = entry (4), split = fork (2); the padding and the empty frame are type 0"
    assert out["y"][0].nonzero().tolist() == [[0, 1], [1, 2]] and int(out["y"][1].sum()) == 0 and not out["ylong"].any() and torch.isnan(out["yscale"]).all()
    assert out["valid"][0].tolist() == [True] * 3 + [False] * 13 and out["gt_out"].tolist() == [0, 0] and out["token"] == ["f0", "f1"]
    assert out["lab"][0]["n_reach"] == 2 and out["lab"][0]["n_pairs"] == 2 and out["lab"][1]["n_gt"] == 0, "both pairs lie inside the gate of pair_geometry on the live tokens (forward along x)"
    tn = out["tokens_np"][0]
    assert tn["pos"].dtype == np.float64 and tn["cell"].dtype == np.int64 and tn["cell"][:3].tolist() == [1027, 1077, 1127] and np.allclose(tn["pos"][:3], [[-9, 5], [-7, 5], [-5, 5]])
    assert sorted(tn) == ["cell", "dirv", "e_p", "flags_p", "pos", "s_p", "valid"], "no GT field in tokens_np"
    # the values of tokens_np by hand (sigmoid of the toy logits: e 5, 4, 3 then -8; s 2, -1, 0 then 0; split flag 1, 0, -1 then 0; every other flag 0), the headings (1, 0), valid for the first three
    fp = np.full((K, 8), 0.5)
    fp[:3, 2] = [0.73105858, 0.5, 0.26894142]
    assert np.allclose(tn["e_p"][:3], [0.99330715, 0.98201379, 0.95257413], atol=1e-6) and np.allclose(tn["e_p"][3:], 3.3535013e-4, atol=1e-7), "e_p = sigmoid(e) of the whole map, gathered"
    assert np.allclose(tn["s_p"][:3], [0.88079708, 0.26894142, 0.5], atol=1e-6) and np.allclose(tn["s_p"][3:], 0.5, atol=1e-6) and np.allclose(tn["flags_p"], fp, atol=1e-6), "s_p and flags_p = their own sigmoids"
    assert np.allclose(tn["dirv"], [[1.0, 0.0]] * K, atol=1e-6) and tn["valid"].tolist() == [True] * 3 + [False] * 13 and tn["valid"].dtype == bool
    # the long chord class follows the TOKEN chord (12 m), not the table's class of the link; a chord of exactly 8 m is short (long = above 8 m)
    o2 = live_batch(live_tokens(toy_no(1, rank_cells=((20, 27), (26, 27))), K), [toy_table([[-9.0, 5.0], [3.0, 5.0]], [(0, 1)], [False])], ["g"], [0], K)
    assert o2["y"][0].nonzero().tolist() == [[0, 1]] and bool(o2["ylong"][0, 0, 1]) and int(o2["ylong"].sum()) == 1
    o4 = live_batch(live_tokens(toy_no(1, rank_cells=((20, 27), (24, 27))), K), [toy_table([[-9.0, 5.0], [-1.0, 5.0]], [(0, 1)], [False])], ["i"], [0], K)
    assert o4["y"][0].nonzero().tolist() == [[0, 1]] and not o4["ylong"].any(), "a chord of exactly 8 m is short"
    # a GT node that no token covers: the link into it is lost (gt_out counts it before the contraction)
    o3 = live_batch(live_tokens(toy_no(1), K), [toy_table([[-9.0, 5.0], [-7.0, 5.0], [-7.0, 40.0]], [(0, 1), (1, 2)], [False, True])], ["h"], [0], K)
    assert o3["gt_out"].tolist() == [1] and o3["lab"][0]["gt_out"] == 1 and o3["lab"][0]["n_links"] == 2 and o3["lab"][0]["n_links_c"] == 1
    # the pair gate only feeds statistics: without it y, ylong, yscale, valid, ytype and gt_out are the same
    og = live_batch(tk, tabs, ["f0", "f1"], [7, 8], K, gate=False)
    for k in ("valid", "ytype", "y", "ylong", "yscale", "gt_out"):
        assert np.array_equal(og[k].numpy(), out[k].numpy(), equal_nan=True), f"{k} differs without the gate"
    assert og["lab"][0]["n_reach"] == 0 and og["lab"][0]["n_pairs"] == 2 and not og["lab"][0]["ypair_gate"].any() and out["lab"][0]["ypair_gate"][:2].all(), "gate=False: no pair is counted as reachable, nothing else changes"
    # a frame over PMAX target pairs is named, with its token and its count (the trainer decides)
    pmax, LC.PMAX = LC.PMAX, 1
    try:
        live_batch(tk, tabs, ["f0", "f1"], [7, 8], K)
        raise RuntimeError("accepted although it must be refused (more target pairs than PMAX)")
    except LabelOverflow as e:
        assert e.token == "f0" and e.pairs == 2 and isinstance(e, RuntimeError) and "exceed PMAX" in str(e), str(e)
    finally:
        LC.PMAX = pmax
    # the loader's token drop: the draw of `jtok_eval.keep_nodes` seeded by (seed, token); a removed token is not valid, has type none and no pair
    from jtok_eval import keep_nodes
    seed = next(s for s in range(1, 50) if not keep_nodes("f0", K, 0.5, s)[:3].all() and keep_nodes("f0", K, 0.5, s)[:3].any() and not np.array_equal(keep_nodes("f0", K, 0.5, s)[:3], keep_nodes("f0", K, 0.5, 0)[:3]))
    keep = torch.from_numpy(keep_nodes("f0", K, 0.5, seed))
    od = live_batch(tk, tabs[:1] * 2, ["f0", "f0"], [7, 7], K, token_drop=0.5, seed=seed)
    assert od["valid"][0].tolist() == (out["valid"][0] & keep).tolist() and od["valid"][0].tolist() != out["valid"][0].tolist() and od["tokens_np"][0]["valid"].tolist() == od["valid"][0].tolist()
    assert (od["ytype"][0][~keep] == 0).all() and not od["y"][0][~keep].any() and not od["y"][0][:, ~keep].any() and int(od["y"][0].sum()) < int(out["y"][0].sum())


def _t_data():
    import tempfile
    with open(os.path.join(ROOT, "data", "splits.json")) as fh:
        rows = json.load(fh)["val"]
    toks = [f"t{i}" for i in range(len(rows))]
    ds = LiveData("val", 0, td=_Toy(rows, toks))
    assert len(ds) == len(rows) and ds[5]["token"] == "t5" and ds[5]["row"] == rows[5] and float(ds[5]["x"].float().mean()) == 5.0 and float(ds[5]["table"]["pos"][0, 1]) == 10.5, "item i is frame i"
    b = collate([ds[3], ds[4]])
    assert b["x"].shape == (2, 39, 4, 4) and b["row"].tolist() == [rows[3], rows[4]] and b["token"] == ["t3", "t4"] and len(b["table"]) == 2 and sorted(b["targets"]) == sorted(NODE_KEYS)
    assert [float(t["pos"][0, 1]) for t in b["table"]] == [6.5, 8.5], "table[i] is the table of frame i"
    assert b["targets"]["exists"].shape == (2, 50, 50) and b["targets"]["exists"][0].nonzero().tolist() == [[20, 28]] and b["targets"]["exists"][1].nonzero().tolist() == [[20, 29]], "targets[i] belong to frame i"
    refuses(lambda: LiveData("val", 0, td=_Toy(rows[::-1], toks)), "VAL cache are not data/splits.json['val']")                   # the TEST guard
    refuses(lambda: LiveData("val", 0, td=_Toy(rows[:-1], toks[:-1])), "VAL cache are not data/splits.json['val']")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "frames.json"), "w") as fh:
            json.dump(dict(rows=[1, 2, 3], tokens=["a", "b", "c"]), fh)
        check_order(d, [1, 2, 3], ["a", "b", "c"])
        refuses(lambda: check_order(d, [1, 3, 2], ["a", "b", "c"]), "not the frames of this data")                  # the rows alone, the tokens alone
        refuses(lambda: check_order(d, [1, 2, 3], ["a", "c", "b"]), "not the frames of this data")
        refuses(lambda: check_order(d, [1, 2], ["a", "b"]), "not the frames of this data")
        with open(os.path.join(d, "frames.json"), "w") as fh:                                                      # LiveData itself runs the order check (here against an injected cache directory)
            json.dump(dict(rows=rows, tokens=toks), fh)
        LiveData("val", 0, td=_Toy(rows, toks), cache_dir=d)
        refuses(lambda: LiveData("val", 0, td=_Toy(rows, toks[::-1]), cache_dir=d), "not the frames of this data")
        os.makedirs(os.path.join(d, "run"))                                                                        # the checkpoint of the cache: sha256("abc") starts ba7816bf8f01cfea
        with open(os.path.join(d, "run", "model.pt"), "wb") as fh:
            fh.write(b"abc")
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(dict(run_sha="ba7816bf8f01cfea"), fh)
        assert check_run_sha(d, os.path.join(d, "run")) == "ba7816bf8f01cfea"
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(dict(run_sha="0123456789abcdef"), fh)
        refuses(lambda: check_run_sha(d, os.path.join(d, "run")), "is not the checkpoint that the cache")


def _t_compare():
    K = 6
    rng = np.random.default_rng(0)
    pair = lambda *s: rng.random(s).astype(np.float32)  # noqa: E731
    a = dict(tok=pair(K, 64), pos=pair(K, 2), dirv=pair(K, 2), e=pair(K), e_raw=pair(K), s=pair(K), uout=pair(K), uin=pair(K), flags=pair(K, 8), valid=np.arange(K) < 4, ytype=np.arange(K) % 3, y=(pair(K, K) > 0.7).astype(np.uint8),
             ylong=pair(K, K) > 0.8, yscale=np.where(pair(K, K, 2) > 0.5, pair(K, K, 2), np.nan).astype(np.float32), gt_out=1, cell=np.arange(K) * 7 + 3,
             label=dict(gt_id=np.arange(K, dtype=np.int32) - 2, gd=np.where(np.arange(K) > 1, pair(K), np.nan).astype(np.float32), gt_flags=pair(K, 8) > 0.5, gt_aux=pair(K, 3) > 0.5),
             tnp=dict(pos=pair(K, 2).astype(np.float64), dirv=pair(K, 2).astype(np.float64), e_p=pair(K).astype(np.float64), flags_p=pair(K, 8).astype(np.float64), s_p=pair(K).astype(np.float64), valid=np.arange(K) < 4))
    cp = lambda d: {k: (cp(v) if isinstance(v, dict) else np.copy(v)) for k, v in d.items()}  # noqa: E731
    c = compare_frame(a, cp(a))
    assert c["same_order"] and c["common"] == K and c["only_a"] == c["only_w"] == 0 and not any(c["floats"].values()) and not any(c["ints"].values()), "a frame equals its copy (NaN = NaN)"
    perm = np.array([3, 0, 5, 1, 4, 2])                                                                          # the same tokens in another order: aligned by cell, nothing differs
    w = dict(cp(a), cell=a["cell"][perm], label={k: v[perm] for k, v in a["label"].items()}, tnp={k: v[perm] for k, v in a["tnp"].items()})
    for k in ("tok", "pos", "dirv", "e", "e_raw", "s", "uout", "uin", "flags", "valid", "ytype"):
        w[k] = a[k][perm]
    for k in ("y", "ylong", "yscale"):
        w[k] = a[k][np.ix_(perm, perm)]
    c = compare_frame(a, w)
    assert not c["same_order"] and c["common"] == K and not any(c["floats"].values()) and not any(c["ints"].values()), "a reordering is not a difference"
    for k, edit, field, kind in (("tok", lambda d: d["tok"].__setitem__((2, 5), d["tok"][2, 5] + 0.5), "tok", "floats"), ("e", lambda d: d["e"].__setitem__(3, d["e"][3] + 1.0), "e", "floats"),
                                 ("y", lambda d: d["y"].__setitem__((1, 3), 1 - d["y"][1, 3]), "y", "ints"), ("ytype", lambda d: d["ytype"].__setitem__(4, 9), "ytype", "ints"),
                                 ("valid", lambda d: d["valid"].__setitem__(0, False), "valid", "ints"), ("ylong", lambda d: d["ylong"].__setitem__((2, 2), ~d["ylong"][2, 2]), "ylong", "ints"),
                                 ("gt_id", lambda d: d["label"]["gt_id"].__setitem__(1, 77), "label", "ints"), ("gt_out", lambda d: d.__setitem__("gt_out", 2), "gt_out", "ints"),
                                 ("yscale", lambda d: d["yscale"].__setitem__((0, 0, 0), np.nan if not np.isnan(d["yscale"][0, 0, 0]) else 0.5), "yscale", "floats"),
                                 ("np_e_p", lambda d: d["tnp"]["e_p"].__setitem__(2, d["tnp"]["e_p"][2] + 0.5), "np_e_p", "floats"), ("np_s_p", lambda d: d["tnp"]["s_p"].__setitem__(3, 0.123), "np_s_p", "floats"),
                                 ("np_flags_p", lambda d: d["tnp"]["flags_p"].__setitem__((1, 4), 0.99), "np_flags_p", "floats"), ("np_dirv", lambda d: d["tnp"]["dirv"].__setitem__((2, 0), 9.0), "np_dirv", "floats"),
                                 ("np_pos", lambda d: d["tnp"]["pos"].__setitem__((4, 1), 9.0), "np_pos", "floats"), ("np_valid", lambda d: d["tnp"]["valid"].__setitem__(1, False), "np_valid", "ints")):
        w = cp(a)
        edit(w)
        c = compare_frame(a, w)
        got = c[kind][field]
        assert got != 0 and (kind == "floats" or got == 1) and sum(1 for f_ in c["floats"].values() if f_ != 0) + sum(1 for i_ in c["ints"].values() if i_ != 0) == 1, (k, c)
    w = cp(a)
    w["tok"][2, 5] += 0.25
    assert abs(compare_frame(a, w)["floats"]["tok"] - 0.25) < 1e-6
    w = cp(a)                                                                                                    # another token at the edge of the budget: counted, and its own fields are not compared
    w["cell"][5], w["tok"][5], w["ytype"][5] = 999, 9.0, 5
    c = compare_frame(a, w)
    assert c["common"] == K - 1 and c["only_a"] == 1 and c["only_w"] == 1 and not any(c["floats"].values()) and c["ints"]["ytype"] == 0 and not c["same_order"]


def _t_guards():
    refuses(lambda: fixture(n=0), "nothing to compare")                                                          # no vacuous PASS
    refuses(lambda: fixture(seeds=()), "nothing to compare")
    refuses(lambda: fixture(n=6, gpu=True, bs=4), "not a multiple of bs")                                        # a short last batch is not a batch of the cache build


def _t_source():
    import jtoken as JK
    K, B = 64, 2
    x = torch.randn(B, 39, 200, 200).half()
    m = _toy_model()
    src = LiveSource(m, K=K)
    names = [n for n, p in src.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("model.nodes.") for n in names) and len(names) == len(list(m.nodes.parameters())) and len(src.trainable()) == len(names), "only the node head trains"
    assert not any(p.requires_grad for n, p in src.named_parameters() if not n.startswith("model.nodes.")), "the pyramid, links, map encoder and assigner are frozen"
    src.train()
    assert not any(mm.training for mm in m.pyr.modules()) and m.nodes.training, "the pyramid stays in eval mode, the node head follows the module"
    src.eval()
    assert not m.nodes.training
    src.train()
    assert not any(p.requires_grad for p in LiveSource(_toy_model(), K=K, train_nodes=False).parameters()), "train_nodes=False: nothing trains"
    # a live forward without a graph, and inside the trainer's autocast: B' stays in float32
    tab0 = [toy_table([[-9.0, 5.0]], [], [])] * B
    with torch.no_grad():
        o0 = src(x, tab0, ["a", "b"], [0, 1])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            o1 = src(x, tab0, ["a", "b"], [0, 1])
        o2 = src(x, tab0, ["a", "b"], [0, 1], token_drop=0.5, seed=1)                                            # the loader's token drop reaches the labels through the source
        no_, tk_ = src.tokens(x)                                                                                 # forward = tokens + labelled, bit for bit
        b_ = src.labelled(tk_, tab0, ["a", "b"], [0, 1])
    assert int(o0["batch"]["valid"].sum()) >= 8 and not o0["batch"]["tok"].requires_grad, "enough valid tokens for the tests (the toy bias)"
    assert all(torch.equal(o0["batch"][k], o1["batch"][k]) for k in GRAPH_KEYS), "B' runs in float32 inside an autocast"
    assert 0 < int(o2["batch"]["valid"].sum()) < int(o0["batch"]["valid"].sum()), "token_drop and seed are passed on"
    assert sorted(o0["no"]) == sorted(no_) and all(torch.equal(o0["no"][k], no_[k]) for k in no_), "forward = tokens + labelled: the node-stage outputs"
    assert sorted(o0["batch"]) == sorted(b_) and all(np.array_equal(o0["batch"][k].numpy(), b_[k].numpy(), equal_nan=True) for k in GRAPH_KEYS + ("valid", "ytype", "y", "ylong", "yscale", "gt_out", "row")), "forward = tokens + labelled: the batch"
    head = JK.JTokenHead(d=32, heads=4, layers=2, ffn=64).eval()                                                 # the head reads the INPUT keys only: the tokens alone give the same output as the whole batch (token_drop 0)
    with torch.no_grad():
        h1, h2 = head(head_inputs(tk_)), head(b_)
    assert all(torch.equal(h1[k], h2[k]) for k in h1), "the head's forward reads only the head inputs"
    # tables made of the tokens themselves: two GT nodes at the first allowed pair of valid tokens, linked
    bt = o0["batch"]
    allowed = JT.pair_geometry(bt["pos"], bt["dirv"], bt["valid"])[1]
    tabs = []
    for b in range(B):
        i, j = [int(v) for v in allowed[b].nonzero()[0]]
        t = toy_table(bt["pos"][b][[i, j]].double().numpy(), [(0, 1)], [False])
        t["dirv"] = bt["dirv"][b][[i, j]].double().numpy()
        tabs.append(t)
    with torch.no_grad():
        og = src(x, tabs, ["a", "b"], [0, 1])["batch"]["lab"]
        on = src(x, tabs, ["a", "b"], [0, 1], gate=False)["batch"]["lab"]
    assert sum(lab["n_reach"] for lab in og) > 0 and all(lab["n_reach"] == 0 for lab in on), "gate=False reaches the labels through the source"
    tg = {k: torch.from_numpy(np.stack([node_targets(t)[k] for t in tabs])) for k in NODE_KEYS}
    r = gradient_check(src, JK.JTokenHead(d=32, heads=4, layers=2, ffn=64), dict(x=x, table=tabs, token=["a", "b"], row=[0, 1], targets=tg))        # the card's gradient check
    assert r["params"] == len(names) and r["grad_norm_from_head"] > 0 and r["grad_norm_from_b"] > 0
    o = LiveSource(_toy_model(), K=K, train_nodes=False)(x, tabs, ["a", "b"], [0, 1])
    assert not o["batch"]["tok"].requires_grad and not o["batch"]["pos"].requires_grad, "train_nodes=False: no graph in the batch"


def selftest():
    torch.manual_seed(0)
    _t_targets()
    _t_tokens()
    _t_labels()
    _t_data()
    _t_compare()
    _t_guards()
    _t_source()
    print("lane_live_tok selftest ok")
