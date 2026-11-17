import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
rep('''def test_score_max():''', '''def test_every():
    """--every N: the sample, the file names, the meta of the dump, and score / sweep reading the subset back (frames 0, 3, 6 of VAL with --max 3; the TRAIN sample every third = 291 frames)"""
    import pickle
    import shutil
    s = E.sample(8414, "train")
    assert len(E.sample(8414, "train", 3)) == 291 and E.sample(8414, "train", 3).tolist() == s[::3].tolist() and E.sample(873, "val", 3).tolist()[:3] == [0, 3, 6] and E.sample(873, "val", 1).tolist() == list(range(873))
    if not CACHE_OK or not os.path.exists(SMOKE) or not os.path.exists(E.planb_path(0)):
        print("every: the sample ok, the dump / score / sweep part SKIPPED (no cache, smoke model or stored plain B)")
        return
    import qgraph_eval as QE
    ds = E.open_cache("val", 0)
    with tempfile.TemporaryDirectory() as tmp:
        runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
        out = os.path.join(scratch, "out")
        os.makedirs(os.path.join(runs, "h"))
        os.makedirs(out)
        ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
        ck["state"] = E.floor_model().state_dict()                                                       # an untrained head that keeps nodes and links
        torch.save(ck, os.path.join(runs, "h", "model.pt"))
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        with patched((E, "SCRATCH", scratch)):
            E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--every", "3", "--max", "3"]), runs)
            path = os.path.join(runs, "h", "token_val_every3_n3.pkl")
            d = pickle.load(open(path, "rb"))
            assert d["meta"]["every"] == 3 and d["meta"]["idx"] == [0, 3, 6] and d["meta"]["tokens"] == [ds.tokens[i] for i in (0, 3, 6)] and [f["token"] for f in d["frames"]] == d["meta"]["tokens"], d["meta"]
            must_fail(E.dump, E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--every", "3", "--max", "3"]), runs)       # never overwritten
            cards = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--every", "3", "--max", "3"]), runs, out)
            assert sorted(os.listdir(out)) == ["h_token_card_argmax_every3_n3.json", "h_token_card_rec_every3_n3.json"], os.listdir(out)
            fr = QE.val_frames()
            want = E.score_preds([E.frame_prediction(ds[i]["tokens_np"], f, "rec") for i, f in zip((0, 3, 6), d["frames"])], [fr[i] for i in (0, 3, 6)], log=lambda *a, **k: None)["graph"]
            c = cards["rec"]
            assert c["frames"] == 3 and all(abs(c["metrics"][k] - want[k]) < 1e-12 for k in E.KEYS) and "topo_F1" in c["info"]["paired_vs_plain_b"] and c["info"]["g1_topo_F1"] is None, "the card scores frames 0, 3, 6"
            tab = json.load(open(E.sweep(E.parse_args(["sweep", "--run", "h", "--seed", "0", "--every", "3", "--max", "3"]), runs, out)))
            assert os.path.exists(os.path.join(out, "h_token_sweep_every3_n3.json")) and tab["frames"] == 3 and len(tab["rows"]) == 24
            rec = next(r for r in tab["rows"] if r["record"])
            assert abs(rec["topo_F1"] - want["topo_F1"]) < 1e-12, "the record cell of the sweep is the record card on the same frames"
            must_fail(E.score, E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, out) if False else None
    print("every: the TRAIN sample every third = 291 frames; dump --every 3 --max 3 takes VAL frames 0, 3, 6 (file token_val_every3_n3.pkl, meta every / idx / tokens, never overwritten), score and sweep read it back (names ..._every3_n3), the card and the sweep's record cell equal a direct scoring of frames 0, 3, 6: ok")


def test_score_max():''')
rep("test_parity_timing, test_sweep, test_score_max]", "test_parity_timing, test_sweep, test_every, test_score_max]")
open(p, "w", encoding="utf-8").write(s)
