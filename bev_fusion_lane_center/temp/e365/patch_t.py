import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
rep('''def test_score_max():''', '''def test_sweep():
    """the sweep on 3 cache frames with the untrained head (it keeps nodes and links): 24 cells, one is the record, the numbers move monotonically in the counts, the table is never overwritten"""
    import shutil
    if not CACHE_OK or not os.path.exists(SMOKE):
        print("sweep: SKIPPED (no cache or no smoke model)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        runs, out = os.path.join(tmp, "runs"), os.path.join(tmp, "out")
        os.makedirs(os.path.join(runs, "h"))
        os.makedirs(out)
        ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
        ck["state"] = E.floor_model().state_dict()
        torch.save(ck, os.path.join(runs, "h", "model.pt"))
        E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
        path = E.sweep(E.parse_args(["sweep", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
        rows = json.load(open(path))["rows"]
        assert len(rows) == 24 and sum(r["record"] for r in rows) == 1 and all(r["frames"] == 3 for r in rows), rows
        cell = lambda lt, kt, g: next(r for r in rows if (r["link_thr"], r["keep_thr"], r["none_gate"]) == (lt, kt, g))  # noqa: E731
        for kt in E.SWEEP_KEEP:
            for g in (True, False):
                lp = [cell(lt, kt, g)["links_per_frame"] for lt in E.SWEEP_LINK]
                assert all(x <= y for x, y in zip(lp, lp[1:])), "a lower link threshold only adds links"
            assert all(cell(lt, kt, True)["links_per_frame"] <= cell(lt, kt, False)["links_per_frame"] for lt in E.SWEEP_LINK), "no none gate only adds links"
        for lt in E.SWEEP_LINK:
            kn = [cell(lt, kt, True)["kept_nodes_per_frame"] for kt in E.SWEEP_KEEP]
            assert all(x <= y for x, y in zip(kn, kn[1:])), "a higher keep threshold only adds nodes"
        assert len({r["topo_F1"] for r in rows}) > 1, "the variants differ in the score"
        must_fail(E.sweep, E.parse_args(["sweep", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
        print(f"sweep: 24 cells on 3 frames of an untrained head (links per frame {min(r['links_per_frame'] for r in rows):.0f} - {max(r['links_per_frame'] for r in rows):.0f}), counts monotone in both thresholds and the gate, the record cell marked, the table never overwritten: ok")


def test_score_max():''')
rep('''            for f in os.listdir(out):
                if f.startswith("h_token_card"):''', '''            v = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3", "--link-thr", "0.1", "--keep-thr", "0.6", "--no-none-gate"]), runs, out)
            assert list(v) == ["rec"] and os.path.exists(os.path.join(out, "h_token_card_rec_lt0.1_kt0.6_nng_n3.json")), (list(v), os.listdir(out))
            dv = v["rec"]["info"]["decode_variant"]
            assert (dv["link_thr"], dv["keep_thr"], dv["none_gate"]) == (0.1, 0.6, False) and "g1_topo_F1" in v["rec"]["info"], dv
            for f in os.listdir(out):
                if f.startswith("h_token_card"):''')
rep("TESTS = [test_keep_rule, test_decode_fork,", "TESTS = [test_keep_rule, test_decode_fork, test_decode_variant,")
rep("test_parity_timing, test_score_max]", "test_parity_timing, test_sweep, test_score_max]")
rep("keep_rule (a node is kept", "decode_variant (the defaults = the record decode, thresholds and the none gate by hand, monotone), sweep (24 cells, counts monotone), keep_rule (a node is kept")
open(p, "w", encoding="utf-8").write(s)
