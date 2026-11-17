import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = s.index("def test_parity_timing():")
b = s.index("TESTS = [")
new = '''def test_parity_timing():
    if not CACHE_OK or not os.path.exists(SMOKE):
        print("parity_timing: SKIPPED (no cache or no smoke model)")
        return
    b_run = os.path.join(E.ROOT, "runs", "e292_a_s0")
    if not os.path.exists(os.path.join(b_run, "model.pt")):
        print("parity_timing: SKIPPED (no B' run)")
        return
    import glob
    with tempfile.TemporaryDirectory() as runs:
        os.makedirs(os.path.join(runs, "h"))
        torch.save(torch.load(SMOKE, map_location="cpu", weights_only=False), os.path.join(runs, "h", "model.pt"))
        rc = E.check_parity(E.parse_args(["check-parity", "--run", "h", "--seed", "0", "--n", "2", "--tol", "0.05"]), runs)
        (pf,) = glob.glob(os.path.join(runs, "h", "token_parity_*.json"))
        par = json.load(open(pf))
        assert rc in (0, 1) and par["frames"] == 2 and par["ok"] == (rc == 0) and "unmatched_share" in par and par["input_max_abs_diff_matched_valid_tokens"]["tok"] < 0.5, par
        K = E.open_cache("val", 0).K
        t_args = ["timing", "--run", "h", "--cpu", "--frames", "2", "--repeats", "1"]
        E.timing(E.parse_args(t_args + ["--b-run", os.path.relpath(b_run, E.ROOT)]), runs)
        E.timing(E.parse_args(t_args), runs)
        files = sorted(glob.glob(os.path.join(runs, "h", "token_timing_cpu_*.json")))
        assert len(files) == 2, files                                          # a second call never skips because a file exists
        ts = [json.load(open(f)) for f in files]
        t = next(x for x in ts if "whole" in x["median_ms"])
        assert all(x["K"] == K and x["quiet"] in (True, False) and x["usable_for_gcost"] is False for x in ts), ts            # K = the cache's K by default; a CPU result is never evidence for G-cost
        assert set(t["median_ms"]) == {"head", "bprime", "whole"} and t["params"] == JK.count_params(JK.JTokenHead()) and 0.05 < t["gmacs_head"] < 20 and t["device"] == "cpu" and t["params_whole_path"] == t["params"] + t["params_bprime"] < 2.74e6, t
        must_fail(E.timing, E.parse_args(t_args + ["--K", str(K + 1)]), runs)                                         # K above the cache's K
        print(f"parity_timing: parity on 2 VAL frames (rc {rc}: dp_none {par['max_dpnone']:.2e}, dp {par['max_dp']:.2e}, unmatched {par['unmatched_share']:.3f}, tok diff {par['input_max_abs_diff_matched_valid_tokens']['tok']:.1e}); timing code path on the CPU, "
              f"K {K}: head {t['median_ms']['head']:.0f} ms, whole {t['median_ms']['whole']:.0f} ms, {t['params']} parameters, {t['gmacs_head']:.2f} GMACs, two stamped files: ok")


def test_score_max():
    """dump then score of 3 frames, end to end on a smoke head: the controls of `ctl --max 3` (bsec0_close: the pickle for the paired interval) and planted hi / rule cards, g1 against the hand formula"""
    import pickle
    import shutil
    if not CACHE_OK or not os.path.exists(SMOKE) or not os.path.exists(E.planb_path(0)):
        print("score_max: SKIPPED (no cache, smoke model or stored plain B)")
        return
    K = E.open_cache("val", 0).K
    with tempfile.TemporaryDirectory() as tmp:
        runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
        out = os.path.join(scratch, "out")
        os.makedirs(os.path.join(runs, "h"))
        os.makedirs(out)
        torch.save(torch.load(SMOKE, map_location="cpu", weights_only=False), os.path.join(runs, "h", "model.pt"))
        shutil.copy(E.planb_path(0), os.path.join(scratch, "planb_s0.pkl"))
        with patched((E, "SCRATCH", scratch)):
            E.ctl(E.parse_args(["ctl", "--seed", "0", "--ident", "e292_a", "--arms", "bsec0_close", "--max", "3"]))
            assert os.path.exists(os.path.join(out, f"bsec0_close_e292_a_s0_K{K}_n3.json")) and os.path.exists(E.bsec0_close_path("e292_a", 0, 3))
            for arm, v in (("hi", 0.8), ("rule", 0.5)):                                                     # planted controls
                json.dump(dict(metrics=dict(topo_F1=v), info=dict(K=K)), open(os.path.join(out, E.ctl_name(arm, "e292_a", 0, K, 3)), "w"))
            E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
            cards = E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
            for dec, c in cards.items():
                assert c["frames"] == 3 and os.path.exists(os.path.join(out, f"h_token_card_{dec}_n3.json")) and not os.path.exists(os.path.join(runs, "h", f"token_card_{dec}.json"))
                want = (c["metrics"]["topo_F1"] - 0.5) / (0.8 - 0.5)                                         # g1 = (S1 - c-rule) / (c-hi - c-rule), by hand
                assert abs(c["info"]["g1_topo_F1"] - want) < 1e-12 and "exceed 1" in c["info"]["g1_note"], c["info"]["g1_topo_F1"]
                assert "topo_F1" in c["info"]["paired_vs_plain_b"] and "topo_F1" in c["info"]["paired_vs_bsec0_close"] and "plain_b_dump" in c["info"]
                assert c["protocol"]["reference"].startswith("val")
            for f in os.listdir(out):
                if f.startswith("h_token_card"):
                    os.remove(os.path.join(out, f))
            d = pickle.load(open(os.path.join(runs, "h", "token_val_n3.pkl"), "rb"))                       # a dump of another run is refused
            d["meta"]["run"] = "other"
            pickle.dump(d, open(os.path.join(runs, "h", "token_val_n3.pkl"), "wb"))
            try:
                E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
            except AssertionError as e:
                assert "not the dump of this read-out" in str(e), e
            else:
                raise RuntimeError("a dump of another run was scored")
            os.remove(os.path.join(scratch, "planb_s0.pkl"))                                                # without the layer-1 record score refuses
            must_fail(E.score, E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
    print("score_max: ctl --max 3 (bsec0_close pickle, K in the file name), dump --max 3, score --max 3 with planted controls: g1 equals the hand formula, the paired intervals vs plain B and vs bsec0_close, the cards in the out folder; a dump of another run and a missing plain B record are refused: ok")


'''
s = s[:a] + new + s[b:]
s = s.replace("test_numbers, test_parity_timing]", "test_numbers, test_parity_timing, test_score_max]")
s = s.replace("parity_timing (cache: the check-parity and the timing code paths on a tiny head, CPU).", "parity_timing (cache: the check-parity and the timing code paths on a tiny head, CPU: stamped files, K from the cache), score_max (cache: `ctl --max` / `dump --max` / `score --max` end to end, g1 by hand).")
open(p, "w", encoding="utf-8").write(s)
