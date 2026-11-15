p = "training/train/jwire_train.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
rep('''            # E325's default path: E312's run, E325's cache names, the default run name; the live stub stops the run after the VAL cache is written
            JD.live_records = lambda *a, **k: (_ for _ in ()).throw(JD._Reached(a, k))
            try:
                run_pipeline(0, None, heads=1, epochs=1, max_train=30, draws=1)
                raise AssertionError("the default run did not reach the live pass")
            except JD._Reached as e:
                assert e.args[1] == dict(pair_run="e312_p_s0"), e.args
            JD.live_records = live_stub
            assert seen["val"][-1] == (0, "e312_p_s0") and os.path.exists(cache("val_records_v1_s0.pkl")) and JD.load_records(cache("val_records_v1_s0.pkl"), 0), seen
            assert os.path.isdir(os.path.join(d, "runs", "e325_w_s0")), "the default run name is e325_w_s<seed>"
''', '''            # E325's default path: E312's run, E325's cache names, the default run name; a SECOND call with the same (default) name is refused and the first run's files stay byte for byte
            run_pipeline(0, None, heads=1, epochs=1, max_train=30, draws=1)
            assert seen["val"][-1] == (0, "e312_p_s0") and seen["live"][-1] == (0, "train", 30, "e312_p_s0"), seen
            assert os.path.exists(cache("val_records_v1_s0.pkl")) and JD.load_records(cache("val_records_v1_s0.pkl"), 0) and os.path.exists(cache("train_records_v1_s0_max30.pkl")), "E325's cache names"
            rd = os.path.join(d, "runs", "e325_w_s0")
            files = lambda: {f: hashlib.sha256(open(os.path.join(rd, f), "rb").read()).hexdigest() for f in sorted(os.listdir(rd))}  # noqa: E731
            first, n_calls = files(), (len(seen["val"]), len(seen["live"]))
            assert {"model.pt", "wire_agreement.json", "log.txt"} <= set(first), first
            for argv in (["run", "--seed", "0", "--heads", "1", "--epochs", "1", "--max-train", "30"], ["run", "--seed", "0", "--name", "e325_w_s0", "--heads", "1", "--epochs", "1", "--max-train", "30"]):
                assert "exists" in (refused(main, argv) or ""), argv
            assert "exists" in (refused(run_pipeline, 0, None, 1, 1, 3, 30, 1) or "") and files() == first and (len(seen["val"]), len(seen["live"])) == n_calls, "a second call rewrote the run or computed something"
            os.makedirs(os.path.join(d, "runs", "e312_p_s0"))                                                # a run folder of another kind (the pair run's model.pt) is not a place for a head either
            open(os.path.join(d, "runs", "e312_p_s0", "model.pt"), "wb").write(b"pair head")
            assert "exists" in (refused(run_pipeline, 0, "e312_p_s0", 1, 1, 3, 30, 1) or "") and open(os.path.join(d, "runs", "e312_p_s0", "model.pt"), "rb").read() == b"pair head"
            for bad in ("../x", "a/b", "", "a b", ".."):                                                       # a name is a folder name, with and without a pair run
                for kw in (dict(), dict(pair_run="e324_s_s0")):
                    assert "run folder name" in (refused(run_pipeline, 0, bad, 1, 1, 3, 30, 1, **kw) or ""), (bad, kw)
            assert not os.path.exists(os.path.join(d, "x")) and not os.path.exists(os.path.join(d, "runs", "x")), "a bad name created a folder"
''')
rep('''    import contextlib
    import io
    import json
    import pickle
    global ROOT''', '''    import contextlib
    import hashlib
    import io
    import json
    import pickle
    global ROOT''')
open(p, "w", encoding="utf-8").write(s)
