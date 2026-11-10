def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("tests/test_evaluation_api.py", [
    ('''    out = subprocess.run([sys.executable, "-m", "evalsuite", "methods"]''', '''    # the identity of a network: its checkpoint (name, size, fingerprint), config, code and the adapter; a retrained checkpoint of the same name is another identity
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as tmp:
        run = os.path.join(tmp, "fake_run")
        os.makedirs(run)
        open(os.path.join(run, "config.json"), "w").write("{}")
        ck = os.path.join(run, "ckpt_00000010.pt")
        open(ck, "wb").write(os.urandom(50_000))
        a = bev_eval.identity(run)
        assert a["checkpoint"] == "ckpt_00000010.pt" and a["size"] == 50_000 and a["adapter"] and a == bev_eval.identity(run), a
        data = bytearray(open(ck, "rb").read())
        data[10] ^= 1
        open(ck, "wb").write(bytes(data))
        assert bev_eval.identity(run)["fingerprint"] != a["fingerprint"], "a retrained checkpoint of the same name and size is another identity"
        open(os.path.join(run, "ckpt_00000020.pt"), "wb").write(bytes(data))
        assert bev_eval.identity(run)["checkpoint"] == "ckpt_00000020.pt", "the last checkpoint is the one evaluated"
        try:
            bev_eval.identity(tmp)
            raise AssertionError("identity of a folder without checkpoint")
        except FileNotFoundError as e:
            assert "no checkpoint" in str(e)
        assert bev_eval.resolve(os.path.relpath(run, ROOT)) == os.path.relpath(run, ROOT) and bev_eval.name(run) == "fake_run" and bev_eval.out(run).endswith(os.path.join("fake_run", "eval"))
    try:
        bev_eval.resolve("no_such_run_anywhere")
        raise AssertionError("an unknown run was resolved")
    except FileNotFoundError as e:
        assert "no run folder" in str(e)
    for bad in ("test", "small.test", "large.test"):                                                          # TEST is refused before anything is built
        try:
            bev_eval.source("runs/none", bad)
            raise AssertionError(f"{bad} was accepted")
        except ValueError as e:
            assert "TEST" in str(e), e
    try:
        bev_eval.task("runs/none", ("maps", "mapz"))
        raise AssertionError("an unknown metric was accepted")
    except ValueError as e:
        assert "unknown metrics" in str(e)
    out = subprocess.run([sys.executable, "-m", "evalsuite", "methods"]'''),
    ('''        assert r["n_frames"] == 1 and r["scenes_run"] == 1 and r["predictions_reused"] is False, r
        rs_dir = os.path.join(net, "eval", "results", r["results"].split("@")[1])
        assert len(os.listdir(os.path.join(rs_dir, "frames"))) == 1, "only the frame asked for is stored (the scene ran whole up to it)"''',
     '''        assert r["n_frames"] == 1 and r["frames_predicted"] == 1 and r["predictions_reused"] is False and r["warnings"] == [], r
        rs_dir = os.path.join(net, "eval", "results", r["results"].split("@")[1])
        assert len(os.listdir(os.path.join(rs_dir, "frames"))) == 1, "only the frame asked for is stored (the scene ran whole up to it)"'''),
    ('''        assert ev.check(net, frames, workers=0)["predictions_stored"] is True''', '''        chk = ev.check(net, frames, workers=0)
        assert chk["ok"] and chk["frames_stored"] == 1 and chk["network_runs"] is False and chk["id"] == r["id"], chk'''),
    ('''        assert {x["kind"] for x in ev.list(net)} == {"results", "evaluation"} and len(ev.list(net)) == 3''', '''        assert {x["kind"] for x in ev.list(net)} == {"predictions", "evaluation"} and len(ev.list(net)) == 3'''),
    ('''  3  `card` gives the contract card (split, subset, protocol carry, a provenance block), `check` and `list` name what is stored, `compare` against itself is refused or zero''',
     '''  3  `card` gives the contract card (split, subset, protocol carry, a provenance block), `check` (the dry run, with the id it would give) and `list` name what is stored
Without option it also tests the adapter's identity of a network (a retrained checkpoint of the same name is another identity), the run resolution, the refusal of TEST and of unknown metrics.'''),
])
print("ok")
