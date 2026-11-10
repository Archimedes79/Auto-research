def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    # frames grammar: the seed is optional
    ('''    for text in ("val", "small.val", "spread:5:val", "random:2:7:small.val", "scenes:demo-0,demo-1:val", "frames:demo-0/1,demo-2/5:val"):
        assert str(Frames.parse(text)) == text, text''', '''    for text in ("val", "small.val", "spread:5:val", "random:2:7:small.val", "scenes:demo-0,demo-1:val", "frames:demo-0/1,demo-2/5:val"):
        assert str(Frames.parse(text)) == text, text
    assert Frames.parse("random:3:small.val") == Frames.random(3, 0, "small.val") and Frames.parse("random:3") == Frames.random(3, 0) and Frames.parse("random:3:7") == Frames.random(3, 7)
    assert Frames.parse("random:3:7:small.val").args == (3, 7) and str(Frames.parse("random:3")) == "random:3:0:val", "the seed is optional and shown when it is not"
    assert Frames.parse("spread:9:small.val").of == "small.val" and Frames.parse("scenes:a,b").args == (("a", "b"),)'''),
    ('''    for bad in ("spread:x", "random:3", "scenes", "frames:a:b:c:d", "spread:1:2:3"):''', '''    for bad in ("spread:x", "random", "random:x", "scenes", "frames:a:b:c:d", "spread:1:2:3", "random:3:7:small.val:x"):'''),
    # the result set: key check, no manifest counters, format, lock with a token
    ('''    rs.add(refs[0], pred, gt="g1", device="cpu")
    back = rs.load(refs[0])''', '''    rs.add(refs[0], pred)
    back = rs.load(refs[0])'''),
    ('''    for r in refs[1:]:
        rs.add(r, pred, gt="g2", device="cpu")
    rs.close(wall_s=1.0, cost={"median_ms": 2.5})
    again = ResultSet.open(os.path.join(tmp, rid))
    assert [r.label for r in again.refs()] == [r.label for r in refs] and again.manifest["gts"] == {"g1": 1, "g2": 2} and again.manifest["devices"] == {"cpu": 3}''',
     '''    for r in refs[1:]:
        rs.add(r, pred)
    rs.close(wall_s=1.0, cost={"median_ms": 2.5})
    again = ResultSet.open(os.path.join(tmp, rid))
    assert [r.label for r in again.refs()] == [r.label for r in refs] and again.manifest["format"] == 1 and again.n_frames() == 3'''),
    ('''    assert rw.tap_names(refs[1]) == ["a_b", "feat", "t"] and rw.manifest["gts"] == {"g1": 1, "g2": 2}, "outputs of a stored frame are kept, a second add counts nothing"''',
     '''    assert rw.tap_names(refs[1]) == ["a_b", "feat", "t"], "outputs of a stored frame are kept, a second add only adds taps"
    # the dataset key is stored with the frame: another key at the same index (a rebuilt data set) is an error, not a wrong score
    keyed = [FrameRef("demo-9", i, f"tok{i}") for i in range(2)]
    rk = ResultSet.create(os.path.join(tmp, rid), ident)
    rk.add(keyed[0], pred)
    assert rk.load(FrameRef("demo-9", 0, "tok0")).info == {"predict_ms": 2.5} and rk.load(FrameRef("demo-9", 0)).info, "the same key, or none, reads"
    _raises(IOError, "the dataset changed", rk.load, FrameRef("demo-9", 0, "other"))
    old = json.load(open(os.path.join(tmp, rid, "manifest.json")))
    old["format"] = 0
    json.dump(old, open(os.path.join(tmp, rid + "_old.json"), "w"))
    os.makedirs(os.path.join(tmp, "oldset"))
    json.dump(old, open(os.path.join(tmp, "oldset", "manifest.json"), "w"))
    _raises(ValueError, "result set format 0", ResultSet.open, os.path.join(tmp, "oldset"))'''),
    ('''    with open(target + ".lock", "w") as f:
        f.write(f"4194301 {socket.gethostname()}")
    with writer_lock(target):
        pass
    with open(target + ".lock", "w") as f:
        f.write("4194301 another-host")
    _raises(Locked, "another-host", writer_lock(target).__enter__)
    os.remove(target + ".lock")''', '''    with open(target + ".lock", "w") as f:
        f.write(f"4194301 {socket.gethostname()} deadbeef")
    with writer_lock(target):
        pass
    assert not os.path.exists(target + ".lock")
    with open(target + ".lock", "w") as f:
        f.write("4194301 another-host deadbeef")
    _raises(Locked, "another-host", writer_lock(target).__enter__)
    os.remove(target + ".lock")
    with open(target + ".lock", "w") as f:                                      # an empty lock: a writer that is just creating it is waited for, a damaged one (old) is taken over
        pass
    os.utime(target + ".lock", (1, 1))
    with writer_lock(target):
        pass
    with writer_lock(target):                                                   # a holder removes only a lock that carries its own token
        with open(target + ".lock", "w") as f:
            f.write(f"{os.getpid()} {socket.gethostname()} someone-else")
    assert os.path.exists(target + ".lock") and "someone-else" in open(target + ".lock").read()
    os.remove(target + ".lock")'''),
    ('''from .contracts import Metric, Prediction, TapSpec''', '''from .contracts import FrameRef, Metric, Prediction, TapSpec'''),
    # the cost now lists the devices of the frames
    ('''assert allr.cost()["predict"]["n"] == 18 and allr.cost()["devices"] is not None''', '''assert allr.cost()["predict"]["n"] == 18 and allr.cost()["devices"] == ["cuda"]'''),
    # unknown set, frames_to_run
    ('''    chk = ev.check(net, "frames:demo-1/4")
    assert chk["ok"] and chk["scenes_run"] == 1 and chk["frames_kept"] == 1 and chk["frames_stored"] == 1 and chk["network_runs"] is False and chk["id"].startswith("thr0.5@e")''',
     '''    chk = ev.check(net, "frames:demo-1/4")
    assert chk["ok"] and chk["scenes_run"] == 1 and chk["frames_kept"] == 1 and chk["frames_stored"] == 1 and chk["network_runs"] is False and chk["id"].startswith("thr0.5@e")
    assert chk["frames_to_run"] == 0 and ev.check("runs/thr0.9", "frames:demo-1/4")["frames_to_run"] == 5, "frames 0..4 of the scene run (the memory), nothing after the kept one"
    assert ev.check("runs/thr0.9", "val")["frames_to_run"] == 18
    nosuch = ev.check(net, "nosuch")
    assert not nosuch["ok"] and "not one of" in nosuch["problems"][0] and "TEST" in ev.check(net, "test")["problems"][0]
    _raises(EvaluationError, "not one of", ev.run, net, "nosuch")'''),
])
print("ok")
