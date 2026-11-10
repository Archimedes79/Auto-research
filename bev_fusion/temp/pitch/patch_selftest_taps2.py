def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ("from .contracts import Metric, Prediction\n", "from .contracts import Metric, Prediction, TapSpec\n"),
    ('STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, fail_at=None)', 'STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, fail_at=None, memory="ok")\nMEMORY_TAP = "memory_conf"'),
    ('''    def predict(self, sample, taps=()):
        STATE.calls += 1
        if STATE.fail_at and STATE.calls >= STATE.fail_at:
            raise RuntimeError("killed")
        thr = self.threshold - 0.04 * self.seen
        self.seen += 1
        return Prediction(outputs=sample.inputs > thr, info={"predict_ms": 1.0, "loss": 1.0 if STATE.gt == "g1" else 5.0})''',
     '''    def taps(self):
        return [TapSpec("memory_conf", "map", "memory", "the fill of the memory"), TapSpec("feat", "map", "input", "the input picture")]

    def predict(self, sample, taps=()):
        STATE.calls += 1
        if STATE.fail_at and STATE.calls >= STATE.fail_at:
            raise RuntimeError("killed")
        thr = self.threshold - 0.04 * self.seen
        fill = {"ok": 0.1 * self.seen, "stuck": 0.3, "flat": 0.0}[STATE.memory]      # empty at the start of a scene and growing / full from the start / never filling
        self.seen += 1
        tapped = {"memory_conf": np.full((2, 2), fill), "feat": np.asarray(sample.inputs, np.float32)}
        return Prediction(outputs=sample.inputs > thr, taps={k: v for k, v in tapped.items() if k in taps}, info={"predict_ms": 1.0, "loss": 1.0 if STATE.gt == "g1" else 5.0})'''),
    ('''    assert ResultSet.create(os.path.join(tmp, rid), ident).has(refs[2]), "the stored set of the same identity is found again"''',
     '''    assert ResultSet.create(os.path.join(tmp, rid), ident).has(refs[2]), "the stored set of the same identity is found again"
    # taps are stored next to the frame; a frame without a tap does not have it, adding a tap keeps the outputs
    assert not again.has(refs[1], ("feat",)) and again.tap_names(refs[1]) == []
    rw = ResultSet.open(os.path.join(tmp, rid), "a")
    rw.add(refs[1], Prediction(outputs="other", taps={"feat": np.arange(4.0).reshape(2, 2), "a/b": np.ones(1)}, info={}))
    got = ResultSet.open(os.path.join(tmp, rid)).load(refs[1], ("feat", "a/b"))
    assert rw.has(refs[1], ("feat",)) and not rw.has(refs[1], ("nope",)) and np.array_equal(got.taps["feat"], np.arange(4.0).reshape(2, 2)) and got.outputs["dt"][0.0]["dets"] == [{"x": 1.5, "c": "car"}]
    assert rw.tap_names(refs[1]) == ["a_b", "feat"] and rw.manifest["gts"] == {"g1": 1, "g2": 2}, "outputs of a stored frame are kept, a second add counts nothing"
    _raises(KeyError, "with the taps", rw.load, refs[2], ("feat",))'''),
    ('''    # the dry run: what would happen, and the problems before anything is run''', '''    # intermediate results: asked for with the predictions; a frame stored without them is predicted again, one with them is reused; unknown taps are refused before the network runs
    net5, STATE.calls = "runs/thr0.6", 0
    plain = ev.predict(net5, "scenes:demo-0:val")
    assert plain["frames_predicted"] == 6
    with_tap = ev.predict(net5, "scenes:demo-0:val", taps=("feat",))
    assert with_tap["frames_predicted"] == 6 and with_tap["results"] == plain["results"], "stored without the tap: predicted again to get it"
    STATE.calls = 0
    assert ev.predict(net5, "scenes:demo-0:val", taps=("feat",))["frames_predicted"] == 0 and STATE.calls == 0 and ev.predict(net5, "scenes:demo-0:val")["frames_predicted"] == 0, "and reused"
    assert ev.tap(plain["results"], "demo-0/2")["taps"] == ["feat"]
    t = ev.tap(plain["results"], "demo-0/2", "feat")
    assert t["shape"] == [32, 32] and np.load(t["path"]).shape == (32, 32)
    _raises(EvaluationError, "has no tap", ev.tap, plain["results"], "demo-0/2", "memory_conf")
    _raises(EvaluationError, "unknown taps", ev.predict, net5, "val", taps=("nothing",))
    # the health of the memory (D70): healthy = no warning; full from the start, or never filling = a warning in the predictions' result
    assert plain["warnings"] == [] and ev.run(net5, "scenes:demo-0:val", progress=0)["warnings"] == []
    for mode, text in (("stuck", "not empty at the first frame"), ("flat", "does not fill")):
        STATE.memory = mode
        r = ev.run(f"runs/thr0.{4 if mode == 'stuck' else 2}", "scenes:demo-1:val", progress=0)
        assert len(r["warnings"]) == 1 and text in r["warnings"][0] and "demo-1" in r["warnings"][0], (mode, r["warnings"])
    STATE.memory = "ok"
    # the dry run: what would happen, and the problems before anything is run'''),
])
print("ok")
