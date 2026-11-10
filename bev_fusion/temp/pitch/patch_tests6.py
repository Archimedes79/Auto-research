def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ("Runner.run + score = Evaluation.run;", "Runner.predict + score = Evaluation.run;"),
    ("    # Runner.run + score = Evaluation.run\n    pr = ev.runner.run(net, \"val\")", "    # Runner.predict + score = Evaluation.run\n    pr = ev.runner.predict(net, \"val\")"),
    ('''STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, free_loads=0, fail_at=None, memory="ok", bump=0, change=False)''', '''STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, free_loads=0, fail_at=None, memory="ok", bump=0, change=False, ev="x")'''),
    ('''"scenes": len(self.scenes()), "gt_signature": STATE.gt, "gt_version": None, "evaluator_hash": "x"}''', '''"scenes": len(self.scenes()), "gt_signature": STATE.gt, "gt_version": None, "evaluator_hash": STATE.ev}'''),
    ('''class FreeSource(minimal.MinSource):
    """the data the Runner reads: no ground truth (reference None), its own load counter; it describes what the network is fed like the labelled source does"""
    def __init__(self, of):
        self.of = of
''', '''class FreeSource(minimal.MinSource):
    """the data the Runner reads: no ground truth (reference None), its own load counter; it describes what the network is fed like the labelled source does"""
    def __init__(self, of, control=""):
        self.of, self.control = of, control
'''),
    ('''        return {"class": "FreeSource", "dataset": "toy", "split": self.of, "scenes": len(self.scenes())}''', '''        return {"class": "FreeSource", "dataset": "toy", "split": self.of, "scenes": len(self.scenes()), "control": self.control or None}'''),
    ('''def free_source(net, of, workers=0, control=""): return FreeSource(of)               # without: what the Runner runs on''', '''def free_source(net, of, workers=0, control=""): return FreeSource(of, control)       # without: what the Runner runs on'''),
    ("ev.runner.run, \"runs/thr0.8\"", "ev.runner.predict, \"runs/thr0.8\""),
    ('''    plain = ev.runner.run(net5, "scenes:demo-0:val")''', '''    plain = ev.runner.predict(net5, "scenes:demo-0:val")'''),
    ('''    with_tap = ev.runner.run(net5, "scenes:demo-0:val", taps=("feat",))''', '''    with_tap = ev.runner.predict(net5, "scenes:demo-0:val", taps=("feat",))'''),
    ('''    assert ev.runner.run(net5, "scenes:demo-0:val", taps=("feat",))["frames_predicted"] == 0 and STATE.calls == 0 and ev.runner.run(net5, "scenes:demo-0:val")["frames_predicted"] == 0, "and reused"''', '''    assert ev.runner.predict(net5, "scenes:demo-0:val", taps=("feat",))["frames_predicted"] == 0 and STATE.calls == 0 and ev.runner.predict(net5, "scenes:demo-0:val")["frames_predicted"] == 0, "and reused"'''),
    ('''    _raises(EvaluationError, "unknown taps", ev.runner.run, net5, "val", taps=("nothing",))
    assert ev.runner.run(net5, "scenes:demo-0:val", taps="feat")["frames_predicted"] == 0 and ev.runner.check(net5, "scenes:demo-0:val", taps="feat")["frames_stored"] == 6, "a bare string is one tap, not its letters"''', '''    _raises(EvaluationError, "unknown taps", ev.runner.predict, net5, "val", taps=("nothing",))
    assert ev.runner.predict(net5, "scenes:demo-0:val", taps="feat")["frames_predicted"] == 0 and ev.runner.plan(net5, "scenes:demo-0:val", taps="feat")["frames_stored"] == 6, "a bare string is one tap, not its letters"
    assert ev.check(net5, "scenes:demo-0:val", taps=("feat",))["frames_stored"] == 6 and ev.check(net5, "scenes:demo-0:val", taps=("memory_conf",))["frames_stored"] == 0, "check asks for taps like run"'''),
    # ---- what the ids depend on, the set clash, the evaluator warning, the fallback adapter, the score-side lock
    ('''    # the adapter: free_source and free_subject go together''', '''    # the identity of the predictions: a checkpoint that changed, or another control, is another ResultSet (never the stored frames of the old one)
    base = ev.runner.predict(net, "scenes:demo-0:val")["results"]
    STATE.bump = 1
    try:
        moved = ev.runner.predict(net, "scenes:demo-0:val")
        assert moved["results"] != base and moved["frames_predicted"] == 6, "another checkpoint: predicted again under another id"
    finally:
        STATE.bump = 0
    ctl = ev.runner.predict(net, "scenes:demo-0:val", control="camera_off")
    assert ctl["results"] not in (base, moved["results"]) and ev.check(net, "scenes:demo-0:val", control="camera_off")["results"] == ctl["results"], "another control: another set"
    # the same frames asked through another set: another evaluation id (never a clash in one store), a clear message otherwise
    one, two = ev.run(net, "scenes:demo-0:val", progress=0), ev.run(net, "scenes:demo-0:small", progress=0)
    assert one["id"] != two["id"] and one["results"] == two["results"], "the set is part of the question; the predictions are shared"
    # the evaluator code of two evaluations differs: compare says so
    STATE.ev = "y"
    try:
        low_y = ev.run(net, "val", progress=0)
    finally:
        STATE.ev = "x"
    assert low_y["id"] != allr["id"] and low_y["results"] == allr["results"], "another evaluator hash: scored again from the same predictions"
    assert any("different evaluator code" in w for w in ev.compare(allr["id"], low_y["id"]).get("warnings", [])), "compare names a difference of the evaluator"
    # the evaluation of an id that is being written is refused with the holder's name
    if results_mod.fcntl is not None:
        eid = ev.check(net, "val", metrics=("maps",))["id"].partition("@")[2]
        with writer_lock(os.path.join(out(net), "suite", "quant", eid)):
            _raises(EvaluationError, "being written by pid", ev.score, allr["results"], "val", None, ("maps",))
        assert ev.score(allr["results"], "val", None, ("maps",))["id"].endswith(eid)
    # an adapter without the label-free pair: the Runner reads the labelled source and subject (more data than it needs, the same numbers)
    plain_ad = SimpleNamespace(**{f: getattr(ME, f) for f in ADAPTER_FUNCTIONS}, reference=reference, card=card)
    fb = Evaluation(adapter=plain_ad).run("runs/thr0.45", "val", progress=0)
    _same(Evaluation(adapter=plain_ad).show_numbers(fb["id"])["numbers"], _engine("runs/thr0.45", Frames.parse("val")), "the fallback path")
    # the adapter: free_source and free_subject go together'''),
    # ---- the guard of a card / a picture fails closed when the predictions are gone
    ('''    pic = ev.draw_frame(allr["id"], "demo-1/4")
    assert os.path.isfile(pic["images"]["overview"]) and "iou.iou" in pic["scores"]''', '''    pic = ev.draw_frame(allr["id"], "demo-1/4")
    assert os.path.isfile(pic["images"]["overview"]) and "iou.iou" in pic["scores"]
    gone = os.path.join(out(net), "results", allr["results"].partition("@")[2], "manifest.json")
    os.rename(gone, gone + ".away")                                                     # the predictions are gone (package.py clean keeps the manifest, a person may not): a card or a picture cannot be certified
    try:
        _raises(EvaluationError, "are gone", ev.make_card, allr["id"])
        _raises(EvaluationError, "are gone", ev.draw_frame, allr["id"], "demo-1/4")
    finally:
        os.rename(gone + ".away", gone)'''),
])
patch("../bev_fusion/tests/test_evaluation_api.py", [
    ('''    finally:
        bev_eval._CODE.update(saved)''', '''        bev_eval._CODE["path"] = "/some/other/snapshot"                           # every way into the network's code claims it: a task, a timing and a probe too
        for call in (lambda: bev_eval.task("runs/t16_full_s0", ("maps", "boxes")), lambda: bev_eval.time("runs/t16_full_s0"), lambda: bev_eval.probe("runs/t16_full_s0")):
            try:
                call()
                raise AssertionError("a second code snapshot was accepted by a task / timing / probe")
            except EvaluationError as e:
                assert "one network per process" in str(e), e
    finally:
        bev_eval._CODE.update(saved)'''),
    ('''    assert hasattr(bev_eval, "free_source") and hasattr(bev_eval, "free_subject"), "the Runner's pair: both or none"''', '''    assert hasattr(bev_eval, "free_source") and hasattr(bev_eval, "free_subject"), "the Runner's pair: both or none"
    assert len(bev_eval.evaluator_extra()) == 12, "the adapter's own code is part of the evaluation id"'''),
])
print("ok")
