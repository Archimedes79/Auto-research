def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ("has seen since the scene started), an IoU metric and a metric that reads a number the network computes itself (a loss, which depends on the ground truth of the moment). No data, no GPU,",
     "has seen since the scene started) and an IoU metric; its Runner side (free_source / free_subject) reads no ground truth. No data, no GPU,"),
    ('''    """a network with a memory: the threshold falls by 0.04 for every frame seen since reset(); it also computes a 'loss' of its own, which depends on the ground truth of the moment"""''',
     '''    """a network with a memory: the threshold falls by 0.04 for every frame seen since reset()"""'''),
    ('''info={"predict_ms": 1.0, "loss": 1.0 if STATE.gt == "g1" else 5.0})''', '''info={"predict_ms": 1.0})'''),
    ('''class Loss(Metric):
    name, headline = "loss", ("total",)

    def frame(self, sample, pred):
        return {"v": np.array([float(pred.info["loss"]), 1.0])}

    def reduce(self, counts, keys=None):
        s = sum((c["v"] for c in counts), np.zeros(2))
        return {"total": float(s[0] / max(s[1], 1.0))}

    def direction(self, key):
        return "low"


class Task(minimal.MinTask):
    def metrics(self):
        return [minimal.IoU(), Loss()]


''', '''class FreeSource(minimal.MinSource):
    """the data the Runner reads: no ground truth (reference None), its own load counter; it describes what the network is fed like the labelled source does"""
    def __init__(self, of):
        self.of = of

    def scenes(self):
        return ["demo-0"] if self.of == "small" else super().scenes()

    def load(self, ref):
        STATE.free_loads += 1
        s = super().load(ref)
        return Sample(s.ref, inputs=s.inputs, reference=None, meta={"token": ref.key})

    def describe(self):
        return {"class": "FreeSource", "dataset": "toy", "split": self.of, "scenes": len(self.scenes())}


'''),
    ("from .contracts import FrameRef, Metric, Prediction, TapSpec", "from .contracts import FrameRef, Prediction, Sample, TapSpec"),
    ('STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, fail_at=None, memory="ok")', 'STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, free_loads=0, fail_at=None, memory="ok")'),
    ('''def source(net, of, workers=0, control=""): return Source(of)''', '''def source(net, of, workers=0, control=""): return Source(of)                       # with ground truth: what the Evaluation scores with
def free_source(net, of, workers=0, control=""): return FreeSource(of)               # without: what the Runner runs on'''),
    ('''def subject(net, src, device): return Subject(float(net.split("thr")[-1]))''', '''def subject(net, src, device): return Subject(float(net.split("thr")[-1]))
free_subject = subject                                                               # (the toy network never reads a ground truth)'''),
    ('''    return Task()''', '''    return minimal.MinTask()'''),
    ('''    suite = EvalSuite(src, subj, Task(), out=tempfile.mkdtemp(), state="carry")''', '''    suite = EvalSuite(src, subj, minimal.MinTask(), out=tempfile.mkdtemp(), state="carry")'''),
    ('''    STATE.root, STATE.gt, STATE.shift, STATE.calls, STATE.loads, STATE.fail_at = tempfile.mkdtemp(), "g1", 0, 0, 0, None
    ev = Evaluation(adapter=ME)''', '''    STATE.root, STATE.gt, STATE.shift, STATE.calls, STATE.loads, STATE.free_loads, STATE.fail_at = tempfile.mkdtemp(), "g1", 0, 0, 0, 0, None
    ev = Evaluation(adapter=ME)'''),
    ('''    _same(r.quant()["numbers"], _engine(net, fr), spec)
        assert r["n_frames"] == len(fr.resolve(Source("val"))[1]) and r["warnings"] == []''', '''    _same(r.quant()["numbers"], _engine(net, fr), spec)
        assert r["n_frames"] == len(fr.resolve(Source("val"))[1]) and r["warnings"] == []
    assert STATE.free_loads > 0 and STATE.loads == 0, "the Runner reads the frames through free_source; the labelled Source.load is never called (no ground truth, no inputs of the labelled data)"'''),
    ('''    # a new ground truth: another evaluation id, scored from the stored predictions, no network and no input loaded; the loss the network computed itself is flagged''', '''    # a new ground truth: another evaluation id, scored from the stored predictions, no network and no input loaded'''),
    ('''    assert len(r3["warnings"]) == 1 and "ground truth" in r3["warnings"][0] and r3.quant()["numbers"]["loss.total"] == allr.quant()["numbers"]["loss.total"] == 1.0, "the stale loss is said"''', '''    assert r3["warnings"] == [] and "loss.total" not in r3.quant()["numbers"], "the predictions carry no loss: nothing computed against an earlier ground truth"'''),
    ("    assert ev.tap(plain[\"results\"], \"demo-0/2\")[\"taps\"] == [\"feat\"]\n    t = ev.tap(plain[\"results\"], \"demo-0/2\", \"feat\")", "    assert ev.runner.tap(plain[\"results\"], \"demo-0/2\")[\"taps\"] == [\"feat\"]\n    t = ev.runner.tap(plain[\"results\"], \"demo-0/2\", \"feat\")"),
    ('''    _raises(EvaluationError, "has no tap", ev.tap, plain["results"], "demo-0/2", "memory_conf")''', '''    _raises(EvaluationError, "has no tap", ev.runner.tap, plain["results"], "demo-0/2", "memory_conf")'''),
])
print("ok")
