def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ("from .contracts import Metric, Prediction\n", "from .contracts import Metric, Prediction, TapSpec\n"),
    ('STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, fail_at=None)', 'STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, fail_at=None, memory="ok")\nMEMORY_TAP = "memory_conf"'),
    ('''        thr = self.threshold - 0.04 * self.seen
        self.seen += 1
        return Prediction(outputs=sample.inputs > thr, info={"predict_ms": 1.0, "loss": 1.0 if STATE.gt == "g1" else 5.0})''',
     '''        thr = self.threshold - 0.04 * self.seen
        fill = {"ok": 0.1 * self.seen, "stuck": 0.3, "flat": 0.0}[STATE.memory]            # the fill of the memory: empty at the start and growing, or full from the start, or never filling
        self.seen += 1
        taps = {}
        if "memory_conf" in taps_asked(sample, locals().get("taps_arg")):
            pass
        return Prediction(outputs=sample.inputs > thr, taps=self._taps(sample, fill, self._asked), info={"predict_ms": 1.0, "loss": 1.0 if STATE.gt == "g1" else 5.0})'''),
])
print("ok")
