def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    ('''`run` is `predict` + `score`. The class that executes a network is the Runner (runner_api.py, D72: it reads no ground truth); the Evaluation calls it (`Evaluation(runner=...)` injects another),
`predict` is its `run`. Everything else here reads what was stored''', '''`Evaluation.run` is `Runner.run` + `score`. The class that executes a network is the Runner (runner_api.py, D72: it reads no ground truth); the Evaluation calls it (`Evaluation(runner=...)`
injects another). Everything else here reads what was stored'''),
    ('''    def predict(self, network: str, frames: str = "val", taps: Sequence[str] = (), device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """the Runner's `run`: predict what is not stored yet (no ground truth is read) and store it, with the intermediate results `taps` when asked"""
        return self.runner.run(network, frames, taps, device, workers, control, progress)

''', ''),
    ('''`predict` them first, or use `run`")''', '''run them first (`Runner.run`), or use `Evaluation.run`")'''),
    ("    # ---------------------------------------------------------------------------------------- predicting and scoring\n", "    # ---------------------------------------------------------------------------------------- scoring (the network is run by the Runner)\n"),
])
patch("evalsuite/selftest_api.py", [
    ("predict + score = run; `check` finds the problems", "Runner.run + score = Evaluation.run; `check` finds the problems"),
    ("    # predict + score = run\n    pr = ev.predict(net, \"val\")", "    # Runner.run + score = Evaluation.run\n    pr = ev.runner.run(net, \"val\")"),
    ("    plain = ev.predict(net5, \"scenes:demo-0:val\")", "    plain = ev.runner.run(net5, \"scenes:demo-0:val\")"),
    ("    with_tap = ev.predict(net5, \"scenes:demo-0:val\", taps=(\"feat\",))", "    with_tap = ev.runner.run(net5, \"scenes:demo-0:val\", taps=(\"feat\",))"),
    ("    assert ev.predict(net5, \"scenes:demo-0:val\", taps=(\"feat\",))[\"frames_predicted\"] == 0 and STATE.calls == 0 and ev.predict(net5, \"scenes:demo-0:val\")[\"frames_predicted\"] == 0, \"and reused\"",
     "    assert ev.runner.run(net5, \"scenes:demo-0:val\", taps=(\"feat\",))[\"frames_predicted\"] == 0 and STATE.calls == 0 and ev.runner.run(net5, \"scenes:demo-0:val\")[\"frames_predicted\"] == 0, \"and reused\""),
    ("    _raises(EvaluationError, \"unknown taps\", ev.predict, net5, \"val\", taps=(\"nothing\",))", "    _raises(EvaluationError, \"unknown taps\", ev.runner.run, net5, \"val\", taps=(\"nothing\",))"),
])
print("ok")
