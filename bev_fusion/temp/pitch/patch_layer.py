def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    ('''`Evaluation.run` is `Runner.run` + `score`. The class that executes a network is the Runner (runner_api.py, D72: it reads no ground truth); the Evaluation calls it (`Evaluation(runner=...)`
injects another). Everything else here reads what was stored (only `draw_frame`, which draws a frame, runs the network on it again).''', '''Layers: command line / caller -> Evaluation -> Runner. `Evaluation.run` is `Runner.run` + `score`. The class that executes a network is the Runner (runner_api.py, D72: it reads no ground
truth); the Evaluation is the one front door and calls it (`Evaluation(runner=...)` injects another); `tap`, `time` and `probe` are the Runner's, passed through. Everything else here reads what
was stored (only `draw_frame`, which draws a frame, runs the network on it again). The Runner has its own command line (runsuite) for predictions of frames that have no labels.'''),
    ('''    def selftest(self) -> Dict[str, Any]:
        """the checks of this interface on a toy project (a few seconds; no data, no network)"""''', '''    # ---------------------------------------------------------------------------------------- what executes the network belongs to the Runner: passed through
    def tap(self, results: str, frame: str, name: str = "") -> Dict[str, Any]:
        """the intermediate results stored with a frame's predictions: without `name` which taps there are, with it the path of its .npy file, its shape and dtype (the Runner's)"""
        return self.runner.tap(results, frame, name)

    def time(self, network: str, n: int = 30) -> Dict[str, Any]:
        """milliseconds per module and per frame at batch 1 (idle GPU only: the adapter refuses a busy one; the Runner's)"""
        return self.runner.time(network, n)

    def probe(self, network: str, frames: int = 3, device: str = "cpu") -> Dict[str, Any]:
        """the module probe: statistics and dead channels of every module, the response when the camera or the radar is removed (frames = how many frames of the first scene; the Runner's)"""
        return self.runner.probe(network, frames, device)

    def selftest(self) -> Dict[str, Any]:
        """the checks of this interface on a toy project (a few seconds; no data, no network)"""'''),
])
print("ok")
