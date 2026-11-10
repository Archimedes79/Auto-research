def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/runner_api.py", [
    # ---- the helpers go to a base class: only the commands are methods of Runner itself
    ('''class Runner:
    METHODS = ()                                              # filled below: the methods are the commands

    def __init__(self, project: Optional[str] = None, adapter: Any = None):
        self.ad = load_adapter(adapter, project)
        self._networks: Dict[str, str] = {}                   # the networks this object has seen: name -> path (an id carries only the name)
''', '''class _Helpers:
    """what the Runner and the Evaluation both need: the adapter, the names of networks, the identity of predictions, the stored sets. Not commands."""

    def __init__(self, project: Optional[str] = None, adapter: Any = None):
        self.ad = load_adapter(adapter, project)
        self._networks: Dict[str, str] = {}                   # the networks this object has seen: name -> path (an id carries only the name)

    def resolve(self, network: str) -> str:
        try:
            net = self.ad.resolve(network) if hasattr(self.ad, "resolve") else network
        except OSError as e:
            raise EvaluationError(str(e)) from None
        self._networks[self.ad.name(net)] = net
        return net

    def net_identity(self, net: str) -> Dict[str, Any]:
        try:
            return self.ad.identity(net)
        except OSError as e:
            raise EvaluationError(f"{net}: {e}") from None

    def identity(self, net_identity: Dict[str, Any], source: Source) -> Dict[str, Any]:
        """what the predictions depend on: the network, what it was fed (what a source says about the ground truth and the set is not part of it) and the memory mode"""
        inputs = self.ad.inputs(source) if hasattr(self.ad, "inputs") else {k: v for k, v in source.describe().items() if k not in RESULT_DROP}
        return {"network": net_identity, "inputs": inputs, "state": CARRY}

    def stored(self, net: str, rid: str) -> Optional[ResultSet]:
        path = os.path.join(self.ad.out(net), "results", rid)
        return ResultSet.open(path) if os.path.exists(os.path.join(path, "manifest.json")) else None

    @staticmethod
    def health(rs: Optional[ResultSet], scenes) -> List[str]:
        """the warnings recorded for these scenes (the memory's health)"""
        return [msg for sc, msg in rs.manifest.get("health", {}).items() if sc in scenes] if rs is not None else []

    def _release(self, source: Source) -> None:
        if hasattr(self.ad, "release"):
            self.ad.release(source)


class Runner(_Helpers):
    METHODS = ()                                              # filled below: the methods are the commands
'''),
    ('''    @staticmethod
    def health(rs: Optional[ResultSet], scenes) -> List[str]:
        """the warnings recorded for these scenes (the memory's health)"""
        return [msg for sc, msg in rs.manifest.get("health", {}).items() if sc in scenes] if rs is not None else []

    # ---------------------------------------------------------------------------------------- about the network itself''', '''    # ---------------------------------------------------------------------------------------- about the network itself'''),
    ('''    # ---------------------------------------------------------------------------------------- helpers (also used by the Evaluation)
    def resolve(self, network: str) -> str:
        try:
            net = self.ad.resolve(network) if hasattr(self.ad, "resolve") else network
        except OSError as e:
            raise EvaluationError(str(e)) from None
        self._networks[self.ad.name(net)] = net
        return net

    def net_identity(self, net: str) -> Dict[str, Any]:
        try:
            return self.ad.identity(net)
        except OSError as e:
            raise EvaluationError(f"{net}: {e}") from None

    def identity(self, net_identity: Dict[str, Any], source: Source) -> Dict[str, Any]:
        """what the predictions depend on: the network, what it was fed (what a source says about the ground truth and the set is not part of it) and the memory mode"""
        inputs = self.ad.inputs(source) if hasattr(self.ad, "inputs") else {k: v for k, v in source.describe().items() if k not in RESULT_DROP}
        return {"network": net_identity, "inputs": inputs, "state": CARRY}

    def stored(self, net: str, rid: str) -> Optional[ResultSet]:
        path = os.path.join(self.ad.out(net), "results", rid)
        return ResultSet.open(path) if os.path.exists(os.path.join(path, "manifest.json")) else None

    def _source(self, net: str, of: str, workers: int, control: str) -> Source:''', '''    # ---------------------------------------------------------------------------------------- the data
    def _source(self, net: str, of: str, workers: int, control: str) -> Source:'''),
    ('''        except (ValueError, KeyError, OSError) as e:
            raise EvaluationError(str(e.args[0] if isinstance(e, KeyError) and e.args else e)) from None

    def _release(self, source: Source) -> None:
        if hasattr(self.ad, "release"):
            self.ad.release(source)


Runner.METHODS = tuple(n for n, f in vars(Runner).items() if callable(f) and not n.startswith("_") and n not in ("health",))      # the methods are the commands: no second list''', '''        except (ValueError, KeyError, OSError) as e:
            raise EvaluationError(str(e.args[0] if isinstance(e, KeyError) and e.args else e)) from None


Runner.METHODS = tuple(n for n, f in vars(Runner).items() if callable(f) and not n.startswith("_"))      # the methods are the commands: no second list'''),
    # ---- _run_network without the dir() hack
    ('''    def _run_network(self, net, source, scenes, keep, rs, ident, taps, device, progress):
        subject = getattr(self.ad, "free_subject", self.ad.subject)(net, source, device)
        try:''', '''    def _run_network(self, net, source, scenes, keep, rs, ident, taps, device, progress):
        want, t0, n, ms = {r.label for r in keep}, time.time(), 0, []
        subject = getattr(self.ad, "free_subject", self.ad.subject)(net, source, device)
        try:'''),
    ('''            runner = FrameRunner(source, subject, CARRY)
            want, t0, n, ms = {r.label for r in keep}, time.time(), 0, []
            total = sum''', '''            runner = FrameRunner(source, subject, CARRY)
            total = sum'''),
    ('''            rs.close(wall_s=time.time() - t0 if "t0" in dir() else 0.0, cost={"median_ms": sorted(ms)[len(ms) // 2], "n": len(ms)} if "ms" in dir() and ms else None)
            subject.close()''', '''            rs.close(wall_s=time.time() - t0, cost={"median_ms": sorted(ms)[len(ms) // 2], "n": len(ms)} if ms else None)
            subject.close()'''),
])
print("ok")
