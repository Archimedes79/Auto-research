# ------------------------------------------------------------------------------------------------ the class
class _Helpers:
    """what the Runner (runner_api.py) and the Evaluation both need: the adapter, the names of networks, the identity of predictions, the stored sets. Not commands."""

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


class Evaluation(_Helpers):
    def __init__(self, project: Optional[str] = None, adapter: Any = None, runner: Any = None):
        super().__init__(project, adapter)
        if runner is None:
            from .runner_api import Runner
            runner = Runner(adapter=self.ad)                  # the class that executes the network (D72): any object with its run / check / list can be injected
        self.runner = runner
        self._tasks: Dict[Tuple[str, Tuple[str, ...]], Any] = {}

    # ---------------------------------------------------------------------------------------- what exists, and the dry run
    def sets(self) -> Dict[str, Any]:
        """the sets of scenes the project defines: name -> number of scenes (`val` and `train` are the network's own lists: no number)"""
        return dict(self.ad.sets())

    def list(self, network: str) -> List[Dict[str, Any]]:
        """the stored predictions and evaluations of a network, with their size"""
        net = self.resolve(network)
        rows, name, base = list(self.runner.list(net)), self.ad.name(net), os.path.join(self.ad.out(net), "suite", "quant")
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            try:
                with open(os.path.join(base, d, "manifest.json"), encoding="utf-8") as f:
                    ev = json.load(f).get("evaluation")
            except (OSError, ValueError):
                continue
            if ev:
                rows.append({"id": f"{name}@{d}", "kind": "evaluation", "frames": ev["n_frames"], "set": ev["frames"], "gt": ev["gt"], "metrics": ev["metrics"]})
        return rows

    def check(self, network: str, frames: str = "val", metrics: Sequence[str] = ("maps", "boxes"), workers: int = 0) -> Dict[str, Any]:
        """the dry run: what `run` would do with these frames (the Runner's check) and what is wrong with the ground truth and the metrics (`ok`, `problems`). No frame runs; the data
        and the checkpoint are opened (a BEV network is built on the CPU)"""
        res = dict(self.runner.check(network, frames, workers=workers))
        problems: List[str] = res["problems"]
        if "results" not in res:                                              # the Runner could not even plan: its problems are the answer
            return res
        try:
            net, fr = self.resolve(network), Frames.parse(frames)
            self._task(net, tuple(metrics))
            source = self._source(net, fr.of, workers, "")
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
            res["ok"] = False
            return res
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            res["gt"] = desc.get("gt_version") or desc.get("gt_signature")
            res["id"] = f"{self.ad.name(net)}@{self._eval_id(res['results'].partition('@')[2], keep, desc, metrics)}"
            if not desc.get("gt_signature"):
                problems.append("the data has no ground-truth signature: the numbers could not be told from another GT's")
            self._probe_reference(source, keep, problems)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
        finally:
            self._release(source)
        res["ok"] = not problems
        return res

    # ---------------------------------------------------------------------------------------- predicting and scoring
    def predict(self, network: str, frames: str = "val", taps: Sequence[str] = (), device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """the Runner's `run`: predict what is not stored yet (no ground truth is read) and store it, with the intermediate results `taps` when asked"""
        return self.runner.run(network, frames, taps, device, workers, control, progress)

    def score(self, results: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Sequence[str] = ("maps", "boxes"), workers: int = 0, progress: int = 50) -> "Result":
        """score stored predictions (an id `<network>@r<hash>`) on the frames against the ground truth; the network is not run; frames that were not predicted are an error"""
        name, _, rid = str(results).partition("@")
        net, fr = self.resolve(self._networks.get(name) or self.ad.network(name)), Frames.parse(frames)
        rs = self.stored(net, rid)
        if rs is None:
            raise EvaluationError(f"no predictions {results}: nothing at {os.path.join(self.ad.out(net), 'results', rid)}")
        source = self._source(net, fr.of, workers, rs.manifest["identity"]["inputs"].get("control") or "")
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)
            missing = sum(not rs.has(r) for r in keep)
            if missing:
                raise EvaluationError(f"{missing} of the {len(keep)} frames are not in {results}: `predict` them first, or use `run`")
            return self._score(net, source, desc, rs, keep, fr, metrics, 0.0, 0, True, progress)
        finally:
            self._release(source)

    def run(self, network: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Sequence[str] = ("maps", "boxes"), taps: Sequence[str] = (), device: str = "cuda",
            workers: int = 2, control: str = "", progress: int = 50) -> "Result":
        """predict (what is not stored yet, through the Runner), then score against the ground truth; gt_key = the GT version the numbers must be on (refused when the data has another);
        metrics = what the project offers (BEV: maps, boxes, curves, fusion, velocity). -> a Result: a dict with the id, whose methods read the evaluation"""
        net, fr, metrics = self.resolve(network), Frames.parse(frames), tuple(metrics)
        self._task(net, metrics)                                          # a typo in the metrics fails here, not after the network ran
        source = self._source(net, fr.of, workers, control)
        try:
            desc = source.describe()
            self._need_gt(desc, gt_key)
            scenes, keep = fr.resolve(source)
            problems: List[str] = []
            self._probe_reference(source, keep, problems)                 # the ground truth of the first kept frame is readable: before the network runs, not after
            if problems:
                raise EvaluationError(problems[0])
            pinfo = self.runner.run(net, fr, taps, device, workers, control, progress)
            rs = self.stored(net, pinfo["results"].partition("@")[2])
            missing = [r.label for r in keep if rs is None or not rs.has(r)]
            if missing:
                raise EvaluationError(f"the Runner did not store {len(missing)} of the {len(keep)} frames the labelled data lists (e.g. {missing[0]}): the two sources disagree about the set")
            return self._score(net, source, desc, rs, keep, fr, metrics, pinfo["seconds_network"], pinfo["frames_predicted"], pinfo["frames_predicted"] == 0, progress)
        finally:
            self._release(source)

    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress):
        out, name, metrics = self.ad.out(net), self.ad.name(net), tuple(metrics)
        eid = self._eval_id(rs.id, keep, desc, metrics)
        task = self._task(net, metrics)
        ref_of = (lambda ref: self.ad.reference(source, ref)) if hasattr(self.ad, "reference") else None
        quant = Quantitative(SimpleNamespace(out=os.path.join(out, "suite"), source=source, subject=None, task=task, state=CARRY))
        t1 = time.time()
        try:
            with writer_lock(os.path.join(out, "suite", "quant", eid)):
                store = quant.run(frames=keep, state=CARRY, name=eid, results=rs, reference=ref_of, progress=progress)
        except Locked as e:
            raise EvaluationError(str(e)) from None
        info = store.manifest.get("evaluation") or {"id": f"{name}@{eid}", "results": f"{name}@{rs.id}", "network": net, "frames": str(fr), "gt": desc.get("gt_version") or desc.get("gt_signature"),
                                                    "gt_signature": desc.get("gt_signature"), "metrics": list(metrics), "n_frames": len(store.frames), "n_scenes": len(store.scene_names()),
                                                    "control": rs.manifest["identity"]["inputs"].get("control") or "", "state": CARRY, "evaluator": desc.get("evaluator_hash")}
        if "evaluation" not in store.manifest:
            store.manifest["evaluation"] = info                               # written once: a later call of the same question does not overwrite its record
            store._write_manifest()
        return Result(self, {**info, "warnings": self._warnings(rs, {r.scene for r in keep}, store), "predictions_reused": reused, "frames_predicted": predicted,
                             "seconds_network": seconds_network, "seconds_score": round(time.time() - t1, 1)})

    # ------------------------------------------------------------ small helpers
    @staticmethod
    def _warnings(rs: Optional[ResultSet], scenes, store=None) -> List[str]:
        """what the reader of a number should know: the devices the frames were predicted on (from the frames the evaluation scored) and the memory's health"""
        out = []
        if store is not None:
            devs = sorted({fr["info"]["device"] for fr in store.frames if fr["info"].get("device")})
            if len(devs) > 1:
                out.append(f"the frames were predicted on different devices {devs}: their float results differ slightly")
        return out + _Helpers.health(rs, scenes)

    @staticmethod
    def _eval_id(rid: str, keep: Sequence[FrameRef], desc: Dict[str, Any], metrics: Sequence[str]) -> str:
        return "e" + digest([rid, digest([[r.label, r.key] for r in keep], 40), desc.get("gt_signature"), sorted(metrics), desc.get("evaluator_hash")])

    @staticmethod
    def _need_gt(desc: Dict[str, Any], gt_key: Optional[str]) -> None:
        if not desc.get("gt_signature"):
            raise EvaluationError("the data has no ground-truth signature: numbers could not be told from those of another GT; fix the data adapter")
        if gt_key and gt_key not in (desc.get("gt_version"), desc.get("gt_signature")):
            raise EvaluationError(f"the data has ground truth {desc.get('gt_version') or desc.get('gt_signature')} ({desc.get('gt_signature')}), not {gt_key}")

    def _source(self, net: str, of: str, workers: int, control: str) -> Source:
        """the Source of a set WITH its ground truth (the adapter's gt_source, else its source); a set the project does not define is refused before anything is opened, and whatever the
        adapter raises (SystemExit included) is an EvaluationError"""
        known = self.ad.sets()
        if of not in known:
            raise EvaluationError(f"set {of!r}: not one of {sorted(known)}" + (" (the TEST split is not offered)" if "test" in of else ""))
        try:
            return getattr(self.ad, "gt_source", self.ad.source)(net, of, workers=workers, control=control)
        except SystemExit as e:
            raise EvaluationError(str(e.code) if e.code is not None else "the data adapter stopped") from None
        except (ValueError, KeyError, OSError) as e:
            raise EvaluationError(str(e.args[0] if isinstance(e, KeyError) and e.args else e)) from None

    def _probe_reference(self, source: Source, keep: Sequence[FrameRef], problems: List[str]) -> None:
        try:
            (self.ad.reference(source, keep[0]) if hasattr(self.ad, "reference") else source.load(keep[0]).reference)
        except Exception as e:                                                # noqa: BLE001  whatever the data adapter raises: its message is the problem
            problems.append(f"the ground truth of {keep[0].label} cannot be read: {type(e).__name__}: {e}")

    def _task(self, net: str, metrics: Sequence[str]):
        key = (net, tuple(metrics))
        if key not in self._tasks:
            try:
                self._tasks[key] = self.ad.task(net, tuple(metrics))
            except ValueError as e:
                raise EvaluationError(str(e)) from None
        return self._tasks[key]

    # ---------------------------------------------------------------------------------------- reading what was stored
    def _open(self, eval_id: str):
        """(info, store, quantitative) of a stored evaluation"""
        name, _, eid = str(eval_id).partition("@")
        if not eid.startswith("e"):
            raise EvaluationError(f"{eval_id!r} is not an evaluation id (<network>@e<hash>; `list NETWORK` shows them)")
        net = self._networks.get(name) or self.resolve(self.ad.network(name))
        path = os.path.join(self.ad.out(net), "suite", "quant", eid)
        from .store import RunStore
        try:
            store = RunStore.open(path)
        except OSError:
            raise EvaluationError(f"no evaluation {eval_id}: nothing at {path}") from None
        info = store.manifest.get("evaluation")
        if not info:
            raise EvaluationError(f"{eval_id} is a store without an evaluation record")
        task = self._task(info["network"], tuple(info["metrics"]))
        return info, store, Quantitative(SimpleNamespace(out=os.path.join(self.ad.out(info["network"]), "suite"), source=None, subject=None, task=task, state=CARRY))

    def _predictions(self, info: Dict[str, Any]) -> Optional[ResultSet]:
        return self.stored(info["network"], info["results"].partition("@")[2])

    def quant(self, id: str, match: Optional[str] = None) -> Dict[str, Any]:
        """the dataset-level numbers of an evaluation ('metric.key' -> value, only those containing `match`) and which way is better ('high' / 'low')"""
        info, store, q = self._open(id)
        tab = q.table(store)
        better = {f"{m.name}.{k}": m.direction(k) for m in q._metrics().values() for k in m.headline}
        keep = (lambda k: match in k) if match else (lambda k: True)
        return {"id": id, "n_frames": info["n_frames"], "gt": info["gt"], "numbers": {k: v for k, v in tab.items() if keep(k)}, "better": {k: v for k, v in better.items() if keep(k)},
                "warnings": self._warnings(self._predictions(info), {fr["scene"] for fr in store.frames}, store)}

    def by(self, id: str, axis: str = "scene") -> Dict[str, Any]:
        """the headline numbers per scene (axis 'scene') or the scores of every frame (axis 'frame')"""
        _, store, q = self._open(id)
        if axis == "scene":
            return q.per_scene(store)
        if axis == "frame":
            return {fr["label"]: fr["scores"] for fr in store.frames}
        raise EvaluationError(f"axis {axis!r}: scene or frame")

    def worst(self, id: str, match: Optional[str] = None, n: int = 10) -> Dict[str, List[dict]]:
        """where the numbers come from: the scenes whose removal improves a headline number most and the frames furthest out in the bad direction (rows that contain `match`); a
        stored evaluation has no plausibility flags (they need the inputs)"""
        _, store, q = self._open(id)
        res = q.outliers(store, k=n)
        if match:
            res = {k: [row for row in v if match in json.dumps(row, default=str)] for k, v in res.items()}
        return res

    def compare(self, a: str, b: str, boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
        """paired scene bootstrap of the headline numbers, b - a (a = the control, b = the variant); refused when the two do not answer the same question (other GT, other frames)"""
        _, sa, qa = self._open(a)
        _, sb, _ = self._open(b)
        try:
            return qa.compare(sa, sb, boot, seed, ignore=getattr(self.ad, "COMPARE_IGNORE", None))
        except ValueError as e:
            raise EvaluationError(str(e)) from None

    def cost(self, id: str) -> Dict[str, Any]:
        """ms per frame of the network as it ran (the median over the frames of the last session that ran frames, on that day's machine)"""
        info, store, q = self._open(id)
        rs = self._predictions(info)
        return {"id": id, "predict": rs.manifest.get("cost") if rs else None, "devices": sorted({fr["info"]["device"] for fr in store.frames if fr["info"].get("device")})}

    def report(self, id: str) -> str:
        """the html page of an evaluation (numbers, scenes by drag, outliers); returns its path"""
        info, store, q = self._open(id)
        return q.report(store, render=0)

    def card(self, id: str) -> Dict[str, Any]:
        """the metric card in the project's contract (what its decision tool reads); refused when the checkpoint changed since the predictions"""
        if not hasattr(self.ad, "card"):
            raise EvaluationError("this project's adapter defines no card")
        info, store, q = self._open(id)
        rs = self._predictions(info)
        self._same_network(info, rs)
        return self.ad.card(info, store, q, rs)

    def _same_network(self, info: Dict[str, Any], rs: Optional[ResultSet]) -> None:
        if rs is not None and self.net_identity(info["network"]) != rs.manifest["identity"]["network"]:
            raise EvaluationError(f"{info['network']} is not the network of these predictions any more (another checkpoint, config or code): "
                                  "what it would draw or certify is not what was evaluated")

    # ---------------------------------------------------------------------------------------- a picture
    def qual_frame(self, id: str, frame: str, view: str = "overview", device: str = "cpu", workers: int = 0) -> Dict[str, Any]:
        """draw one frame of an evaluation. Drawing runs the network on that frame again (the engine's own path, with the history of its scene, on `device`; its scores can differ slightly
        from the stored ones when the evaluation ran on another device): the one place besides the Runner that executes the network. Returns the picture paths and the frame's scores"""
        info, store, q = self._open(id)
        self._same_network(info, self._predictions(info))
        from .suite import EvalSuite
        fr = Frames.parse(info["frames"])
        source = self._source(info["network"], fr.of, workers, info.get("control", ""))
        suite = EvalSuite(source, self.ad.subject(info["network"], source, device), self._task(info["network"], tuple(info["metrics"])),
                          out=os.path.join(self.ad.out(info["network"]), "suite"), state=CARRY)
        try:
            res = suite.qual.frame(frame, views=(view,), history="full")
        finally:
            suite.close()
            self._release(source)
        return {"frame": res.ref, "images": res.images, "scores": res.scores}

    def selftest(self) -> Dict[str, Any]:
        """the checks of this interface on a toy project (a few seconds; no data, no network)"""
        from .selftest_api import run_all
        return run_all()


Evaluation.METHODS = tuple(n for n, f in vars(Evaluation).items() if callable(f) and not n.startswith("_"))      # the methods are the commands: no second list


class Result(dict):
    """what `run` returns: a plain dict (the id, the frames, the GT, what was done, warnings) whose methods read the evaluation of this id; pickled or copied it is the plain dict"""

    def __init__(self, ev: Evaluation, info: Dict[str, Any]):
        super().__init__(info)
        self._ev = ev

    def __reduce__(self):
        return (dict, (dict(self),))

    def quant(self, match: Optional[str] = None):
        return self._ev.quant(self["id"], match)

    def by(self, axis: str = "scene"):
        return self._ev.by(self["id"], axis)

    def worst(self, match: Optional[str] = None, n: int = 10):
        return self._ev.worst(self["id"], match, n)

    def compare(self, a: "Result", boot: int = 1000, seed: int = 0):
        """this evaluation (b) against `a` (the control): b - a"""
        return self._ev.compare(a["id"], self["id"], boot, seed)

    def cost(self):
        return self._ev.cost(self["id"])

    def report(self):
        return self._ev.report(self["id"])

    def card(self):
        return self._ev.card(self["id"])
