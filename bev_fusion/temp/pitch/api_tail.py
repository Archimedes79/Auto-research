# ------------------------------------------------------------------------------------------------ the adapter
ADAPTER_FUNCTIONS = ("name", "out", "network", "sets", "source", "subject", "task", "identity")


def load_adapter(spec: Any, project: Optional[str]):
    """the adapter module: given (a module or object with the functions), named in `spec` (a module path), or named in <project>/evaluation/evalsuite.json"""
    project = os.path.abspath(project or os.environ.get("AR_PROJECT") or ".")
    if spec is not None and not isinstance(spec, str):
        mod = spec
    else:
        if spec is None:
            cfg = os.path.join(project, "evaluation", "evalsuite.json")
            try:
                with open(cfg, encoding="utf-8") as f:
                    spec = json.load(f)["adapter"]
            except (OSError, ValueError, KeyError):
                raise EvaluationError(f"no adapter: pass one, or write {cfg} as {{\"adapter\": \"module.path\"}}") from None
        if project not in sys.path:
            sys.path.insert(0, project)
        try:
            mod = importlib.import_module(spec)
        except ImportError as e:
            raise EvaluationError(f"adapter {spec!r} cannot be imported: {e}") from None
    missing = [f for f in ADAPTER_FUNCTIONS if not hasattr(mod, f)]
    if missing:
        raise EvaluationError(f"the adapter lacks {', '.join(missing)}")
    return mod


def _hash(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ the class
class Evaluation:
    def __init__(self, project: Optional[str] = None, adapter: Any = None):
        self.ad = load_adapter(adapter, project)
        self._tasks: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
        self._networks: Dict[str, str] = {}                               # the networks this object has seen: name -> path (an id carries only the name)

    # ---------------------------------------------------------------------------------------- what exists, and the dry run
    def sets(self) -> Dict[str, Any]:
        """the sets of scenes the project defines: name -> number of scenes (`val` and `train` are the network's own lists: no number)"""
        return dict(self.ad.sets())

    def list(self, network: str) -> List[Dict[str, Any]]:
        """the stored predictions and evaluations of a network, with their size"""
        net = self._net(network)
        out, name, rows = self.ad.out(net), self.ad.name(net), []
        base = os.path.join(out, "results")
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            try:
                rs = ResultSet.open(os.path.join(base, d))
            except (OSError, ValueError):
                continue
            rows.append({"id": f"{name}@{d}", "kind": "predictions", "frames": len(rs._listing()), "megabytes": rs.size_mb(), "created": rs.manifest.get("created")})
        base = os.path.join(out, "suite", "quant")
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
        """the dry run: what `run` would do with these frames, and what is wrong (`ok`, `problems`). Starts no network (the data is opened)"""
        problems: List[str] = []
        res: Dict[str, Any] = {"network": network, "frames": str(frames), "ok": False, "problems": problems}
        try:
            net, fr = self._net(network), Frames.parse(frames)
            res["network"] = net
            ident = self._net_identity(net)
            self._task(net, tuple(metrics))
            source = self.ad.source(net, fr.of, workers=workers, control="")
        except (EvaluationError, ValueError, KeyError, OSError, SystemExit) as e:
            problems.append(str(e))
            return res
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            rid = result_id(self._identity(ident, source))
            rs = self._stored(net, rid)
            have = sum(1 for r in keep if rs and rs.has(r))
            res.update({"scenes_run": len(scenes), "frames_kept": len(keep), "frames_stored": have, "results": f"{self.ad.name(net)}@{rid}", "out": self.ad.out(net),
                        "gt": desc.get("gt_version") or desc.get("gt_signature"), "state": "carry (whole scenes in time order, D70)",
                        "network_runs": have < len(keep), "id": f"{self.ad.name(net)}@{self._eval_id(rid, keep, desc, metrics)}"})
            if not desc.get("gt_signature"):
                problems.append("the data has no ground-truth signature: the numbers could not be told from another GT's")
            self._probe_reference(source, keep, problems)
        except (EvaluationError, ValueError, KeyError, OSError, SystemExit) as e:
            problems.append(str(e))
        finally:
            self._release(source)
        res["ok"] = not problems
        return res

    # ---------------------------------------------------------------------------------------- running (the only code that executes the network)
    def predict(self, network: str, frames: str = "val", device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """run the network on the frames that are not stored yet (whole scenes in time order, warm: D70) and store what it predicts; -> the id of the predictions and what was done"""
        net, fr = self._net(network), Frames.parse(frames)
        ident = self._net_identity(net)
        source = self.ad.source(net, fr.of, workers=workers, control=control)
        try:
            desc = source.describe()
            self._need_gt(desc, None)
            scenes, keep = fr.resolve(source)
            return self._predict(net, source, desc, ident, scenes, keep, device, progress)[1]
        finally:
            self._release(source)

    def score(self, results: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Sequence[str] = ("maps", "boxes"), workers: int = 0, progress: int = 50) -> "Result":
        """score stored predictions (an id `<network>@r<hash>`) on the frames against the ground truth; the network is not run; frames that were not predicted are an error"""
        name, _, rid = str(results).partition("@")
        net, fr = self._net(self.ad.network(name) if name not in self._networks else self._networks[name]), Frames.parse(frames)
        rs = self._stored(net, rid)
        if rs is None:
            raise EvaluationError(f"no predictions {results}: nothing at {os.path.join(self.ad.out(net), 'results', rid)}")
        source = self.ad.source(net, fr.of, workers=workers, control=rs.manifest["identity"]["inputs"].get("control") or "")
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            if self._need_gt(desc, gt_key) and any(not rs.has(r) for r in keep):
                raise EvaluationError(f"{sum(not rs.has(r) for r in keep)} of the {len(keep)} frames are not in {results}: `predict` them first, or use `run`")
            return self._score(net, source, desc, rs, keep, fr, metrics, 0.0, 0.0, False, progress)
        finally:
            self._release(source)

    def run(self, network: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Sequence[str] = ("maps", "boxes"), device: str = "cuda", workers: int = 2,
            control: str = "", progress: int = 50) -> "Result":
        """predict (what is not stored yet), then score against the ground truth; gt_key = the GT version the numbers must be on (refused when the data has another);
        metrics = what the project offers (BEV: maps, boxes, curves, fusion, velocity). -> a Result: a dict with the id, whose methods read the evaluation"""
        net, fr, metrics = self._net(network), Frames.parse(frames), tuple(metrics)
        ident = self._net_identity(net)
        self._task(net, metrics)                                          # a typo in the metrics fails here, not after the network ran
        source = self.ad.source(net, fr.of, workers=workers, control=control)
        try:
            desc = source.describe()
            self._need_gt(desc, gt_key)
            scenes, keep = fr.resolve(source)
            problems: List[str] = []
            self._probe_reference(source, keep, problems)                 # the ground truth of the first kept frame is readable: before the network runs, not after
            if problems:
                raise EvaluationError(problems[0])
            rs, pinfo = self._predict(net, source, desc, ident, scenes, keep, device, progress)
            return self._score(net, source, desc, rs, keep, fr, metrics, pinfo["seconds_network"], pinfo["frames_predicted"], pinfo["frames_predicted"] == 0, progress, control)
        finally:
            self._release(source)

    # ------------------------------------------------------------ the two halves
    def _predict(self, net, source, desc, ident, scenes, keep, device, progress):
        out = self.ad.out(net)
        identity = self._identity(ident, source)
        rid = result_id(identity)
        rs = ResultSet.create(os.path.join(out, "results", rid), identity)
        todo = [r for r in keep if not rs.has(r)]
        t0 = time.time()
        if todo:
            try:
                with writer_lock(rs.path):
                    rs = ResultSet.open(rs.path, "a")                          # another writer may have stored frames while we waited for the lock
                    todo = [r for r in keep if not rs.has(r)]
                    if todo:
                        self._run_network(net, source, scenes, keep, rs, desc.get("gt_signature"), device, progress)
            except Locked as e:
                raise EvaluationError(str(e)) from None
            if self._net_identity(net) != ident:
                raise EvaluationError(f"{net} changed while the network ran (a new checkpoint?): the predictions belong to the old one; run again")
        return rs, {"results": f"{self.ad.name(net)}@{rid}", "frames_kept": len(keep), "frames_predicted": len(todo), "seconds_network": round(time.time() - t0, 1),
                    "predictions_reused": not todo}

    def _run_network(self, net, source, scenes, keep, rs, gt, device, progress):
        subject = self.ad.subject(net, source, device)
        runner = Runner(source, subject, "carry")
        want, t0, n, ms = {r.label for r in keep}, time.time(), 0, []
        total = sum(1 for r in keep if not rs.has(r))
        try:
            for sc in scenes:
                kept = [r for r in source.frames(sc) if r.label in want]
                if all(rs.has(r) for r in kept):
                    continue
                refs = source.frames(sc)[: kept[-1].index + 1]                 # nothing after the last frame asked for is needed; the frames before it always run (the memory)
                for sample, pred in runner.iter_scene(sc, refs):
                    if sample.ref.label in want and not rs.has(sample.ref):
                        rs.add(sample.ref, pred, gt=gt, device=str(getattr(subject, "device", device)))
                        n += 1
                        ms.append(float(pred.info.get("predict_ms", 0.0)))
                        if progress and n % progress == 0:
                            print(f"[predict {rs.id}] {n}/{total} frames, {(time.time() - t0) / n:.2f} s per frame", file=sys.stderr, flush=True)
        finally:
            rs.close(wall_s=time.time() - t0, cost={"median_ms": sorted(ms)[len(ms) // 2], "n": len(ms)} if ms else None)
            subject.close()

    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress, control=""):
        out, name, metrics = self.ad.out(net), self.ad.name(net), tuple(metrics)
        eid = self._eval_id(rs.id, keep, desc, metrics)
        task = self._task(net, metrics)
        ref_of = (lambda ref: self.ad.reference(source, ref)) if hasattr(self.ad, "reference") else None
        quant = Quantitative(SimpleNamespace(out=os.path.join(out, "suite"), source=source, subject=None, task=task, state="carry"))
        t1 = time.time()
        try:
            with writer_lock(os.path.join(out, "suite", "quant", eid)):
                store = quant.run(frames=keep, state="carry", name=eid, results=rs, reference=ref_of, progress=progress)
        except Locked as e:
            raise EvaluationError(str(e)) from None
        gts = rs.manifest.get("gts", {})
        warnings = []
        if gts and set(gts) != {desc.get("gt_signature")}:
            warnings.append(f"some predictions were made while the data had ground truth {sorted(gts)}, now it is {desc.get('gt_signature')}: the numbers a prediction computes itself "
                            "(the val loss) belong to that earlier ground truth; maps, boxes and the rest are scored against the present one")
        info = store.manifest.get("evaluation") or {"id": f"{name}@{eid}", "results": f"{name}@{rs.id}", "network": net, "frames": str(fr), "gt": desc.get("gt_version") or desc.get("gt_signature"),
                                                    "gt_signature": desc.get("gt_signature"), "metrics": list(metrics), "n_frames": len(store.frames), "n_scenes": len(store.scene_names()),
                                                    "control": control, "state": "carry", "evaluator": desc.get("evaluator_hash")}
        if "evaluation" not in store.manifest:
            store.manifest["evaluation"] = info                               # written once: a later call of the same question does not overwrite its record
            store._write_manifest()
        return Result(self, {**info, "warnings": warnings, "predictions_reused": reused, "frames_predicted": predicted, "seconds_network": seconds_network,
                             "seconds_score": round(time.time() - t1, 1)})

    # ------------------------------------------------------------ small helpers
    def _net(self, network: str) -> str:
        net = self.ad.resolve(network) if hasattr(self.ad, "resolve") else network
        self._networks[self.ad.name(net)] = net
        return net

    def _net_identity(self, net: str) -> Dict[str, Any]:
        try:
            return self.ad.identity(net)
        except OSError as e:
            raise EvaluationError(f"{net}: {e}") from None

    def _identity(self, net_identity: Dict[str, Any], source: Source) -> Dict[str, Any]:
        inputs = self.ad.inputs(source) if hasattr(self.ad, "inputs") else {k: v for k, v in source.describe().items() if k not in RESULT_DROP}
        return {"network": net_identity, "inputs": inputs, "state": "carry"}

    @staticmethod
    def _eval_id(rid: str, keep: Sequence[FrameRef], desc: Dict[str, Any], metrics: Sequence[str]) -> str:
        return "e" + _hash(rid, hashlib.sha1("\n".join(f"{r.label} {r.key}" for r in keep).encode()).hexdigest(), desc.get("gt_signature"), sorted(metrics), desc.get("evaluator_hash"))

    @staticmethod
    def _need_gt(desc: Dict[str, Any], gt_key: Optional[str]) -> bool:
        if not desc.get("gt_signature"):
            raise EvaluationError("the data has no ground-truth signature: numbers could not be told from those of another GT; fix the data adapter")
        if gt_key and gt_key not in (desc.get("gt_version"), desc.get("gt_signature")):
            raise EvaluationError(f"the data has ground truth {desc.get('gt_version') or desc.get('gt_signature')} ({desc.get('gt_signature')}), not {gt_key}")
        return True

    def _probe_reference(self, source: Source, keep: Sequence[FrameRef], problems: List[str]) -> None:
        try:
            (self.ad.reference(source, keep[0]) if hasattr(self.ad, "reference") else source.load(keep[0]).reference)
        except Exception as e:                                                # noqa: BLE001  whatever the data adapter raises: its message is the problem
            problems.append(f"the ground truth of {keep[0].label} cannot be read: {type(e).__name__}: {e}")

    def _release(self, source: Source) -> None:
        if hasattr(self.ad, "release"):
            self.ad.release(source)

    def _stored(self, net: str, rid: str) -> Optional[ResultSet]:
        path = os.path.join(self.ad.out(net), "results", rid)
        return ResultSet.open(path) if os.path.exists(os.path.join(path, "manifest.json")) else None

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
        net = self._networks.get(name) or self._net(self.ad.network(name))
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
        return info, store, Quantitative(SimpleNamespace(out=os.path.join(self.ad.out(info["network"]), "suite"), source=None, subject=None, task=task, state="carry"))

    def _predictions(self, info: Dict[str, Any]) -> Optional[ResultSet]:
        return self._stored(info["network"], info["results"].partition("@")[2])

    def quant(self, id: str, match: Optional[str] = None) -> Dict[str, Any]:
        """the dataset-level numbers of an evaluation ('metric.key' -> value, only those containing `match`) and which way is better ('high' / 'low')"""
        info, store, q = self._open(id)
        tab = q.table(store)
        better = {f"{m.name}.{k}": m.direction(k) for m in q._metrics().values() for k in m.headline}
        keep = (lambda k: match in k) if match else (lambda k: True)
        return {"id": id, "n_frames": info["n_frames"], "gt": info["gt"], "numbers": {k: v for k, v in tab.items() if keep(k)}, "better": {k: v for k, v in better.items() if keep(k)}}

    def by(self, id: str, axis: str = "scene") -> Dict[str, Any]:
        """the headline numbers per scene (axis 'scene') or the scores of every frame (axis 'frame')"""
        info, store, q = self._open(id)
        if axis == "scene":
            return q.per_scene(store)
        if axis == "frame":
            return {fr["label"]: fr["scores"] for fr in store.frames}
        raise EvaluationError(f"axis {axis!r}: scene or frame")

    def worst(self, id: str, match: Optional[str] = None, n: int = 10) -> Dict[str, List[dict]]:
        """where the numbers come from: the scenes whose removal improves a headline number most and the frames furthest out in the bad direction (rows that contain `match`); a
        stored evaluation has no plausibility flags (they need the inputs)"""
        info, store, q = self._open(id)
        res = q.outliers(store, k=n)
        if match:
            res = {k: [row for row in v if match in json.dumps(row, default=str)] for k, v in res.items()}
        return res

    def compare(self, control: str, variant: str, boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
        """paired scene bootstrap of the headline numbers, variant - control; refused when the two do not answer the same question (other GT, other frames)"""
        ia, sa, qa = self._open(control)
        _, sb, _ = self._open(variant)
        try:
            return qa.compare(sa, sb, boot, seed, ignore=getattr(self.ad, "COMPARE_IGNORE", None))
        except ValueError as e:
            raise EvaluationError(str(e)) from None

    def cost(self, id: str) -> Dict[str, Any]:
        """ms per frame of the network as it ran (the median over the frames of the last session that ran frames, on that day's machine)"""
        info, store, q = self._open(id)
        rs = self._predictions(info)
        return {"id": id, "predict": rs.manifest.get("cost") if rs else None, "devices": rs.manifest.get("devices") if rs else None}

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
        if rs is not None and self._net_identity(info["network"]) != rs.manifest["identity"]["network"]:
            raise EvaluationError(f"{info['network']} is not the network of these predictions any more (another checkpoint, config or code): "
                                  "what it would draw or certify is not what was evaluated")

    # ---------------------------------------------------------------------------------------- about the network itself
    def qual_frame(self, id: str, frame: str, view: str = "overview", device: str = "cpu", workers: int = 0) -> Dict[str, Any]:
        """draw one frame of an evaluation (the network runs on it with the history of its scene, as it did in the evaluation); returns the picture paths and the frame's scores"""
        info, store, q = self._open(id)
        self._same_network(info, self._predictions(info))
        from .suite import EvalSuite
        fr = Frames.parse(info["frames"])
        source = self.ad.source(info["network"], fr.of, workers=workers, control=info.get("control", ""))
        suite = EvalSuite(source, self.ad.subject(info["network"], source, device), self._task(info["network"], tuple(info["metrics"])),
                          out=os.path.join(self.ad.out(info["network"]), "suite"), state="carry")
        try:
            res = suite.qual.frame(frame, views=(view,), history="full")
        finally:
            suite.close()
            self._release(source)
        return {"frame": res.ref, "images": res.images, "scores": res.scores}

    def time(self, network: str, n: int = 30) -> Dict[str, Any]:
        """milliseconds per module and per frame at batch 1 (idle GPU only: the adapter refuses a busy one)"""
        if not hasattr(self.ad, "time"):
            raise EvaluationError("this project's adapter defines no timing")
        return self.ad.time(self._net(network), n)

    def probe(self, network: str, frames: int = 3, device: str = "cpu") -> Dict[str, Any]:
        """the module probe: statistics and dead channels of every module, the response when the camera or the radar is removed (frames = how many frames of the first scene)"""
        if not hasattr(self.ad, "probe"):
            raise EvaluationError("this project's adapter defines no probe")
        return self.ad.probe(self._net(network), frames, device)

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

    def compare(self, control: "Result", boot: int = 1000, seed: int = 0):
        """this evaluation (the variant) against `control`"""
        return self._ev.compare(control["id"], self["id"], boot, seed)

    def cost(self):
        return self._ev.cost(self["id"])

    def report(self):
        return self._ev.report(self["id"])

    def card(self):
        return self._ev.card(self["id"])
