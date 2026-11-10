def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/results.py", [
    ("""identity, so a stored frame is valid whatever else was asked. Layout: `<path>/manifest.json`, `<path>/frames/<scene>_<index>.pkl` (zlib level 1 over a pickle of {outputs, taps, info}):
the files are the program's own cache, not an exchange format; open a ResultSet only from a folder this program made.""",
     """identity, so a stored frame is valid whatever else was asked. Layout: `<path>/manifest.json`, `<path>/frames/<scene>_<index>.pkl` (zlib level 1 over a pickle of {outputs, info}: the files
are the program's own cache, not an exchange format; open a ResultSet only from a folder this program made) and, for the intermediate results asked for with the predictions (taps),
`<path>/taps/<scene>_<index>/<tap>.npy`. A frame stored without a tap does not have it: `has(ref, taps)` says no, and the frame is predicted again to get it."""),
    ("Numpy-free; the core knows nothing about the network or the data.", "The core knows nothing about the network or the data."),
    ("import time\nimport zlib\nfrom typing import Any, Dict, Iterator, List, Optional\n\nfrom .contracts", "import time\nimport zlib\nfrom typing import Any, Dict, Iterator, List, Optional, Sequence\n\nimport numpy as np\n\nfrom .contracts"),
    ('''"created": time.strftime("%Y-%m-%d %H:%M:%S"), "gts": {}, "devices": {}, "wall_s": 0.0}, "a")''',
     '''"created": time.strftime("%Y-%m-%d %H:%M:%S"), "gts": {}, "devices": {}, "health": {}, "wall_s": 0.0}, "a")'''),
    ('''    def has(self, ref: FrameRef) -> bool:
        return os.path.basename(self._file(ref)) in self._listing()''', '''    def _tap_file(self, ref: FrameRef, name: str) -> str:
        return os.path.join(self.path, "taps", ref.slug, name.replace("/", "_") + ".npy")

    def has(self, ref: FrameRef, taps: Sequence[str] = ()) -> bool:
        """the frame is stored (with these taps, when asked)"""
        return os.path.basename(self._file(ref)) in self._listing() and all(os.path.exists(self._tap_file(ref, t)) for t in taps)

    def tap_names(self, ref: FrameRef) -> List[str]:
        folder = os.path.join(self.path, "taps", ref.slug)
        return sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".npy")) if os.path.isdir(folder) else []'''),
    ('''    def load(self, ref: FrameRef) -> Prediction:
        if not self.has(ref):
            raise KeyError(f"{ref.label} is not in result set {self.id}")
        try:
            with open(self._file(ref), "rb") as f:
                d = pickle.loads(zlib.decompress(f.read()))
        except (OSError, zlib.error, pickle.UnpicklingError, EOFError) as e:
            raise IOError(f"{self._file(ref)} is unreadable ({type(e).__name__}: {e}); remove the file, the next run predicts the frame again") from None
        return Prediction(outputs=d["outputs"], taps=d["taps"], info=d["info"])''', '''    def load(self, ref: FrameRef, taps: Sequence[str] = ()) -> Prediction:
        """the stored prediction; `taps` = the intermediate results to load with it (a KeyError for one that was not stored)"""
        if not self.has(ref, taps):
            raise KeyError(f"{ref.label} is not in result set {self.id}" + (f" with the taps {list(taps)}" if taps else ""))
        try:
            with open(self._file(ref), "rb") as f:
                d = pickle.loads(zlib.decompress(f.read()))
            arrays = {t: np.load(self._tap_file(ref, t)) for t in taps}
        except (OSError, ValueError, zlib.error, pickle.UnpicklingError, EOFError) as e:
            raise IOError(f"{self._file(ref)} (or a tap of it) is unreadable ({type(e).__name__}: {e}); remove the file, the next run predicts the frame again") from None
        return Prediction(outputs=d["outputs"], taps=arrays, info=d["info"])'''),
    ('''        """store a frame; `gt` = the ground truth that was current while the network ran (the loss of a prediction is computed against it), `device` = where it ran"""
        if self.mode == "r":
            raise IOError("result set opened read-only")
        _write_atomic(self._file(ref), zlib.compress(pickle.dumps({"outputs": pred.outputs, "taps": pred.taps, "info": pred.info}, protocol=4), 1))
        self._listing().add(os.path.basename(self._file(ref)))
        for key, val in (("gts", gt), ("devices", device)):
            if val is not None:
                self.manifest[key][str(val)] = self.manifest[key].get(str(val), 0) + 1''', '''        """store a frame; `gt` = the ground truth that was current while the network ran (the loss of a prediction is computed against it), `device` = where it ran. The taps of the
        prediction are stored next to it; a frame that is stored already keeps its outputs (a second call only adds taps)"""
        if self.mode == "r":
            raise IOError("result set opened read-only")
        new = not self.has(ref)
        if new:
            _write_atomic(self._file(ref), zlib.compress(pickle.dumps({"outputs": pred.outputs, "info": pred.info}, protocol=4), 1))
            self._listing().add(os.path.basename(self._file(ref)))
        for name, arr in pred.taps.items():
            os.makedirs(os.path.dirname(self._tap_file(ref, name)), exist_ok=True)
            tmp = self._tap_file(ref, name) + f".tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                np.save(f, np.asarray(arr))
            os.replace(tmp, self._tap_file(ref, name))
        for key, val in (("gts", gt), ("devices", device)):
            if val is not None and new:
                self.manifest[key][str(val)] = self.manifest[key].get(str(val), 0) + 1'''),
])

patch("evalsuite/api.py", [
    ("import hashlib\nimport importlib\nimport json\nimport os\nimport sys\nimport time\nfrom types import SimpleNamespace", "import hashlib\nimport importlib\nimport json\nimport os\nimport sys\nimport time\nfrom types import SimpleNamespace"),
    ("from .contracts import FrameRef, Source\n", "import numpy as np\n\nfrom .contracts import FrameRef, Prediction, Source\n"),
    ('''RESULT_DROP = (''', '''MEMORY_START_MAX, MEMORY_GROWTH_MIN, MEMORY_MIN_FRAMES = 0.05, 0.02, 4        # the health of a carried memory (D70): empty at the first frame of a scene, filling over its frames
RESULT_DROP = ('''),
    ('''  optional: resolve(text) -> network path (a run name, a relative path), release(source), reference(source, ref) -> the ground truth of a frame without loading the inputs,
  inputs(source) -> what the source was fed, COMPARE_IGNORE, card(...), time(network, n), probe(network, frames, device).''', '''  optional: resolve(text) -> network path (a run name, a relative path), release(source), reference(source, ref) -> the ground truth of a frame without loading the inputs,
  inputs(source) -> what the source was fed, COMPARE_IGNORE, card(...), time(network, n), probe(network, frames, device), MEMORY_TAP = the name of a tap whose mean is the fill of the
  network's memory (the health check: below).
Intermediate results: `predict(..., taps=("fused", ...))` stores the named taps of the Subject (its `taps()`) with the predictions; a frame stored without a tap is predicted again when the
tap is asked for. The health of the memory (D70) is checked on every scene the network runs, from the MEMORY_TAP means: empty at the first frame, growing over the frames; a scene that
fails is a warning in the result, never an error.'''),
    ('''    def predict(self, network: str, frames: str = "val", device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """run the network on the frames that are not stored yet (whole scenes in time order, warm: D70) and store what it predicts; -> the id of the predictions and what was done"""''',
     '''    def predict(self, network: str, frames: str = "val", taps: Sequence[str] = (), device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """run the network on the frames that are not stored yet (whole scenes in time order, warm: D70) and store what it predicts, with the intermediate results `taps` when asked;
        -> the id of the predictions, what was done and the warnings"""'''),
    ('''            return self._predict(net, source, desc, ident, scenes, keep, device, progress)[1]''', '''            return self._predict(net, source, desc, ident, scenes, keep, tuple(taps), device, progress)[1]'''),
    ('''    def run(self, network: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Sequence[str] = ("maps", "boxes"), device: str = "cuda", workers: int = 2,
            control: str = "", progress: int = 50) -> "Result":''', '''    def run(self, network: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Sequence[str] = ("maps", "boxes"), taps: Sequence[str] = (), device: str = "cuda",
            workers: int = 2, control: str = "", progress: int = 50) -> "Result":'''),
    ('''            rs, pinfo = self._predict(net, source, desc, ident, scenes, keep, device, progress)''', '''            rs, pinfo = self._predict(net, source, desc, ident, scenes, keep, tuple(taps), device, progress)'''),
    ('''    def _predict(self, net, source, desc, ident, scenes, keep, device, progress):
        out = self.ad.out(net)
        identity = self._identity(ident, source)
        rid = result_id(identity)
        rs = ResultSet.create(os.path.join(out, "results", rid), identity)
        todo = [r for r in keep if not rs.has(r)]''', '''    def _predict(self, net, source, desc, ident, scenes, keep, taps, device, progress):
        out = self.ad.out(net)
        identity = self._identity(ident, source)
        rid = result_id(identity)
        rs = ResultSet.create(os.path.join(out, "results", rid), identity)
        todo = [r for r in keep if not rs.has(r, taps)]'''),
    ('''                    rs = ResultSet.open(rs.path, "a")                          # another writer may have stored frames while we waited for the lock
                    todo = [r for r in keep if not rs.has(r)]
                    if todo:
                        self._run_network(net, source, scenes, keep, rs, desc.get("gt_signature"), device, progress)''', '''                    rs = ResultSet.open(rs.path, "a")                          # another writer may have stored frames while we waited for the lock
                    todo = [r for r in keep if not rs.has(r, taps)]
                    if todo:
                        self._run_network(net, source, scenes, keep, rs, desc.get("gt_signature"), taps, device, progress)'''),
    ('''        return rs, {"results": f"{self.ad.name(net)}@{rid}", "frames_kept": len(keep), "frames_predicted": len(todo), "seconds_network": round(time.time() - t0, 1),
                    "predictions_reused": not todo}''', '''        return rs, {"results": f"{self.ad.name(net)}@{rid}", "frames_kept": len(keep), "frames_predicted": len(todo), "seconds_network": round(time.time() - t0, 1),
                    "predictions_reused": not todo, "warnings": self._warnings(rs, keep, desc)}'''),
    ('''    def _run_network(self, net, source, scenes, keep, rs, gt, device, progress):
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
        finally:''', '''    def _run_network(self, net, source, scenes, keep, rs, gt, taps, device, progress):
        subject = self.ad.subject(net, source, device)
        declared = {t.name for t in subject.taps()}
        bad = [t for t in taps if t not in declared]
        if bad:
            subject.close()
            raise EvaluationError(f"unknown taps {bad}: the network offers {sorted(declared)}")
        mem = getattr(self.ad, "MEMORY_TAP", None)
        mem = mem if mem in declared else None                                 # the health check needs the project's memory tap
        names = list(taps) + ([mem] if mem and mem not in taps else [])
        runner = Runner(source, subject, "carry")
        want, t0, n, ms = {r.label for r in keep}, time.time(), 0, []
        total = sum(1 for r in keep if not rs.has(r, taps))
        try:
            for sc in scenes:
                kept = [r for r in source.frames(sc) if r.label in want]
                if all(rs.has(r, taps) for r in kept):
                    continue
                refs = source.frames(sc)[: kept[-1].index + 1]                 # nothing after the last frame asked for is needed; the frames before it always run (the memory)
                fill = []
                for sample, pred in runner.iter_scene(sc, refs, taps=names):
                    if mem and mem in pred.taps:
                        fill.append(float(np.mean(pred.taps[mem])))
                    if sample.ref.label in want and not rs.has(sample.ref, taps):
                        stored = Prediction(pred.outputs, {k: v for k, v in pred.taps.items() if k in taps}, {**pred.info, **({"memory_fill": fill[-1]} if fill else {})})
                        rs.add(sample.ref, stored, gt=gt, device=str(getattr(subject, "device", device)))
                        n += 1
                        ms.append(float(pred.info.get("predict_ms", 0.0)))
                        if progress and n % progress == 0:
                            print(f"[predict {rs.id}] {n}/{total} frames, {(time.time() - t0) / n:.2f} s per frame", file=sys.stderr, flush=True)
                self._memory_health(rs, sc, fill)
        finally:'''),
    ('''    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress, control=""):''', '''    @staticmethod
    def _memory_health(rs: ResultSet, scene: str, fill: List[float]) -> None:
        """D70: a carried memory is empty at the first frame of a scene and fills over its frames; a scene where it is not is recorded in the result set (a warning to every evaluation of it)"""
        rs.manifest.setdefault("health", {}).pop(scene, None)
        if not fill:
            return
        if fill[0] > MEMORY_START_MAX:
            rs.manifest["health"][scene] = f"{scene}: the memory is not empty at the first frame ({fill[0]:.2f} > {MEMORY_START_MAX}): not initialised"
        elif len(fill) >= MEMORY_MIN_FRAMES and max(fill) - fill[0] < MEMORY_GROWTH_MIN:
            rs.manifest["health"][scene] = f"{scene}: the memory does not fill over {len(fill)} frames ({fill[0]:.2f} -> {fill[-1]:.2f}): not used"

    @staticmethod
    def _warnings(rs: ResultSet, keep: Sequence[FrameRef], desc: Dict[str, Any]) -> List[str]:
        out = []
        gts = rs.manifest.get("gts", {})
        if gts and set(gts) != {desc.get("gt_signature")}:
            out.append(f"some predictions were made while the data had ground truth {sorted(gts)}, now it is {desc.get('gt_signature')}: the numbers a prediction computes itself "
                       "(the val loss) belong to that earlier ground truth; maps, boxes and the rest are scored against the present one")
        if len(rs.manifest.get("devices", {})) > 1:
            out.append(f"the predictions were made on different devices {sorted(rs.manifest['devices'])}: their float results differ slightly")
        scenes = {r.scene for r in keep}
        out += [msg for sc, msg in rs.manifest.get("health", {}).items() if sc in scenes]
        return out

    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress, control=""):'''),
    ('''        gts = rs.manifest.get("gts", {})
        warnings = []
        if gts and set(gts) != {desc.get("gt_signature")}:
            warnings.append(f"some predictions were made while the data had ground truth {sorted(gts)}, now it is {desc.get('gt_signature')}: the numbers a prediction computes itself "
                            "(the val loss) belong to that earlier ground truth; maps, boxes and the rest are scored against the present one")
        info = store.manifest.get("evaluation")''', '''        warnings = self._warnings(rs, keep, desc)
        info = store.manifest.get("evaluation")'''),
    ('''    def quant(self, id: str, match: Optional[str] = None) -> Dict[str, Any]:''', '''    def tap(self, results: str, frame: str, name: str = "") -> Dict[str, Any]:
        """the intermediate results stored with a frame's predictions: without `name` which taps there are, with it the path of its .npy file, its shape and dtype"""
        rname, _, rid = str(results).partition("@")
        net = self._networks.get(rname) or self._net(self.ad.network(rname))
        rs = self._stored(net, rid)
        parsed = FrameRef.parse(frame)
        if rs is None or parsed is None:
            raise EvaluationError(f"no predictions {results}" if rs is None else f"{frame!r} is not scene/index")
        ref = FrameRef(parsed[0], parsed[1])
        if not rs.has(ref):
            raise EvaluationError(f"{frame} is not in {results}")
        if not name:
            return {"results": results, "frame": ref.label, "taps": rs.tap_names(ref)}
        if name not in rs.tap_names(ref):
            raise EvaluationError(f"{frame} of {results} has no tap {name!r}; it has {rs.tap_names(ref)} (ask for it with `predict --taps {name}`)")
        arr = np.load(rs._tap_file(ref, name), mmap_mode="r")
        return {"results": results, "frame": ref.label, "tap": name, "path": rs._tap_file(ref, name), "shape": list(arr.shape), "dtype": str(arr.dtype)}

    def quant(self, id: str, match: Optional[str] = None) -> Dict[str, Any]:'''),
])
print("ok")
