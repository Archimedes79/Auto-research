def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


# ------------------------------------------------------------------------------------------------ the Runner speaks the vocabulary of runsuite's Engine (a drop-in later)
patch("evalsuite/runner_api.py", [
    ('''    rn.check("runs/t23_C_s1", "large.val")                         # the dry run: scenes, frames, what is stored, what would run, the problems
    rn.run("runs/t23_C_s1", "large.val", taps=("fused",))          # predict what is not stored yet, store it (ResultSet, results.py): the id `<network>@r<hash>`
    rn.tap(id, "scene-0061/12", "fused");  rn.list("runs/t23_C_s1")''', '''    rn.plan("runs/t23_C_s1", "large.val")                          # the dry run: scenes, frames, what is stored, what would run, the problems
    rn.predict("runs/t23_C_s1", "large.val", taps=("fused",))      # predict what is not stored yet, store it (ResultSet, results.py): the id `<network>@r<hash>`
    rn.tap(id, "scene-0061/12", "fused");  rn.list_results("runs/t23_C_s1")'''),
    ('''in the result of `run`, never an error.''', '''in the result of `predict`, never an error.'''),
    ('''`run` is not `autoresearch/tools/run.py`: no ledger row, no GPU slot.''', '''`predict` is not `autoresearch/tools/run.py`: no ledger row, no GPU slot.'''),
    ('''    def check(self, network: str, frames: str = "val", taps: Sequence[str] = (), workers: int = 0, control: str = "") -> Dict[str, Any]:
        """the dry run: what `run` would do with these frames and what is wrong (`ok`, `problems`).''', '''    def plan(self, network: str, frames: str = "val", taps: Sequence[str] = (), workers: int = 0, control: str = "") -> Dict[str, Any]:
        """the dry run of `predict`: what it would do with these frames and what is wrong (`ok`, `problems`).'''),
    ('''    def list(self, network: str) -> List[Dict[str, Any]]:''', '''    def list_results(self, network: str) -> List[Dict[str, Any]]:'''),
    ('''(ask for it with `run --taps {name}`)''', '''(ask for it with `predict --taps {name}`)'''),
    ('''    def run(self, network: str, frames: str = "val", taps: Sequence[str] = (), device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:''', '''    def predict(self, network: str, frames: str = "val", taps: Sequence[str] = (), device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:'''),
])

# ------------------------------------------------------------------------------------------------ api.py
patch("evalsuite/api.py", [
    ('''Layers: command line / caller -> Evaluation -> Runner. `Evaluation.run` is `Runner.run` + `score`.''', '''Layers: command line / caller -> Evaluation -> Runner. `Evaluation.run` is `Runner.predict` + `score`.'''),
    ('''runner = Runner(adapter=self.ad)                  # the class that executes the network (D72): any object with its run / check / list can be injected''', '''runner = Runner(adapter=self.ad)                  # the class that executes the network (D72): any object with its predict / plan / list_results / tap / time / probe can be injected'''),
    ('''rows, name, base = list(self.runner.list(net)), self.ad.name(net)''', '''rows, name, base = list(self.runner.list_results(net)), self.ad.name(net)'''),
    # ---- check: taps and control like run
    ('''    def check(self, network: str, frames: str = "val", metrics: Sequence[str] = ("maps", "boxes"), workers: int = 0) -> Dict[str, Any]:
        """the dry run: what `run` would do with these frames (the Runner's check) and what is wrong with the ground truth and the metrics (`ok`, `problems`). No frame runs; the data
        and the checkpoint are opened (a BEV network is built on the CPU)"""
        metrics = names(metrics)
        res = dict(self.runner.check(network, str(frames), workers=workers))''', '''    def check(self, network: str, frames: str = "val", metrics: Sequence[str] = ("maps", "boxes"), taps: Sequence[str] = (), workers: int = 0, control: str = "") -> Dict[str, Any]:
        """the dry run: what `run` would do with these frames (the Runner's plan) and what is wrong with the ground truth and the metrics (`ok`, `problems`). No frame runs; the data
        and the checkpoint are opened (a BEV network is built on the CPU)"""
        metrics = names(metrics)
        res = dict(self.runner.plan(network, str(frames), names(taps), workers, control))'''),
    ('''            source = self._source(net, fr.of, workers, "", True)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
            res["ok"] = False
            return res
        try:
            desc = source.describe()''', '''            source = self._source(net, fr.of, workers, control, True)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
            res["ok"] = False
            return res
        try:
            self._release(source)                                             # only the ground truth is needed from here: the network a source may hold is given back
            desc = source.describe()'''),
    # ---- run
    ('''        pinfo = self.runner.run(net, str(fr), names(taps), device, workers, control, progress)''', '''        pinfo = self.runner.predict(net, str(fr), names(taps), device, workers, control, progress)'''),
    # ---- _score_stored: release the network early, hint
    ('''        source = self._source(net, fr.of, workers, rs.manifest["identity"]["inputs"].get("control") or "", True)
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)
            missing = [r.label for r in keep if not rs.has(r)]
            if missing:
                raise EvaluationError(f"{len(missing)} of the {len(keep)} frames (e.g. {missing[0]}) are not in {results}: run them first (`Runner.run`), or use `Evaluation.run`")''', '''        source = self._source(net, fr.of, workers, rs.manifest["identity"]["inputs"].get("control") or "", True)
        try:
            self._release(source)                                             # scoring needs the ground truth, not the network a source may hold: give it back now
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)
            missing = [r.label for r in keep if not rs.has(r)]
            if missing:
                raise EvaluationError(f"{len(missing)} of the {len(keep)} frames (e.g. {missing[0]}) are not in {results}: predict them first (`run`, or the Runner's `predict`)")'''),
    # ---- _score: the record inside the lock, a clash of the same question on another set is a message
    ('''        t1 = time.time()
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
        return (''', '''        t1 = time.time()
        try:
            with writer_lock(os.path.join(out, "suite", "quant", eid)):
                store = quant.run(frames=keep, state=CARRY, name=eid, results=rs, reference=ref_of, progress=progress)
                info = store.manifest.get("evaluation") or {"id": f"{name}@{eid}", "results": f"{name}@{rs.id}", "network": net, "frames": str(fr), "gt": desc.get("gt_version") or desc.get("gt_signature"),
                                                            "gt_signature": desc.get("gt_signature"), "metrics": list(metrics), "n_frames": len(store.frames), "n_scenes": len(store.scene_names()),
                                                            "control": rs.manifest["identity"]["inputs"].get("control") or "", "state": CARRY, "evaluator": desc.get("evaluator_hash")}
                if "evaluation" not in store.manifest:
                    store.manifest["evaluation"] = info                       # written once, under the lock: a later call of the same question does not overwrite its record
                    store._write_manifest()
        except Locked as e:
            raise EvaluationError(str(e)) from None
        except ValueError as e:                                               # the store of this id holds another question (the same frames asked through another set, ...)
            raise EvaluationError(str(e)) from None
        return ('''),
    # ---- the evaluation id: the set, and what the adapter says shapes its numbers
    ('''    @staticmethod
    def _eval_id(rid: str, keep: Sequence[FrameRef], desc: Dict[str, Any], metrics: Sequence[str]) -> str:
        return "e" + digest([rid, digest([[r.label, r.key] for r in keep], 40), desc.get("gt_signature"), sorted(metrics), desc.get("evaluator_hash")])''', '''    def _eval_id(self, rid: str, keep: Sequence[FrameRef], desc: Dict[str, Any], metrics: Sequence[str]) -> str:
        """what the numbers depend on: the predictions, the frames (with their dataset keys), the set they were asked through, the ground truth, the metrics, the evaluator code (the source's
        hash, and the adapter's own `evaluator_extra()` for the code that shapes numbers outside it)"""
        extra = self.ad.evaluator_extra() if hasattr(self.ad, "evaluator_extra") else None
        return "e" + digest([rid, digest([[r.label, r.key] for r in keep], 40), desc.get("split"), desc.get("set_hash"), desc.get("gt_signature"), sorted(metrics), desc.get("evaluator_hash"), extra])'''),
    # ---- reading: a store without a Task (compare opens the variant that way), the evaluator warning
    ('''    def _open(self, eval_id: str):
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
        task = self._task(info["network"], tuple(info["metrics"]))''', '''    def _open_store(self, eval_id: str):
        """(info, store) of a stored evaluation: the counts and the record of what was asked, no metric definitions (no Task is built)"""
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
        return info, store

    def _open(self, eval_id: str):
        """(info, store, quantitative) of a stored evaluation"""
        info, store = self._open_store(eval_id)
        task = self._task(info["network"], tuple(info["metrics"]))'''),
    ('''        _, sa, qa = self._open(a)
        _, sb, _ = self._open(b)
        try:
            return qa.compare(sa, sb, boot, seed, ignore=getattr(self.ad, "COMPARE_IGNORE", None))
        except ValueError as e:
            raise EvaluationError(str(e)) from None''', '''        ia, sa, qa = self._open(a)
        ib, sb = self._open_store(b)                                          # only the control's metric definitions are used; the variant may come from other code
        try:
            res = qa.compare(sa, sb, boot, seed, ignore=getattr(self.ad, "COMPARE_IGNORE", None))
        except ValueError as e:
            raise EvaluationError(str(e)) from None
        if ia.get("evaluator") != ib.get("evaluator"):
            res = {**res, "warnings": [f"the two were scored with different evaluator code ({ia.get('evaluator')} / {ib.get('evaluator')}): the difference may come from the evaluator"]}
        return res'''),
    # ---- the guard fails closed
    ('''    def _same_network(self, info: Dict[str, Any], rs: Optional[ResultSet]) -> None:
        if rs is not None and self.net_identity(info["network"]) != rs.manifest["identity"]["network"]:''', '''    def _same_network(self, info: Dict[str, Any], rs: Optional[ResultSet]) -> None:
        if rs is None:
            raise EvaluationError(f"the predictions {info['results']} of {info['id']} are gone: whether {info['network']} is still the network that was evaluated cannot be told; score it again (`run`)")
        if self.net_identity(info["network"]) != rs.manifest["identity"]["network"]:'''),
])

# ------------------------------------------------------------------------------------------------ the BEV adapter
patch("../bev_fusion/evaluation/bev_eval.py", [
    ('''def _abs(net: str) -> str:
    return net if os.path.isabs(net) else os.path.join(ROOT, net)
''', '''def _abs(net: str) -> str:
    return net if os.path.isabs(net) else os.path.join(ROOT, net)


def _claim_code(net: str) -> str:
    """the code of this run (its snapshot, else the live training/) is imported into this process once and for all: any later network with other code is refused, whatever it is asked for
    (a task, a timing, a probe, a source). -> the code folder"""
    from evalsuite.api import EvaluationError
    code = os.path.abspath(os.path.join(_abs(net), "code")) if os.path.isdir(os.path.join(_abs(net), "code")) else os.path.join(ROOT, "training")
    if _CODE["path"] not in (None, code):
        raise EvaluationError(f"one network per process: this process imported the code of {_CODE['path']}, {net} has {code}; run it in another process (compare the evaluations by their ids)")
    _CODE["path"] = code
    return code
'''),
    ('''    from evalsuite.api import EvaluationError
    from evaluation import bev_sets
    bev_sets.parse(of)                                                  # TEST and malformed names are refused before anything is built
    code = os.path.abspath(os.path.join(_abs(net), "code")) if os.path.isdir(os.path.join(_abs(net), "code")) else os.path.join(ROOT, "training")
    if _CODE["path"] not in (None, code):
        raise EvaluationError(f"one network per process: this process imported the code of {_CODE['path']}, {net} has {code}; run it in another process (compare the evaluations by their ids)")
    _CODE["path"] = code
    src, subj, _ = _A().build''', '''    from evaluation import bev_sets
    bev_sets.parse(of)                                                  # TEST and malformed names are refused before anything is built
    _claim_code(net)
    src, subj, _ = _A().build'''),
    ('''    t = bev_suite._task_only(net, "curves" in metrics''', '''    _claim_code(net)
    t = bev_suite._task_only(net, "curves" in metrics'''),
    ('''    from evaluation import bev_timing
    try:
        return bev_timing.run_timing(_abs(net), n=n)''', '''    from evaluation import bev_timing
    _claim_code(net)
    try:
        return bev_timing.run_timing(_abs(net), n=n)'''),
    ('''    from evaluation import bev_modules
    return bev_modules.probe(_abs(net), frames=frames, device=device)''', '''    from evaluation import bev_modules
    _claim_code(net)
    return bev_modules.probe(_abs(net), frames=frames, device=device)'''),
    ('''COMPARE_IGNORE = bev_suite.COMPARE_IGNORE''', '''def evaluator_extra() -> str:
    """the code that shapes the numbers outside the source's evaluator hash (bev_adapter.NUMBER_PATHS): this adapter (the reference, the metrics without the loss) and the label-free path;
    it goes into the id of an evaluation, so an edit of either scores the stored predictions again instead of serving old numbers"""
    return hashlib.sha1("".join(_file_hash(os.path.join(ROOT, "evaluation", f)) for f in ("bev_eval.py", "bev_free.py")).encode()).hexdigest()[:12]


COMPARE_IGNORE = bev_suite.COMPARE_IGNORE'''),
])
patch("../bev_fusion/tests/test_evaluation_api.py", [
    ('''runs/t16_full_s0`''', '''runs/t20_base_s0`'''),
])
print("ok")
