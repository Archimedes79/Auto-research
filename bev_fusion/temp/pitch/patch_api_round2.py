def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    # ---- docstring: one network per process, compare(a, b), frames grammar
    ('''    r.compare(control)                                             # paired bootstrap, r - control;   ev.qual_frame(r["id"], "scene-0061/12")
    ev.check("runs/t23_C_s1", "large.val")                         # the dry run: what `run` would do, with its problems; starts nothing
''', '''    ev.compare(a_id, b_id)                                         # paired bootstrap, b - a (a = the control);   ev.qual_frame(r["id"], "scene-0061/12")
    ev.check("runs/t23_C_s1", "large.val")                         # the dry run: what `run` would do, with its problems; the data and the checkpoint are opened, no frame runs

One network per process: a project's network code (its modules) is imported once per process, so a second network of another code snapshot cannot be run in the same one (the adapter may
refuse it). Run each network in its own process and compare the two evaluations by their ids, which only reads what was stored.
'''),
    ('''Frames: a text or a Frames object. `val`, `small.val` = every frame of a set; `spread:1000:large.val` = 1000 frames evenly spaced over all its frames; `random:3:7:val` = 3 random scenes (seed
7); `scenes:scene-0061,scene-0103:val` = those scenes; `frames:scene-0061/12,scene-0003/5:val` = those frames (their scenes run whole). The set is last and defaults to `val`.''',
     '''Frames: a text or a Frames object. `val`, `small.val` = every frame of a set; `spread:1000:large.val` = 1000 frames evenly spaced over all its frames (the scenes they lie in run up to
the last of them); `random:3[:SEED]:val` = 3 random scenes (seed 0 when none is named); `scenes:scene-0061,scene-0103:val` = those scenes; `frames:scene-0061/12,scene-0003/5:val` = those frames
(their scenes run whole up to them). The set is last and defaults to `val`; it must be one the project defines (`sets`).'''),
    # ---- imports and helpers
    ("import hashlib\nimport importlib\nimport json\nimport os\nimport sys\nimport time\n", "import importlib\nimport json\nimport os\nimport sys\nimport time\n"),
    ("from .results import Locked, ResultSet, result_id, writer_lock\n", "from .results import Locked, ResultSet, digest, result_id, writer_lock\n"),
    ('''MEMORY_START_MAX, MEMORY_GROWTH_MIN, MEMORY_MIN_FRAMES = 0.05, 0.02, 4''', '''CARRY = "carry"                                                               # the memory is always carried along a scene (D70)
MEMORY_START_MAX, MEMORY_GROWTH_MIN, MEMORY_MIN_FRAMES = 0.05, 0.02, 4'''),
    ('''def _hash(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


''', ''),
    # ---- Frames.parse
    ('''        parts = str(text).split(":")
        if parts[0] not in cls.KINDS or parts[0] == "all":
            return cls.all(str(text))                                           # a bare set name
        kind, n = parts[0], cls.KINDS[parts[0]]
        if len(parts) < 1 + n or len(parts) > 2 + n:
            raise EvaluationError(f"frames {text!r}: {kind}:" + ":".join(["ARG"] * n) + "[:SET]")
        args = parts[1:1 + n]
        of = parts[1 + n] if len(parts) == 2 + n else "val"
        try:
            conv = {"spread": lambda a: (int(a[0]),), "random": lambda a: (int(a[0]), int(a[1])),
                    "scenes": lambda a: (tuple(x for x in a[0].split(",") if x),), "frames": lambda a: (tuple(x for x in a[0].split(",") if x),)}[kind](args)
        except ValueError:
            raise EvaluationError(f"frames {text!r}: a number is expected in {kind}:...") from None
        return cls(kind, conv, of)''', '''        parts = str(text).split(":")
        kind = parts[0]
        if kind not in cls.KINDS or kind == "all":
            return cls.all(str(text))                                           # a bare set name
        usage = {"spread": "spread:N[:SET]", "random": "random:N[:SEED][:SET]", "scenes": "scenes:NAME,NAME[:SET]", "frames": "frames:SCENE/INDEX,SCENE/INDEX[:SET]"}[kind]
        rest = parts[1:]
        if not rest or len(rest) > 3:
            raise EvaluationError(f"frames {text!r}: {usage}")
        try:
            if kind == "random":
                n, seed = int(rest[0]), 0
                if len(rest) > 1 and rest[1].lstrip("-").isdigit():              # random:3:7:val, random:3:7, random:3:val
                    seed, rest = int(rest[1]), rest[:1] + rest[2:]
                args, of = (n, seed), (rest[1] if len(rest) > 1 else "val")
            elif kind == "spread":
                args, of = (int(rest[0]),), (rest[1] if len(rest) > 1 else "val")
            else:
                args, of = (tuple(x for x in rest[0].split(",") if x),), (rest[1] if len(rest) > 1 else "val")
        except ValueError:
            raise EvaluationError(f"frames {text!r}: a number is expected: {usage}") from None
        if len(rest) > 2:
            raise EvaluationError(f"frames {text!r}: {usage}")
        return cls(kind, args, of)'''),
    # ---- list uses the public count
    ('''"frames": len(rs._listing()), "megabytes": rs.size_mb()''', '''"frames": rs.n_frames(), "megabytes": rs.size_mb()'''),
    # ---- check
    ('''        """the dry run: what `run` would do with these frames, and what is wrong (`ok`, `problems`). Starts no network (the data is opened)"""''',
     '''        """the dry run: what `run` would do with these frames and what is wrong (`ok`, `problems`). No frame runs; the data and the checkpoint are opened (a BEV network is built on the CPU)"""'''),
    ('''            self._task(net, tuple(metrics))
            source = self.ad.source(net, fr.of, workers=workers, control="")
        except (EvaluationError, ValueError, KeyError, OSError, SystemExit) as e:
            problems.append(str(e))
            return res''', '''            self._task(net, tuple(metrics))
            source = self._source(net, fr.of, workers, "")
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
            return res'''),
    ('''            have = sum(1 for r in keep if rs and rs.has(r))
            res.update({"scenes_run": len(scenes), "frames_kept": len(keep), "frames_stored": have,''', '''            have = sum(1 for r in keep if rs and rs.has(r))
            want = {r.label for r in keep}
            to_run = 0                                                           # the network runs each scene up to its last kept frame, the frames before it included (the memory)
            for sc in scenes:
                kept = [r for r in source.frames(sc) if r.label in want]
                to_run += (kept[-1].index + 1) if any(not (rs and rs.has(r)) for r in kept) else 0
            res.update({"scenes_run": len(scenes), "frames_kept": len(keep), "frames_stored": have, "frames_to_run": to_run,'''),
    ('''        except (EvaluationError, ValueError, KeyError, OSError, SystemExit) as e:
            problems.append(str(e))
        finally:
            self._release(source)
        res["ok"] = not problems''', '''        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
        finally:
            self._release(source)
        res["ok"] = not problems'''),
    ('''"gt": desc.get("gt_version") or desc.get("gt_signature"), "state": "carry (whole scenes in time order, D70)",''', '''"gt": desc.get("gt_version") or desc.get("gt_signature"), "state": "carry (whole scenes in time order, D70)",'''),
    # ---- predict / score / run use _source
    ('''        net, fr = self._net(network), Frames.parse(frames)
        ident = self._net_identity(net)
        source = self.ad.source(net, fr.of, workers=workers, control=control)
        try:
            desc = source.describe()
            self._need_gt(desc, None)''', '''        net, fr = self._net(network), Frames.parse(frames)
        ident = self._net_identity(net)
        source = self._source(net, fr.of, workers, control)
        try:
            desc = source.describe()
            self._need_gt(desc, None)'''),
    ('''        source = self.ad.source(net, fr.of, workers=workers, control=rs.manifest["identity"]["inputs"].get("control") or "")
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)
            missing = sum(not rs.has(r) for r in keep)
            if missing:
                raise EvaluationError(f"{missing} of the {len(keep)} frames are not in {results}: `predict` them first, or use `run`")
            return self._score(net, source, desc, rs, keep, fr, metrics, 0.0, 0, True, progress)''', '''        source = self._source(net, fr.of, workers, rs.manifest["identity"]["inputs"].get("control") or "")
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)
            missing = sum(not rs.has(r) for r in keep)
            if missing:
                raise EvaluationError(f"{missing} of the {len(keep)} frames are not in {results}: `predict` them first, or use `run`")
            return self._score(net, source, desc, rs, keep, fr, metrics, 0.0, 0, True, progress)'''),
    ('''        self._task(net, metrics)                                          # a typo in the metrics fails here, not after the network ran
        source = self.ad.source(net, fr.of, workers=workers, control=control)''', '''        self._task(net, metrics)                                          # a typo in the metrics fails here, not after the network ran
        source = self._source(net, fr.of, workers, control)'''),
    ('''            return self._score(net, source, desc, rs, keep, fr, metrics, pinfo["seconds_network"], pinfo["frames_predicted"], pinfo["frames_predicted"] == 0, progress, control)''',
     '''            return self._score(net, source, desc, rs, keep, fr, metrics, pinfo["seconds_network"], pinfo["frames_predicted"], pinfo["frames_predicted"] == 0, progress)'''),
    # ---- the network run: identity re-read right after the checkpoint was loaded, ground truth and device with every frame
    ('''                    if todo:
                        self._run_network(net, source, scenes, keep, rs, desc.get("gt_signature"), taps, device, progress)''', '''                    if todo:
                        self._run_network(net, source, scenes, keep, rs, ident, desc.get("gt_signature"), taps, device, progress)'''),
    ('''    def _run_network(self, net, source, scenes, keep, rs, gt, taps, device, progress):
        subject = self.ad.subject(net, source, device)
        declared = {t.name for t in subject.taps()}''', '''    def _run_network(self, net, source, scenes, keep, rs, ident, gt, taps, device, progress):
        subject = self.ad.subject(net, source, device)
        if self._net_identity(net) != ident:                                   # a training may have written a new checkpoint while the network was loaded: refuse before a frame is stored
            subject.close()
            raise EvaluationError(f"{net} changed while it was loaded (a new checkpoint?): nothing was stored; run again")
        declared = {t.name for t in subject.taps()}'''),
    ('''        runner = Runner(source, subject, "carry")
        want, t0, n, ms = {r.label for r in keep}, time.time(), 0, []''', '''        runner = Runner(source, subject, CARRY)
        want, t0, n, ms = {r.label for r in keep}, time.time(), 0, []'''),
    ('''                        stored = Prediction(pred.outputs, {k: v for k, v in pred.taps.items() if k in taps}, {**pred.info, **({"memory_fill": fill[-1]} if fill else {})})
                        rs.add(sample.ref, stored, gt=gt, device=str(getattr(subject, "device", device)))''', '''                        info = {**pred.info, "gt_at_predict": gt, "device": str(getattr(subject, "device", device)), **({"memory_fill": fill[-1]} if fill else {})}
                        rs.add(sample.ref, Prediction(pred.outputs, {k: v for k, v in pred.taps.items() if k in taps}, info))'''),
    # ---- warnings from what the evaluation really used
    ('''                    "predictions_reused": not todo, "warnings": self._warnings(rs, keep, desc)}''', '''                    "predictions_reused": not todo, "warnings": self._warnings(rs, {r.scene for r in keep})}'''),
    ('''    @staticmethod
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
        return out''', '''    @staticmethod
    def _warnings(rs: Optional[ResultSet], scenes, store=None, scored_gt: Optional[str] = None) -> List[str]:
        """what the reader of a number should know: from the frames the evaluation scored (the ground truth and the device they were predicted under) and from the memory's health"""
        out = []
        if store is not None:
            gts = sorted({str(fr["info"]["gt_at_predict"]) for fr in store.frames if fr["info"].get("gt_at_predict")})
            if gts and gts != [str(scored_gt)]:
                out.append(f"frames were predicted while the data had ground truth {gts}, the numbers are scored against {scored_gt}: what a prediction computes itself "
                           "(the val loss) belongs to the earlier ground truth; maps, boxes and the rest are scored against the present one")
            devs = sorted({fr["info"]["device"] for fr in store.frames if fr["info"].get("device")})
            if len(devs) > 1:
                out.append(f"the frames were predicted on different devices {devs}: their float results differ slightly")
        if rs is not None:
            out += [msg for sc, msg in rs.manifest.get("health", {}).items() if sc in scenes]
        return out'''),
    ('''    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress, control=""):''', '''    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress):'''),
    ('''        quant = Quantitative(SimpleNamespace(out=os.path.join(out, "suite"), source=source, subject=None, task=task, state="carry"))
        t1 = time.time()''', '''        quant = Quantitative(SimpleNamespace(out=os.path.join(out, "suite"), source=source, subject=None, task=task, state=CARRY))
        t1 = time.time()'''),
    ('''                store = quant.run(frames=keep, state="carry", name=eid, results=rs, reference=ref_of, progress=progress)''', '''                store = quant.run(frames=keep, state=CARRY, name=eid, results=rs, reference=ref_of, progress=progress)'''),
    ('''        warnings = self._warnings(rs, keep, desc)
        info = store.manifest.get("evaluation") or {''', '''        info = store.manifest.get("evaluation") or {'''),
    ('''                                                    "control": control, "state": "carry", "evaluator": desc.get("evaluator_hash")}''', '''                                                    "control": rs.manifest["identity"]["inputs"].get("control") or "", "state": CARRY, "evaluator": desc.get("evaluator_hash")}'''),
    ('''        return Result(self, {**info, "warnings": warnings, "predictions_reused": reused,''', '''        return Result(self, {**info, "warnings": self._warnings(rs, {r.scene for r in keep}, store, info["gt_signature"]), "predictions_reused": reused,'''),
    # ---- helpers
    ('''    def _identity(self, net_identity: Dict[str, Any], source: Source) -> Dict[str, Any]:
        inputs = self.ad.inputs(source) if hasattr(self.ad, "inputs") else {k: v for k, v in source.describe().items() if k not in RESULT_DROP}
        return {"network": net_identity, "inputs": inputs, "state": "carry"}

    @staticmethod
    def _eval_id(rid: str, keep: Sequence[FrameRef], desc: Dict[str, Any], metrics: Sequence[str]) -> str:
        return "e" + _hash(rid, hashlib.sha1("\\n".join(f"{r.label} {r.key}" for r in keep).encode()).hexdigest(), desc.get("gt_signature"), sorted(metrics), desc.get("evaluator_hash"))

    @staticmethod
    def _need_gt(desc: Dict[str, Any], gt_key: Optional[str]) -> bool:
        if not desc.get("gt_signature"):
            raise EvaluationError("the data has no ground-truth signature: numbers could not be told from those of another GT; fix the data adapter")
        if gt_key and gt_key not in (desc.get("gt_version"), desc.get("gt_signature")):
            raise EvaluationError(f"the data has ground truth {desc.get('gt_version') or desc.get('gt_signature')} ({desc.get('gt_signature')}), not {gt_key}")
        return True''', '''    def _identity(self, net_identity: Dict[str, Any], source: Source) -> Dict[str, Any]:
        inputs = self.ad.inputs(source) if hasattr(self.ad, "inputs") else {k: v for k, v in source.describe().items() if k not in RESULT_DROP}
        return {"network": net_identity, "inputs": inputs, "state": CARRY}

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
        """the Source of a set; a set the project does not define is refused before anything is opened, and whatever the adapter raises (SystemExit included) is an EvaluationError"""
        known = self.ad.sets()
        if of not in known:
            raise EvaluationError(f"set {of!r}: not one of {sorted(known)}" + (" (the TEST split is not offered)" if "test" in of else ""))
        try:
            return self.ad.source(net, of, workers=workers, control=control)
        except SystemExit as e:
            raise EvaluationError(str(e.code) if e.code is not None else "the data adapter stopped") from None
        except (ValueError, KeyError, OSError) as e:
            raise EvaluationError(str(e.args[0] if isinstance(e, KeyError) and e.args else e)) from None'''),
    # ---- reading methods: quant shows the warnings, by / worst do not unpack what they do not use, compare(a, b)
    ('''        info, store, q = self._open(id)
        tab = q.table(store)
        better = {f"{m.name}.{k}": m.direction(k) for m in q._metrics().values() for k in m.headline}
        keep = (lambda k: match in k) if match else (lambda k: True)
        return {"id": id, "n_frames": info["n_frames"], "gt": info["gt"], "numbers": {k: v for k, v in tab.items() if keep(k)}, "better": {k: v for k, v in better.items() if keep(k)}}''',
     '''        info, store, q = self._open(id)
        tab = q.table(store)
        better = {f"{m.name}.{k}": m.direction(k) for m in q._metrics().values() for k in m.headline}
        keep = (lambda k: match in k) if match else (lambda k: True)
        return {"id": id, "n_frames": info["n_frames"], "gt": info["gt"], "numbers": {k: v for k, v in tab.items() if keep(k)}, "better": {k: v for k, v in better.items() if keep(k)},
                "warnings": self._warnings(self._predictions(info), {fr["scene"] for fr in store.frames}, store, info["gt_signature"])}'''),
    ('''        info, store, q = self._open(id)
        if axis == "scene":
            return q.per_scene(store)''', '''        _, store, q = self._open(id)
        if axis == "scene":
            return q.per_scene(store)'''),
    ('''        info, store, q = self._open(id)
        res = q.outliers(store, k=n)''', '''        _, store, q = self._open(id)
        res = q.outliers(store, k=n)'''),
    ('''    def compare(self, control: str, variant: str, boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
        """paired scene bootstrap of the headline numbers, variant - control; refused when the two do not answer the same question (other GT, other frames)"""
        ia, sa, qa = self._open(control)
        _, sb, _ = self._open(variant)''', '''    def compare(self, a: str, b: str, boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
        """paired scene bootstrap of the headline numbers, b - a (a = the control, b = the variant); refused when the two do not answer the same question (other GT, other frames)"""
        _, sa, qa = self._open(a)
        _, sb, _ = self._open(b)'''),
    ('''        info, store, q = self._open(id)
        rs = self._predictions(info)
        return {"id": id, "predict": rs.manifest.get("cost") if rs else None, "devices": rs.manifest.get("devices") if rs else None}''', '''        info, store, q = self._open(id)
        rs = self._predictions(info)
        return {"id": id, "predict": rs.manifest.get("cost") if rs else None, "devices": sorted({fr["info"]["device"] for fr in store.frames if fr["info"].get("device")})}'''),
    ('''        """draw one frame of an evaluation (the network runs on it with the history of its scene, as it did in the evaluation); returns the picture paths and the frame's scores"""
        info, store, q = self._open(id)''', '''        """draw one frame of an evaluation: the network runs on it again, on `device` and with the history of its scene (its scores can differ slightly from the stored ones when the
        evaluation ran on another device); returns the picture paths and the frame's scores"""
        info, store, q = self._open(id)'''),
    ('''        source = self.ad.source(info["network"], fr.of, workers=workers, control=info.get("control", ""))
        suite = EvalSuite(source, self.ad.subject(info["network"], source, device), self._task(info["network"], tuple(info["metrics"])),
                          out=os.path.join(self.ad.out(info["network"]), "suite"), state="carry")''', '''        source = self._source(info["network"], fr.of, workers, info.get("control", ""))
        suite = EvalSuite(source, self.ad.subject(info["network"], source, device), self._task(info["network"], tuple(info["metrics"])),
                          out=os.path.join(self.ad.out(info["network"]), "suite"), state=CARRY)'''),
    ('''        return info, store, Quantitative(SimpleNamespace(out=os.path.join(self.ad.out(info["network"]), "suite"), source=None, subject=None, task=task, state="carry"))''', '''        return info, store, Quantitative(SimpleNamespace(out=os.path.join(self.ad.out(info["network"]), "suite"), source=None, subject=None, task=task, state=CARRY))'''),
    ('''    def compare(self, control: "Result", boot: int = 1000, seed: int = 0):
        """this evaluation (the variant) against `control`"""
        return self._ev.compare(control["id"], self["id"], boot, seed)''', '''    def compare(self, a: "Result", boot: int = 1000, seed: int = 0):
        """this evaluation (b) against `a` (the control): b - a"""
        return self._ev.compare(a["id"], self["id"], boot, seed)'''),
])
print("ok")
