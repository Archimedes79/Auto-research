def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    # ---- the adapter: free_source and free_subject go together
    ('''    missing = [f for f in ADAPTER_FUNCTIONS if not hasattr(mod, f)]
    if missing:
        raise EvaluationError(f"the adapter lacks {', '.join(missing)}")
    return mod''', '''    missing = [f for f in ADAPTER_FUNCTIONS if not hasattr(mod, f)]
    if missing:
        raise EvaluationError(f"the adapter lacks {', '.join(missing)}")
    if hasattr(mod, "free_source") != hasattr(mod, "free_subject"):
        raise EvaluationError("the adapter defines only one of free_source / free_subject: they go together (a Source without ground truth and the Subject that fits it)")
    return mod


def names(x: "str | Sequence[str]") -> Tuple[str, ...]:
    """a list of names from a name or a list (a bare string is one name, not its letters)"""
    return (x,) if isinstance(x, str) else tuple(x)'''),
    # ---- Helpers: one _source for both classes
    ('''    def _release(self, source: Source) -> None:
        if hasattr(self.ad, "release"):
            self.ad.release(source)
''', '''    def _release(self, source: Source) -> None:
        if hasattr(self.ad, "release"):
            self.ad.release(source)

    def _source(self, net: str, of: str, workers: int, control: str, labelled: bool) -> Source:
        """the Source of a set: with its ground truth (the adapter's gt_source, else source) for the Evaluation, without (free_source, else source) for the Runner. A set the project does not
        define is refused before anything is opened, and whatever the adapter raises (SystemExit included) is an EvaluationError"""
        known = self.ad.sets()
        if of not in known:
            raise EvaluationError(f"set {of!r}: not one of {sorted(known)}" + (" (the TEST split is not offered)" if "test" in of else ""))
        try:
            return getattr(self.ad, "gt_source" if labelled else "free_source", self.ad.source)(net, of, workers=workers, control=control)
        except SystemExit as e:
            raise EvaluationError(str(e.code) if e.code is not None else "the data adapter stopped") from None
        except (ValueError, KeyError, OSError) as e:
            raise EvaluationError(str(e.args[0] if isinstance(e, KeyError) and e.args else e)) from None
'''),
    # ---- Evaluation.check
    ('''        res = dict(self.runner.check(network, frames, workers=workers))''', '''        metrics = names(metrics)
        res = dict(self.runner.check(network, str(frames), workers=workers))'''),
    ('''            self._task(net, tuple(metrics))
            source = self._source(net, fr.of, workers, "")
        except (EvaluationError, ValueError, KeyError, OSError) as e:''', '''            self._task(net, metrics)
            source = self._source(net, fr.of, workers, "", True)
        except (EvaluationError, ValueError, KeyError, OSError) as e:'''),
    # ---- score / run
    ('''        """score stored predictions (an id `<network>@r<hash>`) on the frames against the ground truth; the network is not run; frames that were not predicted are an error"""
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
                raise EvaluationError(f"{missing} of the {len(keep)} frames are not in {results}: run them first (`Runner.run`), or use `Evaluation.run`")
            return self._score(net, source, desc, rs, keep, fr, metrics, 0.0, 0, True, progress)
        finally:
            self._release(source)
''', '''        """score stored predictions (an id `<network>@r<hash>`) on the frames against the ground truth; the network is not run; frames that were not predicted are an error"""
        return self._score_stored(results, Frames.parse(frames), gt_key, names(metrics), workers, progress, 0.0, 0, True)
'''),
    ('''        net, fr, metrics = self.resolve(network), Frames.parse(frames), tuple(metrics)
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

    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress):''', '''        net, fr, metrics = self.resolve(network), Frames.parse(frames), names(metrics)
        self._task(net, metrics)                                          # a typo in the metrics fails here, not after the network ran
        pinfo = self.runner.run(net, str(fr), names(taps), device, workers, control, progress)       # the network runs and is released; the labelled data is opened after it
        try:
            return self._score_stored(pinfo["results"], fr, gt_key, metrics, workers, progress, pinfo["seconds_network"], pinfo["frames_predicted"], pinfo["frames_predicted"] == 0)
        except EvaluationError as e:
            raise EvaluationError(f"{e} (the predictions are stored as {pinfo['results']}: `score` them when this is fixed)") from None

    def _score_stored(self, results, fr, gt_key, metrics, workers, progress, seconds_network, predicted, reused):
        name, _, rid = str(results).partition("@")
        net = self.resolve(self._networks.get(name) or self.ad.network(name))
        rs = self.stored(net, rid)
        if rs is None:
            raise EvaluationError(f"no predictions {results}: nothing at {os.path.join(self.ad.out(net), 'results', rid)}")
        source = self._source(net, fr.of, workers, rs.manifest["identity"]["inputs"].get("control") or "", True)
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)
            missing = [r.label for r in keep if not rs.has(r)]
            if missing:
                raise EvaluationError(f"{len(missing)} of the {len(keep)} frames (e.g. {missing[0]}) are not in {results}: run them first (`Runner.run`), or use `Evaluation.run`")
            return self._score(net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress)
        finally:
            self._release(source)

    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress):'''),
    # ---- the merged _source is gone from the Evaluation
    ('''    def _source(self, net: str, of: str, workers: int, control: str) -> Source:
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

''', ''),
])
patch("evalsuite/runner_api.py", [
    ('''from .api import CARRY, MEMORY_GROWTH_MIN, MEMORY_MIN_FRAMES, MEMORY_START_MAX, EvaluationError, Frames, Helpers''', '''from .api import CARRY, MEMORY_GROWTH_MIN, MEMORY_MIN_FRAMES, MEMORY_START_MAX, EvaluationError, Frames, Helpers, names'''),
    ('''            source = self._source(net, fr.of, workers, control)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
            return res''', '''            source = self._source(net, fr.of, workers, control, False)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
            return res'''),
    ('''        net, fr, taps = self.resolve(network), Frames.parse(frames), tuple(taps)
        ident = self.net_identity(net)
        source = self._source(net, fr.of, workers, control)
        try:
            scenes, keep = fr.resolve(source)
            identity''', '''        net, fr, taps = self.resolve(network), Frames.parse(frames), names(taps)
        ident = self.net_identity(net)
        source = self._source(net, fr.of, workers, control, False)
        try:
            scenes, keep = fr.resolve(source)
            identity'''),
    ('''        problems: List[str] = []
        res: Dict[str, Any] = {"network": network, "frames": str(frames), "ok": False, "problems": problems}''', '''        problems: List[str] = []
        taps = names(taps)
        res: Dict[str, Any] = {"network": network, "frames": str(frames), "ok": False, "problems": problems}'''),
    ('''            have = sum(1 for r in keep if rs and rs.has(r, tuple(taps)))''', '''            have = sum(1 for r in keep if rs and rs.has(r, taps))'''),
    ('''                to_run += (kept[-1].index + 1) if any(not (rs and rs.has(r, tuple(taps))) for r in kept) else 0''', '''                to_run += (kept[-1].index + 1) if any(not (rs and rs.has(r, taps)) for r in kept) else 0'''),
])
# the Runner._source (no longer there) and the old import
s = open("evalsuite/runner_api.py", encoding="utf-8").read()
a = s.index("    # ---------------------------------------------------------------------------------------- the data\n    def _source(")
b = s.index("\n\nRunner.METHODS")
s = s[:a].rstrip() + "\n" + s[b:]
s = s.replace("from .contracts import FrameRef, Prediction, Source\n", "from .contracts import Prediction\n")
open("evalsuite/runner_api.py", "w", encoding="utf-8").write(s)
print("ok")
