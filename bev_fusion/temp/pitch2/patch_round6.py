def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    # ---- the project names its Runner
    ('''    if hasattr(mod, "free_source") != hasattr(mod, "free_subject"):
        raise EvaluationError("the adapter defines only one of free_source / free_subject: they go together (a Source without ground truth and the Subject that fits it)")
    return mod
''', '''    if hasattr(mod, "free_source") != hasattr(mod, "free_subject"):
        raise EvaluationError("the adapter defines only one of free_source / free_subject: they go together (a Source without ground truth and the Subject that fits it)")
    return mod


def load_runner(project: Optional[str], adapter: Any, use_config: bool = True):
    """the Runner of a project: the class named by the key "runner" ("module:Class", the module relative to the project root like the adapter) in <project>/evaluation/evalsuite.json,
    built with (project, adapter); the generic Runner of evalsuite (runner_api.py) when the project names none, or when the adapter was handed in (`use_config` False: a test's adapter
    must not get the Runner of whatever AR_PROJECT is)"""
    root = os.path.abspath(project or os.environ.get("AR_PROJECT") or ".")
    spec = None
    if use_config:
        try:
            with open(os.path.join(root, "evaluation", "evalsuite.json"), encoding="utf-8") as f:
                spec = json.load(f).get("runner")
        except (OSError, ValueError, AttributeError):
            spec = None
    if not spec:
        from .runner_api import Runner
        return Runner(root, adapter)
    module, sep, cls = str(spec).partition(":")
    if not sep or not module or not cls:
        raise EvaluationError(f'"runner" in {root}/evaluation/evalsuite.json is "module:Class", got {spec!r}')
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return getattr(importlib.import_module(module), cls)(root, adapter)
    except (ImportError, AttributeError) as e:
        raise EvaluationError(f"the runner {spec!r} named in {root}/evaluation/evalsuite.json cannot be built: {e}") from None
'''),
    ('''        if runner is None:
            from .runner_api import Runner
            runner = Runner(adapter=self.ad)                  # the class that executes the network (D72): any object with its predict / plan / list_results / tap / time / probe can be injected
        self.runner = runner''', '''        if runner is None:                                    # the class that executes the network (D72): the project's own (evalsuite.json "runner"), else the generic one; any object with
            runner = load_runner(project, self.ad, use_config=adapter is None)     # predict / plan / list_results / tap / time / probe can be injected'''),
    ('''Layers: command line / caller -> Evaluation -> Runner. `Evaluation.run` is `Runner.predict` + `score`. The class that executes a network is the Runner (runner_api.py, D72: it reads no ground
truth); the Evaluation is the one front door and calls it (`Evaluation(runner=...)` injects another); `tap`, `time` and `probe` are the Runner's, passed through. Everything else here reads what
was stored (only `draw_frame`, which draws a frame, runs the network on it again). The Runner has its own command line (runsuite) for predictions of frames that have no labels.''', '''Layers: command line / caller -> Evaluation -> Runner. `Evaluation.run` is `Runner.predict` + `score`. The class that executes a network is the Runner (D72: it reads no ground truth); the
Evaluation is the one front door and calls it; `tap`, `time` and `probe` are the Runner's, passed through. Which Runner: the class a project names in `evaluation/evalsuite.json` ("runner":
"module:Class", BEV: runner.runner:Runner, whose command line is `python -m runner`), else the generic one of runner_api.py; `Evaluation(runner=...)` injects another. Everything else here reads
what was stored (only `draw_frame`, which draws a frame, runs the network on it again).'''),
    # ---- compare: what the reader of an interval should know
    ('''        ia, sa, qa = self._open(a)
        ib, sb = self._open_store(b)                                          # only the control's metric definitions are used; the variant may come from other code
        try:
            res = qa.compare(sa, sb, boot, seed, ignore=getattr(self.ad, "COMPARE_IGNORE", None))
        except ValueError as e:
            raise EvaluationError(str(e)) from None
        if ia.get("evaluator") != ib.get("evaluator"):
            res = {**res, "warnings": [f"the two were scored with different evaluator code ({ia.get('evaluator')} / {ib.get('evaluator')}): the difference may come from the evaluator"]}
        return res''', '''        ia, sa, qa = self._open(a)
        ib, sb = self._open_store(b)                                          # only the control's metric definitions are used; the variant may come from other code
        try:
            res = qa.compare(sa, sb, boot, seed, ignore=getattr(self.ad, "COMPARE_IGNORE", None))
        except ValueError as e:
            raise EvaluationError(str(e)) from None
        both = {fr["label"] for fr in sa.frames} & {fr["label"] for fr in sb.frames}
        n_scenes = len({fr["scene"] for fr in sa.frames if fr["label"] in both})
        warn = []
        if ia.get("evaluator") != ib.get("evaluator"):
            warn.append(f"the two were scored with different evaluator code ({ia.get('evaluator')} / {ib.get('evaluator')}): the difference may come from the evaluator")
        if res.get("only_a") or res.get("only_b"):
            warn.append(f"the two share {res['n_common_frames']} frames only ({res.get('only_a', 0)} only in the control, {res.get('only_b', 0)} only in the variant): the interval is on the shared frames")
        if n_scenes < MIN_SCENES_FOR_INTERVAL:
            warn.append(f"only {n_scenes} scenes: the scene bootstrap is too narrow below about {MIN_SCENES_FOR_INTERVAL} scenes (simulated: 12 % false 'excludes 0' with 10 scenes, 5 % nominal)")
        return {**res, "n_scenes": n_scenes, "warnings": warn}'''),
    ('''RESULT_DROP = ''', '''MIN_SCENES_FOR_INTERVAL = 30                                                  # the percentile scene bootstrap is anti-conservative below this (the review chat simulated it: 12.1 % false at 10 scenes, 7.4 % at 30)
RESULT_DROP = '''),
    # ---- show_numbers: the numbers are re-reduced with today's code
    ('''        return {"id": id, "n_frames": info["n_frames"], "gt": info["gt"], "numbers": {k: v for k, v in tab.items() if keep(k)}, "better": {k: v for k, v in better.items() if keep(k)},
                "warnings": self._warnings(self._predictions(info), {fr["scene"] for fr in store.frames}, store)}''', '''        warn = self._warnings(self._predictions(info), {fr["scene"] for fr in store.frames}, store)
        now = self.ad.evaluator_now() if hasattr(self.ad, "evaluator_now") else None
        if now and info.get("evaluator") and now != info["evaluator"]:
            warn.append(f"scored with evaluator code {info['evaluator']}, the code is now {now}: the counts are the old ones but they are reduced with today's metrics; score again (`run`) for a clean number")
        return {"id": id, "n_frames": info["n_frames"], "gt": info["gt"], "numbers": {k: v for k, v in tab.items() if keep(k)}, "better": {k: v for k, v in better.items() if keep(k)},
                "warnings": warn}'''),
    ('''        """paired scene bootstrap of the headline numbers, b - a (a = the control, b = the variant); refused when the two do not answer the same question (other GT, other frames)"""''', '''        """paired scene bootstrap of the headline numbers, b - a (a = the control, b = the variant); refused when the two do not answer the same question (other source, task or GT, or a frame
        that is another frame in the two); `warnings` say what the interval needs a reader to know (frames in only one of the two, fewer than 30 scenes, other evaluator code)"""'''),
])

patch("evalsuite/runner_api.py", [
    ('''"""Runner: the class that executes a network and stores what it predicts (D70, D72). The layer below the Evaluation (api.py), which is the front door and calls it (`tap`, `time`, `probe` are passed through); it also works alone in Python, to try a network on frames that
have no labels. It has no command line here: the commands of the Runner are runsuite's.''', '''"""Runner: the generic class that executes a network and stores what it predicts (D70, D72), used when a project names none of its own (`"runner": "module:Class"` in its evalsuite.json; BEV has
runner.runner:Runner with `python -m runner`). The layer below the Evaluation (api.py), which is the front door and calls it (`tap`, `time`, `probe` are passed through); it also works alone in
Python, to try a network on frames that have no labels. It has no command line here.'''),
])
print("ok")
