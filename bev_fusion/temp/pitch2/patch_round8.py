def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ("from .adapters.minimal_runner import Runner\n", ""),
    ("from .api import ADAPTER_FUNCTIONS, Evaluation, EvaluationError, Frames, load_adapter", "from .api import ADAPTER_FUNCTIONS, Evaluation, EvaluationError, Frames, load_adapter, load_runner"),
    ('''def toy_eval(adapter=None):
    """an Evaluation of the toy project with the reference Runner injected (a project names its own Runner in evalsuite.json)"""
    adapter = adapter or ME
    return Evaluation(adapter=adapter, runner=Runner(adapter=adapter))
''', '''RUNNER = {"cls": None}


def runner_class():
    """THE Runner class (there is one: the project's, named in its evalsuite.json): handed in by the caller (the BEV gate, `Evaluation.selftest`), else the class AR_PROJECT names. evalsuite has no
    Runner of its own, so this selftest runs the Evaluation together with the real one on the toy adapter"""
    if RUNNER["cls"] is None:
        RUNNER["cls"] = type(load_runner(os.environ.get("AR_PROJECT"), ME))
    return RUNNER["cls"]


def toy_eval(adapter=None):
    """an Evaluation of the toy project with the project's Runner class injected (built with the toy adapter)"""
    adapter = adapter or ME
    return Evaluation(adapter=adapter, runner=runner_class()(None, adapter))
'''),
    ('''class ToyRunner(Runner):
    """a project's own Runner, named in its evalsuite.json"""


''', ""),
    ('''has seen since the scene started) and an IoU metric; its Runner side (free_source / free_subject) reads no ground truth. No data, no GPU,''', '''has seen since the scene started) and an IoU metric; its Runner side (free_source / free_subject) reads no ground truth. No data, no GPU,'''),
    ('''    # the project names its Runner (evalsuite.json "runner"); an adapter handed in gets the generic one; a bad key is a message''', '''    # the project names its Runner (evalsuite.json "runner"): the Evaluation always includes it, whichever way the adapter comes; no key and a bad key are messages'''),
    ('''    config(f"{__name__}:ToyRunner")
    assert type(Evaluation(project=proj).runner) is ToyRunner and type(Evaluation(project=proj, adapter=ME).runner) is ToyRunner, "the project's Runner, whichever way the adapter comes"
    assert type(Evaluation(project=proj, runner=Runner(adapter=ME)).runner) is Runner, "an injected Runner is used"
    config(None)
    _raises(EvaluationError, "names no runner", Evaluation, proj)                       # no fallback: the Evaluation always includes the project's Runner
    assert type(toy_eval().runner) is Runner''', '''    cls = runner_class()
    config(f"{cls.__module__}:{cls.__name__}")
    assert type(Evaluation(project=proj).runner) is cls and type(Evaluation(project=proj, adapter=ME).runner) is cls, "the project's Runner, whichever way the adapter comes"
    assert type(Evaluation(project=proj, runner=cls(None, ME)).runner) is cls and type(toy_eval().runner) is cls, "an injected Runner is used"
    config(None)
    _raises(EvaluationError, "names no runner", Evaluation, proj)                       # no fallback: evalsuite has no Runner of its own'''),
    ('''        Runner._memory_health(rs_, "s", fill)''', '''        runner_class()._memory_health(rs_, "s", fill)'''),
    ('''        json.dump({"adapter": __name__, "runner": "evalsuite.adapters.minimal_runner:Runner"}, f)''', '''        json.dump({"adapter": __name__, "runner": f"{runner_class().__module__}:{runner_class().__name__}"}, f)'''),
    ('''    assert call("Runner.run", "runs/thr0.5", "val", *base)[0] is None, "the Runner is not on this command line (runsuite has it)"''', '''    assert call("Runner.predict", "runs/thr0.5", "val", *base)[0] is None, "the Runner is not on this command line (the project's own has it)"'''),
    ('''def run_all():
    check_frames()''', '''def run_all(runner_cls=None):
    """the checks; `runner_cls` = the project's Runner class (default: the one AR_PROJECT names in its evalsuite.json)"""
    RUNNER["cls"] = runner_cls or RUNNER["cls"]
    check_frames()'''),
    ("    # Runner.predict + score = Evaluation.run\n", "    # Runner.predict + score = Evaluation.run\n"),
])
patch("evalsuite/api.py", [
    ('''        from .selftest_api import run_all
        return run_all()''', '''        from .selftest_api import run_all
        return run_all(type(self.runner))'''),
    # ---- the engine's workbench is built here, never by a caller: no SimpleNamespace stand-in for it
    ('''        quant = Quantitative(SimpleNamespace(out=os.path.join(out, "suite"), source=source, subject=None, task=task, state=CARRY))
''', '''        quant = EvalSuite(source, None, task, out=os.path.join(out, "suite"), state=CARRY).quant
'''),
    ('''        return info, store, Quantitative(SimpleNamespace(out=os.path.join(self.ad.out(info["network"]), "suite"), source=None, subject=None, task=task, state=CARRY))''', '''        return info, store, EvalSuite(None, None, task, out=os.path.join(self.ad.out(info["network"]), "suite"), state=CARRY).quant'''),
    ('''        from .suite import EvalSuite
        fr = Frames.parse(info["frames"])''', '''        fr = Frames.parse(info["frames"])'''),
    ('''from .quantitative import Quantitative
''', '''from .suite import EvalSuite
'''),
    ('''from types import SimpleNamespace
''', ''),
])
patch("evalsuite/suite.py", [
    ('''class EvalSuite:
    def __init__(self, source: Source, subject: Subject, task: Task, out: str = "eval_out", state: str = "carry"):''', '''class EvalSuite:
    """the engine's workbench: a Source, a Subject (None when nothing is run: scoring stored predictions, reading counts) and a Task, with the qualitative and the quantitative part on them.
    The class interface (api.py, `Evaluation`) builds one per call and is what a person uses; `EvalSuite` itself is the older entry point of bev_suite.py, the lane project and the selftests."""
    def __init__(self, source: Source, subject: Optional[Subject], task: Task, out: str = "eval_out", state: str = "carry"):'''),
    ('''    def close(self) -> None:
        self.subject.close()''', '''    def close(self) -> None:
        if self.subject is not None:
            self.subject.close()'''),
    ("from typing import Any, Dict, List\n", "from typing import Any, Dict, List, Optional\n"),
])
print("ok")
