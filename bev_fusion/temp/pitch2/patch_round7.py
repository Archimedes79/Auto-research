def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


# ------------------------------------------------------------------------------------------------ the reference Runner is a fixture next to the toy adapter, not a part of the core
patch("evalsuite/adapters/minimal_runner.py", [
    ('''"""Runner: the generic class that executes a network and stores what it predicts (D70, D72), used when a project names none of its own (`"runner": "module:Class"` in its evalsuite.json; BEV has
runner.runner:Runner with `python -m runner`). The layer below the Evaluation (api.py), which is the front door and calls it (`tap`, `time`, `probe` are passed through); it also works alone in
Python, to try a network on frames that have no labels. It has no command line here.''', '''"""The reference Runner: the class of the toy project, and the code another project's Runner can start from. The Evaluation (api.py) always includes the Runner its project names in
`evaluation/evalsuite.json` (`"runner": "module:Class"`; BEV: runner.runner:Runner with `python -m runner`); evalsuite has none of its own. This one executes a network and stores what it
predicts (D70, D72) and is what the toy selftest (selftest_api.py) injects. `tap`, `time` and `probe` are passed through by the Evaluation. It has no command line.'''),
    ("from .api import CARRY,", "from ..api import CARRY,"),
    ("from .contracts import FrameRef, Prediction", "from ..contracts import FrameRef, Prediction"),
    ("from .results import Locked,", "from ..results import Locked,"),
    ("from .runner import Runner as FrameRunner", "from ..runner import Runner as FrameRunner"),
])
patch("evalsuite/__init__.py", [
    ("The class interface of a project (api.py, runner_api.py, the pattern of ../README.md):  from evalsuite import Evaluation, Runner;  python -m evalsuite Evaluation.run NETWORK FRAMES --gt-key v8", "The class interface of a project (api.py, the pattern of ../README.md):  from evalsuite import Evaluation;  python -m evalsuite run NETWORK FRAMES --gt-key v8  (the Runner is the project's)"),
    ('''"Evaluation", "Frames", "EvaluationError", "Runner"]''', '''"Evaluation", "Frames", "EvaluationError"]'''),
    ('''    if name == "Runner":
        from . import runner_api
        return runner_api.Runner
''', ""),
])
# ------------------------------------------------------------------------------------------------ api.py: no fallback
patch("evalsuite/api.py", [
    ('''def load_runner(project: Optional[str], adapter: Any, use_config: bool = True):
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
    module, sep, cls = str(spec).partition(":")''', '''def load_runner(project: Optional[str], adapter: Any):
    """the Runner of a project: the class named by the key "runner" ("module:Class", the module relative to the project root like the adapter) in <project>/evaluation/evalsuite.json,
    built with (project, adapter). The Evaluation always includes it: evalsuite has no Runner of its own, a project that names none is an error (`Evaluation(runner=...)` injects one)"""
    root = os.path.abspath(project or os.environ.get("AR_PROJECT") or ".")
    try:
        with open(os.path.join(root, "evaluation", "evalsuite.json"), encoding="utf-8") as f:
            spec = json.load(f).get("runner")
    except (OSError, ValueError, AttributeError):
        spec = None
    if not spec:
        raise EvaluationError(f'the project names no runner: write "runner": "module:Class" in {root}/evaluation/evalsuite.json (the Evaluation always includes the project\\'s Runner class), '
                              "or inject one: Evaluation(runner=...)")
    module, sep, cls = str(spec).partition(":")'''),
    ('''            runner = load_runner(project, self.ad, use_config=adapter is None)     # predict / plan / list_results / tap / time / probe can be injected''', '''            runner = load_runner(project, self.ad)             # predict / plan / list_results / tap / time / probe can be injected'''),
    ("""Evaluation is the one front door and calls it; `tap`, `time` and `probe` are the Runner's, passed through. Which Runner: the class a project names in `evaluation/evalsuite.json` ("runner":
"module:Class", BEV: runner.runner:Runner, whose command line is `python -m runner`), else the generic one of runner_api.py; `Evaluation(runner=...)` injects another. Everything else here reads
what was stored (only `draw_frame`, which draws a frame, runs the network on it again).""", """Evaluation is the one front door and ALWAYS includes the Runner of its project: the class it names in `evaluation/evalsuite.json` ("runner": "module:Class", BEV: runner.runner:Runner, whose
command line is `python -m runner`), built with (project, adapter); evalsuite has none of its own (a project that names none is an error; `Evaluation(runner=...)` injects one, the toy selftest does).
`tap`, `time` and `probe` are the Runner's, passed through. Everything else here reads what was stored (only `draw_frame`, which draws a frame, runs the network on it again)."""),
    ("(the health check, runner_api.py).", "(the health check of the Runner)."),
    ('''    """what the Runner (runner_api.py) and the Evaluation both need:''', '''    """what the Runner (the project's class) and the Evaluation both need:'''),
])
patch("evalsuite/apicli.py", [
    ("it passes to the Runner (runner_api.py, the layer below; its own command line is runsuite's).", "it passes to the Runner (the project's class, the layer below; BEV's own command line is `python -m runner`)."),
])
# ------------------------------------------------------------------------------------------------ the toy selftest injects its Runner / names it in a toy config
patch("evalsuite/selftest_api.py", [
    ("from .runner_api import Runner\n", "from .adapters.minimal_runner import Runner\n"),
    ('''    ev = Evaluation(adapter=ME)
    net = "runs/thr0.5"''', '''    ev = toy_eval()
    net = "runs/thr0.5"'''),
    ('''def evaluator_now(): return STATE.ev ''', '''def toy_eval(adapter=None):
    """an Evaluation of the toy project with the reference Runner injected (a project names its own Runner in evalsuite.json)"""
    adapter = adapter or ME
    return Evaluation(adapter=adapter, runner=Runner(adapter=adapter))


def evaluator_now(): return STATE.ev '''),
    ('''    assert type(Evaluation(project=proj).runner) is ToyRunner and type(Evaluation(project=proj, adapter=ME).runner) is Runner, "the key is read when the adapter is the project's too"
    config(None)
    assert type(Evaluation(project=proj).runner) is Runner, "no key: the generic Runner"
    for bad, text in (("nonsense", "module:Class"), (f"{__name__}:Nothing", "cannot be built"), ("no_such_module_here:Runner", "cannot be built")):''', '''    assert type(Evaluation(project=proj).runner) is ToyRunner and type(Evaluation(project=proj, adapter=ME).runner) is ToyRunner, "the project's Runner, whichever way the adapter comes"
    assert type(Evaluation(project=proj, runner=Runner(adapter=ME)).runner) is Runner, "an injected Runner is used"
    config(None)
    _raises(EvaluationError, "names no runner", Evaluation, proj)                       # no fallback: the Evaluation always includes the project's Runner
    assert type(toy_eval().runner) is Runner
    for bad, text in (("nonsense", "module:Class"), (f"{__name__}:Nothing", "cannot be built"), ("no_such_module_here:Runner", "cannot be built")):'''),
    ('''    fb = Evaluation(adapter=plain_ad).run("runs/thr0.45", "val", progress=0)
    _same(Evaluation(adapter=plain_ad).show_numbers(fb["id"])["numbers"]''', '''    fb = toy_eval(plain_ad).run("runs/thr0.45", "val", progress=0)
    _same(toy_eval(plain_ad).show_numbers(fb["id"])["numbers"]'''),
    ('''    assert Evaluation(adapter=ME).show_numbers(allr["id"])["numbers"] == ev.show_numbers(allr["id"])["numbers"]''', '''    assert toy_eval().show_numbers(allr["id"])["numbers"] == ev.show_numbers(allr["id"])["numbers"]'''),
    ('''== Evaluation(adapter=ME).show_numbers(info["id"])["numbers"]["iou.iou"]''', '''== toy_eval().show_numbers(info["id"])["numbers"]["iou.iou"]'''),
    ('''    base = ["--adapter", __name__, "--project", STATE.root]
''', '''    base = ["--adapter", __name__, "--project", STATE.root]
    os.makedirs(os.path.join(STATE.root, "evaluation"), exist_ok=True)                 # the toy project names its Runner like a real one
    with open(os.path.join(STATE.root, "evaluation", "evalsuite.json"), "w") as f:
        json.dump({"adapter": __name__, "runner": "evalsuite.adapters.minimal_runner:Runner"}, f)
'''),
])
patch("../bev_fusion/tests/test_evaluation_api.py", [
    ('''    assert type(Evaluation(adapter="evaluation.bev_eval").runner).__module__ == "evalsuite.runner_api", "an adapter handed in gets the generic Runner, not the project's"''', '''    assert type(Evaluation(adapter="evaluation.bev_eval").runner).__module__ == "runner.runner", "the Evaluation always includes the project's Runner, whichever way the adapter is named"'''),
])
print("ok")
