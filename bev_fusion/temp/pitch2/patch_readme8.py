def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/README.md", [
    (" Runner (the project's class; generic: runner_api.py)  -- the only class that executes the network   (additive counts per frame: small)", " Runner (the project's class, BEV: runner/runner.py)  -- the only class that executes the network   (additive counts per frame: small)"),
    ('''it predicts. Which class: the one a project names in `<project>/evaluation/evalsuite.json` as `"runner": "module:Class"` (module relative to the project root, like the adapter; built with
`(project, adapter)`; BEV: `runner.runner:Runner`, whose command line is `python -m runner` from `bev_fusion/`, for frames without labels), else the generic Runner of `runner_api.py`; an adapter
handed to `Evaluation(adapter=...)` gets the generic one, and `Evaluation(runner=...)` injects any object with `predict`, `plan`, `list_results`, `tap`, `time`, `probe`. evalsuite never imports a project's Runner
by name.''', '''it predicts. There is one Runner and the Evaluation always includes it: the class a project names in `<project>/evaluation/evalsuite.json` as `"runner": "module:Class"` (module relative to the
project root, like the adapter; built with `(project, adapter)`; BEV: `runner.runner:Runner`, whose command line is `python -m runner` from `bev_fusion/`, for frames without labels). evalsuite has no
Runner of its own and never imports a project's by name; a project that names none is an error, and `Evaluation(runner=...)` injects any object with `predict`, `plan`, `list_results`, `tap`, `time`,
`probe`. The toy selftest of this interface (`AR_PROJECT=../bev_fusion python -m evalsuite.selftest_api`) therefore runs the Evaluation together with the BEV Runner on a toy adapter.'''),
    ('''The Evaluation counts and draws with this engine; it also works alone for a network that has no project adapter. You write three small classes for your network; the suite then gives you pictures, numbers and cost.''', '''The Evaluation counts and draws with this engine and builds its workbench (`EvalSuite`: a Source, a Subject or none, a Task, with `.quant` and `.qual`) for each call; a person does not construct one. `EvalSuite`
is the older entry point of `bev_suite.py`, the lane project and the engine's selftests, and it works alone for a network that has no project adapter. You write three small classes for your network; the suite then gives you pictures, numbers and cost.'''),
])
patch("../bev_fusion/tests/test_evaluation_api.py", [
    ('''    from evalsuite.selftest_api import run_all
    r = run_all()''', '''    from evalsuite.selftest_api import run_all
    from evalsuite.api import Evaluation as _Evaluation
    r = run_all(type(_Evaluation(project=ROOT).runner))                                                    # the Evaluation together with THE Runner (bev_fusion/runner) on the toy adapter'''),
])
print("ok")
