def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/README.md", [
    (" Runner (runner_api.py)  -- the only class that executes the network                   (additive counts per frame: small)", " Runner (the project's class; generic: runner_api.py)  -- the only class that executes the network   (additive counts per frame: small)"),
    ("evaluator_extra MEMORY_TAP COMPARE_IGNORE", "evaluator_extra evaluator_now MEMORY_TAP COMPARE_IGNORE"),
    ('''it predicts; it has no command line here (runsuite's Runner offers `plan` / `predict` for frames without labels; its `Engine` has the same methods as this Runner, so one can stand for the other:
`Evaluation(runner=...)`).''', '''it predicts. Which class: the one a project names in `<project>/evaluation/evalsuite.json` as `"runner": "module:Class"` (module relative to the project root, like the adapter; built with
`(project, adapter)`; BEV: `runner.runner:Runner`, whose command line is `python -m runner` from `bev_fusion/`, for frames without labels), else the generic Runner of `runner_api.py`; an adapter
handed to `Evaluation(adapter=...)` gets the generic one, and `Evaluation(runner=...)` injects any object with `predict`, `plan`, `list_results`, `tap`, `time`, `probe`. evalsuite never imports a project's Runner
by name.'''),
    ("| `show_numbers(id, match)`, `split_numbers(id, axis)`, `find_worst(id, match, n)` | the numbers, the numbers per scene or frame, where they come from |", "| `show_numbers(id, match)`, `split_numbers(id, axis)`, `find_worst(id, match, n)` | the numbers (with `warnings`: devices, the memory's health, evaluator code changed since), the numbers per scene or frame, where they come from |"),
    ("| `compare(a, b)` | paired bootstrap, b - a (a = the control); refused on another GT or other frames; `warnings` when the two were scored with different evaluator code |", "| `compare(a, b)` | paired bootstrap, b - a (a = the control); refused on another source, task or GT; `n_scenes` and `warnings`: frames in only one of the two, fewer than 30 scenes (the interval is too narrow then), other evaluator code |"),
])
print("ok")
