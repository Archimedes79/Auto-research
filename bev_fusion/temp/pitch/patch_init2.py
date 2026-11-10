def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/__init__.py", [
    ('''    if name in ("Evaluation", "Frames", "EvaluationError"):
        from . import api
        return getattr(api, name)''', '''    if name in ("Evaluation", "Frames", "EvaluationError"):
        from . import api
        return getattr(api, name)
    if name == "Runner":
        from . import runner_api
        return runner_api.Runner'''),
    ('''"Evaluation", "Frames", "EvaluationError"]''', '''"Evaluation", "Frames", "EvaluationError", "Runner"]'''),
    ("The class interface of a project (api.py, the pattern of ../README.md):  from evalsuite import Evaluation;", "The class interface of a project (api.py, runner_api.py, the pattern of ../README.md):  from evalsuite import Evaluation, Runner;"),
])
print("ok")
