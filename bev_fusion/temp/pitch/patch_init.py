p = "evalsuite/__init__.py"
s = open(p, encoding="utf-8").read()
old1 = '''Command line: python -m evalsuite --adapter pkg.module:build [--arg key=value] <qual|quant|selftest|selftest-adapter> ...
'''
new1 = '''Command line: python -m evalsuite --adapter pkg.module:build [--arg key=value] <qual|quant|selftest|selftest-adapter> ...
The class interface of a project (api.py, the pattern of ../INTERFACE.md):  from evalsuite import Evaluation;  python -m evalsuite Evaluation.run NETWORK FRAMES --gt-key v8
'''
old2 = '''__all__ = ["EvalSuite", "Source", "Subject", "Task", "Metric", "View", "FrameRef", "Sample", "Prediction", "Page", "Panel", "Shape",
           "TapSpec", "Flag", "ViewContext", "Counts"]
'''
new2 = '''__all__ = ["EvalSuite", "Source", "Subject", "Task", "Metric", "View", "FrameRef", "Sample", "Prediction", "Page", "Panel", "Shape",
           "TapSpec", "Flag", "ViewContext", "Counts", "Evaluation", "Frames", "EvaluationError"]


def __getattr__(name):                                  # the class interface is loaded when it is used: the core stays small
    if name in ("Evaluation", "Frames", "EvaluationError"):
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module 'evalsuite' has no attribute {name!r}")


'''
for o, n in ((old1, new1), (old2, new2)):
    assert s.count(o) == 1, o[:60]
    s = s.replace(o, n)
open(p, "w", encoding="utf-8").write(s)
print("ok")
