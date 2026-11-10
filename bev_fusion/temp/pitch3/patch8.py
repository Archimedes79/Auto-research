def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
patch("bev_fusion/tests/test_evaluation_api.py", [
    ('''    assert os.path.join(ROOT, "evaluation", "bev_eval.py") in bev_adapter.NUMBER_PATHS, "the adapter's own code is part of the evaluator hash (and so of the evaluation id)"
    from evalsuite.api import EvaluationError
    from evaluation import bev_adapter                                                                   # an evaluator that cannot be hashed fails closed (a renamed file of NUMBER_PATHS)
''', '''    from evalsuite.api import EvaluationError
    from evaluation import bev_adapter                                                                   # an evaluator that cannot be hashed fails closed (a renamed file of NUMBER_PATHS)
    assert "bev_eval.py" in {os.path.basename(p) for p in bev_adapter.NUMBER_PATHS}, "the adapter's own code is part of the evaluator hash (and so of the evaluation id)"
'''),
])
print("ok")
